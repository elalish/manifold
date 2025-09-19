// Copyright 2022-2025 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {Evaluator} from '../lib/evaluate';
import {Export3MF} from '../lib/export-3mf';
import {ExportGLTF} from '../lib/export-gltf';
import * as scenebuilder from '../lib/scene-builder';
import {GlobalDefaults} from '../lib/scene-builder';

// Swallow informational logs in testing framework
function log(...args: any[]) {
  if (typeof self !== 'undefined' && self.console) {
    self.console.log(...args);
  }
}

// Setup the evaluator and it's context.
const evaluator = new Evaluator();
export const module = evaluator.getModule();  // Used in tests.

// Exporters.
// The end user can download either.
// GLB (Binary GLTF) is used to send the model from this worker
// to the viewer.
const export3mf = new Export3MF();
const exportGltf = new ExportGLTF();


// These are methods that generate Manifold
// or CrossSection objects.  Tell the evaluator to intercept
// the calls, and add any created objects to the clean up list.
evaluator.addContextMethodWithCleanup('show', scenebuilder.show)
evaluator.addContextMethodWithCleanup('only', scenebuilder.only)
evaluator.addContextMethodWithCleanup('setMaterial', scenebuilder.setMaterial)

// Add additional context.  These need no garbage collection.
evaluator.addContext({
  GLTFNode: scenebuilder.GLTFNode,
  setMorphStart: scenebuilder.setMorphStart,
  setMorphEnd: scenebuilder.setMorphEnd,
});

// Clean up the evaluator and scene builder between runs.
export function cleanup() {
  evaluator.cleanup();
  scenebuilder.cleanup();
}

export async function evaluateCADToModel(code: string) {
  // Global defaults can be populated by the script.  It's set per
  // evaluation, while the rest of evaluator context doesn't change from
  // run to run.
  // This can be used to set parameters elsewhere in ManifoldCAD.  For
  // example, the GLTF exporter will look for animation type and
  // framerate.
  const globalDefaults = {} as GlobalDefaults;
  evaluator.context.globalDefaults = globalDefaults;

  const t0 = performance.now();
  const manifold = await evaluator.evaluate(code);
  const t1 = performance.now();

  log(`Manifold took ${
      (Math.round((t1 - t0) / 10) / 100).toLocaleString()} seconds`);

  // If we don't actually have a model, complain.
  if (!manifold && !scenebuilder.hasGLTFNodes()) {
    log('No output because "result" is undefined and no "GLTFNode"s were created.');
    return ({
      glbURL: URL.createObjectURL(new Blob([])),
      threeMFURL: URL.createObjectURL(new Blob([]))
    })
  }

  // Create a gltf-transform document.
  const doc = scenebuilder.hasGLTFNodes() ?
      scenebuilder.GLTFNodesToGLTFDoc(
          scenebuilder.getGLTFNodes(), globalDefaults) :
      scenebuilder.manifoldToGLTFDoc(manifold, globalDefaults);

  const blobs = {
    glbURL: URL.createObjectURL(await exportGltf.asBlob(doc)),
    threeMFURL: URL.createObjectURL(await export3mf.asBlob(doc))
  };

  const t2 = performance.now();
  log(`Exporting GLB & 3MF took ${
      (Math.round((t2 - t1) / 10) / 100).toLocaleString()} seconds`);

  return blobs;
}
