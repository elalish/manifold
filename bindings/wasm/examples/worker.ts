// Copyright 2022 The Manifold Authors.
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

import * as glMatrix from 'gl-matrix'

import {Evaluator} from '../lib/evaluate';
import * as exporter from '../lib/export';
import {GlobalDefaults} from '../lib/export';

// Setup the evaluator and it's context.
const evaluator = new Evaluator();
export const module = evaluator.getModule();

// Faster on modern browsers than Float32Array
glMatrix.glMatrix.setMatrixArrayType(Array);
evaluator.addContext({glMatrix})


// These are exporter methods that generate Manifold
// or CrossSection objects.  Tell the evaluator to intercept
// the calls, and add any created objects to the clean up list.
for (const name of ['show', 'only', 'setMaterial']) {
  evaluator.addContextMethodWithCleanup(name, (exporter as any)[name]);
}

// Add additional exporter context.  These need no garbage collection.
evaluator.addContext({
  GLTFNode: exporter.GLTFNode,
  setMorphStart: exporter.setMorphStart,
  setMorphEnd: exporter.setMorphEnd,
});

// Clean up the evaluator and exporter between runs.
export function cleanup() {
  evaluator.cleanup();
  exporter.cleanup();
}

export async function evaluateCADToModel(code: string) {
  // Global defaults can be populated by the script.  It's set per
  // evaluation, while the rest of evaluator context doesn't change from
  // run to run.
  // This can be used to set parameters elsewhere in ManifoldCAD.  For
  // example, the GLTF exporter will look for animation type and
  // framerate.
  const globalDefaults = {};

  evaluator.context.globalDefaults = globalDefaults;
  const manifold = evaluator.evaluate(code);

  const exports =
      await exporter.exportModels(globalDefaults as GlobalDefaults, manifold);

  return ({
    glbURL: URL.createObjectURL(exports?.glbBlob ?? new Blob([])),
    threeMFURL: URL.createObjectURL(exports?.threeMFBlob ?? new Blob([]))
  });
}
