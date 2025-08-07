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

import Module from './built/manifold';
import {Evaluator} from './lib/evaluate';
import * as exporter from './lib/export';
import {GlobalDefaults} from './lib/export';

export const module = await Module() as any;
module.setup();

// Setup the evaluator and it's context.
const evaluator = new Evaluator(module);

// Faster on modern browsers than Float32Array
glMatrix.glMatrix.setMatrixArrayType(Array);

evaluator.context = {
  ...evaluator.context,
  glMatrix,
  GLTFNode: exporter.GLTFNode,
  setMaterial: exporter.setMaterial,
  setMorphStart: exporter.setMorphStart,
  setMorphEnd: exporter.setMorphEnd,
  show: exporter.show,
  only: exporter.only,
};

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

  return await exporter.exportModels(
      globalDefaults as GlobalDefaults, manifold);
}
