// Copyright 2025 The Manifold Authors.
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

/**
 * Relative precision for the models a script builds.
 *
 * Libraries should not change this value, and if run through manifoldCAD or
 * the manifoldCAD CLI, will not be able to.
 * @packageDocumentation
 * @group ManifoldCAD
 * @category Modelling
 */

import {getManifoldModuleSync} from './wasm.ts';

export function setRelativePrecision(precision: number) {
  getManifoldModuleSync()?.setRelativePrecision(precision);
}

export function getRelativePrecision() {
  return getManifoldModuleSync()?.getRelativePrecision()!;
}

/**
 * Resets the relative precision to its default.
 *
 * @internal
 */
export const cleanup = () => {
  getManifoldModuleSync()?.resetToCircularDefaults();
};
