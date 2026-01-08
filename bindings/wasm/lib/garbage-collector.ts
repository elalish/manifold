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
 * Objects created by the manifold WASM module will not be garbage collected
 * by the JavaScript runtime.  This module tracks those objects for later
 * cleanup.
 *
 * @packageDocumentation
 * @group manifoldCAD Runtime
 * @category Core
 */

import type {CrossSection, Manifold} from '../manifold-encapsulated-types.d.ts';
import type {ManifoldToplevel} from '../manifold.d.ts';

const memoryRegistry = Array<Manifold|CrossSection>();

// manifold static methods (that return a new manifold)
const manifoldStaticFunctions = [
  'cube', 'cylinder', 'sphere', 'tetrahedron', 'extrude', 'revolve', 'compose',
  'union', 'difference', 'intersection', 'levelSet', 'smooth', 'ofMesh', 'hull'
];
// manifold member functions (that return a new manifold)
const manifoldMemberFunctions = [
  'add',
  'subtract',
  'intersect',
  'decompose',
  'warp',
  'transform',
  'translate',
  'rotate',
  'scale',
  'mirror',
  'calculateCurvature',
  'calculateNormals',
  'smoothByNormals',
  'smoothOut',
  'refine',
  'refineToLength',
  'refineToTolerance',
  'setProperties',
  'setTolerance',
  'simplify',
  'asOriginal',
  'trimByPlane',
  'split',
  'splitByPlane',
  'slice',
  'project',
  'hull'
];

// CrossSection static methods (that return a new cross-section)
const crossSectionStaticFunctions = [
  'square', 'circle', 'union', 'difference', 'intersection', 'compose',
  'ofPolygons', 'hull'
];
// CrossSection member functions (that return a new cross-section)
const crossSectionMemberFunctions = [
  'add', 'subtract', 'intersect', 'rectClip', 'decompose', 'transform',
  'translate', 'rotate', 'scale', 'mirror', 'simplify', 'offset', 'hull'
];

/**
 * Delete any objects tagged for garbage collection.
 */
export const cleanup = () => {
  for (const obj of memoryRegistry) {
    // decompose result is an array of manifolds
    if (obj instanceof Array)
      for (const elem of obj) elem.delete();
    else
      obj.delete();
  }
  memoryRegistry.length = 0;
};

/**
 * Intercept function calls for garbage collection.
 *
 * The returned object of the call will be added to the garbage collection list.
 * When `cleanup()` called, the `delete()` method on that object will be called.
 *
 * @param originalFn
 * @returns
 */
export const garbageCollectFunction = (originalFn: any): any => {
  return (...args: any) => {
    //@ts-ignore
    const result = originalFn(...args);
    memoryRegistry.push(result);
    return result;
  };
};

const interceptMethods = (target: any, methodNames: Array<string>) => {
  for (const name of methodNames) {
    const originalFn = target[name];
    target[name] = garbageCollectFunction(originalFn);
  }
};

/**
 * Set up garbage collection for a white listed set of methods belonging
 * to the Manifold WASM module.
 */
export const garbageCollectManifold =
    (target: ManifoldToplevel): ManifoldToplevel => {
      interceptMethods(target.Manifold, manifoldStaticFunctions);
      interceptMethods(target.Manifold.prototype, manifoldMemberFunctions);
      interceptMethods(target.CrossSection, crossSectionStaticFunctions);
      interceptMethods(
          target.CrossSection.prototype, crossSectionMemberFunctions);

      return target;
    };
