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
 * @group ManifoldCAD
 * @category Core
 */

import type {ManifoldToplevel} from 'manifold-3d';
import type {CrossSection, Manifold} from 'manifold-3d/manifold-encapsulated-types.d.ts';

type ManagedGeometry = (Manifold|CrossSection)&{isDeleted?: () => boolean};
const memoryRegistry = new Set<ManagedGeometry>();

// manifold static methods (that return a new manifold)
const manifoldStaticFunctions = [
  'cube', 'cylinder', 'sphere', 'tetrahedron', 'extrude', 'revolve', 'union',
  'difference', 'intersection', 'levelSet', 'hull'
];
// manifold member functions (that return a new manifold)
const manifoldMemberFunctions = [
  'add',
  'subtract',
  'intersect',
  'decompose',
  'warp',
  'warpBatch',
  'minkowskiSum',
  'minkowskiDifference',
  'withContext',
  'transform',
  'translate',
  'rotate',
  'scale',
  'mirror',
  'calculateCurvature',
  'calculateNormals',
  'smoothByNormals',
  'refine',
  'refineToLength',
  'refineToTolerance',
  'setProperties',
  'simplify',
  'asOriginal',
  'trimByPlane',
  'split',
  'splitByPlane',
  'slice',
  'project',
  'hull',
  'withPrecision'
];

// CrossSection static methods (that return a new cross-section)
const crossSectionStaticFunctions = [
  'square', 'circle', 'union', 'difference', 'intersection', 'evenOdd',
  'nonZero', 'hull'
];
// CrossSection member functions (that return new geometry)
const crossSectionMemberFunctions = [
  'add', 'subtract', 'intersect', 'decompose', 'transform', 'translate',
  'rotate', 'scale', 'mirror', 'warp', 'warpBatch', 'simplify', 'offset',
  'hull', 'extrude', 'revolve'
];

/**
 * Delete any objects tagged for garbage collection.
 */
export const cleanup = () => {
  for (const obj of memoryRegistry) {
    // Embind handles may also have been deleted explicitly by the caller.
    if (!obj.isDeleted?.()) obj.delete();
  }
  memoryRegistry.clear();
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
  return function(this: any, ...args: any[]) {
    const result = originalFn.apply(this, args);
    // A factory or scene helper may return a result already registered by a
    // member call. Track each handle only once, including array results.
    for (const obj of Array.isArray(result) ? result : [result]) {
      memoryRegistry.add(obj);
    }
    return result;
  };
};

const interceptMethods = (target: any, methodNames: Array<string>) => {
  for (const name of methodNames) {
    const originalFn = target[name];
    if (typeof originalFn !== 'function') continue;
    // The public constructors inherit from the native prototypes, but native
    // factory results use those original prototypes directly. Replace the
    // method where it is defined so both kinds of instance reach the wrapper.
    let owner = target;
    while (!Object.hasOwn(owner, name)) owner = Object.getPrototypeOf(owner);
    owner[name] = garbageCollectFunction(originalFn);
  }
};

/**
 * Set up garbage collection for the constructors and a white listed set of
 * methods belonging to the Manifold WASM module.
 */
export const garbageCollectManifold =
    (target: ManifoldToplevel): ManifoldToplevel => {
      interceptMethods(target.Manifold, manifoldStaticFunctions);
      interceptMethods(target.Manifold.prototype, manifoldMemberFunctions);
      interceptMethods(target.CrossSection, crossSectionStaticFunctions);
      interceptMethods(
          target.CrossSection.prototype, crossSectionMemberFunctions);

      // A Proxy keeps statics, prototype and instanceof of the original class,
      // and registers each object it builds, with or without `new`.
      const track = (result: ManagedGeometry) => {
        memoryRegistry.add(result);
        return result;
      };
      const handler = {
        construct: (cls: any, args: any[], newTarget: Function) =>
            track(Reflect.construct(cls, args, newTarget)),
        apply: (cls: any, self: any, args: any[]) =>
            track(Reflect.apply(cls, self, args)),
      };
      target.Manifold = new Proxy(target.Manifold, handler);
      target.CrossSection = new Proxy(target.CrossSection, handler);

      return target;
    };
