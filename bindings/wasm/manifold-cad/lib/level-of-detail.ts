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

import type {ManifoldToplevel} from 'manifold-3d';

import {getManifoldModuleSync} from './wasm.ts';

const defaultRelativePrecision = 1e-12;

let relativePrecision: number = defaultRelativePrecision;

/**
 * Set the relative precision of the Manifolds this script creates from then
 * on: shapes, extrusions, revolutions, level sets, hulls of points and
 * Manifolds made from meshes. Like `withPrecision()`, it applies to later
 * operations on them: shapes, level sets and hulls of points are built at the
 * default. Booleans, hulls and other operations on several Manifolds use the
 * smallest precision among their non-empty inputs.
 *
 * @param precision The relative precision, which defines what counts as
 * coplanar or degenerate. Values below the default of 1e-12 use the default.
 */
export function setRelativePrecision(precision: number) {
  relativePrecision =
      Number.isFinite(precision) && precision > defaultRelativePrecision ?
      precision :
      defaultRelativePrecision;
}

/**
 * Get the relative precision set by `setRelativePrecision()`.
 */
export function getRelativePrecision() {
  return relativePrecision;
}

// Manifold.extrude and revolve call the CrossSection methods.
const manifoldFactories =
    ['cube', 'cylinder', 'sphere', 'tetrahedron', 'levelSet'];
const crossSectionToManifold = ['extrude', 'revolve'];

/**
 * Apply the precision set by `setRelativePrecision()` to the Manifolds created
 * by the given module's factories. Apply garbage collection after this, so it
 * tracks the returned Manifolds.
 *
 * @internal
 */
export const applyRelativePrecision =
    (target: ManifoldToplevel): ManifoldToplevel => {
      const stamp =
          (owner: any, name: string,
           hasPrecision: (args: any[]) => boolean = () => false) => {
            const originalFn = owner[name];
            owner[name] = function(this: any, ...args: any) {
              const result = originalFn.apply(this, args);
              if (hasPrecision(args) ||
                  !(relativePrecision > result.getPrecision()))
                return result;
              const stamped = result.withPrecision(relativePrecision);
              result.delete();
              return stamped;
            };
          };
      for (const name of manifoldFactories) stamp(target.Manifold, name);
      // A hull of non-empty Manifolds keeps their smallest precision.
      const hasManifold = (args: any[]) => {
        const items = args.length == 1 ? args[0] : args;
        return items.some(
            (m: any) => m instanceof target.Manifold && !m.isEmpty());
      };
      stamp(target.Manifold, 'hull', hasManifold);
      // CrossSection objects from the module inherit from the parent of
      // CrossSection.prototype, which holds these methods.
      const crossSectionMethods =
          Object.getPrototypeOf(target.CrossSection.prototype);
      for (const name of crossSectionToManifold) {
        stamp(crossSectionMethods, name);
      }
      // Mesh input goes through the constructor or ofMesh, and both default
      // to the script's precision. The static methods and instanceof are
      // inherited from the original constructor.
      const ManifoldCtor: any = target.Manifold;
      const Manifold: any = function(mesh: any, precision = relativePrecision) {
        return new ManifoldCtor(mesh, precision);
      };
      Object.setPrototypeOf(Manifold, ManifoldCtor);
      Manifold.prototype = ManifoldCtor.prototype;
      Manifold.ofMesh = (mesh: any, precision = relativePrecision) =>
          new ManifoldCtor(mesh, precision);
      target.Manifold = Manifold;
      return target;
    };

/**
 * Resets the relative precision to its default.
 *
 * @internal
 */
export const cleanup = () => {
  getManifoldModuleSync()?.resetToCircularDefaults();
  relativePrecision = defaultRelativePrecision;
};
