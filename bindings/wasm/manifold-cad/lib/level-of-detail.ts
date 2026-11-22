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
 * Control detail level for the
 * {@link manifold.CrossSection.circle | CrossSection.circle},
 * {@link manifold.CrossSection.revolve | CrossSection.revolve},
 * {@link manifold.Manifold.cylinder | Manifold.cylinder}, and
 * {@link manifold.Manifold.sphere | Manifold.sphere} constructors.
 *
 * Libraries should not change these values, and if run through manifoldCAD or
 * the manifoldCAD CLI, will not be able to.  Libraries may get values to
 * determine their own level of detail.
 * @packageDocumentation
 * @group ManifoldCAD
 * @category Modelling
 */

import type {ManifoldToplevel} from 'manifold-3d';
import type {CircularQuality} from 'manifold-3d/manifold-global-types.d.ts';

import {getManifoldModuleSync} from './wasm.ts';

let circularSegments: number = 0;
let minCircularAngle: number = 10.0;
let minCircularEdgeLength: number = 1.0;

/**
 * The quality the circular constructors use for this run: the values set here,
 * overridden by any the caller passes. A number below minSegments leaves the
 * run's segments in place, as each constructor ignores such a count.
 */
function runQuality(quality: CircularQuality|undefined, minSegments: number) {
  const run = {
    segments: circularSegments,
    minAngle: minCircularAngle,
    minEdgeLength: minCircularEdgeLength
  };
  if (quality === undefined || quality === null) return run;
  if (typeof quality !== 'object') {
    // A count converts to an int first, as it did when it went straight to the
    // module.
    const segments = (quality as number) | 0;
    return segments >= minSegments ? {...run, segments} : run;
  }
  const set = Object.entries(quality).filter(([, v]) => v !== undefined);
  return {...run, ...Object.fromEntries(set)};
}

/**
 * Make the circular constructors of a Manifold WASM module use this run's
 * quality.
 *
 * @internal
 */
export const applyLevelOfDetail =
    (target: ManifoldToplevel): ManifoldToplevel => {
      const wrap = (obj: any, name: string, index: number, minSegments = 3) => {
        const originalFn = obj[name];
        obj[name] = function(this: any, ...args: any[]) {
          args[index] = runQuality(args[index], minSegments);
          return originalFn.apply(this, args);
        };
      };
      wrap(target.Manifold, 'cylinder', 3);
      // Sphere takes any positive count.
      wrap(target.Manifold, 'sphere', 1, 1);
      wrap(target.CrossSection, 'circle', 1);
      // bindings.js replaces CrossSection.prototype with an object derived
      // from the prototype that instances actually use. Manifold.revolve
      // calls this revolve.
      const instanceProto =
          Object.getPrototypeOf(target.CrossSection.prototype);
      wrap(instanceProto, 'revolve', 0);
      wrap(instanceProto, 'offset', 3);
      return target;
    };

export function setRelativePrecision(precision: number) {
  getManifoldModuleSync()?.setRelativePrecision(precision);
}

export function getRelativePrecision() {
  return getManifoldModuleSync()?.getRelativePrecision()!;
}

/**
 * Set an angle constraint when calculating the number of segments in a circle.
 * The number of segments will be rounded
 * up to the nearest factor of four.
 *
 * @param angle The minimum angle in degrees between consecutive segments. The
 * angle will increase if the the segments hit the minimum edge length.
 * Default is 10 degrees.
 */
export function setMinCircularAngle(angle: number) {
  if (!(angle > 0)) return;  // also rejects NaN
  minCircularAngle = angle;
}

/**
 * Set a length constraint when calculating the number segments in a circle.
 * The number of segments will be rounded
 * up to the nearest factor of four.
 *
 * @param length The minimum length of segments. The length will
 * increase if the the segments hit the minimum angle. Default is 1.0.
 */
export function setMinCircularEdgeLength(length: number) {
  if (!(length > 0)) return;  // also rejects NaN
  minCircularEdgeLength = length;
};

/**
 * Set the default number of segments in a circle.
 * Overrides the edge length and angle
 * constraints and sets the number of segments to exactly this value.
 *
 * A partial revolve uses the matching fraction of this count, rounded down,
 * with at least one. A sphere rounds it up to a multiple of four.
 *
 * @param segments Number of circular segments. Default is 0, meaning no
 * constraint is applied.
 */
export function setCircularSegments(segments: number) {
  segments |= 0;
  if (segments < 3 && segments != 0) return;
  circularSegments = segments;
}

/**
 * Reset the circular construction parameters to their defaults if
 * `setMinCircularAngle()`, `setMinCircularEdgeLength()`, or
 * `setCircularSegments()` have been called.
 */
export function resetToCircularDefaults() {
  getManifoldModuleSync()?.resetToCircularDefaults();
  circularSegments = 0;
  minCircularAngle = 10;
  minCircularEdgeLength = 1;
};

/**
 * Get the current angle constraint.
 *
 * @returns The minimum angle in degrees between consecutive segments. The
 * angle will increase if the the segments hit the minimum edge length.
 * Default is 10 degrees.
 */
export function getMinCircularAngle() {
  return minCircularAngle;
}

/**
 * Get the current edge length constraint.
 *
 * @returns The minimum length of segments. The length will
 * increase if the the segments hit the minimum angle. Default is 1.0.
 */
export function getMinCircularEdgeLength() {
  return minCircularEdgeLength;
}

/**
 * Determine the appropriate number of segments for a given radius.
 *
 * @param radius For a given radius of circle, determine how many default
 * segments there will be.
 */
export function getCircularSegments(radius: number) {
  return getManifoldModuleSync()?.segmentsFor(runQuality(undefined, 3), radius)!
      ;
}

/**
 * @internal
 */
export const cleanup = () => {
  resetToCircularDefaults();
};
