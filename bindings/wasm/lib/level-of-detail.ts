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

import {getManifoldModuleSync} from './wasm.ts';

let minCircularAngle: number = 10.0;
let minCircularEdgeLength: number = 1.0;

/**
 * Set an angle constraint when calculating the number of segments in a circle.
 * The number of segments will be rounded
 * up to the nearest factor of four.
 *
 * @param angle The minimum angle in degrees between consecutive segments. The
 * angle will increase if the the segments hit the minimum edge length.
 * Default is 10 degrees.
 */
export const setMinCircularAngle = (angle: number) => {
  minCircularAngle = angle;
  getManifoldModuleSync()?.setMinCircularAngle(angle);
};

/**
 * Set a length constraint when calculating the number segments in a circle.
 * The number of segments will be rounded
 * up to the nearest factor of four.
 *
 * @param length The minimum length of segments. The length will
 * increase if the the segments hit the minimum angle. Default is 1.0.
 */
export const setMinCircularEdgeLength = (length: number) => {
  minCircularEdgeLength = length;
  getManifoldModuleSync()?.setMinCircularEdgeLength(length);
};

/**
 * Set the default number of segments in a circle.
 * Overrides the edge length and angle
 * constraints and sets the number of segments to exactly this value.
 *
 * @param segments Number of circular segments. Default is 0, meaning no
 * constraint is applied.
 */
export const setCircularSegments = (segments: number) =>
    getManifoldModuleSync()?.setCircularSegments(segments);

/**
 * Reset the circular construction parameters to their defaults if
 * `setMinCircularAngle()`, `setMinCircularEdgeLength()`, or 
 * `setCircularSegments()` have been called.
 */
export const resetToCircularDefaults = () => {
  getManifoldModuleSync()?.resetToCircularDefaults();
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
export const getMinCircularAngle = () => minCircularAngle;

/**
 * Get the current edge length constraint.
 *
 * @returns The minimum length of segments. The length will
 * increase if the the segments hit the minimum angle. Default is 1.0.
 */
export const getMinCircularEdgeLength = () => minCircularEdgeLength;

/**
 * Determine the result of `setMinCircularAngle()`, `setMinCircularEdgeLength()`, and `setCircularSegments()` for a given radius.
 *
 * @param radius For a given radius of circle, determine how many default
 * segments there will be.
 */
export const getCircularSegments = (radius: number) =>
    getManifoldModuleSync()?.getCircularSegments(radius)!;

export const cleanup = () => {
  resetToCircularDefaults();
};