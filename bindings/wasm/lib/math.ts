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

/**
 * @group ManifoldCAD
 * @packageDocumentation
 */

import type {Vec3} from '../manifold.d.ts';

const {cos, sin, PI} = Math;

/**
 * A quaternion in `XYZW` order.
 *
 * ManifoldCAD, like OpenGL and many other libraries represent quaternions in
 * `XYZW` order, with the scalar last. This differs from most math textbooks
 * (and just as many other libraries) that use `WXYZ` order.
 */
export type Quat = [number, number, number, number];
export type Vec4 = Quat;

/**
 * Convert Euler angles to a quaternion.
 *
 * Rotations are applied in `XYZ` order.
 * That is roll first, then pitch and finally yaw.
 * This matches the behaviour of `Manifold.rotate()`.
 *
 * @param rotation Degrees, in `XYZ` format.
 */
export function euler2quat(rotation: Vec3): Quat {
  const [cx, cy, cz] = rotation.map(r => cos(r * PI / 360));
  const [sx, sy, sz] = rotation.map(r => sin(r * PI / 360));

  return [
    sx * cy * cz + cx * sy * sz,  // X
    cx * sy * cz - sx * cy * sz,  // Y
    cx * cy * sz + sx * sy * cz,  // Z
    cx * cy * cz - sx * sy * sz   // W
  ];
}

/**
 * Multiply two quaternions together.  This is useful for stacking rotations.
 */
export function multiplyQuat(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;

  return [
    ax * bw + aw * bx + ay * bz - az * by,  // X
    ay * bw + aw * by + az * bx - ax * bz,  // Y
    az * bw + aw * bz + ax * by - ay * bx,  // Z
    aw * bw - ax * bx - ay * by - az * bz   // W
  ];
}