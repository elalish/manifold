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

import type {Mat4, Vec3} from '../manifold.d.ts';

const {atan2, cos, sin, sqrt, PI} = Math;
const TAU = PI * 2;

export type Vec4 = [number, number, number, number];
export type Quat = Vec4;

export function euler2quat(rotation: Vec3): Quat {
  // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Source_code
  const cr = cos(rotation[2] * 0.5 * TAU / 360);
  const sr = sin(rotation[2] * 0.5 * TAU / 360);
  const cp = cos(rotation[1] * 0.5 * TAU / 360);
  const sp = sin(rotation[1] * 0.5 * TAU / 360);
  const cy = cos(rotation[0] * 0.5 * TAU / 360);
  const sy = sin(rotation[0] * 0.5 * TAU / 360);

  const q: Quat = [0, 0, 0, 0];
  q[3] = cr * cp * cy + sr * sp * sy;
  q[2] = sr * cp * cy - cr * sp * sy;
  q[1] = cr * sp * cy + sr * cp * sy;
  q[0] = cr * cp * sy - sr * sp * cy;

  return q;
}

export function quat2euler(quat: Quat): Vec3 {
  // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Source_code
  // https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
  const rotation: Vec3 = [0, 0, 0];

  const sr_cp = 2 * (quat[0] * quat[1] + quat[2] * quat[3]);
  const cr_cp = 1 - 2 * (quat[1] * quat[1] + quat[2] * quat[2]);
  rotation[0] = atan2(sr_cp, cr_cp) * 360 / TAU;

  const sp = sqrt(1 + 2 * (quat[0] * quat[1] - quat[2] * quat[3]));
  const cp = sqrt(1 - 2 * (quat[0] * quat[2] - quat[1] * quat[3]));
  rotation[1] = atan2(sp, cp) * 360 / TAU;

  const sy_cp = 2 * (quat[0] * quat[3] + quat[1] * quat[2]);
  const cy_cp = 1 - 2 * (quat[2] * quat[2] + quat[3] * quat[3]);
  rotation[2] = atan2(sy_cp, cy_cp) * 360 / TAU;

  return rotation;
}

/**
 * Return an identity matrix.
 *
 * @group Transformation Matrix
 */
export function identityMat4(): Mat4 {
  // Column major storage, so, uh, just imagine it flipped around a little.
  // No that it matters for an identity matrix; they look the same in both
  // row major and column major order.

  // clang-format off
  return [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];
  // clang-format on
}

// Get storage location for a matrix element.
const idx4 = (col: number, row: number) => col * 4 + row;

function lengthVec3(v: Vec3) {
  const [x, y, z] = v;
  return sqrt(x ** 2 + y ** 2 + z ** 2);
}

function normalizeVec3(v: Vec3) {
  const result = [0, 0, 0];
  const len = lengthVec3(v);
  result[0] = v[0] / len;
  result[1] = v[1] / len;
  result[2] = v[2] / len;
  return result;
}

/**
 * Multiply two matrices.
 * @group Transformation Matrix
 */
export function multiplyMat4(a: Mat4, b: Mat4): Mat4 {
  const range = [...Array(4).keys()];
  const result = identityMat4();
  for (const i of range) {
    for (const j of range) {
      for (const k of range) {
        result[idx4(i, j)] += a[idx4(i, k)] * b[idx4(k, j)];
      }
    }
  }
  return result;
}

/**
 * Create a rotation matrix, in ZYX order.
 * @group Transformation Matrix
 */
export function rotateMat4(x: number, y: number, z: number): Mat4 {
  const result = identityMat4();
  const sx = sin(x * TAU / 360);
  const cx = cos(x * TAU / 360);
  const sy = sin(y * TAU / 360);
  const cy = cos(y * TAU / 360);
  const sz = sin(z * TAU / 360);
  const cz = cos(z * TAU / 360);

  result[idx4(0, 0)] = sz * cy;
  result[idx4(1, 0)] = cz * sy * sx - sz * cx;
  result[idx4(2, 0)] = cz * sy * cx + sz * sx;

  result[idx4(0, 1)] = sz * cy;
  result[idx4(1, 1)] = sz * sy * sx + cz * cx;
  result[idx4(2, 1)] = cz * sy * cx + sz * sx;

  result[idx4(0, 2)] = -sy;
  result[idx4(1, 2)] = cz * sy;
  result[idx4(2, 2)] = cz * cy;

  return result
}

/**
 * Create a translation matrix.
 * @group Transformation Matrix
 */
export function translateMat4(x: number, y: number, z: number): Mat4 {
  const result = identityMat4();
  result[idx4(3, 0)] = x
  result[idx4(3, 1)] = y
  result[idx4(3, 2)] = z
  return result
}

/**
 * Create a scaling matrix.
 * @group Transformation Matrix
 */
export function scaleMat4(x: number, y: number, z: number): Mat4 {
  const result = identityMat4();
  result[idx4(0, 0)] = x
  result[idx4(1, 1)] = y
  result[idx4(2, 2)] = z
  return result
}

/**
 * Create a reflection matrix.
 * @param normal The normal vector of the plane to be mirrored over
 * @group Transformation Matrix
 */

export function mirrorMat4(normal: Vec3): Mat4 {
  const result = identityMat4();
  const n = normalizeVec3(normal);
  const range = [...Array(3).keys()];
  for (const j of range) {
    for (const i of range) {
      result[idx4(i, j)] -= 2 * n[i] * n[j];
    }
  }
  return result;
}
