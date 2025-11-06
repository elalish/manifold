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

import {Vec3} from '../manifold-global-types';

const {cos, sin, PI} = Math;
const TAU = PI * 2;

export type Vec4 = [number, number, number, number];

export function euler2quat(rotation: Vec3): Vec4 {
  // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Source_code
  const cr = cos(rotation[2] * 0.5 * TAU / 360);
  const sr = sin(rotation[2] * 0.5 * TAU / 360);
  const cp = cos(rotation[1] * 0.5 * TAU / 360);
  const sp = sin(rotation[1] * 0.5 * TAU / 360);
  const cy = cos(rotation[0] * 0.5 * TAU / 360);
  const sy = sin(rotation[0] * 0.5 * TAU / 360);

  const q: Vec4 = [0, 0, 0, 0];
  q[3] = cr * cp * cy + sr * sp * sy;
  q[2] = sr * cp * cy - cr * sp * sy;
  q[1] = cr * sp * cy + sr * cp * sy;
  q[0] = cr * cp * sy - sr * sp * cy;

  return q;
};
