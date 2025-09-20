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

import {quat} from 'gl-matrix';

import {Quat} from '../examples/public/editor';
import {Vec3} from '../manifold-global-types';

export function euler2quat(rotation: Vec3): Quat {
  const deg2rad = Math.PI / 180;
  const q = [0, 0, 0, 1] as Quat;
  quat.rotateZ(q, q, deg2rad * rotation[2]);
  quat.rotateY(q, q, deg2rad * rotation[1]);
  quat.rotateX(q, q, deg2rad * rotation[0]);
  return q;
}