// Copyright 2023 The Manifold Authors.
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

export interface SealedUint32Array<N extends number> extends Uint32Array {
  length: N;
}

export interface SealedFloat32Array<N extends number> extends Float32Array {
  length: N;
}

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
// 4x4 matrix stored in column-major order
export type Mat4 = [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];
export type SimplePolygon = Vec2[];
export type Polygons = SimplePolygon|SimplePolygon[];
export type Rect = {
  min: Vec2,
  max: Vec2
};
export type Box = {
  min: Vec3,
  max: Vec3
};
export type Smoothness = {
  halfedge: number,
  smoothness: number
};
export type Properties = {
  surfaceArea: number,
  volume: number
};
export type FillRule = 'EvenOdd'|'NonZero'|'Positive'|'Negative'
export type JoinType = 'Square'|'Round'|'Miter'
