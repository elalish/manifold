// Copyright 2023-2025 The Manifold Authors.
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
 * @inline
 * @hidden
 */
export interface SealedUint32Array<N extends number> extends Uint32Array {
  length: N;
}

/**
 * @inline
 * @hidden
 */
export interface SealedFloat32Array<N extends number> extends Float32Array {
  length: N;
}

/**
 * A vector in two dimensional space.
 */
export type Vec2 = [number, number];

/**
 * A vector in three dimensional space.
 */
export type Vec3 = [number, number, number];

/**
 * 3x3 matrix stored in column-major order.
 */
export type Mat3 = [
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

/**
 * 4x4 matrix stored in column-major order.
 */
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

/**
 * A two dimensional rectangle, aligned to the coordinate system.
 * @see {@link CrossSection.bounds}
 */
export type Rect = {
  min: Vec2,
  max: Vec2
};

/**
 * A three dimensional box, aligned to the coordinate system.
 *
 * @see {@link Manifold.boundingBox}
 * @see {@link Manifold.levelSet}
 */
export type Box = {
  min: Vec3,
  max: Vec3
};

export type Smoothness = {
  halfedge: number,
  smoothness: number
};

export type FillRule = 'EvenOdd'|'NonZero'|'Positive'|'Negative';

export type JoinType = 'Square'|'Round'|'Miter';

/**
 * @see {@link Manifold.status}
 */
export type ErrorStatus = 'NoError'|'NonFiniteVertex'|'NotManifold'|
    'VertexOutOfBounds'|'PropertiesWrongLength'|'MissingPositionProperties'|
    'MergeVectorsDifferentLengths'|'MergeIndexOutOfBounds'|
    'TransformWrongLength'|'RunIndexWrongLength'|'FaceIDWrongLength'|
    'InvalidConstruction';
