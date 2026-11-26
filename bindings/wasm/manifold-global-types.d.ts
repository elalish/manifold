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

// Type-only import for the ctx-aware factories on ExecutionContext below.
import type {Manifold, Mesh, Mesh64} from './manifold-encapsulated-types';

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
 * @inline
 * @hidden
 */
export interface SealedFloat64Array<N extends number> extends Float64Array {
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

/**
 * One hit of {@link Manifold.rayCast}.
 *
 * @property originalID The originalID of the input mesh the hit triangle came
 * from, as in Mesh.runOriginalID. It is 0 if that mesh was not tracked, e.g. a
 * primitive that never went through asOriginal().
 * @property faceID The faceID the hit triangle got from its input mesh, as in
 * Mesh.faceID. It is 0 if that mesh had no faceID.
 * @property distance The parametric distance along the ray segment, in [0, 1],
 * where 0 is the origin and 1 is the endpoint.
 * @property position The hit point.
 * @property normal The geometric face normal at the hit.
 * @property crossing +1 if the ray enters the solid here, -1 if it exits.
 * Summing these along the ray gives the change in winding number from origin
 * to endpoint. Prefer this over the sign of the normal along the ray, which
 * can disagree in degenerate cases.
 */
export type RayHit = {
  originalID: number,
  faceID: number,
  distance: number,
  position: Vec3,
  normal: Vec3,
  crossing: number
};

export type JoinType = 'Square'|'Round'|'Miter'|'Bevel';

/**
 * Controls how circular shapes are quantized on construction. A number sets
 * `segments`. Missing fields use their defaults.
 *
 * If segments is greater than two (for a sphere, positive), it takes
 * precedence. Otherwise the minimum is used of the segments calculated based on
 * edge length and angle, rounded up to the nearest multiple of four.
 *
 * - `segments`: Number of segments in a full 360-degree circle, the same as
 *   passing a number. Partial revolves use the matching fraction of it,
 *   rounded up, ignoring an excess of one unit in the last place so
 *   floating-point error cannot add a division. A sphere rounds it up to a
 *   multiple of four. Default is 0, meaning no constraint is applied.
 * - `minAngle`: Minimum angle in degrees between consecutive segments. Default
 *   is 10.
 * - `minEdgeLength`: Minimum length of segments. Default is 1.
 */
export type CircularQuality =
    number|{segments?: number, minAngle?: number, minEdgeLength?: number};

/**
 * @see {@link Manifold.status}
 */
export type ErrorStatus = 'NoError'|'NonFiniteVertex'|'NotManifold'|
    'VertexOutOfBounds'|'PropertiesWrongLength'|'MissingPositionProperties'|
    'MergeVectorsDifferentLengths'|'MergeIndexOutOfBounds'|
    'TransformWrongLength'|'RunIndexWrongLength'|'FaceIDWrongLength'|
    'InvalidConstruction'|'ResultTooLarge'|'InvalidTangents'|'Cancelled';

/**
 * Observe and control a long-running Manifold evaluation. Attach to a
 * Manifold via Manifold.withContext(); the next eager op invoked on the
 * result (status(), withPrecision(), getPrecision(), one of the refine*
 * family, hull(), or minkowskiSum() / minkowskiDifference()) snapshots the ctx
 * and reports progress / observes cancellation through it. Deferred ops
 * (Boolean, transforms, batch ops) ignore any attached ctx. Safe to read/write
 * from any thread/worker.
 *
 * Cancellation is permanent for a Manifold: once cancelled and detected,
 * the Manifold's status becomes 'Cancelled' and stays 'Cancelled'.
 */
export interface ExecutionContext {
  /** Request cancellation. Can be called from any context. Idempotent. */
  cancel(): void;
  /** Has cancellation been requested? */
  cancelled(): boolean;
  /**
   * Normalized progress in [0, 1]. Monotonic within an evaluation.
   * Returns 1 when no work has been scheduled (interpreted as trivially
   * complete -- e.g. a single-leaf manifold has nothing to evaluate).
   */
  progress(): number;

  // ctx-aware static factories: like new Manifold / levelSet, but
  // run under this context so progress / cancellation are observed (these ops
  // have no source Manifold to attach via Manifold.withContext).

  /** Like `new Manifold`, observed/cancellable via this context. */
  fromMesh(mesh: Mesh|Mesh64): Manifold;
  /** Like {@link Manifold.levelSet}, observed/cancellable via this context. */
  levelSet(
      sdf: (point: Vec3) => number, bounds: Box, edgeLength: number,
      level?: number, tolerance?: number): Manifold;

  // Memory

  /**
   * Frees the WASM memory of this ExecutionContext, since these cannot be
   * garbage-collected automatically.
   * @group Basics
   */
  delete(): void;
}
