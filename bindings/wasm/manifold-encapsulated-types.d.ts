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

import {Box, Curvature, Mat4, Polygons, Properties, SealedFloat32Array, SealedUint32Array, Smoothness, Vec2, Vec3} from './manifold-global-types';

/**
 * Constructs a unit cube (edge lengths all one), by default in the first
 * octant, touching the origin.
 *
 * @param size The X, Y, and Z dimensions of the box.
 * @param center Set to true to shift the center to the origin.
 */
export function cube(size?: Vec3|number, center?: boolean): Manifold;

/**
 * A convenience constructor for the common case of extruding a circle. Can also
 * form cones if both radii are specified.
 *
 * @param height Z-extent
 * @param radiusLow Radius of bottom circle. Must be positive.
 * @param radiusHigh Radius of top circle. Can equal zero. Default is equal to
 * radiusLow.
 * @param circularSegments How many line segments to use around the circle.
 * Default is calculated by the static Defaults.
 * @param center Set to true to shift the center to the origin. Default is
 * origin at the bottom.
 */
export function cylinder(
    height: number, radiusLow: number, radiusHigh?: number,
    circularSegments?: number, center?: boolean): Manifold;

/**
 * Constructs a geodesic sphere of a given radius.
 *
 * @param radius Radius of the sphere. Must be positive.
 * @param circularSegments Number of segments along its
 * diameter. This number will always be rounded up to the nearest factor of
 * four, as this sphere is constructed by refining an octahedron. This means
 * there are a circle of vertices on all three of the axis planes. Default is
 * calculated by the static Defaults.
 */
export function sphere(radius: number, circularSegments?: number): Manifold;


/**
 * Constructs a smooth version of the input mesh by creating tangents; this
 * method will throw if you have supplied tangents with your mesh already. The
 * actual triangle resolution is unchanged; use the Refine() method to
 * interpolate to a higher-resolution curve.
 *
 * By default, every edge is calculated for maximum smoothness (very much
 * approximately), attempting to minimize the maximum mean Curvature magnitude.
 * No higher-order derivatives are considered, as the interpolation is
 * independent per triangle, only sharing constraints on their boundaries.
 *
 * @param mesh input Mesh.
 * @param sharpenedEdges If desired, you can supply a vector of sharpened
 * halfedges, which should in general be a small subset of all halfedges. Order
 * of entries doesn't matter, as each one specifies the desired smoothness
 * (between zero and one, with one the default for all unspecified halfedges)
 * and the halfedge index (3 * triangle index + [0,1,2] where 0 is the edge
 * between triVert 0 and 1, etc).
 *
 * At a smoothness value of zero, a sharp crease is made. The smoothness is
 * interpolated along each edge, so the specified value should be thought of as
 * an average. Where exactly two sharpened edges meet at a vertex, their
 * tangents are rotated to be colinear so that the sharpened edge can be
 * continuous. Vertices with only one sharpened edge are completely smooth,
 * allowing sharpened edges to smoothly vanish at termination. A single vertex
 * can be sharpened by sharping all edges that are incident on it, allowing
 * cones to be formed.
 */
export function smooth(mesh: Mesh, sharpenedEdges?: Smoothness[]): Manifold;

/**
 * Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
 * and the rest at similarly symmetric points.
 */
export function tetrahedron(): Manifold;

/**
 * Constructs a manifold from a set of polygons by extruding them along the
 * Z-axis.
 *
 * @param crossSection A set of non-overlapping polygons to extrude.
 * @param height Z-extent of extrusion.
 * @param nDivisions Number of extra copies of the crossSection to insert into
 * the shape vertically; especially useful in combination with twistDegrees to
 * avoid interpolation artifacts. Default is none.
 * @param twistDegrees Amount to twist the top crossSection relative to the
 * bottom, interpolated linearly for the divisions in between.
 * @param scaleTop Amount to scale the top (independently in X and Y). If the
 * scale is {0, 0}, a pure cone is formed with only a single vertex at the top.
 * Default {1, 1}.
 */
export function extrude(
    crossSection: Polygons, height: number, nDivisions?: number,
    twistDegrees?: number, scaleTop?: Vec2): Manifold;

/**
 * Constructs a manifold from a set of polygons by revolving this cross-section
 * around its Y-axis and then setting this as the Z-axis of the resulting
 * manifold. If the polygons cross the Y-axis, only the part on the positive X
 * side is used. Geometrically valid input will result in geometrically valid
 * output.
 *
 * @param crossSection A set of non-overlapping polygons to revolve.
 * @param circularSegments Number of segments along its diameter. Default is
 * calculated by the static Defaults.
 */
export function revolve(
    crossSection: Polygons, circularSegments?: number): Manifold;

export function union(a: Manifold, b: Manifold): Manifold;
export function difference(a: Manifold, b: Manifold): Manifold;
export function intersection(a: Manifold, b: Manifold): Manifold;

export function union(manifolds: Manifold[]): Manifold;
export function difference(manifolds: Manifold[]): Manifold;
export function intersection(manifolds: Manifold[]): Manifold;

/**
 * Constructs a new manifold from a list of other manifolds. This is a purely
 * topological operation, so care should be taken to avoid creating
 * overlapping results. It is the inverse operation of Decompose().
 *
 * @param manifolds A list of Manifolds to lazy-union together.
 */
export function compose(manifolds: Manifold[]): Manifold;

/**
 * Constructs a level-set Mesh from the input Signed-Distance Function (SDF).
 * This uses a form of Marching Tetrahedra (akin to Marching Cubes, but better
 * for manifoldness). Instead of using a cubic grid, it uses a body-centered
 * cubic grid (two shifted cubic grids). This means if your function's interior
 * exceeds the given bounds, you will see a kind of egg-crate shape closing off
 * the manifold, which is due to the underlying grid.
 *
 * @param sdf The signed-distance function which returns the signed distance of
 * a given point in R^3. Positive values are inside, negative outside.
 * @param bounds An axis-aligned box that defines the extent of the grid.
 * @param edgeLength Approximate maximum edge length of the triangles in the
 * final result. This affects grid spacing, and hence has a strong effect on
 * performance.
 * @param level You can inset your Mesh by using a positive value, or outset
 * it with a negative value.
 */
export function levelSet(
    sdf: (point: Vec3) => number, bounds: Box, edgeLength: number,
    level?: number): Manifold;

/**
 * @name Defaults
 * These static properties control how circular shapes are quantized by
 * default on construction. If circularSegments is specified, it takes
 * precedence. If it is zero, then instead the minimum is used of the segments
 * calculated based on edge length and angle, rounded up to the nearest
 * multiple of four. To get numbers not divisible by four, circularSegments
 * must be specified.
 */
///@{
export function setMinCircularAngle(angle: number): void;
export function setMinCircularEdgeLength(length: number): void;
export function setCircularSegments(segments: number): void;
export function getCircularSegments(radius: number): number;
///@}

/**
 * Returns the first of n sequential new unique mesh IDs for marking sets of
 * triangles that can be looked up after further operations. Assign to
 * Mesh.runOriginalID vector.
 */
export function reserveIDs(count: number): number;

export class Manifold {
  /**
   * Convert a Mesh into a Manifold, retaining its properties and merging only
   * the positions according to the merge vectors. Will throw an error if the
   * result is not an oriented 2-manifold. Will collapse degenerate triangles
   * and unnecessary vertices.
   *
   * All fields are read, making this structure suitable for a lossless
   * round-trip of data from getMesh(). For multi-material input, use
   * reserveIDs() to set a unique originalID for each material, and sort the
   * materials into triangle runs.
   */
  constructor(mesh: Mesh);
  /**
   * Transform this Manifold in space. Stored in column-major order. This
   * operation can be chained. Transforms are combined and applied lazily.
   *
   * @param m The affine transformation matrix to apply to all the vertices. The
   *     last row is ignored.
   */
  transform(m: Mat4): Manifold;

  /**
   * Move this Manifold in space. This operation can be chained. Transforms are
   * combined and applied lazily.
   *
   * @param v The vector to add to every vertex.
   */
  translate(v: Vec3): Manifold;

  /**
   * Applies an Euler angle rotation to the manifold, first about the X axis,
   * then Y, then Z, in degrees. We use degrees so that we can minimize rounding
   * error, and eliminate it completely for any multiples of 90 degrees.
   * Additionally, more efficient code paths are used to update the manifold
   * when the transforms only rotate by multiples of 90 degrees. This operation
   * can be chained. Transforms are combined and applied lazily.
   *
   * @param v [X, Y, Z] rotation in degrees.
   */
  rotate(v: Vec3): Manifold;

  /**
   * Scale this Manifold in space. This operation can be chained. Transforms are
   * combined and applied lazily.
   *
   * @param v The vector to multiply every vertex by per component.
   */
  scale(v: Vec3|number): Manifold;

  /**
   * Mirror this Manifold over the plane described by the unit form of the given
   * normal vector. If the length of the normal is zero, an empty Manifold is
   * returned. This operation can be chained. Transforms are combined and
   * applied lazily.
   *
   * @param normal The normal vector of the plane to be mirrored over
   */
  mirror(v: Vec3): Manifold;

  /**
   * Boolean union
   */
  add(other: Manifold): Manifold;

  /**
   * Boolean difference
   */
  subtract(other: Manifold): Manifold;

  /**
   * Boolean intersection
   */
  intersect(other: Manifold): Manifold;

  /**
   * Removes everything behind the given half-space plane.
   *
   * @param normal This vector is normal to the cutting plane and its length
   *     does not matter. The result is in the direction of this vector from the
   *     plane.
   * @param originOffset The distance of the plane from the origin in the
   *     direction of the normal vector.
   */
  trimByPlane(normal: vec3, originOffset: number): Manifold;

  /**
   * Increase the density of the mesh by splitting every edge into n pieces. For
   * instance, with n = 2, each triangle will be split into 4 triangles. These
   * will all be coplanar (and will not be immediately collapsed) unless the
   * Mesh/Manifold has halfedgeTangents specified (e.g. from the Smooth()
   * constructor), in which case the new vertices will be moved to the
   * interpolated surface according to their barycentric coordinates.
   *
   * @param n The number of pieces to split every edge into. Must be > 1.
   */
  refine(n: number): Manifold;

  /**
   * This function does not change the topology, but allows the vertices to be
   * moved according to any arbitrary input function. It is easy to create a
   * function that warps a geometrically valid object into one which overlaps,
   * but that is not checked here, so it is up to the user to choose their
   * function with discretion.
   *
   * @param warpFunc A function that modifies a given vertex position.
   */
  warp(warpFunc: (vert: Vec3) => void): Manifold;

  /**
   * This operation returns a vector of Manifolds that are topologically
   * disconnected. If everything is connected, the vector is length one,
   * containing a copy of the original. It is the inverse operation of
   * Compose().
   */
  decompose(): Manifold[];

  /**
   * Does the Manifold have any triangles?
   */
  isEmpty(): boolean;

  /**
   * The number of vertices in the Manifold.
   */
  numVert(): number;

  /**
   * The number of triangles in the Manifold.
   */
  numTri(): number;

  /**
   * The number of edges in the Manifold.
   */
  numEdge(): number;

  /**
   * Returns the axis-aligned bounding box of all the Manifold's vertices.
   */
  boundingBox(): Box;

  /**
   * Returns the precision of this Manifold's vertices, which tracks the
   * approximate rounding error over all the transforms and operations that have
   * led to this state. Any triangles that are colinear within this precision
   * are considered degenerate and removed. This is the value of &epsilon;
   * defining
   * [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
   */
  precision(): number;

  /**
   * The genus is a topological property of the manifold, representing the
   * number of "handles". A sphere is 0, torus 1, etc. It is only meaningful for
   * a single mesh, so it is best to call Decompose() first.
   */
  genus(): number;

  /**
   * Returns the surface area and volume of the manifold. These properties are
   * clamped to zero for a given face if they are within the Precision(). This
   * means degenerate manifolds can by identified by testing these properties as
   * == 0.
   */
  getProperties(): Properties;

  /**
   * Curvature is the inverse of the radius of curvature, and signed such that
   * positive is convex and negative is concave. There are two orthogonal
   * principal curvatures at any point on a manifold, with one maximum and the
   * other minimum. Gaussian curvature is their product, while mean
   * curvature is their sum. This approximates them for every vertex (returned
   * as vectors in the structure) and also returns their minimum and maximum
   * values.
   */
  getCurvature(): Curvature;

  /**
   * Returns a Mesh that is designed to easily push into a renderer, including
   * all interleaved vertex properties that may have been input. It also
   * includes relations to all the input meshes that form a part of this result
   * and the transforms applied to each.
   *
   * @param normalIdx If the original Mesh inputs that formed this manifold had
   * properties corresponding to normal vectors, you can specify which property
   * channels these are (x, y, z), which will cause this output Mesh to
   * automatically update these normals according to the applied transforms and
   * front/back side. Each channel must be >= 3 and < numProp, and all original
   * Meshes must use the same channels for their normals.
   */
  getMesh(normalIdx?: Vec3): Mesh;

  /**
   * If you copy a manifold, but you want this new copy to have new properties
   * (e.g. a different UV mapping), you can reset its IDs to a new original,
   * meaning it will now be referenced by its descendants instead of the meshes
   * it was built from, allowing you to differentiate the copies when applying
   * your properties to the final result.
   *
   * This function also condenses all coplanar faces in the relation, and
   * collapses those edges. If you want to have inconsistent properties across
   * these faces, meaning you want to preserve some of these edges, you should
   * instead call GetMesh(), calculate your properties and use these to
   * construct a new manifold.
   */
  asOriginal(): Manifold;

  /**
   * If this mesh is an original, this returns its ID that can be referenced
   * by product manifolds. If this manifold is a product, this
   * returns -1.
   */
  originalID(): number;
}

export class Mesh {
  constructor(options: {
    numProp: number,
    vertProperties: Float32Array,
    triVerts: Uint32Array,
    mergeFromVert?: Uint32Array,
    mergeToVert?: Uint32Array,
    runIndex?: Uint32Array,
    runOriginalID?: Uint32Array,
    runTransform?: Float32Array,
    faceID?: Uint32Array,
    halfedgeTangent?: Float32Array
  });
  numProp: number;
  vertProperties: Float32Array;
  triVerts: Uint32Array;
  mergeFromVert?: Uint32Array;
  mergeToVert?: Uint32Array;
  runIndex?: Uint32Array;
  runOriginalID?: Uint32Array;
  runTransform?: Float32Array;
  faceID?: Uint32Array;
  halfedgeTangent?: Float32Array;
  get numTri(): number;
  get numVert(): number;
  get numRun(): number;
  verts(tri: number): SealedUint32Array<3>;
  position(vert: number): SealedFloat32Array<3>;
  extras(vert: number): Float32Array;
  tangent(halfedge: number): SealedFloat32Array<4>;
  transform(run: number): Mat4;
}