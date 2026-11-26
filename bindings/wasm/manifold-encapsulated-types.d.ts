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
 * @primaryExport
 */

import {Box, CircularQuality, ErrorStatus, ExecutionContext, JoinType, Mat3, Mat4, Polygons, RayHit, Rect, SealedFloat32Array, SealedFloat64Array, SealedUint32Array, SimplePolygon, Vec2, Vec3} from './manifold-global-types';

export {ExecutionContext} from './manifold-global-types';

/**
 * Triangulates a set of epsilon-valid polygons.
 *
 * @param polygons The set of polygons, wound CCW and representing multiple
 * polygons and/or holes.
 * @param epsilon The value of epsilon, bounding the uncertainty of the input.
 * @param allowConvex If true (default), the triangulator will use a fast
 * triangulation if the input is convex, falling back to ear-clipping if not.
 * The triangle quality may be lower, so set to false to disable this
 * optimization.
 * @return The triangles, referencing the original polygon points in order.
 *
 * @see {@link https://manifoldcad.org/docs/html/group___triangulation.html | C++ API: Triangulation}
 */
export function triangulate(
    polygons: Polygons, epsilon?: number, allowConvex?: boolean): Vec3[];

/**
 * The finest relative precision a Manifold can have, 1e-12. Pass it to
 * {@link Manifold.withPrecision} to request the finest.
 */
export const MIN_RELATIVE_PRECISION: number;

/**
 * Returns the number of segments a circle of this radius gets with this
 * quality.
 *
 * @param radius For a given radius of circle, determine how many segments
 * there will be.
 * @param quality The circular quality, see {@link CircularQuality}. Omit it
 * for the defaults.
 */
export function segmentsFor(radius: number, quality?: CircularQuality): number;

/**
 * Returns a quality that makes {@link Manifold.revolve} divide the given angle
 * into exactly this many divisions. Like revolve, it uses the angle's
 * magnitude, clamped to 360 degrees. A revolve of more than 120 degrees needs
 * at least two divisions, and of more than 240 degrees at least three, and a
 * tiny angle cannot get more divisions than the largest int count allows.
 * Past those limits it returns the closest count revolve allows. A non-finite
 * or zero angle, which revolve rejects, gives the default quality.
 *
 * @param divisions Number of divisions. Values below 1 are treated as 1.
 * @param degrees Angle revolve will sweep.
 */
export function forDivisions(divisions: number, degrees: number):
    {segments: number, minAngle: number, minEdgeLength: number};

/**
 * Returns the version of the Manifold library.
 *
 * @return Version as [major, minor, patch].
 */
export function version(): Vec3;
///@}

/**
 * Two-dimensional cross sections guaranteed to be without self-intersections,
 * or overlaps between polygons (from construction onwards). Polygon clipping
 * (boolean) and offsetting use Manifold's own robust floating-point predicates.
 *
 * @see {@link https://manifoldcad.org/docs/html/classmanifold_1_1_cross_section.html | C++ API: CrossSection Class Reference}
 * @group Basics
 */
export class CrossSection {
  /**
   * Create a 2d cross-section from a set of contours (complex polygons). A
   * boolean union operation (with the Positive filling rule) is performed to
   * combine overlapping polygons and ensure the resulting CrossSection is free
   * of intersections.
   *
   * @param contours A set of closed paths describing zero or more complex
   * polygons.
   * @group Basics
   */
  constructor(contours: Polygons);

  // Shapes

  /**
   * Constructs a square with the given XY dimensions. By default it is
   * positioned in the first quadrant, touching the origin. If any dimensions in
   * size are negative or non-finite, or if all are zero, an empty CrossSection
   * will be returned.
   *
   * @param size The X, and Y dimensions of the square.
   * @param center Set to true to shift the center to the origin.
   * @group Constructors
   */
  static square(size?: Readonly<Vec2>|number, center?: boolean): CrossSection;

  /**
   * Constructs a circle of a given radius.
   *
   * @param radius Radius of the circle. Must be positive.
   * @param quality Number of segments in a full circle, see
   * {@link CircularQuality}.
   * @group Constructors
   */
  static circle(radius: number, quality?: CircularQuality): CrossSection;

  // Extrusions (2d to 3d manifold)

  /**
   * Constructs a manifold by extruding the cross-section along Z-axis.
   *
   * @param height Z-extent of extrusion.
   * @param nDivisions Number of extra copies of the crossSection to insert into
   * the shape vertically; especially useful in combination with twistDegrees to
   * avoid interpolation artifacts. Default is none.
   * @param twistDegrees Amount to twist the top crossSection relative to the
   * bottom, interpolated linearly for the divisions in between.
   * @param scaleTop Amount to scale the top (independently in X and Y). A
   * single number scales X and Y equally. If the scale is {0, 0}, a pure cone
   * is formed with only a single vertex at the top. Default {1, 1}.
   * @param center If true, the extrusion is centered on the z-axis through the
   *     origin
   * as opposed to resting on the XY plane as is default.
   * @group Transformations
   */
  extrude(
      height: number, nDivisions?: number, twistDegrees?: number,
      scaleTop?: Readonly<Vec2>|number, center?: boolean): Manifold;

  /**
   * Constructs a manifold by revolving this cross-section around its Y-axis and
   * then setting this as the Z-axis of the resulting manifold. If the contours
   * cross the Y-axis, only the part on the positive X side is used.
   * Geometrically valid input will result in geometrically valid output.
   *
   * @param quality Number of segments in a full circle, see
   * {@link CircularQuality}. A partial revolve uses the matching fraction of
   * it, rounded up, ignoring an excess of one unit in the last place. For an
   * exact number of divisions, use {@link forDivisions}.
   * @param revolveDegrees Number of degrees to revolve. Must be finite and
   * nonzero. A negative value revolves the other way. Default is 360 degrees.
   * @group Transformations
   */
  revolve(quality?: CircularQuality, revolveDegrees?: number): Manifold;

  // Transformations

  /**
   * Transform this CrossSection in space. Stored in column-major order. This
   * operation can be chained. Transforms are combined and applied lazily.
   *
   * @param m The affine transformation matrix to apply to all the vertices. The
   *     last row is ignored.
   * @group Transformations
   */
  transform(m: Mat3): CrossSection;

  /**
   * Move this CrossSection in space. This operation can be chained. Transforms
   * are combined and applied lazily.
   *
   * @param v The vector to add to every vertex.
   * @group Transformations
   */
  translate(v: Readonly<Vec2>): CrossSection;
  translate(x: number, y?: number): CrossSection;

  /**
   * Applies a (Z-axis) rotation to the CrossSection, in degrees. This operation
   * can be chained. Transforms are combined and applied lazily.
   *
   * @param degrees degrees about the Z-axis to rotate.
   * @group Transformations
   */
  rotate(degrees: number): CrossSection;

  /**
   * Scale this CrossSection in space. This operation can be chained. Transforms
   * are combined and applied lazily.
   *
   * @param v The vector to multiply every vertex by per component.
   * @group Transformations
   */
  scale(v: Readonly<Vec2>|number): CrossSection;


  /**
   * Mirror this CrossSection over the arbitrary axis described by the unit form
   * of the given vector. If the length of the vector is zero, an empty
   * CrossSection is returned. This operation can be chained. Transforms are
   * combined and applied lazily.
   *
   * @param ax the axis to be mirrored over
   * @group Transformations
   */
  mirror(ax: Readonly<Vec2>): CrossSection;

  /**
   * Move the vertices of this CrossSection (creating a new one) according to
   * any arbitrary input function, followed by a union operation (with a
   * Positive fill rule) that ensures any introduced intersections are not
   * included in the result.
   *
   * @param warpFunc A function that modifies a given vertex position.
   * @group Transformations
   */
  warp(warpFunc: (vert: Vec2) => void): CrossSection;

  /**
   * Batch version of warp(). The callback receives the vertices of all
   * contours, in contour order, as a flat array of xyxy... (length = count *
   * 2) and may modify it in-place. Rethrows any error from warpFunc.
   *
   * @param warpFunc A function that modifies the given vertex positions.
   * @group Transformations
   */
  warpBatch(warpFunc: (verts: Float64Array, count: number) => void):
      CrossSection;

  /**
   * Inflate the contours in CrossSection by the specified delta, handling
   * corners according to the given JoinType.
   *
   * @param delta Positive deltas will cause the expansion of outlining contours
   * to expand, and retraction of inner (hole) contours. Negative deltas will
   * have the opposite effect.
   * @param joinType The join type specifying the treatment of contour joins
   * (corners). Defaults to Round
   * @param miterLimit The maximum distance in multiples of delta that vertices
   * can be offset from their original positions with before squaring is
   * applied, **when the join type is Miter** (default is 2, which is the
   * minimum allowed). See the [Clipper2
   * MiterLimit](http://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/Properties/MiterLimit.htm)
   * page for a visual example.
   * @param quality Number of segments per 360 degrees of
   * <B>JoinType::Round</B> corners (roughly, the number of vertices that
   * will be added to each contour), see {@link CircularQuality}.
   * @group Transformations
   */
  offset(
      delta: number, joinType?: JoinType, miterLimit?: number,
      quality?: CircularQuality): CrossSection;

  /**
   * Remove vertices from the contours in this CrossSection that are less than
   * the specified distance tolerance from an imaginary line that passes through
   * its two adjacent vertices. Near duplicate vertices and collinear points
   * will be removed at lower tolerances, with elimination of line segments
   * becoming increasingly aggressive with larger tolerances.
   *
   * It is recommended to apply this function following Offset, in order to
   * clean up any spurious tiny line segments introduced that do not improve
   * quality in any meaningful way. This is particularly important if further
   * offseting operations are to be performed, which would compound the issue.
   *
   * @param tolerance minimum distance vertices must diverge from the
   *     hypothetical outline without them in order to be included in the
   *     output. Default 0 uses the cross-section's own tolerance, which is
   *     geometry-scale-derived and may be larger than a fixed epsilon for
   *     large-coordinate geometry. Pass an explicit value to override.
   * @group Transformations
   */
  simplify(tolerance?: number): CrossSection;

  // Clipping Operations

  /**
   * Boolean union
   * @group Boolean
   */
  add(other: CrossSection|Polygons): CrossSection;

  /**
   * Boolean difference
   * @group Boolean
   */
  subtract(other: CrossSection|Polygons): CrossSection;

  /**
   * Boolean intersection
   * @group Boolean
   */
  intersect(other: CrossSection|Polygons): CrossSection;

  /**
   * Boolean union of the cross-sections a and b
   * @group Boolean
   */
  static union(a: CrossSection|Polygons, b: CrossSection|Polygons):
      CrossSection;

  /**
   * Boolean difference of the cross-section b from the cross-section a
   * @group Boolean
   */
  static difference(a: CrossSection|Polygons, b: CrossSection|Polygons):
      CrossSection;

  /**
   * Boolean intersection of the cross-sections a and b
   * @group Boolean
   */
  static intersection(a: CrossSection|Polygons, b: CrossSection|Polygons):
      CrossSection;

  /**
   * Boolean union of a list of cross-sections
   * @group Boolean
   */
  static union(polygons: readonly(CrossSection|Polygons)[]): CrossSection;

  /**
   * Boolean difference of the tail of a list of cross-sections from its head
   * @group Boolean
   */
  static difference(polygons: readonly(CrossSection|Polygons)[]): CrossSection;

  /**
   * Boolean intersection of a list of cross-sections
   */
  static intersection(polygons: readonly(CrossSection|Polygons)[]):
      CrossSection;

  // Convex Hulls

  /**
   * Compute the convex hull of the contours in this CrossSection.
   * @group Convex Hull
   */
  hull(): CrossSection;

  /**
   * Compute the convex hull of all points in a list of polygons/cross-sections.
   * @group Convex Hull
   */
  static hull(polygons: readonly(CrossSection|Polygons)[]): CrossSection;

  // Topological Operations

  /**
   * This operation returns a vector of CrossSections that are topologically
   * disconnected, each containing one outline contour with zero or more
   * holes.
   * @group Constructors
   */
  decompose(): CrossSection[];

  // Polygon Conversion

  /**
   * Create a 2d cross-section from a set of contours, reading them by the
   * even-odd rule: a sub-region is filled when its winding number is odd, so
   * overlapping an odd number of contours fills and overlapping an even number
   * leaves a hole. Use this to take input authored for an even-odd tool such
   * as OpenSCAD. The result is regularized like any other CrossSection, so
   * every later operation behaves the same.
   *
   * @param contours A set of closed paths describing zero or more complex
   * polygons.
   * @group Input & Output
   */
  static evenOdd(contours: Polygons): CrossSection;

  /**
   * Create a 2d cross-section from a set of contours, reading them by the
   * nonzero rule: a sub-region is filled when its signed winding number is
   * not zero. Both clockwise and counter-clockwise contours can enclose solid
   * regions; opposite windings cancel where their sum is zero. Unlike even-odd
   * fill, a winding of two remains filled. The result is regularized like any
   * other CrossSection, so every later operation behaves the same.
   *
   * @param contours A set of closed paths describing zero or more complex
   * polygons.
   * @group Input & Output
   */
  static nonZero(contours: Polygons): CrossSection;

  /**
   * Return the contours of this CrossSection as a list of simple polygons.
   * @group Input & Output
   */
  toPolygons(): SimplePolygon[];

  // Properties

  /**
   * Return the total area covered by complex polygons making up the
   * CrossSection.
   * @group Information
   */
  area(): number;

  /**
   * Does the CrossSection (not) have any contours?
   * @group Information
   */
  isEmpty(): boolean;

  /**
   * The number of vertices in the CrossSection.
   * @group Information
   */
  numVert(): number;

  /**
   * The number of contours in the CrossSection.
   * @group Information
   */
  numContour(): number;

  /**
   * Returns the axis-aligned bounding rectangle of all the CrossSection's
   * vertices.
   * @group Information
   */
  bounds(): Rect;

  // Memory

  /**
   * Frees the WASM memory of this CrossSection, since these cannot be
   * garbage-collected automatically.
   * @group Basics
   */
  delete(): void;
}

/**
 * This library's internal representation of an oriented, 2-manifold, triangle
 * mesh - a simple boundary-representation of a solid object. Use this class to
 * store and operate on solids, and use Mesh for input and output.
 *
 * In addition to storing geometric data, a Manifold can also store an arbitrary
 * number of vertex properties. These could be anything, e.g. normals, UV
 * coordinates, colors, etc, but this library is completely agnostic. All
 * properties are merely float values indexed by channel number. It is up to the
 * user to associate channel numbers with meaning.
 *
 * Manifold allows vertex properties to be shared for efficient storage, or to
 * have multiple property verts associated with a single geometric vertex,
 * allowing sudden property changes, e.g. at Boolean intersections, without
 * sacrificing manifoldness.
 *
 * Manifolds also keep track of their relationships to their inputs, via
 * OriginalIDs and the faceIDs and transforms accessible through Mesh. This
 * allows object-level properties to be re-associated with the output after many
 * operations, particularly useful for materials. Since separate object's
 * properties are not mixed, there is no requirement that channels have
 * consistent meaning between different inputs.
 *
 * @see {@link https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html | C++ API: Manifold Class Reference}
 * @group Basics
 */
export class Manifold {
  /**
   * Convert a Mesh into a Manifold, retaining its properties and merging only
   * the positions according to the merge vectors. Throws a ManifoldError, with
   * the error status as its code, if the mesh is invalid or the result is not
   * an oriented 2-manifold. Triangles with a repeated vertex after merging are
   * dropped. Other degenerate triangles are kept. Use simplify() to collapse
   * them. For cancellable ingest that reports progress, use
   * {@link ExecutionContext.fromMesh}.
   *
   * All fields are read, making this structure suitable for a lossless
   * round-trip of data from getMesh(). For multi-material input, use
   * reserveIDs() to set a unique originalID for each material, and sort the
   * materials into triangle runs.
   *
   * A Mesh64 is read in double precision, so the input positions are not
   * rounded to float. A plain object is read as a Mesh. To read a plain object
   * in double precision, wrap it with new Mesh64(obj).
   *
   * The result has relative precision float epsilon, about 1.2e-7, for a Mesh
   * and {@link MIN_RELATIVE_PRECISION} (1e-12) for a Mesh64. Precision is not
   * stored in the mesh, so use {@link withPrecision} to change it.
   *
   * @group Basics
   */
  constructor(mesh: Mesh|Mesh64);

  // Shapes

  /**
   * Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
   * and the rest at similarly symmetric points.
   * @group Constructors
   */
  static tetrahedron(): Manifold;

  /**
   * Constructs a unit cube (edge lengths all one), by default in the first
   * octant, touching the origin.
   *
   * @param size The X, Y, and Z dimensions of the box.
   * @param center Set to true to shift the center to the origin.
   * @group Constructors
   */
  static cube(size?: Readonly<Vec3>|number, center?: boolean): Manifold;

  /**
   * A convenience constructor for the common case of extruding a circle. Can
   * also form cones if both radii are specified.
   *
   * @param height Z-extent
   * @param radiusLow Radius of bottom circle. Must be positive.
   * @param radiusHigh Radius of top circle. Can equal zero. Default is equal to
   * radiusLow.
   * @param quality How many line segments to use around the circle, see
   * {@link CircularQuality}.
   * @param center Set to true to shift the center to the origin. Default is
   * origin at the bottom.
   * @group Constructors
   */
  static cylinder(
      height: number, radiusLow: number, radiusHigh?: number,
      quality?: CircularQuality, center?: boolean): Manifold;

  /**
   * Constructs a geodesic sphere of a given radius.
   *
   * @param radius Radius of the sphere. Must be positive.
   * @param quality Number of segments in a full circle, see
   * {@link CircularQuality}. This number will always be rounded up to the
   * nearest factor of four, as this sphere is constructed by refining an
   * octahedron. This means there are a circle of vertices on all three of the
   * axis planes.
   * @group Constructors
   */
  static sphere(radius: number, quality?: CircularQuality): Manifold;

  // Extrusions from 2d shapes

  /**
   * Constructs a manifold from a set of polygons/cross-section by extruding
   * them along the Z-axis.
   *
   * @param polygons A set of non-overlapping polygons to extrude.
   * @param height Z-extent of extrusion.
   * @param nDivisions Number of extra copies of the crossSection to insert into
   * the shape vertically; especially useful in combination with twistDegrees to
   * avoid interpolation artifacts. Default is none.
   * @param twistDegrees Amount to twist the top crossSection relative to the
   * bottom, interpolated linearly for the divisions in between.
   * @param scaleTop Amount to scale the top (independently in X and Y). A
   * single number scales X and Y equally. If the scale is {0, 0}, a pure cone
   * is formed with only a single vertex at the top. Default {1, 1}.
   * @param center If true, the extrusion is centered on the z-axis through the
   *     origin
   * as opposed to resting on the XY plane as is default.
   * @group Polygons
   */
  static extrude(
      polygons: CrossSection|Polygons, height: number, nDivisions?: number,
      twistDegrees?: number, scaleTop?: Readonly<Vec2>|number,
      center?: boolean): Manifold;

  /**
   * Constructs a manifold from a set of polygons/cross-section by revolving
   * them around the Y-axis and then setting this as the Z-axis of the resulting
   * manifold. If the polygons cross the Y-axis, only the part on the positive X
   * side is used. Geometrically valid input will result in geometrically valid
   * output.
   *
   * @param polygons A set of non-overlapping polygons to revolve.
   * @param quality Number of segments in a full circle, see
   * {@link CircularQuality}. A partial revolve uses the matching fraction of
   * it, rounded up, ignoring an excess of one unit in the last place. For an
   * exact number of divisions, use {@link forDivisions}.
   * @param revolveDegrees Number of degrees to revolve. Must be finite and
   * nonzero. A negative value revolves the other way. Default is 360 degrees.
   * @group Polygons
   */
  static revolve(
      polygons: CrossSection|Polygons, quality?: CircularQuality,
      revolveDegrees?: number): Manifold;

  // Signed Distance Functions

  /**
   * Constructs a level-set manifold from the input Signed-Distance Function
   * (SDF). This uses a form of Marching Tetrahedra (akin to Marching Cubes, but
   * better for manifoldness). Instead of using a cubic grid, it uses a
   * body-centered cubic grid (two shifted cubic grids). This means if your
   * function's interior exceeds the given bounds, you will see a kind of
   * egg-crate shape closing off the manifold, which is due to the underlying
   * grid.
   *
   * @param sdf The signed-distance function which returns the signed distance
   *     of
   * a given point in R^3. Positive values are inside, negative outside.
   * @param bounds An axis-aligned box that defines the extent of the grid.
   * @param edgeLength Approximate maximum edge length of the triangles in the
   * final result. This affects grid spacing, and hence has a strong effect on
   * performance.
   * @param level You can inset your Mesh by using a positive value, or outset
   * it with a negative value.
   * @param tolerance Ensure each vertex is within this distance of the true
   * surface. Defaults to -1, which will return the interpolated
   * crossing-point based on the two nearest grid points. Small positive values
   * will require more sdf evaluations per output vertex.
   * @group Constructors
   */
  static levelSet(
      sdf: (point: Vec3) => number, bounds: Box, edgeLength: number,
      level?: number, tolerance?: number): Manifold;

  // Transformations

  /**
   * Transform this Manifold in space. Stored in column-major order. This
   * operation can be chained. Transforms are combined and applied lazily.
   *
   * @param m The affine transformation matrix to apply to all the vertices. The
   *     last row is ignored.
   * @group Transformations
   */
  transform(m: Mat4): Manifold;

  /**
   * Move this Manifold in space. This operation can be chained. Transforms are
   * combined and applied lazily.
   *
   * @param v The vector to add to every vertex.
   * @group Transformations
   */
  translate(v: Readonly<Vec3>): Manifold;
  translate(x: number, y?: number, z?: number): Manifold;

  /**
   * Applies an Euler or Tait-Bryan angle rotation to the manifold.  This
   * operation can be chained. Transforms are combined and applied lazily.
   *
   * We use degrees so that we can minimize rounding error, and eliminate it
   * completely for any multiples of 90 degrees. Additionally, more efficient
   * code paths are used to update the manifold when the transforms only rotate
   * by multiples of 90 degrees.
   *
   * From the reference frame of the model being rotated, rotations are applied
   * in *z-y'-x"* order. That is yaw first, then pitch and finally roll.
   *
   * From the global reference frame, a model will be rotated in *x-y-z* order.
   * That is about the global X axis, then global Y axis, and finally global Z.
   *
   * @param v [X, Y, Z] rotation in degrees.
   * @group Transformations
   */
  rotate(v: Readonly<Vec3>): Manifold;
  rotate(x: number, y?: number, z?: number): Manifold;

  /**
   * Scale this Manifold in space. This operation can be chained. Transforms are
   * combined and applied lazily.
   *
   * @param v The vector to multiply every vertex by per component.
   * @group Transformations
   */
  scale(v: Readonly<Vec3>|number): Manifold;

  /**
   * Mirror this Manifold over the plane described by the unit form of the given
   * normal vector. If the length of the normal is zero, an empty Manifold is
   * returned. This operation can be chained. Transforms are combined and
   * applied lazily.
   *
   * @param normal The normal vector of the plane to be mirrored over
   * @group Transformations
   */
  mirror(normal: Readonly<Vec3>): Manifold;

  /**
   * This function does not change the topology, but allows the vertices to be
   * moved according to any arbitrary input function. It is easy to create a
   * function that warps a geometrically valid object into one which overlaps,
   * but that is not checked here, so it is up to the user to choose their
   * function with discretion. Throws a ManifoldError if the result has an
   * error status, such as a non-finite vertex, and rethrows any error from
   * warpFunc.
   *
   * @param warpFunc A function that modifies a given vertex position.
   * @group Transformations
   */
  warp(warpFunc: (vert: Vec3) => void): Manifold;

  /**
   * Batch version of warp(). The callback receives a flat array of xyzxyz...
   * (length = count * 3) and may modify it in-place. Throws like warp().
   */
  warpBatch(warpFunc: (verts: Float64Array, count: number) => void): Manifold;

  /**
   * Smooths out the Manifold by filling in the halfedgeTangent vectors. The
   * geometry will remain unchanged until refine(), refineToLength() or
   * refineToTolerance() is called to interpolate the surface. This version
   * uses standard vertex normal properties to define the tangent vectors. Fewer
   * than three properties count as missing normals. Any vertex where multiple
   * normals exist and don't agree will result in a sharp edge.
   * @group Smoothing
   */
  smoothByNormals(): Manifold;

  /**
   * Increase the density of the mesh by splitting every edge into n pieces. For
   * instance, with n = 2, each triangle will be split into 4 triangles. These
   * will all be coplanar (and will not be immediately collapsed) unless the
   * Mesh/Manifold has halfedgeTangents specified (e.g. from
   * smoothByNormals()), in which case the new vertices will be moved to the
   * interpolated surface according to their barycentric coordinates.
   *
   * @param n The number of pieces to split every edge into. Must be > 1.
   * @group Smoothing
   */
  refine(n: number): Manifold;

  /**
   * Increase the density of the mesh by splitting each edge into pieces of
   * roughly the input length. Interior verts are added to keep the rest of the
   * triangulation edges also of roughly the same length. If halfedgeTangents
   * are present (e.g. from smoothByNormals()), the new vertices will be
   * moved to the interpolated surface according to their barycentric
   * coordinates.
   *
   * @param length The length that edges will be broken down to. Its sign is
   * ignored, and zero or NaN is an error.
   * @group Smoothing
   */
  refineToLength(length: number): Manifold;

  /**
   * Increase the density of the mesh by splitting each edge into pieces such
   * that any point on the resulting triangles is roughly within tolerance of
   * the smoothly curved surface defined by the tangent vectors. This means
   * tightly curving regions will be divided more finely than smoother regions.
   * If halfedgeTangents are not present, the result will simply be a copy of
   * the original. Quads will ignore their interior triangle bisector.
   *
   * @param tolerance The desired maximum distance between the faceted mesh
   * produced and the exact smoothly curving surface. All vertices are exactly
   * on the surface, within rounding error. Its sign is ignored, and zero or NaN
   * is an error.
   * @group Smoothing
   */
  refineToTolerance(tolerance: number): Manifold;

  /**
   * Create a new copy of this manifold with updated vertex properties by
   * supplying a function that takes the existing position and properties as
   * input. You may specify any number of output properties, allowing creation
   * and removal of channels. Note: undefined behavior will result if you read
   * past the number of input properties or write past the number of output
   * properties.
   *
   * @param numProp The new number of properties per vertex.
   * @param propFunc A function that modifies the properties of a given vertex.
   * @group Properties
   */
  setProperties(
      numProp: number,
      propFunc: (newProp: number[], position: Vec3, oldProp: number[]) => void):
      Manifold;

  /**
   * Curvature is the inverse of the radius of curvature, and signed such that
   * positive is convex and negative is concave. There are two orthogonal
   * principal curvatures at any point on a manifold, with one maximum and the
   * other minimum. Gaussian curvature is their product, while mean
   * curvature is their sum. This approximates them for every vertex and assigns
   * them as vertex properties on the given channels.
   *
   * @param gaussianIdx The property channel index in which to store the
   *     Gaussian curvature. An index < 0 will be ignored (stores nothing). The
   *     property set will be automatically expanded to include the channel
   *     index specified.
   * @param meanIdx The property channel index in which to store the mean
   *     curvature. An index < 0 will be ignored (stores nothing). The property
   *     set will be automatically expanded to include the channel index
   *     specified.
   * @group Properties
   */
  calculateCurvature(gaussianIdx: number, meanIdx: number): Manifold;

  /**
   * Fills in vertex properties for normal vectors, calculated from the mesh
   * geometry. Flat faces composed of three or more triangles will remain flat.
   *
   * Normals are stored in the first three extra property channels. The
   * Manifold records them so a subsequent getMesh() returns world-frame
   * normals. Normals supplied by setProperties() are not recorded; use this
   * instead of setProperties() on every operand when normals must follow
   * transforms.
   *
   * @param minSharpAngle Any edges with angles greater than this value will
   * remain sharp, getting different normal vector properties on each side of
   * the edge. Default is 52.5 degrees. With a value of zero, the model is
   * faceted and all normals match their triangle normals, but in this case it
   * would be better not to calculate normals at all.
   * @group Properties
   */
  calculateNormals(minSharpAngle?: number): Manifold;

  // Boolean Operations

  /**
   * Boolean union
   *
   * @group Boolean
   */
  add(other: Manifold): Manifold;

  /**
   * Boolean difference
   *
   * @group Boolean
   */
  subtract(other: Manifold): Manifold;

  /**
   * Boolean intersection
   *
   * @group Boolean
   */
  intersect(other: Manifold): Manifold;

  /**
   * Boolean union of the manifolds a and b
   *
   * @group Boolean
   */
  static union(a: Manifold, b: Manifold): Manifold;

  /**
   * Boolean difference of the manifold b from the manifold a
   *
   * @group Boolean
   */
  static difference(a: Manifold, b: Manifold): Manifold;

  /**
   * Boolean intersection of the manifolds a and b
   *
   * @group Boolean
   */
  static intersection(a: Manifold, b: Manifold): Manifold;

  /**
   * Boolean union of a list of manifolds
   *
   * @group Boolean
   */
  static union(manifolds: readonly Manifold[]): Manifold;

  /**
   * Boolean difference of the tail of a list of manifolds from its head
   *
   * @group Boolean
   */
  static difference(manifolds: readonly Manifold[]): Manifold;

  /**
   * Boolean intersection of a list of manifolds
   *
   * @group Boolean
   */
  static intersection(manifolds: readonly Manifold[]): Manifold;

  /**
   * Split cuts this manifold in two using the cutter manifold. The first result
   * is the intersection, second is the difference. This is more efficient than
   * doing them separately.
   *
   * @param cutter
   * @group Boolean
   */
  split(cutter: Manifold): [Manifold, Manifold];

  /**
   * Convenient version of split() for a half-space.
   *
   * @param normal This vector is normal to the cutting plane and its length
   *     does
   * not matter. The first result is in the direction of this vector, the second
   * result is on the opposite side.
   * @param originOffset The distance of the plane from the origin in the
   * direction of the normal vector. Default 0.
   * @group Boolean
   */
  splitByPlane(normal: Readonly<Vec3>, originOffset?: number):
      [Manifold, Manifold];

  /**
   * Removes everything behind the given half-space plane.
   *
   * @param normal This vector is normal to the cutting plane and its length
   *     does not matter. The result is in the direction of this vector from the
   *     plane.
   * @param originOffset The distance of the plane from the origin in the
   *     direction of the normal vector. Default 0.
   *
   * @group Boolean
   */
  trimByPlane(normal: Readonly<Vec3>, originOffset?: number): Manifold;

  /**
   * Compute the minkowski sum of this manifold with another.
   * This corresponds to the morphological dilation of the manifold.
   *
   * @param other The other manifold to minkowski sum to this one.
   * @group Boolean
   */
  minkowskiSum(other: Manifold): Manifold;

  /**
   * Subtract the sweep of the other manifold across this manifold's surface.
   * This corresponds to the morphological erosion of the manifold.
   *
   * @param other The other manifold to minkowski subtract from this one.
   * @group Boolean
   */
  minkowskiDifference(other: Manifold): Manifold;

  /**
   * Returns the cross section of this object parallel to the X-Y plane at the
   * specified height. Using a height equal to the bottom
   * of the bounding box will return the bottom faces, while using a height
   * equal to the top of the bounding box will return empty.
   *
   * @param height Z-level of slice. Default 0.
   * @group Polygons
   */
  slice(height?: number): CrossSection;

  /**
   * Returns a cross section representing the projected outline of this object
   * onto the X-Y plane.
   *
   * @group Polygons
   */
  project(): CrossSection;

  // Convex Hulls

  /**
   * Compute the convex hull of all points in this Manifold.
   *
   * @group Convex Hull
   */
  hull(): Manifold;

  /**
   * Compute the convex hull of all points contained within a set of Manifolds
   * and point vectors. The hull uses the smallest precision among the
   * non-empty input Manifolds.
   *
   * @group Convex Hull
   */
  static hull(points: readonly(Manifold|Vec3)[]): Manifold;

  // Topological Operations

  /**
   * This operation returns a vector of Manifolds that are topologically
   * disconnected. If everything is connected, the vector is length one,
   * containing a copy of the original.
   *
   * @group Constructors
   */
  decompose(): Manifold[];

  // Property Access

  /**
   * Does the Manifold have any triangles?
   * @group Information
   */
  isEmpty(): boolean;

  /**
   * The number of vertices in the Manifold.
   * @group Information
   */
  numVert(): number;

  /**
   * The number of triangles in the Manifold.
   * @group Information
   */
  numTri(): number;

  /**
   * The number of edges in the Manifold.
   * @group Information
   */
  numEdge(): number;

  /**
   * The number of properties per vertex in the Manifold.
   * @group Information
   */
  numProp(): number;

  /**
   * The number of property vertices in the Manifold. This will always be >=
   * numVert, as some physical vertices may be duplicated to account for
   * different properties on different neighboring triangles.
   * @group Information
   */
  numPropVert(): number

  /**
   * Returns the axis-aligned bounding box of all the Manifold's vertices.
   * @group Information
   */
  boundingBox(): Box;

  /**
   * Return a simplified copy of the manifold. The tolerance controls the local
   * error when collapsing edges, measured relative to nearby face planes.
   * Simplification tends to remove features thinner than the tolerance.
   *
   * @param tolerance Simplification tolerance, lower-bounded by the manifold's
   *     epsilon.
   * @group Transformations
   */
  simplify(tolerance?: number): Manifold;

  /**
   * The genus is a topological property of the manifold, representing the
   * number of "handles". A sphere is 0, torus 1, etc. It is only meaningful for
   * a single mesh, so it is best to call decompose() first.
   * @group Information
   */
  genus(): number;

  /**
   * Returns the surface area of the manifold.
   *
   * @group Measurement
   */
  surfaceArea(): number;

  /**
   * Returns the volume of the manifold. Negative volume is considered a class
   * of overlapped geometry and should be avoided as much as self-intersection.
   *
   * @group Measurement
   */
  volume(): number;

  /**
   * Returns the minimum gap between two manifolds. Returns a float between
   * 0 and searchLength.
   *
   * @group Measurement
   */
  minGap(other: Manifold, searchLength: number): number;

  /**
   * Cast a ray segment, returning all hits sorted by distance.
   *
   * @param origin The start point of the ray segment.
   * @param endpoint The end point of the ray segment.
   * @returns Array of RayHit sorted by distance, empty on miss.
   *
   * @group Spatial Queries
   */
  rayCast(origin: Vec3, endpoint: Vec3): RayHit[];

  /**
   * Returns the winding number of this manifold around each of the given
   * points. Returns 0 for points outside and 1 for points
   * inside a non-overlapped manifold. Any other values indicate overlapping
   * regions, e.g. -1 for inverted regions, 2 for double-enclosed regions.
   *
   * @param points The 3D query points.
   *
   * @group Spatial Queries
   */
  windingNumber(points: Vec3[]): number[];



  /**
   * Returns the reason for an error that made this Manifold empty. The
   * constructor throws instead, but {@link ExecutionContext.fromMesh} returns
   * an empty Manifold with this status for an invalid mesh, and a cancelled
   * evaluation gives 'Cancelled'. This Status will carry on through operations
   * like NaN propagation, ensuring an errored mesh doesn't get mysteriously
   * lost. Empty meshes may still show NoError, for instance the intersection
   * of non-overlapping meshes.
   *
   * @group Information
   */
  status(): ErrorStatus;

  /**
   * Returns a copy of this Manifold with the given ExecutionContext attached.
   * The attachment is consumed by the next eager op invoked on the result:
   * status(), withPrecision() or getPrecision() for a deferred CSG tree,
   * refine() / refineToLength() / refineToTolerance(), hull(), or
   * minkowskiSum() / minkowskiDifference().
   * Deferred ops (Boolean operators, transforms, batch ops) ignore any
   * attached ctx and produce a result with no attached ctx. See
   * ExecutionContext for the full model.
   *
   * @group Information
   */
  withContext(ctx: ExecutionContext): Manifold;

  /**
   * Evaluates this Manifold and returns it with the given relative precision,
   * which defines what counts as coplanar or degenerate. The geometry is not
   * recomputed: the precision applies only to later operations. Booleans and
   * other operations on several Manifolds use the smallest precision among
   * their non-empty inputs.
   *
   * @param relPrecision The relative precision. 0, negative values, NaN and
   * infinity give float epsilon. Positive values below
   * {@link MIN_RELATIVE_PRECISION} (1e-12) are raised to it, so pass it for the
   * finest.
   * @group Information
   */
  withPrecision(relPrecision: number): Manifold;

  /**
   * Returns the relative precision of this Manifold. A Manifold made from a
   * Mesh or built by a factory such as cube(), extrude(), levelSet() or hull()
   * of points has float epsilon, about 1.2e-7. One made from a Mesh64 has
   * {@link MIN_RELATIVE_PRECISION} (1e-12). withPrecision() sets it, other
   * operations on one Manifold keep it, and operations on several take the
   * smallest among their non-empty inputs. An empty Boolean or Minkowski result
   * takes the smallest among all its inputs.
   *
   * @group Information
   */
  getPrecision(): number;

  // Export

  /**
   * Returns a Mesh that is designed to easily push into a renderer, including
   * all interleaved vertex properties that may have been input. It also
   * includes relations to all the input meshes that form a part of this result
   * and the transforms applied to each.
   *
   * For each run that records normals (runFlags bit 1), the first three extra
   * property channels contain world-frame normals. Non-original results
   * normalize them for graphics output. Unflagged values in those channels are
   * ordinary properties. To have normals follow transforms, call
   * calculateNormals() on every operand or import a Mesh with runFlags bit 1
   * set.
   * @group Input & Output
   */
  getMesh(): Mesh;

  /**
   * Like getMesh(), but returns the full double precision used internally
   * (MeshGL64): vertProperties, runTransform and halfedgeTangent are
   * Float64Array.
   * @group Input & Output
   */
  getMesh64(): Mesh64;

  // ID Management

  /**
   * This removes all relations (originalID, faceID, transform) to ancestor
   * meshes and this new Manifold is marked an original. It also recreates faces
   * - these don't get joined at boundaries where originalID changes, so the
   * reset may allow triangles of flat faces to be further collapsed with
   * simplify().
   *
   * If only some of the input recorded normals, the result keeps two runs
   * with this ID, one for the triangles with normals and one for the rest.
   *
   * @param id The ID to assign to this manifold. If negative (the default), a
   *     new ID is assigned. Use zero to match all default-constructed
   *     manifolds, thus not keeping track of the joints between input
   *     manifolds. Ensure separate IDs are used for mesh inputs containing
   *     properties, generally by calling this function without arguments just
   *     after construction.
   */
  asOriginal(id?: number): Manifold;

  /**
   * If this mesh is an original, this returns its ID that can be referenced
   * by product manifolds. If this manifold is a product, this
   * returns -1.
   *
   * @group Mesh ID
   */
  originalID(): number;

  /**
   * Returns the first of n sequential new unique mesh IDs for marking sets of
   * triangles that can be looked up after further operations. Assign to
   * Mesh.runOriginalID vector.
   *
   * @group Mesh ID
   */
  static reserveIDs(count: number): number;

  // Memory

  /**
   * Frees the WASM memory of this Manifold, since these cannot be
   * garbage-collected automatically.
   * @group Basics
   */
  delete(): void;
}

/**
 * @group Input & Output
 * @internal
 */
export interface MeshOptions {
  numProp: number;
  vertProperties: Float32Array;
  triVerts: Uint32Array;
  mergeFromVert?: Uint32Array;
  mergeToVert?: Uint32Array;
  runIndex?: Uint32Array;
  runOriginalID?: Uint32Array;
  runTransform?: Float32Array;
  runFlags?: Uint8Array;
  faceID?: Uint32Array;
  halfedgeTangent?: Float32Array;
}

/**
 * A mesh for input and output, suitable for pushing into graphics libraries
 * directly. This may not be manifold since the verts are duplicated
 * along property boundaries that do not match. The additional merge vectors
 * store this missing information, allowing the manifold to be reconstructed.
 *
 * @see {@link https://manifoldcad.org/docs/html/structmanifold_1_1_mesh_g_l_p.html | C++ API: MeshGLP< Precision, I > Struct Template Reference}
 * @group Input & Output
 * @internal
 */
export class Mesh {
  constructor(options: MeshOptions);

  /**
   * Number of properties per vertex, always >= 3.
   */
  numProp: number;

  /**
   * Flat, GL-style interleaved list of all vertex properties: propVal =
   * vertProperties[vert * numProp + propIdx]. The first three properties are
   * always the position x, y, z.
   */
  vertProperties: Float32Array;

  /**
   * The vertex indices of the three triangle corners in CCW (from the outside)
   * order, for each triangle.
   */
  triVerts: Uint32Array;

  /**
   * Optional: A list of only the vertex indicies that need to be merged to
   * reconstruct the manifold.
   */
  mergeFromVert: Uint32Array;

  /**
   * Optional: The same length as mergeFromVert, and the corresponding value
   * contains the vertex to merge with. It will have an identical position, but
   * the other properties may differ.
   */
  mergeToVert: Uint32Array;

  /**
   * Optional: Indicates runs of triangles that correspond to a particular
   * input mesh instance. The runs encompass all of triVerts and are sorted
   * by runOriginalID. Run i begins at triVerts[runIndex[i]] and ends at
   * triVerts[runIndex[i+1]]. All runIndex values are divisible by 3. Returned
   * runIndex will always be 1 longer than runOriginalID, but same length is
   * also allowed as input: triVerts.size() will be automatically appended in
   * this case.
   */
  runIndex: Uint32Array;

  /**
   * Optional: The OriginalID of the mesh this triangle run came from. This ID
   * is ideal for reapplying materials to the output mesh. Multiple runs may
   * have the same ID, e.g. representing different copies of the same input
   * mesh. If you create an input Mesh that you want to be able to reference
   * as one or more originals, be sure to set unique values from reserveIDs().
   */
  runOriginalID: Uint32Array;

  /**
   * Optional: For each run, a 3x4 transform is stored representing how the
   * corresponding original mesh was transformed to create this triangle run.
   * This matrix is stored in column-major order and the length of the overall
   * vector is either 12 * runOriginalID.size() or 0. It is empty in the output
   * of a Manifold that is an original (see originalID()), and every run then
   * has the identity transform.
   */
  runTransform: Float32Array;

  /**
   * Optional: For each run, a bitmask of flags. Bit 0 = backside (this run
   * is on the backside of its original mesh, e.g. from a subtraction).
   * Bit 1 = hasNormals (the first three extra-property channels of this run
   * hold world-frame vertex normals). See `backside(run)` / `hasNormals(run)`.
   */
  runFlags: Uint8Array;

  /**
   * Optional: Length NumTri, contains the source face ID this triangle comes
   * from. Simplification will maintain all edges between triangles with
   * different faceIDs. Input faceIDs will be maintained to the outputs,
   * otherwise this vector will be empty.
   */
  faceID: Uint32Array;

  /**
   * Optional: The X-Y-Z-W weighted tangent vectors for smooth refine(). If
   * non-empty, must be exactly four times as long as Mesh.triVerts. Indexed
   * as 4 * (3 * tri + i) + j, i < 3, j < 4, representing the tangent value
   * Mesh.triVerts[tri][i] along the CCW edge. If empty, mesh is faceted.
   */
  halfedgeTangent: Float32Array;

  /**
   * Number of triangles
   */
  get numTri(): number;

  /**
   * Number of property vertices
   */
  get numVert(): number;

  /**
   * Number of triangle runs. Each triangle run is a set of consecutive
   * triangles that all come from the same instance of the same input mesh.
   */
  get numRun(): number;

  /**
   * Returns a copy with mergeFromVert and mergeToVert vectors updated in order
   * to create a manifold solid. If the Mesh is already manifold, the merge
   * vectors are unchanged. Otherwise, verts along open edges are merged within
   * tolerance (the maximum of the specified tolerance and the baseline
   * bounding-box tolerance), keeping any from the existing merge vectors. The
   * baseline is the bounding box scale times float epsilon, the precision a
   * Manifold made from this mesh gets.
   *
   * There is no guarantee the result will be manifold - this is a best-effort
   * helper function designed primarily to aid in the case where a manifold
   * multi-material Mesh was produced, but its merge vectors were lost due to
   * a round-trip through a file format. Constructing a Manifold from the result
   * will report a Status if it is not manifold.
   *
   * @param tolerance Default 0, with a bounding-box and precision-based floor.
   */
  merge(tolerance?: number): Mesh;

  /**
   * Gets the three vertex indices of this triangle in CCW order.
   *
   * @param tri triangle index.
   */
  verts(tri: number): SealedUint32Array<3>;

  /**
   * Gets the x, y, z position of this vertex.
   *
   * @param vert vertex index.
   */
  position(vert: number): SealedFloat32Array<3>;

  /**
   * Gets any other properties associated with this vertex.
   *
   * @param vert vertex index.
   */
  extras(vert: number): Float32Array;

  /**
   * Gets the tangent vector starting at verts(tri)[j] pointing to the next
   * Bezier point along the CCW edge. The fourth value is its weight.
   *
   * @param halfedge halfedge index: 3 * tri + j, where j is 0, 1, or 2.
   */
  tangent(halfedge: number): SealedFloat32Array<4>;

  /**
   * Gets the column-major 4x4 matrix transform from the original mesh to these
   * related triangles, or the identity if runTransform does not hold one.
   *
   * @param run triangle run index.
   */
  transform(run: number): Mat4;

  /**
   * Returns true if this triangle run is on the backside compared to the
   * original mesh, e.g. from a subtraction. Informational only - the
   * framework already orients stored normals so the standard `getMesh()`
   * flow returns world-frame values regardless of this bit.
   *
   * @param run triangle run index.
   */
  backside(run: number): boolean;

  /**
   * Returns true if the first three extra-property channels of this run
   * carry world-frame vertex normals (set by `calculateNormals()` and
   * round-tripped via `runFlags` bit 1). Consumers should treat the slot
   * as normals and skip re-applying `runTransform` to it.
   *
   * hasNormals is per-run, so different runs may set it differently.
   * Behavior is undefined when a single propVert is shared by triangles
   * from runs that disagree - the slot has one interpretation, and a
   * transform rotates it for hasNormals=true and clobbers any
   * hasNormals=false sharer. Standard `calculateNormals` and boolean
   * outputs never produce that shape.
   *
   * @param run triangle run index.
   */
  hasNormals(run: number): boolean;
}

/**
 * @group Input & Output
 * @internal
 */
export interface Mesh64Options {
  numProp: number;
  vertProperties: Float64Array;
  triVerts: Uint32Array;
  mergeFromVert?: Uint32Array;
  mergeToVert?: Uint32Array;
  runIndex?: Uint32Array;
  runOriginalID?: Uint32Array;
  runTransform?: Float64Array;
  runFlags?: Uint8Array;
  faceID?: Uint32Array;
  halfedgeTangent?: Float64Array;
}

/**
 * The double-precision version of Mesh, matching MeshGL64 in C++: the vertex
 * properties, run transforms and tangents are Float64Array. Indices are
 * Uint32Array, as in Mesh. See Mesh for the meaning of each field and method.
 *
 * @see {@link https://manifoldcad.org/docs/html/structmanifold_1_1_mesh_g_l_p.html | C++ API: MeshGLP< Precision, I > Struct Template Reference}
 * @group Input & Output
 * @internal
 */
export class Mesh64 {
  constructor(options: Mesh64Options);

  numProp: number;
  vertProperties: Float64Array;
  triVerts: Uint32Array;
  mergeFromVert: Uint32Array;
  mergeToVert: Uint32Array;
  runIndex: Uint32Array;
  runOriginalID: Uint32Array;
  runTransform: Float64Array;
  runFlags: Uint8Array;
  faceID: Uint32Array;
  halfedgeTangent: Float64Array;

  get numTri(): number;
  get numVert(): number;
  get numRun(): number;

  /**
   * Like Mesh.merge(), in double precision. The baseline is the bounding box
   * scale times 1e-12, the precision a Manifold made from a Mesh64 gets.
   *
   * @param tolerance Default 0, with a bounding-box and precision-based floor.
   */
  merge(tolerance?: number): Mesh64;

  verts(tri: number): SealedUint32Array<3>;

  /** Gets the x, y, z position of this vertex in double precision. */
  position(vert: number): SealedFloat64Array<3>;

  /** Gets any other properties of this vertex in double precision. */
  extras(vert: number): Float64Array;

  /** Like Mesh.tangent(), in double precision. */
  tangent(halfedge: number): SealedFloat64Array<4>;

  transform(run: number): Mat4;
  backside(run: number): boolean;
  hasNormals(run: number): boolean;
}

/**
 * The error thrown when a Manifold operation fails with an error status, for
 * example by the Manifold constructor on an invalid mesh, or by warp() when the
 * result has a non-finite vertex. Its message is a short description of the
 * status.
 *
 * @group Basics
 */
export class ManifoldError extends Error {
  /**
   * @param code The error status.
   */
  constructor(code: ErrorStatus);

  /**
   * Always 'ManifoldError'.
   */
  name: 'ManifoldError';

  /**
   * The error status that caused this error.
   */
  code: ErrorStatus;
}
