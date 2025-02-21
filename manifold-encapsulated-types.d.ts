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

import {Box, FillRule, JoinType, Mat3, Mat4, Polygons, Rect, SealedFloat32Array, SealedUint32Array, SimplePolygon, Smoothness, Vec2, Vec3} from './manifold-global-types';

/**
 * Triangulates a set of /epsilon-valid polygons.
 *
 * @param polygons The set of polygons, wound CCW and representing multiple
 * polygons and/or holes.
 * @param epsilon The value of epsilon, bounding the uncertainty of the input
 * @return The triangles, referencing the original polygon points in order.
 */
export function triangulate(polygons: Polygons, epsilon?: number): Vec3[];

/**
 * Sets an angle constraint the default number of circular segments for the
 * {@link CrossSection.circle}, {@link Manifold.cylinder}, {@link
 * Manifold.sphere}, and
 * {@link Manifold.revolve} constructors. The number of segments will be rounded
 * up to the nearest factor of four.
 *
 * @param angle The minimum angle in degrees between consecutive segments. The
 * angle will increase if the the segments hit the minimum edge length.
 * Default is 10 degrees.
 */
export function setMinCircularAngle(angle: number): void;

/**
 * Sets a length constraint the default number of circular segments for the
 * {@link CrossSection.circle}, {@link Manifold.cylinder}, {@link
 * Manifold.sphere}, and
 * {@link Manifold.revolve} constructors. The number of segments will be rounded
 * up to the nearest factor of four.
 *
 * @param length The minimum length of segments. The length will
 * increase if the the segments hit the minimum angle. Default is 1.0.
 */
export function setMinCircularEdgeLength(length: number): void;

/**
 * Sets the default number of circular segments for the
 * {@link CrossSection.circle}, {@link Manifold.cylinder}, {@link
 * Manifold.sphere}, and
 * {@link Manifold.revolve} constructors. Overrides the edge length and angle
 * constraints and sets the number of segments to exactly this value.
 *
 * @param segments Number of circular segments. Default is 0, meaning no
 * constraint is applied.
 */
export function setCircularSegments(segments: number): void;

/**
 * Determine the result of the {@link setMinCircularAngle},
 * {@link setMinCircularEdgeLength}, and {@link setCircularSegments} defaults.
 *
 * @param radius For a given radius of circle, determine how many default
 * segments there will be.
 */
export function getCircularSegments(radius: number): number;

/**
 * Resets the circular construction parameters to their defaults if
 * {@link setMinCircularAngle}, {@link setMinCircularEdgeLength}, or {@link
 * setCircularSegments} have been called.
 */
export function resetToCircularDefaults(): void;
///@}

export class CrossSection {
  /**
   * Create a 2d cross-section from a set of contours (complex polygons). A
   * boolean union operation (with Positive filling rule by default) is
   * performed to combine overlapping polygons and ensure the resulting
   * CrossSection is free of intersections.
   *
   * @param contours A set of closed paths describing zero or more complex
   * polygons.
   * @param fillRule The filling rule used to interpret polygon sub-regions in
   * contours.
   */
  constructor(contours: Polygons, fillRule?: FillRule);

  // Shapes

  /**
   * Constructs a square with the given XY dimensions. By default it is
   * positioned in the first quadrant, touching the origin. If any dimensions in
   * size are negative, or if all are zero, an empty Manifold will be returned.
   *
   * @param size The X, and Y dimensions of the square.
   * @param center Set to true to shift the center to the origin.
   */
  static square(size?: Vec2|number, center?: boolean): CrossSection;

  /**
   * Constructs a circle of a given radius.
   *
   * @param radius Radius of the circle. Must be positive.
   * @param circularSegments Number of segments along its diameter. Default is
   * calculated by the static Quality defaults according to the radius.
   */
  static circle(radius: number, circularSegments?: number): CrossSection;

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
   * @param scaleTop Amount to scale the top (independently in X and Y). If the
   * scale is {0, 0}, a pure cone is formed with only a single vertex at the
   * top. Default {1, 1}.
   * @param center If true, the extrusion is centered on the z-axis through the
   *     origin
   * as opposed to resting on the XY plane as is default.
   */
  extrude(
      height: number, nDivisions?: number, twistDegrees?: number,
      scaleTop?: Vec2|number, center?: boolean): Manifold;

  /**
   * Constructs a manifold by revolving this cross-section around its Y-axis and
   * then setting this as the Z-axis of the resulting manifold. If the contours
   * cross the Y-axis, only the part on the positive X side is used.
   * Geometrically valid input will result in geometrically valid output.
   *
   * @param circularSegments Number of segments along its diameter. Default is
   * calculated by the static Defaults.
   */
  revolve(circularSegments?: number, revolveDegrees?: number): Manifold;

  // Transformations

  /**
   * Transform this CrossSection in space. Stored in column-major order. This
   * operation can be chained. Transforms are combined and applied lazily.
   *
   * @param m The affine transformation matrix to apply to all the vertices. The
   *     last row is ignored.
   */
  transform(m: Mat3): CrossSection;

  /**
   * Move this CrossSection in space. This operation can be chained. Transforms
   * are combined and applied lazily.
   *
   * @param v The vector to add to every vertex.
   */
  translate(v: Vec2): CrossSection;
  translate(x: number, y?: number): CrossSection;

  /**
   * Applies a (Z-axis) rotation to the CrossSection, in degrees. This operation
   * can be chained. Transforms are combined and applied lazily.
   *
   * @param degrees degrees about the Z-axis to rotate.
   */
  rotate(degrees: number): CrossSection;

  /**
   * Scale this CrossSection in space. This operation can be chained. Transforms
   * are combined and applied lazily.
   *
   * @param v The vector to multiply every vertex by per component.
   */
  scale(v: Vec2|number): CrossSection;


  /**
   * Mirror this CrossSection over the arbitrary axis described by the unit form
   * of the given vector. If the length of the vector is zero, an empty
   * CrossSection is returned. This operation can be chained. Transforms are
   * combined and applied lazily.
   *
   * @param ax the axis to be mirrored over
   */
  mirror(ax: Vec2): CrossSection;

  /**
   * Move the vertices of this CrossSection (creating a new one) according to
   * any arbitrary input function, followed by a union operation (with a
   * Positive fill rule) that ensures any introduced intersections are not
   * included in the result.
   *
   * @param warpFunc A function that modifies a given vertex position.
   */
  warp(warpFunc: (vert: Vec2) => void): CrossSection;

  /**
   * Inflate the contours in CrossSection by the specified delta, handling
   * corners according to the given JoinType.
   *
   * @param delta Positive deltas will cause the expansion of outlining contours
   * to expand, and retraction of inner (hole) contours. Negative deltas will
   * have the opposite effect.
   * @param joinType The join type specifying the treatment of contour joins
   * (corners).
   * @param miterLimit The maximum distance in multiples of delta that vertices
   * can be offset from their original positions with before squaring is
   * applied, **when the join type is Miter** (default is 2, which is the
   * minimum allowed). See the [Clipper2
   * MiterLimit](http://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/Properties/MiterLimit.htm)
   * page for a visual example.
   * @param circularSegments Number of segments per 360 degrees of
   * <B>JoinType::Round</B> corners (roughly, the number of vertices that
   * will be added to each contour). Default is calculated by the static Quality
   * defaults according to the radius.
   */
  offset(
      delta: number, joinType?: JoinType, miterLimit?: number,
      circularSegments?: number): CrossSection;

  /**
   * Remove vertices from the contours in this CrossSection that are less than
   * the specified distance epsilon from an imaginary line that passes through
   * its two adjacent vertices. Near duplicate vertices and collinear points
   * will be removed at lower epsilons, with elimination of line segments
   * becoming increasingly aggressive with larger epsilons.
   *
   * It is recommended to apply this function following Offset, in order to
   * clean up any spurious tiny line segments introduced that do not improve
   * quality in any meaningful way. This is particularly important if further
   * offseting operations are to be performed, which would compound the issue.
   *
   * @param epsilon minimum distance vertices must diverge from the hypothetical
   *     outline without them in order to be included in the output (default
   *     1e-6)
   */
  simplify(epsilon?: number): CrossSection;

  // Clipping Operations

  /**
   * Boolean union
   */
  add(other: CrossSection|Polygons): CrossSection;

  /**
   * Boolean difference
   */
  subtract(other: CrossSection|Polygons): CrossSection;

  /**
   * Boolean intersection
   */
  intersect(other: CrossSection|Polygons): CrossSection;

  /**
   * Boolean union of the cross-sections a and b
   */
  static union(a: CrossSection|Polygons, b: CrossSection|Polygons):
      CrossSection;

  /**
   * Boolean difference of the cross-section b from the cross-section a
   */
  static difference(a: CrossSection|Polygons, b: CrossSection|Polygons):
      CrossSection;

  /**
   * Boolean intersection of the cross-sections a and b
   */
  static intersection(a: CrossSection|Polygons, b: CrossSection|Polygons):
      CrossSection;

  /**
   * Boolean union of a list of cross-sections
   */
  static union(polygons: (CrossSection|Polygons)[]): CrossSection;

  /**
   * Boolean difference of the tail of a list of cross-sections from its head
   */
  static difference(polygons: (CrossSection|Polygons)[]): CrossSection;

  /**
   * Boolean intersection of a list of cross-sections
   */
  static intersection(polygons: (CrossSection|Polygons)[]): CrossSection;

  // Convex Hulls

  /**
   * Compute the convex hull of the contours in this CrossSection.
   */
  hull(): CrossSection;

  /**
   * Compute the convex hull of all points in a list of polygons/cross-sections.
   */
  static hull(polygons: (CrossSection|Polygons)[]): CrossSection;

  // Topological Operations

  /**
   * Construct a CrossSection from a vector of other Polygons (batch
   * boolean union).
   */
  static compose(polygons: (CrossSection|Polygons)[]): CrossSection;

  /**
   * This operation returns a vector of CrossSections that are topologically
   * disconnected, each containing one outline contour with zero or more
   * holes.
   */
  decompose(): CrossSection[];

  // Polygon Conversion

  /**
   * Create a 2d cross-section from a set of contours (complex polygons). A
   * boolean union operation (with Positive filling rule by default) is
   * performed to combine overlapping polygons and ensure the resulting
   * CrossSection is free of intersections.
   *
   * @param contours A set of closed paths describing zero or more complex
   * polygons.
   * @param fillRule The filling rule used to interpret polygon sub-regions in
   * contours.
   */
  static ofPolygons(contours: Polygons, fillRule?: FillRule): CrossSection;

  /**
   * Return the contours of this CrossSection as a list of simple polygons.
   */
  toPolygons(): SimplePolygon[];

  // Properties

  /**
   * Return the total area covered by complex polygons making up the
   * CrossSection.
   */
  area(): number;

  /**
   * Does the CrossSection (not) have any contours?
   */
  isEmpty(): boolean;

  /**
   * The number of vertices in the CrossSection.
   */
  numVert(): number;

  /**
   * The number of contours in the CrossSection.
   */
  numContour(): number;

  /**
   * Returns the axis-aligned bounding rectangle of all the CrossSection's
   * vertices.
   */
  bounds(): Rect;

  // Memory

  /**
   * Frees the WASM memory of this CrossSection, since these cannot be
   * garbage-collected automatically.
   */
  delete(): void;
}

/**
 * This library's internal representation of an oriented, 2-manifold, triangle
 * mesh - a simple boundary-representation of a solid object. Use this class to
 * store and operate on solids, and use MeshGL for input and output, or
 * potentially Mesh if only basic geometry is required.
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
 * OriginalIDs and the faceIDs and transforms accessible through MeshGL. This
 * allows object-level properties to be re-associated with the output after many
 * operations, particularly useful for materials. Since separate object's
 * properties are not mixed, there is no requirement that channels have
 * consistent meaning between different inputs.
 */
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

  // Shapes

  /**
   * Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
   * and the rest at similarly symmetric points.
   */
  static tetrahedron(): Manifold;

  /**
   * Constructs a unit cube (edge lengths all one), by default in the first
   * octant, touching the origin.
   *
   * @param size The X, Y, and Z dimensions of the box.
   * @param center Set to true to shift the center to the origin.
   */
  static cube(size?: Vec3|number, center?: boolean): Manifold;

  /**
   * A convenience constructor for the common case of extruding a circle. Can
   * also form cones if both radii are specified.
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
  static cylinder(
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
  static sphere(radius: number, circularSegments?: number): Manifold;

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
   * @param scaleTop Amount to scale the top (independently in X and Y). If the
   * scale is {0, 0}, a pure cone is formed with only a single vertex at the
   * top. Default {1, 1}.
   * @param center If true, the extrusion is centered on the z-axis through the
   *     origin
   * as opposed to resting on the XY plane as is default.
   */
  static extrude(
      polygons: CrossSection|Polygons, height: number, nDivisions?: number,
      twistDegrees?: number, scaleTop?: Vec2|number,
      center?: boolean): Manifold;

  /**
   * Constructs a manifold from a set of polygons/cross-section by revolving
   * them around the Y-axis and then setting this as the Z-axis of the resulting
   * manifold. If the polygons cross the Y-axis, only the part on the positive X
   * side is used. Geometrically valid input will result in geometrically valid
   * output.
   *
   * @param polygons A set of non-overlapping polygons to revolve.
   * @param circularSegments Number of segments along its diameter. Default is
   * calculated by the static Defaults.
   * @param revolveDegrees Number of degrees to revolve. Default is 360 degrees.
   */
  static revolve(
      polygons: CrossSection|Polygons, circularSegments?: number,
      revolveDegrees?: number): Manifold;

  // Mesh Conversion

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
  static ofMesh(mesh: Mesh): Manifold;

  /**
   * Constructs a smooth version of the input mesh by creating tangents; this
   * method will throw if you have supplied tangents with your mesh already. The
   * actual triangle resolution is unchanged; use the Refine() method to
   * interpolate to a higher-resolution curve.
   *
   * By default, every edge is calculated for maximum smoothness (very much
   * approximately), attempting to minimize the maximum mean Curvature
   * magnitude. No higher-order derivatives are considered, as the interpolation
   * is independent per triangle, only sharing constraints on their boundaries.
   *
   * @param mesh input Mesh.
   * @param sharpenedEdges If desired, you can supply a vector of sharpened
   * halfedges, which should in general be a small subset of all halfedges.
   * Order of entries doesn't matter, as each one specifies the desired
   * smoothness (between zero and one, with one the default for all unspecified
   * halfedges) and the halfedge index (3 * triangle index + [0,1,2] where 0 is
   * the edge between triVert 0 and 1, etc).
   *
   * At a smoothness value of zero, a sharp crease is made. The smoothness is
   * interpolated along each edge, so the specified value should be thought of
   * as an average. Where exactly two sharpened edges meet at a vertex, their
   * tangents are rotated to be colinear so that the sharpened edge can be
   * continuous. Vertices with only one sharpened edge are completely smooth,
   * allowing sharpened edges to smoothly vanish at termination. A single vertex
   * can be sharpened by sharping all edges that are incident on it, allowing
   * cones to be formed.
   */
  static smooth(mesh: Mesh, sharpenedEdges?: Smoothness[]): Manifold;

  // Signed Distance Functions

  /**
   * Constructs a level-set Mesh from the input Signed-Distance Function (SDF).
   * This uses a form of Marching Tetrahedra (akin to Marching Cubes, but better
   * for manifoldness). Instead of using a cubic grid, it uses a body-centered
   * cubic grid (two shifted cubic grids). This means if your function's
   * interior exceeds the given bounds, you will see a kind of egg-crate shape
   * closing off the manifold, which is due to the underlying grid.
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
   */
  transform(m: Mat4): Manifold;

  /**
   * Move this Manifold in space. This operation can be chained. Transforms are
   * combined and applied lazily.
   *
   * @param v The vector to add to every vertex.
   */
  translate(v: Vec3): Manifold;
  translate(x: number, y?: number, z?: number): Manifold;

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
  rotate(x: number, y?: number, z?: number): Manifold;

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
  mirror(normal: Vec3): Manifold;

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
   * Smooths out the Manifold by filling in the halfedgeTangent vectors. The
   * geometry will remain unchanged until Refine or RefineToLength is called to
   * interpolate the surface. This version uses the supplied vertex normal
   * properties to define the tangent vectors.
   *
   * @param normalIdx The first property channel of the normals. NumProp must be
   * at least normalIdx + 3. Any vertex where multiple normals exist and don't
   * agree will result in a sharp edge.
   */
  smoothByNormals(normalIdx: number): Manifold;

  /**
   * Smooths out the Manifold by filling in the halfedgeTangent vectors. The
   * geometry will remain unchanged until Refine or RefineToLength is called to
   * interpolate the surface. This version uses the geometry of the triangles
   * and pseudo-normals to define the tangent vectors.
   *
   * @param minSharpAngle degrees, default 60. Any edges with angles greater
   * than this value will remain sharp. The rest will be smoothed to G1
   * continuity, with the caveat that flat faces of three or more triangles will
   * always remain flat. With a value of zero, the model is faceted, but in this
   * case there is no point in smoothing.
   *
   * @param minSmoothness range: 0 - 1, default 0. The smoothness applied to
   * sharp angles. The default gives a hard edge, while values > 0 will give a
   * small fillet on these sharp edges. A value of 1 is equivalent to a
   * minSharpAngle of 180 - all edges will be smooth.
   */
  smoothOut(minSharpAngle?: number, minSmoothness?: number): Manifold;

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
   * Increase the density of the mesh by splitting each edge into pieces of
   * roughly the input length. Interior verts are added to keep the rest of the
   * triangulation edges also of roughly the same length. If halfedgeTangents
   * are present (e.g. from the Smooth() constructor), the new vertices will be
   * moved to the interpolated surface according to their barycentric
   * coordinates.
   *
   * @param length The length that edges will be broken down to.
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
   * on the surface, within rounding error.
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
   */
  calculateCurvature(gaussianIdx: number, meanIdx: number): Manifold;

  /**
   * Fills in vertex properties for normal vectors, calculated from the mesh
   * geometry. Flat faces composed of three or more triangles will remain flat.
   *
   * @param normalIdx The property channel in which to store the X
   * values of the normals. The X, Y, and Z channels will be sequential. The
   * property set will be automatically expanded to include up through normalIdx
   * + 2.
   *
   * @param minSharpAngle Any edges with angles greater than this value will
   * remain sharp, getting different normal vector properties on each side of
   * the edge. By default, no edges are sharp and all normals are shared. With a
   * value of zero, the model is faceted and all normals match their triangle
   * normals, but in this case it would be better not to calculate normals at
   * all.
   */
  calculateNormals(normalIdx: number, minSharpAngle: number): Manifold;

  // Boolean Operations

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
   * Boolean union of the manifolds a and b
   */
  static union(a: Manifold, b: Manifold): Manifold;

  /**
   * Boolean difference of the manifold b from the manifold a
   */
  static difference(a: Manifold, b: Manifold): Manifold;

  /**
   * Boolean intersection of the manifolds a and b
   */
  static intersection(a: Manifold, b: Manifold): Manifold;

  /**
   * Boolean union of a list of manifolds
   */
  static union(manifolds: Manifold[]): Manifold;

  /**
   * Boolean difference of the tail of a list of manifolds from its head
   */
  static difference(manifolds: Manifold[]): Manifold;

  /**
   * Boolean intersection of a list of manifolds
   */
  static intersection(manifolds: Manifold[]): Manifold;

  /**
   * Split cuts this manifold in two using the cutter manifold. The first result
   * is the intersection, second is the difference. This is more efficient than
   * doing them separately.
   *
   * @param cutter
   */
  split(cutter: Manifold): Manifold[];

  /**
   * Convenient version of Split() for a half-space.
   *
   * @param normal This vector is normal to the cutting plane and its length
   *     does
   * not matter. The first result is in the direction of this vector, the second
   * result is on the opposite side.
   * @param originOffset The distance of the plane from the origin in the
   * direction of the normal vector.
   */
  splitByPlane(normal: Vec3, originOffset: number): Manifold[];

  /**
   * Removes everything behind the given half-space plane.
   *
   * @param normal This vector is normal to the cutting plane and its length
   *     does not matter. The result is in the direction of this vector from the
   *     plane.
   * @param originOffset The distance of the plane from the origin in the
   *     direction of the normal vector.
   */
  trimByPlane(normal: Vec3, originOffset: number): Manifold;

  /**
   * Returns the cross section of this object parallel to the X-Y plane at the
   * specified height. Using a height equal to the bottom
   * of the bounding box will return the bottom faces, while using a height
   * equal to the top of the bounding box will return empty.
   *
   * @param height Z-level of slice.
   */
  slice(height: number): CrossSection;

  /**
   * Returns a cross section representing the projected outline of this object
   * onto the X-Y plane.
   */
  project(): CrossSection;

  // Convex Hulls

  /**
   * Compute the convex hull of all points in this Manifold.
   */
  hull(): Manifold;

  /**
   * Compute the convex hull of all points contained within a set of Manifolds
   * and point vectors.
   */
  static hull(points: (Manifold|Vec3)[]): Manifold;

  // Topological Operations

  /**
   * Constructs a new manifold from a list of other manifolds. This is a purely
   * topological operation, so care should be taken to avoid creating
   * overlapping results. It is the inverse operation of Decompose().
   *
   * @param manifolds A list of Manifolds to lazy-union together.
   */
  static compose(manifolds: Manifold[]): Manifold;

  /**
   * This operation returns a vector of Manifolds that are topologically
   * disconnected. If everything is connected, the vector is length one,
   * containing a copy of the original. It is the inverse operation of
   * Compose().
   */
  decompose(): Manifold[];

  // Property Access

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
   * The number of properties per vertex in the Manifold.
   */
  numProp(): number;

  /**
   * The number of property vertices in the Manifold. This will always be >=
   * numVert, as some physical vertices may be duplicated to account for
   * different properties on different neighboring triangles.
   */
  numPropVert(): number

  /**
   * Returns the axis-aligned bounding box of all the Manifold's vertices.
   */
  boundingBox(): Box;

  /**
   * Returns the tolerance of this Manifold's vertices, which tracks the
   * approximate rounding error over all the transforms and operations that have
   * led to this state. Any triangles that are colinear within this tolerance
   * are considered degenerate and removed. This is the value of &epsilon;
   * defining
   * [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
   */
  tolerance(): number;

  /**
   * Return a copy of the manifold with the set tolerance value.
   * This performs mesh simplification when the tolerance value is increased.
   */
  setTolerance(tolerance: number): Manifold;

  /**
   * The genus is a topological property of the manifold, representing the
   * number of "handles". A sphere is 0, torus 1, etc. It is only meaningful for
   * a single mesh, so it is best to call Decompose() first.
   */
  genus(): number;

  /**
   * Returns the surface area of the manifold.
   */
  surfaceArea(): number;

  /**
   * Returns the volume of the manifold.
   */
  volume(): number;

  /**
   * Returns the minimum gap between two manifolds. Returns a float between
   * 0 and searchLength.
   */
  minGap(other: Manifold, searchLength: number): number;

  // Export

  /**
   * Returns a Mesh that is designed to easily push into a renderer, including
   * all interleaved vertex properties that may have been input. It also
   * includes relations to all the input meshes that form a part of this result
   * and the transforms applied to each.
   *
   * @param normalIdx If the original MeshGL inputs that formed this manifold
   * had properties corresponding to normal vectors, you can specify the first
   * of the three consecutive property channels forming the (x, y, z) normals,
   * which will cause this output MeshGL to automatically update these normals
   * according to the applied transforms and front/back side. normalIdx + 3 must
   * be <= numProp, and all original MeshGLs must use the same channels for
   * their normals.
   */
  getMesh(normalIdx?: number): Mesh;

  // ID Management

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

  /**
   * Returns the first of n sequential new unique mesh IDs for marking sets of
   * triangles that can be looked up after further operations. Assign to
   * Mesh.runOriginalID vector.
   */
  static reserveIDs(count: number): number;

  // Memory

  /**
   * Frees the WASM memory of this Manifold, since these cannot be
   * garbage-collected automatically.
   */
  delete(): void;
}

export interface MeshOptions {
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
}

/**
 * An alternative to Mesh for output suitable for pushing into graphics
 * libraries directly. This may not be manifold since the verts are duplicated
 * along property boundaries that do not match. The additional merge vectors
 * store this missing information, allowing the manifold to be reconstructed.
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
   * mesh. If you create an input MeshGL that you want to be able to reference
   * as one or more originals, be sure to set unique values from ReserveIDs().
   */
  runOriginalID: Uint32Array;

  /**
   * Optional: For each run, a 3x4 transform is stored representing how the
   * corresponding original mesh was transformed to create this triangle run.
   * This matrix is stored in column-major order and the length of the overall
   * vector is 12 * runOriginalID.size().
   */
  runTransform: Float32Array;

  /**
   * Optional: Length NumTri, contains an ID of the source face this triangle
   * comes from. When auto-generated, this ID will be a triangle index into the
   * original mesh. All neighboring coplanar triangles from that input mesh
   * will refer to a single triangle of that group as the faceID. When
   * supplying faceIDs, ensure that triangles with the same ID are in fact
   * coplanar and have consistent properties (within some tolerance) or the
   * output will be surprising.
   */
  faceID: Uint32Array;

  /**
   * Optional: The X-Y-Z-W weighted tangent vectors for smooth Refine(). If
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
   * Updates the mergeFromVert and mergeToVert vectors in order to create a
   * manifold solid. If the MeshGL is already manifold, no change will occur and
   * the function will return false. Otherwise, this will merge verts along open
   * edges within tolerance (the maximum of the MeshGL tolerance and the
   * baseline bounding-box tolerance), keeping any from the existing merge
   * vectors.
   *
   * There is no guarantee the result will be manifold - this is a best-effort
   * helper function designed primarily to aid in the case where a manifold
   * multi-material MeshGL was produced, but its merge vectors were lost due to
   * a round-trip through a file format. Constructing a Manifold from the result
   * will report a Status if it is not manifold.
   */
  merge(): boolean;

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
   * related triangles.
   *
   * @param run triangle run index.
   */
  transform(run: number): Mat4;
}
