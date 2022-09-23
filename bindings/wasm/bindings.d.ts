type Vec3 = [number, number, number];
type Matrix4x3 = [Vec3, Vec3, Vec3, Vec3];
type Vec2 = [number, number];
type SimplePolygon = Vec2[];
type Polygons = SimplePolygon[];

declare class Manifold {
  /**
   * Transform this Manifold in space. The first three columns form a 3x3 matrix
   * transform and the last is a translation vector. This operation can be
   * chained. Transforms are combined and applied lazily.
   *
   * @param m The affine transform matrix to apply to all the vertices.
   */
  transform(m: Matrix4x3): Manifold;

  /** 
   * Move this Manifold in space. This operation can be chained. Transforms are
   * combined and applied lazily.
   *
   * @param v The vector to add to every vertex.
   */
  translate(v: Vec3): Manifold;

  /**
   * Applys an Euler angle rotation to the manifold, first about the X axis, then
   * Y, then Z, in degrees. We use degrees so that we can minimize rounding error,
   * and elimiate it completely for any multiples of 90 degrees. Addtionally, more
   * efficient code paths are used to update the manifold when the transforms only
   * rotate by multiples of 90 degrees. This operation can be chained. Transforms
   * are combined and applied lazily.
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
  scale(v: Vec3): Manifold;

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
}

/**
 * Constructs a unit cube (edge lengths all one), by default in the first
 * octant, touching the origin.
 *
 * @param size The X, Y, and Z dimensions of the box.
 * @param center Set to true to shift the center to the origin.
 */
declare function cube(size?: Vec3, center?: boolean): Manifold;

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
declare function cylinder(
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
declare function sphere(radius: number, circularSegments?: number): Manifold;

/**
 * Constructs a manifold from a set of polygons by extruding them along the
 * Z-axis.
 *
 * @param crossSection A set of non-overlapping polygons to extrude.
 * @param height Z-extent of extrusion.
 * @param nDivisions Number of extra copies of the crossSection to insert into
 * the shape vertically; especially useful in combnation with twistDegrees to
 * avoid interpolation artifacts. Default is none.
 * @param twistDegrees Amount to twist the top crossSection relative to the
 * bottom, interpolated linearly for the divisions in between.
 * @param scaleTop Amount to scale the top (independently in X and Y). If the
 * scale is {0, 0}, a pure cone is formed with only a single vertex at the top.
 * Default {1, 1}.
 */
declare function extrude(
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
declare function revolve(crossSection: Polygons, circularSegments?: number): Manifold;

declare function union(a: Manifold, b: Manifold): Manifold;
declare function difference(a: Manifold, b: Manifold): Manifold;
declare function intersection(a: Manifold, b: Manifold): Manifold;

