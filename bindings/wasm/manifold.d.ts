// Copyright 2022 The Manifold Authors.
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

type Vec2 = [number, number];
type Vec3 = [number, number, number];
type Vec4 = [number, number, number, number];
type Matrix3x4 = [Vec3, Vec3, Vec3, Vec3];
type SimplePolygon = Vec2[];
type Polygons = SimplePolygon|SimplePolygon[];
type Box = {
  min: Vec3,
  max: Vec3
};
type Smoothness = {
  halfedge: number,
  smoothness: number
};
type Properties = {
  surfaceArea: number,
  volume: number
};
type BaryRef = {
  meshID: number,
  originalID: number,
  tri: number,
  vertBary: Vec3
};
type Curvature = {
  maxMeanCurvature: number,
  minMeanCurvature: number,
  maxGaussianCurvature: number,
  minGaussianCurvature: number,
  vertMeanCurvature: number[],
  vertGaussianCurvature: number[]
};
type MeshRelation = {
  barycentric: Vec3[],
  triBary: BaryRef[],
};

declare class Mesh {
  vertPos: Float32Array;
  triVerts: Uint32Array;
  vertNormal?: Float32Array;
  halfedgeTangent?: Float32Array;
  get numTri(): number;
  get numVert(): number;
  verts(tri: number): Uint32Array<3>;
  position(vert: number): Float32Array<3>;
  normal(vert: number): Float32Array<3>;
  tangent(halfedge: number): Float32Array<4>;
}

declare class Manifold {
  /**
   * Create a Manifold from a Mesh object.
   */
  constructor(mesh: Mesh);
  /**
   * Transform this Manifold in space. The first three columns form a 3x3 matrix
   * transform and the last is a translation vector. This operation can be
   * chained. Transforms are combined and applied lazily.
   *
   * @param m The affine transform matrix to apply to all the vertices.
   */
  transform(m: Matrix3x4): Manifold;

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
   * This returns a Mesh of simple vectors of vertices and triangles suitable
   * for saving or other operations outside of the context of this library.
   */
  getMesh(): Mesh;

  /**
   * Gets the relationship to the previous meshes, for the purpose of assigning
   * properties like texture coordinates. The triBary vector is the same length
   * as Mesh.triVerts: BaryRef.originalID indicates the source mesh and
   * BaryRef.tri is that mesh's triangle index to which these barycentric
   * coordinates refer. BaryRef.vertBary gives an index for each vertex into the
   * barycentric vector if that index is >= 0, indicating it is a new vertex. If
   * the index is < 0, this indicates it is an original vertex, the index + 3
   * vert of the referenced triangle.
   *
   * BaryRef.meshID is a unique ID to the particular instance of a given mesh.
   * For instance, if you want to convert the triangle mesh to a polygon mesh,
   * all the triangles from a given face will have the same .meshID and .tri
   * values.
   */
  getMeshRelation(): MeshRelation;

  /**
   * If you copy a manifold, but you want this new copy to have new properties
   * (e.g. a different UV mapping), you can reset its meshIDs to a new original,
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
   * If this mesh is an original, this returns its meshID that can be referenced
   * by product manifolds' MeshRelation. If this manifold is a product, this
   * returns -1.
   */
  originalID(): number;
}

/**
 * Constructs a unit cube (edge lengths all one), by default in the first
 * octant, touching the origin.
 *
 * @param size The X, Y, and Z dimensions of the box.
 * @param center Set to true to shift the center to the origin.
 */
declare function cube(size?: Vec3|number, center?: boolean): Manifold;

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
declare function smooth(mesh: Mesh, sharpenedEdges?: Smoothness[]): Manifold;

/**
 * Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
 * and the rest at similarly symmetric points.
 */
declare function tetrahedron(): Manifold;

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
declare function revolve(
    crossSection: Polygons, circularSegments?: number): Manifold;

declare function union(a: Manifold, b: Manifold): Manifold;
declare function difference(a: Manifold, b: Manifold): Manifold;
declare function intersection(a: Manifold, b: Manifold): Manifold;

declare function union(manifolds: Manifold[]): Manifold;
declare function difference(manifolds: Manifold[]): Manifold;
declare function intersection(manifolds: Manifold[]): Manifold;

/**
 * Constructs a new manifold from a list of other manifolds. This is a purely
 * topological operation, so care should be taken to avoid creating
 * overlapping results. It is the inverse operation of Decompose().
 *
 * @param manifolds A list of Manifolds to lazy-union together.
 */
declare function compose(manifolds: Manifold[]): Manifold;

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
declare function levelSet(
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
declare function setMinCircularAngle(angle: number): void;
declare function setMinCircularEdgeLength(length: number): void;
declare function setCircularSegments(segments: number): void;
declare function getCircularSegments(radius: number): number;
///@}

/**
 * Create a Manifold from a serialized Mesh object (MeshVec). Unlike the
 * constructor, this method does not dispose the Mesh after using it.
 *
 * @param meshVec The serialized Mesh object to convert into a Manifold.
 */
declare function ManifoldFromMeshVec(meshVec: MeshVec): Manifold;

declare interface ManifoldStatic {
  cube: typeof cube;
  cylinder: typeof cylinder;
  sphere: typeof sphere;
  smooth: typeof smooth;
  tetrahedron: typeof tetrahedron;
  extrude: typeof extrude;
  revolve: typeof revolve;
  union: typeof union;
  difference: typeof difference;
  intersection: typeof intersection;
  compose: typeof compose;
  levelSet: typeof levelSet;
  setMinCircularAngle: typeof setMinCircularAngle;
  setMinCircularEdgeLength: typeof setMinCircularEdgeLength;
  setCircularSegments: typeof setCircularSegments;
  getCircularSegments: typeof getCircularSegments;
  ManifoldFromMeshVec: typeof ManifoldFromMeshVec;
  Manifold: typeof Manifold;
  setup: () => void;
}

declare function Module(): Promise<ManifoldStatic>;