#pragma once
namespace manifold_docs {
const auto manifold_cube = R"___()___";
const auto manifold_manifold = R"___(Construct an empty Manifold.)___";
const auto manifold_manifold_1 = R"___()___";
const auto manifold_operator_eq = R"___()___";
const auto manifold_manifold_2 = R"___()___";
const auto manifold_manifold_3 = R"___()___";
const auto manifold_manifold_4 = R"___()___";
const auto manifold_invalid = R"___()___";
const auto manifold_operator_eq_1 = R"___()___";
const auto manifold_get_csg_leaf_node = R"___()___";
const auto manifold_manifold_5 = R"___(Convert a MeshGL into a Manifold, retaining its properties and merging only
the positions according to the merge vectors. Will return an empty Manifold
and set an Error Status if the result is not an oriented 2-manifold. Will
collapse degenerate triangles and unnecessary vertices.
All fields are read, making this structure suitable for a lossless round-trip
of data from GetMeshGL. For multi-material input, use ReserveIDs to set a
unique originalID for each material, and sort the materials into triangle
runs.
:param mesh_gl: The input MeshGL.
:param property_tolerance: A vector of precision values for each property
beyond position. If specified, the propertyTolerance vector must have size =
numProp - 3. This is the amount of interpolation error allowed before two
neighboring triangles are considered to be on a property boundary edge.
Property boundary edges will be retained across operations even if the
triangles are coplanar. Defaults to 1e-5, which works well for most
properties in the [-1, 1] range.)___";
const auto manifold_manifold_6 = R"___(Convert a Mesh into a Manifold. Will return an empty Manifold
and set an Error Status if the Mesh is not an oriented 2-manifold. Will
collapse degenerate triangles and unnecessary vertices.
:param mesh: The input Mesh.)___";
const auto manifold_get_mesh = R"___(This returns a Mesh of simple vectors of vertices and triangles suitable for
saving or other operations outside of the context of this library.)___";
const auto manifold_get_mesh_gl = R"___(The most complete output of this library, returning a MeshGL that is designed
to easily push into a renderer, including all interleaved vertex properties
that may have been input. It also includes relations to all the input meshes
that form a part of this result and the transforms applied to each.
:param normal_idx: If the original MeshGL inputs that formed this manifold had
properties corresponding to normal vectors, you can specify which property
channels these are (x, y, z), which will cause this output MeshGL to
automatically update these normals according to the applied transforms and
front/back side. Each channel must be >= 3 and < numProp, and all original
MeshGLs must use the same channels for their normals.)___";
const auto manifold_is_empty = R"___(Does the Manifold have any triangles?)___";
const auto manifold_status = R"___(Returns the reason for an input Mesh producing an empty Manifold. This Status
only applies to Manifolds newly-created from an input Mesh - once they are
combined into a new Manifold via operations, the status reverts to NoError,
simply processing the problem mesh as empty. Likewise, empty meshes may still
show NoError, for instance if they are small enough relative to their
precision to be collapsed to nothing.)___";
const auto manifold_num_vert = R"___(The number of vertices in the Manifold.)___";
const auto manifold_num_edge = R"___(The number of edges in the Manifold.)___";
const auto manifold_num_tri = R"___(The number of triangles in the Manifold.)___";
const auto manifold_num_prop = R"___(The number of properties per vertex in the Manifold.)___";
const auto manifold_num_prop_vert = R"___(The number of property vertices in the Manifold. This will always be >=
NumVert, as some physical vertices may be duplicated to account for different
properties on different neighboring triangles.)___";
const auto manifold_bounding_box = R"___(Returns the axis-aligned bounding box of all the Manifold's vertices.)___";
const auto manifold_precision = R"___(Returns the precision of this Manifold's vertices, which tracks the
approximate rounding error over all the transforms and operations that have
led to this state. Any triangles that are colinear within this precision are
considered degenerate and removed. This is the value of &epsilon; defining
[&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).)___";
const auto manifold_genus = R"___(The genus is a topological property of the manifold, representing the number
of "handles". A sphere is 0, torus 1, etc. It is only meaningful for a single
mesh, so it is best to call Decompose() first.)___";
const auto manifold_get_properties = R"___(Returns the surface area and volume of the manifold.)___";
const auto manifold_original_id = R"___(If this mesh is an original, this returns its meshID that can be referenced
by product manifolds' MeshRelation. If this manifold is a product, this
returns -1.)___";
const auto manifold_as_original = R"___(This function condenses all coplanar faces in the relation, and
collapses those edges. In the process the relation to ancestor meshes is lost
and this new Manifold is marked an original. Properties are preserved, so if
they do not match across an edge, that edge will be kept.)___";
const auto manifold_reserve_ids = R"___(Returns the first of n sequential new unique mesh IDs for marking sets of
triangles that can be looked up after further operations. Assign to
MeshGL.runOriginalID vector.)___";
const auto manifold_impl__reserve_ids = R"___()___";
const auto manifold_matches_tri_normals = R"___(The triangle normal vectors are saved over the course of operations rather
than recalculated to avoid rounding error. This checks that triangles still
match their normal vectors within Precision().)___";
const auto manifold_num_degenerate_tris = R"___(The number of triangles that are colinear within Precision(). This library
attempts to remove all of these, but it cannot always remove all of them
without changing the mesh by too much.)___";
const auto manifold_num_overlaps = R"___(This is a checksum-style verification of the collider, simply returning the
total number of edge-face bounding box overlaps between this and other.
:param other: A Manifold to overlap with.)___";
const auto manifold_translate = R"___(Move this Manifold in space. This operation can be chained. Transforms are
combined and applied lazily.
:param v: The vector to add to every vertex.)___";
const auto manifold_scale = R"___(Scale this Manifold in space. This operation can be chained. Transforms are
combined and applied lazily.
:param v: The vector to multiply every vertex by per component.)___";
const auto manifold_rotate = R"___(Applies an Euler angle rotation to the manifold, first about the X axis, then
Y, then Z, in degrees. We use degrees so that we can minimize rounding error,
and eliminate it completely for any multiples of 90 degrees. Additionally,
more efficient code paths are used to update the manifold when the transforms
only rotate by multiples of 90 degrees. This operation can be chained.
Transforms are combined and applied lazily.
:param x_degrees: First rotation, degrees about the X-axis.
:param y_degrees: Second rotation, degrees about the Y-axis.
:param z_degrees: Third rotation, degrees about the Z-axis.)___";
const auto manifold_transform = R"___(Transform this Manifold in space. The first three columns form a 3x3 matrix
transform and the last is a translation vector. This operation can be
chained. Transforms are combined and applied lazily.
:param m: The affine transform matrix to apply to all the vertices.)___";
const auto manifold_mirror = R"___(Mirror this Manifold over the plane described by the unit form of the given
normal vector. If the length of the normal is zero, an empty Manifold is
returned. This operation can be chained. Transforms are combined and applied
lazily.
:param normal: The normal vector of the plane to be mirrored over)___";
const auto manifold_warp = R"___(This function does not change the topology, but allows the vertices to be
moved according to any arbitrary input function. It is easy to create a
function that warps a geometrically valid object into one which overlaps, but
that is not checked here, so it is up to the user to choose their function
with discretion.
:param warp_func: A function that modifies a given vertex position.)___";
const auto manifold_warp_batch = R"___(Same as Manifold::Warp but calls warpFunc with with
a VecView which is roughly equivalent to std::span
pointing to all vec3 elements to be modified in-place
:param warp_func: A function that modifies multiple vertex positions.)___";
const auto manifold_set_properties = R"___(Create a new copy of this manifold with updated vertex properties by
supplying a function that takes the existing position and properties as
input. You may specify any number of output properties, allowing creation and
removal of channels. Note: undefined behavior will result if you read past
the number of input properties or write past the number of output properties.
:param num_prop: The new number of properties per vertex.
:param prop_func: A function that modifies the properties of a given vertex.)___";
const auto manifold_calculate_curvature = R"___(Curvature is the inverse of the radius of curvature, and signed such that
positive is convex and negative is concave. There are two orthogonal
principal curvatures at any point on a manifold, with one maximum and the
other minimum. Gaussian curvature is their product, while mean
curvature is their sum. This approximates them for every vertex and assigns
them as vertex properties on the given channels.
:param gaussian_idx: The property channel index in which to store the Gaussian
curvature. An index < 0 will be ignored (stores nothing). The property set
will be automatically expanded to include the channel index specified.
:param mean_idx: The property channel index in which to store the mean
curvature. An index < 0 will be ignored (stores nothing). The property set
will be automatically expanded to include the channel index specified.)___";
const auto manifold_refine = R"___(Increase the density of the mesh by splitting every edge into n pieces. For
instance, with n = 2, each triangle will be split into 4 triangles. These
will all be coplanar (and will not be immediately collapsed) unless the
Mesh/Manifold has halfedgeTangents specified (e.g. from the Smooth()
constructor), in which case the new vertices will be moved to the
interpolated surface according to their barycentric coordinates.
:param n: The number of pieces to split every edge into. Must be > 1.)___";
const auto manifold_boolean = R"___(The central operation of this library: the Boolean combines two manifolds
into another by calculating their intersections and removing the unused
portions.
[&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid)
inputs will produce &epsilon;-valid output. &epsilon;-invalid input may fail
triangulation.
These operations are optimized to produce nearly-instant results if either
input is empty or their bounding boxes do not overlap.
:param second: The other Manifold.
:param op: The type of operation to perform.)___";
const auto manifold_batch_boolean = R"___()___";
const auto manifold_operator_plus = R"___(Shorthand for Boolean Union.)___";
const auto manifold_operator_plus_eq = R"___(Shorthand for Boolean Union assignment.)___";
const auto manifold_operator_minus = R"___(Shorthand for Boolean Difference.)___";
const auto manifold_operator_minus_eq = R"___(Shorthand for Boolean Difference assignment.)___";
const auto manifold_operator_xor = R"___(Shorthand for Boolean Intersection.)___";
const auto manifold_operator_xor_eq = R"___(Shorthand for Boolean Intersection assignment.)___";
const auto manifold_split = R"___(Split cuts this manifold in two using the cutter manifold. The first result
is the intersection, second is the difference. This is more efficient than
doing them separately.
:param cutter:)___";
const auto manifold_split_by_plane = R"___(Convenient version of Split() for a half-space.
:param normal: This vector is normal to the cutting plane and its length does
not matter. The first result is in the direction of this vector, the second
result is on the opposite side.
:param origin_offset: The distance of the plane from the origin in the
direction of the normal vector.)___";
const auto manifold_trim_by_plane = R"___(Identical to SplitByPlane(), but calculating and returning only the first
result.
:param normal: This vector is normal to the cutting plane and its length does
not matter. The result is in the direction of this vector from the plane.
:param origin_offset: The distance of the plane from the origin in the
direction of the normal vector.)___";
const auto manifold_slice = R"___(Returns the cross section of this object parallel to the X-Y plane at the
specified Z height, defaulting to zero. Using a height equal to the bottom of
the bounding box will return the bottom faces, while using a height equal to
the top of the bounding box will return empty.)___";
const auto manifold_project = R"___(Returns a cross section representing the projected outline of this object
onto the X-Y plane.)___";
const auto manifold_hull = R"___(Compute the convex hull of a set of points. If the given points are fewer
than 4, or they are all coplanar, an empty Manifold will be returned.
:param pts: A vector of 3-dimensional points over which to compute a convex
hull.)___";
const auto manifold_hull_1 = R"___(Compute the convex hull of this manifold.)___";
const auto manifold_hull_2 = R"___(Compute the convex hull enveloping a set of manifolds.
:param manifolds: A vector of manifolds over which to compute a convex hull.)___";
const auto triangulate = R"___(@brief Triangulates a set of &epsilon;-valid polygons. If the input is not
&epsilon;-valid, the triangulation may overlap, but will always return a
manifold result that matches the input edge directions.
:param polygons: The set of polygons, wound CCW and representing multiple
polygons and/or holes.
:param precision: The value of &epsilon;, bounding the uncertainty of the
input.
@return std::vector<glm::ivec3> The triangles, referencing the original
polygon points in order.)___";
const auto triangulate_poly = R"___()___";
const auto triangulate_idx = R"___()___";
const auto set_min_circular_angle = R"___(Sets an angle constraint the default number of circular segments for the
CrossSection::circle, Manifold::cylinder, Manifold::sphere, and
Manifold::revolve constructors. The number of segments will be rounded up
to the nearest factor of four.
:param angle: The minimum angle in degrees between consecutive segments. The
angle will increase if the the segments hit the minimum edge length.
Default is 10 degrees.)___";
const auto set_min_circular_edge_length = R"___(Sets a length constraint the default number of circular segments for the
CrossSection::circle, Manifold::cylinder, Manifold::sphere, and
Manifold::revolve constructors. The number of segments will be rounded up
to the nearest factor of four.
:param length: The minimum length of segments. The length will
increase if the the segments hit the minimum angle. Default is 1.0.)___";
const auto set_circular_segments = R"___(Sets the default number of circular segments for the
CrossSection::circle, Manifold::cylinder, Manifold::sphere, and
Manifold::revolve constructors. Overrides the edge length and angle
constraints and sets the number of segments to exactly this value.
:param number: Number of circular segments. Default is 0, meaning no
constraint is applied.)___";
const auto get_circular_segments = R"___(Determine the result of the set_min_circular_angle,
set_min_circular_edge_length, and set_circular_segments defaults.
:param radius: For a given radius of circle, determine how many default
segments there will be.)___";
} // namespace manifold_docs