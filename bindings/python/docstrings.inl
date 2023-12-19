#pragma once
#include <string>
namespace manifold_docs {
const auto cross_section__area_f5458809be32 = R"___(Return the total area covered by complex polygons making up the
CrossSection.)___";
const auto cross_section__batch_boolean_d17f25dd9195 = R"___(Perform the given boolean operation on a list of CrossSections. In case of
Subtract, all CrossSections in the tail are differenced from the head.)___";
const auto cross_section__boolean_d0a145760f3d = R"___(Perform the given boolean operation between this and another CrossSection.)___";
const auto cross_section__bounds_47149b83cada = R"___(Returns the axis-aligned bounding rectangle of all the CrossSections'
vertices.)___";
const auto cross_section__circle_72e80d0e2b44 = R"___(Constructs a circle of a given radius.
:param radius: Radius of the circle. Must be positive.
:param circular_segments: Number of segments along its diameter. Default is
calculated by the static Quality defaults according to the radius.)___";
const auto cross_section__compose_1a076380de50 = R"___(Construct a CrossSection from a vector of other CrossSections (batch
boolean union).)___";
const auto cross_section__cross_section_138b32c68b57 = R"___(Create a 2d cross-section from an axis-aligned rectangle (bounding box).
:param rect: An axis-aligned rectangular bounding box.)___";
const auto cross_section__cross_section_381df1ec382b = R"___(The default constructor is an empty cross-section (containing no contours).)___";
const auto cross_section__cross_section_4459865c1e2f = R"___(Create a 2d cross-section from a set of contours (complex polygons). A
boolean union operation (with Positive filling rule by default) is
performed to combine overlapping polygons and ensure the resulting
CrossSection is free of intersections.
:param contours: A set of closed paths describing zero or more complex
polygons.
:param fillrule: The filling rule used to interpret polygon sub-regions in
contours.)___";
const auto cross_section__cross_section_aad37c257207 = R"___(Create a 2d cross-section from a single contour. A boolean union operation
(with Positive filling rule by default) is performed to ensure the
resulting CrossSection is free of self-intersections.
:param contour: A closed path outlining the desired cross-section.
:param fillrule: The filling rule used to interpret polygon sub-regions
created by self-intersections in contour.)___";
const auto cross_section__cross_section_c56006cfa78c = R"___(The copy constructor avoids copying the underlying paths vector (sharing
with its parent via shared_ptr), however subsequent transformations, and
their application will not be shared. It is generally recommended to avoid
this, opting instead to simply create CrossSections with the available
const methods.)___";
const auto cross_section__decompose_17ae5159e6e5 = R"___(This operation returns a vector of CrossSections that are topologically
disconnected, each containing one outline contour with zero or more
holes.)___";
const auto cross_section__hull_014f76304d06 = R"___(Compute the convex hull enveloping a set of cross-sections.
:param cross_sections: A vector of cross-sections over which to compute a
convex hull.)___";
const auto cross_section__hull_3f1ad9eaa499 = R"___(Compute the convex hull of this cross-section.)___";
const auto cross_section__hull_c94ccc3c0fe6 = R"___(Compute the convex hull of a set of points/polygons. If the given points are
fewer than 3, an empty CrossSection will be returned.
:param pts: A vector of vectors of 2-dimensional points over which to compute
a convex hull.)___";
const auto cross_section__hull_e609862eb256 = R"___(Compute the convex hull of a set of points. If the given points are fewer
than 3, an empty CrossSection will be returned.
:param pts: A vector of 2-dimensional points over which to compute a convex
hull.)___";
const auto cross_section__is_empty_25b4b2d4e0ad = R"___(Does the CrossSection contain any contours?)___";
const auto cross_section__mirror_a10119e40f21 = R"___(Mirror this CrossSection over the arbitrary axis described by the unit form
of the given vector. If the length of the vector is zero, an empty
CrossSection is returned. This operation can be chained. Transforms are
combined and applied lazily.
:param ax: the axis to be mirrored over)___";
const auto cross_section__num_contour_5894fa74e5f5 = R"___(Return the number of contours (both outer and inner paths) in the
CrossSection.)___";
const auto cross_section__num_vert_9dd2efd31062 = R"___(Return the number of vertices in the CrossSection.)___";
const auto cross_section__offset_b3675b4b0ed0 = R"___(Inflate the contours in CrossSection by the specified delta, handling
corners according to the given JoinType.
:param delta: Positive deltas will cause the expansion of outlining contours
to expand, and retraction of inner (hole) contours. Negative deltas will
have the opposite effect.
:param jt: The join type specifying the treatment of contour joins
(corners).
:param miter_limit: The maximum distance in multiples of delta that vertices
can be offset from their original positions with before squaring is
applied, <B>when the join type is Miter</B> (default is 2, which is the
minimum allowed). See the [Clipper2
MiterLimit](http://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/Properties/MiterLimit.htm)
page for a visual example.
:param circular_segments: Number of segments per 360 degrees of
<B>JoinType::Round</B> corners (roughly, the number of vertices that
will be added to each contour). Default is calculated by the static Quality
defaults according to the radius.)___";
const auto cross_section__operator_minus_04b4d727817f = R"___(Compute the boolean difference of a (clip) cross-section from another
(subject).)___";
const auto cross_section__operator_minus_eq_bdfdaab4f47a = R"___(Compute the boolean difference of a (clip) cross-section from a another
(subject), assigning the result to the subject.)___";
const auto cross_section__operator_plus_d3c26b9c5ca3 = R"___(Compute the boolean union between two cross-sections.)___";
const auto cross_section__operator_plus_eq_9daa44be9a1d = R"___(Compute the boolean union between two cross-sections, assigning the result
to the first.)___";
const auto cross_section__operator_xor_76de317c9be1 = R"___(Compute the boolean intersection between two cross-sections.)___";
const auto cross_section__operator_xor_eq_705353363fb1 = R"___(Compute the boolean intersection between two cross-sections, assigning the
result to the first.)___";
const auto cross_section__rotate_7c6bae9524e7 = R"___(Applies a (Z-axis) rotation to the CrossSection, in degrees. This operation
can be chained. Transforms are combined and applied lazily.
:param degrees: degrees about the Z-axis to rotate.)___";
const auto cross_section__scale_8913c878f656 = R"___(Scale this CrossSection in space. This operation can be chained. Transforms
are combined and applied lazily.
:param v: The vector to multiply every vertex by per component.)___";
const auto cross_section__simplify_dbac3e60acf4 = R"___(Remove vertices from the contours in this CrossSection that are less than
the specified distance epsilon from an imaginary line that passes through
its two adjacent vertices. Near duplicate vertices and collinear points
will be removed at lower epsilons, with elimination of line segments
becoming increasingly aggressive with larger epsilons.
It is recommended to apply this function following Offset, in order to
clean up any spurious tiny line segments introduced that do not improve
quality in any meaningful way. This is particularly important if further
offseting operations are to be performed, which would compound the issue.)___";
const auto cross_section__square_67088e89831e = R"___(Constructs a square with the given XY dimensions. By default it is
positioned in the first quadrant, touching the origin. If any dimensions in
size are negative, or if all are zero, an empty Manifold will be returned.
:param size: The X, and Y dimensions of the square.
:param center: Set to true to shift the center to the origin.)___";
const auto cross_section__to_polygons_6f4cb60dbd78 = R"___(Return the contours of this CrossSection as a Polygons.)___";
const auto cross_section__transform_baddfca7ede3 = R"___(Transform this CrossSection in space. The first two columns form a 2x2
matrix transform and the last is a translation vector. This operation can
be chained. Transforms are combined and applied lazily.
:param m: The affine transform matrix to apply to all the vertices.)___";
const auto cross_section__translate_339895387e15 = R"___(Move this CrossSection in space. This operation can be chained. Transforms
are combined and applied lazily.
:param v: The vector to add to every vertex.)___";
const auto cross_section__warp_180cafeaaad1 = R"___(Move the vertices of this CrossSection (creating a new one) according to
any arbitrary input function, followed by a union operation (with a
Positive fill rule) that ensures any introduced intersections are not
included in the result.
:param warp_func: A function that modifies a given vertex position.)___";
const auto cross_section__warp_batch_f843ce28c677 = R"___(Same as CrossSection::Warp but calls warpFunc with
a VecView which is roughly equivalent to std::span
pointing to all vec2 elements to be modified in-place
:param warp_func: A function that modifies multiple vertex positions.)___";
const auto get_circular_segments_3f31ccff2bbc = R"___(Determine the result of the SetMinCircularAngle(),
SetMinCircularEdgeLength(), and SetCircularSegments() defaults.
:param radius: For a given radius of circle, determine how many default
segments there will be.)___";
const auto manifold__as_original_2653ccf900dd = R"___(This function condenses all coplanar faces in the relation, and
collapses those edges. In the process the relation to ancestor meshes is lost
and this new Manifold is marked an original. Properties are preserved, so if
they do not match across an edge, that edge will be kept.)___";
const auto manifold__boolean_f506a86f19f9 = R"___(The central operation of this library: the Boolean combines two manifolds
into another by calculating their intersections and removing the unused
portions.
[&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid)
inputs will produce &epsilon;-valid output. &epsilon;-invalid input may fail
triangulation.
These operations are optimized to produce nearly-instant results if either
input is empty or their bounding boxes do not overlap.
:param second: The other Manifold.
:param op: The type of operation to perform.)___";
const auto manifold__bounding_box_f0b931ac75e5 = R"___(Returns the axis-aligned bounding box of all the Manifold's vertices.)___";
const auto manifold__calculate_curvature_0bb0af918c76 = R"___(Curvature is the inverse of the radius of curvature, and signed such that
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
const auto manifold__compose_6c382bb1612b = R"___(Constructs a new manifold from a vector of other manifolds. This is a purely
topological operation, so care should be taken to avoid creating
overlapping results. It is the inverse operation of Decompose().
:param manifolds: A vector of Manifolds to lazy-union together.)___";
const auto manifold__cube_64d1c43c52ed = R"___(Constructs a unit cube (edge lengths all one), by default in the first
octant, touching the origin. If any dimensions in size are negative, or if
all are zero, an empty Manifold will be returned.
:param size: The X, Y, and Z dimensions of the box.
:param center: Set to true to shift the center to the origin.)___";
const auto manifold__cylinder_af7b1b7dc893 = R"___(A convenience constructor for the common case of extruding a circle. Can also
form cones if both radii are specified.
:param height: Z-extent
:param radius_low: Radius of bottom circle. Must be positive.
:param radius_high: Radius of top circle. Can equal zero. Default is equal to
radiusLow.
:param circular_segments: How many line segments to use around the circle.
Default is calculated by the static Defaults.
:param center: Set to true to shift the center to the origin. Default is
origin at the bottom.)___";
const auto manifold__decompose_88ccab82c740 = R"___(This operation returns a vector of Manifolds that are topologically
disconnected. If everything is connected, the vector is length one,
containing a copy of the original. It is the inverse operation of Compose().)___";
const auto manifold__extrude_bc84f1554abe = R"___(Constructs a manifold from a set of polygons by extruding them along the
Z-axis.
Note that high twistDegrees with small nDivisions may cause
self-intersection. This is not checked here and it is up to the user to
choose the correct parameters.
:param cross_section: A set of non-overlapping polygons to extrude.
:param height: Z-extent of extrusion.
:param n_divisions: Number of extra copies of the crossSection to insert into
the shape vertically; especially useful in combination with twistDegrees to
avoid interpolation artifacts. Default is none.
:param twist_degrees: Amount to twist the top crossSection relative to the
bottom, interpolated linearly for the divisions in between.
:param scale_top: Amount to scale the top (independently in X and Y). If the
scale is {0, 0}, a pure cone is formed with only a single vertex at the top.
Note that scale is applied after twist.
Default {1, 1}.)___";
const auto manifold__genus_75c215a950f8 = R"___(The genus is a topological property of the manifold, representing the number
of "handles". A sphere is 0, torus 1, etc. It is only meaningful for a single
mesh, so it is best to call Decompose() first.)___";
const auto manifold__get_mesh_5654c880b26b = R"___(This returns a Mesh of simple vectors of vertices and triangles suitable for
saving or other operations outside of the context of this library.)___";
const auto manifold__get_mesh_gl_731af09ec81f = R"___(The most complete output of this library, returning a MeshGL that is designed
to easily push into a renderer, including all interleaved vertex properties
that may have been input. It also includes relations to all the input meshes
that form a part of this result and the transforms applied to each.
:param normal_idx: If the original MeshGL inputs that formed this manifold had
properties corresponding to normal vectors, you can specify which property
channels these are (x, y, z), which will cause this output MeshGL to
automatically update these normals according to the applied transforms and
front/back side. Each channel must be >= 3 and < numProp, and all original
MeshGLs must use the same channels for their normals.)___";
const auto manifold__get_properties_ba109b37a202 = R"___(Returns the surface area and volume of the manifold.)___";
const auto manifold__hull_1fd92449f7cb = R"___(Compute the convex hull of this manifold.)___";
const auto manifold__hull_b0113f48020a = R"___(Compute the convex hull of a set of points. If the given points are fewer
than 4, or they are all coplanar, an empty Manifold will be returned.
:param pts: A vector of 3-dimensional points over which to compute a convex
hull.)___";
const auto manifold__hull_e1568f7f16f2 = R"___(Compute the convex hull enveloping a set of manifolds.
:param manifolds: A vector of manifolds over which to compute a convex hull.)___";
const auto manifold__is_empty_8f3c4e98cca8 = R"___(Does the Manifold have any triangles?)___";
const auto manifold__manifold_37129c244d43 = R"___(Convert a MeshGL into a Manifold, retaining its properties and merging only
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
const auto manifold__manifold_6dd024847cd9 = R"___(Convert a Mesh into a Manifold. Will return an empty Manifold
and set an Error Status if the Mesh is not an oriented 2-manifold. Will
collapse degenerate triangles and unnecessary vertices.
:param mesh: The input Mesh.)___";
const auto manifold__manifold_c130481162c9 = R"___(Construct an empty Manifold.)___";
const auto manifold__matches_tri_normals_c662986590f4 = R"___(The triangle normal vectors are saved over the course of operations rather
than recalculated to avoid rounding error. This checks that triangles still
match their normal vectors within Precision().)___";
const auto manifold__mirror_d798a49656cc = R"___(Mirror this Manifold over the plane described by the unit form of the given
normal vector. If the length of the normal is zero, an empty Manifold is
returned. This operation can be chained. Transforms are combined and applied
lazily.
:param normal: The normal vector of the plane to be mirrored over)___";
const auto manifold__num_degenerate_tris_d86f985c281a = R"___(The number of triangles that are colinear within Precision(). This library
attempts to remove all of these, but it cannot always remove all of them
without changing the mesh by too much.)___";
const auto manifold__num_edge_61b0f3dc7f99 = R"___(The number of edges in the Manifold.)___";
const auto manifold__num_overlaps_4a8cd704b45a = R"___(This is a checksum-style verification of the collider, simply returning the
total number of edge-face bounding box overlaps between this and other.
:param other: A Manifold to overlap with.)___";
const auto manifold__num_prop_745ca800e017 = R"___(The number of properties per vertex in the Manifold.)___";
const auto manifold__num_prop_vert_a7ba865d3e11 = R"___(The number of property vertices in the Manifold. This will always be >=
NumVert, as some physical vertices may be duplicated to account for different
properties on different neighboring triangles.)___";
const auto manifold__num_tri_56a67713dff8 = R"___(The number of triangles in the Manifold.)___";
const auto manifold__num_vert_93a4106c8a53 = R"___(The number of vertices in the Manifold.)___";
const auto manifold__operator_minus_2ecfe0a1eb86 = R"___(Shorthand for Boolean Difference.)___";
const auto manifold__operator_minus_eq_f6df5d8bc05c = R"___(Shorthand for Boolean Difference assignment.)___";
const auto manifold__operator_plus_a473f00d1659 = R"___(Shorthand for Boolean Union.)___";
const auto manifold__operator_plus_eq_b7ba3403e755 = R"___(Shorthand for Boolean Union assignment.)___";
const auto manifold__operator_xor_4da4601e403b = R"___(Shorthand for Boolean Intersection.)___";
const auto manifold__operator_xor_eq_0116ab409eee = R"___(Shorthand for Boolean Intersection assignment.)___";
const auto manifold__original_id_d1c064807f94 = R"___(If this mesh is an original, this returns its meshID that can be referenced
by product manifolds' MeshRelation. If this manifold is a product, this
returns -1.)___";
const auto manifold__precision_bb888ab8ec11 = R"___(Returns the precision of this Manifold's vertices, which tracks the
approximate rounding error over all the transforms and operations that have
led to this state. Any triangles that are colinear within this precision are
considered degenerate and removed. This is the value of &epsilon; defining
[&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).)___";
const auto manifold__project_e28980e0f682 = R"___(Returns a cross section representing the projected outline of this object
onto the X-Y plane.)___";
const auto manifold__refine_c64a4fc78137 = R"___(Increase the density of the mesh by splitting every edge into n pieces. For
instance, with n = 2, each triangle will be split into 4 triangles. These
will all be coplanar (and will not be immediately collapsed) unless the
Mesh/Manifold has halfedgeTangents specified (e.g. from the Smooth()
constructor), in which case the new vertices will be moved to the
interpolated surface according to their barycentric coordinates.
:param n: The number of pieces to split every edge into. Must be > 1.)___";
const auto manifold__reserve_ids_a514f84c6343 = R"___(Returns the first of n sequential new unique mesh IDs for marking sets of
triangles that can be looked up after further operations. Assign to
MeshGL.runOriginalID vector.)___";
const auto manifold__revolve_c916603e7a75 = R"___(Constructs a manifold from a set of polygons by revolving this cross-section
around its Y-axis and then setting this as the Z-axis of the resulting
manifold. If the polygons cross the Y-axis, only the part on the positive X
side is used. Geometrically valid input will result in geometrically valid
output.
:param cross_section: A set of non-overlapping polygons to revolve.
:param circular_segments: Number of segments along its diameter. Default is
calculated by the static Defaults.
:param revolve_degrees: Number of degrees to revolve. Default is 360 degrees.)___";
const auto manifold__rotate_f0bb7b7d2f38 = R"___(Applies an Euler angle rotation to the manifold, first about the X axis, then
Y, then Z, in degrees. We use degrees so that we can minimize rounding error,
and eliminate it completely for any multiples of 90 degrees. Additionally,
more efficient code paths are used to update the manifold when the transforms
only rotate by multiples of 90 degrees. This operation can be chained.
Transforms are combined and applied lazily.
:param x_degrees: First rotation, degrees about the X-axis.
:param y_degrees: Second rotation, degrees about the Y-axis.
:param z_degrees: Third rotation, degrees about the Z-axis.)___";
const auto manifold__scale_244be87a307d = R"___(Scale this Manifold in space. This operation can be chained. Transforms are
combined and applied lazily.
:param v: The vector to multiply every vertex by per component.)___";
const auto manifold__set_properties_6a3457d9ec71 = R"___(Create a new copy of this manifold with updated vertex properties by
supplying a function that takes the existing position and properties as
input. You may specify any number of output properties, allowing creation and
removal of channels. Note: undefined behavior will result if you read past
the number of input properties or write past the number of output properties.
:param num_prop: The new number of properties per vertex.
:param prop_func: A function that modifies the properties of a given vertex.)___";
const auto manifold__slice_7d90a75e7913 = R"___(Returns the cross section of this object parallel to the X-Y plane at the
specified Z height, defaulting to zero. Using a height equal to the bottom of
the bounding box will return the bottom faces, while using a height equal to
the top of the bounding box will return empty.)___";
const auto manifold__smooth_66eaffd9331b = R"___(Constructs a smooth version of the input mesh by creating tangents; this
method will throw if you have supplied tangents with your mesh already. The
actual triangle resolution is unchanged; use the Refine() method to
interpolate to a higher-resolution curve.
By default, every edge is calculated for maximum smoothness (very much
approximately), attempting to minimize the maximum mean Curvature magnitude.
No higher-order derivatives are considered, as the interpolation is
independent per triangle, only sharing constraints on their boundaries.
:param mesh_gl: input MeshGL.
:param sharpened_edges: If desired, you can supply a vector of sharpened
halfedges, which should in general be a small subset of all halfedges. Order
of entries doesn't matter, as each one specifies the desired smoothness
(between zero and one, with one the default for all unspecified halfedges)
and the halfedge index (3 * triangle index + [0,1,2] where 0 is the edge
between triVert 0 and 1, etc).
At a smoothness value of zero, a sharp crease is made. The smoothness is
interpolated along each edge, so the specified value should be thought of as
an average. Where exactly two sharpened edges meet at a vertex, their
tangents are rotated to be colinear so that the sharpened edge can be
continuous. Vertices with only one sharpened edge are completely smooth,
allowing sharpened edges to smoothly vanish at termination. A single vertex
can be sharpened by sharping all edges that are incident on it, allowing
cones to be formed.)___";
const auto manifold__smooth_f788c5a0c633 = R"___(Constructs a smooth version of the input mesh by creating tangents; this
method will throw if you have supplied tangents with your mesh already. The
actual triangle resolution is unchanged; use the Refine() method to
interpolate to a higher-resolution curve.
By default, every edge is calculated for maximum smoothness (very much
approximately), attempting to minimize the maximum mean Curvature magnitude.
No higher-order derivatives are considered, as the interpolation is
independent per triangle, only sharing constraints on their boundaries.
:param mesh: input Mesh.
:param sharpened_edges: If desired, you can supply a vector of sharpened
halfedges, which should in general be a small subset of all halfedges. Order
of entries doesn't matter, as each one specifies the desired smoothness
(between zero and one, with one the default for all unspecified halfedges)
and the halfedge index (3 * triangle index + [0,1,2] where 0 is the edge
between triVert 0 and 1, etc).
At a smoothness value of zero, a sharp crease is made. The smoothness is
interpolated along each edge, so the specified value should be thought of as
an average. Where exactly two sharpened edges meet at a vertex, their
tangents are rotated to be colinear so that the sharpened edge can be
continuous. Vertices with only one sharpened edge are completely smooth,
allowing sharpened edges to smoothly vanish at termination. A single vertex
can be sharpened by sharping all edges that are incident on it, allowing
cones to be formed.)___";
const auto manifold__sphere_6781451731f0 = R"___(Constructs a geodesic sphere of a given radius.
:param radius: Radius of the sphere. Must be positive.
:param circular_segments: Number of segments along its
diameter. This number will always be rounded up to the nearest factor of
four, as this sphere is constructed by refining an octahedron. This means
there are a circle of vertices on all three of the axis planes. Default is
calculated by the static Defaults.)___";
const auto manifold__split_by_plane_f411533a14aa = R"___(Convenient version of Split() for a half-space.
:param normal: This vector is normal to the cutting plane and its length does
not matter. The first result is in the direction of this vector, the second
result is on the opposite side.
:param origin_offset: The distance of the plane from the origin in the
direction of the normal vector.)___";
const auto manifold__split_fc2847c7afae = R"___(Split cuts this manifold in two using the cutter manifold. The first result
is the intersection, second is the difference. This is more efficient than
doing them separately.
:param cutter:)___";
const auto manifold__status_b1c2b69ee41e = R"___(Returns the reason for an input Mesh producing an empty Manifold. This Status
only applies to Manifolds newly-created from an input Mesh - once they are
combined into a new Manifold via operations, the status reverts to NoError,
simply processing the problem mesh as empty. Likewise, empty meshes may still
show NoError, for instance if they are small enough relative to their
precision to be collapsed to nothing.)___";
const auto manifold__tetrahedron_7e95f682f35b = R"___(Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
and the rest at similarly symmetric points.)___";
const auto manifold__transform_0390744b2b46 = R"___(Transform this Manifold in space. The first three columns form a 3x3 matrix
transform and the last is a translation vector. This operation can be
chained. Transforms are combined and applied lazily.
:param m: The affine transform matrix to apply to all the vertices.)___";
const auto manifold__translate_fcadc2ca8d6d = R"___(Move this Manifold in space. This operation can be chained. Transforms are
combined and applied lazily.
:param v: The vector to add to every vertex.)___";
const auto manifold__trim_by_plane_066ac34a84b0 = R"___(Identical to SplitByPlane(), but calculating and returning only the first
result.
:param normal: This vector is normal to the cutting plane and its length does
not matter. The result is in the direction of this vector from the plane.
:param origin_offset: The distance of the plane from the origin in the
direction of the normal vector.)___";
const auto manifold__warp_4cc67905f424 = R"___(This function does not change the topology, but allows the vertices to be
moved according to any arbitrary input function. It is easy to create a
function that warps a geometrically valid object into one which overlaps, but
that is not checked here, so it is up to the user to choose their function
with discretion.
:param warp_func: A function that modifies a given vertex position.)___";
const auto manifold__warp_batch_0b44f7bbe36b = R"___(Same as Manifold::Warp but calls warpFunc with with
a VecView which is roughly equivalent to std::span
pointing to all vec3 elements to be modified in-place
:param warp_func: A function that modifies multiple vertex positions.)___";
const auto set_circular_segments_7ebd2a75022a = R"___(Sets the default number of circular segments for the
CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and
Manifold::Revolve() constructors. Overrides the edge length and angle
constraints and sets the number of segments to exactly this value.
:param number: Number of circular segments. Default is 0, meaning no
constraint is applied.)___";
const auto set_min_circular_angle_69463d0b8fac = R"___(Sets an angle constraint the default number of circular segments for the
CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and
Manifold::Revolve() constructors. The number of segments will be rounded up
to the nearest factor of four.
:param angle: The minimum angle in degrees between consecutive segments. The
angle will increase if the the segments hit the minimum edge length.
Default is 10 degrees.)___";
const auto set_min_circular_edge_length_bb1c70addca7 = R"___(Sets a length constraint the default number of circular segments for the
CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and
Manifold::Revolve() constructors. The number of segments will be rounded up
to the nearest factor of four.
:param length: The minimum length of segments. The length will
increase if the the segments hit the minimum angle. Default is 1.0.)___";
const auto triangulate_025ab0b4046f = R"___(@brief Triangulates a set of &epsilon;-valid polygons. If the input is not
&epsilon;-valid, the triangulation may overlap, but will always return a
manifold result that matches the input edge directions.
:param polygons: The set of polygons, wound CCW and representing multiple
polygons and/or holes.
:param precision: The value of &epsilon;, bounding the uncertainty of the
input.
@return std::vector<glm::ivec3> The triangles, referencing the original
polygon points in order.)___";
const auto triangulate_8f1ad11752db = R"___(Ear-clipping triangulator based on David Eberly's approach from Geometric
Tools, but adjusted to handle epsilon-valid polygons, and including a
fallback that ensures a manifold triangulation even for overlapping polygons.
This is an O(n^2) algorithm, but hopefully this is not a big problem as the
number of edges in a given polygon is generally much less than the number of
triangles in a mesh, and relatively few faces even need triangulation.
The main adjustments for robustness involve clipping the sharpest ears first
(a known technique to get higher triangle quality), and doing an exhaustive
search to determine ear convexity exactly if the first geometric result is
within precision.)___";
const auto triangulate_idx_9847f0a1f0f8 = R"___(@brief Triangulates a set of &epsilon;-valid polygons. If the input is not
&epsilon;-valid, the triangulation may overlap, but will always return a
manifold result that matches the input edge directions.
:param polys: The set of polygons, wound CCW and representing multiple
polygons and/or holes. These have 2D-projected positions as well as
references back to the original vertices.
:param precision: The value of &epsilon;, bounding the uncertainty of the
input.
@return std::vector<glm::ivec3> The triangles, referencing the original
vertex indicies.)___";
} // namespace manifold_docs