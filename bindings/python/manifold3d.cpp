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

#include "cross_section.h"
#include "manifold.h"
#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

using namespace manifold;

typedef std::tuple<float, float> Float2;
typedef std::tuple<float, float, float> Float3;

PYBIND11_MODULE(manifold3d, m) {
  m.doc() = "Python binding for the Manifold library.";

  m.def("set_min_circular_angle", Quality::SetMinCircularAngle,
        "Sets an angle constraint the default number of circular segments for "
        "the CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), "
        "and Manifold::Revolve() constructors. The number of segments will be "
        "rounded up to the nearest factor of four."
        "\n\n"
        ":param angle: The minimum angle in degrees between consecutive "
        "segments. The angle will increase if the the segments hit the minimum "
        "edge length.\n"
        "Default is 10 degrees.");

  m.def("set_min_circular_edge_length", Quality::SetMinCircularEdgeLength,
        "Sets a length constraint the default number of circular segments for "
        "the CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), "
        "and Manifold::Revolve() constructors. The number of segments will be "
        "rounded up to the nearest factor of four."
        "\n\n"
        ":param length: The minimum length of segments. The length will "
        "increase if the the segments hit the minimum angle. Default is 1.0.");

  m.def("set_circular_segments", Quality::SetCircularSegments,
        "Sets the default number of circular segments for the "
        "CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and "
        "Manifold::Revolve() constructors. Overrides the edge length and angle "
        "constraints and sets the number of segments to exactly this value."
        "\n\n"
        ":param number: Number of circular segments. Default is 0, meaning no "
        "constraint is applied.");

  m.def("get_circular_segments", Quality::GetCircularSegments,
        "Determine the result of the SetMinCircularAngle(), "
        "SetMinCircularEdgeLength(), and SetCircularSegments() defaults."
        "\n\n"
        ":param radius: For a given radius of circle, determine how many "
        "default");

  py::class_<Manifold>(m, "Manifold")
      .def(py::init<>())
      .def(py::init([](std::vector<Manifold> &manifolds) {
             Manifold result;
             if (manifolds.size() >= 1) {
               // for some reason using Manifold() as the initial object
               // will cause failure for python specifically
               // unable to reproduce with c++ directly
               Manifold first = manifolds[0];
               for (int i = 1; i < manifolds.size(); i++) first += manifolds[i];
               return first;
             } else {
               return Manifold();
             }
           }),
           "Construct manifold as the union of a set of manifolds.")
      .def(py::self + py::self, "Boolean union.")
      .def(py::self - py::self, "Boolean difference.")
      .def(py::self ^ py::self, "Boolean intersection.")
      .def(
          "hull", [](Manifold &self) { return self.Hull(); },
          "Compute the convex hull of all points in this manifold.")
      .def_static(
          "batch_hull",
          [](std::vector<Manifold> &ms) { return Manifold::Hull(ms); },
          "Compute the convex hull enveloping a set of manifolds.")
      .def_static(
          "hull_points",
          [](std::vector<Float3> &pts) {
            std::vector<glm::vec3> vec(pts.size());
            for (int i = 0; i < pts.size(); i++) {
              vec[i] = {std::get<0>(pts[i]), std::get<1>(pts[i]),
                        std::get<2>(pts[i])};
            }
            return Manifold::Hull(vec);
          },
          "Compute the convex hull enveloping a set of 3d points.")
      .def(
          "transform",
          [](Manifold &self, py::array_t<float> &mat) {
            auto mat_view = mat.unchecked<2>();
            if (mat_view.shape(0) != 3 || mat_view.shape(1) != 4)
              throw std::runtime_error("Invalid matrix shape");
            glm::mat4x3 mat_glm;
            for (int i = 0; i < 3; i++) {
              for (int j = 0; j < 4; j++) {
                mat_glm[j][i] = mat_view(i, j);
              }
            }
            return self.Transform(mat_glm);
          },
          py::arg("m"),
          "Transform this Manifold in space. The first three columns form a "
          "3x3 matrix transform and the last is a translation vector. This "
          "operation can be chained. Transforms are combined and applied "
          "lazily.\n"
          "\n\n"
          ":param m: The affine transform matrix to apply to all the vertices.")
      .def(
          "translate",
          [](Manifold &self, float x = 0.0f, float y = 0.0f, float z = 0.0f) {
            return self.Translate(glm::vec3(x, y, z));
          },
          py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f,
          "Move this Manifold in space. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param x: X axis translation. (default 0.0).\n"
          ":param y: Y axis translation. (default 0.0).\n"
          ":param z: Z axis translation. (default 0.0).")
      .def(
          "translate",
          [](Manifold &self, py::array_t<float> &t) {
            auto t_view = t.unchecked<1>();
            if (t.shape(0) != 3)
              throw std::runtime_error("Invalid vector shape");
            return self.Translate(glm::vec3(t_view(0), t_view(1), t_view(2)));
          },
          py::arg("t"),
          "Move this Manifold in space. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param v: The vector to add to every vertex.")
      .def(
          "scale",
          [](Manifold &self, float scale) {
            return self.Scale(glm::vec3(scale));
          },
          py::arg("scale"),
          "Scale this Manifold in space. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param scale: The scalar multiplier for each component of every "
          "vertices.")
      .def(
          "scale",
          [](Manifold &self, py::array_t<float> &scale) {
            auto scale_view = scale.unchecked<1>();
            if (scale_view.shape(0) != 3)
              throw std::runtime_error("Invalid vector shape");
            glm::vec3 v(scale_view(0), scale_view(1), scale_view(2));
            return self.Scale(v);
          },
          py::arg("v"),
          "Scale this Manifold in space. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param v: The vector to multiply every vertex by component.")
      .def(
          "mirror",
          [](Manifold &self, py::array_t<float> &mirror) {
            auto v_view = mirror.unchecked<1>();
            if (v_view.shape(0) != 3)
              throw std::runtime_error("Invalid vector shape");
            glm::vec3 v(v_view(0), v_view(1), v_view(2));
            return self.Mirror(v);
          },
          py::arg("v"),
          "Mirror this Manifold in space. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param mirror: The vector defining the axis of mirroring.")
      .def(
          "rotate",
          [](Manifold &self, py::array_t<float> &v) {
            auto v_view = v.unchecked<1>();
            if (v_view.shape(0) != 3)
              throw std::runtime_error("Invalid vector shape");
            return self.Rotate(v_view(0), v_view(1), v_view(2));
          },
          py::arg("v"),
          "Applies an Euler angle rotation to the manifold, first about the X "
          "axis, then Y, then Z, in degrees. We use degrees so that we can "
          "minimize rounding error, and eliminate it completely for any "
          "multiples of 90 degrees. Additionally, more efficient code paths "
          "are used to update the manifold when the transforms only rotate by "
          "multiples of 90 degrees. This operation can be chained. Transforms "
          "are combined and applied lazily."
          "\n\n"
          ":param v: [X, Y, Z] rotation in degrees.")
      .def(
          "rotate",
          [](Manifold &self, float xDegrees = 0.0f, float yDegrees = 0.0f,
             float zDegrees = 0.0f) {
            return self.Rotate(xDegrees, yDegrees, zDegrees);
          },
          py::arg("x_degrees") = 0.0f, py::arg("y_degrees") = 0.0f,
          py::arg("z_degrees") = 0.0f,
          "Applies an Euler angle rotation to the manifold, first about the X "
          "axis, then Y, then Z, in degrees. We use degrees so that we can "
          "minimize rounding error, and eliminate it completely for any "
          "multiples of 90 degrees. Additionally, more efficient code paths "
          "are used to update the manifold when the transforms only rotate by "
          "multiples of 90 degrees. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param x: X rotation in degrees. (default 0.0).\n"
          ":param y: Y rotation in degrees. (default 0.0).\n"
          ":param z: Z rotation in degrees. (default 0.0).")
      .def(
          "warp",
          [](Manifold &self, const std::function<Float3(Float3)> &f) {
            return self.Warp([&f](glm::vec3 &v) {
              Float3 fv = f(std::make_tuple(v.x, v.y, v.z));
              v.x = std::get<0>(fv);
              v.y = std::get<1>(fv);
              v.z = std::get<2>(fv);
            });
          },
          py::arg("f"),
          "This function does not change the topology, but allows the vertices "
          "to be moved according to any arbitrary input function. It is easy "
          "to create a function that warps a geometrically valid object into "
          "one which overlaps, but that is not checked here, so it is up to "
          "the user to choose their function with discretion."
          "\n\n"
          ":param warpFunc: A function that modifies a given vertex position.")
      .def(
          "refine", [](Manifold &self, int n) { return self.Refine(n); },
          py::arg("n"),
          "Increase the density of the mesh by splitting every edge into n "
          "pieces. For instance, with n = 2, each triangle will be split into "
          "4 triangles. These will all be coplanar (and will not be "
          "immediately collapsed) unless the Mesh/Manifold has "
          "halfedgeTangents specified (e.g. from the Smooth() constructor), "
          "in which case the new vertices will be moved to the interpolated "
          "surface according to their barycentric coordinates.\n"
          "\n"
          ":param n: The number of pieces to split every edge into. Must be > "
          "1.")
      .def("to_mesh", &Manifold::GetMesh)
      .def("num_vert", &Manifold::NumVert,
           "The number of vertices in the Manifold.")
      .def("num_edge", &Manifold::NumEdge,
           "The number of edges in the Manifold.")
      .def("num_tri", &Manifold::NumTri,
           "The number of triangles in the Manifold.")
      .def("num_prop", &Manifold::NumProp,
           "The number of properties per vertex in the Manifold")
      .def("num_prop_vert", &Manifold::NumPropVert,
           "The number of property vertices in the Manifold. This will always "
           "be >= NumVert, as some physical vertices may be duplicated to "
           "account for different properties on different neighboring "
           "triangles.")
      .def("precision", &Manifold::Precision,
           "Returns the precision of this Manifold's vertices, which tracks "
           "the approximate rounding error over all the transforms and "
           "operations that have led to this state. Any triangles that are "
           "colinear within this precision are considered degenerate and "
           "removed. This is the value of &epsilon; defining "
           "[&epsilon;-valid](https://github.com/elalish/manifold/wiki/"
           "Manifold-Library#definition-of-%CE%B5-valid).")
      .def("genus", &Manifold::Genus,
           "The genus is a topological property of the manifold, representing "
           "the number of \"handles\". A sphere is 0, torus 1, etc. It is only "
           "meaningful for a single mesh, so it is best to call Decompose() "
           "first.")
      .def(
          "get_volume",
          [](Manifold &self) { return self.GetProperties().volume; },
          "Get the volume of the manifold\n This is clamped to zero for a "
          "given face if they are within the Precision().")
      .def(
          "get_surface_area",
          [](Manifold &self) { return self.GetProperties().surfaceArea; },
          "Get the surface area of the manifold\n This is clamped to zero for "
          "a given face if they are within the Precision().")
      .def("original_id", &Manifold::OriginalID,
           "If this mesh is an original, this returns its meshID that can be "
           "referenced by product manifolds' MeshRelation. If this manifold is "
           "a product, this returns -1.")
      .def("as_original", &Manifold::AsOriginal,
           "This function condenses all coplanar faces in the relation, and "
           "collapses those edges. In the process the relation to ancestor "
           "meshes is lost and this new Manifold is marked an original. "
           "Properties are preserved, so if they do not match across an edge, "
           "that edge will be kept.")
      .def("is_empty", &Manifold::IsEmpty,
           "Does the Manifold have any triangles?")
      .def(
          "decompose", [](Manifold &self) { return self.Decompose(); },
          "This operation returns a vector of Manifolds that are "
          "topologically disconnected. If everything is connected, the vector "
          "is length one, containing a copy of the original. It is the inverse "
          "operation of Compose().")
      .def("split", &Manifold::Split,
           "Split cuts this manifold in two using the cutter manifold. The "
           "first result is the intersection, second is the difference. This "
           "is more efficient than doing them separately.")
      .def(
          "split_by_plane",
          [](Manifold &self, Float3 normal, float originOffset) {
            return self.SplitByPlane(
                {std::get<0>(normal), std::get<1>(normal), std::get<2>(normal)},
                originOffset);
          },
          py::arg("normal"), py::arg("origin_offset"),
          "Convenient version of Split() for a half-space."
          "\n\n"
          ":param normal: This vector is normal to the cutting plane and its "
          "length does not matter. The first result is in the direction of "
          "this vector, the second result is on the opposite side.\n"
          ":param originOffset: The distance of the plane from the origin in "
          "the direction of the normal vector.")
      .def(
          "trim_by_plane",
          [](Manifold &self, Float3 normal, float originOffset) {
            return self.TrimByPlane(
                {std::get<0>(normal), std::get<1>(normal), std::get<2>(normal)},
                originOffset);
          },
          py::arg("normal"), py::arg("origin_offset"),
          "Identical to SplitByPlane(), but calculating and returning only the "
          "first result."
          "\n\n"
          ":param normal: This vector is normal to the cutting plane and its "
          "length does not matter. The result is in the direction of this "
          "vector from the plane.\n"
          ":param originOffset: The distance of the plane from the origin in "
          "the direction of the normal vector.")
      .def_property_readonly(
          "bounding_box",
          [](Manifold &self) {
            auto b = self.BoundingBox();
            py::tuple box = py::make_tuple(b.min[0], b.min[1], b.min[2],
                                           b.max[0], b.max[1], b.max[2]);
            return box;
          },
          "Gets the manifold bounding box as a tuple "
          "(xmin, ymin, zmin, xmax, ymax, zmax).")
      .def_static(
          "smooth", [](const Mesh &mesh) { return Manifold::Smooth(mesh); },
          "Constructs a smooth version of the input mesh by creating tangents; "
          "this method will throw if you have supplied tangents with your "
          "mesh already. The actual triangle resolution is unchanged; use the "
          "Refine() method to interpolate to a higher-resolution curve."
          "\n\n"
          "By default, every edge is calculated for maximum smoothness (very "
          "much approximately), attempting to minimize the maximum mean "
          "Curvature magnitude. No higher-order derivatives are considered, "
          "as the interpolation is independent per triangle, only sharing "
          "constraints on their boundaries."
          "\n\n"
          ":param mesh: input Mesh.\n"
          ":param sharpenedEdges: If desired, you can supply a vector of "
          "sharpened halfedges, which should in general be a small subset of "
          "all halfedges. Order of entries doesn't matter, as each one "
          "specifies the desired smoothness (between zero and one, with one "
          "the default for all unspecified halfedges) and the halfedge index "
          "(3 * triangle index + [0,1,2] where 0 is the edge between triVert 0 "
          "and 1, etc)."
          "\n\n"
          "At a smoothness value of zero, a sharp crease is made. The "
          "smoothness is interpolated along each edge, so the specified value "
          "should be thought of as an average. Where exactly two sharpened "
          "edges meet at a vertex, their tangents are rotated to be colinear "
          "so that the sharpened edge can be continuous. Vertices with only "
          "one sharpened edge are completely smooth, allowing sharpened edges "
          "to smoothly vanish at termination. A single vertex can be sharpened "
          "by sharping all edges that are incident on it, allowing cones to be "
          "formed.")
      .def_static(
          "from_mesh", [](const Mesh &mesh) { return Manifold(mesh); },
          py::arg("mesh"))
      .def_static(
          "tetrahedron", []() { return Manifold::Tetrahedron(); },
          "Constructs a tetrahedron centered at the origin with one vertex at "
          "(1,1,1) and the rest at similarly symmetric points.")
      .def_static(
          "cube",
          [](Float3 size, bool center = false) {
            return Manifold::Cube(
                glm::vec3(std::get<0>(size), std::get<1>(size),
                          std::get<2>(size)),
                center);
          },
          py::arg("size") = std::make_tuple(1.0f, 1.0f, 1.0f),
          py::arg("center") = false,
          "Constructs a unit cube (edge lengths all one), by default in the "
          "first octant, touching the origin."
          "\n\n"
          ":param size: The X, Y, and Z dimensions of the box.\n"
          ":param center: Set to true to shift the center to the origin.")
      .def_static(
          "cube",
          [](py::array_t<float> &size, bool center = false) {
            auto size_view = size.unchecked<1>();
            if (size_view.shape(0) != 3)
              throw std::runtime_error("Invalid vector shape");
            return Manifold::Cube(
                glm::vec3(size_view(0), size_view(1), size_view(2)), center);
          },
          py::arg("size"), py::arg("center") = false,
          "Constructs a unit cube (edge lengths all one), by default in the "
          "first octant, touching the origin."
          "\n\n"
          ":param size: The X, Y, and Z dimensions of the box.\n"
          ":param center: Set to true to shift the center to the origin.")
      .def_static(
          "cube",
          [](float x, float y, float z, bool center = false) {
            return Manifold::Cube(glm::vec3(x, y, z), center);
          },
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("center") = false,
          "Constructs a unit cube (edge lengths all one), by default in the "
          "first octant, touching the origin."
          "\n\n"
          ":param x: The X dimensions of the box.\n"
          ":param y: The Y dimensions of the box.\n"
          ":param z: The Z dimensions of the box.\n"
          ":param center: Set to true to shift the center to the origin.")
      .def_static(
          "cylinder",
          [](float height, float radiusLow, float radiusHigh = -1.0f,
             int circularSegments = 0, bool center = false) {
            return Manifold::Cylinder(height, radiusLow, radiusHigh,
                                      circularSegments, center);
          },
          py::arg("height"), py::arg("radius_low"),
          py::arg("radius_high") = -1.0f, py::arg("circular_segments") = 0,
          py::arg("center") = false,
          "A convenience constructor for the common case of extruding a "
          "circle. Can also form cones if both radii are specified."
          "\n\n"
          ":param height: Z-extent\n"
          ":param radiusLow: Radius of bottom circle. Must be positive.\n"
          ":param radiusHigh: Radius of top circle. Can equal zero. Default "
          "(-1) is equal to radiusLow.\n"
          ":param circularSegments: How many line segments to use around the "
          "circle. Default (-1) is calculated by the static Defaults.\n"
          ":param center: Set to true to shift the center to the origin. "
          "Default is origin at the bottom.")
      .def_static(
          "sphere",
          [](float radius, int circularSegments = 0) {
            return Manifold::Sphere(radius, circularSegments);
          },
          py::arg("radius"), py::arg("circular_segments") = 0,
          "Constructs a geodesic sphere of a given radius.\n"
          "\n"
          ":param radius: Radius of the sphere. Must be positive.\n"
          ":param circularSegments: Number of segments along its diameter. "
          "This number will always be rounded up to the nearest factor of "
          "four, as this sphere is constructed by refining an octahedron. This "
          "means there are a circle of vertices on all three of the axis "
          "planes. Default is calculated by the static Defaults.")
      .def_static("reserve_ids", Manifold::ReserveIDs, py::arg("n"),
                  "Returns the first of n sequential new unique mesh IDs for "
                  "marking sets of triangles that can be looked up after "
                  "further operations. Assign to MeshGL.runOriginalID vector");

  py::class_<Mesh>(m, "Mesh")
      .def(
          py::init([](py::array_t<float> &vertPos, py::array_t<int> &triVerts,
                      std::optional<py::array_t<float>> &vertNormal,
                      std::optional<py::array_t<float>> &halfedgeTangent) {
            auto vertPos_view = vertPos.unchecked<2>();
            auto triVerts_view = triVerts.unchecked<2>();
            if (vertPos_view.shape(1) != 3)
              throw std::runtime_error("Invalid vert_pos shape");
            if (triVerts_view.shape(1) != 3)
              throw std::runtime_error("Invalid tri_verts shape");
            std::vector<glm::vec3> vertPos_vec(vertPos_view.shape(0));
            std::vector<glm::ivec3> triVerts_vec(triVerts_view.shape(0));
            for (int i = 0; i < vertPos_view.shape(0); i++)
              for (const int j : {0, 1, 2})
                vertPos_vec[i][j] = vertPos_view(i, j);
            for (int i = 0; i < triVerts_view.shape(0); i++)
              for (const int j : {0, 1, 2})
                triVerts_vec[i][j] = triVerts_view(i, j);

            // optional arguments
            if (!vertNormal.has_value() && !halfedgeTangent.has_value()) {
              return Mesh({vertPos_vec, triVerts_vec});
            } else {
              auto vertNormal_view = vertNormal.value().unchecked<2>();
              auto halfedgeTangent_view =
                  halfedgeTangent.value().unchecked<2>();
              if (vertNormal_view.shape(0) != 0) {
                if (vertNormal_view.shape(1) != 3)
                  throw std::runtime_error("Invalid vert_normal shape");
                if (vertNormal_view.shape(0) != vertPos_view.shape(0))
                  throw std::runtime_error(
                      "vert_normal must have the same length as vert_pos");
              }
              if (halfedgeTangent_view.shape(0) != 0) {
                if (halfedgeTangent_view.shape(1) != 4)
                  throw std::runtime_error("Invalid halfedge_tangent shape");
                if (halfedgeTangent_view.shape(0) != triVerts_view.shape(0) * 3)
                  throw std::runtime_error(
                      "halfedge_tangent must be three times as long as "
                      "tri_verts");
              }
              std::vector<glm::vec3> vertNormal_vec(vertNormal_view.shape(0));
              std::vector<glm::vec4> halfedgeTangent_vec(
                  halfedgeTangent_view.shape(0));
              for (int i = 0; i < vertNormal_view.shape(0); i++)
                for (const int j : {0, 1, 2})
                  vertNormal_vec[i][j] = vertNormal_view(i, j);
              for (int i = 0; i < halfedgeTangent_view.shape(0); i++)
                for (const int j : {0, 1, 2, 3})
                  halfedgeTangent_vec[i][j] = halfedgeTangent_view(i, j);

              return Mesh({vertPos_vec, triVerts_vec, vertNormal_vec,
                           halfedgeTangent_vec});
            }
          }),
          py::arg("vert_pos"), py::arg("tri_verts"),
          py::arg("vert_normal") = py::none(),
          py::arg("halfedge_tangent") = py::none())
      .def_property_readonly("vert_pos",
                             [](Mesh &self) {
                               const int numVert = self.vertPos.size();
                               py::array_t<float> vert_pos({numVert, 3});
                               auto vert_pos_view =
                                   vert_pos.mutable_unchecked<2>();

                               for (int i = 0; i < numVert; ++i)
                                 for (const int j : {0, 1, 2})
                                   vert_pos_view(i, j) = self.vertPos[i][j];
                               return vert_pos;
                             })
      .def_property_readonly("tri_verts",
                             [](Mesh &self) {
                               const int numTri = self.triVerts.size();
                               py::array_t<int> tri_verts({numTri, 3});
                               auto tri_verts_view =
                                   tri_verts.mutable_unchecked<2>();

                               for (int i = 0; i < numTri; ++i)
                                 for (const int j : {0, 1, 2})
                                   tri_verts_view(i, j) = self.triVerts[i][j];
                               return tri_verts;
                             })
      .def_property_readonly(
          "vert_normal",
          [](Mesh &self) {
            const int numVert = self.vertNormal.size();
            py::array_t<float> vert_normal({numVert, 3});
            auto vert_normal_view = vert_normal.mutable_unchecked<2>();

            for (int i = 0; i < numVert; ++i)
              for (const int j : {0, 1, 2})
                vert_normal_view(i, j) = self.vertNormal[i][j];
            return vert_normal;
          })
      .def_property_readonly("halfedge_tangent", [](Mesh &self) {
        const int numEdge = self.halfedgeTangent.size();
        py::array_t<float> halfedge_tangent({numEdge, 4});
        auto halfedge_tangent_view = halfedge_tangent.mutable_unchecked<2>();

        for (int i = 0; i < numEdge; ++i)
          for (const int j : {0, 1, 2, 3})
            halfedge_tangent_view(i, j) = self.halfedgeTangent[i][j];
        return halfedge_tangent;
      });

  py::enum_<CrossSection::FillRule>(m, "FillRule")
      .value("EvenOdd", CrossSection::FillRule::EvenOdd,
             "Only odd numbered sub-regions are filled.")
      .value("NonZero", CrossSection::FillRule::NonZero,
             "Only non-zero sub-regions are filled.")
      .value("Positive", CrossSection::FillRule::Positive,
             "Only sub-regions with winding counts > 0 are filled.")
      .value("Negative", CrossSection::FillRule::Negative,
             "Only sub-regions with winding counts < 0 are filled.");

  py::enum_<CrossSection::JoinType>(m, "JoinType")
      .value("Square", CrossSection::JoinType::Square,
             "Squaring is applied uniformly at all joins where the internal "
             "join angle is less that 90 degrees. The squared edge will be at "
             "exactly the offset distance from the join vertex.")
      .value(
          "Round", CrossSection::JoinType::Square,
          "Rounding is applied to all joins that have convex external angles, "
          "and it maintains the exact offset distance from the join vertex.")
      .value(
          "Miter", CrossSection::JoinType::Miter,
          "There's a necessary limit to mitered joins (to avoid narrow angled "
          "joins producing excessively long and narrow "
          "[spikes](http://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/"
          "Classes/ClipperOffset/Properties/MiterLimit.htm)). So where mitered "
          "joins would exceed a given maximum miter distance (relative to the "
          "offset distance), these are 'squared' instead.");

  py::class_<CrossSection>(
      m, "CrossSection",
      "Two-dimensional cross sections guaranteed to be without "
      "self-intersections, or overlaps between polygons (from construction "
      "onwards). This class makes use of the "
      "[Clipper2](http://www.angusj.com/clipper2/Docs/Overview.htm) library "
      "for polygon clipping (boolean) and offsetting operations.")
      .def(py::init<>())
      .def(py::init([](std::vector<std::vector<Float2>> &polygons,
                       CrossSection::FillRule fillrule) {
             std::vector<SimplePolygon> simplePolygons(polygons.size());
             for (int i = 0; i < polygons.size(); i++) {
               simplePolygons[i] = std::vector<glm::vec2>(polygons[i].size());
               for (int j = 0; j < polygons[i].size(); j++) {
                 simplePolygons[i][j] = {std::get<0>(polygons[i][j]),
                                         std::get<1>(polygons[i][j])};
               }
             }
             return CrossSection(simplePolygons, fillrule);
           }),
           py::arg("polygons"),
           py::arg("fillrule") = CrossSection::FillRule::Positive,
           "Create a 2d cross-section from a set of contours (complex "
           "polygons). A boolean union operation (with Positive filling rule "
           "by default) performed to combine overlapping polygons and ensure "
           "the resulting CrossSection is free of intersections."
           "\n\n"
           ":param contours: A set of closed paths describing zero or more "
           "complex polygons.\n"
           ":param fillrule: The filling rule used to interpret polygon "
           "sub-regions in contours.")
      .def("area", &CrossSection::Area,
           "Return the total area covered by complex polygons making up the "
           "CrossSection.")
      .def("num_vert", &CrossSection::NumVert,
           "Return the number of vertices in the CrossSection.")
      .def("num_contour", &CrossSection::NumContour,
           "Return the number of contours (both outer and inner paths) in the "
           "CrossSection.")
      .def("is_empty", &CrossSection::IsEmpty,
           "Does the CrossSection contain any contours?")
      .def(
          "translate",
          [](CrossSection self, Float2 v) {
            return self.Translate({std::get<0>(v), std::get<1>(v)});
          },
          "Move this CrossSection in space. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param v: The vector to add to every vertex.")
      .def("rotate", &CrossSection::Rotate,
           "Applies a (Z-axis) rotation to the CrossSection, in degrees. This "
           "operation can be chained. Transforms are combined and applied "
           "lazily."
           "\n\n"
           ":param degrees: degrees about the Z-axis to rotate.")
      .def(
          "scale",
          [](CrossSection self, Float2 s) {
            return self.Scale({std::get<0>(s), std::get<1>(s)});
          },
          "Scale this CrossSection in space. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param v: The vector to multiply every vertex by per component.")
      .def(
          "mirror",
          [](CrossSection self, Float2 ax) {
            return self.Mirror({std::get<0>(ax), std::get<1>(ax)});
          },
          "Mirror this CrossSection over the arbitrary axis described by the "
          "unit form of the given vector. If the length of the vector is zero, "
          "an empty CrossSection is returned. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param ax: the axis to be mirrored over")
      .def(
          "transform",
          [](CrossSection self, py::array_t<float> &mat) {
            auto mat_view = mat.unchecked<2>();
            if (mat_view.shape(0) != 2 || mat_view.shape(1) != 3)
              throw std::runtime_error("Invalid matrix shape");
            glm::mat3x2 mat_glm;
            for (int i = 0; i < 2; i++) {
              for (int j = 0; j < 3; j++) {
                mat_glm[j][i] = mat_view(i, j);
              }
            }
            return self.Transform(mat_glm);
          },
          "Transform this CrossSection in space. The first two columns form a "
          "2x2 matrix transform and the last is a translation vector. This "
          "operation can be chained. Transforms are combined and applied "
          "lazily."
          "\n\n"
          ":param m: The affine transform matrix to apply to all the vertices.")
      .def(
          "warp",
          [](CrossSection self, const std::function<Float2(Float2)> &f) {
            return self.Warp([&f](glm::vec2 &v) {
              Float2 fv = f(std::make_tuple(v.x, v.y));
              v.x = std::get<0>(fv);
              v.y = std::get<1>(fv);
            });
          },
          py::arg("f"),
          "Move the vertices of this CrossSection (creating a new one) "
          "according to any arbitrary input function, followed by a union "
          "operation (with a Positive fill rule) that ensures any introduced "
          "intersections are not included in the result."
          "\n\n"
          ":param warpFunc: A function that modifies a given vertex position.")
      .def("simplify", &CrossSection::Simplify,
           "Remove vertices from the contours in this CrossSection that are "
           "less than the specified distance epsilon from an imaginary line "
           "that passes through its two adjacent vertices. Near duplicate "
           "vertices and collinear points will be removed at lower epsilons, "
           "with elimination of line segments becoming increasingly aggressive "
           "with larger epsilons."
           "\n\n"
           "It is recommended to apply this function following Offset, in "
           "order to clean up any spurious tiny line segments introduced that "
           "do not improve quality in any meaningful way. This is particularly "
           "important if further offseting operations are to be performed, "
           "which would compound the issue.")
      .def("offset", &CrossSection::Offset, py::arg("delta"),
           py::arg("join_type"), py::arg("miter_limit") = 2.0,
           py::arg("arc_tolerance") = 0.0,
           "Inflate the contours in CrossSection by the specified delta, "
           "handling corners according to the given JoinType."
           "\n\n"
           ":param delta: Positive deltas will cause the expansion of "
           "outlining contours to expand, and retraction of inner (hole) "
           "contours. Negative deltas will have the opposite effect.\n"
           ":param jt: The join type specifying the treatment of contour joins "
           "(corners).\n"
           ":param miter_limit: The maximum distance in multiples of delta "
           "that vertices can be offset from their original positions with "
           "before squaring is applied, <B>when the join type is Miter</B> "
           "(default is 2, which is the minimum allowed). See the [Clipper2 "
           "MiterLimit](http://www.angusj.com/clipper2/Docs/Units/"
           "Clipper.Offset/Classes/ClipperOffset/Properties/MiterLimit.htm) "
           "page for a visual example.\n"
           ":param circularSegments: Number of segments per 360 degrees of "
           "<B>JoinType::Round</B> corners (roughly, the number of vertices "
           "that will be added to each contour). Default is calculated by the "
           "static Quality defaults according to the radius.")
      .def(py::self + py::self, "Boolean union.")
      .def(py::self - py::self, "Boolean difference.")
      .def(py::self ^ py::self, "Boolean intersection.")
      .def(
          "hull", [](CrossSection &self) { return self.Hull(); },
          "Compute the convex hull of this cross-section.")
      .def_static(
          "batch_hull",
          [](std::vector<CrossSection> &cs) { return CrossSection::Hull(cs); },
          "Compute the convex hull enveloping a set of cross-sections.")
      .def_static(
          "hull_points",
          [](std::vector<Float2> &pts) {
            std::vector<glm::vec2> poly(pts.size());
            for (int i = 0; i < pts.size(); i++) {
              poly[i] = {std::get<0>(pts[i]), std::get<1>(pts[i])};
            }
            return CrossSection::Hull(poly);
          },
          "Compute the convex hull enveloping a set of 2d points.")
      .def("decompose", &CrossSection::Decompose,
           "This operation returns a vector of CrossSections that are "
           "topologically disconnected, each containing one outline contour "
           "with zero or more holes.")
      .def(
          "to_polygons",
          [](CrossSection self) {
            const Polygons &data = self.ToPolygons();
            py::list polygon_list;
            for (int i = 0; i < data.size(); ++i) {
              py::list polygon;
              for (int j = 0; j < data[i].size(); ++j) {
                auto f = data[i][j];
                py::tuple vertex = py::make_tuple(f[0], f[1]);
                polygon.append(vertex);
              }
              polygon_list.append(polygon);
            }
            return polygon_list;
          },
          "Returns the vertices of the cross-section's polygons as a "
          "List[List[Tuple[float, float]]].")
      .def(
          "extrude",
          [](CrossSection self, float height, int nDivisions = 0,
             float twistDegrees = 0.0f,
             Float2 scaleTop = std::make_tuple(1.0f, 1.0f)) {
            glm::vec2 scaleTopVec(std::get<0>(scaleTop), std::get<1>(scaleTop));
            return Manifold::Extrude(self, height, nDivisions, twistDegrees,
                                     scaleTopVec);
          },
          py::arg("height"), py::arg("n_divisions") = 0,
          py::arg("twist_degrees") = 0.0f,
          py::arg("scale_top") = std::make_tuple(1.0f, 1.0f),
          "Constructs a manifold from the set of polygons by extruding them "
          "along the Z-axis.\n"
          "\n"
          ":param height: Z-extent of extrusion.\n"
          ":param nDivisions: Number of extra copies of the crossSection to "
          "insert into the shape vertically; especially useful in combination "
          "with twistDegrees to avoid interpolation artifacts. Default is "
          "none.\n"
          ":param twistDegrees: Amount to twist the top crossSection relative "
          "to the bottom, interpolated linearly for the divisions in between.\n"
          ":param scaleTop: Amount to scale the top (independently in X and "
          "Y). If the scale is (0, 0), a pure cone is formed with only a "
          "single vertex at the top. Default (1, 1).")
      .def(
          "revolve",
          [](CrossSection self, int circularSegments = 0) {
            return Manifold::Revolve(self, circularSegments);
          },
          py::arg("circular_segments") = 0,
          "Constructs a manifold from the set of polygons by revolving this "
          "cross-section around its Y-axis and then setting this as the Z-axis "
          "of the resulting manifold. If the polygons cross the Y-axis, only "
          "the part on the positive X side is used. Geometrically valid input "
          "will result in geometrically valid output.\n"
          "\n"
          ":param circularSegments: Number of segments along its diameter. "
          "Default is calculated by the static Defaults.")
      .def_static(
          "square",
          [](Float2 dims, bool center) {
            return CrossSection::Square({std::get<0>(dims), std::get<1>(dims)},
                                        center);
          },
          py::arg("dims"), py::arg("center") = false,
          "Constructs a square with the given XY dimensions. By default it is "
          "positioned in the first quadrant, touching the origin. If any "
          "dimensions in size are negative, or if all are zero, an empty "
          "Manifold will be returned."
          "\n\n"
          ":param size: The X, and Y dimensions of the square.\n"
          ":param center: Set to true to shift the center to the origin.")
      .def_static(
          "circle",
          [](float radius, int circularSegments) {
            return CrossSection::Circle(radius, circularSegments);
          },
          py::arg("radius"), py::arg("circularSegments") = 0,
          "Constructs a circle of a given radius."
          "\n\n"
          ":param radius: Radius of the circle. Must be positive.\n"
          ":param circularSegments: Number of segments along its diameter. "
          "Default is calculated by the static Quality defaults according to "
          "the radius.");
}
