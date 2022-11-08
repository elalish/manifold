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

struct PolygonsWrapper {
  std::unique_ptr<Polygons> polygons;
};

PYBIND11_MODULE(pymanifold, m) {
  m.doc() =
      "Python binding for the manifold library. Please check the C++ "
      "documentation for APIs.\n"
      "This binding will perform copying to make the API more familiar to "
      "OpenSCAD users.";
  py::class_<Manifold>(m, "Manifold")
      .def(py::init<>())
      .def(py::init([](std::vector<Manifold> &manifolds) {
             Manifold result;
             for (Manifold &manifold : manifolds) result += manifold;
             return result;
           }),
           "Construct manifold as the union of a set of manifolds.")
      .def(py::self + py::self, "Boolean union.")
      .def(py::self - py::self, "Boolean difference.")
      .def(py::self ^ py::self, "Boolean intersection.")
      .def(
          "transform",
          [](Manifold self, py::array_t<float> &mat) {
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
          "3x3 matrix\n"
          "transform and the last is a translation vector. This operation can "
          "be\n"
          "chained. Transforms are combined and applied lazily.\n"
          "\n"
          ":param m: The affine transform matrix to apply to all the vertices.")
      .def(
          "translate",
          [](Manifold self, float x = 0.0f, float y = 0.0f, float z = 0.0f) {
            return self.Translate(glm::vec3(x, y, z));
          },
          py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f,
          "Move this Manifold in space. This operation can be chained. "
          "Transforms are\n"
          "combined and applied lazily.\n"
          "\n"
          ":param x: X axis translation. (default 0.0).\n"
          ":param y: Y axis translation. (default 0.0).\n"
          ":param z: Z axis translation. (default 0.0).")
      .def(
          "translate",
          [](Manifold self, py::array_t<float> &t) {
            auto t_view = t.unchecked<1>();
            if (t.shape(0) != 3)
              throw std::runtime_error("Invalid vector shape");
            return self.Translate(glm::vec3(t_view(0), t_view(1), t_view(2)));
          },
          py::arg("t"),
          "Move this Manifold in space. This operation can be chained. "
          "Transforms are\n"
          "combined and applied lazily.\n"
          "\n"
          ":param v: The vector to add to every vertex.")
      .def("scale",
           static_cast<Manifold (*)(Manifold, float)>(
               [](Manifold self, float scale) {
                 return self.Scale(glm::vec3(scale));
               }),
           py::arg("scale"),
           "Scale this Manifold in space. This operation can be chained. "
           "Transforms are\n"
           "combined and applied lazily.\n"
           "\n"
           ":param scale: The scalar multiplier for each component of every "
           "vertices.")
      .def(
          "scale",
          [](Manifold self, py::array_t<float> &scale) {
            auto scale_view = scale.unchecked<1>();
            if (scale_view.shape(0) != 3)
              throw std::runtime_error("Invalid vector shape");
            glm::vec3 v(scale_view(0), scale_view(1), scale_view(2));
            return self.Scale(v);
          },
          py::arg("v"),
          "Scale this Manifold in space. This operation can be chained. "
          "Transforms are\n"
          "combined and applied lazily.\n"
          "\n"
          ":param v: The vector to multiply every vertex by component.")
      .def(
          "rotate",
          [](Manifold self, py::array_t<float> &v) {
            auto v_view = v.unchecked<1>();
            if (v_view.shape(0) != 3)
              throw std::runtime_error("Invalid vector shape");
            return self.Rotate(v_view(0), v_view(1), v_view(2));
          },
          py::arg("v"),
          "Applies an Euler angle rotation to the manifold, first about the X "
          "axis, then\n"
          "Y, then Z, in degrees. We use degrees so that we can minimize "
          "rounding error,\n"
          "and eliminate it completely for any multiples of 90 degrees. "
          "Additionally, more\n"
          "efficient code paths are used to update the manifold when the "
          "transforms only\n"
          "rotate by multiples of 90 degrees. This operation can be chained. "
          "Transforms\n"
          "are combined and applied lazily.\n"
          "\n"
          ":param v: [X, Y, Z] rotation in degrees.")
      .def(
          "rotate",
          [](Manifold self, float xDegrees = 0.0f, float yDegrees = 0.0f,
             float zDegrees = 0.0f) {
            return self.Rotate(xDegrees, yDegrees, zDegrees);
          },
          py::arg("x_degrees") = 0.0f, py::arg("y_degrees") = 0.0f,
          py::arg("z_degrees") = 0.0f,
          "Applies an Euler angle rotation to the manifold, first about the X "
          "axis, then\n"
          "Y, then Z, in degrees. We use degrees so that we can minimize "
          "rounding error,\n"
          "and eliminate it completely for any multiples of 90 degrees. "
          "Additionally, more\n"
          "efficient code paths are used to update the manifold when the "
          "transforms only\n"
          "rotate by multiples of 90 degrees. This operation can be chained. "
          "Transforms\n"
          "are combined and applied lazily.\n"
          "\n"
          ":param x: X rotation in degrees. (default 0.0).\n"
          ":param y: Y rotation in degrees. (default 0.0).\n"
          ":param z: Z rotation in degrees. (default 0.0).")
      .def(
          "warp",
          [](Manifold self, const std::function<Float3(Float3)> &f) {
            return self.Warp([&f](glm::vec3 &v) {
              Float3 fv = f(std::make_tuple(v.x, v.y, v.z));
              v.x = std::get<0>(fv);
              v.y = std::get<1>(fv);
              v.z = std::get<2>(fv);
            });
          },
          py::arg("f"))
      .def(
          "refine", [](Manifold self, int n) { return self.Refine(n); },
          py::arg("n"),
          "Increase the density of the mesh by splitting every edge into n "
          "pieces. For\n"
          "instance, with n = 2, each triangle will be split into 4 triangles. "
          "These\n"
          "will all be coplanar (and will not be immediately collapsed) unless "
          "the\n"
          "Mesh/Manifold has halfedgeTangents specified (e.g. from the "
          "Smooth()\n"
          "constructor), in which case the new vertices will be moved to the\n"
          "interpolated surface according to their barycentric coordinates.\n"
          "\n"
          ":param n: The number of pieces to split every edge into. Must be > "
          "1.")
      .def("to_mesh", &Manifold::GetMesh)
      .def_static("smooth", Manifold::Smooth,

                  "Constructs a smooth version of the input mesh by creating "
                  "tangents; this\n"
                  "method will throw if you have supplied tangents with your "
                  "mesh already. The\n"
                  "actual triangle resolution is unchanged; use the Refine() "
                  "method to\n"
                  "interpolate to a higher-resolution curve.\n"
                  "\n"
                  "By default, every edge is calculated for maximum smoothness "
                  "(very much\n"
                  "approximately), attempting to minimize the maximum mean "
                  "Curvature magnitude.\n"
                  "No higher-order derivatives are considered, as the "
                  "interpolation is\n"
                  "independent per triangle, only sharing constraints on their "
                  "boundaries.\n"
                  "\n"
                  ":param mesh: input Mesh.\n"
                  ":param sharpenedEdges: If desired, you can supply a vector "
                  "of sharpened\n"
                  "halfedges, which should in general be a small subset of all "
                  "halfedges. Order\n"
                  "of entries doesn't matter, as each one specifies the "
                  "desired smoothness\n"
                  "(between zero and one, with one the default for all "
                  "unspecified halfedges)\n"
                  "and the halfedge index (3 * triangle index + [0,1,2] where "
                  "0 is the edge\n"
                  "between triVert 0 and 1, etc).\n"
                  "\n"
                  "At a smoothness value of zero, a sharp crease is made. The "
                  "smoothness is\n"
                  "interpolated along each edge, so the specified value should "
                  "be thought of as\n"
                  "an average. Where exactly two sharpened edges meet at a "
                  "vertex, their\n"
                  "tangents are rotated to be colinear so that the sharpened "
                  "edge can be\n"
                  "continuous. Vertices with only one sharpened edge are "
                  "completely smooth,\n"
                  "allowing sharpened edges to smoothly vanish at termination. "
                  "A single vertex\n"
                  "can be sharpened by sharping all edges that are incident on "
                  "it, allowing\n"
                  "cones to be formed.")
      .def_static(
          "from_mesh", [](const Mesh &mesh) { return Manifold(mesh); },
          py::arg("mesh"))
      .def_static(
          "tetrahedron", []() { return Manifold::Tetrahedron(); },
          "Constructs a tetrahedron centered at the origin with one vertex at "
          "(1,1,1)\n"
          "and the rest at similarly symmetric points.")
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
          "first\n"
          "octant, touching the origin.\n"
          "\n"
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
          "first\n"
          "octant, touching the origin.\n"
          "\n"
          ":param size: The X, Y, and Z dimensions of the box.\n"
          ":param center: Set to true to shift the center to the origin.")
      .def_static(
          "cube",
          [](float x, float y, float z, bool center = false) {
            return Manifold::Cube(glm::vec3(x, y, z), center);
          },
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("center") = false,
          "Constructs a unit cube (edge lengths all one), by default in the "
          "first\n"
          "octant, touching the origin.\n"
          "\n"
          ":param x: The X dimensions of the box.\n"
          ":param y: The Y dimensions of the box.\n"
          ":param z: The Z dimensions of the box.\n"
          ":param center: Set to true to shift the center to the origin.")
      .def_static(
          "cylinder",
          [](float height, float radiusLow, float radiusHigh = -1.0f,
             int circularSegments = 0) {
            return Manifold::Cylinder(height, radiusLow, radiusHigh,
                                      circularSegments);
          },
          py::arg("height"), py::arg("radius_low"),
          py::arg("radius_high") = -1.0f, py::arg("circular_segments") = 0,
          "A convenience constructor for the common case of extruding a "
          "circle. Can also\n"
          "form cones if both radii are specified.\n"
          "\n"
          ":param height: Z-extent\n"
          ":param radiusLow: Radius of bottom circle. Must be positive.\n"
          ":param radiusHigh: Radius of top circle. Can equal zero. Default "
          "(-1) is equal to\n"
          "radiusLow.\n"
          ":param circularSegments: How many line segments to use around the "
          "circle.\n"
          "Default (-1) is calculated by the static Defaults.\n"
          ":param center: Set to true to shift the center to the origin. "
          "Default is\n"
          "origin at the bottom.")
      .def_static(
          "sphere",
          [](float radius, int circularSegments = 0) {
            return Manifold::Sphere(radius, circularSegments);
          },
          py::arg("radius"), py::arg("circular_segments") = 0,
          "Constructs a geodesic sphere of a given radius.\n"
          "\n"
          ":param radius: Radius of the sphere. Must be positive.\n"
          ":param circularSegments: Number of segments along its\n"
          "diameter. This number will always be rounded up to the nearest "
          "factor of\n"
          "four, as this sphere is constructed by refining an octahedron. This "
          "means\n"
          "there are a circle of vertices on all three of the axis planes. "
          "Default is\n"
          "calculated by the static Defaults.");

  py::class_<PolygonsWrapper>(m, "Polygons")
      .def(py::init([](std::vector<std::vector<Float2>> &polygons) {
             std::vector<SimplePolygon> simplePolygons(polygons.size());
             for (int i = 0; i < polygons.size(); i++) {
               std::vector<PolyVert> vertices(polygons[i].size());
               for (int j = 0; j < polygons[i].size(); j++) {
                 vertices[j] = {
                     {std::get<0>(polygons[i][j]), std::get<1>(polygons[i][j])},
                     j};
               }
               simplePolygons[i] = {vertices};
             }
             return PolygonsWrapper{std::make_unique<Polygons>(simplePolygons)};
           }),
           py::arg("polygons"),
           "Construct the Polygons object from a list of simple polygon, "
           "where each simple polygon is a list of points (pair of floats).")
      .def(
          "extrude",
          [](PolygonsWrapper &self, float height, int nDivisions = 0,
             float twistDegrees = 0.0f,
             Float2 scaleTop = std::make_tuple(1.0f, 1.0f)) {
            glm::vec2 scaleTopVec(std::get<0>(scaleTop), std::get<1>(scaleTop));
            return Manifold::Extrude(*self.polygons, height, nDivisions,
                                     twistDegrees, scaleTopVec);
          },
          py::arg("height"), py::arg("n_divisions") = 0,
          py::arg("twist_degrees") = 0.0f,
          py::arg("scale_top") = std::make_tuple(1.0f, 1.0f),
          "Constructs a manifold from the set of polygons by extruding them "
          "along the\n"
          "Z-axis.\n"
          "\n"
          ":param height: Z-extent of extrusion.\n"
          ":param nDivisions: Number of extra copies of the crossSection to "
          "insert into\n"
          "the shape vertically; especially useful in combination with "
          "twistDegrees to\n"
          "avoid interpolation artifacts. Default is none.\n"
          ":param twistDegrees: Amount to twist the top crossSection relative "
          "to the\n"
          "bottom, interpolated linearly for the divisions in between.\n"
          ":param scaleTop: Amount to scale the top (independently in X and "
          "Y). If the\n"
          "scale is (0, 0), a pure cone is formed with only a single vertex at "
          "the top.\n"
          "Default (1, 1).")
      .def(
          "revolve",
          [](PolygonsWrapper &self, int circularSegments = 0) {
            return Manifold::Revolve(*self.polygons, circularSegments);
          },
          py::arg("circular_segments") = 0,
          "Constructs a manifold from the set of polygons by revolving this "
          "cross-section\n"
          "around its Y-axis and then setting this as the Z-axis of the "
          "resulting\n"
          "manifold. If the polygons cross the Y-axis, only the part on the "
          "positive X\n"
          "side is used. Geometrically valid input will result in "
          "geometrically valid\n"
          "output.\n"
          "\n"
          ":param circularSegments: Number of segments along its diameter. "
          "Default is\n"
          "calculated by the static Defaults.");

  py::class_<Mesh>(m, "Mesh")
      .def(py::init([](py::array_t<float> &vertPos, py::array_t<int> &triVerts,
                       py::array_t<float> &vertNormal,
                       py::array_t<float> &halfedgeTangent) {
             auto vertPos_view = vertPos.unchecked<2>();
             auto triVerts_view = triVerts.unchecked<2>();
             auto vertNormal_view = vertNormal.unchecked<2>();
             auto halfedgeTangent_view = halfedgeTangent.unchecked<2>();
             if (vertPos_view.shape(1) != 3)
               throw std::runtime_error("Invalid vert_pos shape");
             if (triVerts_view.shape(1) != 3)
               throw std::runtime_error("Invalid tri_verts shape");
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
             std::vector<glm::vec3> vertPos_vec(vertPos_view.shape(0));
             std::vector<glm::ivec3> triVerts_vec(triVerts_view.shape(0));
             std::vector<glm::vec3> vertNormal_vec(vertNormal_view.shape(0));
             std::vector<glm::vec4> halfedgeTangent_vec(
                 halfedgeTangent_view.shape(0));
             for (int i = 0; i < vertPos_view.shape(0); i++)
               for (const int j : {0, 1, 2})
                 vertPos_vec[i][j] = vertPos_view(i, j);
             for (int i = 0; i < triVerts_view.shape(0); i++)
               for (const int j : {0, 1, 2})
                 triVerts_vec[i][j] = triVerts_view(i, j);
             for (int i = 0; i < vertNormal_view.shape(0); i++)
               for (const int j : {0, 1, 2})
                 vertNormal_vec[i][j] = vertNormal_view(i, j);
             for (int i = 0; i < halfedgeTangent_view.shape(0); i++)
               for (const int j : {0, 1, 2, 3})
                 halfedgeTangent_vec[i][j] = halfedgeTangent_view(i, j);
             return Mesh({vertPos_vec, triVerts_vec, vertNormal_vec,
                          halfedgeTangent_vec});
           }),
           py::arg("vert_pos"), py::arg("tri_verts"), py::arg("vert_normal"),
           py::arg("halfedge_tangent"))
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
}
