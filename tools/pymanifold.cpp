#include "manifold.h"
#include "meshIO.h"
#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

using namespace manifold;

void exportManifold(Manifold &m, std::string name) {
  Mesh out = m.GetMesh();
  manifold::ExportOptions options;
  options.faceted = true;
  options.mat.roughness = 0.2;
  options.mat.metalness = 0.0;
  manifold::ExportMesh(name, out, options);
};

typedef std::tuple<float, float> Float2;
typedef std::tuple<float, float, float> Float3;

struct PolygonsWrapper {
  std::unique_ptr<Polygons> polygons;
};

PYBIND11_MODULE(pymanifold, m) {
  m.doc() = "Python binding for the manifold library. Please check the C++ documentation for APIs.\n"
      "This binding will perform copying to make the API more familiar to OpenSCAD users.";
  py::class_<Manifold>(m, "Manifold")
      .def(py::init<>())
      .def(py::init([](std::vector<Manifold> &manifolds) {
        Manifold result;
        for (Manifold &manifold : manifolds)
          result += manifold;
        return result;
      }), "Construct manifold as the union of a set of manifolds.")
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def("__and__", [](Manifold &a, Manifold &b) { return a ^ b; })
      .def("transform",
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
           })
      .def(
          "translate",
          [](Manifold self, float x, float y, float z = 0.0f) {
            return self.Translate(glm::vec3(x, y, z));
          },
          py::arg("x"), py::arg("y"), py::arg("z") = 0.0f)
      .def("scale", static_cast<Manifold (*)(Manifold, float)>(
                        [](Manifold self, float scale) {
                          return self.Scale(glm::vec3(scale));
                        }), py::arg("scale"))
      .def("scale", static_cast<Manifold (*)(Manifold, py::array_t<float> &)>(
                        [](Manifold self, py::array_t<float> &scale) {
                          auto scale_view = scale.unchecked<1>();
                          if (scale_view.shape(0) != 3)
                            throw std::runtime_error("Invalid vector shape");
                          glm::vec3 v(scale_view(0), scale_view(1),
                                      scale_view(2));
                          return self.Scale(v);
                        }), py::arg("v"))
      .def("rotate",
           [](Manifold self, float xDegrees, float yDegrees, float zDegrees) {
             return self.Rotate(xDegrees, yDegrees, zDegrees);
           }, py::arg("xDegrees"), py::arg("yDegrees"), py::arg("zDegrees"))
      .def("export",
           [](Manifold &self, std::string name) { exportManifold(self, name); },
        py::arg("filename"),
           "Export the manifold object to file, where the file type is determined from file extension.")
      .def("warp",
           [](Manifold self, const std::function<Float3(Float3)> &f) {
             return self.Warp([&f](glm::vec3 &v) {
               Float3 fv = f(std::make_tuple(v.x, v.y, v.z));
               v.x = std::get<0>(fv);
               v.y = std::get<1>(fv);
               v.z = std::get<2>(fv);
             });
           }, py::arg("f"))
      .def("refine", [](Manifold self, int n) { return self.Refine(n); }, py::arg("n"))
      .def_static(
          "FromMesh",
          [](py::array_t<float> &vertPos, py::array_t<int> &triVerts) {
            auto vertPos_view = vertPos.unchecked<2>();
            auto triVerts_view = triVerts.unchecked<2>();
            if (vertPos_view.shape(1) != 3)
              throw std::runtime_error("Invalid vertex position shape");
            if (triVerts_view.shape(1) != 3)
              throw std::runtime_error("Invalid triangle vertices shape");
            std::vector<glm::vec3> vertPos_vec(vertPos_view.shape(0));
            std::vector<glm::ivec3> triVerts_vec(triVerts_view.shape(0));
            for (int i = 0; i < vertPos_view.shape(0); i++)
              vertPos_vec[i] = {vertPos_view(i, 0), vertPos_view(i, 1),
                                vertPos_view(i, 2)};
            for (int i = 0; i < triVerts_view.shape(0); i++)
              triVerts_vec[i] = {triVerts_view(i, 0), triVerts_view(i, 1),
                                 triVerts_view(i, 2)};
            return Manifold({vertPos_vec, triVerts_vec});
          }, py::arg("vertPos"), py::arg("triVerts"))
      .def_static("Tetrahedron", []() { return Manifold::Tetrahedron(); })
      .def_static(
          "Cube",
          [](float x, float y, float z,
             bool center =
                 false) { return Manifold::Cube(glm::vec3(x, y, z), center); },
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("center") = false)
      .def_static(
          "Cylinder",
          [](float height, float radiusLow, float radiusHigh = -1.0f,
             int circularSegments = 0) {
            return Manifold::Cylinder(height, radiusLow, radiusHigh,
                                      circularSegments);
          },
          py::arg("height"), py::arg("radiusLow"),
          py::arg("radiusHigh") = -1.0f, py::arg("circularSegments") = 0)
      .def_static(
          "Sphere",
          [](float radius,
             int circularSegments =
                 0) { return Manifold::Sphere(radius, circularSegments); },
          py::arg("radius"), py::arg("circularSegments") = 0);
  ;

  py::class_<PolygonsWrapper>(m, "Polygons")
      .def(py::init([](std::vector<std::vector<Float2>> &polygons) {
        std::vector<SimplePolygon> simplePolygons(polygons.size());
        for (int i = 0; i < polygons.size(); i++) {
          std::vector<PolyVert> vertices(polygons[i].size());
          for (int j = 0; j < polygons[i].size(); j++) {
            vertices[j] = {
                {std::get<0>(polygons[i][j]), std::get<1>(polygons[i][j])}, j};
          }
          simplePolygons[i] = {vertices};
        }
        return PolygonsWrapper{std::make_unique<Polygons>(simplePolygons)};
      }), py::arg("polygons"),
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
          py::arg("height"), py::arg("nDivisions") = 0,
          py::arg("twistDegrees") = 0.0f,
          py::arg("scaleTop") = std::make_tuple(1.0f, 1.0f))
      .def(
          "revolve",
          [](PolygonsWrapper &self, int circularSegments = 0) {
            return Manifold::Revolve(*self.polygons, circularSegments);
          },
          py::arg("circularSegments") = 0);
}
