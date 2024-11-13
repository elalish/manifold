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

#include <optional>
#include <string>

#include "autogen_docstrings.inl"  // generated in build folder
#include "manifold/cross_section.h"
#include "manifold/manifold.h"
#include "manifold/polygon.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/operators.h"
#include "nanobind/stl/function.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/vector.h"

namespace nb = nanobind;
using namespace manifold;

template <class T>
struct la_name {};
template <>
struct la_name<vec3> {
  static constexpr char const name[] = "Doublex3";
  static constexpr char const multi_name[] = "DoubleNx3";
};
template <>
struct la_name<vec2> {
  static constexpr char const name[] = "Doublex2";
  static constexpr char const multi_name[] = "DoubleNx2";
};
template <>
struct la_name<ivec3> {
  static constexpr char const name[] = "Intx3";
  static constexpr char const multi_name[] = "IntNx3";
};
template <>
struct la_name<mat3x4> {
  static constexpr char const name[] = "Double3x4";
};
template <>
struct la_name<mat2x3> {
  static constexpr char const name[] = "Double2x3";
};

// handle la::vecN
template <class T, int N>
struct nb::detail::type_caster<la::vec<T, N>> {
  using la_type = la::vec<T, N>;
  NB_TYPE_CASTER(la_type, const_name(la_name<la_type>::name));

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    int size = PyObject_Size(src.ptr());  // negative on failure
    if (size != N) return false;
    make_caster<T> t_cast;
    for (int i = 0; i < size; i++) {
      if (!t_cast.from_python(src[i], flags, cleanup)) return false;
      value[i] = t_cast.value;
    }
    return true;
  }
  static handle from_cpp(la_type vec, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    nb::list out;
    for (int i = 0; i < N; i++) out.append(vec[i]);
    return out.release();
  }
};

// handle la::matMxN
template <class T, int C, int R>
struct nb::detail::type_caster<la::mat<T, R, C>> {
  using la_type = la::mat<T, R, C>;
  using numpy_type = nb::ndarray<nb::numpy, T, nb::shape<R, C>>;
  NB_TYPE_CASTER(la_type, const_name(la_name<la_type>::name));

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    int rows = PyObject_Size(src.ptr());  // negative on failure
    if (rows != R) return false;
    for (int i = 0; i < R; i++) {
      const nb::object &slice = src[i];
      int cols = PyObject_Size(slice.ptr());  // negative on failure
      if (cols != C) return false;
      for (int j = 0; j < C; j++) {
        make_caster<T> t_cast;
        if (!t_cast.from_python(slice[j], flags, cleanup)) return false;
        value[j][i] = t_cast.value;
      }
    }
    return true;
  }
  static handle from_cpp(la_type mat, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    std::array<T, R * C> buffer;
    for (int i = 0; i < R; i++) {
      for (int j = 0; j < C; j++) {
        // py is (Rows, Cols), la is (Cols, Rows)
        buffer[i * C + j] = mat[j][i];
      }
    }
    numpy_type arr{buffer, {R, C}, nb::handle()};
    // we must copy the underlying data
    return make_caster<numpy_type>::from_cpp(arr, rv_policy::copy, cleanup);
  }
};

// handle std::vector<la::vecN>
template <class T, int N>
struct nb::detail::type_caster<std::vector<la::vec<T, N>>> {
  using la_type = la::vec<T, N>;
  using numpy_type = nb::ndarray<nb::numpy, T, nb::shape<-1, N>>;
  NB_TYPE_CASTER(std::vector<la_type>,
                 const_name(la_name<la_type>::multi_name));

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    make_caster<numpy_type> arr_cast;
    if (arr_cast.from_python(src, flags, cleanup)) {
      size_t num_vec = arr_cast.value.shape(0);
      value.resize(num_vec);
      for (size_t i = 0; i < num_vec; i++) {
        for (int j = 0; j < N; j++) {
          value[i][j] = arr_cast.value(i, j);
        }
      }
    } else {
      size_t num_vec = PyObject_Size(src.ptr());  // negative on failure
      if (num_vec == static_cast<size_t>(-1)) return false;
      value.resize(num_vec);
      for (size_t i = 0; i < num_vec; i++) {
        make_caster<la_type> vec_cast;
        if (!vec_cast.from_python(src[i], flags, cleanup)) return false;
        value[i] = vec_cast.value;
      }
    }
    return true;
  }
  static handle from_cpp(Value vec, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    size_t num_vec = vec.size();
    T *buffer = new T[num_vec * N];
    nb::capsule mem_mgr(buffer, [](void *p) noexcept { delete[] (T *)p; });
    for (size_t i = 0; i < num_vec; i++) {
      for (int j = 0; j < N; j++) {
        buffer[i * N + j] = vec[i][j];
      }
    }
    numpy_type arr{buffer, {num_vec, N}, std::move(mem_mgr)};
    // we can just do a move because we already did the copying
    return make_caster<numpy_type>::from_cpp(arr, rv_policy::move, cleanup);
  }
};

// handle VecView<la::vecN>
template <class T, int N>
struct nb::detail::type_caster<manifold::VecView<la::vec<T, N>>> {
  using la_type = la::vec<T, N>;
  using numpy_type = nb::ndarray<nb::numpy, T, nb::shape<-1, N>>;
  NB_TYPE_CASTER(manifold::VecView<la_type>,
                 const_name(la_name<la_type>::multi_name));

  static handle from_cpp(Value vec, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    size_t num_vec = vec.size();
    // assume packed struct
    static_assert(alignof(la::vec<T, N>) <= (N * sizeof(T)),
                  "VecView -> numpy requires packed structs");
    static_assert(sizeof(la::vec<T, N>) == (N * sizeof(T)),
                  "VecView -> numpy requires packed structs");
    numpy_type arr{vec.data(), {num_vec, N}, nb::handle()};
    // we must copy the underlying data
    return make_caster<numpy_type>::from_cpp(arr, rv_policy::copy, cleanup);
  }
};

template <typename T>
std::vector<T> toVector(const T *arr, size_t size) {
  return std::vector<T>(arr, arr + size);
}

using namespace manifold_docstrings;

// strip original :params: and replace with ours
const std::string manifold__rotate_xyz =
    manifold__rotate__x_degrees__y_degrees__z_degrees;
const std::string manifold__rotate__v =
    manifold__rotate_xyz.substr(0, manifold__rotate_xyz.find(":param")) +
    ":param v: [X, Y, Z] rotation in degrees.";

NB_MODULE(manifold3d, m) {
  m.doc() = "Python binding for the Manifold library.";

  m.def("set_min_circular_angle", Quality::SetMinCircularAngle,
        nb::arg("angle"), set_min_circular_angle__angle);

  m.def("set_min_circular_edge_length", Quality::SetMinCircularEdgeLength,
        nb::arg("length"), set_min_circular_edge_length__length);

  m.def("set_circular_segments", Quality::SetCircularSegments,
        nb::arg("number"), set_circular_segments__number);

  m.def("get_circular_segments", Quality::GetCircularSegments,
        nb::arg("radius"), get_circular_segments__radius);

  m.def("triangulate", &Triangulate, nb::arg("polygons"),
        nb::arg("epsilon") = -1, triangulate__polygons__epsilon);

  nb::class_<Manifold>(m, "Manifold")
      .def(nb::init<>(), manifold__manifold)
      .def(nb::init<const MeshGL &>(), nb::arg("mesh"),
           manifold__manifold__mesh_gl)
      .def(nb::init<const MeshGL64 &>(), nb::arg("mesh"),
           manifold__manifold__mesh_gl64)
      .def(nb::self + nb::self, manifold__operator_plus__q)
      .def(nb::self - nb::self, manifold__operator_minus__q)
      .def(nb::self ^ nb::self, manifold__operator_xor__q)
      .def(
          "hull", [](const Manifold &self) { return self.Hull(); },
          manifold__hull)
      .def_static(
          "batch_hull",
          [](std::vector<Manifold> ms) { return Manifold::Hull(ms); },
          nb::arg("manifolds"), manifold__hull__manifolds)
      .def_static(
          "hull_points",
          [](std::vector<vec3> pts) { return Manifold::Hull(pts); },
          nb::arg("pts"), manifold__hull__pts)
      .def("transform", &Manifold::Transform, nb::arg("m"),
           manifold__transform__m)
      .def("translate", &Manifold::Translate, nb::arg("t"),
           manifold__translate__v)
      .def("scale", &Manifold::Scale, nb::arg("v"), manifold__scale__v)
      .def(
          "scale",
          [](const Manifold &self, double s) { self.Scale({s, s, s}); },
          nb::arg("s"),
          "Scale this Manifold in space. This operation can be chained. "
          "Transforms are combined and applied lazily.\n\n"
          ":param s: The scalar to multiply every vertex by component.")
      .def("mirror", &Manifold::Mirror, nb::arg("v"), manifold__mirror__normal)
      .def(
          "rotate",
          [](const Manifold &self, vec3 v) {
            return self.Rotate(v.x, v.y, v.z);
          },
          nb::arg("v"), manifold__rotate__v.c_str())
      .def(
          "warp",
          [](const Manifold &self, std::function<vec3(vec3)> warp_func) {
            // need a wrapper because python cant modify a reference in-place
            return self.Warp([&warp_func](vec3 &v) { v = warp_func(v); });
          },
          nb::arg("warp_func"), manifold__warp__warp_func)
      .def(
          "warp_batch",
          [](const Manifold &self,
             std::function<nb::object(VecView<vec3>)> warp_func) {
            // need a wrapper because python cant modify a reference in-place
            return self.WarpBatch([&warp_func](VecView<vec3> v) {
              auto tmp = warp_func(v);
              nb::ndarray<double, nb::shape<-1, 3>, nanobind::c_contig> tmpnd;
              if (!nb::try_cast(tmp, tmpnd) || tmpnd.ndim() != 2)
                throw std::runtime_error(
                    "Invalid vector shape, expected (:, 3)");
              std::copy(tmpnd.data(), tmpnd.data() + v.size() * 3,
                        &v.data()->x);
            });
          },
          nb::arg("warp_func"), manifold__warp_batch__warp_func)
      .def(
          "set_properties",
          [](const Manifold &self, int newNumProp,
             const std::function<nb::object(
                 vec3, const nb::ndarray<nb::numpy, const double, nb::c_contig>
                           &)> &f) {
            const int oldNumProp = self.NumProp();
            return self.SetProperties(newNumProp, [newNumProp, oldNumProp, &f](
                                                      double *newProps, vec3 v,
                                                      const double *oldProps) {
              auto result =
                  f(v, nb::ndarray<nb::numpy, const double, nb::c_contig>(
                           oldProps, {static_cast<unsigned long>(oldNumProp)},
                           nb::handle()));
              nb::ndarray<double, nb::shape<-1>> array;
              std::vector<double> vec;
              if (nb::try_cast(result, array)) {
                if (array.ndim() != 1 ||
                    array.shape(0) != static_cast<size_t>(newNumProp))
                  throw std::runtime_error("Invalid vector shape, expected (" +
                                           std::to_string(newNumProp) + ")");
                for (int i = 0; i < newNumProp; i++) newProps[i] = array(i);
              } else if (nb::try_cast(result, vec)) {
                for (int i = 0; i < newNumProp; i++) newProps[i] = vec[i];
              } else {
                throw std::runtime_error(
                    "Callback in set_properties should return an array");
              }
            });
          },
          nb::arg("new_num_prop"), nb::arg("f"),
          manifold__set_properties__num_prop__prop_func)
      .def("calculate_curvature", &Manifold::CalculateCurvature,
           nb::arg("gaussian_idx"), nb::arg("mean_idx"),
           manifold__calculate_curvature__gaussian_idx__mean_idx)
      .def("min_gap", &Manifold::MinGap, nb::arg("other"),
           nb::arg("search_length"),
           "Returns the minimum gap between two manifolds."
           "Returns a double between 0 and searchLength.")
      .def("calculate_normals", &Manifold::CalculateNormals,
           nb::arg("normal_idx"), nb::arg("min_sharp_angle") = 60,
           manifold__calculate_normals__normal_idx__min_sharp_angle)
      .def("smooth_by_normals", &Manifold::SmoothByNormals,
           nb::arg("normal_idx"), manifold__smooth_by_normals__normal_idx)
      .def("smooth_out", &Manifold::SmoothOut, nb::arg("min_sharp_angle") = 60,
           nb::arg("min_smoothness") = 0,
           manifold__smooth_out__min_sharp_angle__min_smoothness)
      .def("refine", &Manifold::Refine, nb::arg("n"), manifold__refine__n)
      .def("refine_to_length", &Manifold::RefineToLength, nb::arg("length"),
           manifold__refine_to_length__length)
      .def("refine_to_tolerance", &Manifold::RefineToTolerance,
           nb::arg("tolerance"), manifold__refine_to_tolerance__tolerance)
      .def("to_mesh", &Manifold::GetMeshGL, nb::arg("normal_idx") = -1,
           manifold__get_mesh_gl__normal_idx)
      .def("to_mesh64", &Manifold::GetMeshGL64, nb::arg("normal_idx") = -1,
           manifold__get_mesh_gl64__normal_idx)
      .def("num_vert", &Manifold::NumVert, manifold__num_vert)
      .def("num_edge", &Manifold::NumEdge, manifold__num_edge)
      .def("num_tri", &Manifold::NumTri, manifold__num_tri)
      .def("num_prop", &Manifold::NumProp, manifold__num_prop)
      .def("num_prop_vert", &Manifold::NumPropVert, manifold__num_prop_vert)
      .def("genus", &Manifold::Genus, manifold__genus)
      .def(
          "volume", [](const Manifold &self) { return self.Volume(); },
          "Get the volume of the manifold\n This is clamped to zero for a "
          "given face if they are within the Epsilon().")
      .def(
          "surface_area",
          [](const Manifold &self) { return self.SurfaceArea(); },
          "Get the surface area of the manifold\n This is clamped to zero for "
          "a given face if they are within the Epsilon().")
      .def("original_id", &Manifold::OriginalID, manifold__original_id)
      .def("get_tolerance", &Manifold::GetTolerance, manifold__get_tolerance)
      .def("set_tolerance", &Manifold::SetTolerance,
           manifold__set_tolerance__tolerance)
      .def("as_original", &Manifold::AsOriginal, manifold__as_original)
      .def("is_empty", &Manifold::IsEmpty, manifold__is_empty)
      .def("decompose", &Manifold::Decompose, manifold__decompose)
      .def("split", &Manifold::Split, nb::arg("cutter"),
           manifold__split__cutter)
      .def("split_by_plane", &Manifold::SplitByPlane, nb::arg("normal"),
           nb::arg("origin_offset"),
           manifold__split_by_plane__normal__origin_offset)
      .def("trim_by_plane", &Manifold::TrimByPlane, nb::arg("normal"),
           nb::arg("origin_offset"),
           manifold__trim_by_plane__normal__origin_offset)
      .def(
          "slice",
          [](const Manifold &self, double height) {
            return CrossSection(self.Slice(height));
          },
          nb::arg("height"), manifold__slice__height)
      .def(
          "project",
          [](const Manifold &self) {
            return CrossSection(self.Project()).Simplify(self.GetEpsilon());
          },
          manifold__project)
      .def("status", &Manifold::Status, manifold__status)
      .def(
          "bounding_box",
          [](const Manifold &self) {
            Box b = self.BoundingBox();
            return nb::make_tuple(b.min[0], b.min[1], b.min[2], b.max[0],
                                  b.max[1], b.max[2]);
          },
          "Gets the manifold bounding box as a tuple "
          "(xmin, ymin, zmin, xmax, ymax, zmax).")
      .def_static(
          "smooth",
          [](const MeshGL &mesh, std::vector<size_t> sharpened_edges,
             std::vector<double> edge_smoothness) {
            if (sharpened_edges.size() != edge_smoothness.size()) {
              throw std::runtime_error(
                  "sharpened_edges.size() != edge_smoothness.size()");
            }
            std::vector<Smoothness> vec(sharpened_edges.size());
            for (size_t i = 0; i < vec.size(); i++) {
              vec[i] = {sharpened_edges[i], edge_smoothness[i]};
            }
            return Manifold::Smooth(mesh, vec);
          },
          nb::arg("mesh"), nb::arg("sharpened_edges") = nb::list(),
          nb::arg("edge_smoothness") = nb::list(),
          manifold__smooth__mesh_gl__sharpened_edges)
      .def_static(
          "smooth",
          [](const MeshGL64 &mesh, std::vector<size_t> sharpened_edges,
             std::vector<double> edge_smoothness) {
            if (sharpened_edges.size() != edge_smoothness.size()) {
              throw std::runtime_error(
                  "sharpened_edges.size() != edge_smoothness.size()");
            }
            std::vector<Smoothness> vec(sharpened_edges.size());
            for (size_t i = 0; i < vec.size(); i++) {
              vec[i] = {sharpened_edges[i], edge_smoothness[i]};
            }
            return Manifold::Smooth(mesh, vec);
          },
          nb::arg("mesh"), nb::arg("sharpened_edges") = nb::list(),
          nb::arg("edge_smoothness") = nb::list(),
          // note: this is not a typo, the documentation is essentially the same
          // so we just use the 32 byte variant to avoid duplicating docstring
          // override...
          manifold__smooth__mesh_gl__sharpened_edges)
      .def_static("batch_boolean", &Manifold::BatchBoolean,
                  nb::arg("manifolds"), nb::arg("op"),
                  manifold__batch_boolean__manifolds__op)
      .def_static("compose", &Manifold::Compose, nb::arg("manifolds"),
                  manifold__compose__manifolds)
      .def_static("tetrahedron", &Manifold::Tetrahedron, manifold__tetrahedron)
      .def_static("cube", &Manifold::Cube,
                  nb::arg("size") = std::make_tuple(1.0, 1.0, 1.0),
                  nb::arg("center") = false, manifold__cube__size__center)
      .def_static(
          "extrude",
          [](const CrossSection &crossSection, double height, int nDivisions,
             double twistDegrees, vec2 scaleTop) {
            return Manifold::Extrude(crossSection.ToPolygons(), height,
                                     nDivisions, twistDegrees, scaleTop);
          },
          nb::arg("crossSection"), nb::arg("height"),
          nb::arg("n_divisions") = 0, nb::arg("twist_degrees") = 0.0f,
          nb::arg("scale_top") = std::make_tuple(1.0f, 1.0f),
          manifold__extrude__cross_section__height__n_divisions__twist_degrees__scale_top)
      .def_static(
          "revolve",
          [](const CrossSection &crossSection, int circularSegments,
             double revolveDegrees) {
            return Manifold::Revolve(crossSection.ToPolygons(),
                                     circularSegments, revolveDegrees);
          },
          nb::arg("crossSection"), nb::arg("circular_segments") = 0,
          nb::arg("revolve_degrees") = 360.0,
          manifold__revolve__cross_section__circular_segments__revolve_degrees)
      .def_static(
          "level_set",
          [](const std::function<double(double, double, double)> &f,
             std::vector<double> bounds, double edgeLength, double level = 0.0,
             double tolerance = -1) {
            // Same format as Manifold.bounding_box
            Box bound = {vec3(bounds[0], bounds[1], bounds[2]),
                         vec3(bounds[3], bounds[4], bounds[5])};

            std::function<double(vec3)> cppToPython = [&f](vec3 v) {
              return f(v.x, v.y, v.z);
            };
            return Manifold::LevelSet(cppToPython, bound, edgeLength, level,
                                      tolerance, false);
          },
          nb::arg("f"), nb::arg("bounds"), nb::arg("edgeLength"),
          nb::arg("level") = 0.0, nb::arg("tolerance") = -1,
          manifold__level_set__sdf__bounds__edge_length__level__tolerance__can_parallel)
      .def_static(
          "cylinder", &Manifold::Cylinder, nb::arg("height"),
          nb::arg("radius_low"), nb::arg("radius_high") = -1.0f,
          nb::arg("circular_segments") = 0, nb::arg("center") = false,
          manifold__cylinder__height__radius_low__radius_high__circular_segments__center)
      .def_static("sphere", &Manifold::Sphere, nb::arg("radius"),
                  nb::arg("circular_segments") = 0,
                  manifold__sphere__radius__circular_segments)
      .def_static("reserve_ids", Manifold::ReserveIDs, nb::arg("n"),
                  manifold__reserve_ids__n);

  nb::class_<MeshGL>(m, "Mesh")
      .def(
          // note that reshape requires mutable ndarray, but this will not
          // affect the original array passed into the function
          "__init__",
          [](MeshGL *self,
             nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> &vertProp,
             nb::ndarray<uint32_t, nb::shape<-1, 3>, nb::c_contig> &triVerts,
             const std::optional<nb::ndarray<uint32_t, nb::shape<-1>,
                                             nb::c_contig>> &mergeFromVert,
             const std::optional<nb::ndarray<uint32_t, nb::shape<-1>,
                                             nb::c_contig>> &mergeToVert,
             const std::optional<
                 nb::ndarray<uint32_t, nb::shape<-1>, nb::c_contig>> &runIndex,
             const std::optional<nb::ndarray<uint32_t, nb::shape<-1>,
                                             nb::c_contig>> &runOriginalID,
             std::optional<nb::ndarray<float, nb::shape<-1, 4, 3>,
                                       nb::c_contig>> &runTransform,
             const std::optional<
                 nb::ndarray<uint32_t, nb::shape<-1>, nb::c_contig>> &faceID,
             const std::optional<nb::ndarray<float, nb::shape<-1, 3, 4>,
                                             nb::c_contig>> &halfedgeTangent,
             float tolerance) {
            new (self) MeshGL();
            MeshGL &out = *self;
            out.numProp = vertProp.shape(1);
            out.vertProperties = toVector(vertProp.data(), vertProp.size());

            out.triVerts = toVector(triVerts.data(), triVerts.size());

            if (mergeFromVert.has_value())
              out.mergeFromVert =
                  toVector(mergeFromVert->data(), mergeFromVert->size());

            if (mergeToVert.has_value())
              out.mergeToVert =
                  toVector(mergeToVert->data(), mergeToVert->size());

            if (runIndex.has_value())
              out.runIndex = toVector(runIndex->data(), runIndex->size());

            if (runOriginalID.has_value())
              out.runOriginalID =
                  toVector(runOriginalID->data(), runOriginalID->size());

            if (runTransform.has_value()) {
              out.runTransform =
                  toVector(runTransform->data(), runTransform->size());
            }

            if (faceID.has_value())
              out.faceID = toVector(faceID->data(), faceID->size());

            if (halfedgeTangent.has_value()) {
              out.halfedgeTangent =
                  toVector(halfedgeTangent->data(), halfedgeTangent->size());
            }
          },
          nb::arg("vert_properties"), nb::arg("tri_verts"),
          nb::arg("merge_from_vert") = nb::none(),
          nb::arg("merge_to_vert") = nb::none(),
          nb::arg("run_index") = nb::none(),
          nb::arg("run_original_id") = nb::none(),
          nb::arg("run_transform") = nb::none(),
          nb::arg("face_id") = nb::none(),
          nb::arg("halfedge_tangent") = nb::none(), nb::arg("tolerance") = 0)
      .def_prop_ro(
          "vert_properties",
          [](const MeshGL &self) {
            return nb::ndarray<nb::numpy, const float, nb::c_contig>(
                self.vertProperties.data(),
                {self.vertProperties.size() / self.numProp, self.numProp},
                nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "tri_verts",
          [](const MeshGL &self) {
            return nb::ndarray<nb::numpy, const int, nb::c_contig>(
                self.triVerts.data(), {self.triVerts.size() / 3, 3},
                nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "run_transform",
          [](const MeshGL &self) {
            return nb::ndarray<nb::numpy, const float, nb::c_contig>(
                self.runTransform.data(), {self.runTransform.size() / 12, 4, 3},
                nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "halfedge_tangent",
          [](const MeshGL &self) {
            return nb::ndarray<nb::numpy, const float, nb::c_contig>(
                self.halfedgeTangent.data(),
                {self.halfedgeTangent.size() / 12, 3, 4}, nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_ro("merge_from_vert", &MeshGL::mergeFromVert)
      .def_ro("merge_to_vert", &MeshGL::mergeToVert)
      .def_ro("run_index", &MeshGL::runIndex)
      .def_ro("run_original_id", &MeshGL::runOriginalID)
      .def_ro("face_id", &MeshGL::faceID)
      .def("merge", &MeshGL::Merge, mesh_gl__merge);

  nb::class_<MeshGL64>(m, "Mesh64")
      .def(
          // note that reshape requires mutable ndarray, but this will not
          // affect the original array passed into the function
          "__init__",
          [](MeshGL64 *self,
             nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig> &vertProp,
             nb::ndarray<uint64_t, nb::shape<-1, 3>, nb::c_contig> &triVerts,
             const std::optional<nb::ndarray<uint64_t, nb::shape<-1>,
                                             nb::c_contig>> &mergeFromVert,
             const std::optional<nb::ndarray<uint64_t, nb::shape<-1>,
                                             nb::c_contig>> &mergeToVert,
             const std::optional<
                 nb::ndarray<uint64_t, nb::shape<-1>, nb::c_contig>> &runIndex,
             const std::optional<nb::ndarray<uint32_t, nb::shape<-1>,
                                             nb::c_contig>> &runOriginalID,
             std::optional<nb::ndarray<double, nb::shape<-1, 4, 3>,
                                       nb::c_contig>> &runTransform,
             const std::optional<
                 nb::ndarray<uint64_t, nb::shape<-1>, nb::c_contig>> &faceID,
             const std::optional<nb::ndarray<double, nb::shape<-1, 3, 4>,
                                             nb::c_contig>> &halfedgeTangent,
             float tolerance) {
            new (self) MeshGL64();
            MeshGL64 &out = *self;
            out.numProp = vertProp.shape(1);
            out.vertProperties = toVector(vertProp.data(), vertProp.size());

            out.triVerts = toVector(triVerts.data(), triVerts.size());

            if (mergeFromVert.has_value())
              out.mergeFromVert =
                  toVector(mergeFromVert->data(), mergeFromVert->size());

            if (mergeToVert.has_value())
              out.mergeToVert =
                  toVector(mergeToVert->data(), mergeToVert->size());

            if (runIndex.has_value())
              out.runIndex = toVector(runIndex->data(), runIndex->size());

            if (runOriginalID.has_value())
              out.runOriginalID =
                  toVector(runOriginalID->data(), runOriginalID->size());

            if (runTransform.has_value()) {
              out.runTransform =
                  toVector(runTransform->data(), runTransform->size());
            }

            if (faceID.has_value())
              out.faceID = toVector(faceID->data(), faceID->size());

            if (halfedgeTangent.has_value()) {
              out.halfedgeTangent =
                  toVector(halfedgeTangent->data(), halfedgeTangent->size());
            }
          },
          nb::arg("vert_properties"), nb::arg("tri_verts"),
          nb::arg("merge_from_vert") = nb::none(),
          nb::arg("merge_to_vert") = nb::none(),
          nb::arg("run_index") = nb::none(),
          nb::arg("run_original_id") = nb::none(),
          nb::arg("run_transform") = nb::none(),
          nb::arg("face_id") = nb::none(),
          nb::arg("halfedge_tangent") = nb::none(), nb::arg("tolerance") = 0)
      .def_prop_ro(
          "vert_properties",
          [](const MeshGL64 &self) {
            return nb::ndarray<nb::numpy, const double, nb::c_contig>(
                self.vertProperties.data(),
                {self.vertProperties.size() / self.numProp, self.numProp},
                nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "tri_verts",
          [](const MeshGL64 &self) {
            return nb::ndarray<nb::numpy, const uint64_t, nb::c_contig>(
                self.triVerts.data(), {self.triVerts.size() / 3, 3},
                nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "run_transform",
          [](const MeshGL64 &self) {
            return nb::ndarray<nb::numpy, const double, nb::c_contig>(
                self.runTransform.data(), {self.runTransform.size() / 12, 4, 3},
                nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "halfedge_tangent",
          [](const MeshGL64 &self) {
            return nb::ndarray<nb::numpy, const double, nb::c_contig>(
                self.halfedgeTangent.data(),
                {self.halfedgeTangent.size() / 12, 3, 4}, nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_ro("merge_from_vert", &MeshGL64::mergeFromVert)
      .def_ro("merge_to_vert", &MeshGL64::mergeToVert)
      .def_ro("run_index", &MeshGL64::runIndex)
      .def_ro("run_original_id", &MeshGL64::runOriginalID)
      .def_ro("face_id", &MeshGL64::faceID)
      .def("merge", &MeshGL64::Merge, mesh_gl__merge);

  nb::enum_<Manifold::Error>(m, "Error")
      .value("NoError", Manifold::Error::NoError)
      .value("NonFiniteVertex", Manifold::Error::NonFiniteVertex)
      .value("NotManifold", Manifold::Error::NotManifold)
      .value("VertexOutOfBounds", Manifold::Error::VertexOutOfBounds)
      .value("PropertiesWrongLength", Manifold::Error::PropertiesWrongLength)
      .value("MissingPositionProperties",
             Manifold::Error::MissingPositionProperties)
      .value("MergeVectorsDifferentLengths",
             Manifold::Error::MergeVectorsDifferentLengths)
      .value("MergeIndexOutOfBounds", Manifold::Error::MergeIndexOutOfBounds)
      .value("TransformWrongLength", Manifold::Error::TransformWrongLength)
      .value("RunIndexWrongLength", Manifold::Error::RunIndexWrongLength)
      .value("FaceIDWrongLength", Manifold::Error::FaceIDWrongLength)
      .value("InvalidConstruction", Manifold::Error::InvalidConstruction);

  nb::enum_<CrossSection::FillRule>(m, "FillRule")
      .value("EvenOdd", CrossSection::FillRule::EvenOdd,
             "Only odd numbered sub-regions are filled.")
      .value("NonZero", CrossSection::FillRule::NonZero,
             "Only non-zero sub-regions are filled.")
      .value("Positive", CrossSection::FillRule::Positive,
             "Only sub-regions with winding counts > 0 are filled.")
      .value("Negative", CrossSection::FillRule::Negative,
             "Only sub-regions with winding counts < 0 are filled.");

  nb::enum_<CrossSection::JoinType>(m, "JoinType")
      .value("Square", CrossSection::JoinType::Square,
             "Squaring is applied uniformly at all joins where the internal "
             "join angle is less that 90 degrees. The squared edge will be at "
             "exactly the offset distance from the join vertex.")
      .value(
          "Round", CrossSection::JoinType::Round,
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

  nb::enum_<OpType>(m, "OpType", "Operation types for batch_boolean")
      .value("Add", OpType::Add)
      .value("Subtract", OpType::Subtract)
      .value("Intersect", OpType::Intersect);

  nb::class_<CrossSection>(
      m, "CrossSection",
      "Two-dimensional cross sections guaranteed to be without "
      "self-intersections, or overlaps between polygons (from construction "
      "onwards). This class makes use of the "
      "[Clipper2](http://www.angusj.com/clipper2/Docs/Overview.htm) library "
      "for polygon clipping (boolean) and offsetting operations.")
      .def(nb::init<>(), cross_section__cross_section)
      .def(nb::init<std::vector<std::vector<vec2>>, CrossSection::FillRule>(),
           nb::arg("contours"),
           nb::arg("fillrule") = CrossSection::FillRule::Positive,
           cross_section__cross_section__contours__fillrule)
      .def("area", &CrossSection::Area, cross_section__area)
      .def("num_vert", &CrossSection::NumVert, cross_section__num_vert)
      .def("num_contour", &CrossSection::NumContour, cross_section__num_contour)
      .def("is_empty", &CrossSection::IsEmpty, cross_section__is_empty)
      .def(
          "bounds",
          [](const CrossSection &self) {
            Rect r = self.Bounds();
            return nb::make_tuple(r.min[0], r.min[1], r.max[0], r.max[1]);
          },
          "Return bounding box of CrossSection as tuple("
          "min_x, min_y, max_x, max_y)")
      .def("translate", &CrossSection::Translate, nb::arg("v"),
           cross_section__translate__v)
      .def("rotate", &CrossSection::Rotate, nb::arg("degrees"),
           cross_section__rotate__degrees)
      .def("scale", &CrossSection::Scale, nb::arg("scale"),
           cross_section__scale__scale)
      .def(
          "scale",
          [](const CrossSection &self, double s) { self.Scale({s, s}); },
          nb::arg("s"),
          "Scale this CrossSection in space. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param s: The scalar to multiply every vertex by per component.")
      .def("mirror", &CrossSection::Mirror, nb::arg("ax"),
           cross_section__mirror__ax)
      .def("transform", &CrossSection::Transform, nb::arg("m"),
           cross_section__transform__m)
      .def(
          "warp",
          [](const CrossSection &self, std::function<vec2(vec2)> warp_func) {
            // need a wrapper because python cant modify a reference in-place
            return self.Warp([&warp_func](vec2 &v) { v = warp_func(v); });
          },
          nb::arg("warp_func"), cross_section__warp__warp_func)
      .def("warp_batch", &CrossSection::WarpBatch, nb::arg("warp_func"),
           cross_section__warp_batch__warp_func)

      .def(
          "warp_batch",
          [](const CrossSection &self,
             std::function<nb::object(VecView<vec2>)> warp_func) {
            // need a wrapper because python cant modify a reference in-place
            return self.WarpBatch([&warp_func](VecView<vec2> v) {
              auto tmp = warp_func(v);
              nb::ndarray<double, nb::shape<-1, 2>, nanobind::c_contig> tmpnd;
              if (!nb::try_cast(tmp, tmpnd) || tmpnd.ndim() != 2)
                throw std::runtime_error(
                    "Invalid vector shape, expected (:, 2)");
              std::copy(tmpnd.data(), tmpnd.data() + v.size() * 2,
                        &v.data()->x);
            });
          },
          nb::arg("warp_func"), cross_section__warp_batch__warp_func)
      .def("simplify", &CrossSection::Simplify, nb::arg("epsilon") = 1e-6,
           cross_section__simplify__epsilon)
      .def(
          "offset", &CrossSection::Offset, nb::arg("delta"),
          nb::arg("join_type"), nb::arg("miter_limit") = 2.0,
          nb::arg("circular_segments") = 0,
          cross_section__offset__delta__jointype__miter_limit__circular_segments)
      .def(nb::self + nb::self, cross_section__operator_plus__q)
      .def(nb::self - nb::self, cross_section__operator_minus__q)
      .def(nb::self ^ nb::self, cross_section__operator_xor__q)
      .def(
          "hull", [](const CrossSection &self) { return self.Hull(); },
          cross_section__hull)
      .def_static(
          "batch_hull",
          [](std::vector<CrossSection> cs) { return CrossSection::Hull(cs); },
          nb::arg("cross_sections"), cross_section__hull__cross_sections)
      .def_static(
          "hull_points",
          [](std::vector<vec2> pts) { return CrossSection::Hull(pts); },
          nb::arg("pts"), cross_section__hull__pts)
      .def("decompose", &CrossSection::Decompose, cross_section__decompose)
      .def_static("batch_boolean", &CrossSection::BatchBoolean,
                  nb::arg("cross_sections"), nb::arg("op"),
                  cross_section__batch_boolean__cross_sections__op)
      .def_static("compose", &CrossSection::Compose, nb::arg("cross_sections"),
                  cross_section__compose__cross_sections)
      .def("to_polygons", &CrossSection::ToPolygons, cross_section__to_polygons)
      .def(
          "extrude",
          [](const CrossSection &self, double height, int nDivisions,
             double twistDegrees, vec2 scaleTop) {
            return Manifold::Extrude(self.ToPolygons(), height, nDivisions,
                                     twistDegrees, scaleTop);
          },
          nb::arg("height"), nb::arg("n_divisions") = 0,
          nb::arg("twist_degrees") = 0.0f,
          nb::arg("scale_top") = std::make_tuple(1.0f, 1.0f),
          manifold__extrude__cross_section__height__n_divisions__twist_degrees__scale_top)
      .def(
          "revolve",
          [](const CrossSection &self, int circularSegments,
             double revolveDegrees) {
            return Manifold::Revolve(self.ToPolygons(), circularSegments,
                                     revolveDegrees);
          },
          nb::arg("circular_segments") = 0, nb::arg("revolve_degrees") = 360.0,
          manifold__revolve__cross_section__circular_segments__revolve_degrees)

      .def_static("square", &CrossSection::Square, nb::arg("size"),
                  nb::arg("center") = false,
                  cross_section__square__size__center)
      .def_static("circle", &CrossSection::Circle, nb::arg("radius"),
                  nb::arg("circular_segments") = 0,
                  cross_section__circle__radius__circular_segments);
}
