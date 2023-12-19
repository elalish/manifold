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

#include "cross_section.h"
#include "docstrings.inl"
#include "manifold.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/operators.h"
#include "nanobind/stl/function.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/vector.h"
#include "polygon.h"
#include "sdf.h"

namespace nb = nanobind;
using namespace manifold;

template <class T>
struct glm_name {};
template <>
struct glm_name<glm::vec3> {
  static constexpr char const name[] = "Floatx3";
  static constexpr char const multi_name[] = "FloatNx3";
};
template <>
struct glm_name<glm::vec2> {
  static constexpr char const name[] = "Floatx2";
  static constexpr char const multi_name[] = "FloatNx2";
};
template <>
struct glm_name<glm::ivec3> {
  static constexpr char const name[] = "Intx3";
  static constexpr char const multi_name[] = "IntNx3";
};
template <>
struct glm_name<glm::mat4x3> {
  static constexpr char const name[] = "Float3x4";
};
template <>
struct glm_name<glm::mat3x2> {
  static constexpr char const name[] = "Float2x3";
};

// handle glm::vecN
template <class T, int N, glm::qualifier Q>
struct nb::detail::type_caster<glm::vec<N, T, Q>> {
  using glm_type = glm::vec<N, T, Q>;
  NB_TYPE_CASTER(glm_type, const_name(glm_name<glm_type>::name));

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    int size = PyObject_Size(src.ptr());  // negative on failure
    if (size != N) return false;
    make_caster<T> t_cast;
    for (size_t i = 0; i < size; i++) {
      if (!t_cast.from_python(src[i], flags, cleanup)) return false;
      value[i] = t_cast.value;
    }
    return true;
  }
  static handle from_cpp(glm_type vec, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    nb::list out;
    for (int i = 0; i < N; i++) out.append(vec[i]);
    return out.release();
  }
};

// handle glm::matMxN
template <class T, int C, int R, glm::qualifier Q>
struct nb::detail::type_caster<glm::mat<C, R, T, Q>> {
  using glm_type = glm::mat<C, R, T, Q>;
  using numpy_type = nb::ndarray<nb::numpy, T, nb::shape<R, C>>;
  NB_TYPE_CASTER(glm_type, const_name(glm_name<glm_type>::name));

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    int rows = PyObject_Size(src.ptr());  // negative on failure
    if (rows != R) return false;
    for (size_t i = 0; i < R; i++) {
      const nb::object &slice = src[i];
      int cols = PyObject_Size(slice.ptr());  // negative on failure
      if (cols != C) return false;
      for (size_t j = 0; j < C; j++) {
        make_caster<T> t_cast;
        if (!t_cast.from_python(slice[j], flags, cleanup)) return false;
        value[j][i] = t_cast.value;
      }
    }
    return true;
  }
  static handle from_cpp(glm_type mat, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    T *buffer = new T[R * C];
    nb::capsule mem_mgr(buffer, [](void *p) noexcept { delete[](T *) p; });
    for (int i = 0; i < R; i++) {
      for (int j = 0; j < C; j++) {
        // py is (Rows, Cols), glm is (Cols, Rows)
        buffer[i * C + j] = mat[j][i];
      }
    }
    numpy_type arr{buffer, {R, C}, std::move(mem_mgr)};
    return ndarray_wrap(arr.handle(), int(ndarray_framework::numpy), policy,
                        cleanup);
  }
};

// handle std::vector<glm::vecN>
template <class T, int N, glm::qualifier Q>
struct nb::detail::type_caster<std::vector<glm::vec<N, T, Q>>> {
  using glm_type = glm::vec<N, T, Q>;
  using numpy_type = nb::ndarray<nb::numpy, T, nb::shape<nb::any, N>>;
  NB_TYPE_CASTER(std::vector<glm_type>,
                 const_name(glm_name<glm_type>::multi_name));

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    make_caster<numpy_type> arr_cast;
    if (arr_cast.from_python(src, flags, cleanup)) {
      int num_vec = arr_cast.value.shape(0);
      value.resize(num_vec);
      for (int i = 0; i < num_vec; i++) {
        for (int j = 0; j < N; j++) {
          value[i][j] = arr_cast.value(i, j);
        }
      }
    } else {
      int num_vec = PyObject_Size(src.ptr());  // negative on failure
      if (num_vec < 0) return false;
      value.resize(num_vec);
      for (int i = 0; i < num_vec; i++) {
        make_caster<glm_type> vec_cast;
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
    nb::capsule mem_mgr(buffer, [](void *p) noexcept { delete[](T *) p; });
    for (int i = 0; i < num_vec; i++) {
      for (int j = 0; j < N; j++) {
        buffer[i * N + j] = vec[i][j];
      }
    }
    numpy_type arr{buffer, {num_vec, N}, std::move(mem_mgr)};
    return ndarray_wrap(arr.handle(), int(ndarray_framework::numpy), policy,
                        cleanup);
  }
};

// handle VecView<glm::vec*>
template <class T, int N, glm::qualifier Q>
struct nb::detail::type_caster<manifold::VecView<glm::vec<N, T, Q>>> {
  using glm_type = glm::vec<N, T, Q>;
  using numpy_type = nb::ndarray<nb::numpy, T, nb::shape<nb::any, N>>;
  NB_TYPE_CASTER(manifold::VecView<glm_type>,
                 const_name(glm_name<glm_type>::multi_name));

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    make_caster<numpy_type> arr_cast;
    if (!arr_cast.from_python(src, flags, cleanup)) return false;
    // TODO try 2d iterators if numpy cast fails
    int num_vec = arr_cast.value.shape(0);
    if (num_vec != value.size()) return false;
    for (int i = 0; i < num_vec; i++) {
      for (int j = 0; j < N; j++) {
        value[i][j] = arr_cast.value(i, j);
      }
    }
    return true;
  }
  static handle from_cpp(Value vec, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    size_t num_vec = vec.size();
    static_assert(sizeof(vec[0]) == (N * sizeof(T)),
                  "VecView -> numpy requires packed structs");
    numpy_type arr{&vec[0], {num_vec, N}};
    return ndarray_wrap(arr.handle(), int(ndarray_framework::numpy), policy,
                        cleanup);
  }
};

template <typename T>
std::vector<T> toVector(const T *arr, size_t size) {
  return std::vector<T>(arr, arr + size);
}

using namespace manifold_docs;

// strip original :params: and replace with ours
const std::string manifold__rotate_vec =
    std::string(manifold__rotate_f0bb7b7d2f38)
        .substr(0, std::string(manifold__rotate_f0bb7b7d2f38).find(":param")) +
    ":param v: [X, Y, Z] rotation in degrees.";

NB_MODULE(manifold3d, m) {
  m.doc() = "Python binding for the Manifold library.";

  m.def("set_min_circular_angle", Quality::SetMinCircularAngle,
        nb::arg("angle"), set_min_circular_angle_69463d0b8fac);

  m.def("set_min_circular_edge_length", Quality::SetMinCircularEdgeLength,
        nb::arg("length"), set_min_circular_edge_length_bb1c70addca7);

  m.def("set_circular_segments", Quality::SetCircularSegments,
        nb::arg("number"), set_circular_segments_7ebd2a75022a);

  m.def("get_circular_segments", Quality::GetCircularSegments,
        nb::arg("radius"), get_circular_segments_3f31ccff2bbc);

  m.def("triangulate", &Triangulate, nb::arg("polygons"),
        nb::arg("precision") = -1,  // TODO document
        triangulate_025ab0b4046f);

  nb::class_<Manifold>(m, "Manifold")
      .def(nb::init<>(), manifold__manifold_c130481162c9)
      .def(nb::init<const MeshGL &, const std::vector<float> &>(),
           nb::arg("mesh"), nb::arg("property_tolerance") = nb::list(),
           manifold__manifold_37129c244d43)
      .def(nb::self + nb::self, "Boolean union.")
      .def(nb::self - nb::self, "Boolean difference.")
      .def(nb::self ^ nb::self, "Boolean intersection.")
      .def(
          "hull", [](const Manifold &self) { return self.Hull(); },
          manifold__hull_1fd92449f7cb)
      .def_static(
          "batch_hull",
          [](std::vector<Manifold> ms) { return Manifold::Hull(ms); },
          nb::arg("manifolds"), manifold__hull_e1568f7f16f2)
      .def_static(
          "hull_points",
          [](std::vector<glm::vec3> pts) { return Manifold::Hull(pts); },
          nb::arg("pts"), manifold__hull_b0113f48020a)
      .def("transform", &Manifold::Transform, nb::arg("m"),
           manifold__transform_0390744b2b46)
      .def("translate", &Manifold::Translate, nb::arg("t"),
           manifold__translate_fcadc2ca8d6d)
      .def("scale", &Manifold::Scale, nb::arg("v"),
           manifold__scale_244be87a307d)
      .def(
          "scale",
          [](const Manifold &m, float s) {
            m.Scale({s, s, s});
          },
          nb::arg("s"),
          "Scale this Manifold in space. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param s: The scalar to multiply every vertex by component.")
      .def("mirror", &Manifold::Mirror, nb::arg("v"),
           manifold__mirror_d798a49656cc)
      .def(
          "rotate",
          [](const Manifold &self, glm::vec3 v) {
            return self.Rotate(v.x, v.y, v.z);
          },
          nb::arg("v"), manifold__rotate_vec.c_str())
      .def("warp", &Manifold::Warp, nb::arg("f"), manifold__warp_4cc67905f424)
      .def("warp_batch", &Manifold::WarpBatch, nb::arg("f"),
           manifold__warp_batch_0b44f7bbe36b)
      .def(
          "set_properties",  // TODO this needs a batch version!
          [](const Manifold &self, int newNumProp,
             const std::function<nb::object(
                 glm::vec3, const nb::ndarray<nb::numpy, const float,
                                              nb::c_contig> &)> &f) {
            const int oldNumProp = self.NumProp();
            return self.SetProperties(newNumProp, [newNumProp, oldNumProp, &f](
                                                      float *newProps,
                                                      glm::vec3 v,
                                                      const float *oldProps) {
              auto result =
                  f(v, nb::ndarray<nb::numpy, const float, nb::c_contig>(
                           oldProps, {static_cast<unsigned long>(oldNumProp)}));
              nb::ndarray<float, nb::shape<nb::any>> array;
              std::vector<float> vec;
              if (nb::try_cast(result, array)) {
                if (array.ndim() != 1 || array.shape(0) != newNumProp)
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
          manifold__set_properties_6a3457d9ec71)
      .def("calculate_curvature", &Manifold::CalculateCurvature,
           nb::arg("gaussian_idx"), nb::arg("mean_idx"),
           manifold__calculate_curvature_0bb0af918c76)
      .def("refine", &Manifold::Refine, nb::arg("n"),
           manifold__refine_c64a4fc78137)
      .def("to_mesh", &Manifold::GetMeshGL,
           nb::arg("normal_idx") = glm::ivec3(0),
           manifold__get_mesh_gl_731af09ec81f)
      .def("num_vert", &Manifold::NumVert, manifold__num_vert_93a4106c8a53)
      .def("num_edge", &Manifold::NumEdge, manifold__num_edge_61b0f3dc7f99)
      .def("num_tri", &Manifold::NumTri, manifold__num_tri_56a67713dff8)
      .def("num_prop", &Manifold::NumProp, manifold__num_prop_745ca800e017)
      .def("num_prop_vert", &Manifold::NumPropVert,
           manifold__num_prop_vert_a7ba865d3e11)
      .def("precision", &Manifold::Precision, manifold__precision_bb888ab8ec11)
      .def("genus", &Manifold::Genus, manifold__genus_75c215a950f8)
      .def(
          "volume",
          [](const Manifold &self) { return self.GetProperties().volume; },
          "Get the volume of the manifold\n This is clamped to zero for a "
          "given face if they are within the Precision().")
      .def(
          "surface_area",
          [](const Manifold &self) { return self.GetProperties().surfaceArea; },
          "Get the surface area of the manifold\n This is clamped to zero for "
          "a given face if they are within the Precision().")
      .def("original_id", &Manifold::OriginalID,
           manifold__original_id_d1c064807f94)
      .def("as_original", &Manifold::AsOriginal,
           manifold__as_original_2653ccf900dd)
      .def("is_empty", &Manifold::IsEmpty, manifold__is_empty_8f3c4e98cca8)
      .def("decompose", &Manifold::Decompose, manifold__decompose_88ccab82c740)
      .def("split", &Manifold::Split, nb::arg("cutter"),
           manifold__split_fc2847c7afae)
      .def("split_by_plane", &Manifold::SplitByPlane, nb::arg("normal"),
           nb::arg("origin_offset"), manifold__split_by_plane_f411533a14aa)
      .def("trim_by_plane", &Manifold::TrimByPlane, nb::arg("normal"),
           nb::arg("origin_offset"), manifold__trim_by_plane_066ac34a84b0)
      .def("slice", &Manifold::Slice, nb::arg("height"),
           manifold__slice_7d90a75e7913)
      .def("project", &Manifold::Project, manifold__project_e28980e0f682)
      .def("status", &Manifold::Status, manifold__status_b1c2b69ee41e)
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
          [](const MeshGL &mesh, std::vector<int> sharpened_edges,
             std::vector<float> edge_smoothness) {
            if (sharpened_edges.size() != edge_smoothness.size()) {
              throw std::runtime_error(
                  "sharpened_edges.size() != edge_smoothness.size()");
            }
            std::vector<Smoothness> vec(sharpened_edges.size());
            for (int i = 0; i < vec.size(); i++) {
              vec[i] = {sharpened_edges[i], edge_smoothness[i]};
            }
            return Manifold::Smooth(mesh, vec);
          },
          nb::arg("mesh"), nb::arg("sharpened_edges") = nb::list(),
          nb::arg("edge_smoothness") = nb::list(),
          // todo params slightly diff
          manifold__smooth_66eaffd9331b)
      .def_static("compose", &Manifold::Compose, nb::arg("manifolds"),
                  manifold__compose_6c382bb1612b)
      .def_static("tetrahedron", &Manifold::Tetrahedron,
                  manifold__tetrahedron_7e95f682f35b)
      .def_static("cube", &Manifold::Cube, nb::arg("size") = glm::vec3{1, 1, 1},
                  nb::arg("center") = false, manifold__cube_64d1c43c52ed)
      .def_static("cylinder", &Manifold::Cylinder, nb::arg("height"),
                  nb::arg("radius_low"), nb::arg("radius_high") = -1.0f,
                  nb::arg("circular_segments") = 0, nb::arg("center") = false,
                  manifold__cylinder_af7b1b7dc893)
      .def_static("sphere", &Manifold::Sphere, nb::arg("radius"),
                  nb::arg("circular_segments") = 0,
                  manifold__sphere_6781451731f0)
      .def_static("reserve_ids", Manifold::ReserveIDs, nb::arg("n"),
                  manifold__reserve_ids_a514f84c6343);

  nb::class_<MeshGL>(m, "Mesh")
      .def(
          // note that reshape requires mutable ndarray, but this will not
          // affect the original array passed into the function
          "__init__",
          [](MeshGL *self,
             nb::ndarray<float, nb::shape<nb::any, nb::any>, nb::c_contig>
                 &vertProp,
             nb::ndarray<uint32_t, nb::shape<nb::any, 3>, nb::c_contig>
                 &triVerts,
             const std::optional<nb::ndarray<uint32_t, nb::shape<nb::any>,
                                             nb::c_contig>> &mergeFromVert,
             const std::optional<nb::ndarray<uint32_t, nb::shape<nb::any>,
                                             nb::c_contig>> &mergeToVert,
             const std::optional<nb::ndarray<uint32_t, nb::shape<nb::any>,
                                             nb::c_contig>> &runIndex,
             const std::optional<nb::ndarray<uint32_t, nb::shape<nb::any>,
                                             nb::c_contig>> &runOriginalID,
             std::optional<nb::ndarray<float, nb::shape<nb::any, 4, 3>,
                                       nb::c_contig>> &runTransform,
             const std::optional<nb::ndarray<uint32_t, nb::shape<nb::any>,
                                             nb::c_contig>> &faceID,
             const std::optional<nb::ndarray<float, nb::shape<nb::any, 3, 4>,
                                             nb::c_contig>> &halfedgeTangent,
             float precision) {
    new (self) MeshGL();
    MeshGL &out = *self;
    out.numProp = vertProp.shape(1);
    out.vertProperties = toVector<float>(vertProp.data(), vertProp.size());

    out.triVerts = toVector<uint32_t>(triVerts.data(), triVerts.size());

    if (mergeFromVert.has_value())
      out.mergeFromVert =
          toVector<uint32_t>(mergeFromVert->data(), mergeFromVert->size());

    if (mergeToVert.has_value())
      out.mergeToVert =
          toVector<uint32_t>(mergeToVert->data(), mergeToVert->size());

    if (runIndex.has_value())
      out.runIndex = toVector<uint32_t>(runIndex->data(), runIndex->size());

    if (runOriginalID.has_value())
      out.runOriginalID =
          toVector<uint32_t>(runOriginalID->data(), runOriginalID->size());

    if (runTransform.has_value()) {
      out.runTransform =
          toVector<float>(runTransform->data(), runTransform->size());
    }

    if (faceID.has_value())
      out.faceID = toVector<uint32_t>(faceID->data(), faceID->size());

    if (halfedgeTangent.has_value()) {
      out.halfedgeTangent =
          toVector<float>(halfedgeTangent->data(), halfedgeTangent->size());
    }
          },
          nb::arg("vert_properties"), nb::arg("tri_verts"),
          nb::arg("merge_from_vert") = nb::none(),
          nb::arg("merge_to_vert") = nb::none(),
          nb::arg("run_index") = nb::none(),
          nb::arg("run_original_id") = nb::none(),
          nb::arg("run_transform") = nb::none(),
          nb::arg("face_id") = nb::none(),
          nb::arg("halfedge_tangent") = nb::none(), nb::arg("precision") = 0)
      .def_prop_ro("vert_properties",
                   [](const MeshGL &self) {
    return nb::ndarray<nb::numpy, const float, nb::c_contig>(
        self.vertProperties.data(),
        {self.vertProperties.size() / self.numProp, self.numProp});
                   }, nb::rv_policy::reference_internal)
      .def_prop_ro("tri_verts",
                   [](const MeshGL &self) {
    return nb::ndarray<nb::numpy, const int, nb::c_contig>(
        self.triVerts.data(), {self.triVerts.size() / 3, 3});
                   }, nb::rv_policy::reference_internal)
      .def_prop_ro("run_transform",
                   [](const MeshGL &self) {
    return nb::ndarray<nb::numpy, const float, nb::c_contig>(
        self.runTransform.data(), {self.runTransform.size() / 12, 4, 3});
                   }, nb::rv_policy::reference_internal)
      .def_prop_ro("halfedge_tangent",
                   [](const MeshGL &self) {
    return nb::ndarray<nb::numpy, const float, nb::c_contig>(
        self.halfedgeTangent.data(), {self.halfedgeTangent.size() / 12, 3, 4});
                   }, nb::rv_policy::reference_internal)
      .def_ro("merge_from_vert", &MeshGL::mergeFromVert)
      .def_ro("merge_to_vert", &MeshGL::mergeToVert)
      .def_ro("run_index", &MeshGL::runIndex)
      .def_ro("run_original_id", &MeshGL::runOriginalID)
      .def_ro("face_id", &MeshGL::faceID)
      .def_static(
          "level_set", 
          [](const std::function<float(float, float, float)> &f,
             std::vector<float> bounds, float edgeLength, float level = 0.0) {
                // Same format as Manifold.bounding_box
                Box bound = {glm::vec3(bounds[0], bounds[1], bounds[2]),
                            glm::vec3(bounds[3], bounds[4], bounds[5])};

                std::function<float(glm::vec3)> cppToPython = [&f](glm::vec3 v) {
                  return f(v.x, v.y, v.z);
                };
                Mesh result = LevelSet(cppToPython, bound, edgeLength, level, false);
                return MeshGL(result);
          },
          nb::arg("f"), nb::arg("bounds"), nb::arg("edgeLength"),
          nb::arg("level") = 0.0,
          "Constructs a level-set Mesh from the input Signed-Distance Function "
          "(SDF) This uses a form of Marching Tetrahedra (akin to Marching "
          "Cubes, but better for manifoldness). Instead of using a cubic grid, "
          "it uses a body-centered cubic grid (two shifted cubic grids). This "
          "means if your function's interior exceeds the given bounds, you "
          "will see a kind of egg-crate shape closing off the manifold, which "
          "is due to the underlying grid."
          "\n\n"
          ":param f: The signed-distance functor, containing this function "
          "signature: `def sdf(xyz : tuple) -> float:`, which returns the "
          "signed distance of a given point in R^3. Positive values are "
          "inside, negative outside."
          ":param bounds: An axis-aligned box that defines the extent of the "
          "grid."
          ":param edgeLength: Approximate maximum edge length of the triangles "
          "in the final result.  This affects grid spacing, and hence has a "
          "strong effect on performance."
          ":param level: You can inset your Mesh by using a positive value, or "
          "outset it with a negative value."
          ":return Mesh: This mesh is guaranteed to be manifold."
          "Use Manifold.from_mesh(mesh) to create a Manifold");

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

  nb::class_<CrossSection>(
      m, "CrossSection",
      "Two-dimensional cross sections guaranteed to be without "
      "self-intersections, or overlaps between polygons (from construction "
      "onwards). This class makes use of the "
      "[Clipper2](http://www.angusj.com/clipper2/Docs/Overview.htm) library "
      "for polygon clipping (boolean) and offsetting operations.")
      .def(nb::init<>(), cross_section__cross_section_381df1ec382b)
      .def(nb::init<std::vector<std::vector<glm::vec2>>,
                    CrossSection::FillRule>(),
           nb::arg("polygons"),
           nb::arg("fillrule") = CrossSection::FillRule::Positive,
           cross_section__cross_section_4459865c1e2f)
      .def("area", &CrossSection::Area, cross_section__area_f5458809be32)
      .def("num_vert", &CrossSection::NumVert,
           cross_section__num_vert_9dd2efd31062)
      .def("num_contour", &CrossSection::NumContour,
           cross_section__num_contour_5894fa74e5f5)
      .def("is_empty", &CrossSection::IsEmpty,
           cross_section__is_empty_25b4b2d4e0ad)
      .def(
          "bounds",
          [](const CrossSection &self) {
            Rect r = self.Bounds();
            return nb::make_tuple(r.min[0], r.min[1], r.max[0], r.max[1]);
          },
          "Return bounding box of CrossSection as tuple("
          "min_x, min_y, max_x, max_y)")
      .def("translate", &CrossSection::Translate, nb::arg("v"),
           cross_section__translate_339895387e15)
      .def("rotate", &CrossSection::Rotate, nb::arg("angle"),
           cross_section__rotate_7c6bae9524e7)
      .def("scale", &CrossSection::Scale, nb::arg("v"),
           cross_section__scale_8913c878f656)
      .def(
          "scale",
          [](const CrossSection &self, float s) {
            self.Scale({s, s});
          },
          nb::arg("s"),
          "Scale this CrossSection in space. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param s: The scalar to multiply every vertex by per component.")
      .def("mirror", &CrossSection::Mirror, nb::arg("ax"),
           cross_section__mirror_a10119e40f21)
      .def("transform", &CrossSection::Transform, nb::arg("m"),
           cross_section__transform_baddfca7ede3)
      .def("warp", &CrossSection::Warp, nb::arg("f"),
           cross_section__warp_180cafeaaad1)
      .def("warp_batch", &CrossSection::WarpBatch, nb::arg("f"),
           cross_section__warp_batch_f843ce28c677)
      .def("simplify", &CrossSection::Simplify, nb::arg("epsilon") = 1e-6,
           cross_section__simplify_dbac3e60acf4)
      .def("offset", &CrossSection::Offset, nb::arg("delta"),
           nb::arg("join_type"), nb::arg("miter_limit") = 2.0,
           nb::arg("circular_segments") = 0, cross_section__offset_b3675b4b0ed0)
      .def(nb::self + nb::self, cross_section__operator_plus_d3c26b9c5ca3)
      .def(nb::self - nb::self, cross_section__operator_minus_04b4d727817f)
      .def(nb::self ^ nb::self, cross_section__operator_xor_76de317c9be1)
      .def(
          "hull", [](const CrossSection &self) { return self.Hull(); },
          cross_section__hull_3f1ad9eaa499)
      .def_static(
          "batch_hull",
          [](std::vector<CrossSection> cs) { return CrossSection::Hull(cs); },
          nb::arg("cross_sections"), cross_section__hull_014f76304d06)
      .def_static(
          "hull_points",
          [](std::vector<glm::vec2> pts) { return CrossSection::Hull(pts); },
          nb::arg("pts"), cross_section__hull_c94ccc3c0fe6)
      .def("decompose", &CrossSection::Decompose,
           cross_section__decompose_17ae5159e6e5)
      .def("to_polygons", &CrossSection::ToPolygons,
           cross_section__to_polygons_6f4cb60dbd78)
      .def("extrude", &Manifold::Extrude, nb::arg("height"),
           nb::arg("n_divisions") = 0, nb::arg("twist_degrees") = 0.0f,
           nb::arg("scale_top") = std::make_tuple(1.0f, 1.0f),
           manifold__extrude_bc84f1554abe)
      .def("revolve", &Manifold::Revolve, nb::arg("circular_segments") = 0,
           nb::arg("revolve_degrees") = 360.0, manifold__revolve_c916603e7a75)
      .def_static("square", &CrossSection::Square, nb::arg("size"),
                  nb::arg("center") = false, cross_section__square_67088e89831e)
      .def_static("circle", &CrossSection::Circle, nb::arg("radius"),
                  nb::arg("circular_segments") = 0,
                  cross_section__circle_72e80d0e2b44);
}
