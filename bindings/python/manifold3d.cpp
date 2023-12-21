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

NB_MODULE(manifold3d, m) {
  m.doc() = "Python binding for the Manifold library.";

  m.def("set_min_circular_angle", Quality::SetMinCircularAngle,
        nb::arg("angle"),
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
        nb::arg("length"),
        "Sets a length constraint the default number of circular segments for "
        "the CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), "
        "and Manifold::Revolve() constructors. The number of segments will be "
        "rounded up to the nearest factor of four."
        "\n\n"
        ":param length: The minimum length of segments. The length will "
        "increase if the the segments hit the minimum angle. Default is 1.0.");

  m.def("set_circular_segments", Quality::SetCircularSegments,
        nb::arg("number"),
        "Sets the default number of circular segments for the "
        "CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and "
        "Manifold::Revolve() constructors. Overrides the edge length and angle "
        "constraints and sets the number of segments to exactly this value."
        "\n\n"
        ":param number: Number of circular segments. Default is 0, meaning no "
        "constraint is applied.");

  m.def("get_circular_segments", Quality::GetCircularSegments,
        nb::arg("radius"),
        "Determine the result of the SetMinCircularAngle(), "
        "SetMinCircularEdgeLength(), and SetCircularSegments() defaults."
        "\n\n"
        ":param radius: For a given radius of circle, determine how many "
        "default");

  m.def("triangulate", &Triangulate, nb::arg("polygons"),
        nb::arg("precision") = -1,  // TODO document
        "Given a list polygons (each polygon shape=(N,2) dtype=float), "
        "returns the indices of the triangle vertices as a "
        "numpy.ndarray(shape=(N,3), dtype=np.uint32).");

  nb::class_<Manifold>(m, "Manifold")
      .def(nb::init<>(), "Construct empty Manifold object")
      .def(nb::init<const MeshGL &, const std::vector<float> &>(),
           nb::arg("mesh"), nb::arg("property_tolerance") = nb::list(),
           "Convert a MeshGL into a Manifold, retaining its properties and "
           "merging onlythe positions according to the merge vectors. Will "
           "return an empty Manifoldand set an Error Status if the result is "
           "not an oriented 2-manifold. Willcollapse degenerate triangles and "
           "unnecessary vertices.\n\n"
           "All fields are read, making this structure suitable for a lossless "
           "round-tripof data from GetMeshGL. For multi-material input, use "
           "ReserveIDs to set aunique originalID for each material, and sort "
           "the materials into triangleruns.\n\n"
           ":param meshGL: The input MeshGL.\n"
           ":param propertyTolerance: A vector of precision values for each "
           "property beyond position. If specified, the propertyTolerance "
           "vector must have size = numProp - 3. This is the amount of "
           "interpolation error allowed before two neighboring triangles are "
           "considered to be on a property boundary edge. Property boundary "
           "edges will be retained across operations even if thetriangles are "
           "coplanar. Defaults to 1e-5, which works well for most properties "
           "in the [-1, 1] range.")
      .def(nb::self + nb::self, "Boolean union.")
      .def(nb::self - nb::self, "Boolean difference.")
      .def(nb::self ^ nb::self, "Boolean intersection.")
      .def(
          "hull", [](const Manifold &self) { return self.Hull(); },
          "Compute the convex hull of all points in this manifold.")
      .def_static(
          "batch_hull",
          [](std::vector<Manifold> ms) { return Manifold::Hull(ms); },
          nb::arg("manifolds"),
          "Compute the convex hull enveloping a set of manifolds.")
      .def_static(
          "hull_points",
          [](std::vector<glm::vec3> pts) { return Manifold::Hull(pts); },
          nb::arg("pts"),
          "Compute the convex hull enveloping a set of 3d points.")
      .def(
          "transform", &Manifold::Transform, nb::arg("m"),
          "Transform this Manifold in space. The first three columns form a "
          "3x3 matrix transform and the last is a translation vector. This "
          "operation can be chained. Transforms are combined and applied "
          "lazily.\n"
          "\n\n"
          ":param m: The affine transform matrix to apply to all the vertices.")
      .def("translate", &Manifold::Translate, nb::arg("t"),
           "Move this Manifold in space. This operation can be chained. "
           "Transforms are combined and applied lazily."
           "\n\n"
           ":param t: The vector to add to every vertex.")
      .def("scale", &Manifold::Scale, nb::arg("v"),
           "Scale this Manifold in space. This operation can be chained. "
           "Transforms are combined and applied lazily."
           "\n\n"
           ":param v: The vector to multiply every vertex by component.")
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
           "Mirror this Manifold in space. This operation can be chained. "
           "Transforms are combined and applied lazily."
           "\n\n"
           ":param mirror: The vector defining the axis of mirroring.")
      .def(
          "rotate",
          [](const Manifold &self, glm::vec3 v) {
            return self.Rotate(v.x, v.y, v.z);
          },
          nb::arg("v"),
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
          "warp", &Manifold::Warp, nb::arg("f"),
          "This function does not change the topology, but allows the vertices "
          "to be moved according to any arbitrary input function. It is easy "
          "to create a function that warps a geometrically valid object into "
          "one which overlaps, but that is not checked here, so it is up to "
          "the user to choose their function with discretion."
          "\n\n"
          ":param f: A function that modifies a given vertex position.")
      .def("warp_batch", &Manifold::WarpBatch, nb::arg("f"),
           "Same as Manifold.warp but calls `f` with a "
           "ndarray(shape=(N,3), dtype=float) and expects an ndarray "
           "of the same shape and type in return. The input array can be "
           "modified and returned if desired. "
           "\n\n"
           ":param f: A function that modifies multiple vertex positions.")
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
          "Create a new copy of this manifold with updated vertex properties "
          "by supplying a function that takes the existing position and "
          "properties as input. You may specify any number of output "
          "properties, allowing creation and removal of channels. Note: "
          "undefined behavior will result if you read past the number of input "
          "properties or write past the number of output properties."
          "\n\n"
          ":param new_num_prop: The new number of properties per vertex."
          ":param f: A function that modifies the properties of a given "
          "vertex.")
      .def(
          "calculate_curvature", &Manifold::CalculateCurvature,
          nb::arg("gaussian_idx"), nb::arg("mean_idx"),
          "Curvature is the inverse of the radius of curvature, and signed "
          "such that positive is convex and negative is concave. There are two "
          "orthogonal principal curvatures at any point on a manifold, with "
          "one maximum and the other minimum. Gaussian curvature is their "
          "product, while mean curvature is their sum. This approximates them "
          "for every vertex and assigns them as vertex properties on the given "
          "channels."
          "\n\n"
          ":param gaussianIdx: The property channel index in which to store "
          "the Gaussian curvature. An index < 0 will be ignored (stores "
          "nothing). The property set will be automatically expanded to "
          "include the channel index specified.\n"
          ":param meanIdx: The property channel index in which to store the "
          "mean curvature. An index < 0 will be ignored (stores nothing). The "
          "property set will be automatically expanded to include the channel "
          "index specified.")
      .def("refine", &Manifold::Refine, nb::arg("n"),
           "Increase the density of the mesh by splitting every edge into n "
           "pieces. For instance, with n = 2, each triangle will be split into "
           "4 triangles. These will all be coplanar (and will not be "
           "immediately collapsed) unless the Mesh/Manifold has "
           "halfedgeTangents specified (e.g. from the Smooth() constructor), "
           "in which case the new vertices will be moved to the interpolated "
           "surface according to their barycentric coordinates."
           "\n\n"
           ":param n: The number of pieces to split every edge into. Must be > "
           "1.")
      .def("to_mesh", &Manifold::GetMeshGL,
           nb::arg("normal_idx") = glm::ivec3(0),
           "The most complete output of this library, returning a MeshGL that "
           "is designed to easily push into a renderer, including all "
           "interleaved vertex properties that may have been input. It also "
           "includes relations to all the input meshes that form a part of "
           "this result and the transforms applied to each."
           "\n\n"
           ":param normal_idx: If the original MeshGL inputs that formed this "
           "manifold had properties corresponding to normal vectors, you can "
           "specify which property channels these are (x, y, z), which will "
           "cause this output MeshGL to automatically update these normals "
           "according to the applied transforms and front/back side. Each "
           "channel must be >= 3 and < numProp, and all original MeshGLs must "
           "use the same channels for their normals.")
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
          "decompose", &Manifold::Decompose,
          "This operation returns a vector of Manifolds that are "
          "topologically disconnected. If everything is connected, the vector "
          "is length one, containing a copy of the original. It is the inverse "
          "operation of Compose().")
      .def("split", &Manifold::Split, nb::arg("cutter"),
           "Split cuts this manifold in two using the cutter manifold. The "
           "first result is the intersection, second is the difference. This "
           "is more efficient than doing them separately."
           "\n\n"
           ":param cutter: This is the manifold to cut by.\n")
      .def("split_by_plane", &Manifold::SplitByPlane, nb::arg("normal"),
           nb::arg("origin_offset"),
           "Convenient version of Split() for a half-space."
           "\n\n"
           ":param normal: This vector is normal to the cutting plane and its "
           "length does not matter. The first result is in the direction of "
           "this vector, the second result is on the opposite side.\n"
           ":param origin_offset: The distance of the plane from the origin in "
           "the direction of the normal vector.")
      .def(
          "trim_by_plane", &Manifold::TrimByPlane, nb::arg("normal"),
          nb::arg("origin_offset"),
          "Identical to SplitByPlane(), but calculating and returning only the "
          "first result."
          "\n\n"
          ":param normal: This vector is normal to the cutting plane and its "
          "length does not matter. The result is in the direction of this "
          "vector from the plane.\n"
          ":param origin_offset: The distance of the plane from the origin in "
          "the direction of the normal vector.")
      .def("minkowski_sum", &Manifold::MinkowskiSum, nb::arg("other"),
           "Compute the minkowski sum of this manifold with another."
           "This corresponds to the morphological dilation of the manifold."
           "\n\n"
           ":param other: The other manifold to minkowski sum to this one.")
      .def("minkowski_difference", &Manifold::MinkowskiDifference,
           nb::arg("other"),
           "Subtract the sweep of the other manifold across this manifold's "
           "surface."
           "This corresponds to the morphological erosion of the manifold."
           "\n\n"
           ":param other: The other manifold to minkowski subtract from this "
           "one.")
      .def("slice", &Manifold::Slice, nb::arg("height"),
           "Returns the cross section of this object parallel to the X-Y plane "
           "at the specified height. Using a height equal to the bottom of the "
           "bounding box will return the bottom faces, while using a height "
           "equal to the top of the bounding box will return empty."
           "\n\n"
           ":param height: The Z-level of the slice, defaulting to zero.")
      .def("project", &Manifold::Project,
           "Returns a cross section representing the projected outline of this "
           "object onto the X-Y plane.")
      .def("status", &Manifold::Status,
           "Returns the reason for an input Mesh producing an empty Manifold. "
           "This Status only applies to Manifolds newly-created from an input "
           "Mesh - once they are combined into a new Manifold via operations, "
           "the status reverts to NoError, simply processing the problem mesh "
           "as empty. Likewise, empty meshes may still show NoError, for "
           "instance if they are small enough relative to their precision to "
           "be collapsed to nothing.")
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
          ":param sharpened_edges: If desired, you can supply a vector of "
          "sharpened halfedges, which should in general be a small subset of "
          "all halfedges. The halfedge index is "
          "(3 * triangle index + [0,1,2] where 0 is the edge between triVert 0 "
          "and 1, etc)."
          ":param edge_smoothness: must be same length as shapened_edges. "
          "Each entry specifies the desired smoothness (between zero and one, "
          "with one being the default for all unspecified halfedges)"
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
      .def_static("compose", &Manifold::Compose, nb::arg("manifolds"),
                  "combine several manifolds into one without checking for "
                  "intersections.")
      .def_static(
          "tetrahedron", &Manifold::Tetrahedron,
          "Constructs a tetrahedron centered at the origin with one vertex at "
          "(1,1,1) and the rest at similarly symmetric points.")
      .def_static(
          "cube", &Manifold::Cube, nb::arg("size") = glm::vec3{1, 1, 1},
          nb::arg("center") = false,
          "Constructs a unit cube (edge lengths all one), by default in the "
          "first octant, touching the origin."
          "\n\n"
          ":param size: The X, Y, and Z dimensions of the box.\n"
          ":param center: Set to true to shift the center to the origin.")
      .def_static(
          "cylinder", &Manifold::Cylinder, nb::arg("height"),
          nb::arg("radius_low"), nb::arg("radius_high") = -1.0f,
          nb::arg("circular_segments") = 0, nb::arg("center") = false,
          "A convenience constructor for the common case of extruding a "
          "circle. Can also form cones if both radii are specified."
          "\n\n"
          ":param height: Z-extent\n"
          ":param radius_low: Radius of bottom circle. Must be positive.\n"
          ":param radius_high: Radius of top circle. Can equal zero. Default "
          "(-1) is equal to radius_low.\n"
          ":param circular_segments: How many line segments to use around the "
          "circle. Default (-1) is calculated by the static Defaults.\n"
          ":param center: Set to true to shift the center to the origin. "
          "Default is origin at the bottom.")
      .def_static(
          "sphere", &Manifold::Sphere, nb::arg("radius"),
          nb::arg("circular_segments") = 0,
          "Constructs a geodesic sphere of a given radius.\n"
          "\n"
          ":param radius: Radius of the sphere. Must be positive.\n"
          ":param circular_segments: Number of segments along its diameter. "
          "This number will always be rounded up to the nearest factor of "
          "four, as this sphere is constructed by refining an octahedron. This "
          "means there are a circle of vertices on all three of the axis "
          "planes. Default is calculated by the static Defaults.")
      .def_static("reserve_ids", Manifold::ReserveIDs, nb::arg("n"),
                  "Returns the first of n sequential new unique mesh IDs for "
                  "marking sets of triangles that can be looked up after "
                  "further operations. Assign to MeshGL.runOriginalID vector");

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
      .def(nb::init<>(), "Construct empty CrossSection object")
      .def(nb::init<std::vector<std::vector<glm::vec2>>,
                    CrossSection::FillRule>(),
           nb::arg("polygons"),
           nb::arg("fillrule") = CrossSection::FillRule::Positive,
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
          "bounds",
          [](const CrossSection &self) {
            Rect r = self.Bounds();
            return nb::make_tuple(r.min[0], r.min[1], r.max[0], r.max[1]);
          },
          "Return bounding box of CrossSection as tuple("
          "min_x, min_y, max_x, max_y)")
      .def("translate", &CrossSection::Translate, nb::arg("v"),
           "Move this CrossSection in space. This operation can be chained. "
           "Transforms are combined and applied lazily."
           "\n\n"
           ":param v: The vector to add to every vertex.")
      .def("rotate", &CrossSection::Rotate, nb::arg("angle"),
           "Applies a (Z-axis) rotation to the CrossSection, in degrees. This "
           "operation can be chained. Transforms are combined and applied "
           "lazily."
           "\n\n"
           ":param degrees: degrees about the Z-axis to rotate.")
      .def("scale", &CrossSection::Scale, nb::arg("v"),
           "Scale this CrossSection in space. This operation can be chained. "
           "Transforms are combined and applied lazily."
           "\n\n"
           ":param v: The vector to multiply every vertex by per component.")
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
      .def(
          "mirror", &CrossSection::Mirror, nb::arg("ax"),
          "Mirror this CrossSection over the arbitrary axis described by the "
          "unit form of the given vector. If the length of the vector is zero, "
          "an empty CrossSection is returned. This operation can be chained. "
          "Transforms are combined and applied lazily."
          "\n\n"
          ":param ax: the axis to be mirrored over")
      .def(
          "transform", &CrossSection::Transform, nb::arg("m"),
          "Transform this CrossSection in space. The first two columns form a "
          "2x2 matrix transform and the last is a translation vector. This "
          "operation can be chained. Transforms are combined and applied "
          "lazily."
          "\n\n"
          ":param m: The affine transform matrix to apply to all the vertices.")
      .def("warp", &CrossSection::Warp, nb::arg("f"),
           "Move the vertices of this CrossSection (creating a new one) "
           "according to any arbitrary input function, followed by a union "
           "operation (with a Positive fill rule) that ensures any introduced "
           "intersections are not included in the result."
           "\n\n"
           ":param warpFunc: A function that modifies a given vertex position.")
      .def("warp_batch", &CrossSection::WarpBatch, nb::arg("f"),
           "Same as CrossSection.warp but calls `f` with a "
           "ndarray(shape=(N,2), dtype=float) and expects an ndarray "
           "of the same shape and type in return. The input array can be "
           "modified and returned if desired. "
           "\n\n"
           ":param f: A function that modifies multiple vertex positions.")
      .def("simplify", &CrossSection::Simplify, nb::arg("epsilon") = 1e-6,
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
      .def("offset", &CrossSection::Offset, nb::arg("delta"),
           nb::arg("join_type"), nb::arg("miter_limit") = 2.0,
           nb::arg("circular_segments") = 0,
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
           ":param circular_segments: Number of segments per 360 degrees of "
           "<B>JoinType::Round</B> corners (roughly, the number of vertices "
           "that will be added to each contour). Default is calculated by the "
           "static Quality defaults according to the radius.")
      .def(nb::self + nb::self, "Boolean union.")
      .def(nb::self - nb::self, "Boolean difference.")
      .def(nb::self ^ nb::self, "Boolean intersection.")
      .def(
          "hull", [](const CrossSection &self) { return self.Hull(); },
          "Compute the convex hull of this cross-section.")
      .def_static(
          "batch_hull",
          [](std::vector<CrossSection> cs) { return CrossSection::Hull(cs); },
          nb::arg("cross_sections"),
          "Compute the convex hull enveloping a set of cross-sections.")
      .def_static(
          "hull_points",
          [](std::vector<glm::vec2> pts) { return CrossSection::Hull(pts); },
          nb::arg("pts"),
          "Compute the convex hull enveloping a set of 2d points.")
      .def("decompose", &CrossSection::Decompose,
           "This operation returns a vector of CrossSections that are "
           "topologically disconnected, each containing one outline contour "
           "with zero or more holes.")
      .def("to_polygons", &CrossSection::ToPolygons,
           "Returns the vertices of the cross-section's polygons "
           "as a List[ndarray(shape=(*,2), dtype=float)].")
      .def(
          "extrude", &Manifold::Extrude, nb::arg("height"),
          nb::arg("n_divisions") = 0, nb::arg("twist_degrees") = 0.0f,
          nb::arg("scale_top") = std::make_tuple(1.0f, 1.0f),
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
          "revolve", &Manifold::Revolve, nb::arg("circular_segments") = 0,
          nb::arg("revolve_degrees") = 360.0,
          "Constructs a manifold from the set of polygons by revolving this "
          "cross-section around its Y-axis and then setting this as the Z-axis "
          "of the resulting manifold. If the polygons cross the Y-axis, only "
          "the part on the positive X side is used. Geometrically valid input "
          "will result in geometrically valid output.\n"
          "\n"
          ":param circular_segments: Number of segments along its diameter. "
          "Default is calculated by the static Defaults.\n"
          ":param revolve_degrees: rotation angle for the sweep.")
      .def_static(
          "square", &CrossSection::Square, nb::arg("size"),
          nb::arg("center") = false,
          "Constructs a square with the given XY dimensions. By default it is "
          "positioned in the first quadrant, touching the origin. If any "
          "dimensions in size are negative, or if all are zero, an empty "
          "Manifold will be returned."
          "\n\n"
          ":param size: The X, and Y dimensions of the square.\n"
          ":param center: Set to true to shift the center to the origin.")
      .def_static(
          "circle", &CrossSection::Circle, nb::arg("radius"),
          nb::arg("circular_segments") = 0,
          "Constructs a circle of a given radius."
          "\n\n"
          ":param radius: Radius of the circle. Must be positive.\n"
          ":param circular_segments: Number of segments along its diameter. "
          "Default is calculated by the static Quality defaults according to "
          "the radius.");
}
