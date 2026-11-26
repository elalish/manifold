// Copyright 2023 The Manifold Authors.
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

#include "manifold/manifoldc.h"

#ifndef MANIFOLD_NO_IOSTREAM
#include <sstream>
#endif
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "conv.h"
#include "manifold/common.h"
#include "manifold/manifold.h"
#include "manifold/polygon.h"
#include "manifold/types.h"

using namespace manifold;

static_assert(MANIFOLD_MIN_RELATIVE_PRECISION == kMinRelativePrecision);

namespace {
ManifoldManifold* level_set(double (*sdf_context)(double, double, double,
                                                  void*),
                            const ManifoldBox* bounds, double edge_length,
                            double level, double tolerance, bool seq, void* ctx,
                            ExecutionContext* ec = nullptr) {
  // Bind function with context argument to one without
  using namespace std::placeholders;
  std::function<double(double, double, double)> sdf =
      std::bind(sdf_context, _1, _2, _3, ctx);
  std::function<double(vec3)> fun = [sdf](vec3 v) {
    return (sdf(v.x, v.y, v.z));
  };
  // Run under the ExecutionContext when given, so progress/cancel are observed.
  Manifold result = ec ? ec->LevelSet(fun, *from_c(bounds), edge_length, level,
                                      tolerance, !seq)
                       : Manifold::LevelSet(fun, *from_c(bounds), edge_length,
                                            level, tolerance, !seq);
  return to_c(new Manifold(result));
}

template <typename Mesh, typename Float, typename Index, typename Options>
Mesh* meshgl_w_options(const Float* vert_props, size_t n_verts, size_t n_props,
                       const Index* tri_verts, size_t n_tris,
                       const Options* options) {
  // The v4.0 struct ends at face_id. Read a field added after it only when
  // struct_size covers it, otherwise treat it as empty.
  constexpr size_t minSize =
      offsetof(Options, face_id) + sizeof(Options::face_id);
  if (options != nullptr && options->struct_size < minSize) {
    std::fprintf(stderr,
                 "manifold: options struct_size %zu is smaller than %zu\n",
                 options->struct_size, minSize);
    std::abort();
  }

  auto mesh = new Mesh();
  mesh->numProp = n_props;
  mesh->vertProperties = vector_of_array(vert_props, n_verts * n_props);
  mesh->triVerts = vector_of_array(tri_verts, n_tris * 3);
  if (options == nullptr) return mesh;

  if (options->halfedge_tangents != nullptr) {
    mesh->halfedgeTangent =
        vector_of_array(options->halfedge_tangents, n_tris * 3 * 4);
  }
  if (options->run_indices != nullptr) {
    mesh->runIndex =
        vector_of_array(options->run_indices, options->run_indices_length);
  }
  if (options->run_original_ids != nullptr) {
    mesh->runOriginalID = vector_of_array(options->run_original_ids,
                                          options->run_original_ids_length);
  }
  if (options->merge_from_vert != nullptr) {
    mesh->mergeFromVert =
        vector_of_array(options->merge_from_vert, options->merge_verts_length);
  }
  if (options->merge_to_vert != nullptr) {
    mesh->mergeToVert =
        vector_of_array(options->merge_to_vert, options->merge_verts_length);
  }
  if (options->run_transform != nullptr) {
    mesh->runTransform =
        vector_of_array(options->run_transform, options->run_transform_length);
  }
  if (options->run_flags != nullptr) {
    mesh->runFlags =
        vector_of_array(options->run_flags, options->run_flags_length);
  }
  if (options->face_id != nullptr) {
    mesh->faceID = vector_of_array(options->face_id, n_tris);
  }
  return mesh;
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

ManifoldSimplePolygon* manifold_simple_polygon(const ManifoldVec2* ps,
                                               size_t length) {
  auto vec = new std::vector<vec2>;
  vec->reserve(length);
  for (size_t i = 0; i < length; ++i) {
    vec->push_back({ps[i].x, ps[i].y});
  }
  return to_c(vec);
}

ManifoldPolygons* manifold_polygons(const ManifoldSimplePolygon* const* ps,
                                    size_t length) {
  auto vec = new std::vector<SimplePolygon>;
  vec->reserve(length);
  auto polys = reinterpret_cast<const SimplePolygon* const*>(ps);
  for (size_t i = 0; i < length; ++i) {
    vec->push_back(*polys[i]);
  }
  return to_c(vec);
}

size_t manifold_simple_polygon_length(const ManifoldSimplePolygon* p) {
  return from_c(p)->size();
}

size_t manifold_polygons_length(const ManifoldPolygons* ps) {
  return from_c(ps)->size();
}

size_t manifold_polygons_simple_length(const ManifoldPolygons* ps, size_t idx) {
  return (*from_c(ps))[idx].size();
}

ManifoldVec2 manifold_simple_polygon_get_point(const ManifoldSimplePolygon* p,
                                               size_t idx) {
  return to_c((*from_c(p))[idx]);
}

ManifoldSimplePolygon* manifold_polygons_get_simple(const ManifoldPolygons* ps,
                                                    size_t idx) {
  return to_c(new SimplePolygon((*from_c(ps))[idx]));
}

ManifoldVec2 manifold_polygons_get_point(const ManifoldPolygons* ps,
                                         size_t simple_idx, size_t pt_idx) {
  return to_c((*from_c(ps))[simple_idx][pt_idx]);
}

ManifoldManifoldVec* manifold_manifold_empty_vec() {
  return to_c(new ManifoldVec());
}

ManifoldManifoldVec* manifold_manifold_vec(size_t sz) {
  return to_c(new ManifoldVec(sz));
}

void manifold_manifold_vec_reserve(ManifoldManifoldVec* ms, size_t sz) {
  from_c(ms)->reserve(sz);
}

size_t manifold_manifold_vec_length(const ManifoldManifoldVec* ms) {
  return from_c(ms)->size();
}

ManifoldManifold* manifold_manifold_vec_get(const ManifoldManifoldVec* ms,
                                            size_t idx) {
  auto m = (*from_c(ms))[idx];
  return to_c(new Manifold(m));
}

void manifold_manifold_vec_set(ManifoldManifoldVec* ms, size_t idx,
                               const ManifoldManifold* m) {
  (*from_c(ms))[idx] = *from_c(m);
}

void manifold_manifold_vec_push_back(ManifoldManifoldVec* ms,
                                     const ManifoldManifold* m) {
  return from_c(ms)->push_back(*from_c(m));
}

ManifoldManifold* manifold_boolean(const ManifoldManifold* a,
                                   const ManifoldManifold* b,
                                   ManifoldOpType op) {
  auto m = from_c(a)->Boolean(*from_c(b), from_c(op));
  return to_c(new Manifold(m));
}

ManifoldManifold* manifold_batch_boolean(const ManifoldManifoldVec* ms,
                                         ManifoldOpType op) {
  auto m = Manifold::BatchBoolean(*from_c(ms), from_c(op));
  return to_c(new Manifold(m));
}

ManifoldManifold* manifold_union(const ManifoldManifold* a,
                                 const ManifoldManifold* b) {
  auto m = (*from_c(a)) + (*from_c(b));
  return to_c(new Manifold(m));
}

ManifoldManifold* manifold_difference(const ManifoldManifold* a,
                                      const ManifoldManifold* b) {
  auto m = (*from_c(a)) - (*from_c(b));
  return to_c(new Manifold(m));
}

ManifoldManifold* manifold_intersection(const ManifoldManifold* a,
                                        const ManifoldManifold* b) {
  auto m = (*from_c(a)) ^ (*from_c(b));
  return to_c(new Manifold(m));
}

ManifoldManifoldPair manifold_split(const ManifoldManifold* a,
                                    const ManifoldManifold* b) {
  auto pair = from_c(a)->Split(*from_c(b));
  auto first = new Manifold(pair.first);
  auto second = new Manifold(pair.second);
  return {to_c(first), to_c(second)};
}

ManifoldManifoldPair manifold_split_by_plane(const ManifoldManifold* m,
                                             double normal_x, double normal_y,
                                             double normal_z, double offset) {
  auto normal = vec3(normal_x, normal_y, normal_z);
  auto pair = from_c(m)->SplitByPlane(normal, offset);
  auto first = new Manifold(pair.first);
  auto second = new Manifold(pair.second);
  return {to_c(first), to_c(second)};
}

ManifoldManifold* manifold_trim_by_plane(const ManifoldManifold* m,
                                         double normal_x, double normal_y,
                                         double normal_z, double offset) {
  auto normal = vec3(normal_x, normal_y, normal_z);
  auto trimmed = from_c(m)->TrimByPlane(normal, offset);
  return to_c(new Manifold(trimmed));
}

ManifoldManifold* manifold_minkowski_sum(const ManifoldManifold* a,
                                         const ManifoldManifold* b) {
  auto m = (*from_c(a)).MinkowskiSum(*from_c(b));
  return to_c(new Manifold(m));
}

ManifoldManifold* manifold_minkowski_difference(const ManifoldManifold* a,
                                                const ManifoldManifold* b) {
  auto m = (*from_c(a)).MinkowskiDifference(*from_c(b));
  return to_c(new Manifold(m));
}

ManifoldPolygons* manifold_slice(const ManifoldManifold* m, double height) {
  auto poly = from_c(m)->Slice(height);
  return to_c(new Polygons(poly));
}

ManifoldPolygons* manifold_project(const ManifoldManifold* m) {
  auto poly = from_c(m)->Project();
  return to_c(new Polygons(poly));
}

ManifoldManifold* manifold_hull(const ManifoldManifold* m) {
  auto hulled = from_c(m)->Hull();
  return to_c(new Manifold(hulled));
}

ManifoldManifold* manifold_batch_hull(const ManifoldManifoldVec* ms) {
  auto hulled = Manifold::Hull(*from_c(ms));
  return to_c(new Manifold(hulled));
}

ManifoldManifold* manifold_hull_pts(const ManifoldVec3* ps, size_t length) {
  std::vector<vec3> vec(length);
  for (size_t i = 0; i < length; ++i) {
    vec[i] = {ps[i].x, ps[i].y, ps[i].z};
  }
  auto hulled = Manifold::Hull(vec);
  return to_c(new Manifold(hulled));
}

ManifoldManifold* manifold_translate(const ManifoldManifold* m, double x,
                                     double y, double z) {
  auto v = vec3(x, y, z);
  auto translated = from_c(m)->Translate(v);
  return to_c(new Manifold(translated));
}

ManifoldManifold* manifold_rotate(const ManifoldManifold* m, double x, double y,
                                  double z) {
  auto rotated = from_c(m)->Rotate(x, y, z);
  return to_c(new Manifold(rotated));
}

ManifoldManifold* manifold_scale(const ManifoldManifold* m, double x, double y,
                                 double z) {
  auto s = vec3(x, y, z);
  auto scaled = from_c(m)->Scale(s);
  return to_c(new Manifold(scaled));
}

ManifoldManifold* manifold_transform(const ManifoldManifold* m, double x1,
                                     double y1, double z1, double x2, double y2,
                                     double z2, double x3, double y3, double z3,
                                     double x4, double y4, double z4) {
  auto mat = mat3x4({x1, y1, z1}, {x2, y2, z2}, {x3, y3, z3}, {x4, y4, z4});
  auto transformed = from_c(m)->Transform(mat);
  return to_c(new Manifold(transformed));
}

ManifoldManifold* manifold_mirror(const ManifoldManifold* m, double nx,
                                  double ny, double nz) {
  auto mirrored = from_c(m)->Mirror({nx, ny, nz});
  return to_c(new Manifold(mirrored));
}

ManifoldManifold* manifold_warp(const ManifoldManifold* m,
                                ManifoldVec3 (*fun)(double, double, double,
                                                    void*),
                                void* ctx) {
  // Bind function with context argument to one without
  using namespace std::placeholders;
  std::function<ManifoldVec3(double, double, double)> f3 =
      std::bind(fun, _1, _2, _3, ctx);
  std::function<void(vec3 & v)> warp = [f3](vec3& v) {
    v = from_c(f3(v.x, v.y, v.z));
  };
  auto warped = from_c(m)->Warp(warp);
  return to_c(new Manifold(warped));
}

ManifoldManifold* manifold_warp_batch(const ManifoldManifold* m,
                                      void (*fun)(double* xyz, size_t count,
                                                  void* ctx),
                                      void* ctx) {
  static_assert(sizeof(vec3) == 3 * sizeof(double));
  auto warped = from_c(m)->WarpBatch([fun, ctx](VecView<vec3> verts) {
    fun(reinterpret_cast<double*>(verts.data()), verts.size(), ctx);
  });
  return to_c(new Manifold(warped));
}

ManifoldManifold* manifold_level_set(
    double (*sdf)(double, double, double, void*), const ManifoldBox* bounds,
    double edge_length, double level, double tolerance, void* ctx) {
  return level_set(sdf, bounds, edge_length, level, tolerance, false, ctx);
}

ManifoldManifold* manifold_level_set_seq(
    double (*sdf)(double, double, double, void*), const ManifoldBox* bounds,
    double edge_length, double level, double tolerance, void* ctx) {
  return level_set(sdf, bounds, edge_length, level, tolerance, true, ctx);
}

ManifoldManifold* manifold_smooth_by_normals(const ManifoldManifold* m) {
  auto smoothed = from_c(m)->SmoothByNormals();
  return to_c(new Manifold(smoothed));
}

ManifoldManifold* manifold_refine(const ManifoldManifold* m, int refine) {
  auto refined = from_c(m)->Refine(refine);
  return to_c(new Manifold(refined));
}

ManifoldManifold* manifold_refine_to_length(const ManifoldManifold* m,
                                            double length) {
  auto refined = from_c(m)->RefineToLength(length);
  return to_c(new Manifold(refined));
}

ManifoldManifold* manifold_refine_to_tolerance(const ManifoldManifold* m,
                                               double tolerance) {
  auto refined = from_c(m)->RefineToTolerance(tolerance);
  return to_c(new Manifold(refined));
}

ManifoldManifold* manifold_simplify(const ManifoldManifold* m,
                                    double tolerance) {
  auto simplified = from_c(m)->Simplify(tolerance);
  return to_c(new Manifold(simplified));
}

ManifoldManifold* manifold_empty() { return to_c(new Manifold()); }

ManifoldManifold* manifold_copy(const ManifoldManifold* m) {
  return to_c(new Manifold(*from_c(m)));
}

ManifoldManifold* manifold_tetrahedron() {
  auto m = Manifold::Tetrahedron();
  return to_c(new Manifold(m));
}

ManifoldManifold* manifold_cube(double x, double y, double z, int center) {
  auto size = vec3(x, y, z);
  auto m = Manifold::Cube(size, center);
  return to_c(new Manifold(m));
}

ManifoldManifold* manifold_cylinder(double height, double radius_low,
                                    double radius_high,
                                    const ManifoldCircularQuality* quality,
                                    int center) {
  auto m = Manifold::Cylinder(height, radius_low, radius_high, from_c(quality),
                              center);
  return to_c(new Manifold(m));
}

ManifoldManifold* manifold_sphere(double radius,
                                  const ManifoldCircularQuality* quality) {
  auto m = Manifold::Sphere(radius, from_c(quality));
  return to_c(new Manifold(m));
}

ManifoldMeshGL* manifold_meshgl(const float* vert_props, size_t n_verts,
                                size_t n_props, const uint32_t* tri_verts,
                                size_t n_tris) {
  auto mesh = new MeshGL();
  mesh->numProp = n_props;
  mesh->vertProperties = vector_of_array(vert_props, n_verts * n_props);
  mesh->triVerts = vector_of_array(tri_verts, n_tris * 3);
  return to_c(mesh);
}

ManifoldMeshGL* manifold_meshgl_w_tangents(const float* vert_props,
                                           size_t n_verts, size_t n_props,
                                           const uint32_t* tri_verts,
                                           size_t n_tris,
                                           const float* halfedge_tangent) {
  auto mesh = new MeshGL();
  mesh->numProp = n_props;
  mesh->vertProperties = vector_of_array(vert_props, n_verts * n_props);
  mesh->triVerts = vector_of_array(tri_verts, n_tris * 3);
  mesh->halfedgeTangent = vector_of_array(halfedge_tangent, n_tris * 3 * 4);
  return to_c(mesh);
}

ManifoldMeshGL* manifold_meshgl_w_options(
    const float* vert_props, size_t n_verts, size_t n_props,
    const uint32_t* tri_verts, size_t n_tris,
    const ManifoldMeshGLOptions* options) {
  return to_c(meshgl_w_options<MeshGL>(vert_props, n_verts, n_props, tri_verts,
                                       n_tris, options));
}

ManifoldMeshGL64* manifold_meshgl64(const double* vert_props, size_t n_verts,
                                    size_t n_props, const uint64_t* tri_verts,
                                    size_t n_tris) {
  auto mesh = new MeshGL64();
  mesh->numProp = n_props;
  mesh->vertProperties = vector_of_array(vert_props, n_verts * n_props);
  mesh->triVerts = vector_of_array(tri_verts, n_tris * 3);
  return to_c(mesh);
}

ManifoldMeshGL64* manifold_meshgl64_w_tangents(const double* vert_props,
                                               size_t n_verts, size_t n_props,
                                               const uint64_t* tri_verts,
                                               size_t n_tris,
                                               const double* halfedge_tangent) {
  auto mesh = new MeshGL64();
  mesh->numProp = n_props;
  mesh->vertProperties = vector_of_array(vert_props, n_verts * n_props);
  mesh->triVerts = vector_of_array(tri_verts, n_tris * 3);
  mesh->halfedgeTangent = vector_of_array(halfedge_tangent, n_tris * 3 * 4);
  return to_c(mesh);
}

ManifoldMeshGL64* manifold_meshgl64_w_options(
    const double* vert_props, size_t n_verts, size_t n_props,
    const uint64_t* tri_verts, size_t n_tris,
    const ManifoldMeshGL64Options* options) {
  return to_c(meshgl_w_options<MeshGL64>(vert_props, n_verts, n_props,
                                         tri_verts, n_tris, options));
}

ManifoldManifold* manifold_of_meshgl(const ManifoldMeshGL* mesh) {
  auto m = Manifold(*from_c(mesh));
  return to_c(new Manifold(m));
}

ManifoldManifold* manifold_of_meshgl64(const ManifoldMeshGL64* mesh) {
  auto m = Manifold(*from_c(mesh));
  return to_c(new Manifold(m));
}

ManifoldManifold* manifold_extrude(const ManifoldPolygons* cs, double height,
                                   int slices, double twist_degrees,
                                   double scale_x, double scale_y) {
  auto scale = vec2(scale_x, scale_y);
  auto m = Manifold::Extrude(*from_c(cs), height, slices, twist_degrees, scale);
  return to_c(new Manifold(m));
}

ManifoldManifold* manifold_revolve(const ManifoldPolygons* cs,
                                   const ManifoldCircularQuality* quality,
                                   double revolve_degrees) {
  auto m = Manifold::Revolve(*from_c(cs), from_c(quality), revolve_degrees);
  return to_c(new Manifold(m));
}

ManifoldManifoldVec* manifold_decompose(const ManifoldManifold* m) {
  return to_c(new std::vector<Manifold>(from_c(m)->Decompose()));
}

ManifoldMeshGL* manifold_get_meshgl(const ManifoldManifold* m) {
  return to_c(new MeshGL(from_c(m)->GetMeshGL()));
}

ManifoldMeshGL* manifold_meshgl_copy(const ManifoldMeshGL* m) {
  return to_c(new MeshGL(*from_c(m)));
}

ManifoldMeshGL* manifold_meshgl_merge(const ManifoldMeshGL* m,
                                      float tolerance) {
  return to_c(new MeshGL(from_c(m)->Merge(tolerance)));
}

ManifoldMeshGL64* manifold_get_meshgl64(const ManifoldManifold* m) {
  return to_c(new MeshGL64(from_c(m)->GetMeshGL64()));
}

ManifoldMeshGL64* manifold_meshgl64_copy(const ManifoldMeshGL64* m) {
  return to_c(new MeshGL64(*from_c(m)));
}

ManifoldMeshGL64* manifold_meshgl64_merge(const ManifoldMeshGL64* m,
                                          double tolerance) {
  return to_c(new MeshGL64(from_c(m)->Merge(tolerance)));
}

size_t manifold_meshgl_num_prop(const ManifoldMeshGL* m) {
  return from_c(m)->numProp;
}
size_t manifold_meshgl_num_vert(const ManifoldMeshGL* m) {
  return from_c(m)->NumVert();
}
size_t manifold_meshgl_num_tri(const ManifoldMeshGL* m) {
  return from_c(m)->NumTri();
}

size_t manifold_meshgl_vert_properties_length(const ManifoldMeshGL* m) {
  return from_c(m)->vertProperties.size();
}

size_t manifold_meshgl_tri_length(const ManifoldMeshGL* m) {
  return from_c(m)->triVerts.size();
}

size_t manifold_meshgl_merge_length(const ManifoldMeshGL* m) {
  return from_c(m)->mergeFromVert.size();
}

size_t manifold_meshgl_run_index_length(const ManifoldMeshGL* m) {
  return from_c(m)->runIndex.size();
}

size_t manifold_meshgl_run_original_id_length(const ManifoldMeshGL* m) {
  return from_c(m)->runOriginalID.size();
}

size_t manifold_meshgl_run_transform_length(const ManifoldMeshGL* m) {
  return from_c(m)->runTransform.size();
}

size_t manifold_meshgl_face_id_length(const ManifoldMeshGL* m) {
  return from_c(m)->faceID.size();
}

size_t manifold_meshgl_tangent_length(const ManifoldMeshGL* m) {
  return from_c(m)->halfedgeTangent.size();
}

const float* manifold_meshgl_vert_properties(const ManifoldMeshGL* m) {
  return from_c(m)->vertProperties.data();
}

const uint32_t* manifold_meshgl_tri_verts(const ManifoldMeshGL* m) {
  return from_c(m)->triVerts.data();
}

const uint32_t* manifold_meshgl_merge_from_vert(const ManifoldMeshGL* m) {
  return from_c(m)->mergeFromVert.data();
}

const uint32_t* manifold_meshgl_merge_to_vert(const ManifoldMeshGL* m) {
  return from_c(m)->mergeToVert.data();
}

const uint32_t* manifold_meshgl_run_index(const ManifoldMeshGL* m) {
  return from_c(m)->runIndex.data();
}

const uint32_t* manifold_meshgl_run_original_id(const ManifoldMeshGL* m) {
  return from_c(m)->runOriginalID.data();
}

const float* manifold_meshgl_run_transform(const ManifoldMeshGL* m) {
  return from_c(m)->runTransform.data();
}

const uint32_t* manifold_meshgl_face_id(const ManifoldMeshGL* m) {
  return from_c(m)->faceID.data();
}

const float* manifold_meshgl_halfedge_tangent(const ManifoldMeshGL* m) {
  return from_c(m)->halfedgeTangent.data();
}

size_t manifold_meshgl_run_flags_length(const ManifoldMeshGL* m) {
  return from_c(m)->runFlags.size();
}

const uint8_t* manifold_meshgl_run_flags(const ManifoldMeshGL* m) {
  return from_c(m)->runFlags.data();
}

size_t manifold_meshgl_num_run(const ManifoldMeshGL* m) {
  return from_c(m)->NumRun();
}

int manifold_meshgl_backside(const ManifoldMeshGL* m, size_t run) {
  return from_c(m)->Backside(run) ? 1 : 0;
}

int manifold_meshgl_has_normals(const ManifoldMeshGL* m, size_t run) {
  return from_c(m)->HasNormals(run) ? 1 : 0;
}

size_t manifold_meshgl64_num_prop(const ManifoldMeshGL64* m) {
  return from_c(m)->numProp;
}
size_t manifold_meshgl64_num_vert(const ManifoldMeshGL64* m) {
  return from_c(m)->NumVert();
}
size_t manifold_meshgl64_num_tri(const ManifoldMeshGL64* m) {
  return from_c(m)->NumTri();
}
size_t manifold_meshgl64_vert_properties_length(const ManifoldMeshGL64* m) {
  return from_c(m)->vertProperties.size();
}
size_t manifold_meshgl64_tri_length(const ManifoldMeshGL64* m) {
  return from_c(m)->triVerts.size();
}
size_t manifold_meshgl64_merge_length(const ManifoldMeshGL64* m) {
  return from_c(m)->mergeFromVert.size();
}
size_t manifold_meshgl64_run_index_length(const ManifoldMeshGL64* m) {
  return from_c(m)->runIndex.size();
}
size_t manifold_meshgl64_run_original_id_length(const ManifoldMeshGL64* m) {
  return from_c(m)->runOriginalID.size();
}
size_t manifold_meshgl64_run_transform_length(const ManifoldMeshGL64* m) {
  return from_c(m)->runTransform.size();
}
size_t manifold_meshgl64_face_id_length(const ManifoldMeshGL64* m) {
  return from_c(m)->faceID.size();
}
size_t manifold_meshgl64_tangent_length(const ManifoldMeshGL64* m) {
  return from_c(m)->halfedgeTangent.size();
}
const double* manifold_meshgl64_vert_properties(const ManifoldMeshGL64* m) {
  return from_c(m)->vertProperties.data();
}
const uint64_t* manifold_meshgl64_tri_verts(const ManifoldMeshGL64* m) {
  return from_c(m)->triVerts.data();
}
const uint64_t* manifold_meshgl64_merge_from_vert(const ManifoldMeshGL64* m) {
  return from_c(m)->mergeFromVert.data();
}
const uint64_t* manifold_meshgl64_merge_to_vert(const ManifoldMeshGL64* m) {
  return from_c(m)->mergeToVert.data();
}
const uint64_t* manifold_meshgl64_run_index(const ManifoldMeshGL64* m) {
  return from_c(m)->runIndex.data();
}
const uint32_t* manifold_meshgl64_run_original_id(const ManifoldMeshGL64* m) {
  return from_c(m)->runOriginalID.data();
}
const double* manifold_meshgl64_run_transform(const ManifoldMeshGL64* m) {
  return from_c(m)->runTransform.data();
}
const uint64_t* manifold_meshgl64_face_id(const ManifoldMeshGL64* m) {
  return from_c(m)->faceID.data();
}
const double* manifold_meshgl64_halfedge_tangent(const ManifoldMeshGL64* m) {
  return from_c(m)->halfedgeTangent.data();
}

size_t manifold_meshgl64_run_flags_length(const ManifoldMeshGL64* m) {
  return from_c(m)->runFlags.size();
}

const uint8_t* manifold_meshgl64_run_flags(const ManifoldMeshGL64* m) {
  return from_c(m)->runFlags.data();
}

size_t manifold_meshgl64_num_run(const ManifoldMeshGL64* m) {
  return from_c(m)->NumRun();
}

int manifold_meshgl64_backside(const ManifoldMeshGL64* m, size_t run) {
  return from_c(m)->Backside(run) ? 1 : 0;
}

int manifold_meshgl64_has_normals(const ManifoldMeshGL64* m, size_t run) {
  return from_c(m)->HasNormals(run) ? 1 : 0;
}

ManifoldManifold* manifold_as_original(const ManifoldManifold* m, int id) {
  auto orig = from_c(m)->AsOriginal(id);
  return to_c(new Manifold(orig));
}

int manifold_original_id(const ManifoldManifold* m) {
  return from_c(m)->OriginalID();
}

int manifold_is_empty(const ManifoldManifold* m) {
  return from_c(m)->IsEmpty();
}

ManifoldError manifold_status(const ManifoldManifold* m) {
  auto error = from_c(m)->Status();
  return to_c(error);
}

const char* manifold_error_string(ManifoldError error) {
  return ToString(static_cast<Manifold::Error>(error));
}

ManifoldManifold* manifold_with_context(const ManifoldManifold* m,
                                        const ManifoldExecutionContext* ctx) {
  auto attached = from_c(m)->WithContext(*from_c(ctx));
  return to_c(new Manifold(attached));
}

size_t manifold_num_vert(const ManifoldManifold* m) {
  return from_c(m)->NumVert();
}
size_t manifold_num_edge(const ManifoldManifold* m) {
  return from_c(m)->NumEdge();
}
size_t manifold_num_tri(const ManifoldManifold* m) {
  return from_c(m)->NumTri();
}
size_t manifold_num_prop(const ManifoldManifold* m) {
  return from_c(m)->NumProp();
};
int manifold_genus(const ManifoldManifold* m) { return from_c(m)->Genus(); }

double manifold_surface_area(const ManifoldManifold* m) {
  return from_c(m)->SurfaceArea();
}
double manifold_volume(const ManifoldManifold* m) {
  return from_c(m)->Volume();
}

ManifoldBox* manifold_bounding_box(const ManifoldManifold* m) {
  auto box = from_c(m)->BoundingBox();
  return to_c(new Box(box));
}

double manifold_epsilon(const ManifoldManifold* m) {
  return from_c(m)->GetEpsilon();
}

ManifoldManifold* manifold_with_precision(const ManifoldManifold* m,
                                          double rel_precision) {
  return to_c(new Manifold(from_c(m)->WithPrecision(rel_precision)));
}

double manifold_get_precision(const ManifoldManifold* m) {
  return from_c(m)->GetPrecision();
}

size_t manifold_num_prop_vert(const ManifoldManifold* m) {
  return from_c(m)->NumPropVert();
}

uint32_t manifold_reserve_ids(uint32_t n) { return Manifold::ReserveIDs(n); }

ManifoldManifold* manifold_set_properties(
    const ManifoldManifold* m, int num_prop,
    void (*fun)(double* new_prop, ManifoldVec3 position, const double* old_prop,
                void* ctx),
    void* ctx) {
  if (fun == nullptr) {
    return to_c(new Manifold(from_c(m)->SetProperties(num_prop, nullptr)));
  }
  // Bind function with context argument to one without
  using namespace std::placeholders;
  std::function<void(double*, ManifoldVec3, const double*)> f3 =
      std::bind(fun, _1, _2, _3, ctx);
  std::function<void(double*, vec3, const double*)> f =
      [f3](double* new_prop, vec3 v, const double* old_prop) {
        return (f3(new_prop, to_c(v), old_prop));
      };
  auto man = from_c(m)->SetProperties(num_prop, f);
  return to_c(new Manifold(man));
};

ManifoldManifold* manifold_calculate_curvature(const ManifoldManifold* m,
                                               int gaussian_idx, int mean_idx) {
  auto man = from_c(m)->CalculateCurvature(gaussian_idx, mean_idx);
  return to_c(new Manifold(man));
}

double manifold_min_gap(const ManifoldManifold* m,
                        const ManifoldManifold* other, double search_length) {
  return from_c(m)->MinGap(*from_c(other), search_length);
}

ManifoldRayHitVec* manifold_ray_cast(const ManifoldManifold* m, double origin_x,
                                     double origin_y, double origin_z,
                                     double end_x, double end_y, double end_z) {
  auto hits = from_c(m)->RayCast(vec3(origin_x, origin_y, origin_z),
                                 vec3(end_x, end_y, end_z));
  return to_c(new RayHitVec(std::move(hits)));
}

size_t manifold_ray_hit_vec_length(const ManifoldRayHitVec* v) {
  return from_c(v)->size();
}

ManifoldRayHit manifold_ray_hit_vec_get(const ManifoldRayHitVec* v,
                                        size_t idx) {
  const auto& hit = (*from_c(v))[idx];
  return {hit.originalID,     hit.faceID,       hit.distance,
          to_c(hit.position), to_c(hit.normal), hit.crossing};
}

int manifold_winding_number(const ManifoldManifold* m, double x, double y,
                            double z) {
  const auto result = from_c(m)->WindingNumber({{x, y, z}});
  return result.empty() ? 0 : result[0];
}

ManifoldExecutionContext* manifold_execution_context() {
  return to_c(new ExecutionContext());
}

void manifold_execution_context_cancel(ManifoldExecutionContext* ctx) {
  from_c(ctx)->Cancel();
}

int manifold_execution_context_cancelled(const ManifoldExecutionContext* ctx) {
  return from_c(ctx)->Cancelled() ? 1 : 0;
}

double manifold_execution_context_progress(
    const ManifoldExecutionContext* ctx) {
  return from_c(ctx)->Progress();
}

// ctx-aware static factories: the FromMeshGL / LevelSet ops have no
// source Manifold to attach via manifold_with_context, so they run directly on
// the ExecutionContext. Each mirrors its plain factory but observes progress /
// cancellation. `sdf_context` is the SDF callback's user-data (distinct from
// the ExecutionContext `ec`).
ManifoldManifold* manifold_execution_context_level_set(
    ManifoldExecutionContext* ec, double (*sdf)(double, double, double, void*),
    const ManifoldBox* bounds, double edge_length, double level,
    double tolerance, void* sdf_context) {
  return level_set(sdf, bounds, edge_length, level, tolerance, false,
                   sdf_context, from_c(ec));
}

ManifoldManifold* manifold_execution_context_level_set_seq(
    ManifoldExecutionContext* ec, double (*sdf)(double, double, double, void*),
    const ManifoldBox* bounds, double edge_length, double level,
    double tolerance, void* sdf_context) {
  return level_set(sdf, bounds, edge_length, level, tolerance, true,
                   sdf_context, from_c(ec));
}

ManifoldManifold* manifold_execution_context_of_meshgl(
    ManifoldExecutionContext* ec, const ManifoldMeshGL* mesh) {
  return to_c(new Manifold(from_c(ec)->FromMeshGL(*from_c(mesh))));
}

ManifoldManifold* manifold_execution_context_of_meshgl64(
    ManifoldExecutionContext* ec, const ManifoldMeshGL64* mesh) {
  return to_c(new Manifold(from_c(ec)->FromMeshGL(*from_c(mesh))));
}

ManifoldManifold* manifold_calculate_normals(const ManifoldManifold* m,
                                             double min_sharp_angle) {
  auto man = from_c(m)->CalculateNormals(min_sharp_angle);
  return to_c(new Manifold(man));
}

// Version

void manifold_version(int version[3]) {
  const auto v = manifold::Version();
  version[0] = v[0];
  version[1] = v[1];
  version[2] = v[2];
}

int manifold_circular_quality_segments_for(
    double radius, const ManifoldCircularQuality* quality) {
  return from_c(quality).SegmentsFor(radius);
}

ManifoldCircularQuality manifold_circular_quality_for_divisions(
    int divisions, double degrees) {
  const CircularQuality q = CircularQuality::ForDivisions(divisions, degrees);
  return {q.segments, q.minAngle, q.minEdgeLength};
}

ManifoldTriangulation* manifold_triangulate(const ManifoldPolygons* ps,
                                            double epsilon) {
  auto triangulation = manifold::Triangulate(*from_c(ps), epsilon);
  auto buffer = new std::vector<int>;
  buffer->reserve(triangulation.size() * 3);
  for (int i = 0; i < triangulation.size(); i++)
    for (int j : {0, 1, 2}) buffer->push_back(triangulation[i][j]);
  return to_c(buffer);
}

size_t manifold_triangulation_num_tri(const ManifoldTriangulation* m) {
  return from_c(m)->size() / 3;
}

const int* manifold_triangulation_tri_verts(const ManifoldTriangulation* m) {
  auto& triangulation = *from_c(m);
  return triangulation.data();
}

// pointer free + destruction
void manifold_delete_cross_section(ManifoldCrossSection* c) {
  delete from_c(c);
}
void manifold_delete_cross_section_vec(ManifoldCrossSectionVec* csv) {
  delete from_c(csv);
}
void manifold_delete_ray_hit_vec(ManifoldRayHitVec* v) { delete from_c(v); }
void manifold_delete_simple_polygon(ManifoldSimplePolygon* p) {
  delete from_c(p);
}
void manifold_delete_polygons(ManifoldPolygons* p) { delete from_c(p); }
void manifold_delete_manifold(ManifoldManifold* m) { delete from_c(m); }
void manifold_delete_manifold_vec(ManifoldManifoldVec* ms) {
  delete from_c(ms);
}
void manifold_delete_meshgl(ManifoldMeshGL* m) { delete from_c(m); }
void manifold_delete_meshgl64(ManifoldMeshGL64* m) { delete from_c(m); }
void manifold_delete_box(ManifoldBox* b) { delete from_c(b); }
void manifold_delete_rect(ManifoldRect* r) { delete from_c(r); }
void manifold_delete_triangulation(ManifoldTriangulation* m) {
  delete from_c(m);
}
void manifold_delete_execution_context(ManifoldExecutionContext* ctx) {
  delete from_c(ctx);
}

// IO

#ifndef MANIFOLD_NO_IOSTREAM
ManifoldManifold* manifold_read_obj(const char* obj_file) {
  std::istringstream iss(obj_file);
  Manifold m = Manifold::ReadOBJ(iss);
  return to_c(new Manifold(m));
}

ManifoldMeshGL64* manifold_meshgl64_read_obj(const char* obj_file) {
  std::istringstream iss(obj_file);
  MeshGL64 m = ReadOBJ(iss);
  return to_c(new MeshGL64(std::move(m)));
}

void manifold_write_obj(const ManifoldManifold* manifold,
                        void (*callback)(char*, void*), void* args) {
  std::stringstream ss;
  const Manifold* m = from_c(manifold);
  m->WriteOBJ(ss);
  callback(ss.str().data(), args);
}

void manifold_meshgl64_write_obj(const ManifoldMeshGL64* mesh,
                                 void (*callback)(char*, void*), void* args) {
  std::stringstream ss;
  const MeshGL64* m = from_c(mesh);
  WriteOBJ(ss, *m);
  callback(ss.str().data(), args);
}
#endif

#ifdef __cplusplus
}
#endif
