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

#include <vector>

#include "./conv.h"
#include "manifold/common.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"
#include "manifold/types.h"

using namespace manifold;

namespace {
ManifoldManifold *level_set(
    void *mem, double (*sdf_context)(double, double, double, void *),
    ManifoldBox *bounds, double edge_length, double level, double precision,
    bool seq, void *ctx) {
  // Bind function with context argument to one without
  using namespace std::placeholders;
  std::function<double(double, double, double)> sdf =
      std::bind(sdf_context, _1, _2, _3, ctx);
  std::function<double(vec3)> fun = [sdf](vec3 v) {
    return (sdf(v.x, v.y, v.z));
  };
  return to_c(new (mem) Manifold(Manifold::LevelSet(
      fun, *from_c(bounds), edge_length, level, precision, !seq)));
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

ManifoldSimplePolygon *manifold_simple_polygon(void *mem, ManifoldVec2 *ps,
                                               size_t length) {
  auto vec = new (mem) std::vector<vec2>;
  for (size_t i = 0; i < length; ++i) {
    vec->push_back({ps[i].x, ps[i].y});
  }
  return to_c(vec);
}

ManifoldPolygons *manifold_polygons(void *mem, ManifoldSimplePolygon **ps,
                                    size_t length) {
  auto vec = new (mem) std::vector<SimplePolygon>;
  auto polys = reinterpret_cast<SimplePolygon **>(ps);
  for (size_t i = 0; i < length; ++i) {
    vec->push_back(*polys[i]);
  }
  return to_c(vec);
}

size_t manifold_simple_polygon_length(ManifoldSimplePolygon *p) {
  return from_c(p)->size();
}

size_t manifold_polygons_length(ManifoldPolygons *ps) {
  return from_c(ps)->size();
}

size_t manifold_polygons_simple_length(ManifoldPolygons *ps, int idx) {
  return (*from_c(ps))[idx].size();
}

ManifoldVec2 manifold_simple_polygon_get_point(ManifoldSimplePolygon *p,
                                               int idx) {
  return to_c((*from_c(p))[idx]);
}

ManifoldSimplePolygon *manifold_polygons_get_simple(void *mem,
                                                    ManifoldPolygons *ps,
                                                    int idx) {
  auto sp = (*from_c(ps))[idx];
  return to_c(new (mem) SimplePolygon(sp));
}

ManifoldVec2 manifold_polygons_get_point(ManifoldPolygons *ps, int simple_idx,
                                         int pt_idx) {
  return to_c((*from_c(ps))[simple_idx][pt_idx]);
}

ManifoldManifoldVec *manifold_manifold_empty_vec(void *mem) {
  return to_c(new (mem) ManifoldVec());
}

ManifoldManifoldVec *manifold_manifold_vec(void *mem, size_t sz) {
  return to_c(new (mem) ManifoldVec(sz));
}

void manifold_manifold_vec_reserve(ManifoldManifoldVec *ms, size_t sz) {
  from_c(ms)->reserve(sz);
}

size_t manifold_manifold_vec_length(ManifoldManifoldVec *ms) {
  return from_c(ms)->size();
}

ManifoldManifold *manifold_manifold_vec_get(void *mem, ManifoldManifoldVec *ms,
                                            int idx) {
  auto m = (*from_c(ms))[idx];
  return to_c(new (mem) Manifold(m));
}

void manifold_manifold_vec_set(ManifoldManifoldVec *ms, int idx,
                               ManifoldManifold *m) {
  (*from_c(ms))[idx] = *from_c(m);
}

void manifold_manifold_vec_push_back(ManifoldManifoldVec *ms,
                                     ManifoldManifold *m) {
  return from_c(ms)->push_back(*from_c(m));
}

ManifoldManifold *manifold_boolean(void *mem, ManifoldManifold *a,
                                   ManifoldManifold *b, ManifoldOpType op) {
  auto m = from_c(a)->Boolean(*from_c(b), from_c(op));
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_batch_boolean(void *mem, ManifoldManifoldVec *ms,
                                         ManifoldOpType op) {
  auto m = Manifold::BatchBoolean(*from_c(ms), from_c(op));
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_union(void *mem, ManifoldManifold *a,
                                 ManifoldManifold *b) {
  auto m = (*from_c(a)) + (*from_c(b));
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_difference(void *mem, ManifoldManifold *a,
                                      ManifoldManifold *b) {
  auto m = (*from_c(a)) - (*from_c(b));
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_intersection(void *mem, ManifoldManifold *a,
                                        ManifoldManifold *b) {
  auto m = (*from_c(a)) ^ (*from_c(b));
  return to_c(new (mem) Manifold(m));
}

ManifoldManifoldPair manifold_split(void *mem_first, void *mem_second,
                                    ManifoldManifold *a, ManifoldManifold *b) {
  auto pair = from_c(a)->Split(*from_c(b));
  auto first = new (mem_first) Manifold(pair.first);
  auto second = new (mem_second) Manifold(pair.second);
  return {to_c(first), to_c(second)};
}

ManifoldManifoldPair manifold_split_by_plane(void *mem_first, void *mem_second,
                                             ManifoldManifold *m,
                                             double normal_x, double normal_y,
                                             double normal_z, double offset) {
  auto normal = vec3(normal_x, normal_y, normal_z);
  auto pair = from_c(m)->SplitByPlane(normal, offset);
  auto first = new (mem_first) Manifold(pair.first);
  auto second = new (mem_second) Manifold(pair.second);
  return {to_c(first), to_c(second)};
}

ManifoldManifold *manifold_trim_by_plane(void *mem, ManifoldManifold *m,
                                         double normal_x, double normal_y,
                                         double normal_z, double offset) {
  auto normal = vec3(normal_x, normal_y, normal_z);
  auto trimmed = from_c(m)->TrimByPlane(normal, offset);
  return to_c(new (mem) Manifold(trimmed));
}

ManifoldPolygons *manifold_slice(void *mem, ManifoldManifold *m,
                                 double height) {
  auto poly = from_c(m)->Slice(height);
  return to_c(new (mem) Polygons(poly));
}

ManifoldPolygons *manifold_project(void *mem, ManifoldManifold *m) {
  auto poly = from_c(m)->Project();
  return to_c(new (mem) Polygons(poly));
}

ManifoldManifold *manifold_hull(void *mem, ManifoldManifold *m) {
  auto hulled = from_c(m)->Hull();
  return to_c(new (mem) Manifold(hulled));
}

ManifoldManifold *manifold_batch_hull(void *mem, ManifoldManifoldVec *ms) {
  auto hulled = Manifold::Hull(*from_c(ms));
  return to_c(new (mem) Manifold(hulled));
}

ManifoldManifold *manifold_hull_pts(void *mem, ManifoldVec3 *ps,
                                    size_t length) {
  std::vector<vec3> vec(length);
  for (size_t i = 0; i < length; ++i) {
    vec[i] = {ps[i].x, ps[i].y, ps[i].z};
  }
  auto hulled = Manifold::Hull(vec);
  return to_c(new (mem) Manifold(hulled));
}

ManifoldManifold *manifold_translate(void *mem, ManifoldManifold *m, double x,
                                     double y, double z) {
  auto v = vec3(x, y, z);
  auto translated = from_c(m)->Translate(v);
  return to_c(new (mem) Manifold(translated));
}

ManifoldManifold *manifold_rotate(void *mem, ManifoldManifold *m, double x,
                                  double y, double z) {
  auto rotated = from_c(m)->Rotate(x, y, z);
  return to_c(new (mem) Manifold(rotated));
}

ManifoldManifold *manifold_scale(void *mem, ManifoldManifold *m, double x,
                                 double y, double z) {
  auto s = vec3(x, y, z);
  auto scaled = from_c(m)->Scale(s);
  return to_c(new (mem) Manifold(scaled));
}

ManifoldManifold *manifold_transform(void *mem, ManifoldManifold *m, double x1,
                                     double y1, double z1, double x2, double y2,
                                     double z2, double x3, double y3, double z3,
                                     double x4, double y4, double z4) {
  auto mat = mat4x3(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4);
  auto transformed = from_c(m)->Transform(mat);
  return to_c(new (mem) Manifold(transformed));
}

ManifoldManifold *manifold_mirror(void *mem, ManifoldManifold *m, double nx,
                                  double ny, double nz) {
  auto mirrored = from_c(m)->Mirror({nx, ny, nz});
  return to_c(new (mem) Manifold(mirrored));
}

ManifoldManifold *manifold_warp(void *mem, ManifoldManifold *m,
                                ManifoldVec3 (*fun)(double, double, double,
                                                    void *),
                                void *ctx) {
  // Bind function with context argument to one without
  using namespace std::placeholders;
  std::function<ManifoldVec3(double, double, double)> f3 =
      std::bind(fun, _1, _2, _3, ctx);
  std::function<void(vec3 & v)> warp = [f3](vec3 &v) {
    v = from_c(f3(v.x, v.y, v.z));
  };
  auto warped = from_c(m)->Warp(warp);
  return to_c(new (mem) Manifold(warped));
}

ManifoldManifold *manifold_level_set(
    void *mem, double (*sdf)(double, double, double, void *),
    ManifoldBox *bounds, double edge_length, double level, double precision,
    void *ctx) {
  return level_set(mem, sdf, bounds, edge_length, level, precision, false, ctx);
}

ManifoldManifold *manifold_level_set_seq(
    void *mem, double (*sdf)(double, double, double, void *),
    ManifoldBox *bounds, double edge_length, double level, double precision,
    void *ctx) {
  return level_set(mem, sdf, bounds, edge_length, level, precision, true, ctx);
}

ManifoldManifold *manifold_smooth_by_normals(void *mem, ManifoldManifold *m,
                                             int normalIdx) {
  auto smoothed = from_c(m)->SmoothByNormals(normalIdx);
  return to_c(new (mem) Manifold(smoothed));
}

ManifoldManifold *manifold_smooth_out(void *mem, ManifoldManifold *m,
                                      double minSharpAngle,
                                      double minSmoothness) {
  auto smoothed = from_c(m)->SmoothOut(minSharpAngle, minSmoothness);
  return to_c(new (mem) Manifold(smoothed));
}

ManifoldManifold *manifold_refine(void *mem, ManifoldManifold *m, int refine) {
  auto refined = from_c(m)->Refine(refine);
  return to_c(new (mem) Manifold(refined));
}

ManifoldManifold *manifold_refine_to_length(void *mem, ManifoldManifold *m,
                                            double length) {
  auto refined = from_c(m)->RefineToLength(length);
  return to_c(new (mem) Manifold(refined));
}

ManifoldManifold *manifold_refine_to_precision(void *mem, ManifoldManifold *m,
                                               double precision) {
  auto refined = from_c(m)->RefineToPrecision(precision);
  return to_c(new (mem) Manifold(refined));
}

ManifoldManifold *manifold_empty(void *mem) {
  return to_c(new (mem) Manifold());
}

ManifoldManifold *manifold_copy(void *mem, ManifoldManifold *m) {
  return to_c(new (mem) Manifold(*from_c(m)));
}

ManifoldManifold *manifold_tetrahedron(void *mem) {
  auto m = Manifold::Tetrahedron();
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_cube(void *mem, double x, double y, double z,
                                int center) {
  auto size = vec3(x, y, z);
  auto m = Manifold::Cube(size, center);
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_cylinder(void *mem, double height, double radius_low,
                                    double radius_high, int circular_segments,
                                    int center) {
  auto m = Manifold::Cylinder(height, radius_low, radius_high,
                              circular_segments, center);
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_sphere(void *mem, double radius,
                                  int circular_segments) {
  auto m = Manifold::Sphere(radius, circular_segments);
  return to_c(new (mem) Manifold(m));
}

ManifoldMeshGL *manifold_meshgl(void *mem, float *vert_props, size_t n_verts,
                                size_t n_props, uint32_t *tri_verts,
                                size_t n_tris) {
  auto mesh = new (mem) MeshGL();
  mesh->numProp = n_props;
  mesh->vertProperties = vector_of_array(vert_props, n_verts * n_props);
  mesh->triVerts = vector_of_array(tri_verts, n_tris * 3);
  return to_c(mesh);
}

ManifoldMeshGL *manifold_meshgl_w_tangents(void *mem, float *vert_props,
                                           size_t n_verts, size_t n_props,
                                           uint32_t *tri_verts, size_t n_tris,
                                           float *halfedge_tangent) {
  auto mesh = new (mem) MeshGL();
  mesh->numProp = n_props;
  mesh->vertProperties = vector_of_array(vert_props, n_verts * n_props);
  mesh->triVerts = vector_of_array(tri_verts, n_tris * 3);
  mesh->halfedgeTangent = vector_of_array(halfedge_tangent, n_tris * 3 * 4);
  return to_c(mesh);
}

ManifoldManifold *manifold_smooth(void *mem, ManifoldMeshGL *mesh,
                                  size_t *half_edges, double *smoothness,
                                  size_t n_edges) {
  auto smooth = std::vector<Smoothness>();
  for (size_t i = 0; i < n_edges; ++i) {
    smooth.push_back({half_edges[i], smoothness[i]});
  }
  auto m = Manifold::Smooth(*from_c(mesh), smooth);
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_of_meshgl(void *mem, ManifoldMeshGL *mesh) {
  auto m = Manifold(*from_c(mesh));
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_extrude(void *mem, ManifoldPolygons *cs,
                                   double height, int slices,
                                   double twist_degrees, double scale_x,
                                   double scale_y) {
  auto scale = vec2(scale_x, scale_y);
  auto m = Manifold::Extrude(*from_c(cs), height, slices, twist_degrees, scale);
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_revolve(void *mem, ManifoldPolygons *cs,
                                   int circular_segments) {
  auto m = Manifold::Revolve(*from_c(cs), circular_segments);
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_compose(void *mem, ManifoldManifoldVec *ms) {
  auto composed = Manifold::Compose(*from_c(ms));
  return to_c(new (mem) Manifold(composed));
}

ManifoldManifoldVec *manifold_decompose(void *mem, ManifoldManifold *m) {
  auto comps = from_c(m)->Decompose();
  return to_c(new (mem) std::vector<Manifold>(comps));
}

ManifoldMeshGL *manifold_get_meshgl(void *mem, ManifoldManifold *m) {
  auto mesh = from_c(m)->GetMeshGL();
  return to_c(new (mem) MeshGL(mesh));
}

ManifoldMeshGL *manifold_meshgl_copy(void *mem, ManifoldMeshGL *m) {
  return to_c(new (mem) MeshGL(*from_c(m)));
}

ManifoldMeshGL *manifold_meshgl_merge(void *mem, ManifoldMeshGL *m) {
  auto duplicate = new (mem) MeshGL(*from_c(m));
  if (duplicate->Merge()) {
    return to_c(duplicate);
  }
  delete duplicate;
  return m;
}

int manifold_meshgl_num_prop(ManifoldMeshGL *m) { return from_c(m)->numProp; }
int manifold_meshgl_num_vert(ManifoldMeshGL *m) { return from_c(m)->NumVert(); }
int manifold_meshgl_num_tri(ManifoldMeshGL *m) { return from_c(m)->NumTri(); }

size_t manifold_meshgl_vert_properties_length(ManifoldMeshGL *m) {
  return from_c(m)->vertProperties.size();
}

size_t manifold_meshgl_tri_length(ManifoldMeshGL *m) {
  return from_c(m)->triVerts.size();
}

size_t manifold_meshgl_merge_length(ManifoldMeshGL *m) {
  return from_c(m)->mergeFromVert.size();
}

size_t manifold_meshgl_run_index_length(ManifoldMeshGL *m) {
  return from_c(m)->runIndex.size();
}

size_t manifold_meshgl_run_original_id_length(ManifoldMeshGL *m) {
  return from_c(m)->runOriginalID.size();
}

size_t manifold_meshgl_run_transform_length(ManifoldMeshGL *m) {
  return from_c(m)->runTransform.size();
}

size_t manifold_meshgl_face_id_length(ManifoldMeshGL *m) {
  return from_c(m)->faceID.size();
}

size_t manifold_meshgl_tangent_length(ManifoldMeshGL *m) {
  return from_c(m)->halfedgeTangent.size();
}

float *manifold_meshgl_vert_properties(void *mem, ManifoldMeshGL *m) {
  return copy_data(mem, from_c(m)->vertProperties);
}

uint32_t *manifold_meshgl_tri_verts(void *mem, ManifoldMeshGL *m) {
  return copy_data(mem, from_c(m)->triVerts);
}

uint32_t *manifold_meshgl_merge_from_vert(void *mem, ManifoldMeshGL *m) {
  return copy_data(mem, from_c(m)->mergeFromVert);
}

uint32_t *manifold_meshgl_merge_to_vert(void *mem, ManifoldMeshGL *m) {
  return copy_data(mem, from_c(m)->mergeToVert);
}

uint32_t *manifold_meshgl_run_index(void *mem, ManifoldMeshGL *m) {
  return copy_data(mem, from_c(m)->runIndex);
}

uint32_t *manifold_meshgl_run_original_id(void *mem, ManifoldMeshGL *m) {
  return copy_data(mem, from_c(m)->runOriginalID);
}

float *manifold_meshgl_run_transform(void *mem, ManifoldMeshGL *m) {
  return copy_data(mem, from_c(m)->runTransform);
}

uint32_t *manifold_meshgl_face_id(void *mem, ManifoldMeshGL *m) {
  return copy_data(mem, from_c(m)->faceID);
}

float *manifold_meshgl_halfedge_tangent(void *mem, ManifoldMeshGL *m) {
  return copy_data(mem, from_c(m)->halfedgeTangent);
}

ManifoldManifold *manifold_as_original(void *mem, ManifoldManifold *m) {
  auto orig = from_c(m)->AsOriginal();
  return to_c(new (mem) Manifold(orig));
}

int manifold_original_id(ManifoldManifold *m) {
  return from_c(m)->OriginalID();
}

int manifold_is_empty(ManifoldManifold *m) { return from_c(m)->IsEmpty(); }

ManifoldError manifold_status(ManifoldManifold *m) {
  auto error = from_c(m)->Status();
  return to_c(error);
}

size_t manifold_num_vert(ManifoldManifold *m) { return from_c(m)->NumVert(); }
size_t manifold_num_edge(ManifoldManifold *m) { return from_c(m)->NumEdge(); }
size_t manifold_num_tri(ManifoldManifold *m) { return from_c(m)->NumTri(); }
int manifold_genus(ManifoldManifold *m) { return from_c(m)->Genus(); }

ManifoldProperties manifold_get_properties(ManifoldManifold *m) {
  return to_c(from_c(m)->GetProperties());
}

ManifoldBox *manifold_bounding_box(void *mem, ManifoldManifold *m) {
  auto box = from_c(m)->BoundingBox();
  return to_c(new (mem) Box(box));
}

double manifold_precision(ManifoldManifold *m) {
  return from_c(m)->Precision();
}

uint32_t manifold_reserve_ids(uint32_t n) { return Manifold::ReserveIDs(n); }

ManifoldManifold *manifold_set_properties(
    void *mem, ManifoldManifold *m, int num_prop,
    void (*fun)(double *new_prop, ManifoldVec3 position, const double *old_prop,
                void *ctx),
    void *ctx) {
  // Bind function with context argument to one without
  using namespace std::placeholders;
  std::function<void(double *, ManifoldVec3, const double *)> f3 =
      std::bind(fun, _1, _2, _3, ctx);
  std::function<void(double *, vec3, const double *)> f =
      [f3](double *new_prop, vec3 v, const double *old_prop) {
        return (f3(new_prop, to_c(v), old_prop));
      };
  auto man = from_c(m)->SetProperties(num_prop, f);
  return to_c(new (mem) Manifold(man));
};

ManifoldManifold *manifold_calculate_curvature(void *mem, ManifoldManifold *m,
                                               int gaussian_idx, int mean_idx) {
  auto man = from_c(m)->CalculateCurvature(gaussian_idx, mean_idx);
  return to_c(new (mem) Manifold(man));
}

double manifold_min_gap(ManifoldManifold *m, ManifoldManifold *other,
                        double searchLength) {
  return from_c(m)->MinGap(*from_c(other), searchLength);
}

ManifoldManifold *manifold_calculate_normals(void *mem, ManifoldManifold *m,
                                             int normal_idx,
                                             int min_sharp_angle) {
  auto man = from_c(m)->CalculateNormals(normal_idx, min_sharp_angle);
  return to_c(new (mem) Manifold(man));
}

// Static Quality Globals

void manifold_set_min_circular_angle(double degrees) {
  Quality::SetMinCircularAngle(degrees);
}

void manifold_set_min_circular_edge_length(double length) {
  Quality::SetMinCircularEdgeLength(length);
}

void manifold_set_circular_segments(int number) {
  Quality::SetCircularSegments(number);
}

int manifold_get_circular_segments(double radius) {
  return Quality::GetCircularSegments(radius);
}

void manifold_reset_to_circular_defaults() { Quality::ResetToDefaults(); }

// memory size
size_t manifold_cross_section_size() { return sizeof(CrossSection); }
size_t manifold_cross_section_vec_size() {
  return sizeof(std::vector<CrossSection>);
}
size_t manifold_simple_polygon_size() { return sizeof(SimplePolygon); }
size_t manifold_polygons_size() { return sizeof(Polygons); }
size_t manifold_manifold_size() { return sizeof(Manifold); }
size_t manifold_manifold_vec_size() { return sizeof(std::vector<Manifold>); }
size_t manifold_manifold_pair_size() { return sizeof(ManifoldManifoldPair); }
size_t manifold_meshgl_size() { return sizeof(MeshGL); }
size_t manifold_box_size() { return sizeof(Box); }
size_t manifold_rect_size() { return sizeof(Rect); }

// pointer free + destruction
void manifold_delete_cross_section(ManifoldCrossSection *c) {
  delete from_c(c);
}
void manifold_delete_cross_section_vec(ManifoldCrossSectionVec *csv) {
  delete from_c(csv);
}
void manifold_delete_simple_polygon(ManifoldSimplePolygon *p) {
  delete from_c(p);
}
void manifold_delete_polygons(ManifoldPolygons *p) { delete from_c(p); }
void manifold_delete_manifold(ManifoldManifold *m) { delete from_c(m); }
void manifold_delete_manifold_vec(ManifoldManifoldVec *ms) {
  delete from_c(ms);
}
void manifold_delete_meshgl(ManifoldMeshGL *m) { delete from_c(m); }
void manifold_delete_box(ManifoldBox *b) { delete from_c(b); }
void manifold_delete_rect(ManifoldRect *r) { delete from_c(r); }

// destruction
void manifold_destruct_cross_section(ManifoldCrossSection *cs) {
  from_c(cs)->~CrossSection();
}
void manifold_destruct_cross_section_vec(ManifoldCrossSectionVec *csv) {
  from_c(csv)->~CrossSectionVec();
}
void manifold_destruct_simple_polygon(ManifoldSimplePolygon *p) {
  from_c(p)->~SimplePolygon();
}
void manifold_destruct_polygons(ManifoldPolygons *p) { from_c(p)->~Polygons(); }
void manifold_destruct_manifold(ManifoldManifold *m) { from_c(m)->~Manifold(); }
void manifold_destruct_manifold_vec(ManifoldManifoldVec *ms) {
  from_c(ms)->~ManifoldVec();
}
void manifold_destruct_meshgl(ManifoldMeshGL *m) { from_c(m)->~MeshGL(); }
void manifold_destruct_box(ManifoldBox *b) { from_c(b)->~Box(); }
void manifold_destruct_rect(ManifoldRect *r) { from_c(r)->~Rect(); }

#ifdef __cplusplus
}
#endif
