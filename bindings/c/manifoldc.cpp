#include <conv.h>
#include <manifold.h>
#include <manifoldc.h>
#include <meshIO.h>
#include <public.h>
#include <sdf.h>

#include "box.cpp"
#include "include/conv.h"
#include "include/types.h"
#include "meshio.cpp"
#include "types.h"

using namespace manifold;

namespace {
ManifoldMeshGL *level_set(void *mem, float (*sdf)(float, float, float),
                          ManifoldBox *bounds, float edge_length, float level,
                          bool seq) {
  // typing with std::function rather than auto compiles when CUDA is on,
  // passing it into GPU (and crashing) is avoided dynamically in `sdf.h`
  std::function<float(glm::vec3)> fun = [sdf](glm::vec3 v) {
    return (sdf(v.x, v.y, v.z));
  };
  auto pol = seq ? std::make_optional(Seq) : std::nullopt;
  auto mesh = LevelSet(fun, *from_c(bounds), edge_length, level, pol);
  return to_c(new (mem) MeshGL(mesh));
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

ManifoldSimplePolygon *manifold_simple_polygon(void *mem, ManifoldVec2 *ps,
                                               size_t length) {
  auto vec = new (mem) std::vector<PolyVert>;
  for (int i = 0; i < length; ++i) {
    vec->push_back({{ps[i].x, ps[i].y}, i});
  }
  return to_c(vec);
}

ManifoldPolygons *manifold_polygons(void *mem, ManifoldSimplePolygon **ps,
                                    size_t length) {
  auto vec = new (mem) std::vector<SimplePolygon>;
  auto polys = reinterpret_cast<SimplePolygon **>(ps);
  for (int i = 0; i < length; ++i) {
    vec->push_back(*polys[i]);
  }
  return to_c(vec);
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
                                             float normal_x, float normal_y,
                                             float normal_z, float offset) {
  auto normal = glm::vec3(normal_x, normal_y, normal_z);
  auto pair = from_c(m)->SplitByPlane(normal, offset);
  auto first = new (mem_first) Manifold(pair.first);
  auto second = new (mem_second) Manifold(pair.second);
  return {to_c(first), to_c(second)};
}

ManifoldManifold *manifold_trim_by_plane(void *mem, ManifoldManifold *m,
                                         float normal_x, float normal_y,
                                         float normal_z, float offset) {
  auto normal = glm::vec3(normal_x, normal_y, normal_z);
  auto trimmed = from_c(m)->TrimByPlane(normal, offset);
  return to_c(new (mem) Manifold(trimmed));
}

ManifoldManifold *manifold_translate(void *mem, ManifoldManifold *m, float x,
                                     float y, float z) {
  auto v = glm::vec3(x, y, z);
  auto translated = from_c(m)->Translate(v);
  return to_c(new (mem) Manifold(translated));
}

ManifoldManifold *manifold_rotate(void *mem, ManifoldManifold *m, float x,
                                  float y, float z) {
  auto rotated = from_c(m)->Rotate(x, y, z);
  return to_c(new (mem) Manifold(rotated));
}

ManifoldManifold *manifold_scale(void *mem, ManifoldManifold *m, float x,
                                 float y, float z) {
  auto s = glm::vec3(x, y, z);
  auto scaled = from_c(m)->Scale(s);
  return to_c(new (mem) Manifold(scaled));
}

ManifoldManifold *manifold_transform(void *mem, ManifoldManifold *m, float x1,
                                     float y1, float z1, float x2, float y2,
                                     float z2, float x3, float y3, float z3,
                                     float x4, float y4, float z4) {
  auto mat = glm::mat4x3(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4);
  auto transformed = from_c(m)->Transform(mat);
  return to_c(new (mem) Manifold(transformed));
}

ManifoldManifold *manifold_warp(void *mem, ManifoldManifold *m,
                                ManifoldVec3 (*fun)(float, float, float)) {
  std::function<void(glm::vec3 & v)> warp = [fun](glm::vec3 &v) {
    v = from_c(fun(v.x, v.y, v.z));
  };
  auto warped = from_c(m)->Warp(warp);
  return to_c(new (mem) Manifold(warped));
}

ManifoldMeshGL *manifold_level_set(void *mem, float (*sdf)(float, float, float),
                                   ManifoldBox *bounds, float edge_length,
                                   float level) {
  return level_set(mem, sdf, bounds, edge_length, level, false);
}

ManifoldMeshGL *manifold_level_set_seq(void *mem,
                                       float (*sdf)(float, float, float),
                                       ManifoldBox *bounds, float edge_length,
                                       float level) {
  return level_set(mem, sdf, bounds, edge_length, level, true);
}

ManifoldManifold *manifold_refine(void *mem, ManifoldManifold *m, int refine) {
  auto refined = from_c(m)->Refine(refine);
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

ManifoldManifold *manifold_cube(void *mem, float x, float y, float z,
                                int center) {
  auto size = glm::vec3(x, y, z);
  auto m = Manifold::Cube(size, center);
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_cylinder(void *mem, float height, float radius_low,
                                    float radius_high, int circular_segments,
                                    int center) {
  auto m = Manifold::Cylinder(height, radius_low, radius_high,
                              circular_segments, center);
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_sphere(void *mem, float radius,
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
                                  int *half_edges, float *smoothness,
                                  int n_edges) {
  auto smooth = std::vector<Smoothness>();
  for (int i = 0; i < n_edges; ++i) {
    smooth.push_back({half_edges[i], smoothness[i]});
  }
  auto m = Manifold::Smooth(*from_c(mesh), smooth);
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_of_meshgl(void *mem, ManifoldMeshGL *mesh) {
  auto m = Manifold(*from_c(mesh));
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_extrude(void *mem, ManifoldPolygons *polygons,
                                   float height, int slices,
                                   float twist_degrees, float scale_x,
                                   float scale_y) {
  auto scale = glm::vec2(scale_x, scale_y);
  auto m = Manifold::Extrude(*from_c(polygons), height, slices, twist_degrees,
                             scale);
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_revolve(void *mem, ManifoldPolygons *polygons,

                                   int circular_segments) {
  auto m = Manifold::Revolve(*from_c(polygons), circular_segments);
  return to_c(new (mem) Manifold(m));
}

ManifoldManifold *manifold_compose(void *mem, ManifoldManifold **ms,
                                   size_t length) {
  auto vec = std::vector<Manifold>();
  auto manifolds = reinterpret_cast<Manifold **>(ms);
  for (int i = 0; i < length; ++i) {
    vec.push_back(*manifolds[i]);
  }
  auto composed = Manifold::Compose(vec);
  return to_c(new (mem) Manifold(composed));
}

ManifoldComponents *manifold_get_components(void *mem, ManifoldManifold *m) {
  return to_c(new (mem) Components(from_c(m)->GetComponents()));
}

size_t manifold_components_length(ManifoldComponents *components) {
  return from_c(components)->numComponents;
}

ManifoldManifold **manifold_decompose(void **mem, ManifoldManifold *m,
                                      ManifoldComponents *cs) {
  auto components = *from_c(cs);
  auto manifolds = from_c(m)->Decompose(components);
  ManifoldManifold **ms = reinterpret_cast<ManifoldManifold **>(mem);
  for (int i = 0; i < components.numComponents; ++i) {
    ms[i] = to_c(new (mem[i]) Manifold(manifolds[i]));
  }
  return ms;
}

ManifoldMeshGL *manifold_get_meshgl(void *mem, ManifoldManifold *m) {
  auto mesh = from_c(m)->GetMeshGL();
  return to_c(new (mem) MeshGL(mesh));
}

ManifoldMeshGL *manifold_meshgl_copy(void *mem, ManifoldMeshGL *m) {
  return to_c(new (mem) MeshGL(*from_c(m)));
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

size_t manifold_meshgl_original_id_length(ManifoldMeshGL *m) {
  return from_c(m)->originalID.size();
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

uint32_t *manifold_meshgl_original_id(void *mem, ManifoldMeshGL *m) {
  return copy_data(mem, from_c(m)->originalID);
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

int manifold_num_vert(ManifoldManifold *m) { return from_c(m)->NumVert(); }
int manifold_num_edge(ManifoldManifold *m) { return from_c(m)->NumEdge(); }
int manifold_num_tri(ManifoldManifold *m) { return from_c(m)->NumTri(); }
int manifold_genus(ManifoldManifold *m) { return from_c(m)->Genus(); }

ManifoldProperties manifold_get_properties(ManifoldManifold *m) {
  return to_c(from_c(m)->GetProperties());
}

ManifoldBox *manifold_bounding_box(void *mem, ManifoldManifold *m) {
  auto box = from_c(m)->BoundingBox();
  return to_c(new (mem) Box(box));
}

float manifold_precision(ManifoldManifold *m) { return from_c(m)->Precision(); }
ManifoldCurvature *manifold_get_curvature(void *mem, ManifoldManifold *m) {
  auto curv = from_c(m)->GetCurvature();
  return to_c(new (mem) Curvature(curv));
}

ManifoldCurvatureBounds manifold_curvature_bounds(ManifoldCurvature *curv) {
  auto c = *from_c(curv);
  return {c.maxMeanCurvature, c.minMeanCurvature, c.maxGaussianCurvature,
          c.minGaussianCurvature};
}

size_t manifold_curvature_vert_length(ManifoldCurvature *curv) {
  return from_c(curv)->vertMeanCurvature.size();
}

float *manifold_curvature_vert_mean(void *mem, ManifoldCurvature *curv) {
  return copy_data(mem, from_c(curv)->vertMeanCurvature);
}

float *manifold_curvature_vert_gaussian(void *mem, ManifoldCurvature *curv) {
  return copy_data(mem, from_c(curv)->vertGaussianCurvature);
}

// Static Quality Globals
void manifold_set_min_circular_angle(float degrees) {
  Manifold::SetMinCircularAngle(degrees);
}

void manifold_set_min_circular_edge_length(float length) {
  Manifold::SetMinCircularEdgeLength(length);
}

void manifold_set_circular_segments(int number) {
  Manifold::SetCircularSegments(number);
}

int manifold_get_circular_segments(float radius) {
  return Manifold::GetCircularSegments(radius);
}

// memory size
size_t manifold_simple_polygon_size() { return sizeof(SimplePolygon); }
size_t manifold_polygons_size() { return sizeof(Polygons); }
size_t manifold_manifold_size() { return sizeof(Manifold); }
size_t manifold_manifold_pair_size() { return sizeof(ManifoldManifoldPair); }
size_t manifold_meshgl_size() { return sizeof(MeshGL); }
size_t manifold_box_size() { return sizeof(Box); }
size_t manifold_curvature_size() { return sizeof(Curvature); }
size_t manifold_components_size() { return sizeof(Components); }

// pointer free + destruction
void manifold_delete_simple_polygon(ManifoldSimplePolygon *p) {
  delete from_c(p);
}
void manifold_delete_polygons(ManifoldPolygons *p) { delete from_c(p); }
void manifold_delete_manifold(ManifoldManifold *m) { delete from_c(m); }
void manifold_delete_meshgl(ManifoldMeshGL *m) { delete from_c(m); }
void manifold_delete_box(ManifoldBox *b) { delete from_c(b); }
void manifold_delete_curvature(ManifoldCurvature *c) { delete from_c(c); }
void manifold_delete_components(ManifoldComponents *c) { delete from_c(c); }

// destruction
void manifold_destruct_simple_polygon(ManifoldSimplePolygon *p) {
  from_c(p)->~SimplePolygon();
}
void manifold_destruct_polygons(ManifoldPolygons *p) { from_c(p)->~Polygons(); }
void manifold_destruct_manifold(ManifoldManifold *m) { from_c(m)->~Manifold(); }
void manifold_destruct_meshgl(ManifoldMeshGL *m) { from_c(m)->~MeshGL(); }
void manifold_destruct_box(ManifoldBox *b) { from_c(b)->~Box(); }
void manifold_destruct_curvature(ManifoldCurvature *c) {
  from_c(c)->~Curvature();
}
void manifold_destruct_components(ManifoldComponents *c) {
  from_c(c)->~Components();
}

#ifdef __cplusplus
}
#endif
