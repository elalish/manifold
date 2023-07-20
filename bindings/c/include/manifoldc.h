#include <stddef.h>
#include <stdint.h>
#include <types.h>

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#endif

// Polygons

ManifoldSimplePolygon *manifold_simple_polygon(void *mem, ManifoldVec2 *ps,
                                               size_t length);
ManifoldPolygons *manifold_polygons(void *mem, ManifoldSimplePolygon **ps,
                                    size_t length);
size_t manifold_simple_polygon_length(ManifoldSimplePolygon *p);
size_t manifold_polygons_length(ManifoldPolygons *ps);
size_t manifold_polygons_simple_length(ManifoldPolygons *ps, int idx);
ManifoldVec2 manifold_simple_polygon_get_point(ManifoldSimplePolygon *p,
                                               int idx);
ManifoldSimplePolygon *manifold_polygons_get_simple(void *mem,
                                                    ManifoldPolygons *ps,
                                                    int idx);
ManifoldVec2 manifold_polygons_get_point(ManifoldPolygons *ps, int simple_idx,
                                         int pt_idx);

// Mesh Construction

ManifoldMeshGL *manifold_meshgl(void *mem, float *vert_props, size_t n_verts,
                                size_t n_props, uint32_t *tri_verts,
                                size_t n_tris);

ManifoldMeshGL *manifold_meshgl_w_tangents(void *mem, float *vert_props,
                                           size_t n_verts, size_t n_props,
                                           uint32_t *tri_verts, size_t n_tris,
                                           float *halfedge_tangent);
ManifoldMeshGL *manifold_get_meshgl(void *mem, ManifoldManifold *m);
ManifoldMeshGL *manifold_meshgl_copy(void *mem, ManifoldMeshGL *m);

// SDF
// By default, the execution policy (sequential or parallel) of
// manifold_level_set will be chosen automatically depending on the size of the
// job and whether Manifold has been compiled with a PAR backend. If you are
// using these bindings from a language that has a runtime lock preventing the
// parallel execution of closures, then you should use manifold_level_set_seq to
// force sequential execution.
ManifoldMeshGL *manifold_level_set(void *mem, float (*sdf)(float, float, float),
                                   ManifoldBox *bounds, float edge_length,
                                   float level);
ManifoldMeshGL *manifold_level_set_seq(void *mem,
                                       float (*sdf)(float, float, float),
                                       ManifoldBox *bounds, float edge_length,
                                       float level);
// The _context variants of manifold_level_set allow a pointer to be passed
// back to each invocation of the sdf function pointer, for languages that
// need additional data.
ManifoldMeshGL *manifold_level_set_context(
    void *mem, float (*sdf)(float, float, float, void *), ManifoldBox *bounds,
    float edge_length, float level, void *ctx);
ManifoldMeshGL *manifold_level_set_seq_context(
    void *mem, float (*sdf)(float, float, float, void *), ManifoldBox *bounds,
    float edge_length, float level, void *ctx);

// Manifold Vectors

ManifoldManifoldVec *manifold_manifold_empty_vec(void *mem);
ManifoldManifoldVec *manifold_manifold_vec(void *mem, size_t sz);
void manifold_manifold_vec_reserve(ManifoldManifoldVec *ms, size_t sz);
size_t manifold_manifold_vec_length(ManifoldManifoldVec *ms);
ManifoldManifold *manifold_manifold_vec_get(void *mem, ManifoldManifoldVec *ms,
                                            int idx);
void manifold_manifold_vec_set(ManifoldManifoldVec *ms, int idx,
                               ManifoldManifold *m);
void manifold_manifold_vec_push_back(ManifoldManifoldVec *ms,
                                     ManifoldManifold *m);

// Manifold Booleans

ManifoldManifold *manifold_boolean(void *mem, ManifoldManifold *a,
                                   ManifoldManifold *b, ManifoldOpType op);
ManifoldManifold *manifold_batch_boolean(void *mem, ManifoldManifoldVec *ms,
                                         ManifoldOpType op);
ManifoldManifold *manifold_union(void *mem, ManifoldManifold *a,
                                 ManifoldManifold *b);
ManifoldManifold *manifold_difference(void *mem, ManifoldManifold *a,
                                      ManifoldManifold *b);
ManifoldManifold *manifold_intersection(void *mem, ManifoldManifold *a,
                                        ManifoldManifold *b);
ManifoldManifoldPair manifold_split(void *mem_first, void *mem_second,
                                    ManifoldManifold *a, ManifoldManifold *b);
ManifoldManifoldPair manifold_split_by_plane(void *mem_first, void *mem_second,
                                             ManifoldManifold *m,
                                             float normal_x, float normal_y,
                                             float normal_z, float offset);
ManifoldManifold *manifold_trim_by_plane(void *mem, ManifoldManifold *m,
                                         float normal_x, float normal_y,
                                         float normal_z, float offset);

// Convex Hulls

ManifoldManifold *manifold_hull(void *mem, ManifoldManifold *m);
ManifoldManifold *manifold_batch_hull(void *mem, ManifoldManifoldVec *ms);
ManifoldManifold *manifold_hull_pts(void *mem, ManifoldVec3 *ps, size_t length);

// Manifold Transformations

ManifoldManifold *manifold_translate(void *mem, ManifoldManifold *m, float x,
                                     float y, float z);
ManifoldManifold *manifold_rotate(void *mem, ManifoldManifold *m, float x,
                                  float y, float z);
ManifoldManifold *manifold_scale(void *mem, ManifoldManifold *m, float x,
                                 float y, float z);
ManifoldManifold *manifold_transform(void *mem, ManifoldManifold *m, float x1,
                                     float y1, float z1, float x2, float y2,
                                     float z2, float x3, float y3, float z3,
                                     float x4, float y4, float z4);
ManifoldManifold *manifold_mirror(void *mem, ManifoldManifold *m, float nx,
                                  float ny, float nz);
ManifoldManifold *manifold_warp(void *mem, ManifoldManifold *m,
                                ManifoldVec3 (*fun)(float, float, float));
ManifoldManifold *manifold_refine(void *mem, ManifoldManifold *m, int refine);

// Manifold Shapes / Constructors

ManifoldManifold *manifold_empty(void *mem);
ManifoldManifold *manifold_copy(void *mem, ManifoldManifold *m);
ManifoldManifold *manifold_tetrahedron(void *mem);
ManifoldManifold *manifold_cube(void *mem, float x, float y, float z,
                                int center);
ManifoldManifold *manifold_cylinder(void *mem, float height, float radius_low,
                                    float radius_high, int circular_segments,
                                    int center);
ManifoldManifold *manifold_sphere(void *mem, float radius,
                                  int circular_segments);
ManifoldManifold *manifold_of_meshgl(void *mem, ManifoldMeshGL *mesh);
ManifoldManifold *manifold_smooth(void *mem, ManifoldMeshGL *mesh,
                                  int *half_edges, float *smoothness,
                                  int n_idxs);
ManifoldManifold *manifold_extrude(void *mem, ManifoldCrossSection *cs,
                                   float height, int slices,
                                   float twist_degrees, float scale_x,
                                   float scale_y);
ManifoldManifold *manifold_revolve(void *mem, ManifoldCrossSection *cs,
                                   int circular_segments);
ManifoldManifold *manifold_compose(void *mem, ManifoldManifoldVec *ms);
ManifoldManifoldVec *manifold_decompose(void *mem, ManifoldManifold *m);
ManifoldManifold *manifold_as_original(void *mem, ManifoldManifold *m);

// Manifold Info

int manifold_is_empty(ManifoldManifold *m);
ManifoldError manifold_status(ManifoldManifold *m);
int manifold_num_vert(ManifoldManifold *m);
int manifold_num_edge(ManifoldManifold *m);
int manifold_num_tri(ManifoldManifold *m);
ManifoldBox *manifold_bounding_box(void *mem, ManifoldManifold *m);
float manifold_precision(ManifoldManifold *m);
int manifold_genus(ManifoldManifold *m);
ManifoldProperties manifold_get_properties(ManifoldManifold *m);
int manifold_get_circular_segments(float radius);
int manifold_original_id(ManifoldManifold *m);
uint32_t manifold_reserve_ids(uint32_t n);
ManifoldManifold *manifold_set_properties(
    void *mem, ManifoldManifold *m, int num_prop,
    void (*fun)(float *new_prop, ManifoldVec3 position, const float *old_prop));
ManifoldManifold *manifold_calculate_curvature(void *mem, ManifoldManifold *m,
                                               int gaussian_idx, int mean_idx);

// CrossSection Shapes/Constructors

ManifoldCrossSection *manifold_cross_section_empty(void *mem);
ManifoldCrossSection *manifold_cross_section_copy(void *mem,
                                                  ManifoldCrossSection *cs);
ManifoldCrossSection *manifold_cross_section_of_simple_polygon(
    void *mem, ManifoldSimplePolygon *p, ManifoldFillRule fr);
ManifoldCrossSection *manifold_cross_section_of_polygons(void *mem,
                                                         ManifoldPolygons *p,
                                                         ManifoldFillRule fr);
ManifoldCrossSection *manifold_cross_section_square(void *mem, float x, float y,
                                                    int center);
ManifoldCrossSection *manifold_cross_section_circle(void *mem, float radius,
                                                    int circular_segments);
ManifoldCrossSection *manifold_cross_section_compose(
    void *mem, ManifoldCrossSectionVec *csv);
ManifoldCrossSectionVec *manifold_cross_section_decompose(
    void *mem, ManifoldCrossSection *cs);

// CrossSection Vectors

ManifoldCrossSectionVec *manifold_cross_section_empty_vec(void *mem);
ManifoldCrossSectionVec *manifold_cross_section_vec(void *mem, size_t sz);
void manifold_cross_section_vec_reserve(ManifoldCrossSectionVec *csv,
                                        size_t sz);
size_t manifold_cross_section_vec_length(ManifoldCrossSectionVec *csv);
ManifoldCrossSection *manifold_cross_section_vec_get(
    void *mem, ManifoldCrossSectionVec *csv, int idx);
void manifold_cross_section_vec_set(ManifoldCrossSectionVec *csv, int idx,
                                    ManifoldCrossSection *cs);
void manifold_cross_section_vec_push_back(ManifoldCrossSectionVec *csv,
                                          ManifoldCrossSection *cs);

// CrossSection Booleans

ManifoldCrossSection *manifold_cross_section_boolean(void *mem,
                                                     ManifoldCrossSection *a,
                                                     ManifoldCrossSection *b,
                                                     ManifoldOpType op);
ManifoldCrossSection *manifold_cross_section_batch_boolean(
    void *mem, ManifoldCrossSectionVec *csv, ManifoldOpType op);
ManifoldCrossSection *manifold_cross_section_union(void *mem,
                                                   ManifoldCrossSection *a,
                                                   ManifoldCrossSection *b);
ManifoldCrossSection *manifold_cross_section_difference(
    void *mem, ManifoldCrossSection *a, ManifoldCrossSection *b);
ManifoldCrossSection *manifold_cross_section_intersection(
    void *mem, ManifoldCrossSection *a, ManifoldCrossSection *b);

// CrossSection Convex Hulls

ManifoldCrossSection *manifold_cross_section_hull(void *mem,
                                                  ManifoldCrossSection *cs);
ManifoldCrossSection *manifold_cross_section_batch_hull(
    void *mem, ManifoldCrossSectionVec *css);
ManifoldCrossSection *manifold_cross_section_hull_simple_polygon(
    void *mem, ManifoldSimplePolygon *ps);
ManifoldCrossSection *manifold_cross_section_hull_polygons(
    void *mem, ManifoldPolygons *ps);

// CrossSection Transformation

ManifoldCrossSection *manifold_cross_section_translate(void *mem,
                                                       ManifoldCrossSection *cs,
                                                       float x, float y);
ManifoldCrossSection *manifold_cross_section_rotate(void *mem,
                                                    ManifoldCrossSection *cs,
                                                    float deg);
ManifoldCrossSection *manifold_cross_section_scale(void *mem,
                                                   ManifoldCrossSection *cs,
                                                   float x, float y);
ManifoldCrossSection *manifold_cross_section_mirror(void *mem,
                                                    ManifoldCrossSection *cs,
                                                    float ax_x, float ax_y);
ManifoldCrossSection *manifold_cross_section_transform(void *mem,
                                                       ManifoldCrossSection *cs,
                                                       float x1, float y1,
                                                       float x2, float y2,
                                                       float x3, float y3);
ManifoldCrossSection *manifold_cross_section_warp(
    void *mem, ManifoldCrossSection *cs, ManifoldVec2 (*fun)(float, float));
ManifoldCrossSection *manifold_cross_section_simplify(void *mem,
                                                      ManifoldCrossSection *cs,
                                                      double epsilon);
ManifoldCrossSection *manifold_cross_section_offset(
    void *mem, ManifoldCrossSection *cs, double delta, ManifoldJoinType jt,
    double miter_limit, int circular_segments);

// CrossSection Info

double manifold_cross_section_area(ManifoldCrossSection *cs);
int manifold_cross_section_num_vert(ManifoldCrossSection *cs);
int manifold_cross_section_num_contour(ManifoldCrossSection *cs);
int manifold_cross_section_is_empty(ManifoldCrossSection *cs);
ManifoldRect *manifold_cross_section_bounds(void *mem,
                                            ManifoldCrossSection *cs);
ManifoldPolygons *manifold_cross_section_to_polygons(void *mem,
                                                     ManifoldCrossSection *cs);

// Rectangle

ManifoldRect *manifold_rect(void *mem, float x1, float y1, float x2, float y2);
ManifoldVec2 manifold_rect_min(ManifoldRect *r);
ManifoldVec2 manifold_rect_max(ManifoldRect *r);
ManifoldVec2 manifold_rect_dimensions(ManifoldRect *r);
ManifoldVec2 manifold_rect_center(ManifoldRect *r);
float manifold_rect_scale(ManifoldRect *r);
int manifold_rect_contains_pt(ManifoldRect *r, float x, float y);
int manifold_rect_contains_rect(ManifoldRect *a, ManifoldRect *b);
void manifold_rect_include_pt(ManifoldRect *r, float x, float y);
ManifoldRect *manifold_rect_union(void *mem, ManifoldRect *a, ManifoldRect *b);
ManifoldRect *manifold_rect_transform(void *mem, ManifoldRect *r, float x1,
                                      float y1, float x2, float y2, float x3,
                                      float y3);
ManifoldRect *manifold_rect_translate(void *mem, ManifoldRect *r, float x,
                                      float y);
ManifoldRect *manifold_rect_mul(void *mem, ManifoldRect *r, float x, float y);
int manifold_rect_does_overlap_rect(ManifoldRect *a, ManifoldRect *r);
int manifold_rect_is_empty(ManifoldRect *r);
int manifold_rect_is_finite(ManifoldRect *r);
ManifoldCrossSection *manifold_rect_as_cross_section(void *mem,
                                                     ManifoldRect *r);

// Bounding Box

ManifoldBox *manifold_box(void *mem, float x1, float y1, float z1, float x2,
                          float y2, float z2);
ManifoldVec3 manifold_box_min(ManifoldBox *b);
ManifoldVec3 manifold_box_max(ManifoldBox *b);
ManifoldVec3 manifold_box_dimensions(ManifoldBox *b);
ManifoldVec3 manifold_box_center(ManifoldBox *b);
float manifold_box_scale(ManifoldBox *b);
int manifold_box_contains_pt(ManifoldBox *b, float x, float y, float z);
int manifold_box_contains_box(ManifoldBox *a, ManifoldBox *b);
void manifold_box_include_pt(ManifoldBox *b, float x, float y, float z);
ManifoldBox *manifold_box_union(void *mem, ManifoldBox *a, ManifoldBox *b);
ManifoldBox *manifold_box_transform(void *mem, ManifoldBox *b, float x1,
                                    float y1, float z1, float x2, float y2,
                                    float z2, float x3, float y3, float z3,
                                    float x4, float y4, float z4);
ManifoldBox *manifold_box_translate(void *mem, ManifoldBox *b, float x, float y,
                                    float z);
ManifoldBox *manifold_box_mul(void *mem, ManifoldBox *b, float x, float y,
                              float z);
int manifold_box_does_overlap_pt(ManifoldBox *b, float x, float y, float z);
int manifold_box_does_overlap_box(ManifoldBox *a, ManifoldBox *b);
int manifold_box_is_finite(ManifoldBox *b);

// Static Quality Globals

void manifold_set_min_circular_angle(float degrees);
void manifold_set_min_circular_edge_length(float length);
void manifold_set_circular_segments(int number);

// Manifold Mesh Extraction

int manifold_meshgl_num_prop(ManifoldMeshGL *m);
int manifold_meshgl_num_vert(ManifoldMeshGL *m);
int manifold_meshgl_num_tri(ManifoldMeshGL *m);
size_t manifold_meshgl_vert_properties_length(ManifoldMeshGL *m);
size_t manifold_meshgl_tri_length(ManifoldMeshGL *m);
size_t manifold_meshgl_merge_length(ManifoldMeshGL *m);
size_t manifold_meshgl_run_index_length(ManifoldMeshGL *m);
size_t manifold_meshgl_run_original_id_length(ManifoldMeshGL *m);
size_t manifold_meshgl_run_transform_length(ManifoldMeshGL *m);
size_t manifold_meshgl_face_id_length(ManifoldMeshGL *m);
size_t manifold_meshgl_tangent_length(ManifoldMeshGL *m);
float *manifold_meshgl_vert_properties(void *mem, ManifoldMeshGL *m);
uint32_t *manifold_meshgl_tri_verts(void *mem, ManifoldMeshGL *m);
uint32_t *manifold_meshgl_merge_from_vert(void *mem, ManifoldMeshGL *m);
uint32_t *manifold_meshgl_merge_to_vert(void *mem, ManifoldMeshGL *m);
uint32_t *manifold_meshgl_run_index(void *mem, ManifoldMeshGL *m);
uint32_t *manifold_meshgl_run_original_id(void *mem, ManifoldMeshGL *m);
float *manifold_meshgl_run_transform(void *mem, ManifoldMeshGL *m);
uint32_t *manifold_meshgl_face_id(void *mem, ManifoldMeshGL *m);
float *manifold_meshgl_halfedge_tangent(void *mem, ManifoldMeshGL *m);

// memory size

size_t manifold_manifold_size();
size_t manifold_manifold_vec_size();
size_t manifold_cross_section_size();
size_t manifold_cross_section_vec_size();
size_t manifold_simple_polygon_size();
size_t manifold_polygons_size();
size_t manifold_manifold_pair_size();
size_t manifold_meshgl_size();
size_t manifold_box_size();
size_t manifold_rect_size();
size_t manifold_curvature_size();

// destruction

void manifold_destruct_manifold(ManifoldManifold *m);
void manifold_destruct_manifold_vec(ManifoldManifoldVec *ms);
void manifold_destruct_cross_section(ManifoldCrossSection *m);
void manifold_destruct_cross_section_vec(ManifoldCrossSectionVec *csv);
void manifold_destruct_simple_polygon(ManifoldSimplePolygon *p);
void manifold_destruct_polygons(ManifoldPolygons *p);
void manifold_destruct_meshgl(ManifoldMeshGL *m);
void manifold_destruct_box(ManifoldBox *b);
void manifold_destruct_rect(ManifoldRect *b);

// pointer free + destruction

void manifold_delete_manifold(ManifoldManifold *m);
void manifold_delete_manifold_vec(ManifoldManifoldVec *ms);
void manifold_delete_cross_section(ManifoldCrossSection *cs);
void manifold_delete_cross_section_vec(ManifoldCrossSectionVec *csv);
void manifold_delete_simple_polygon(ManifoldSimplePolygon *p);
void manifold_delete_polygons(ManifoldPolygons *p);
void manifold_delete_meshgl(ManifoldMeshGL *m);
void manifold_delete_box(ManifoldBox *b);
void manifold_delete_rect(ManifoldRect *b);

// MeshIO / Export

#ifdef MANIFOLD_EXPORT
ManifoldMaterial *manifold_material(void *mem);
void manifold_material_set_roughness(ManifoldMaterial *mat, float roughness);
void manifold_material_set_metalness(ManifoldMaterial *mat, float metalness);
void manifold_material_set_color(ManifoldMaterial *mat, ManifoldVec4 color);
void manifold_material_set_vert_color(ManifoldMaterial *mat,
                                      ManifoldVec4 *vert_color, size_t n_vert);
ManifoldExportOptions *manifold_export_options(void *mem);
void manifold_export_options_set_faceted(ManifoldExportOptions *options,
                                         int faceted);
void manifold_export_options_set_material(ManifoldExportOptions *options,
                                          ManifoldMaterial *mat);
void manifold_export_meshgl(const char *filename, ManifoldMeshGL *mesh,
                            ManifoldExportOptions *options);
ManifoldMeshGL *manifold_import_meshgl(void *mem, const char *filename,
                                       int force_cleanup);

size_t manifold_material_size();
size_t manifold_export_options_size();

void manifold_destruct_material(ManifoldMaterial *m);
void manifold_destruct_export_options(ManifoldExportOptions *options);

void manifold_delete_material(ManifoldMaterial *m);
void manifold_delete_export_options(ManifoldExportOptions *options);
#endif

#ifdef __cplusplus
}
#endif
