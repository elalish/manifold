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

#include <stddef.h>
#include <stdint.h>

#include "manifold/types.h"

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#endif

// Polygons

ManifoldSimplePolygon* manifold_simple_polygon(ManifoldVec2* ps, size_t length);
ManifoldPolygons* manifold_polygons(ManifoldSimplePolygon** ps, size_t length);
size_t manifold_simple_polygon_length(ManifoldSimplePolygon* p);
size_t manifold_polygons_length(ManifoldPolygons* ps);
size_t manifold_polygons_simple_length(ManifoldPolygons* ps, size_t idx);
ManifoldVec2 manifold_simple_polygon_get_point(ManifoldSimplePolygon* p,
                                               size_t idx);
ManifoldSimplePolygon* manifold_polygons_get_simple(ManifoldPolygons* ps,
                                                    size_t idx);
ManifoldVec2 manifold_polygons_get_point(ManifoldPolygons* ps,
                                         size_t simple_idx, size_t pt_idx);

// Mesh Construction

ManifoldMeshGL* manifold_meshgl(float* vert_props, size_t n_verts,
                                size_t n_props, uint32_t* tri_verts,
                                size_t n_tris);

ManifoldMeshGL* manifold_meshgl_w_tangents(float* vert_props, size_t n_verts,
                                           size_t n_props, uint32_t* tri_verts,
                                           size_t n_tris,
                                           float* halfedge_tangent);
ManifoldMeshGL* manifold_meshgl_w_options(float* vert_props, size_t n_verts,
                                          size_t n_props, uint32_t* tri_verts,
                                          size_t n_tris,
                                          ManifoldMeshGLOptions* options);
ManifoldMeshGL* manifold_get_meshgl(ManifoldManifold* m);
ManifoldMeshGL* manifold_get_meshgl_w_normals(ManifoldManifold* m,
                                              int32_t normalIdx);
ManifoldMeshGL* manifold_meshgl_copy(ManifoldMeshGL* m);
ManifoldMeshGL* manifold_meshgl_merge(ManifoldMeshGL* m);

ManifoldMeshGL64* manifold_meshgl64(double* vert_props, size_t n_verts,
                                    size_t n_props, uint64_t* tri_verts,
                                    size_t n_tris);

ManifoldMeshGL64* manifold_meshgl64_w_tangents(double* vert_props,
                                               size_t n_verts, size_t n_props,
                                               uint64_t* tri_verts,
                                               size_t n_tris,
                                               double* halfedge_tangent);
ManifoldMeshGL64* manifold_meshgl64_w_options(double* vert_props,
                                              size_t n_verts, size_t n_props,
                                              uint64_t* tri_verts,
                                              size_t n_tris,
                                              ManifoldMeshGL64Options* options);
ManifoldMeshGL64* manifold_get_meshgl64(ManifoldManifold* m);
ManifoldMeshGL64* manifold_get_meshgl64_w_normals(ManifoldManifold* m,
                                                  int32_t normalIdx);
ManifoldMeshGL64* manifold_meshgl64_copy(ManifoldMeshGL64* m);
ManifoldMeshGL64* manifold_meshgl64_merge(ManifoldMeshGL64* m);

// SDF
// By default, the execution policy (sequential or parallel) of
// manifold_level_set will be chosen automatically depending on the size of the
// job and whether Manifold has been compiled with a PAR backend. If you are
// using these bindings from a language that has a runtime lock preventing the
// parallel execution of closures, then you should use manifold_level_set_seq to
// force sequential execution.
ManifoldManifold* manifold_level_set(ManifoldSdf sdf, ManifoldBox* bounds,
                                     double edge_length, double level,
                                     double tolerance, void* ctx);
ManifoldManifold* manifold_level_set_seq(ManifoldSdf sdf, ManifoldBox* bounds,
                                         double edge_length, double level,
                                         double tolerance, void* ctx);

// Manifold Vectors

ManifoldManifoldVec* manifold_manifold_empty_vec();
ManifoldManifoldVec* manifold_manifold_vec(size_t sz);
void manifold_manifold_vec_reserve(ManifoldManifoldVec* ms, size_t sz);
size_t manifold_manifold_vec_length(ManifoldManifoldVec* ms);
ManifoldManifold* manifold_manifold_vec_get(ManifoldManifoldVec* ms,
                                            size_t idx);
void manifold_manifold_vec_set(ManifoldManifoldVec* ms, size_t idx,
                               ManifoldManifold* m);
void manifold_manifold_vec_push_back(ManifoldManifoldVec* ms,
                                     ManifoldManifold* m);

// Manifold Booleans

ManifoldManifold* manifold_boolean(ManifoldManifold* a, ManifoldManifold* b,
                                   ManifoldOpType op);
ManifoldManifold* manifold_batch_boolean(ManifoldManifoldVec* ms,
                                         ManifoldOpType op);
ManifoldManifold* manifold_union(ManifoldManifold* a, ManifoldManifold* b);
ManifoldManifold* manifold_difference(ManifoldManifold* a, ManifoldManifold* b);
ManifoldManifold* manifold_intersection(ManifoldManifold* a,
                                        ManifoldManifold* b);
ManifoldManifoldPair manifold_split(ManifoldManifold* a, ManifoldManifold* b);
ManifoldManifoldPair manifold_split_by_plane(ManifoldManifold* m,
                                             double normal_x, double normal_y,
                                             double normal_z, double offset);
ManifoldManifold* manifold_trim_by_plane(ManifoldManifold* m, double normal_x,
                                         double normal_y, double normal_z,
                                         double offset);
ManifoldManifold* manifold_minkowski_sum(ManifoldManifold* a,
                                         ManifoldManifold* b);
ManifoldManifold* manifold_minkowski_difference(ManifoldManifold* a,
                                                ManifoldManifold* b);

// 3D to 2D

ManifoldPolygons* manifold_slice(ManifoldManifold* m, double height);
ManifoldPolygons* manifold_project(ManifoldManifold* m);

// Convex Hulls

ManifoldManifold* manifold_hull(ManifoldManifold* m);
ManifoldManifold* manifold_batch_hull(ManifoldManifoldVec* ms);
ManifoldManifold* manifold_hull_pts(ManifoldVec3* ps, size_t length);

// Manifold Transformations

ManifoldManifold* manifold_translate(ManifoldManifold* m, double x, double y,
                                     double z);
ManifoldManifold* manifold_rotate(ManifoldManifold* m, double x, double y,
                                  double z);
ManifoldManifold* manifold_scale(ManifoldManifold* m, double x, double y,
                                 double z);
ManifoldManifold* manifold_transform(ManifoldManifold* m, double x1, double y1,
                                     double z1, double x2, double y2, double z2,
                                     double x3, double y3, double z3, double x4,
                                     double y4, double z4);
ManifoldManifold* manifold_mirror(ManifoldManifold* m, double nx, double ny,
                                  double nz);
ManifoldManifold* manifold_warp(ManifoldManifold* m,
                                ManifoldVec3 (*fun)(double, double, double,
                                                    void*),
                                void* ctx);
ManifoldManifold* manifold_smooth_by_normals(ManifoldManifold* m,
                                             int normalIdx);
ManifoldManifold* manifold_refine(ManifoldManifold* m, int refine);
ManifoldManifold* manifold_refine_to_length(ManifoldManifold* m, double length);
ManifoldManifold* manifold_refine_to_tolerance(ManifoldManifold* m,
                                               double tolerance);
ManifoldManifold* manifold_simplify(ManifoldManifold* m, double tolerance);

// Manifold Shapes / Constructors

ManifoldManifold* manifold_empty();
ManifoldManifold* manifold_copy(ManifoldManifold* m);
ManifoldManifold* manifold_tetrahedron();
ManifoldManifold* manifold_cube(double x, double y, double z, int center);
ManifoldManifold* manifold_cylinder(double height, double radius_low,
                                    double radius_high, int circular_segments,
                                    int center);
ManifoldManifold* manifold_sphere(double radius, int circular_segments);
ManifoldManifold* manifold_of_meshgl(ManifoldMeshGL* mesh);
ManifoldManifold* manifold_of_meshgl64(ManifoldMeshGL64* mesh);
ManifoldManifold* manifold_extrude(ManifoldPolygons* cs, double height,
                                   int slices, double twist_degrees,
                                   double scale_x, double scale_y);
ManifoldManifold* manifold_revolve(ManifoldPolygons* cs, int circular_segments,
                                   double revolve_degrees);
ManifoldManifold* manifold_compose(ManifoldManifoldVec* ms);
ManifoldManifoldVec* manifold_decompose(ManifoldManifold* m);

// id: the original ID to assign; pass a negative value (e.g. -1) for a new one.
ManifoldManifold* manifold_as_original(ManifoldManifold* m, int id);

// Manifold Info

int manifold_is_empty(ManifoldManifold* m);
ManifoldError manifold_status(ManifoldManifold* m);
// Returns a copy of the manifold with the given ExecutionContext attached.
// The attachment is consumed by the next eager op invoked on the result —
// manifold_status, or one of the manifold_refine* family. Deferred ops
// (Boolean operators, transforms, batch ops) ignore any attached ctx and
// produce a result with no attached ctx. See ExecutionContext in common.h
// for the full model.
ManifoldManifold* manifold_with_context(ManifoldManifold* m,
                                        ManifoldExecutionContext* ctx);
size_t manifold_num_vert(ManifoldManifold* m);
size_t manifold_num_edge(ManifoldManifold* m);
size_t manifold_num_tri(ManifoldManifold* m);
size_t manifold_num_prop(ManifoldManifold* m);
ManifoldBox* manifold_bounding_box(ManifoldManifold* m);
double manifold_epsilon(ManifoldManifold* m);
size_t manifold_num_prop_vert(ManifoldManifold* m);
int manifold_genus(ManifoldManifold* m);
double manifold_surface_area(ManifoldManifold* m);
double manifold_volume(ManifoldManifold* m);
int manifold_get_circular_segments(double radius);
int manifold_original_id(ManifoldManifold* m);
uint32_t manifold_reserve_ids(uint32_t n);
ManifoldManifold* manifold_set_properties(ManifoldManifold* m, int num_prop,
                                          void (*fun)(double* new_prop,
                                                      ManifoldVec3 position,
                                                      const double* old_prop,
                                                      void* ctx),
                                          void* ctx);
ManifoldManifold* manifold_calculate_curvature(ManifoldManifold* m,
                                               int gaussian_idx, int mean_idx);
double manifold_min_gap(ManifoldManifold* m, ManifoldManifold* other,
                        double searchLength);
ManifoldManifold* manifold_calculate_normals(ManifoldManifold* m,
                                             int normal_idx,
                                             double min_sharp_angle);

// Ray Casting
ManifoldRayHitVec* manifold_ray_cast(ManifoldManifold* m, double origin_x,
                                     double origin_y, double origin_z,
                                     double end_x, double end_y, double end_z);
size_t manifold_ray_hit_vec_length(ManifoldRayHitVec* v);
ManifoldRayHit manifold_ray_hit_vec_get(ManifoldRayHitVec* v, size_t idx);

// Point containment
int manifold_winding_number(ManifoldManifold* m, double x, double y, double z);

// ExecutionContext: observe progress and request cancellation of a
// long-running Manifold evaluation. Attach to a manifold via
// manifold_with_context; subsequent operations on that manifold observe
// the context. Safe to read/write from any thread. See the ExecutionContext
// class in common.h for full semantics (sticky cancel, per-boolean
// granularity).
ManifoldExecutionContext* manifold_execution_context();
void manifold_execution_context_cancel(ManifoldExecutionContext* ctx);
int manifold_execution_context_cancelled(ManifoldExecutionContext* ctx);
double manifold_execution_context_progress(ManifoldExecutionContext* ctx);

// ctx-aware static factories. These ops have no source manifold to attach via
// manifold_with_context, so they run on the ExecutionContext directly to report
// progress / observe cancellation. Mirror manifold_level_set /
// manifold_of_meshgl; `sdf_context` is the SDF callback's
// user-data.
ManifoldManifold* manifold_execution_context_level_set(
    ManifoldExecutionContext* ec, ManifoldSdf sdf, ManifoldBox* bounds,
    double edge_length, double level, double tolerance, void* sdf_context);
ManifoldManifold* manifold_execution_context_level_set_seq(
    ManifoldExecutionContext* ec, ManifoldSdf sdf, ManifoldBox* bounds,
    double edge_length, double level, double tolerance, void* sdf_context);
ManifoldManifold* manifold_execution_context_of_meshgl(
    ManifoldExecutionContext* ec, ManifoldMeshGL* mesh);
ManifoldManifold* manifold_execution_context_of_meshgl64(
    ManifoldExecutionContext* ec, ManifoldMeshGL64* mesh);

// CrossSection Shapes/Constructors
ManifoldCrossSection* manifold_cross_section_empty();
ManifoldCrossSection* manifold_cross_section_copy(ManifoldCrossSection* cs);
ManifoldCrossSection* manifold_cross_section_of_simple_polygon(
    ManifoldSimplePolygon* p);
ManifoldCrossSection* manifold_cross_section_of_polygons(ManifoldPolygons* p);
ManifoldCrossSection* manifold_cross_section_even_odd_simple_polygon(
    ManifoldSimplePolygon* p);
ManifoldCrossSection* manifold_cross_section_even_odd_polygons(ManifoldPolygons* p);
ManifoldCrossSection* manifold_cross_section_non_zero_simple_polygon(ManifoldSimplePolygon* p);
ManifoldCrossSection* manifold_cross_section_non_zero_polygons(ManifoldPolygons* p);
ManifoldCrossSection* manifold_cross_section_square(double x, double y, int center);
ManifoldCrossSection* manifold_cross_section_circle(double radius, int circular_segments);
ManifoldCrossSectionVec* manifold_cross_section_decompose(
    ManifoldCrossSection* cs);

// CrossSection Vectors

ManifoldCrossSectionVec* manifold_cross_section_empty_vec();
ManifoldCrossSectionVec* manifold_cross_section_vec(size_t sz);
void manifold_cross_section_vec_reserve(ManifoldCrossSectionVec* csv,
                                        size_t sz);
size_t manifold_cross_section_vec_length(ManifoldCrossSectionVec* csv);
ManifoldCrossSection* manifold_cross_section_vec_get(
    ManifoldCrossSectionVec* csv, size_t idx);
void manifold_cross_section_vec_set(ManifoldCrossSectionVec* csv, size_t idx,
                                    ManifoldCrossSection* cs);
void manifold_cross_section_vec_push_back(ManifoldCrossSectionVec* csv,
                                          ManifoldCrossSection* cs);

// CrossSection Booleans

ManifoldCrossSection* manifold_cross_section_boolean(ManifoldCrossSection* a,
                                                     ManifoldCrossSection* b,
                                                     ManifoldOpType op);
ManifoldCrossSection* manifold_cross_section_batch_boolean(
    ManifoldCrossSectionVec* csv, ManifoldOpType op);
ManifoldCrossSection* manifold_cross_section_union(ManifoldCrossSection* a,
                                                   ManifoldCrossSection* b);
ManifoldCrossSection* manifold_cross_section_difference(
    ManifoldCrossSection* a, ManifoldCrossSection* b);
ManifoldCrossSection* manifold_cross_section_intersection(
    ManifoldCrossSection* a, ManifoldCrossSection* b);

// CrossSection Convex Hulls

ManifoldCrossSection* manifold_cross_section_hull(ManifoldCrossSection* cs);
ManifoldCrossSection* manifold_cross_section_batch_hull(
    ManifoldCrossSectionVec* css);
ManifoldCrossSection* manifold_cross_section_hull_simple_polygon(
    ManifoldSimplePolygon* ps);
ManifoldCrossSection* manifold_cross_section_hull_polygons(
    ManifoldPolygons* ps);

// CrossSection Transformation

ManifoldCrossSection* manifold_cross_section_translate(ManifoldCrossSection* cs,
                                                       double x, double y);
ManifoldCrossSection* manifold_cross_section_rotate(ManifoldCrossSection* cs,
                                                    double deg);
ManifoldCrossSection* manifold_cross_section_scale(ManifoldCrossSection* cs,
                                                   double x, double y);
ManifoldCrossSection* manifold_cross_section_mirror(ManifoldCrossSection* cs,
                                                    double ax_x, double ax_y);
ManifoldCrossSection* manifold_cross_section_transform(ManifoldCrossSection* cs,
                                                       double x1, double y1,
                                                       double x2, double y2,
                                                       double x3, double y3);
ManifoldCrossSection* manifold_cross_section_warp_context(
    ManifoldCrossSection* cs, ManifoldVec2 (*fun)(double, double, void*),
    void* ctx);
ManifoldCrossSection* manifold_cross_section_simplify(ManifoldCrossSection* cs,
                                                      double tolerance);
ManifoldCrossSection* manifold_cross_section_offset(ManifoldCrossSection* cs,
                                                    double delta,
                                                    ManifoldJoinType jt,
                                                    double miter_limit,
                                                    int circular_segments);

// CrossSection Info

double manifold_cross_section_area(ManifoldCrossSection* cs);
size_t manifold_cross_section_num_vert(ManifoldCrossSection* cs);
size_t manifold_cross_section_num_contour(ManifoldCrossSection* cs);
int manifold_cross_section_is_empty(ManifoldCrossSection* cs);
ManifoldRect* manifold_cross_section_bounds(ManifoldCrossSection* cs);
ManifoldPolygons* manifold_cross_section_to_polygons(ManifoldCrossSection* cs);

// Rectangle

ManifoldRect* manifold_rect(double x1, double y1, double x2, double y2);
ManifoldVec2 manifold_rect_min(ManifoldRect* r);
ManifoldVec2 manifold_rect_max(ManifoldRect* r);
ManifoldVec2 manifold_rect_dimensions(ManifoldRect* r);
ManifoldVec2 manifold_rect_center(ManifoldRect* r);
double manifold_rect_scale(ManifoldRect* r);
int manifold_rect_contains_pt(ManifoldRect* r, double x, double y);
int manifold_rect_contains_rect(ManifoldRect* a, ManifoldRect* b);
void manifold_rect_include_pt(ManifoldRect* r, double x, double y);
ManifoldRect* manifold_rect_union(ManifoldRect* a, ManifoldRect* b);
ManifoldRect* manifold_rect_transform(ManifoldRect* r, double x1, double y1,
                                      double x2, double y2, double x3,
                                      double y3);
ManifoldRect* manifold_rect_translate(ManifoldRect* r, double x, double y);
ManifoldRect* manifold_rect_mul(ManifoldRect* r, double x, double y);
int manifold_rect_does_overlap_rect(ManifoldRect* a, ManifoldRect* r);
int manifold_rect_is_empty(ManifoldRect* r);
int manifold_rect_is_finite(ManifoldRect* r);

// Bounding Box

ManifoldBox* manifold_box(double x1, double y1, double z1, double x2, double y2,
                          double z2);
ManifoldVec3 manifold_box_min(ManifoldBox* b);
ManifoldVec3 manifold_box_max(ManifoldBox* b);
ManifoldVec3 manifold_box_dimensions(ManifoldBox* b);
ManifoldVec3 manifold_box_center(ManifoldBox* b);
double manifold_box_scale(ManifoldBox* b);
int manifold_box_contains_pt(ManifoldBox* b, double x, double y, double z);
int manifold_box_contains_box(ManifoldBox* a, ManifoldBox* b);
void manifold_box_include_pt(ManifoldBox* b, double x, double y, double z);
ManifoldBox* manifold_box_union(ManifoldBox* a, ManifoldBox* b);
ManifoldBox* manifold_box_transform(ManifoldBox* b, double x1, double y1,
                                    double z1, double x2, double y2, double z2,
                                    double x3, double y3, double z3, double x4,
                                    double y4, double z4);
ManifoldBox* manifold_box_translate(ManifoldBox* b, double x, double y,
                                    double z);
ManifoldBox* manifold_box_mul(ManifoldBox* b, double x, double y, double z);
int manifold_box_does_overlap_pt(ManifoldBox* b, double x, double y, double z);
int manifold_box_does_overlap_box(ManifoldBox* a, ManifoldBox* b);
int manifold_box_is_finite(ManifoldBox* b);

// Version

void manifold_version(int version[3]);

// Static Quality Globals

void manifold_set_relative_precision(double precision);
double manifold_get_relative_precision();
void manifold_set_min_circular_angle(double degrees);
void manifold_set_min_circular_edge_length(double length);
void manifold_set_circular_segments(int number);
void manifold_reset_to_circular_defaults();

// Manifold Mesh Extraction

size_t manifold_meshgl_num_prop(ManifoldMeshGL* m);
size_t manifold_meshgl_num_vert(ManifoldMeshGL* m);
size_t manifold_meshgl_num_tri(ManifoldMeshGL* m);
size_t manifold_meshgl_vert_properties_length(ManifoldMeshGL* m);
size_t manifold_meshgl_tri_length(ManifoldMeshGL* m);
size_t manifold_meshgl_merge_length(ManifoldMeshGL* m);
size_t manifold_meshgl_run_index_length(ManifoldMeshGL* m);
size_t manifold_meshgl_run_original_id_length(ManifoldMeshGL* m);
size_t manifold_meshgl_run_transform_length(ManifoldMeshGL* m);
size_t manifold_meshgl_face_id_length(ManifoldMeshGL* m);
size_t manifold_meshgl_tangent_length(ManifoldMeshGL* m);
float* manifold_meshgl_vert_properties(ManifoldMeshGL* m);
uint32_t* manifold_meshgl_tri_verts(ManifoldMeshGL* m);
uint32_t* manifold_meshgl_merge_from_vert(ManifoldMeshGL* m);
uint32_t* manifold_meshgl_merge_to_vert(ManifoldMeshGL* m);
uint32_t* manifold_meshgl_run_index(ManifoldMeshGL* m);
uint32_t* manifold_meshgl_run_original_id(ManifoldMeshGL* m);
float* manifold_meshgl_run_transform(ManifoldMeshGL* m);
uint32_t* manifold_meshgl_face_id(ManifoldMeshGL* m);
float* manifold_meshgl_halfedge_tangent(ManifoldMeshGL* m);
float manifold_meshgl_tolerance(ManifoldMeshGL* m);
size_t manifold_meshgl_run_flags_length(ManifoldMeshGL* m);
uint8_t* manifold_meshgl_run_flags(ManifoldMeshGL* m);
size_t manifold_meshgl_num_run(ManifoldMeshGL* m);
int manifold_meshgl_backside(ManifoldMeshGL* m, size_t run);
int manifold_meshgl_has_normals(ManifoldMeshGL* m, size_t run);

size_t manifold_meshgl64_num_prop(ManifoldMeshGL64* m);
size_t manifold_meshgl64_num_vert(ManifoldMeshGL64* m);
size_t manifold_meshgl64_num_tri(ManifoldMeshGL64* m);
size_t manifold_meshgl64_vert_properties_length(ManifoldMeshGL64* m);
size_t manifold_meshgl64_tri_length(ManifoldMeshGL64* m);
size_t manifold_meshgl64_merge_length(ManifoldMeshGL64* m);
size_t manifold_meshgl64_run_index_length(ManifoldMeshGL64* m);
size_t manifold_meshgl64_run_original_id_length(ManifoldMeshGL64* m);
size_t manifold_meshgl64_run_transform_length(ManifoldMeshGL64* m);
size_t manifold_meshgl64_face_id_length(ManifoldMeshGL64* m);
size_t manifold_meshgl64_tangent_length(ManifoldMeshGL64* m);
double* manifold_meshgl64_vert_properties(ManifoldMeshGL64* m);
uint64_t* manifold_meshgl64_tri_verts(ManifoldMeshGL64* m);
uint64_t* manifold_meshgl64_merge_from_vert(ManifoldMeshGL64* m);
uint64_t* manifold_meshgl64_merge_to_vert(ManifoldMeshGL64* m);
uint64_t* manifold_meshgl64_run_index(ManifoldMeshGL64* m);
uint32_t* manifold_meshgl64_run_original_id(ManifoldMeshGL64* m);
double* manifold_meshgl64_run_transform(ManifoldMeshGL64* m);
uint64_t* manifold_meshgl64_face_id(ManifoldMeshGL64* m);
double* manifold_meshgl64_halfedge_tangent(ManifoldMeshGL64* m);
double manifold_meshgl64_tolerance(ManifoldMeshGL64* m);
size_t manifold_meshgl64_run_flags_length(ManifoldMeshGL64* m);
uint8_t* manifold_meshgl64_run_flags(ManifoldMeshGL64* m);
size_t manifold_meshgl64_num_run(ManifoldMeshGL64* m);
int manifold_meshgl64_backside(ManifoldMeshGL64* m, size_t run);
int manifold_meshgl64_has_normals(ManifoldMeshGL64* m, size_t run);

// Triangulation

ManifoldTriangulation* manifold_triangulate(ManifoldPolygons* ps,
                                            double epsilon);
size_t manifold_triangulation_num_tri(ManifoldTriangulation* m);
// buffer size should be 3*manifold_triangulation_num_tri(m)
// this function copies the tri verts out into buffer and return the buffer
// pointer
int* manifold_triangulation_tri_verts(int* buffer, ManifoldTriangulation* m);

// destruction

void manifold_delete_manifold(ManifoldManifold* m);
void manifold_delete_manifold_vec(ManifoldManifoldVec* ms);
void manifold_delete_cross_section(ManifoldCrossSection* cs);
void manifold_delete_cross_section_vec(ManifoldCrossSectionVec* csv);
void manifold_delete_ray_hit_vec(ManifoldRayHitVec* v);
void manifold_delete_simple_polygon(ManifoldSimplePolygon* p);
void manifold_delete_polygons(ManifoldPolygons* p);
void manifold_delete_meshgl(ManifoldMeshGL* m);
void manifold_delete_meshgl64(ManifoldMeshGL64* m);
void manifold_delete_box(ManifoldBox* b);
void manifold_delete_rect(ManifoldRect* b);
void manifold_delete_triangulation(ManifoldTriangulation* m);
void manifold_delete_execution_context(ManifoldExecutionContext* ctx);

#ifndef MANIFOLD_NO_IOSTREAM
// MeshIO / Export

// Import a manifold from a Wavefront obj file.
// The obj_file parameter is the content of the obj file, not the filename,
// and should be null-terminated.
ManifoldManifold* manifold_read_obj(char* obj_file);
// Import a meshgl from a Wavefront obj file.
// The obj_file parameter is the content of the obj file, not the filename,
// and should be null-terminated.
ManifoldMeshGL64* manifold_meshgl64_read_obj(char* obj_file);
// Export a manifold to a Wavefront obj file.
// The callback accepts two parameters:
// 1. Temporary null-terminated string buffer, containing the content of the
//    file. This buffer will be freed automatically after returning from the
//    callback.
// 2. An arg value (the third parameter in the manifold_write_obj function), for
//    passing additional data into the callback.
void manifold_write_obj(ManifoldManifold* manifold,
                        void (*callback)(char*, void*), void* args);
// Export a MeshGL64 to a Wavefront obj file.
// The callback accepts two parameters:
// 1. Temporary null-terminated string buffer, containing the content of the
//    file. This buffer will be freed automatically after returning from the
//    callback.
// 2. An arg value (the third parameter in the manifold_write_obj function), for
//    passing additional data into the callback.
void manifold_meshgl64_write_obj(ManifoldMeshGL64* mesh,
                                 void (*callback)(char*, void*), void* args);
#endif
#ifdef __cplusplus
}
#endif
