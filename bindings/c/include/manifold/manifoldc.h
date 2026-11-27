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

// In general, if the function return type is `T*`,
// you have to cleanup the result by calling `manifold_delete_*`.
// If the function return type is `const T*`,
// the result is borrowed from input arguments.

// Relative precision defines what counts as coplanar or degenerate. A
// manifold's epsilon is its relative precision times its bounding box scale.
// Manifolds made from a MeshGL and built shapes have float epsilon
// (FLT_EPSILON in <float.h>). Manifolds made from a MeshGL64 or read from OBJ
// have MANIFOLD_MIN_RELATIVE_PRECISION, the finest allowed. Booleans,
// Minkowski and hulls of several manifolds use the smallest precision among
// their non-empty inputs. An empty Boolean or Minkowski result uses the
// smallest among all its inputs. Other operations keep the precision of their
// input. Use manifold_with_precision to change it.
#define MANIFOLD_MIN_RELATIVE_PRECISION 1e-12

// Polygons

ManifoldSimplePolygon* manifold_simple_polygon(const ManifoldVec2* ps,
                                               size_t length);
ManifoldPolygons* manifold_polygons(const ManifoldSimplePolygon* const* ps,
                                    size_t length);
size_t manifold_simple_polygon_length(const ManifoldSimplePolygon* p);
size_t manifold_polygons_length(const ManifoldPolygons* ps);
size_t manifold_polygons_simple_length(const ManifoldPolygons* ps, size_t idx);
ManifoldVec2 manifold_simple_polygon_get_point(const ManifoldSimplePolygon* p,
                                               size_t idx);
ManifoldSimplePolygon* manifold_polygons_get_simple(const ManifoldPolygons* ps,
                                                    size_t idx);
ManifoldVec2 manifold_polygons_get_point(const ManifoldPolygons* ps,
                                         size_t simple_idx, size_t pt_idx);

// Mesh Construction

ManifoldMeshGL* manifold_meshgl(const float* vert_props, size_t n_verts,
                                size_t n_props, const uint32_t* tri_verts,
                                size_t n_tris);

ManifoldMeshGL* manifold_meshgl_w_tangents(const float* vert_props,
                                           size_t n_verts, size_t n_props,
                                           const uint32_t* tri_verts,
                                           size_t n_tris,
                                           const float* halfedge_tangent);
ManifoldMeshGL* manifold_meshgl_w_options(const float* vert_props,
                                          size_t n_verts, size_t n_props,
                                          const uint32_t* tri_verts,
                                          size_t n_tris,
                                          const ManifoldMeshGLOptions* options);
ManifoldMeshGL* manifold_get_meshgl(const ManifoldManifold* m);
ManifoldMeshGL* manifold_meshgl_copy(const ManifoldMeshGL* m);
ManifoldMeshGL* manifold_meshgl_merge(const ManifoldMeshGL* m, float tolerance);

ManifoldMeshGL64* manifold_meshgl64(const double* vert_props, size_t n_verts,
                                    size_t n_props, const uint64_t* tri_verts,
                                    size_t n_tris);

ManifoldMeshGL64* manifold_meshgl64_w_tangents(const double* vert_props,
                                               size_t n_verts, size_t n_props,
                                               const uint64_t* tri_verts,
                                               size_t n_tris,
                                               const double* halfedge_tangent);
ManifoldMeshGL64* manifold_meshgl64_w_options(
    const double* vert_props, size_t n_verts, size_t n_props,
    const uint64_t* tri_verts, size_t n_tris,
    const ManifoldMeshGL64Options* options);
ManifoldMeshGL64* manifold_get_meshgl64(const ManifoldManifold* m);
ManifoldMeshGL64* manifold_meshgl64_copy(const ManifoldMeshGL64* m);
ManifoldMeshGL64* manifold_meshgl64_merge(const ManifoldMeshGL64* m,
                                          double tolerance);

// SDF
// By default, the execution policy (sequential or parallel) of
// manifold_level_set will be chosen automatically depending on the size of the
// job and whether Manifold has been compiled with a PAR backend. If you are
// using these bindings from a language that has a runtime lock preventing the
// parallel execution of closures, then you should use manifold_level_set_seq to
// force sequential execution.
ManifoldManifold* manifold_level_set(ManifoldSdf sdf, const ManifoldBox* bounds,
                                     double edge_length, double level,
                                     double tolerance, void* ctx);
ManifoldManifold* manifold_level_set_seq(ManifoldSdf sdf,
                                         const ManifoldBox* bounds,
                                         double edge_length, double level,
                                         double tolerance, void* ctx);

// Manifold Vectors

ManifoldManifoldVec* manifold_manifold_empty_vec(void);
ManifoldManifoldVec* manifold_manifold_vec(size_t sz);
void manifold_manifold_vec_reserve(ManifoldManifoldVec* ms, size_t sz);
size_t manifold_manifold_vec_length(const ManifoldManifoldVec* ms);
ManifoldManifold* manifold_manifold_vec_get(const ManifoldManifoldVec* ms,
                                            size_t idx);
void manifold_manifold_vec_set(ManifoldManifoldVec* ms, size_t idx,
                               const ManifoldManifold* m);
void manifold_manifold_vec_push_back(ManifoldManifoldVec* ms,
                                     const ManifoldManifold* m);

// Manifold Booleans

ManifoldManifold* manifold_boolean(const ManifoldManifold* a,
                                   const ManifoldManifold* b,
                                   ManifoldOpType op);
ManifoldManifold* manifold_batch_boolean(const ManifoldManifoldVec* ms,
                                         ManifoldOpType op);
ManifoldManifold* manifold_union(const ManifoldManifold* a,
                                 const ManifoldManifold* b);
ManifoldManifold* manifold_difference(const ManifoldManifold* a,
                                      const ManifoldManifold* b);
ManifoldManifold* manifold_intersection(const ManifoldManifold* a,
                                        const ManifoldManifold* b);
ManifoldManifoldPair manifold_split(const ManifoldManifold* a,
                                    const ManifoldManifold* b);
ManifoldManifoldPair manifold_split_by_plane(const ManifoldManifold* m,
                                             double normal_x, double normal_y,
                                             double normal_z, double offset);
ManifoldManifold* manifold_trim_by_plane(const ManifoldManifold* m,
                                         double normal_x, double normal_y,
                                         double normal_z, double offset);
ManifoldManifold* manifold_minkowski_sum(const ManifoldManifold* a,
                                         const ManifoldManifold* b);
ManifoldManifold* manifold_minkowski_difference(const ManifoldManifold* a,
                                                const ManifoldManifold* b);

// 3D to 2D

ManifoldPolygons* manifold_slice(const ManifoldManifold* m, double height);
ManifoldPolygons* manifold_project(const ManifoldManifold* m);

// Convex Hulls

ManifoldManifold* manifold_hull(const ManifoldManifold* m);
ManifoldManifold* manifold_batch_hull(const ManifoldManifoldVec* ms);
ManifoldManifold* manifold_hull_pts(const ManifoldVec3* ps, size_t length);

// Manifold Transformations

ManifoldManifold* manifold_translate(const ManifoldManifold* m, double x,
                                     double y, double z);
ManifoldManifold* manifold_rotate(const ManifoldManifold* m, double x, double y,
                                  double z);
ManifoldManifold* manifold_scale(const ManifoldManifold* m, double x, double y,
                                 double z);
ManifoldManifold* manifold_transform(const ManifoldManifold* m, double x1,
                                     double y1, double z1, double x2, double y2,
                                     double z2, double x3, double y3, double z3,
                                     double x4, double y4, double z4);
ManifoldManifold* manifold_mirror(const ManifoldManifold* m, double nx,
                                  double ny, double nz);
ManifoldManifold* manifold_warp(const ManifoldManifold* m,
                                ManifoldVec3 (*fun)(double, double, double,
                                                    void*),
                                void* ctx);
// Like manifold_warp, but fun is called once with all count vertex positions
// as interleaved x, y, z doubles. fun may read and overwrite all count points
// in place. These become the vertices of the returned manifold. m is not
// modified, since the buffer belongs to a copy. The pointer is only valid
// during the call: do not free it, keep it, or access past count points. xyz
// may be NULL when count is 0. fun runs exactly once, synchronously, before
// manifold_warp_batch returns. If m has an error status, fun is not called and
// the result carries that status. If fun writes a non-finite value, the result
// is empty with MANIFOLD_NON_FINITE_VERTEX.
ManifoldManifold* manifold_warp_batch(const ManifoldManifold* m,
                                      void (*fun)(double* xyz, size_t count,
                                                  void* ctx),
                                      void* ctx);
ManifoldManifold* manifold_smooth_by_normals(const ManifoldManifold* m);
ManifoldManifold* manifold_refine(const ManifoldManifold* m, int refine);
ManifoldManifold* manifold_refine_to_length(const ManifoldManifold* m,
                                            double length);
ManifoldManifold* manifold_refine_to_tolerance(const ManifoldManifold* m,
                                               double tolerance);
ManifoldManifold* manifold_simplify(const ManifoldManifold* m,
                                    double tolerance);

// Manifold Shapes / Constructors

ManifoldManifold* manifold_empty(void);
ManifoldManifold* manifold_copy(const ManifoldManifold* m);
ManifoldManifold* manifold_tetrahedron(void);
ManifoldManifold* manifold_cube(double x, double y, double z, int center);
// quality may be NULL for the default CircularQuality.
ManifoldManifold* manifold_cylinder(double height, double radius_low,
                                    double radius_high,
                                    const ManifoldCircularQuality* quality,
                                    int center);
ManifoldManifold* manifold_sphere(double radius,
                                  const ManifoldCircularQuality* quality);
ManifoldManifold* manifold_of_meshgl(const ManifoldMeshGL* mesh);
ManifoldManifold* manifold_of_meshgl64(const ManifoldMeshGL64* mesh);
ManifoldManifold* manifold_extrude(const ManifoldPolygons* cs, double height,
                                   int slices, double twist_degrees,
                                   double scale_x, double scale_y);
ManifoldManifold* manifold_revolve(const ManifoldPolygons* cs,
                                   const ManifoldCircularQuality* quality,
                                   double revolve_degrees);
ManifoldManifoldVec* manifold_decompose(const ManifoldManifold* m);

// id: the original ID to assign; pass a negative value (e.g. -1) for a new one.
ManifoldManifold* manifold_as_original(const ManifoldManifold* m, int id);

// Manifold Info

int manifold_is_empty(const ManifoldManifold* m);
ManifoldError manifold_status(const ManifoldManifold* m);
// Returns a short English description of the error, the same text every
// binding uses. Values outside the enum give "Unknown error". The string is
// static and must not be freed.
const char* manifold_error_string(ManifoldError error);
// Returns a copy of the manifold with the given ExecutionContext attached.
// Only manifold_status, manifold_with_precision, manifold_get_precision and
// the eager ops (the manifold_refine* family, manifold_hull,
// manifold_minkowski_sum and manifold_minkowski_difference) observe it,
// reporting progress and cancellation through it. Other queries that force
// evaluation, such as manifold_volume or manifold_get_meshgl, do not. Every
// other op that returns a manifold ignores the attached ctx and returns a
// result with no attached ctx. Copies (manifold_copy and the
// manifold_manifold_vec get/set/push_back calls) keep the attachment. See
// Manifold::WithContext in manifold.h and ExecutionContext in common.h for
// the full model.
ManifoldManifold* manifold_with_context(const ManifoldManifold* m,
                                        const ManifoldExecutionContext* ctx);
// Evaluates m and returns it with the given relative precision for later
// operations, without recomputing it. 0, negative values, NaN and infinity
// give FLT_EPSILON, even for a manifold made from a MeshGL64. Positive values
// below MANIFOLD_MIN_RELATIVE_PRECISION are raised to it.
ManifoldManifold* manifold_with_precision(const ManifoldManifold* m,
                                          double rel_precision);
// Returns the relative precision of m. See MANIFOLD_MIN_RELATIVE_PRECISION.
double manifold_get_precision(const ManifoldManifold* m);
size_t manifold_num_vert(const ManifoldManifold* m);
size_t manifold_num_edge(const ManifoldManifold* m);
size_t manifold_num_tri(const ManifoldManifold* m);
size_t manifold_num_prop(const ManifoldManifold* m);
ManifoldBox* manifold_bounding_box(const ManifoldManifold* m);
// Returns the relative precision of m times its bounding box scale, or -1 if
// m is empty.
double manifold_epsilon(const ManifoldManifold* m);
size_t manifold_num_prop_vert(const ManifoldManifold* m);
int manifold_genus(const ManifoldManifold* m);
double manifold_surface_area(const ManifoldManifold* m);
double manifold_volume(const ManifoldManifold* m);
int manifold_original_id(const ManifoldManifold* m);
uint32_t manifold_reserve_ids(uint32_t n);
// A NULL fun sets the new properties to zero.
ManifoldManifold* manifold_set_properties(
    const ManifoldManifold* m, int num_prop,
    void (*fun)(double* new_prop, ManifoldVec3 position, const double* old_prop,
                void* ctx),
    void* ctx);
ManifoldManifold* manifold_calculate_curvature(const ManifoldManifold* m,
                                               int gaussian_idx, int mean_idx);
double manifold_min_gap(const ManifoldManifold* m,
                        const ManifoldManifold* other, double search_length);
ManifoldManifold* manifold_calculate_normals(const ManifoldManifold* m,
                                             double min_sharp_angle);

// Ray Casting
ManifoldRayHitVec* manifold_ray_cast(const ManifoldManifold* m, double origin_x,
                                     double origin_y, double origin_z,
                                     double end_x, double end_y, double end_z);
size_t manifold_ray_hit_vec_length(const ManifoldRayHitVec* v);
ManifoldRayHit manifold_ray_hit_vec_get(const ManifoldRayHitVec* v, size_t idx);

// Point containment
int manifold_winding_number(const ManifoldManifold* m, double x, double y,
                            double z);

// ExecutionContext: observe progress and request cancellation of a
// long-running Manifold evaluation. Attach it with manifold_with_context,
// then call manifold_status or one of the eager ops listed there on the
// result. Safe to read/write from any thread. See the ExecutionContext
// class in common.h for full semantics (sticky cancel, cancellation
// granularity).
ManifoldExecutionContext* manifold_execution_context(void);
void manifold_execution_context_cancel(ManifoldExecutionContext* ctx);
int manifold_execution_context_cancelled(const ManifoldExecutionContext* ctx);
double manifold_execution_context_progress(const ManifoldExecutionContext* ctx);

// ctx-aware static factories. These ops have no source manifold to attach via
// manifold_with_context, so they run on the ExecutionContext directly to report
// progress / observe cancellation. Mirror manifold_level_set /
// manifold_of_meshgl; `sdf_context` is the SDF callback's
// user-data.
ManifoldManifold* manifold_execution_context_level_set(
    ManifoldExecutionContext* ec, ManifoldSdf sdf, const ManifoldBox* bounds,
    double edge_length, double level, double tolerance, void* sdf_context);
ManifoldManifold* manifold_execution_context_level_set_seq(
    ManifoldExecutionContext* ec, ManifoldSdf sdf, const ManifoldBox* bounds,
    double edge_length, double level, double tolerance, void* sdf_context);
ManifoldManifold* manifold_execution_context_of_meshgl(
    ManifoldExecutionContext* ec, const ManifoldMeshGL* mesh);
ManifoldManifold* manifold_execution_context_of_meshgl64(
    ManifoldExecutionContext* ec, const ManifoldMeshGL64* mesh);

// CrossSection Shapes/Constructors
ManifoldCrossSection* manifold_cross_section_empty(void);
ManifoldCrossSection* manifold_cross_section_copy(
    const ManifoldCrossSection* cs);
ManifoldCrossSection* manifold_cross_section_of_simple_polygon(
    const ManifoldSimplePolygon* p);
ManifoldCrossSection* manifold_cross_section_of_polygons(
    const ManifoldPolygons* p);
ManifoldCrossSection* manifold_cross_section_even_odd_simple_polygon(
    const ManifoldSimplePolygon* p);
ManifoldCrossSection* manifold_cross_section_even_odd_polygons(
    const ManifoldPolygons* p);
ManifoldCrossSection* manifold_cross_section_non_zero_simple_polygon(
    const ManifoldSimplePolygon* p);
ManifoldCrossSection* manifold_cross_section_non_zero_polygons(
    const ManifoldPolygons* p);
ManifoldCrossSection* manifold_cross_section_square(double x, double y,
                                                    int center);
ManifoldCrossSection* manifold_cross_section_circle(
    double radius, const ManifoldCircularQuality* quality);
ManifoldCrossSectionVec* manifold_cross_section_decompose(
    const ManifoldCrossSection* cs);

// CrossSection Vectors

ManifoldCrossSectionVec* manifold_cross_section_empty_vec(void);
ManifoldCrossSectionVec* manifold_cross_section_vec(size_t sz);
void manifold_cross_section_vec_reserve(ManifoldCrossSectionVec* csv,
                                        size_t sz);
size_t manifold_cross_section_vec_length(const ManifoldCrossSectionVec* csv);
ManifoldCrossSection* manifold_cross_section_vec_get(
    const ManifoldCrossSectionVec* csv, size_t idx);
void manifold_cross_section_vec_set(ManifoldCrossSectionVec* csv, size_t idx,
                                    const ManifoldCrossSection* cs);
void manifold_cross_section_vec_push_back(ManifoldCrossSectionVec* csv,
                                          const ManifoldCrossSection* cs);

// CrossSection Booleans

ManifoldCrossSection* manifold_cross_section_boolean(
    const ManifoldCrossSection* a, const ManifoldCrossSection* b,
    ManifoldOpType op);
ManifoldCrossSection* manifold_cross_section_batch_boolean(
    const ManifoldCrossSectionVec* csv, ManifoldOpType op);
ManifoldCrossSection* manifold_cross_section_union(
    const ManifoldCrossSection* a, const ManifoldCrossSection* b);
ManifoldCrossSection* manifold_cross_section_difference(
    const ManifoldCrossSection* a, const ManifoldCrossSection* b);
ManifoldCrossSection* manifold_cross_section_intersection(
    const ManifoldCrossSection* a, const ManifoldCrossSection* b);

// CrossSection Convex Hulls

ManifoldCrossSection* manifold_cross_section_hull(
    const ManifoldCrossSection* cs);
ManifoldCrossSection* manifold_cross_section_batch_hull(
    const ManifoldCrossSectionVec* css);
ManifoldCrossSection* manifold_cross_section_hull_simple_polygon(
    const ManifoldSimplePolygon* ps);
ManifoldCrossSection* manifold_cross_section_hull_polygons(
    const ManifoldPolygons* ps);

// CrossSection Transformation

ManifoldCrossSection* manifold_cross_section_translate(
    const ManifoldCrossSection* cs, double x, double y);
ManifoldCrossSection* manifold_cross_section_rotate(
    const ManifoldCrossSection* cs, double deg);
ManifoldCrossSection* manifold_cross_section_scale(
    const ManifoldCrossSection* cs, double x, double y);
ManifoldCrossSection* manifold_cross_section_mirror(
    const ManifoldCrossSection* cs, double ax_x, double ax_y);
ManifoldCrossSection* manifold_cross_section_transform(
    const ManifoldCrossSection* cs, double x1, double y1, double x2, double y2,
    double x3, double y3);
ManifoldCrossSection* manifold_cross_section_warp(
    const ManifoldCrossSection* cs, ManifoldVec2 (*fun)(double, double, void*),
    void* ctx);
// Like manifold_cross_section_warp, but fun is called once with all count
// vertex positions as interleaved x, y doubles. fun may read and overwrite all
// count points in place. The returned cross section is built from these
// points, with the fill rule re-applied as in manifold_cross_section_warp. cs
// is not modified, since the buffer belongs to a copy. The pointer is only
// valid during the call: do not free it, keep it, or access past count points.
// xy may be NULL when count is 0. fun runs exactly once, synchronously, before
// manifold_cross_section_warp_batch returns.
ManifoldCrossSection* manifold_cross_section_warp_batch(
    const ManifoldCrossSection* cs,
    void (*fun)(double* xy, size_t count, void* ctx), void* ctx);
ManifoldCrossSection* manifold_cross_section_simplify(
    const ManifoldCrossSection* cs, double tolerance);
ManifoldCrossSection* manifold_cross_section_offset(
    const ManifoldCrossSection* cs, double delta, ManifoldJoinType jt,
    double miter_limit, const ManifoldCircularQuality* quality);

// CrossSection Info

double manifold_cross_section_area(const ManifoldCrossSection* cs);
size_t manifold_cross_section_num_vert(const ManifoldCrossSection* cs);
size_t manifold_cross_section_num_contour(const ManifoldCrossSection* cs);
int manifold_cross_section_is_empty(const ManifoldCrossSection* cs);
ManifoldRect* manifold_cross_section_bounds(const ManifoldCrossSection* cs);
ManifoldPolygons* manifold_cross_section_to_polygons(
    const ManifoldCrossSection* cs);

// Rectangle

ManifoldRect* manifold_rect(double x1, double y1, double x2, double y2);
ManifoldVec2 manifold_rect_min(const ManifoldRect* r);
ManifoldVec2 manifold_rect_max(const ManifoldRect* r);
ManifoldVec2 manifold_rect_dimensions(const ManifoldRect* r);
ManifoldVec2 manifold_rect_center(const ManifoldRect* r);
double manifold_rect_scale(const ManifoldRect* r);
int manifold_rect_contains_pt(const ManifoldRect* r, double x, double y);
int manifold_rect_contains_rect(const ManifoldRect* a, const ManifoldRect* b);
void manifold_rect_include_pt(ManifoldRect* r, double x, double y);
ManifoldRect* manifold_rect_union(const ManifoldRect* a, const ManifoldRect* b);
ManifoldRect* manifold_rect_transform(const ManifoldRect* r, double x1,
                                      double y1, double x2, double y2,
                                      double x3, double y3);
ManifoldRect* manifold_rect_translate(const ManifoldRect* r, double x,
                                      double y);
ManifoldRect* manifold_rect_mul(const ManifoldRect* r, double x, double y);
int manifold_rect_does_overlap_rect(const ManifoldRect* a,
                                    const ManifoldRect* r);
int manifold_rect_is_empty(const ManifoldRect* r);
int manifold_rect_is_finite(const ManifoldRect* r);

// Bounding Box

ManifoldBox* manifold_box(double x1, double y1, double z1, double x2, double y2,
                          double z2);
ManifoldVec3 manifold_box_min(const ManifoldBox* b);
ManifoldVec3 manifold_box_max(const ManifoldBox* b);
ManifoldVec3 manifold_box_dimensions(const ManifoldBox* b);
ManifoldVec3 manifold_box_center(const ManifoldBox* b);
double manifold_box_scale(const ManifoldBox* b);
int manifold_box_contains_pt(const ManifoldBox* b, double x, double y,
                             double z);
int manifold_box_contains_box(const ManifoldBox* a, const ManifoldBox* b);
void manifold_box_include_pt(ManifoldBox* b, double x, double y, double z);
ManifoldBox* manifold_box_union(const ManifoldBox* a, const ManifoldBox* b);
ManifoldBox* manifold_box_transform(const ManifoldBox* b, double x1, double y1,
                                    double z1, double x2, double y2, double z2,
                                    double x3, double y3, double z3, double x4,
                                    double y4, double z4);
ManifoldBox* manifold_box_translate(const ManifoldBox* b, double x, double y,
                                    double z);
ManifoldBox* manifold_box_mul(const ManifoldBox* b, double x, double y,
                              double z);
int manifold_box_does_overlap_pt(const ManifoldBox* b, double x, double y,
                                 double z);
int manifold_box_does_overlap_box(const ManifoldBox* a, const ManifoldBox* b);
int manifold_box_is_finite(const ManifoldBox* b);

// Version

void manifold_version(int version[3]);

// Circular Quality
// quality may be NULL for the default CircularQuality.
int manifold_circular_quality_segments_for(
    double radius, const ManifoldCircularQuality* quality);
ManifoldCircularQuality manifold_circular_quality_for_divisions(int divisions,
                                                                double degrees);

// Manifold Mesh Extraction

size_t manifold_meshgl_num_prop(const ManifoldMeshGL* m);
size_t manifold_meshgl_num_vert(const ManifoldMeshGL* m);
size_t manifold_meshgl_num_tri(const ManifoldMeshGL* m);
size_t manifold_meshgl_vert_properties_length(const ManifoldMeshGL* m);
size_t manifold_meshgl_tri_length(const ManifoldMeshGL* m);
size_t manifold_meshgl_merge_length(const ManifoldMeshGL* m);
size_t manifold_meshgl_run_index_length(const ManifoldMeshGL* m);
size_t manifold_meshgl_run_original_id_length(const ManifoldMeshGL* m);
size_t manifold_meshgl_run_transform_length(const ManifoldMeshGL* m);
size_t manifold_meshgl_face_id_length(const ManifoldMeshGL* m);
size_t manifold_meshgl_tangent_length(const ManifoldMeshGL* m);
size_t manifold_meshgl_run_flags_length(const ManifoldMeshGL* m);
// The buffer is borrowed from `m` and is only valid for the lifetime of `m`.
const float* manifold_meshgl_vert_properties(const ManifoldMeshGL* m);
const uint32_t* manifold_meshgl_tri_verts(const ManifoldMeshGL* m);
const uint32_t* manifold_meshgl_merge_from_vert(const ManifoldMeshGL* m);
const uint32_t* manifold_meshgl_merge_to_vert(const ManifoldMeshGL* m);
const uint32_t* manifold_meshgl_run_index(const ManifoldMeshGL* m);
const uint32_t* manifold_meshgl_run_original_id(const ManifoldMeshGL* m);
const float* manifold_meshgl_run_transform(const ManifoldMeshGL* m);
const uint32_t* manifold_meshgl_face_id(const ManifoldMeshGL* m);
const float* manifold_meshgl_halfedge_tangent(const ManifoldMeshGL* m);
const uint8_t* manifold_meshgl_run_flags(const ManifoldMeshGL* m);
size_t manifold_meshgl_num_run(const ManifoldMeshGL* m);
int manifold_meshgl_backside(const ManifoldMeshGL* m, size_t run);
int manifold_meshgl_has_normals(const ManifoldMeshGL* m, size_t run);

size_t manifold_meshgl64_num_prop(const ManifoldMeshGL64* m);
size_t manifold_meshgl64_num_vert(const ManifoldMeshGL64* m);
size_t manifold_meshgl64_num_tri(const ManifoldMeshGL64* m);
size_t manifold_meshgl64_vert_properties_length(const ManifoldMeshGL64* m);
size_t manifold_meshgl64_tri_length(const ManifoldMeshGL64* m);
size_t manifold_meshgl64_merge_length(const ManifoldMeshGL64* m);
size_t manifold_meshgl64_run_index_length(const ManifoldMeshGL64* m);
size_t manifold_meshgl64_run_original_id_length(const ManifoldMeshGL64* m);
size_t manifold_meshgl64_run_transform_length(const ManifoldMeshGL64* m);
size_t manifold_meshgl64_face_id_length(const ManifoldMeshGL64* m);
size_t manifold_meshgl64_tangent_length(const ManifoldMeshGL64* m);
size_t manifold_meshgl64_run_flags_length(const ManifoldMeshGL64* m);
// The buffer is borrowed from `m` and is only valid for the lifetime of `m`.
const double* manifold_meshgl64_vert_properties(const ManifoldMeshGL64* m);
const uint64_t* manifold_meshgl64_tri_verts(const ManifoldMeshGL64* m);
const uint64_t* manifold_meshgl64_merge_from_vert(const ManifoldMeshGL64* m);
const uint64_t* manifold_meshgl64_merge_to_vert(const ManifoldMeshGL64* m);
const uint64_t* manifold_meshgl64_run_index(const ManifoldMeshGL64* m);
const uint32_t* manifold_meshgl64_run_original_id(const ManifoldMeshGL64* m);
const double* manifold_meshgl64_run_transform(const ManifoldMeshGL64* m);
const uint64_t* manifold_meshgl64_face_id(const ManifoldMeshGL64* m);
const double* manifold_meshgl64_halfedge_tangent(const ManifoldMeshGL64* m);
const uint8_t* manifold_meshgl64_run_flags(const ManifoldMeshGL64* m);
size_t manifold_meshgl64_num_run(const ManifoldMeshGL64* m);
int manifold_meshgl64_backside(const ManifoldMeshGL64* m, size_t run);
int manifold_meshgl64_has_normals(const ManifoldMeshGL64* m, size_t run);

// Triangulation

ManifoldTriangulation* manifold_triangulate(const ManifoldPolygons* ps,
                                            double epsilon);
size_t manifold_triangulation_num_tri(const ManifoldTriangulation* m);
// The buffer is borrowed from `m` and is only valid for the lifetime of `m`.
const int* manifold_triangulation_tri_verts(const ManifoldTriangulation* m);

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
// and should be null-terminated. The result has relative precision
// MANIFOLD_MIN_RELATIVE_PRECISION.
ManifoldManifold* manifold_read_obj(const char* obj_file);
// Import a meshgl from a Wavefront obj file.
// The obj_file parameter is the content of the obj file, not the filename,
// and should be null-terminated.
ManifoldMeshGL64* manifold_meshgl64_read_obj(const char* obj_file);
// Export a manifold to a Wavefront obj file.
// The callback accepts two parameters:
// 1. Temporary null-terminated string buffer, containing the content of the
//    file. This buffer will be freed automatically after returning from the
//    callback.
// 2. An arg value (the third parameter in the manifold_write_obj function), for
//    passing additional data into the callback.
void manifold_write_obj(const ManifoldManifold* manifold,
                        void (*callback)(char*, void*), void* args);
// Export a MeshGL64 to a Wavefront obj file.
// The callback accepts two parameters:
// 1. Temporary null-terminated string buffer, containing the content of the
//    file. This buffer will be freed automatically after returning from the
//    callback.
// 2. An arg value (the third parameter in the manifold_meshgl64_write_obj
//    function), for passing additional data into the callback.
void manifold_meshgl64_write_obj(const ManifoldMeshGL64* mesh,
                                 void (*callback)(char*, void*), void* args);
#endif
#ifdef __cplusplus
}
#endif
