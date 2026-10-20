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

#include "conv.h"
#include "manifold/common.h"
#include "manifold/cross_section.h"
#include "manifold/types.h"

#ifdef __cplusplus
extern "C" {
#endif
ManifoldCrossSection* manifold_cross_section_empty() {
  return to_c(new CrossSection());
}

ManifoldCrossSection* manifold_cross_section_copy(ManifoldCrossSection* cs) {
  auto cross = *from_c(cs);
  return to_c(new CrossSection(cross));
}

ManifoldCrossSection* manifold_cross_section_of_simple_polygon(
    ManifoldSimplePolygon* p) {
  return to_c(new CrossSection(*from_c(p)));
}

ManifoldCrossSection* manifold_cross_section_of_polygons(ManifoldPolygons* p) {
  return to_c(new CrossSection(*from_c(p)));
}

ManifoldCrossSection* manifold_cross_section_even_odd_simple_polygon(
    ManifoldSimplePolygon* p) {
  return to_c(new CrossSection(CrossSection::EvenOdd(*from_c(p))));
}

ManifoldCrossSection* manifold_cross_section_even_odd_polygons(
    ManifoldPolygons* p) {
  return to_c(new CrossSection(CrossSection::EvenOdd(*from_c(p))));
}

ManifoldCrossSectionVec* manifold_cross_section_empty_vec() {
  return to_c(new CrossSectionVec());
}

ManifoldCrossSectionVec* manifold_cross_section_vec(size_t sz) {
  return to_c(new CrossSectionVec(sz));
}

void manifold_cross_section_vec_reserve(ManifoldCrossSectionVec* csv,
                                        size_t sz) {
  from_c(csv)->reserve(sz);
}

size_t manifold_cross_section_vec_length(ManifoldCrossSectionVec* csv) {
  return from_c(csv)->size();
}

ManifoldCrossSection* manifold_cross_section_vec_get(
    ManifoldCrossSectionVec* csv, size_t idx) {
  auto cs = (*from_c(csv))[idx];
  return to_c(new CrossSection(cs));
}

void manifold_cross_section_vec_set(ManifoldCrossSectionVec* csv, size_t idx,
                                    ManifoldCrossSection* cs) {
  (*from_c(csv))[idx] = *from_c(cs);
}

void manifold_cross_section_vec_push_back(ManifoldCrossSectionVec* csv,
                                          ManifoldCrossSection* cs) {
  return from_c(csv)->push_back(*from_c(cs));
}

ManifoldCrossSection* manifold_cross_section_square(double x, double y,
                                                    int center) {
  auto cs = CrossSection::Square(vec2(x, y), center);
  return to_c(new CrossSection(cs));
}

ManifoldCrossSection* manifold_cross_section_circle(double radius,
                                                    int circular_segments) {
  auto cs = CrossSection::Circle(radius, circular_segments);
  return to_c(new CrossSection(cs));
}

ManifoldCrossSection* manifold_cross_section_boolean(ManifoldCrossSection* a,
                                                     ManifoldCrossSection* b,
                                                     ManifoldOpType op) {
  auto cs = from_c(a)->Boolean(*from_c(b), from_c(op));
  return to_c(new CrossSection(cs));
}

ManifoldCrossSection* manifold_cross_section_batch_boolean(
    ManifoldCrossSectionVec* csv, ManifoldOpType op) {
  auto cs = CrossSection::BatchBoolean(*from_c(csv), from_c(op));
  return to_c(new CrossSection(cs));
}

ManifoldCrossSection* manifold_cross_section_union(ManifoldCrossSection* a,
                                                   ManifoldCrossSection* b) {
  auto cs = (*from_c(a)) + (*from_c(b));
  return to_c(new CrossSection(cs));
}

ManifoldCrossSection* manifold_cross_section_difference(
    ManifoldCrossSection* a, ManifoldCrossSection* b) {
  auto cs = (*from_c(a)) - (*from_c(b));
  return to_c(new CrossSection(cs));
}

ManifoldCrossSection* manifold_cross_section_intersection(
    ManifoldCrossSection* a, ManifoldCrossSection* b) {
  auto cs = (*from_c(a)) ^ (*from_c(b));
  return to_c(new CrossSection(cs));
}

ManifoldCrossSection* manifold_cross_section_hull(ManifoldCrossSection* cs) {
  auto hulled = from_c(cs)->Hull();
  return to_c(new CrossSection(hulled));
}

ManifoldCrossSection* manifold_cross_section_batch_hull(
    ManifoldCrossSectionVec* css) {
  auto hulled = CrossSection::Hull(*from_c(css));
  return to_c(new CrossSection(hulled));
}

ManifoldCrossSection* manifold_cross_section_hull_simple_polygon(
    ManifoldSimplePolygon* ps) {
  auto hulled = CrossSection::Hull(*from_c(ps));
  return to_c(new CrossSection(hulled));
}

ManifoldCrossSection* manifold_cross_section_hull_polygons(
    ManifoldPolygons* ps) {
  auto hulled = CrossSection::Hull(*from_c(ps));
  return to_c(new CrossSection(hulled));
}

ManifoldCrossSection* manifold_cross_section_translate(ManifoldCrossSection* cs,
                                                       double x, double y) {
  auto translated = from_c(cs)->Translate(vec2(x, y));
  return to_c(new CrossSection(translated));
}

ManifoldCrossSection* manifold_cross_section_rotate(ManifoldCrossSection* cs,
                                                    double deg) {
  auto rotated = from_c(cs)->Rotate(deg);
  return to_c(new CrossSection(rotated));
}

ManifoldCrossSection* manifold_cross_section_scale(ManifoldCrossSection* cs,
                                                   double x, double y) {
  auto scaled = from_c(cs)->Scale(vec2(x, y));
  return to_c(new CrossSection(scaled));
}

ManifoldCrossSection* manifold_cross_section_mirror(ManifoldCrossSection* cs,
                                                    double ax_x, double ax_y) {
  auto mirrored = from_c(cs)->Mirror(vec2(ax_x, ax_y));
  return to_c(new CrossSection(mirrored));
}

ManifoldCrossSection* manifold_cross_section_transform(ManifoldCrossSection* cs,
                                                       double x1, double y1,
                                                       double x2, double y2,
                                                       double x3, double y3) {
  auto mat = mat2x3({x1, y1}, {x2, y2}, {x3, y3});
  auto transformed = from_c(cs)->Transform(mat);
  return to_c(new CrossSection(transformed));
}

ManifoldCrossSection* manifold_cross_section_warp_context(
    ManifoldCrossSection* cs, ManifoldVec2 (*fun)(double, double, void*),
    void* ctx) {
  // Bind function with context argument to one without
  using namespace std::placeholders;
  std::function<ManifoldVec2(double, double)> f2 = std::bind(fun, _1, _2, ctx);
  std::function<void(vec2 & v)> warp = [f2](vec2& v) {
    v = from_c(f2(v.x, v.y));
  };
  auto warped = from_c(cs)->Warp(warp);
  return to_c(new CrossSection(warped));
}

ManifoldCrossSection* manifold_cross_section_simplify(ManifoldCrossSection* cs,
                                                      double tolerance) {
  auto simplified = from_c(cs)->Simplify(tolerance);
  return to_c(new CrossSection(simplified));
}

ManifoldCrossSection* manifold_cross_section_offset(ManifoldCrossSection* cs,
                                                    double delta,
                                                    ManifoldJoinType jt,
                                                    double miter_limit,
                                                    int circular_segments) {
  auto offset =
      from_c(cs)->Offset(delta, from_c(jt), miter_limit, circular_segments);
  return to_c(new CrossSection(offset));
}

double manifold_cross_section_area(ManifoldCrossSection* cs) {
  return from_c(cs)->Area();
}

size_t manifold_cross_section_num_vert(ManifoldCrossSection* cs) {
  return from_c(cs)->NumVert();
}

size_t manifold_cross_section_num_contour(ManifoldCrossSection* cs) {
  return from_c(cs)->NumContour();
}

int manifold_cross_section_is_empty(ManifoldCrossSection* cs) {
  return from_c(cs)->IsEmpty();
}

ManifoldRect* manifold_cross_section_bounds(ManifoldCrossSection* cs) {
  auto rect = from_c(cs)->Bounds();
  return to_c(new Rect(rect));
}

ManifoldPolygons* manifold_cross_section_to_polygons(ManifoldCrossSection* cs) {
  auto ps = from_c(cs)->ToPolygons();
  return to_c(new Polygons(ps));
}

ManifoldCrossSectionVec* manifold_cross_section_decompose(
    ManifoldCrossSection* cs) {
  auto comps = from_c(cs)->Decompose();
  return to_c(new CrossSectionVec(comps));
}
#ifdef __cplusplus
}
#endif
