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

#include "./conv.h"
#include "manifold/common.h"
#include "manifold/types.h"

#ifdef __cplusplus
extern "C" {
#endif
ManifoldRect *manifold_rect(void *mem, double x1, double y1, double x2,
                            double y2) {
  auto p1 = vec2(x1, y1);
  auto p2 = vec2(x2, y2);
  auto rect = new (mem) Rect(p1, p2);
  return to_c(rect);
}

ManifoldVec2 manifold_rect_min(ManifoldRect *r) {
  return to_c((*from_c(r)).min);
}

ManifoldVec2 manifold_rect_max(ManifoldRect *r) {
  return to_c((*from_c(r)).max);
}

ManifoldVec2 manifold_rect_dimensions(ManifoldRect *r) {
  auto v = from_c(r)->Size();
  return {v.x, v.y};
}

ManifoldVec2 manifold_rect_center(ManifoldRect *r) {
  auto v = from_c(r)->Center();
  return {v.x, v.y};
}

double manifold_rect_scale(ManifoldRect *r) { return from_c(r)->Scale(); }

int manifold_rect_contains_pt(ManifoldRect *r, double x, double y) {
  auto rect = *from_c(r);
  auto p = vec2(x, y);
  return rect.Contains(p);
}

int manifold_rect_contains_rect(ManifoldRect *a, ManifoldRect *b) {
  auto outer = *from_c(a);
  auto inner = *from_c(b);
  return outer.Contains(inner);
}

void manifold_rect_include_pt(ManifoldRect *r, double x, double y) {
  auto rect = *from_c(r);
  auto p = vec2(x, y);
  rect.Union(p);
}

ManifoldRect *manifold_rect_union(void *mem, ManifoldRect *a, ManifoldRect *b) {
  auto rect = from_c(a)->Union(*from_c(b));
  return to_c(new (mem) Rect(rect));
}

ManifoldRect *manifold_rect_transform(void *mem, ManifoldRect *r, double x1,
                                      double y1, double x2, double y2,
                                      double x3, double y3) {
  auto mat = mat3x2(x1, y1, x2, y2, x3, y3);
  auto transformed = from_c(r)->Transform(mat);
  return to_c(new (mem) Rect(transformed));
}

ManifoldRect *manifold_rect_translate(void *mem, ManifoldRect *r, double x,
                                      double y) {
  auto rect = *from_c(r);
  auto p = vec2(x, y);
  auto translated = (*from_c(r)) + p;
  return to_c(new (mem) Rect(translated));
}

ManifoldRect *manifold_rect_mul(void *mem, ManifoldRect *r, double x,
                                double y) {
  auto rect = *from_c(r);
  auto p = vec2(x, y);
  auto scaled = (*from_c(r)) * p;
  return to_c(new (mem) Rect(scaled));
}

int manifold_rect_does_overlap_rect(ManifoldRect *a, ManifoldRect *r) {
  return from_c(a)->DoesOverlap(*from_c(r));
}

int manifold_rect_is_empty(ManifoldRect *r) { return from_c(r)->IsEmpty(); }

int manifold_rect_is_finite(ManifoldRect *r) { return from_c(r)->IsFinite(); }
#ifdef __cplusplus
}
#endif
