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
ManifoldBox *manifold_box(void *mem, double x1, double y1, double z1, double x2,
                          double y2, double z2) {
  auto p1 = vec3(x1, y1, z1);
  auto p2 = vec3(x2, y2, z2);
  auto box = new (mem) Box(p1, p2);
  return to_c(box);
}

ManifoldVec3 manifold_box_min(ManifoldBox *b) { return to_c((*from_c(b)).min); }

ManifoldVec3 manifold_box_max(ManifoldBox *b) { return to_c((*from_c(b)).max); }

ManifoldVec3 manifold_box_dimensions(ManifoldBox *b) {
  auto v = from_c(b)->Size();
  return {v.x, v.y, v.z};
}

ManifoldVec3 manifold_box_center(ManifoldBox *b) {
  auto v = from_c(b)->Center();
  return {v.x, v.y, v.z};
}

double manifold_box_scale(ManifoldBox *b) { return from_c(b)->Scale(); }

int manifold_box_contains_pt(ManifoldBox *b, double x, double y, double z) {
  auto p = vec3(x, y, z);
  return from_c(b)->Contains(p);
}

int manifold_box_contains_box(ManifoldBox *a, ManifoldBox *b) {
  auto outer = *from_c(a);
  auto inner = *from_c(b);
  return outer.Contains(inner);
}

void manifold_box_include_pt(ManifoldBox *b, double x, double y, double z) {
  auto box = *from_c(b);
  auto p = vec3(x, y, z);
  box.Union(p);
}

ManifoldBox *manifold_box_union(void *mem, ManifoldBox *a, ManifoldBox *b) {
  auto box = from_c(a)->Union(*from_c(b));
  return to_c(new (mem) Box(box));
}

ManifoldBox *manifold_box_transform(void *mem, ManifoldBox *b, double x1,
                                    double y1, double z1, double x2, double y2,
                                    double z2, double x3, double y3, double z3,
                                    double x4, double y4, double z4) {
  auto mat = mat4x3(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4);
  auto transformed = from_c(b)->Transform(mat);
  return to_c(new (mem) Box(transformed));
}

ManifoldBox *manifold_box_translate(void *mem, ManifoldBox *b, double x,
                                    double y, double z) {
  auto p = vec3(x, y, z);
  auto translated = (*from_c(b)) + p;
  return to_c(new (mem) Box(translated));
}

ManifoldBox *manifold_box_mul(void *mem, ManifoldBox *b, double x, double y,
                              double z) {
  auto p = vec3(x, y, z);
  auto scaled = (*from_c(b)) * p;
  return to_c(new (mem) Box(scaled));
}

int manifold_box_does_overlap_pt(ManifoldBox *b, double x, double y, double z) {
  auto p = vec3(x, y, z);
  return from_c(b)->DoesOverlap(p);
}

int manifold_box_does_overlap_box(ManifoldBox *a, ManifoldBox *b) {
  return from_c(a)->DoesOverlap(*from_c(b));
}

int manifold_box_is_finite(ManifoldBox *b) { return from_c(b)->IsFinite(); }
#ifdef __cplusplus
}
#endif
