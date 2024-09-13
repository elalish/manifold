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

#pragma once
#include <cstring>
#include <vector>

#include "manifold/common.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"
#include "manifold/types.h"

using namespace manifold;
using ManifoldVec = std::vector<Manifold>;
using CrossSectionVec = std::vector<CrossSection>;

ManifoldManifold *to_c(manifold::Manifold *m);
ManifoldManifoldVec *to_c(ManifoldVec *ms);
ManifoldCrossSection *to_c(manifold::CrossSection *cs);
ManifoldCrossSectionVec *to_c(CrossSectionVec *csv);
ManifoldSimplePolygon *to_c(manifold::SimplePolygon *p);
ManifoldPolygons *to_c(manifold::Polygons *ps);
ManifoldMeshGL *to_c(manifold::MeshGL *m);
ManifoldBox *to_c(manifold::Box *m);
ManifoldRect *to_c(manifold::Rect *m);
ManifoldError to_c(manifold::Manifold::Error error);
ManifoldVec2 to_c(vec2 v);
ManifoldVec3 to_c(vec3 v);
ManifoldIVec3 to_c(ivec3 v);
ManifoldProperties to_c(manifold::Properties p);

const manifold::Manifold *from_c(ManifoldManifold *m);
ManifoldVec *from_c(ManifoldManifoldVec *ms);
const manifold::CrossSection *from_c(ManifoldCrossSection *cs);
CrossSectionVec *from_c(ManifoldCrossSectionVec *csv);
const manifold::SimplePolygon *from_c(ManifoldSimplePolygon *m);
const manifold::Polygons *from_c(ManifoldPolygons *m);
const manifold::MeshGL *from_c(ManifoldMeshGL *m);
OpType from_c(ManifoldOpType op);
CrossSection::FillRule from_c(ManifoldFillRule fillrule);
CrossSection::JoinType from_c(ManifoldJoinType jt);
const manifold::Box *from_c(ManifoldBox *m);
const manifold::Rect *from_c(ManifoldRect *r);
vec2 from_c(ManifoldVec2 v);
vec3 from_c(ManifoldVec3 v);
ivec3 from_c(ManifoldIVec3 v);
vec4 from_c(ManifoldVec4 v);

std::vector<vec3> vector_of_vec_array(ManifoldVec3 *vs, size_t length);
std::vector<ivec3> vector_of_vec_array(ManifoldIVec3 *vs, size_t length);
std::vector<vec4> vector_of_vec_array(ManifoldVec4 *vs, size_t length);

template <typename T>
std::vector<T> vector_of_array(T *ts, size_t length) {
  auto vec = std::vector<T>();
  for (size_t i = 0; i < length; ++i) {
    vec.push_back(ts[i]);
  }
  return vec;
}

template <typename T>
T *copy_data(void *mem, std::vector<T> v) {
  T *ts = reinterpret_cast<T *>(mem);
  memcpy(ts, v.data(), sizeof(T) * v.size());
  return ts;
}
