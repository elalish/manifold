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
using RayHitVec = std::vector<RayHit>;

ManifoldManifold* to_c(manifold::Manifold* m);
ManifoldManifoldVec* to_c(ManifoldVec* ms);
ManifoldCrossSection* to_c(manifold::CrossSection* cs);
ManifoldCrossSectionVec* to_c(CrossSectionVec* csv);
ManifoldSimplePolygon* to_c(manifold::SimplePolygon* p);
ManifoldPolygons* to_c(manifold::Polygons* ps);
ManifoldMeshGL* to_c(manifold::MeshGL* m);
ManifoldMeshGL64* to_c(manifold::MeshGL64* m);
ManifoldBox* to_c(manifold::Box* m);
ManifoldRect* to_c(manifold::Rect* m);
ManifoldError to_c(manifold::Manifold::Error error);
ManifoldVec2 to_c(vec2 v);
ManifoldVec3 to_c(vec3 v);
ManifoldIVec3 to_c(ivec3 v);
ManifoldTriangulation* to_c(std::vector<int>* m);
ManifoldRayHitVec* to_c(RayHitVec* v);
ManifoldExecutionContext* to_c(ExecutionContext* ctx);

const manifold::Manifold* from_c(const ManifoldManifold* m);
const ManifoldVec* from_c(const ManifoldManifoldVec* ms);
ManifoldVec* from_c(ManifoldManifoldVec* ms);
const manifold::CrossSection* from_c(const ManifoldCrossSection* cs);
CrossSectionVec* from_c(ManifoldCrossSectionVec* csv);
const CrossSectionVec* from_c(const ManifoldCrossSectionVec* csv);
const manifold::SimplePolygon* from_c(const ManifoldSimplePolygon* m);
const manifold::Polygons* from_c(const ManifoldPolygons* m);
const manifold::MeshGL* from_c(const ManifoldMeshGL* m);
const manifold::MeshGL64* from_c(const ManifoldMeshGL64* m);
OpType from_c(ManifoldOpType op);
CrossSection::JoinType from_c(ManifoldJoinType jt);
manifold::Box* from_c(ManifoldBox* m);
const manifold::Box* from_c(const ManifoldBox* m);
manifold::Rect* from_c(ManifoldRect* r);
const manifold::Rect* from_c(const ManifoldRect* r);
vec2 from_c(ManifoldVec2 v);
vec3 from_c(ManifoldVec3 v);
ivec3 from_c(ManifoldIVec3 v);
vec4 from_c(ManifoldVec4 v);
const std::vector<int>* from_c(const ManifoldTriangulation* m);
const RayHitVec* from_c(const ManifoldRayHitVec* v);
const ExecutionContext* from_c(const ManifoldExecutionContext* ctx);
ExecutionContext* from_c(ManifoldExecutionContext* ctx);

template <typename T>
std::vector<T> vector_of_array(const T* ts, size_t length) {
  auto vec = std::vector<T>();
  for (size_t i = 0; i < length; ++i) {
    vec.push_back(ts[i]);
  }
  return vec;
}
