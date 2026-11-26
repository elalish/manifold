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

#include <vector>

#include "manifold/common.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"
#include "manifold/types.h"

ManifoldManifold* to_c(manifold::Manifold* m) {
  return reinterpret_cast<ManifoldManifold*>(m);
}

ManifoldManifoldVec* to_c(ManifoldVec* ms) {
  return reinterpret_cast<ManifoldManifoldVec*>(ms);
}

ManifoldCrossSection* to_c(manifold::CrossSection* cs) {
  return reinterpret_cast<ManifoldCrossSection*>(cs);
}

ManifoldCrossSectionVec* to_c(CrossSectionVec* csv) {
  return reinterpret_cast<ManifoldCrossSectionVec*>(csv);
}

ManifoldRayHitVec* to_c(RayHitVec* v) {
  return reinterpret_cast<ManifoldRayHitVec*>(v);
}

ManifoldExecutionContext* to_c(ExecutionContext* ctx) {
  return reinterpret_cast<ManifoldExecutionContext*>(ctx);
}

ManifoldSimplePolygon* to_c(manifold::SimplePolygon* m) {
  return reinterpret_cast<ManifoldSimplePolygon*>(m);
}

ManifoldPolygons* to_c(manifold::Polygons* m) {
  return reinterpret_cast<ManifoldPolygons*>(m);
}

ManifoldMeshGL* to_c(manifold::MeshGL* m) {
  return reinterpret_cast<ManifoldMeshGL*>(m);
}

ManifoldMeshGL64* to_c(manifold::MeshGL64* m) {
  return reinterpret_cast<ManifoldMeshGL64*>(m);
}

// ManifoldError matches Manifold::Error value for value, so the two convert
// with a cast.
using Error = manifold::Manifold::Error;
static_assert(MANIFOLD_NO_ERROR == static_cast<int>(Error::NoError));
static_assert(MANIFOLD_NON_FINITE_VERTEX ==
              static_cast<int>(Error::NonFiniteVertex));
static_assert(MANIFOLD_NOT_MANIFOLD == static_cast<int>(Error::NotManifold));
static_assert(MANIFOLD_VERTEX_OUT_OF_BOUNDS ==
              static_cast<int>(Error::VertexOutOfBounds));
static_assert(MANIFOLD_PROPERTIES_WRONG_LENGTH ==
              static_cast<int>(Error::PropertiesWrongLength));
static_assert(MANIFOLD_MISSING_POSITION_PROPERTIES ==
              static_cast<int>(Error::MissingPositionProperties));
static_assert(MANIFOLD_MERGE_VECTORS_DIFFERENT_LENGTHS ==
              static_cast<int>(Error::MergeVectorsDifferentLengths));
static_assert(MANIFOLD_MERGE_INDEX_OUT_OF_BOUNDS ==
              static_cast<int>(Error::MergeIndexOutOfBounds));
static_assert(MANIFOLD_TRANSFORM_WRONG_LENGTH ==
              static_cast<int>(Error::TransformWrongLength));
static_assert(MANIFOLD_RUN_INDEX_WRONG_LENGTH ==
              static_cast<int>(Error::RunIndexWrongLength));
static_assert(MANIFOLD_FACE_ID_WRONG_LENGTH ==
              static_cast<int>(Error::FaceIDWrongLength));
static_assert(MANIFOLD_INVALID_CONSTRUCTION ==
              static_cast<int>(Error::InvalidConstruction));
static_assert(MANIFOLD_RESULT_TOO_LARGE ==
              static_cast<int>(Error::ResultTooLarge));
static_assert(MANIFOLD_INVALID_TANGENTS ==
              static_cast<int>(Error::InvalidTangents));
static_assert(MANIFOLD_CANCELLED == static_cast<int>(Error::Cancelled));

ManifoldError to_c(manifold::Manifold::Error error) {
  return static_cast<ManifoldError>(error);
}

ManifoldBox* to_c(manifold::Box* m) {
  return reinterpret_cast<ManifoldBox*>(m);
}

ManifoldRect* to_c(manifold::Rect* m) {
  return reinterpret_cast<ManifoldRect*>(m);
}

ManifoldVec2 to_c(vec2 v) { return {v.x, v.y}; }

ManifoldVec3 to_c(vec3 v) { return {v.x, v.y, v.z}; }

ManifoldTriangulation* to_c(std::vector<int>* m) {
  return reinterpret_cast<ManifoldTriangulation*>(m);
}

const manifold::Manifold* from_c(const ManifoldManifold* m) {
  return reinterpret_cast<const manifold::Manifold*>(m);
}

ManifoldVec* from_c(ManifoldManifoldVec* ms) {
  return reinterpret_cast<ManifoldVec*>(ms);
}

const ManifoldVec* from_c(const ManifoldManifoldVec* ms) {
  return reinterpret_cast<const ManifoldVec*>(ms);
}

const manifold::CrossSection* from_c(const ManifoldCrossSection* cs) {
  return reinterpret_cast<const manifold::CrossSection*>(cs);
}

CrossSectionVec* from_c(ManifoldCrossSectionVec* csv) {
  return reinterpret_cast<CrossSectionVec*>(csv);
}

const CrossSectionVec* from_c(const ManifoldCrossSectionVec* csv) {
  return reinterpret_cast<const CrossSectionVec*>(csv);
}

const RayHitVec* from_c(const ManifoldRayHitVec* v) {
  return reinterpret_cast<const RayHitVec*>(v);
}

ExecutionContext* from_c(ManifoldExecutionContext* ctx) {
  return reinterpret_cast<ExecutionContext*>(ctx);
}

const ExecutionContext* from_c(const ManifoldExecutionContext* ctx) {
  return reinterpret_cast<const ExecutionContext*>(ctx);
}

const manifold::SimplePolygon* from_c(const ManifoldSimplePolygon* m) {
  return reinterpret_cast<const manifold::SimplePolygon*>(m);
}

const manifold::Polygons* from_c(const ManifoldPolygons* m) {
  return reinterpret_cast<const manifold::Polygons*>(m);
}

const manifold::MeshGL* from_c(const ManifoldMeshGL* m) {
  return reinterpret_cast<const manifold::MeshGL*>(m);
}

const manifold::MeshGL64* from_c(const ManifoldMeshGL64* m) {
  return reinterpret_cast<const manifold::MeshGL64*>(m);
}

OpType from_c(ManifoldOpType optype) {
  auto op = OpType::Add;
  switch (optype) {
    case MANIFOLD_ADD:
      break;
    case MANIFOLD_SUBTRACT:
      op = OpType::Subtract;
      break;
    case MANIFOLD_INTERSECT:
      op = OpType::Intersect;
      break;
  };
  return op;
}

CrossSection::JoinType from_c(ManifoldJoinType join_type) {
  auto jt = CrossSection::JoinType::Square;
  switch (join_type) {
    case MANIFOLD_JOIN_TYPE_SQUARE:
      break;
    case MANIFOLD_JOIN_TYPE_ROUND:
      jt = CrossSection::JoinType::Round;
      break;
    case MANIFOLD_JOIN_TYPE_MITER:
      jt = CrossSection::JoinType::Miter;
      break;
    case MANIFOLD_JOIN_TYPE_BEVEL:
      jt = CrossSection::JoinType::Bevel;
      break;
  };
  return jt;
}

CircularQuality from_c(const ManifoldCircularQuality* q) {
  if (q == nullptr) return {};
  return {q->segments, q->min_angle, q->min_edge_length};
}

manifold::Box* from_c(ManifoldBox* m) {
  return reinterpret_cast<manifold::Box*>(m);
}

const manifold::Box* from_c(const ManifoldBox* m) {
  return reinterpret_cast<const manifold::Box*>(m);
}

manifold::Rect* from_c(ManifoldRect* m) {
  return reinterpret_cast<manifold::Rect*>(m);
}

const manifold::Rect* from_c(const ManifoldRect* m) {
  return reinterpret_cast<const manifold::Rect*>(m);
}

vec2 from_c(ManifoldVec2 v) { return vec2(v.x, v.y); }

vec3 from_c(ManifoldVec3 v) { return vec3(v.x, v.y, v.z); }

const std::vector<int>* from_c(const ManifoldTriangulation* m) {
  return reinterpret_cast<const std::vector<int>*>(m);
}