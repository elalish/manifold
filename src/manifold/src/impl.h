// Copyright 2021 The Manifold Authors.
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
#include <map>

#include "collider.h"
#include "manifold.h"
#include "optional_assert.h"
#include "polygon.h"
#include "shared.h"
#include "sparse.h"
#include "utils.h"
#include "vec_dh.h"

namespace manifold {

/** @ingroup Private */
struct Manifold::Impl {
  struct Relation {
    int originalID = -1;
    glm::mat4x3 transform = glm::mat4x3(1);
    bool backSide = false;
  };
  struct MeshRelationD {
    /// The originalID of this Manifold if it is an original; -1 otherwise.
    int originalID = -1;
    int numProp = 0;
    VecDH<float> properties;
    std::map<int, Relation> meshIDtransform;
    VecDH<TriRef> triRef;
    VecDH<glm::ivec3> triProperties;
  };

  Box bBox_;
  float precision_ = -1;
  Error status_ = Error::NoError;
  VecDH<glm::vec3> vertPos_;
  VecDH<Halfedge> halfedge_;
  VecDH<glm::vec3> vertNormal_;
  VecDH<glm::vec3> faceNormal_;
  VecDH<glm::vec4> halfedgeTangent_;
  MeshRelationD meshRelation_;
  Collider collider_;

  static std::atomic<uint32_t> meshIDCounter_;
  static uint32_t ReserveIDs(uint32_t);

  Impl() {}
  enum class Shape { Tetrahedron, Cube, Octahedron };
  Impl(Shape);

  Impl(const MeshGL&, std::vector<float> propertyTolerance = {});
  Impl(const Mesh&, const MeshRelationD& relation,
       const std::vector<float>& propertyTolerance = {},
       bool hasFaceIDs = false);

  void CreateFaces(const std::vector<float>& propertyTolerance = {});
  void RemoveUnreferencedVerts(VecDH<glm::ivec3>& triVerts);
  void InitializeOriginal();
  void CreateHalfedges(const VecDH<glm::ivec3>& triVerts);
  void CalculateNormals();
  void IncrementMeshIDs();

  void Update();
  void MarkFailure(Error status);
  void Warp(std::function<void(glm::vec3&)> warpFunc);
  Impl Transform(const glm::mat4x3& transform) const;
  SparseIndices EdgeCollisions(const Impl& B) const;
  SparseIndices VertexCollisionsZ(const VecDH<glm::vec3>& vertsIn) const;

  bool IsEmpty() const { return NumVert() == 0; }
  int NumVert() const { return vertPos_.size(); }
  int NumEdge() const { return halfedge_.size() / 2; }
  int NumTri() const { return halfedge_.size() / 3; }
  int NumProp() const { return meshRelation_.numProp; }
  int NumPropVert() const {
    return NumProp() == 0 ? NumVert()
                          : meshRelation_.properties.size() / NumProp();
  }

  // properties.cu
  Properties GetProperties() const;
  void CalculateCurvature(int gaussianIdx, int meanIdx);
  void CalculateBBox();
  bool IsFinite() const;
  bool IsIndexInBounds(const VecDH<glm::ivec3>& triVerts) const;
  void SetPrecision(float minPrecision = -1);
  bool IsManifold() const;
  bool Is2Manifold() const;
  bool MatchesTriNormals() const;
  int NumDegenerateTris() const;

  // sort.cu
  void Finish();
  void SortVerts();
  void ReindexVerts(const VecDH<int>& vertNew2Old, int numOldVert);
  void CompactProps();
  void GetFaceBoxMorton(VecDH<Box>& faceBox, VecDH<uint32_t>& faceMorton) const;
  void SortFaces(VecDH<Box>& faceBox, VecDH<uint32_t>& faceMorton);
  void GatherFaces(const VecDH<int>& faceNew2Old);
  void GatherFaces(const Impl& old, const VecDH<int>& faceNew2Old);

  // face_op.cu
  void Face2Tri(const VecDH<int>& faceEdge, const VecDH<TriRef>& halfedgeRef);
  PolygonsIdx Face2Polygons(int face, glm::mat3x2 projection,
                            const VecDH<int>& faceEdge) const;

  // edge_op.cu
  void SimplifyTopology();
  void DedupeEdge(int edge);
  void CollapseEdge(int edge);
  void RecursiveEdgeSwap(int edge);
  void RemoveIfFolded(int edge);
  void PairUp(int edge0, int edge1);
  void UpdateVert(int vert, int startEdge, int endEdge);
  void FormLoop(int current, int end);
  void CollapseTri(const glm::ivec3& triEdge);
  void SplitPinchedVerts();

  // smoothing.cu
  void CreateTangents(const std::vector<Smoothness>&);
  VecDH<Barycentric> Subdivide(int n);
  void Refine(int n);
};
}  // namespace manifold
