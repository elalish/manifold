// Copyright 2021 Emmett Lalish
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
#include "collider.h"
#include "manifold.h"
#include "shared.h"
#include "sparse.h"
#include "utils.h"
#include "vec_dh.h"

namespace manifold {

/** @ingroup Private */
struct Manifold::Impl {
  struct MeshRelationD {
    VecDH<glm::vec3> barycentric;
    /// meshID in BaryRef has different meaning in MeshRelation and
    /// MeshRelationD:
    /// - In `MeshRelation`: The original mesh triangle index.
    /// - In `MeshRelationD`: The original mesh triangle index =
    /// `originalID[meshID]`
    ///
    /// @note Triangles coming from different manifolds should have different mesh
    /// ID, otherwise `SimplifyTopology` will not work properly.
    VecDH<BaryRef> triBary;
    /// meshID to originalID mapping.
    std::unordered_map<int, int> originalID;
  };

  Box bBox_;
  float precision_ = -1;
  VecDH<glm::vec3> vertPos_;
  VecDH<Halfedge> halfedge_;
  VecDH<glm::vec3> vertNormal_;
  VecDH<glm::vec3> faceNormal_;
  VecDH<glm::vec4> halfedgeTangent_;
  MeshRelationD meshRelation_;
  Collider collider_;
  glm::mat4x3 transform_ = glm::mat4x3(1.0f);

  static std::atomic<int> meshIDCounter_;

  Impl() {}
  enum class Shape { TETRAHEDRON, CUBE, OCTAHEDRON };
  Impl(Shape);

  Impl(const Mesh&,
       const std::vector<glm::ivec3>& triProperties = std::vector<glm::ivec3>(),
       const std::vector<float>& properties = std::vector<float>(),
       const std::vector<float>& propertyTolerance = std::vector<float>());

  int InitializeNewReference(
      const std::vector<glm::ivec3>& triProperties = std::vector<glm::ivec3>(),
      const std::vector<float>& properties = std::vector<float>(),
      const std::vector<float>& propertyTolerance = std::vector<float>());

  void ReinitializeReference(int meshID);
  void CreateHalfedges(const VecDH<glm::ivec3>& triVerts);
  void CreateAndFixHalfedges(const VecDH<glm::ivec3>& triVerts);
  void CalculateNormals();
  void UpdateMeshIDs(VecDH<int> &meshIDs, VecDH<int> &originalIDs, int startTri=0, int n=-1, int startID=0);

  void Update();
  void ApplyTransform() const;
  void ApplyTransform();
  SparseIndices EdgeCollisions(const Impl& B) const;
  SparseIndices VertexCollisionsZ(const VecDH<glm::vec3>& vertsIn) const;

  bool IsEmpty() const { return NumVert() == 0; }
  int NumVert() const { return vertPos_.size(); }
  int NumEdge() const { return halfedge_.size() / 2; }
  int NumTri() const { return halfedge_.size() / 3; }
  // properties.cu
  Properties GetProperties() const;
  Curvature GetCurvature() const;
  void CalculateBBox();
  void SetPrecision(float minPrecision = -1);
  bool IsManifold() const;
  bool MatchesTriNormals() const;
  int NumDegenerateTris() const;

  // sort.cu
  void Finish();
  void SortVerts();
  void ReindexVerts(const VecDH<int>& vertNew2Old, int numOldVert);
  void GetFaceBoxMorton(VecDH<Box>& faceBox, VecDH<uint32_t>& faceMorton) const;
  void SortFaces(VecDH<Box>& faceBox, VecDH<uint32_t>& faceMorton);
  void GatherFaces(const VecDH<int>& faceNew2Old);
  void GatherFaces(const Impl& old, const VecDH<int>& faceNew2Old);

  // face_op.cu
  void Face2Tri(const VecDH<int>& faceEdge, const VecDH<BaryRef>& faceRef,
                const VecDH<int>& halfedgeBary);
  Polygons Face2Polygons(int face, glm::mat3x2 projection,
                         const VecDH<int>& faceEdge) const;

  // edge_op.cu
  void SimplifyTopology();
  void CollapseEdge(int edge);
  void RecursiveEdgeSwap(int edge);
  void RemoveIfFolded(int edge);
  void PairUp(int edge0, int edge1);
  void UpdateVert(int vert, int startEdge, int endEdge);
  void FormLoop(int current, int end);
  void CollapseTri(const glm::ivec3& triEdge);

  // smoothing.cu
  void CreateTangents(const std::vector<Smoothness>&);
  MeshRelationD Subdivide(int n);
  void Refine(int n);
};
}  // namespace manifold
