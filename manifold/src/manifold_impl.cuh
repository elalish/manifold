// Copyright 2019 Emmett Lalish
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
#include "collider.cuh"
#include "manifold.h"
#include "sparse.cuh"
#include "utils.cuh"
#include "vec_dh.cuh"

namespace manifold {

struct Manifold::Impl {
  Box bBox_;
  VecDH<glm::vec3> vertPos_;
  VecDH<Halfedge> halfedge_;
  VecDH<glm::vec3> vertNormal_;
  VecDH<glm::vec3> faceNormal_;
  Collider collider_;
  glm::mat4x3 transform_ = glm::mat4x3(1.0f);

  Impl() {}
  Impl(const Mesh&);
  enum class Shape { TETRAHEDRON, CUBE, OCTAHEDRON };
  Impl(Shape);

  void CreateHalfedges(const VecDH<glm::ivec3>& triVerts);
  void CreateAndFixHalfedges(const VecDH<glm::ivec3>& triVerts);
  void SplitNonmanifoldVerts();
  void Finish();
  void Update();
  void ApplyTransform() const;
  void ApplyTransform();
  void Refine(int n);

  bool IsEmpty() const { return NumVert() == 0; }
  int NumVert() const { return vertPos_.size(); }
  int NumEdge() const { return halfedge_.size() / 2; }
  int NumTri() const { return halfedge_.size() / 3; }
  Properties GetProperties() const;
  void CalculateBBox();
  bool IsManifold() const;

  void SortVerts();
  void ReindexVerts(const VecDH<int>& vertNew2Old, int numOldVert);
  void SortFaces(VecDH<Box>& faceBox, VecDH<uint32_t>& faceMorton);
  void GatherFaces(const VecDH<Halfedge>& oldHalfedge,
                   const VecDH<int>& faceNew2Old);
  void CalculateNormals();
  void Face2Tri(const VecDH<int>& faceEdge);

  SparseIndices EdgeCollisions(const Impl& B) const;
  SparseIndices VertexCollisionsZ(const VecDH<glm::vec3>& vertsIn) const;
  void GetFaceBoxMorton(VecDH<Box>& faceBox, VecDH<uint32_t>& faceMorton) const;
  Polygons Face2Polygons(int face, glm::mat3x2 projection,
                         const VecH<int>& faceEdge) const;
};
}  // namespace manifold