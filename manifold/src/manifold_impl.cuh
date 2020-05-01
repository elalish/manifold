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
  VecDH<int> vertLabel_;
  int numLabel_ = 1;
  VecDH<Halfedge> halfedge_;
  VecDH<Halfedge*> nextHalfedge_;
  VecDH<int> faceEdge_;

  VecDH<glm::vec3> vertNormal_;
  VecDH<glm::vec3> faceNormal_;
  Collider collider_;
  glm::mat4x3 transform_ = glm::mat4x3(1.0f);

  Impl() {}
  Impl(const Mesh&);
  enum class Shape { TETRAHEDRON, CUBE, OCTAHEDRON };
  Impl(Shape);
  void RemoveChaff();
  void Finish();
  void CreateHalfedges(const VecDH<glm::ivec3>& triVerts);
  void Update();
  void ApplyTransform() const;
  void ApplyTransform();
  void AssembleFaces() const;
  void AssembleFaces();
  bool Tri2Face() const;
  bool Tri2Face();
  bool Face2Tri();
  void Refine(int n);
  bool IsManifold() const;

  int NumVert() const { return vertPos_.size(); }
  int NumEdge() const { return halfedge_.size() / 2; }
  int NumFace() const { return faceEdge_.size() - 1; }
  std::pair<float, float> AreaVolume() const;
  void CalculateBBox();

  void SortVerts();
  void GetFaceBoxMorton(VecDH<Box>& faceBox, VecDH<uint32_t>& faceMorton) const;
  void SortFaces(VecDH<Box>& faceBox, VecDH<uint32_t>& faceMorton);
  void CalculateNormals();

  SparseIndices EdgeCollisions(const Impl& B) const;
  SparseIndices VertexCollisionsZ(const VecDH<glm::vec3>& vertsIn) const;

  static void NextEdges(Halfedge* nextEdge, const Halfedge* edgeBegin,
                        const Halfedge* edgeEnd);
  static Polygons Assemble(const Halfedge* edgeBegin, const Halfedge* edgeEnd,
                           const Halfedge* nextEdge,
                           std::function<glm::vec2(int)> vertProjection);
};
}  // namespace manifold