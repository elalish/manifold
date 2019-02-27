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
#include "mesh.h"
#include "utils.cuh"
#include "vec_dh.cuh"

namespace manifold {

struct Mesh::Impl {
  Box bBox_;
  VecDH<glm::vec3> vertPos_;
  VecDH<EdgeVertsD> edgeVerts_;
  VecDH<TriVerts> triVerts_;
  VecDH<TriEdges> triEdges_;
  Collider collider_;

  Impl() {}
  Impl(const MeshHost&);
  void Append2Host(MeshHost&) const;
  void Transform(const glm::mat4&);
  void TranslateScale(const glm::mat4&);
  bool IsValid() const;
  // Mesh Boolean(const Impl& second, OpType op, int max_overlaps) const;

  int NumVert() const { return vertPos_.size(); }
  int NumEdge() const { return edgeVerts_.size(); }
  int NumTri() const { return triVerts_.size(); }
  void CalculateBBox();

  void SortVerts();
  void CreateEdges();
  void SortHalfedges(VecDH<EdgeVertsD>& halfEdges, VecDH<int>& dir);
  VecDH<Box> GetEdgeBox() const;
  void GetTriBoxMorton(VecDH<Box>& triBox, VecDH<uint32_t>& triMorton) const;
  void SortTris(VecDH<Box>& triBox, VecDH<uint32_t>& triMorton);

  void EdgeCollisions(VecDH<int>& edgesB, VecDH<int>& tris,
                      const Mesh::Impl& B) const;
  void VertexCollisionsZ(VecDH<int>& vertsOut, VecDH<int>& tris,
                         const VecDH<glm::vec3>& vertsIn) const;
};
Mesh::Mesh(const MeshHost& mesh) : pImpl_{std::make_unique<Impl>(mesh)} {}
Mesh::~Mesh() = default;
Mesh::Mesh(Mesh&&) = default;
Mesh& Mesh::operator=(Mesh&&) = default;

Mesh::Mesh(const Mesh& other) : pImpl_(nullptr) {
  if (other.pImpl_) pImpl_ = std::make_unique<Impl>(*other.pImpl_);
  transform_ = other.transform_;
}

Mesh& Mesh::operator=(const Mesh& other) {
  if (!other.pImpl_)
    pImpl_.reset();
  else if (!pImpl_)
    pImpl_ = std::make_unique<Impl>(*other.pImpl_);
  else
    *pImpl_ = *other.pImpl_;
  transform_ = other.transform_;
  return *this;
}
}  // namespace manifold