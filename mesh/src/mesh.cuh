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
#include "sparse.cuh"
#include "utils.cuh"
#include "vec_dh.cuh"

namespace manifold {

using EdgeVertsD = thrust::pair<int, int>;
struct EdgeTrisD {
  int right, left;
};

inline std::ostream& operator<<(std::ostream& stream, const EdgeVertsD& edge) {
  return stream << edge.first << ", " << edge.second;
}

struct Mesh::Impl {
  Box bBox_;
  VecDH<glm::vec3> vertPos_;
  VecDH<EdgeVertsD> edgeVerts_;
  VecDH<EdgeTrisD> edgeTris_;
  VecDH<TriVerts> triVerts_;
  VecDH<TriEdges> triEdges_;
  Collider collider_;

  Impl() {}
  Impl(const MeshHost&);
  enum class Shape { TETRAHEDRON, CUBE, OCTAHEDRON };
  Impl(Shape);
  void Finish();
  void Append2Host(MeshHost&) const;
  void Transform(const glm::mat4&);
  void TranslateScale(const glm::mat4&);
  bool IsValid() const;

  int NumVert() const { return vertPos_.size(); }
  int NumEdge() const { return edgeVerts_.size(); }
  int NumTri() const { return triVerts_.size(); }
  glm::vec3 GetTriNormal(int tri) const;
  void CalculateBBox();

  void SortVerts();
  void CreateEdges();
  void SortHalfedges(VecDH<EdgeVertsD>& halfEdges, VecDH<int>& dir);
  VecDH<Box> GetEdgeBox() const;
  void GetTriBoxMorton(VecDH<Box>& triBox, VecDH<uint32_t>& triMorton) const;
  void SortTris(VecDH<Box>& triBox, VecDH<uint32_t>& triMorton);

  SparseIndices EdgeCollisions(const Mesh::Impl& B) const;
  SparseIndices VertexCollisionsZ(const VecDH<glm::vec3>& vertsIn) const;
};
}  // namespace manifold