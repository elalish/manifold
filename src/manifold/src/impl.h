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
#include "quickhull.h"
#include "shared.h"
#include "sparse.h"
#include "utils.h"
#include "vec.h"

namespace manifold {

/** @ingroup Private */
struct Manifold::Impl {
  struct Relation {
    int originalID = -1;
    mat4x3 transform = mat4x3(1);
    bool backSide = false;
  };
  struct MeshRelationD {
    /// The originalID of this Manifold if it is an original; -1 otherwise.
    int originalID = -1;
    int numProp = 0;
    Vec<double> properties;
    std::map<int, Relation> meshIDtransform;
    Vec<TriRef> triRef;
    Vec<ivec3> triProperties;
  };
  struct BaryIndices {
    int tri, start4, end4;
  };

  Box bBox_;
  double precision_ = -1;
  Error status_ = Error::NoError;
  Vec<vec3> vertPos_;
  Vec<Halfedge> halfedge_;
  Vec<vec3> vertNormal_;
  Vec<vec3> faceNormal_;
  Vec<vec4> halfedgeTangent_;
  MeshRelationD meshRelation_;
  Collider collider_;

  static std::atomic<uint32_t> meshIDCounter_;
  static uint32_t ReserveIDs(uint32_t);

  Impl() {}
  enum class Shape { Tetrahedron, Cube, Octahedron };
  Impl(Shape, const mat4x3 = mat4x3(1));

  Impl(const MeshGL&, std::vector<float> propertyTolerance = {});
  Impl(const Mesh&, const MeshRelationD& relation,
       const std::vector<double>& propertyTolerance = {},
       bool hasFaceIDs = false);

  inline void ForVert(int halfedge, std::function<void(int halfedge)> func) {
    int current = halfedge;
    do {
      current = NextHalfedge(halfedge_[current].pairedHalfedge);
      func(current);
    } while (current != halfedge);
  }

  template <typename T>
  void ForVert(
      int halfedge, std::function<T(int halfedge)> transform,
      std::function<void(int halfedge, const T& here, T& next)> binaryOp) {
    T here = transform(halfedge);
    int current = halfedge;
    do {
      const int nextHalfedge = NextHalfedge(halfedge_[current].pairedHalfedge);
      T next = transform(nextHalfedge);
      binaryOp(current, here, next);
      here = next;
      current = nextHalfedge;
    } while (current != halfedge);
  }

  void CreateFaces(const std::vector<double>& propertyTolerance = {});
  void RemoveUnreferencedVerts();
  void InitializeOriginal();
  void CreateHalfedges(const Vec<ivec3>& triVerts);
  void CalculateNormals();
  void IncrementMeshIDs();

  void Update();
  void MarkFailure(Error status);
  void Warp(std::function<void(vec3&)> warpFunc);
  void WarpBatch(std::function<void(VecView<vec3>)> warpFunc);
  Impl Transform(const mat4x3& transform) const;
  SparseIndices EdgeCollisions(const Impl& B, bool inverted = false) const;
  SparseIndices VertexCollisionsZ(VecView<const vec3> vertsIn,
                                  bool inverted = false) const;

  bool IsEmpty() const { return NumTri() == 0; }
  size_t NumVert() const { return vertPos_.size(); }
  size_t NumEdge() const { return halfedge_.size() / 2; }
  size_t NumTri() const { return halfedge_.size() / 3; }
  size_t NumProp() const { return meshRelation_.numProp; }
  size_t NumPropVert() const {
    return NumProp() == 0 ? NumVert()
                          : meshRelation_.properties.size() / NumProp();
  }

  // properties.cu
  Properties GetProperties() const;
  void CalculateCurvature(int gaussianIdx, int meanIdx);
  void CalculateBBox();
  bool IsFinite() const;
  bool IsIndexInBounds(VecView<const ivec3> triVerts) const;
  void SetPrecision(double minPrecision = -1);
  bool IsManifold() const;
  bool Is2Manifold() const;
  bool MatchesTriNormals() const;
  int NumDegenerateTris() const;
  double MinGap(const Impl& other, double searchLength) const;

  // sort.cu
  void Finish();
  void SortVerts();
  void ReindexVerts(const Vec<int>& vertNew2Old, size_t numOldVert);
  void CompactProps();
  void GetFaceBoxMorton(Vec<Box>& faceBox, Vec<uint32_t>& faceMorton) const;
  void SortFaces(Vec<Box>& faceBox, Vec<uint32_t>& faceMorton);
  void GatherFaces(const Vec<int>& faceNew2Old);
  void GatherFaces(const Impl& old, const Vec<int>& faceNew2Old);

  // face_op.cu
  void Face2Tri(const Vec<int>& faceEdge, const Vec<TriRef>& halfedgeRef);
  PolygonsIdx Face2Polygons(VecView<Halfedge>::IterC start,
                            VecView<Halfedge>::IterC end,
                            mat3x2 projection) const;
  Polygons Slice(double height) const;
  Polygons Project() const;

  // edge_op.cu
  void CleanupTopology();
  void SimplifyTopology();
  void DedupeEdge(int edge);
  void CollapseEdge(int edge, std::vector<int>& edges);
  void RecursiveEdgeSwap(int edge, int& tag, std::vector<int>& visited,
                         std::vector<int>& edgeSwapStack,
                         std::vector<int>& edges);
  void RemoveIfFolded(int edge);
  void PairUp(int edge0, int edge1);
  void UpdateVert(int vert, int startEdge, int endEdge);
  void FormLoop(int current, int end);
  void CollapseTri(const ivec3& triEdge);
  void SplitPinchedVerts();

  // subdivision.cpp
  int GetNeighbor(int tri) const;
  ivec4 GetHalfedges(int tri) const;
  BaryIndices GetIndices(int halfedge) const;
  void FillRetainedVerts(Vec<Barycentric>& vertBary) const;
  Vec<Barycentric> Subdivide(std::function<int(vec3)>);

  // smoothing.cpp
  bool IsInsideQuad(int halfedge) const;
  bool IsMarkedInsideQuad(int halfedge) const;
  vec3 GetNormal(int halfedge, int normalIdx) const;
  vec4 TangentFromNormal(const vec3& normal, int halfedge) const;
  std::vector<Smoothness> UpdateSharpenedEdges(
      const std::vector<Smoothness>&) const;
  Vec<bool> FlatFaces() const;
  Vec<int> VertFlatFace(const Vec<bool>&) const;
  Vec<int> VertHalfedge() const;
  std::vector<Smoothness> SharpenEdges(double minSharpAngle,
                                       double minSmoothness) const;
  void SharpenTangent(int halfedge, double smoothness);
  void SetNormals(int normalIdx, double minSharpAngle);
  void LinearizeFlatTangents();
  void DistributeTangents(const Vec<bool>& fixedHalfedges);
  void CreateTangents(int normalIdx);
  void CreateTangents(std::vector<Smoothness>);
  void Refine(std::function<int(vec3)>);

  // quickhull.cpp
  void Hull(VecView<vec3> vertPos);
};
}  // namespace manifold
