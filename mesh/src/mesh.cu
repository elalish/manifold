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

#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <algorithm>

// #include "boolean3D.cuh"
#include "mesh.cuh"

namespace {
using namespace manifold;

struct PosMin
    : public thrust::binary_function<glm::vec3, glm::vec3, glm::vec3> {
  __host__ __device__ glm::vec3 operator()(glm::vec3 a, glm::vec3 b) {
    return glm::min(a, b);
  }
};

struct PosMax
    : public thrust::binary_function<glm::vec3, glm::vec3, glm::vec3> {
  __host__ __device__ glm::vec3 operator()(glm::vec3 a, glm::vec3 b) {
    return glm::max(a, b);
  }
};

struct Transform3 {
  const glm::mat4 T;

  __host__ __device__ void operator()(glm::vec3& position) {
    position = glm::vec3(T * glm::vec4(position, 1.0f));
  }
};

__host__ __device__ uint32_t SpreadBits3(uint32_t v) {
  v = 0xFF0000FFu & (v * 0x00010001u);
  v = 0x0F00F00Fu & (v * 0x00000101u);
  v = 0xC30C30C3u & (v * 0x00000011u);
  v = 0x49249249u & (v * 0x00000005u);
  return v;
}

__host__ __device__ uint32_t MortonCode(glm::vec3 position, Box bBox) {
  glm::vec3 xyz = (position - bBox.min) / (bBox.max - bBox.min);
  xyz = glm::min(glm::vec3(1023.0f), glm::max(glm::vec3(0.0f), 1024.0f * xyz));
  uint32_t x = SpreadBits3(static_cast<uint32_t>(xyz.x));
  uint32_t y = SpreadBits3(static_cast<uint32_t>(xyz.y));
  uint32_t z = SpreadBits3(static_cast<uint32_t>(xyz.z));
  return x * 4 + y * 2 + z;
}

struct Morton {
  const Box bBox;

  __host__ __device__ void operator()(
      thrust::tuple<uint32_t&, const glm::vec3&> inout) {
    glm::vec3 position = thrust::get<1>(inout);
    thrust::get<0>(inout) = MortonCode(position, bBox);
  }
};

struct Reindex {
  const int* indexInv_;

  __host__ __device__ void operator()(TriVerts& triVerts) {
    for (int i : {0, 1, 2}) triVerts[i] = indexInv_[triVerts[i]];
  }
};

struct TriMortonBox {
  const glm::vec3* vertPos;
  const Box bBox;

  __host__ __device__ void operator()(
      thrust::tuple<uint32_t&, Box&, const TriVerts&> inout) {
    const TriVerts& triVerts = thrust::get<2>(inout);
    glm::vec3 center =
        (vertPos[triVerts[0]] + vertPos[triVerts[1]] + vertPos[triVerts[2]]) /
        3.0f;
    thrust::get<0>(inout) = MortonCode(center, bBox);
    thrust::get<1>(inout) = Box(vertPos[triVerts[0]], vertPos[triVerts[1]]);
    thrust::get<1>(inout).Union(vertPos[triVerts[2]]);
  }
};

struct EdgeBox {
  const glm::vec3* vertPos;

  __host__ __device__ void operator()(thrust::tuple<Box&, EdgeVertsD> inout) {
    EdgeVertsD edgeVerts = thrust::get<1>(inout);
    thrust::get<0>(inout) =
        Box(vertPos[edgeVerts.first], vertPos[edgeVerts.second]);
  }
};

struct MakeHalfedges {
  int i, j;

  __host__ __device__ void operator()(
      thrust::tuple<TriEdges&, int&, EdgeVertsD&, const TriVerts&> inout) {
    const TriVerts& in = thrust::get<3>(inout);
    int V1 = in[i];
    int V2 = in[j];
    TriEdges& triEdges = thrust::get<0>(inout);
    int& dir = thrust::get<1>(inout);
    EdgeVertsD& edgeVerts = thrust::get<2>(inout);
    if (V1 < V2) {  // forward
      dir = 1;
      edgeVerts = thrust::make_pair(V1, V2);
    } else {  // backward
      dir = -1;
      edgeVerts = thrust::make_pair(V2, V1);
    }
    triEdges[i] = EdgeIdx(0, dir);
  }
};

struct AssignEdges {
  int i;

  __host__ __device__ void operator()(thrust::tuple<TriEdges&, int> inout) {
    int idx2 = thrust::get<1>(inout);
    TriEdges& triEdges = thrust::get<0>(inout);
    triEdges[i] = EdgeIdx(idx2 / 2, triEdges[i].Dir());
  }
};

struct OpposedDir {
  __host__ __device__ bool operator()(int a, int b) const { return a + b == 0; }
};

struct CheckTris {
  const EdgeVertsD* edgeVerts;

  __host__ __device__ void operator()(
      thrust::tuple<bool&, const TriVerts&, const TriEdges&> inout) {
    const TriVerts& triVerts = thrust::get<1>(inout);
    const TriEdges& triEdges = thrust::get<2>(inout);
    bool good = true;
    for (int i : {0, 1, 2}) {
      int j = (i + 1) % 3;
      if (triEdges[i].Dir() > 0) {
        good &= triVerts[i] == edgeVerts[triEdges[i].Idx()].first;
        good &= triVerts[j] == edgeVerts[triEdges[i].Idx()].second;
      } else {
        good &= triVerts[i] == edgeVerts[triEdges[i].Idx()].second;
        good &= triVerts[j] == edgeVerts[triEdges[i].Idx()].first;
      }
    }
    thrust::get<0>(inout) = good;
  }
};
}  // namespace

namespace manifold {

Mesh::Mesh() { pImpl_ = nullptr; }

void Mesh::Append2Host(MeshHost& mesh) const { pImpl_->Append2Host(mesh); }

Mesh Mesh::Copy() const { return *this; }

Box Mesh::BoundingBox() const { return pImpl_->bBox_.Transform(transform_); }

void Mesh::Translate(glm::vec3 v) { transform_[3] += glm::vec4(v, 0.0f); }

void Mesh::Scale(glm::vec3 v) {
  glm::mat4 s(1.0f);
  for (int i : {0, 1, 2}) s[i][i] = v[i];
  transform_ *= s;
}

void Mesh::Rotate(glm::mat3 r) { transform_ *= glm::mat4(r); }

bool Mesh::IsValid() const { return pImpl_->IsValid(); }

int Mesh::NumOverlaps(const Mesh& B, int max_overlaps) const {
  ApplyTransform();
  B.ApplyTransform();

  VecDH<int> edgesB(max_overlaps), tris(max_overlaps);
  pImpl_->EdgeCollisions(edgesB, tris, *B.pImpl_);
  int num_overlaps = tris.size();

  edgesB.resize(max_overlaps);
  tris.resize(max_overlaps);
  B.pImpl_->EdgeCollisions(edgesB, tris, *pImpl_);
  return num_overlaps += tris.size();
}

Mesh Mesh::Boolean(const Mesh& second, OpType op, int max_overlaps) const {
  ApplyTransform();
  second.ApplyTransform();
  // Boolean3D boolean(*pImpl_, *second.pImpl_, max_overlaps);
  Mesh result;
  // result.pImpl_ = std::make_unique<Impl>(boolean.Result(op));
  return result;
}

void Mesh::ApplyTransform() const {
  if (transform_ == glm::mat4(1.0f)) return;
  glm::mat4 TS(1.0f);
  for (int i : {0, 1, 2}) {
    TS[i][i] = transform_[i][i];  // scale
    TS[3][i] = transform_[3][i];  // translate
  }
  if (transform_ == TS)
    pImpl_->TranslateScale(transform_);
  else
    pImpl_->Transform(transform_);
  transform_ = glm::mat4(1.0f);
}

Mesh::Impl::Impl(const MeshHost& mesh)
    : vertPos_(mesh.vertPos), triVerts_(mesh.triVerts) {
  CalculateBBox();
  SortVerts();
  VecDH<Box> triBox;
  VecDH<uint32_t> triMorton;
  GetTriBoxMorton(triBox, triMorton);
  SortTris(triBox, triMorton);
  CreateEdges();
  collider_ = Collider(triBox, triMorton);
}

void Mesh::Impl::Append2Host(MeshHost& pImpl_out) const {
  const int start = pImpl_out.vertPos.size();
  std::transform(
      triVerts_.begin(), triVerts_.end(),
      std::back_inserter(pImpl_out.triVerts),
      [start](const glm::ivec3& triVerts) { return triVerts + start; });
  pImpl_out.vertPos.insert(pImpl_out.vertPos.end(), vertPos_.begin(),
                           vertPos_.end());
}

void Mesh::Impl::Transform(const glm::mat4& T) {
  thrust::for_each(vertPos_.beginD(), vertPos_.endD(), Transform3({T}));
  CalculateBBox();
  VecDH<Box> triBox;
  VecDH<uint32_t> triMorton;
  GetTriBoxMorton(triBox, triMorton);
  collider_.UpdateBoxes(triBox);
}

void Mesh::Impl::TranslateScale(const glm::mat4& T) {
  glm::vec3 translate, scale;
  for (int i : {0, 1, 2}) {
    translate[i] = T[3][i];
    scale[i] = T[i][i];
  }
  glm::mat4 transform(1.0f);
  for (int i : {0, 1, 2}) {
    transform[3][i] = translate[i];
    transform[i][i] = scale[i];
  }
  thrust::for_each(vertPos_.beginD(), vertPos_.endD(), Transform3({transform}));
  bBox_ *= scale;
  bBox_ += translate;
  collider_.Scale(scale);
  collider_.Translate(translate);
}

bool Mesh::Impl::IsValid() const {
  VecDH<bool> check(NumTri());
  thrust::for_each_n(
      zip(check.beginD(), triVerts_.cbeginD(), triEdges_.cbeginD()), NumTri(),
      CheckTris({edgeVerts_.cptrD()}));
  return thrust::all_of(check.beginD(), check.endD(), thrust::identity<bool>());
  // return thrust::all_of(zip(triVerts_.beginD(), triEdges_.beginD()),
  //                       zip(triVerts_.endD(), triEdges_.endD()),
  //                       CheckTris({edgeVerts_.ptrD()}));
}

void Mesh::Impl::CalculateBBox() {
  bBox_.min = thrust::reduce(vertPos_.begin(), vertPos_.end(),
                             glm::vec3(1 / 0.0f), PosMin());
  bBox_.max = thrust::reduce(vertPos_.begin(), vertPos_.end(),
                             glm::vec3(-1 / 0.0f), PosMax());
}

void Mesh::Impl::SortVerts() {
  VecDH<uint32_t> vertMorton(NumVert());
  thrust::for_each_n(zip(vertMorton.beginD(), vertPos_.cbeginD()), NumVert(),
                     Morton({bBox_}));

  VecDH<int> vertNew2Old(NumVert());
  thrust::sequence(vertNew2Old.beginD(), vertNew2Old.endD());
  thrust::sort_by_key(vertMorton.beginD(), vertMorton.endD(),
                      zip(vertPos_.beginD(), vertNew2Old.beginD()));

  VecDH<int> vertOld2New(NumVert());
  thrust::scatter(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(NumVert()),
                  vertNew2Old.beginD(), vertOld2New.beginD());
  thrust::for_each(triVerts_.beginD(), triVerts_.endD(),
                   Reindex({vertOld2New.cptrD()}));
}

void Mesh::Impl::CreateEdges() {
  VecDH<EdgeVertsD> halfEdgeVerts(NumTri() * 3);
  VecDH<int> dir(NumTri() * 3);
  edgeVerts_.resize(halfEdgeVerts.size() / 2);
  triEdges_.resize(NumTri(), TriEdges({0, 1}));
  for (int i : {0, 1, 2}) {
    int j = (i + 1) % 3;
    int start = i * NumTri();
    thrust::for_each_n(zip(triEdges_.beginD(), dir.beginD() + start,
                           halfEdgeVerts.beginD() + start, triVerts_.cbeginD()),
                       NumTri(), MakeHalfedges({i, j}));
  }
  SortHalfedges(halfEdgeVerts, dir);
  strided_range<VecDH<EdgeVertsD>::IterD> edgeVerts(halfEdgeVerts.beginD(),
                                                    halfEdgeVerts.endD(), 2);
  thrust::copy(edgeVerts.begin(), edgeVerts.end(), edgeVerts_.beginD());
  // verify
  strided_range<VecDH<EdgeVertsD>::IterD> edgesOdd(halfEdgeVerts.beginD() + 1,
                                                   halfEdgeVerts.endD(), 2);
  ALWAYS_ASSERT(
      thrust::equal(edgeVerts.begin(), edgeVerts.end(), edgesOdd.begin()),
      runtimeErr, "Mesh is not manifold!");
  strided_range<VecDH<int>::IterD> dir1(dir.beginD(), dir.endD(), 2);
  strided_range<VecDH<int>::IterD> dir2(dir.beginD() + 1, dir.endD(), 2);
  ALWAYS_ASSERT(
      thrust::equal(dir1.begin(), dir1.end(), dir2.begin(), OpposedDir()),
      runtimeErr, "Mesh is not oriented!");
}

void Mesh::Impl::SortHalfedges(VecDH<EdgeVertsD>& halfEdgeVerts,
                               VecDH<int>& dir) {
  VecDH<int> halfedgeNew2Old(NumTri() * 3);
  thrust::sequence(halfedgeNew2Old.beginD(), halfedgeNew2Old.endD());
  thrust::sort_by_key(halfEdgeVerts.beginD(), halfEdgeVerts.endD(),
                      zip(dir.beginD(), halfedgeNew2Old.beginD()));

  VecDH<int> halfedgeOld2New(NumTri() * 3);
  thrust::scatter(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator((int)halfedgeNew2Old.size()),
                  halfedgeNew2Old.beginD(), halfedgeOld2New.beginD());
  // assign edge idx to triEdges_ (assumes edge dir is already assigned)
  for (int i : {0, 1, 2}) {
    int start = i * NumTri();
    thrust::for_each_n(
        zip(triEdges_.beginD(), halfedgeOld2New.cbeginD() + start), NumTri(),
        AssignEdges({i}));
  }
}

VecDH<Box> Mesh::Impl::GetEdgeBox() const {
  VecDH<Box> edgeBox(NumEdge());
  thrust::for_each_n(zip(edgeBox.beginD(), edgeVerts_.cbeginD()), NumEdge(),
                     EdgeBox({vertPos_.cptrD()}));
  return edgeBox;
}

void Mesh::Impl::GetTriBoxMorton(VecDH<Box>& triBox,
                                 VecDH<uint32_t>& triMorton) const {
  triBox.resize(NumTri());
  triMorton.resize(NumTri());
  thrust::for_each_n(
      zip(triMorton.beginD(), triBox.beginD(), triVerts_.cbeginD()), NumTri(),
      TriMortonBox({vertPos_.cptrD(), bBox_}));
}

void Mesh::Impl::SortTris(VecDH<Box>& triBox, VecDH<uint32_t>& triMorton) {
  thrust::sort_by_key(triMorton.beginD(), triMorton.endD(),
                      zip(triBox.beginD(), triVerts_.beginD()));
}

void Mesh::Impl::EdgeCollisions(VecDH<int>& edgesB, VecDH<int>& tris,
                                const Mesh::Impl& B) const {
  VecDH<Box> BedgeBB = B.GetEdgeBox();
  collider_.Collisions(edgesB, tris, BedgeBB);
}

void Mesh::Impl::VertexCollisionsZ(VecDH<int>& vertsOut, VecDH<int>& tris,
                                   const VecDH<glm::vec3>& vertsIn) const {
  collider_.Collisions(vertsOut, tris, vertsIn);
}
}  // namespace manifold