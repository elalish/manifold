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

#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <algorithm>

#include "connected_components.cuh"
#include "manifold_impl.cuh"
#include "polygon.h"

namespace {
using namespace manifold;

constexpr float kTolerance = 1e-5;

struct NormalizeTo {
  float length;
  __host__ __device__ void operator()(glm::vec3& v) {
    v = length * glm::normalize(v);
    if (isnan(v.x)) v = glm::vec3(0.0);
  }
};

struct SplitEdges {
  glm::vec3* vertPos;
  const int startIdx;
  const int n;

  __host__ __device__ void operator()(thrust::tuple<int, EdgeVertsD> in) {
    int edge = thrust::get<0>(in);
    EdgeVertsD edgeVerts = thrust::get<1>(in);

    float invTotal = 1.0f / n;
    for (int i = 1; i < n; ++i)
      vertPos[startIdx + (n - 1) * edge + i - 1] =
          (float(n - i) * vertPos[edgeVerts.first] +
           float(i) * vertPos[edgeVerts.second]) *
          invTotal;
  }
};

struct InteriorVerts {
  glm::vec3* vertPos;
  const int startIdx;
  const int n;

  __host__ __device__ void operator()(thrust::tuple<int, glm::ivec3> in) {
    int tri = thrust::get<0>(in);
    glm::ivec3 triVerts = thrust::get<1>(in);

    int vertsPerTri = ((n - 2) * (n - 2) + (n - 2)) / 2;
    float invTotal = 1.0f / n;
    int pos = startIdx + vertsPerTri * tri;
    for (int i = 1; i < n - 1; ++i)
      for (int j = 1; j < n - i; ++j)
        vertPos[pos++] = (float(i) * vertPos[triVerts[2]] +  //
                          float(j) * vertPos[triVerts[0]] +  //
                          float(n - i - j) * vertPos[triVerts[1]]) *
                         invTotal;
  }
};

struct SplitTris {
  glm::ivec3* triVerts;
  const int edgeIdx;
  const int triIdx;
  const int n;

  __host__ __device__ int EdgeVert(int i, EdgeIdx edge) const {
    return edgeIdx + (n - 1) * edge.Idx() +
           (edge.Dir() > 0 ? i - 1 : n - 1 - i);
  }

  __host__ __device__ int TriVert(int i, int j, int tri) const {
    --i;
    --j;
    int m = n - 2;
    int vertsPerTri = (m * m + m) / 2;
    int vertOffset = (i * (2 * m - i + 1)) / 2 + j;
    return triIdx + vertsPerTri * tri + vertOffset;
  }

  __host__ __device__ int Vert(int i, int j, int tri, glm::ivec3 triVert,
                               TriEdges triEdge) const {
    bool edge0 = i == 0;
    bool edge1 = j == 0;
    bool edge2 = j == n - i;
    if (edge0) {
      if (edge1)
        return triVert[1];
      else if (edge2)
        return triVert[0];
      else
        return EdgeVert(n - j, triEdge[0]);
    } else if (edge1) {
      if (edge2)
        return triVert[2];
      else
        return EdgeVert(i, triEdge[1]);
    } else if (edge2)
      return EdgeVert(j, triEdge[2]);
    else
      return TriVert(i, j, tri);
  }

  __host__ __device__ void operator()(
      thrust::tuple<int, glm::ivec3, TriEdges> in) {
    int tri = thrust::get<0>(in);
    glm::ivec3 triVert = thrust::get<1>(in);
    TriEdges triEdge = thrust::get<2>(in);

    int pos = n * n * tri;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n - i; ++j) {
        int a = Vert(i, j, tri, triVert, triEdge);
        int b = Vert(i + 1, j, tri, triVert, triEdge);
        int c = Vert(i, j + 1, tri, triVert, triEdge);
        triVerts[pos++] = glm::ivec3(a, b, c);
        if (j < n - 1 - i) {
          int d = Vert(i + 1, j + 1, tri, triVert, triEdge);
          triVerts[pos++] = glm::ivec3(b, d, c);
        }
      }
    }
  }
};

__host__ __device__ void AtomicAddFloat(float& target, float add) {
#ifdef __CUDA_ARCH__
  atomicAdd(&target, add);
#else
#pragma omp atomic
  target += add;
#endif
}

struct AreaVolume {
  float* surfaceArea;
  float* volume;
  const int* vertLabel;
  const glm::vec3* vertPos;

  __host__ __device__ void operator()(const glm::ivec3& triVerts) {
    glm::vec3 edge[3];
    float perimeter = 0.0f;
    for (int i : {0, 1, 2}) {
      edge[i] = vertPos[triVerts[(i + 1) % 3]] - vertPos[triVerts[i]];
      perimeter += glm::length(edge[i]);
    }
    glm::vec3 crossP = glm::cross(edge[0], edge[1]);
    float area = glm::length(crossP) / 2.0f;
    if (area > perimeter * kTolerance) {
      int comp = vertLabel[triVerts[0]];
      AtomicAddFloat(surfaceArea[comp], area);
      AtomicAddFloat(volume[comp],
                     glm::dot(crossP, vertPos[triVerts[0]]) / 6.0f);
    }
  }
};

struct ClampVolume {
  __host__ __device__ void operator()(thrust::tuple<float&, float> inOut) {
    float& volume = thrust::get<0>(inOut);
    float surfaceArea = thrust::get<1>(inOut);

    if (glm::abs(volume) < surfaceArea * kTolerance) volume = 0.0f;
  }
};

struct NonZero {
  __host__ __device__ bool operator()(float val) { return val != 0.0f; }
};

struct RemoveVert {
  const float* volume;

  __host__ __device__ bool operator()(thrust::tuple<int, int, glm::vec3> in) {
    int vertLabel = thrust::get<0>(in);
    return volume[vertLabel] == 0.0f;
  }
};

struct RemoveTri {
  __host__ __device__ bool operator()(thrust::tuple<glm::ivec3, glm::vec3> in) {
    const glm::ivec3& triVerts = thrust::get<0>(in);
    return triVerts[0] < 0;
  }
};

struct IdxMin
    : public thrust::binary_function<glm::ivec3, glm::ivec3, glm::ivec3> {
  __host__ __device__ int min3(glm::ivec3 a) {
    return glm::min(a.x, glm::min(a.y, a.z));
  }
  __host__ __device__ glm::ivec3 operator()(glm::ivec3 a, glm::ivec3 b) {
    return glm::ivec3(glm::min(min3(a), min3(b)));
  }
};

struct IdxMax
    : public thrust::binary_function<glm::ivec3, glm::ivec3, glm::ivec3> {
  __host__ __device__ int max3(glm::ivec3 a) {
    return glm::max(a.x, glm::max(a.y, a.z));
  }
  __host__ __device__ glm::ivec3 operator()(glm::ivec3 a, glm::ivec3 b) {
    return glm::ivec3(glm::max(max3(a), max3(b)));
  }
};

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

struct Transform {
  const glm::mat4x3 transform;

  __host__ __device__ void operator()(glm::vec3& position) {
    position = transform * glm::vec4(position, 1.0f);
  }
};

struct TransformNormals {
  const glm::mat3 transform;

  __host__ __device__ void operator()(glm::vec3& normal) {
    normal = glm::normalize(transform * normal);
    if (isnan(normal.x)) normal = glm::vec3(0.0f);
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

  __host__ __device__ void operator()(glm::ivec3& triVerts) {
    for (int i : {0, 1, 2}) triVerts[i] = indexInv_[triVerts[i]];
  }
};

struct TriMortonBox {
  const glm::vec3* vertPos;
  const Box bBox;

  __host__ __device__ void operator()(
      thrust::tuple<uint32_t&, Box&, const glm::ivec3&> inout) {
    uint32_t& mortonCode = thrust::get<0>(inout);
    Box& triBox = thrust::get<1>(inout);
    const glm::ivec3& triVerts = thrust::get<2>(inout);

    glm::vec3 center =
        (vertPos[triVerts[0]] + vertPos[triVerts[1]] + vertPos[triVerts[2]]) /
        3.0f;
    mortonCode = MortonCode(center, bBox);
    triBox = Box(vertPos[triVerts[0]], vertPos[triVerts[1]]);
    triBox.Union(vertPos[triVerts[2]]);
  }
};

__host__ __device__ void AtomicAddVec3(glm::vec3& target,
                                       const glm::vec3& add) {
  for (int i : {0, 1, 2}) {
#ifdef __CUDA_ARCH__
    atomicAdd(&target[i], add[i]);
#else
#pragma omp atomic
    target[i] += add[i];
#endif
  }
}

struct AssignNormals {
  glm::vec3* vertNormal;
  const glm::vec3* vertPos;
  const bool calculateTriNormal;

  __host__ __device__ void operator()(
      thrust::tuple<glm::vec3&, const glm::ivec3&, const TriEdges&> in) {
    glm::vec3& triNormal = thrust::get<0>(in);
    const glm::ivec3& triVerts = thrust::get<1>(in);
    const TriEdges& triEdges = thrust::get<2>(in);

    glm::vec3 v0 = vertPos[triVerts[0]];
    glm::vec3 v1 = vertPos[triVerts[1]];
    glm::vec3 v2 = vertPos[triVerts[2]];
    // edge vectors
    glm::vec3 e01 = glm::normalize(v1 - v0);
    glm::vec3 e12 = glm::normalize(v2 - v1);
    glm::vec3 e20 = glm::normalize(v0 - v2);

    if (calculateTriNormal) {
      triNormal = glm::normalize(glm::cross(e01, e12));
      if (isnan(triNormal.x)) triNormal = glm::vec3(0.0);
    }
    // corner angles
    glm::vec3 phi;
    phi[0] = glm::acos(-glm::dot(e01, e12));
    phi[1] = glm::acos(-glm::dot(e12, e20));
    phi[2] = glm::pi<float>() - phi[0] - phi[1];
    // assign weighted sum
    for (int i : {0, 1, 2}) {
      AtomicAddVec3(vertNormal[triVerts[i]],
                    glm::max(phi[i], kTolerance) * triNormal);
    }
  }
};

struct TmpEdge {
  int first, second, halfedgeIdx;

  __host__ __device__ TmpEdge() {}
  __host__ __device__ TmpEdge(int start, int end, int idx) {
    first = glm::min(start, end);
    second = glm::max(start, end);
    halfedgeIdx = idx;
  }

  __host__ __device__ bool operator<(const TmpEdge& other) const {
    return first == other.first ? second < other.second : first < other.first;
  }
};

struct Tri2Halfedges {
  Halfedge* halfedges;
  TmpEdge* edges;

  __host__ __device__ void operator()(
      thrust::tuple<int, const glm::ivec3&> in) {
    const int tri = thrust::get<0>(in);
    const glm::ivec3& triVerts = thrust::get<1>(in);
    for (const int i : {0, 1, 2}) {
      const int j = (i + 1) % 3;
      const int edge = 3 * tri + i;
      halfedges[edge] = {triVerts[i], triVerts[j], -1, tri};
      edges[edge] = TmpEdge(triVerts[i], triVerts[j], edge);
    }
  }
};

struct LinkHalfedges {
  Halfedge* halfedges;
  const TmpEdge* edges;

  __host__ __device__ void operator()(int k) {
    const int i = 2 * k;
    const int j = i + 1;
    const int pair0 = edges[i].halfedgeIdx;
    const int pair1 = edges[j].halfedgeIdx;
    if (halfedges[pair0].startVert != halfedges[pair1].endVert ||
        halfedges[pair0].endVert != halfedges[pair1].startVert ||
        halfedges[pair0].face == halfedges[pair1].face)
      printf("Not manifold!\n");
    halfedges[pair0].pairedHalfedge = pair1;
    halfedges[pair1].pairedHalfedge = pair0;
  }
};

struct MakeHalfedges {
  int i, j;

  __host__ __device__ void operator()(
      thrust::tuple<TriEdges&, int&, EdgeVertsD&, const glm::ivec3&> inout) {
    const glm::ivec3& in = thrust::get<3>(inout);
    int V1 = in[i];
    int V2 = in[j];
    TriEdges& triEdges = thrust::get<0>(inout);
    int& dir = thrust::get<1>(inout);
    EdgeVertsD& edgeVerts = thrust::get<2>(inout);
    if (V1 < V2) {  // forward
      dir = 1;
      edgeVerts = thrust::make_pair(V1, V2);
    } else if (V1 > V2) {  // backward
      dir = -1;
      edgeVerts = thrust::make_pair(V2, V1);
    } else {
      dir = 0;
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
  __host__ __device__ bool operator()(int a, int b) const {
    return a * b == -1;
  }
};

struct LinkEdges2Tris {
  EdgeTrisD* edgeTris;

  __host__ __device__ void operator()(thrust::tuple<int, TriEdges> in) {
    const int tri = thrust::get<0>(in);
    const TriEdges triEdges = thrust::get<1>(in);
    for (int i : {0, 1, 2}) {
      if (triEdges[i].Dir() > 0)
        edgeTris[triEdges[i].Idx()].left = tri;
      else
        edgeTris[triEdges[i].Idx()].right = tri;
    }
  }
};

struct Halfedge2Tmp {
  __host__ __device__ void operator()(
      thrust::tuple<TmpEdge&, const Halfedge&, int> inout) {
    const Halfedge& halfedge = thrust::get<1>(inout);
    int idx = thrust::get<2>(inout);
    if (halfedge.startVert > halfedge.endVert) idx = -1;

    thrust::get<0>(inout) = TmpEdge(halfedge.startVert, halfedge.endVert, idx);
  }
};

struct TmpInvalid {
  __host__ __device__ bool operator()(const TmpEdge& edge) {
    return edge.halfedgeIdx < 0;
  }
};

struct EdgeBox {
  const glm::vec3* vertPos;

  __host__ __device__ void operator()(
      thrust::tuple<Box&, const TmpEdge&> inout) {
    const TmpEdge& edge = thrust::get<1>(inout);
    thrust::get<0>(inout) = Box(vertPos[edge.first], vertPos[edge.second]);
  }
};

struct ReindexEdge {
  const TmpEdge* edges;

  __host__ __device__ void operator()(int& edge) {
    edge = edges[edge].halfedgeIdx;
  }
};

struct CheckTris {
  const EdgeVertsD* edgeVerts;

  __host__ __device__ bool operator()(thrust::tuple<glm::ivec3, TriEdges> in) {
    const glm::ivec3& triVerts = thrust::get<0>(in);
    const TriEdges& triEdges = thrust::get<1>(in);
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
    return good;
  }
};
}  // namespace

namespace manifold {

Manifold::Impl::Impl(const Mesh& manifold)
    : vertPos_(manifold.vertPos), triVerts_(manifold.triVerts) {
  CheckDevice();
  Finish();
}

Manifold::Impl::Impl(Shape shape) {
  std::vector<glm::vec3> vertPos;
  std::vector<glm::ivec3> triVerts;
  switch (shape) {
    case Shape::TETRAHEDRON:
      vertPos = {{-1.0f, -1.0f, 1.0f},
                 {-1.0f, 1.0f, -1.0f},
                 {1.0f, -1.0f, -1.0f},
                 {1.0f, 1.0f, 1.0f}};
      triVerts = {{2, 0, 1}, {0, 3, 1}, {2, 3, 0}, {3, 2, 1}};
      break;
    case Shape::CUBE:
      vertPos = {{0.0f, 0.0f, 0.0f},  //
                 {1.0f, 0.0f, 0.0f},  //
                 {1.0f, 1.0f, 0.0f},  //
                 {0.0f, 1.0f, 0.0f},  //
                 {0.0f, 0.0f, 1.0f},  //
                 {1.0f, 0.0f, 1.0f},  //
                 {1.0f, 1.0f, 1.0f},  //
                 {0.0f, 1.0f, 1.0f}};
      triVerts = {{0, 2, 1}, {0, 3, 2},  //
                  {4, 5, 6}, {4, 6, 7},  //
                  {0, 1, 5}, {0, 5, 4},  //
                  {1, 2, 6}, {1, 6, 5},  //
                  {2, 3, 7}, {2, 7, 6},  //
                  {3, 0, 4}, {3, 4, 7}};
      break;
    case Shape::OCTAHEDRON:
      vertPos = {{1.0f, 0.0f, 0.0f},   //
                 {-1.0f, 0.0f, 0.0f},  //
                 {0.0f, 1.0f, 0.0f},   //
                 {0.0f, -1.0f, 0.0f},  //
                 {0.0f, 0.0f, 1.0f},   //
                 {0.0f, 0.0f, -1.0f}};
      triVerts = {{0, 2, 4}, {1, 5, 3},  //
                  {2, 1, 4}, {3, 5, 0},  //
                  {1, 3, 4}, {0, 5, 2},  //
                  {3, 0, 4}, {2, 5, 1}};
      break;
    default:
      throw logicErr("Unrecognized shape!");
  }
  vertPos_ = vertPos;
  triVerts_ = triVerts;
  Finish();
}

void Manifold::Impl::RemoveChaff() {
  CreateEdges();
  int n_comp = ConnectedComponents(vertLabel_, NumVert(), halfedge_);

  VecDH<float> surfaceArea(n_comp), volume(n_comp);
  thrust::for_each_n(triVerts_.beginD(), NumTri(),
                     AreaVolume({surfaceArea.ptrD(), volume.ptrD(),
                                 vertLabel_.cptrD(), vertPos_.cptrD()}));
  thrust::for_each_n(zip(volume.beginD(), surfaceArea.beginD()), n_comp,
                     ClampVolume());
  numLabel_ = thrust::count_if(volume.beginD(), volume.endD(), NonZero());

  VecDH<int> newVert2Old(NumVert());
  thrust::sequence(newVert2Old.begin(), newVert2Old.end());
  auto begin =
      zip(vertLabel_.beginD(), newVert2Old.beginD(), vertPos_.beginD());
  int newNumVert =
      thrust::remove_if(
          begin, zip(vertLabel_.endD(), newVert2Old.endD(), vertPos_.endD()),
          RemoveVert({volume.cptrD()})) -
      begin;

  VecDH<int> oldVert2New(NumVert(), -1);
  vertPos_.resize(newNumVert);
  vertLabel_.resize(newNumVert);
  thrust::scatter(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(newNumVert),
                  newVert2Old.beginD(), oldVert2New.beginD());

  thrust::for_each(triVerts_.beginD(), triVerts_.endD(),
                   Reindex({oldVert2New.cptrD()}));

  auto start = zip(triVerts_.beginD(), triNormal_.beginD());
  int newNumTri =
      thrust::remove_if(start, zip(triVerts_.endD(), triNormal_.endD()),
                        RemoveTri()) -
      start;
  triVerts_.resize(newNumTri);
  triNormal_.resize(newNumTri);
}

void Manifold::Impl::Finish() {
  if (triVerts_.size() == 0) return;
  ALWAYS_ASSERT(thrust::reduce(triVerts_.beginD(), triVerts_.endD(),
                               glm::ivec3(std::numeric_limits<int>::max()),
                               IdxMin())[0] >= 0,
                runtimeErr, "Negative vertex index!");
  ALWAYS_ASSERT(thrust::reduce(triVerts_.beginD(), triVerts_.endD(),
                               glm::ivec3(-1), IdxMax())[0] < NumVert(),
                runtimeErr, "Vertex index exceeds number of verts!");
  if (vertLabel_.size() != NumVert()) {
    vertLabel_.resize(NumVert());
    numLabel_ = 1;
    thrust::fill(vertLabel_.beginD(), vertLabel_.endD(), 0);
  }
  CalculateBBox();
  SortVerts();
  VecDH<Box> triBox;
  VecDH<uint32_t> triMorton;
  GetTriBoxMorton(triBox, triMorton);
  SortTris(triBox, triMorton);
  CreateEdges();
  CreateHalfedges(triVerts_);
  CalculateNormals();
  collider_ = Collider(triBox, triMorton);
}

void Manifold::Impl::CreateHalfedges(const VecDH<glm::ivec3>& triVerts) {
  const int numTri = triVerts.size();
  halfedge_.resize(3 * numTri);
  VecDH<TmpEdge> edge(3 * numTri);
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), triVerts.beginD()),
                     numTri, Tri2Halfedges({halfedge_.ptrD(), edge.ptrD()}));
  thrust::sort(edge.beginD(), edge.endD());
  thrust::for_each_n(thrust::make_counting_iterator(0), halfedge_.size() / 2,
                     LinkHalfedges({halfedge_.ptrD(), edge.cptrD()}));
}

void Manifold::Impl::Update() {
  CalculateBBox();
  VecDH<Box> triBox;
  VecDH<uint32_t> triMorton;
  GetTriBoxMorton(triBox, triMorton);
  collider_.UpdateBoxes(triBox);
}

void Manifold::Impl::ApplyTransform() const {
  // This const_cast is here because these operations cancel out, leaving the
  // state conceptually unchanged. This enables lazy transformation evaluation.
  const_cast<Impl*>(this)->ApplyTransform();
}

void Manifold::Impl::ApplyTransform() {
  if (transform_ == glm::mat4x3(1.0f)) return;
  thrust::for_each(vertPos_.beginD(), vertPos_.endD(), Transform({transform_}));

  glm::mat3 normalTransform =
      glm::inverse(glm::transpose(glm::mat3(transform_)));
  thrust::for_each(triNormal_.beginD(), triNormal_.endD(),
                   TransformNormals({normalTransform}));
  thrust::for_each(vertNormal_.beginD(), vertNormal_.endD(),
                   TransformNormals({normalTransform}));
  // This optimization does a cheap collider update if the transform is
  // axis-aligned.
  if (!collider_.Transform(transform_)) Update();
  transform_ = glm::mat4x3(1.0f);
  CalculateBBox();
}

bool Manifold::Impl::Tri2Face() const {
  // This const_cast is here because this operation tweaks the internal data
  // structure, but does not change what it represents.
  return const_cast<Impl*>(this)->Tri2Face();
}

bool Manifold::Impl::Tri2Face() {
  if (face_.size() != 0 || halfedge_.size() % 3 != 0) return false;
  face_.resize(halfedge_.size() / 3 + 1);
  thrust::sequence(face_.beginD(), face_.endD(), 0, 3);
  return true;
}

void Manifold::Impl::Refine(int n) {
  // This function doesn't run Finish(), as that is expensive and it'll need to
  // be run after the new vertices have moved, which is a likely scenario after
  // refinement (smoothing).
  int numVert = NumVert();
  int numEdge = NumEdge();
  int numTri = NumTri();
  // Append new verts
  int vertsPerEdge = n - 1;
  int vertsPerTri = ((n - 2) * (n - 2) + (n - 2)) / 2;
  int triVertStart = numVert + numEdge * vertsPerEdge;
  vertPos_.resize(triVertStart + numTri * vertsPerTri);
  thrust::for_each_n(
      zip(thrust::make_counting_iterator(0), edgeVerts_.beginD()), numEdge,
      SplitEdges({vertPos_.ptrD(), numVert, n}));
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), triVerts_.beginD()),
                     numTri, InteriorVerts({vertPos_.ptrD(), triVertStart, n}));
  // Create subtriangles
  VecDH<glm::ivec3> inTri(triVerts_);
  triVerts_.resize(n * n * numTri);
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), inTri.beginD(),
                         triEdges_.beginD()),
                     numTri,
                     SplitTris({triVerts_.ptrD(), numVert, triVertStart, n}));
}

bool Manifold::Impl::IsValid() const {
  return thrust::all_of(zip(triVerts_.beginD(), triEdges_.beginD()),
                        zip(triVerts_.endD(), triEdges_.endD()),
                        CheckTris({edgeVerts_.ptrD()}));
}

void Manifold::Impl::CalculateBBox() {
  bBox_.min = thrust::reduce(vertPos_.begin(), vertPos_.end(),
                             glm::vec3(1 / 0.0f), PosMin());
  bBox_.max = thrust::reduce(vertPos_.begin(), vertPos_.end(),
                             glm::vec3(-1 / 0.0f), PosMax());
  ALWAYS_ASSERT(bBox_.isFinite(), runtimeErr,
                "Input vertices are not all finite!");
}

void Manifold::Impl::SortVerts() {
  VecDH<uint32_t> vertMorton(NumVert());
  thrust::for_each_n(zip(vertMorton.beginD(), vertPos_.cbeginD()), NumVert(),
                     Morton({bBox_}));

  VecDH<int> vertNew2Old(NumVert());
  thrust::sequence(vertNew2Old.beginD(), vertNew2Old.endD());
  thrust::sort_by_key(
      vertMorton.beginD(), vertMorton.endD(),
      zip(vertPos_.beginD(), vertLabel_.beginD(), vertNew2Old.beginD()));

  VecDH<int> vertOld2New(NumVert());
  thrust::scatter(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(NumVert()),
                  vertNew2Old.beginD(), vertOld2New.beginD());
  thrust::for_each(triVerts_.beginD(), triVerts_.endD(),
                   Reindex({vertOld2New.cptrD()}));
}

void Manifold::Impl::CreateEdges() {
  VecDH<EdgeVertsD> halfEdgeVerts(NumTri() * 3);
  VecDH<int> dir(NumTri() * 3);
  edgeVerts_.resize(halfEdgeVerts.size() / 2);
  triEdges_.resize(NumTri());
  edgeTris_.resize(NumEdge());
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

  thrust::for_each_n(zip(thrust::make_counting_iterator(0), triEdges_.beginD()),
                     NumTri(), LinkEdges2Tris({edgeTris_.ptrD()}));
  // verify
  strided_range<VecDH<EdgeVertsD>::IterD> edgesOdd(halfEdgeVerts.beginD() + 1,
                                                   halfEdgeVerts.endD(), 2);
  ALWAYS_ASSERT(
      thrust::equal(edgeVerts.begin(), edgeVerts.end(), edgesOdd.begin()),
      runtimeErr, "Manifold is not manifold!");
  strided_range<VecDH<int>::IterD> dir1(dir.beginD(), dir.endD(), 2);
  strided_range<VecDH<int>::IterD> dir2(dir.beginD() + 1, dir.endD(), 2);
  ALWAYS_ASSERT(
      thrust::equal(dir1.begin(), dir1.end(), dir2.begin(), OpposedDir()),
      runtimeErr, "Manifold is not oriented!");
}

void Manifold::Impl::SortHalfedges(VecDH<EdgeVertsD>& halfEdgeVerts,
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

void Manifold::Impl::GetTriBoxMorton(VecDH<Box>& triBox,
                                     VecDH<uint32_t>& triMorton) const {
  triBox.resize(NumTri());
  triMorton.resize(NumTri());
  thrust::for_each_n(
      zip(triMorton.beginD(), triBox.beginD(), triVerts_.cbeginD()), NumTri(),
      TriMortonBox({vertPos_.cptrD(), bBox_}));
}

void Manifold::Impl::SortTris(VecDH<Box>& triBox, VecDH<uint32_t>& triMorton) {
  if (triNormal_.size() == NumTri()) {
    thrust::sort_by_key(
        triMorton.beginD(), triMorton.endD(),
        zip(triBox.beginD(), triVerts_.beginD(), triNormal_.beginD()));
  } else {
    thrust::sort_by_key(triMorton.beginD(), triMorton.endD(),
                        zip(triBox.beginD(), triVerts_.beginD()));
  }
}

void Manifold::Impl::CalculateNormals() {
  vertNormal_.resize(NumVert());
  bool calculateTriNormal = false;
  if (triNormal_.size() != NumTri()) {
    triNormal_.resize(NumTri());
    calculateTriNormal = true;
  }
  thrust::for_each_n(
      zip(triNormal_.beginD(), triVerts_.beginD(), triEdges_.beginD()),
      NumTri(), AssignNormals({vertNormal_.ptrD(), vertPos_.cptrD(),
                               calculateTriNormal}));
  thrust::for_each(vertNormal_.begin(), vertNormal_.end(), NormalizeTo({1.0}));
}

SparseIndices Manifold::Impl::EdgeCollisions(const Impl& Q) const {
  VecDH<TmpEdge> edges(Q.halfedge_.size());
  thrust::for_each_n(zip(edges.beginD(), Q.halfedge_.beginD(),
                         thrust::make_counting_iterator(0)),
                     edges.size(), Halfedge2Tmp());
  int numEdge = thrust::remove_if(edges.beginD(), edges.endD(), TmpInvalid()) -
                edges.beginD();
  ALWAYS_ASSERT(numEdge == Q.NumEdge(), runtimeErr, "Not oriented!");
  edges.resize(numEdge);

  VecDH<Box> QedgeBB(numEdge);
  thrust::for_each_n(zip(QedgeBB.beginD(), edges.cbeginD()), numEdge,
                     EdgeBox({Q.vertPos_.cptrD()}));

  SparseIndices p2q1 = collider_.Collisions(QedgeBB);

  thrust::for_each(p2q1.beginD(1), p2q1.endD(1), ReindexEdge({edges.cptrD()}));
  return p2q1;
}

SparseIndices Manifold::Impl::VertexCollisionsZ(
    const VecDH<glm::vec3>& vertsIn) const {
  return collider_.Collisions(vertsIn);
}
}  // namespace manifold