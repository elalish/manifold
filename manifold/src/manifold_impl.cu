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

#include <thrust/adjacent_difference.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <map>

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

struct Halfedge2Tmp {
  __host__ __device__ void operator()(
      thrust::tuple<TmpEdge&, const Halfedge&, int> inout) {
    const Halfedge& halfedge = thrust::get<1>(inout);
    int idx = thrust::get<2>(inout);
    if (!halfedge.IsForward()) idx = -1;

    thrust::get<0>(inout) = TmpEdge(halfedge.startVert, halfedge.endVert, idx);
  }
};

struct TmpInvalid {
  __host__ __device__ bool operator()(const TmpEdge& edge) {
    return edge.halfedgeIdx < 0;
  }
};

VecDH<TmpEdge> CreateTmpEdges(const VecDH<Halfedge>& halfedge) {
  VecDH<TmpEdge> edges(halfedge.size());
  thrust::for_each_n(
      zip(edges.beginD(), halfedge.beginD(), thrust::make_counting_iterator(0)),
      edges.size(), Halfedge2Tmp());
  int numEdge = thrust::remove_if(edges.beginD(), edges.endD(), TmpInvalid()) -
                edges.beginD();
  ALWAYS_ASSERT(numEdge == halfedge.size() / 2, runtimeErr, "Not oriented!");
  edges.resize(numEdge);
  return edges;
}

struct ReindexEdge {
  const TmpEdge* edges;

  __host__ __device__ void operator()(int& edge) {
    edge = edges[edge].halfedgeIdx;
  }
};

struct ReindexHalfedge {
  int* half2Edge;

  __host__ __device__ void operator()(thrust::tuple<int, TmpEdge> in) {
    const int edge = thrust::get<0>(in);
    const int halfedge = thrust::get<1>(in).halfedgeIdx;

    half2Edge[halfedge] = edge;
  }
};

struct SplitEdges {
  glm::vec3* vertPos;
  const int startIdx;
  const int n;

  __host__ __device__ void operator()(thrust::tuple<int, TmpEdge> in) {
    int edge = thrust::get<0>(in);
    TmpEdge edgeVerts = thrust::get<1>(in);

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
  const Halfedge* halfedge;

  __host__ __device__ void operator()(int tri) {
    int vertsPerTri = ((n - 2) * (n - 2) + (n - 2)) / 2;
    float invTotal = 1.0f / n;
    int pos = startIdx + vertsPerTri * tri;
    for (int i = 1; i < n - 1; ++i)
      for (int j = 1; j < n - i; ++j)
        vertPos[pos++] =
            (float(i) * vertPos[halfedge[3 * tri + 2].startVert] +  //
             float(j) * vertPos[halfedge[3 * tri].startVert] +      //
             float(n - i - j) * vertPos[halfedge[3 * tri + 1].startVert]) *
            invTotal;
  }
};

struct SplitTris {
  glm::ivec3* triVerts;
  const Halfedge* halfedge;
  const int* half2Edge;
  const int edgeIdx;
  const int triIdx;
  const int n;

  __host__ __device__ int EdgeVert(int i, int inHalfedge) const {
    bool forward = halfedge[inHalfedge].IsForward();
    int edge = forward ? half2Edge[inHalfedge]
                       : half2Edge[halfedge[inHalfedge].pairedHalfedge];
    return edgeIdx + (n - 1) * edge + (forward ? i - 1 : n - 1 - i);
  }

  __host__ __device__ int TriVert(int i, int j, int tri) const {
    --i;
    --j;
    int m = n - 2;
    int vertsPerTri = (m * m + m) / 2;
    int vertOffset = (i * (2 * m - i + 1)) / 2 + j;
    return triIdx + vertsPerTri * tri + vertOffset;
  }

  __host__ __device__ int Vert(int i, int j, int tri) const {
    bool edge0 = i == 0;
    bool edge1 = j == 0;
    bool edge2 = j == n - i;
    if (edge0) {
      if (edge1)
        return halfedge[3 * tri + 1].startVert;
      else if (edge2)
        return halfedge[3 * tri].startVert;
      else
        return EdgeVert(n - j, 3 * tri);
    } else if (edge1) {
      if (edge2)
        return halfedge[3 * tri + 2].startVert;
      else
        return EdgeVert(i, 3 * tri + 1);
    } else if (edge2)
      return EdgeVert(j, 3 * tri + 2);
    else
      return TriVert(i, j, tri);
  }

  __host__ __device__ void operator()(int tri) {
    int pos = n * n * tri;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n - i; ++j) {
        int a = Vert(i, j, tri);
        int b = Vert(i + 1, j, tri);
        int c = Vert(i, j + 1, tri);
        triVerts[pos++] = glm::ivec3(a, b, c);
        if (j < n - 1 - i) {
          int d = Vert(i + 1, j + 1, tri);
          triVerts[pos++] = glm::ivec3(b, d, c);
        }
      }
    }
  }
};

struct FaceAreaVolume {
  const Halfedge* halfedges;
  const int* faceEdge;
  const glm::vec3* vertPos;

  __host__ __device__ thrust::pair<float, float> operator()(int face) {
    float perimeter = 0.0f;
    float area = 0.0f;
    float volume = 0.0f;

    int edge = faceEdge[face];
    const glm::vec3 anchor = vertPos[halfedges[edge].startVert];

    const int end = faceEdge[face + 1];
    while (edge < end) {
      const Halfedge halfedge = halfedges[edge++];
      const glm::vec3 start = vertPos[halfedge.startVert];
      const glm::vec3 edgeVec = vertPos[halfedge.endVert] - start;
      perimeter += glm::length(edgeVec);
      const glm::vec3 crossP = glm::cross(start - anchor, edgeVec);
      area += glm::length(crossP);
      volume += glm::dot(crossP, anchor);
    }

    return area > perimeter * kTolerance
               ? thrust::make_pair(area / 2.0f, volume / 6.0f)
               : thrust::make_pair(0.0f, 0.0f);
  }
};

struct Extrema : public thrust::binary_function<Halfedge, Halfedge, Halfedge> {
  __host__ __device__ void MakeForward(Halfedge& a) {
    if (!a.IsForward()) {
      int tmp = a.startVert;
      a.startVert = a.endVert;
      a.endVert = tmp;
    }
  }

  __host__ __device__ int MaxOrMinus(int a, int b) {
    return glm::min(a, b) < 0 ? -1 : glm::max(a, b);
  }

  __host__ __device__ Halfedge operator()(Halfedge a, Halfedge b) {
    MakeForward(a);
    MakeForward(b);
    a.startVert = glm::min(a.startVert, b.startVert);
    a.endVert = glm::max(a.endVert, b.endVert);
    a.face = MaxOrMinus(a.face, b.face);
    a.pairedHalfedge = MaxOrMinus(a.pairedHalfedge, b.pairedHalfedge);
    return a;
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

struct SumPair : public thrust::binary_function<thrust::pair<float, float>,
                                                thrust::pair<float, float>,
                                                thrust::pair<float, float>> {
  __host__ __device__ thrust::pair<float, float> operator()(
      thrust::pair<float, float> a, thrust::pair<float, float> b) {
    a.first += b.first;
    a.second += b.second;
    return a;
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

struct FaceMortonBox {
  const int* faceEdge;
  const Halfedge* halfedge;
  const glm::vec3* vertPos;
  const Box bBox;

  __host__ __device__ void operator()(
      thrust::tuple<uint32_t&, Box&, int> inout) {
    uint32_t& mortonCode = thrust::get<0>(inout);
    Box& faceBox = thrust::get<1>(inout);
    int face = thrust::get<2>(inout);

    glm::vec3 center;

    int iEdge = faceEdge[face];
    const int end = faceEdge[face + 1];
    const int nEdge = end - iEdge;
    while (iEdge < end) {
      const glm::vec3 pos = vertPos[halfedge[iEdge].startVert];
      center += pos;
      faceBox.Union(pos);
      ++iEdge;
    }
    center /= nEdge;

    mortonCode = MortonCode(center, bBox);
  }
};

struct Reindex {
  const int* indexInv;

  __host__ __device__ void operator()(Halfedge& edge) {
    edge.startVert = indexInv[edge.startVert];
    edge.endVert = indexInv[edge.endVert];
  }
};

struct ReindexFace {
  Halfedge* halfedge;
  const int* faceEdge;
  const Halfedge* oldHalfedge;
  const int* oldFaceEdge;
  const int* faceNew2Old;
  const int* faceOld2New;

  __host__ __device__ void operator()(thrust::tuple<int, int> in) {
    const int newFace = thrust::get<0>(in);
    int outEdge = thrust::get<1>(in);

    const int oldFace = faceNew2Old[newFace];
    int iEdge = oldFaceEdge[oldFace];
    const int end = oldFaceEdge[oldFace + 1];
    while (iEdge < end) {
      Halfedge edge = oldHalfedge[iEdge++];
      edge.face = newFace;
      const int pairedFace = oldHalfedge[edge.pairedHalfedge].face;
      const int offset = edge.pairedHalfedge - oldFaceEdge[pairedFace];
      edge.pairedHalfedge = faceEdge[faceOld2New[pairedFace]] + offset;
      halfedge[outEdge++] = edge;
    }
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
  const Halfedge* halfedges;
  const int* nextHalfedge;
  const int* faceEdge;
  const bool calculateTriNormal;

  __host__ __device__ void operator()(thrust::tuple<glm::vec3&, int> in) {
    glm::vec3& triNormal = thrust::get<0>(in);
    const int face = thrust::get<1>(in);

    if (calculateTriNormal) {
      triNormal = glm::vec3(0.0f);
      int iEdge = faceEdge[face];
      const int end = faceEdge[face + 1];
      Halfedge edge = halfedges[iEdge];
      glm::vec3 edgeVec = vertPos[edge.endVert] - vertPos[edge.startVert];
      while (iEdge < end) {
        Halfedge nextEdge = halfedges[nextHalfedge[iEdge]];
        glm::vec3 nextEdgeVec =
            vertPos[nextEdge.endVert] - vertPos[nextEdge.startVert];
        triNormal += glm::cross(edgeVec, nextEdgeVec);
        edge = nextEdge;
        edgeVec = nextEdgeVec;
        ++iEdge;
      }
      triNormal = glm::normalize(triNormal);
      if (isnan(triNormal.x)) triNormal = glm::vec3(0.0);
    }

    int iEdge = faceEdge[face];
    const int end = faceEdge[face + 1];
    Halfedge edge = halfedges[iEdge];
    glm::vec3 edgeVec =
        glm::normalize(vertPos[edge.endVert] - vertPos[edge.startVert]);
    while (iEdge < end) {
      Halfedge nextEdge = halfedges[nextHalfedge[iEdge]];
      glm::vec3 nextEdgeVec = glm::normalize(vertPos[nextEdge.endVert] -
                                             vertPos[nextEdge.startVert]);
      // corner angle
      float phi = glm::acos(-glm::dot(edgeVec, nextEdgeVec));
      AtomicAddVec3(vertNormal[edge.endVert],
                    glm::max(phi, kTolerance) * triNormal);
      edge = nextEdge;
      edgeVec = nextEdgeVec;
      ++iEdge;
    }
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

struct EdgeBox {
  const glm::vec3* vertPos;

  __host__ __device__ void operator()(
      thrust::tuple<Box&, const TmpEdge&> inout) {
    const TmpEdge& edge = thrust::get<1>(inout);
    thrust::get<0>(inout) = Box(vertPos[edge.first], vertPos[edge.second]);
  }
};

struct CheckManifold {
  const Halfedge* halfedges;
  const int* faces;

  __host__ __device__ bool operator()(int face) {
    bool good = true;
    int edge = faces[face];
    const int end = faces[face + 1];
    while (edge < end) {
      const Halfedge halfedge = halfedges[edge];
      const Halfedge paired = halfedges[halfedge.pairedHalfedge];
      good &= halfedge.face == face;
      good &= paired.pairedHalfedge == edge;
      good &= halfedge.startVert == paired.endVert;
      good &= halfedge.endVert == paired.startVert;
      ++edge;
    }
    // TODO: also test for duplicate halfedge pairs.
    return good;
  }
};

glm::mat3x2 GetAxisAlignedProjection(glm::vec3 normal) {
  glm::vec3 absNormal = glm::abs(normal);
  float xyzMax;
  glm::mat2x3 projection;
  if (absNormal.z > absNormal.x && absNormal.z > absNormal.y) {
    projection = glm::mat2x3(1.0f, 0.0f, 0.0f,  //
                             0.0f, 1.0f, 0.0f);
    xyzMax = normal.z;
  } else if (absNormal.y > absNormal.x) {
    projection = glm::mat2x3(0.0f, 0.0f, 1.0f,  //
                             1.0f, 0.0f, 0.0f);
    xyzMax = normal.y;
  } else {
    projection = glm::mat2x3(0.0f, 1.0f, 0.0f,  //
                             0.0f, 0.0f, 1.0f);
    xyzMax = normal.x;
  }
  if (xyzMax < 0) projection[0] *= -1.0f;
  return glm::transpose(projection);
}
}  // namespace

namespace manifold {

Manifold::Impl::Impl(const Mesh& manifold) : vertPos_(manifold.vertPos) {
  CheckDevice();
  CreateHalfedges(manifold.triVerts);
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
  CreateHalfedges(triVerts);
  Finish();
}

void Manifold::Impl::CreateHalfedges(const VecDH<glm::ivec3>& triVerts) {
  const int numTri = triVerts.size();
  faceEdge_.resize(0);
  halfedge_.resize(3 * numTri);
  VecDH<TmpEdge> edge(3 * numTri);
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), triVerts.beginD()),
                     numTri, Tri2Halfedges({halfedge_.ptrD(), edge.ptrD()}));
  thrust::sort(edge.beginD(), edge.endD());
  thrust::for_each_n(thrust::make_counting_iterator(0), halfedge_.size() / 2,
                     LinkHalfedges({halfedge_.ptrD(), edge.cptrD()}));
  Tri2Face();
}

void Manifold::Impl::LabelVerts() {
  numLabel_ = ConnectedComponents(vertLabel_, NumVert(), halfedge_);
}

void Manifold::Impl::Finish() {
  if (halfedge_.size() == 0) return;
  Halfedge extrema = {0, 0, 0, 0};
  extrema =
      thrust::reduce(halfedge_.beginD(), halfedge_.endD(), extrema, Extrema());

  ALWAYS_ASSERT(extrema.startVert >= 0, runtimeErr,
                "Vertex index is negative!");
  ALWAYS_ASSERT(extrema.endVert < NumVert(), runtimeErr,
                "Vertex index exceeds number of verts!");
  ALWAYS_ASSERT(extrema.face >= 0, runtimeErr, "Face index is negative!");
  ALWAYS_ASSERT(extrema.face < NumFace(), runtimeErr,
                "Face index exceeds number of faces!");
  ALWAYS_ASSERT(extrema.pairedHalfedge >= 0, runtimeErr,
                "Halfedge index is negative!");
  ALWAYS_ASSERT(extrema.pairedHalfedge < 2 * NumEdge(), runtimeErr,
                "Halfedge index exceeds number of halfedges!");
  ALWAYS_ASSERT(faceEdge_.H().front() == 0, runtimeErr,
                "Faces do not start at zero!");
  ALWAYS_ASSERT(faceEdge_.H().back() == 2 * NumEdge(), runtimeErr,
                "Faces do not end at halfedge length!");

  if (vertLabel_.size() != NumVert()) {
    vertLabel_.resize(NumVert());
    numLabel_ = 1;
    thrust::fill(vertLabel_.beginD(), vertLabel_.endD(), 0);
  }
  CalculateBBox();
  SortVerts();
  VecDH<Box> faceBox;
  VecDH<uint32_t> faceMorton;
  GetFaceBoxMorton(faceBox, faceMorton);
  AssembleFaces();
  CalculateNormals();
  SortFaces(faceBox, faceMorton);
  collider_ = Collider(faceBox, faceMorton);
}

void Manifold::Impl::Update() {
  CalculateBBox();
  VecDH<Box> faceBox;
  VecDH<uint32_t> faceMorton;
  GetFaceBoxMorton(faceBox, faceMorton);
  collider_.UpdateBoxes(faceBox);
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
  thrust::for_each(faceNormal_.beginD(), faceNormal_.endD(),
                   TransformNormals({normalTransform}));
  thrust::for_each(vertNormal_.beginD(), vertNormal_.endD(),
                   TransformNormals({normalTransform}));
  // This optimization does a cheap collider update if the transform is
  // axis-aligned.
  if (!collider_.Transform(transform_)) Update();
  transform_ = glm::mat4x3(1.0f);
  CalculateBBox();
}

void Manifold::Impl::AssembleFaces() const {
  // This const_cast is here because this operation tweaks the internal data
  // structure, but does not change what it represents.
  return const_cast<Impl*>(this)->AssembleFaces();
}

void Manifold::Impl::AssembleFaces() {
  nextHalfedge_.resize(halfedge_.size());
  const VecH<int>& faceEdge = faceEdge_.H();
  for (int i = 0; i < NumFace(); ++i) {
    int start = faceEdge[i];
    Manifold::Impl::NextEdges(&nextHalfedge_.H()[0] + start,
                              &halfedge_.H()[0] + start,
                              &halfedge_.H()[0] + faceEdge[i + 1]);
  }
}

bool Manifold::Impl::Tri2Face() const {
  // This const_cast is here because this operation tweaks the internal data
  // structure, but does not change what it represents.
  return const_cast<Impl*>(this)->Tri2Face();
}

bool Manifold::Impl::Tri2Face() {
  if (faceEdge_.size() != 0 || halfedge_.size() % 3 != 0) return false;
  faceEdge_.resize(halfedge_.size() / 3 + 1);
  thrust::sequence(faceEdge_.beginD(), faceEdge_.endD(), 0, 3);
  return true;
}

bool Manifold::Impl::Face2Tri() {
  if (faceEdge_.size() == 0 && halfedge_.size() % 3 == 0) return false;
  VecDH<glm::ivec3> triVertsOut;
  VecDH<glm::vec3> triNormalOut;

  VecH<glm::ivec3>& triVerts = triVertsOut.H();
  VecH<glm::vec3>& triNormal = triNormalOut.H();
  VecH<glm::vec3>& vertPos = vertPos_.H();
  const VecH<int>& face = faceEdge_.H();
  const VecH<Halfedge>& halfedge = halfedge_.H();
  const VecH<glm::vec3>& faceNormal = faceNormal_.H();

  for (int i = 0; i < face.size() - 1; ++i) {
    const int edge = face[i];
    const int lastEdge = face[i + 1];
    const int numEdge = lastEdge - edge;
    ALWAYS_ASSERT(numEdge >= 3, logicErr, "face has less than three edges.");
    const glm::vec3 normal = faceNormal[i];

    if (numEdge == 3) {  // Special case to increase performance

      glm::ivec3 tri(halfedge[edge].startVert, halfedge[edge + 1].startVert,
                     halfedge[edge + 2].startVert);
      glm::ivec3 ends(halfedge[edge].endVert, halfedge[edge + 1].endVert,
                      halfedge[edge + 2].endVert);
      if (ends[0] == tri[2]) {
        std::swap(tri[1], tri[2]);
        std::swap(ends[1], ends[2]);
      }
      ALWAYS_ASSERT(ends[0] == tri[1] && ends[1] == tri[2] && ends[2] == tri[0],
                    runtimeErr, "These 3 edges do not form a triangle!");

      triVerts.push_back(tri);
      triNormal.push_back(normal);
    } else {  // General triangulation
      const glm::mat3x2 projection = GetAxisAlignedProjection(normal);
      Polygons polys =
          Manifold::Impl::Face2Polygons(i, [&vertPos, &projection](int vert) {
            return projection * vertPos[vert];
          });
      std::vector<glm::ivec3> newTris;
      try {
        newTris = Triangulate(polys);
      } catch (const runtimeErr& e) {
        if (PolygonParams().checkGeometry) throw;
        /**
        To ensure the triangulation maintains the mesh as 2-manifold, we
        require it to not create edges connecting non-neighboring vertices
        from the same input edge. This is because if two neighboring
        polygons were to create an edge like this between two of their
        shared vertices, this would create a 4-manifold edge, which is not
        allowed.

        For some self-overlapping polygons, there exists no triangulation
        that adheres to this constraint. In this case, we create an extra
        vertex for each polygon and triangulate them like a wagon wheel,
        which is guaranteed to be manifold. This is very rare and only
        occurs when the input manifolds are self-overlapping.
         */
        for (const auto& poly : polys) {
          glm::vec3 centroid = thrust::transform_reduce(
              poly.begin(), poly.end(),
              [&vertPos](PolyVert v) { return vertPos[v.idx]; },
              glm::vec3(0.0f), [](glm::vec3 a, glm::vec3 b) { return a + b; });
          centroid /= poly.size();
          int newVert = vertPos.size();
          vertPos.push_back(centroid);
          newTris.push_back({poly.back().idx, poly.front().idx, newVert});
          for (int j = 1; j < poly.size(); ++j)
            newTris.push_back({poly[j - 1].idx, poly[j].idx, newVert});
        }
      }
      for (auto tri : newTris) {
        triVerts.push_back(tri);
        triNormal.push_back(normal);
      }
    }
  }
  faceNormal_ = triNormalOut;
  CreateHalfedges(triVertsOut);
  return true;
}

void Manifold::Impl::Refine(int n) {
  // This function doesn't run Finish(), as that is expensive and it'll need to
  // be run after the new vertices have moved, which is a likely scenario after
  // refinement (smoothing).
  Face2Tri();
  int numVert = NumVert();
  int numEdge = NumEdge();
  int numTri = NumFace();
  // Append new verts
  int vertsPerEdge = n - 1;
  int vertsPerTri = ((n - 2) * (n - 2) + (n - 2)) / 2;
  int triVertStart = numVert + numEdge * vertsPerEdge;
  vertPos_.resize(triVertStart + numTri * vertsPerTri);
  VecDH<TmpEdge> edges = CreateTmpEdges(halfedge_);
  VecDH<int> half2Edge(2 * numEdge);
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), edges.beginD()),
                     numEdge, ReindexHalfedge({half2Edge.ptrD()}));
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), edges.beginD()),
                     numEdge, SplitEdges({vertPos_.ptrD(), numVert, n}));
  thrust::for_each_n(
      thrust::make_counting_iterator(0), numTri,
      InteriorVerts({vertPos_.ptrD(), triVertStart, n, halfedge_.ptrD()}));
  // Create subtriangles
  VecDH<glm::ivec3> triVerts(n * n * numTri);
  thrust::for_each_n(thrust::make_counting_iterator(0), numTri,
                     SplitTris({triVerts.ptrD(), halfedge_.cptrD(),
                                half2Edge.cptrD(), numVert, triVertStart, n}));
  CreateHalfedges(triVerts);
}

bool Manifold::Impl::IsManifold() const {
  return thrust::all_of(thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(NumFace()),
                        CheckManifold({halfedge_.cptrD(), faceEdge_.cptrD()}));
}

Manifold::Properties Manifold::Impl::GetProperties() const {
  ApplyTransform();
  thrust::pair<float, float> areaVolume = thrust::transform_reduce(
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(NumFace()),
      FaceAreaVolume({halfedge_.cptrD(), faceEdge_.cptrD(), vertPos_.cptrD()}),
      thrust::make_pair(0.0f, 0.0f), SumPair());
  return {areaVolume.first, areaVolume.second};
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

  ReindexVerts(vertNew2Old, NumVert());
}

void Manifold::Impl::ReindexVerts(const VecDH<int>& vertNew2Old,
                                  int oldNumVert) {
  VecDH<int> vertOld2New(oldNumVert);
  thrust::scatter(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(NumVert()),
                  vertNew2Old.beginD(), vertOld2New.beginD());
  thrust::for_each(halfedge_.beginD(), halfedge_.endD(),
                   Reindex({vertOld2New.cptrD()}));
}

void Manifold::Impl::GetFaceBoxMorton(VecDH<Box>& faceBox,
                                      VecDH<uint32_t>& faceMorton) const {
  faceBox.resize(NumFace());
  faceMorton.resize(NumFace());
  thrust::for_each_n(zip(faceMorton.beginD(), faceBox.beginD(),
                         thrust::make_counting_iterator(0)),
                     NumFace(),
                     FaceMortonBox({faceEdge_.cptrD(), halfedge_.cptrD(),
                                    vertPos_.cptrD(), bBox_}));
}

void Manifold::Impl::SortFaces(VecDH<Box>& faceBox,
                               VecDH<uint32_t>& faceMorton) {
  VecDH<int> faceNew2Old(NumFace());
  thrust::sequence(faceNew2Old.beginD(), faceNew2Old.endD());

  VecDH<int> faceSize = FaceSize();

  thrust::sort_by_key(faceMorton.beginD(), faceMorton.endD(),
                      zip(faceBox.beginD(), faceNew2Old.beginD(),
                          faceSize.beginD() + 1, faceNormal_.beginD()));

  VecDH<Halfedge> oldHalfedge = halfedge_;
  VecDH<int> oldFaceEdge = faceEdge_;
  GatherFaces(oldHalfedge, oldFaceEdge, faceNew2Old, faceSize);
}

VecDH<int> Manifold::Impl::FaceSize() const {
  VecDH<int> faceSize(faceEdge_.size());
  thrust::adjacent_difference(faceEdge_.beginD(), faceEdge_.endD(),
                              faceSize.beginD());
  return faceSize;
}

void Manifold::Impl::GatherFaces(const VecDH<Halfedge>& oldHalfedge,
                                 const VecDH<int>& oldFaceEdge,
                                 const VecDH<int>& faceNew2Old,
                                 const VecDH<int>& newFaceSize) {
  VecDH<int> faceOld2New(oldFaceEdge.size() - 1);
  thrust::scatter(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(NumFace()),
                  faceNew2Old.beginD(), faceOld2New.beginD());

  thrust::inclusive_scan(newFaceSize.beginD() + 1, newFaceSize.endD(),
                         faceEdge_.beginD() + 1);

  thrust::for_each_n(zip(thrust::make_counting_iterator(0), faceEdge_.beginD()),
                     NumFace(),
                     ReindexFace({halfedge_.ptrD(), faceEdge_.cptrD(),
                                  oldHalfedge.cptrD(), oldFaceEdge.cptrD(),
                                  faceNew2Old.cptrD(), faceOld2New.cptrD()}));
}

void Manifold::Impl::CalculateNormals() {
  vertNormal_.resize(NumVert());
  bool calculateTriNormal = false;
  if (faceNormal_.size() != NumFace()) {
    faceNormal_.resize(NumFace());
    calculateTriNormal = true;
  }
  thrust::for_each_n(
      zip(faceNormal_.beginD(), thrust::make_counting_iterator(0)), NumFace(),
      AssignNormals({vertNormal_.ptrD(), vertPos_.cptrD(), halfedge_.cptrD(),
                     nextHalfedge_.cptrD(), faceEdge_.cptrD(),
                     calculateTriNormal}));
  thrust::for_each(vertNormal_.begin(), vertNormal_.end(), NormalizeTo({1.0}));
}

SparseIndices Manifold::Impl::EdgeCollisions(const Impl& Q) const {
  VecDH<TmpEdge> edges = CreateTmpEdges(Q.halfedge_);
  const int numEdge = edges.size();
  VecDH<Box> QedgeBB(numEdge);
  thrust::for_each_n(zip(QedgeBB.beginD(), edges.cbeginD()), numEdge,
                     EdgeBox({Q.vertPos_.cptrD()}));

  SparseIndices q1p2 = collider_.Collisions(QedgeBB);

  thrust::for_each(q1p2.beginD(0), q1p2.endD(0), ReindexEdge({edges.cptrD()}));
  return q1p2;
}

SparseIndices Manifold::Impl::VertexCollisionsZ(
    const VecDH<glm::vec3>& vertsIn) const {
  return collider_.Collisions(vertsIn);
}

void Manifold::Impl::NextEdges(int* nextHalfedge, const Halfedge* edgeBegin,
                               const Halfedge* edgeEnd) {
  int numEdge = edgeEnd - edgeBegin;
  std::map<int, int> vert_edge;
  for (int i = 0; i < numEdge; ++i) {
    ALWAYS_ASSERT(
        vert_edge.emplace(std::make_pair(edgeBegin[i].startVert, i)).second,
        runtimeErr, "polygon has duplicate vertices.");
  }

  auto startEdge = edgeBegin;
  auto thisEdge = edgeBegin;
  for (;;) {
    if (thisEdge == startEdge) {
      if (vert_edge.empty()) break;
      startEdge = std::next(edgeBegin, vert_edge.begin()->second);
      thisEdge = startEdge;
    }
    auto result = vert_edge.find(thisEdge->endVert);
    ALWAYS_ASSERT(result != vert_edge.end(), runtimeErr, "nonmanifold edge");
    auto nextEdge = std::next(edgeBegin, result->second);
    nextHalfedge[thisEdge - edgeBegin] = result->second;
    thisEdge = nextEdge;
    vert_edge.erase(result);
  }
}

Polygons Manifold::Impl::Face2Polygons(
    int face, std::function<glm::vec2(int)> vertProjection) const {
  const VecH<int>& faceEdge = faceEdge_.H();
  const VecH<Halfedge>& halfedge = halfedge_.H();
  const VecH<int>& nextHalfedge = nextHalfedge_.H();
  const int edge = faceEdge[face];
  const int lastEdge = faceEdge[face + 1];

  Polygons polys;
  std::vector<bool> visited(lastEdge - edge, false);
  int startEdge = edge;
  int thisEdge = edge;
  for (;;) {
    if (thisEdge == startEdge) {
      auto next = std::find(visited.begin(), visited.end(), false);
      if (next == visited.end()) break;
      startEdge = next - visited.begin();
      thisEdge = startEdge;
      polys.push_back({});
    }
    int vert = halfedge[thisEdge].startVert;
    polys.back().push_back(
        {vertProjection(vert), vert, halfedge[thisEdge].face});
    thisEdge = nextHalfedge[thisEdge];
  }
  return polys;
}
}  // namespace manifold