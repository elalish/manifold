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

/**
 * Represents the uncertainty of the vertices (greater than or equal to
 * worst-case floating-point precision). Used to determine when face surface
 * area or volume is small enough to clamp to zero. TODO: this should be based
 * on the bounding box, and probably passed through Boolean operations. It
 * should also be passed into the Polygon triangulator, where it is more
 * important.
 */
constexpr float kTolerance = 1e-5;

struct NormalizeTo {
  float length;
  __host__ __device__ void operator()(glm::vec3& v) {
    v = length * glm::normalize(v);
    if (isnan(v.x)) v = glm::vec3(0.0);
  }
};

/**
 * This is a temporary edge strcture which only stores edges forward and
 * references the halfedge it was created from.
 */
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
  const glm::vec3* vertPos;

  __host__ __device__ thrust::pair<float, float> operator()(int face) {
    float perimeter = 0.0f;
    float area = 0.0f;
    float volume = 0.0f;

    glm::vec3 edge[3];
    for (int i : {0, 1, 2}) {
      const int j = (i + 1) % 3;
      edge[i] = vertPos[halfedges[3 * face + j].startVert] -
                vertPos[halfedges[3 * face + i].startVert];
      perimeter += glm::length(edge[i]);
    }
    glm::vec3 crossP = glm::cross(edge[0], edge[1]);

    area += glm::length(crossP);
    volume += glm::dot(crossP, vertPos[halfedges[3 * face].startVert]);

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
  const Halfedge* halfedge;
  const glm::vec3* vertPos;
  const Box bBox;

  __host__ __device__ void operator()(
      thrust::tuple<uint32_t&, Box&, int> inout) {
    uint32_t& mortonCode = thrust::get<0>(inout);
    Box& faceBox = thrust::get<1>(inout);
    int face = thrust::get<2>(inout);

    glm::vec3 center(0.0f);

    for (const int i : {0, 1, 2}) {
      const glm::vec3 pos = vertPos[halfedge[3 * face + i].startVert];
      center += pos;
      faceBox.Union(pos);
    }
    center /= 3;

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
  const bool calculateTriNormal;

  __host__ __device__ void operator()(thrust::tuple<glm::vec3&, int> in) {
    glm::vec3& triNormal = thrust::get<0>(in);
    const int face = thrust::get<1>(in);

    glm::ivec3 triVerts(halfedges[3 * face].startVert,
                        halfedges[3 * face].endVert,
                        halfedges[3 * face + 1].endVert);
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
      if (isnan(phi[i])) phi[i] = 0;
      AtomicAddVec3(vertNormal[triVerts[i]],
                    glm::max(phi[i], kTolerance) * triNormal);
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

  __host__ __device__ bool operator()(int face) {
    bool good = true;
    for (const int i : {0, 1, 2}) {
      const int edge = 3 * face + i;
      const Halfedge halfedge = halfedges[edge];
      const Halfedge paired = halfedges[halfedge.pairedHalfedge];
      good &= halfedge.face == face;
      good &= paired.pairedHalfedge == edge;
      good &= halfedge.startVert != halfedge.endVert;
      good &= halfedge.startVert == paired.endVert;
      good &= halfedge.endVert == paired.startVert;
    }
    return good;
  }
};

struct NoDuplicates {
  const Halfedge* halfedges;

  __host__ __device__ bool operator()(int edge) {
    return halfedges[edge].startVert != halfedges[edge + 1].startVert ||
           halfedges[edge].endVert != halfedges[edge + 1].endVert;
  }
};

/**
 * By using the closest axis-aligned projection to the normal instead of a
 * projection along the normal, we avoid introducing any rounding error.
 */
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

/**
 * Create a manifold from an input triangle Mesh. Will throw if the Mesh is not
 * manifold.
 */
Manifold::Impl::Impl(const Mesh& manifold) : vertPos_(manifold.vertPos) {
  CheckDevice();
  CreateHalfedges(manifold.triVerts);
  Finish();
}

/**
 * Create eiter a unit tetrahedron, cube or octahedron. The cube is in the first
 * octant, while the others are symmetric about the origin.
 */
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

/**
 * Create the halfedge_ data structure from an input triVerts array like Mesh.
 */
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

/**
 * Calculate vertLabels_ by running connected components on the halfedges. This
 * operation is a bit slow and currently CPU-only. Note: by operating on
 * halfedges, connectivity can be broken by faces that are polygons with holes
 * (no edges to attach one polygon to the other, even though they are part of
 * the same face). This style of labeling is consistent with what is needed in
 * the Boolean operation. To separate manifolds topologically, it is best to
 * first triangulate them.
 */
void Manifold::Impl::LabelVerts() {
  numLabel_ = ConnectedComponents(vertLabel_, NumVert(), halfedge_);
}

/**
 * Once halfedge_ and faceEdge_ have been filled in, this function can be called
 * to create the rest of the internal data structures. If vertLabel_ hasn't been
 * filled in, it is assumed the object is simply-connected and numLabel_ is set
 * to 1.
 */
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
  SortFaces(faceBox, faceMorton);
  CalculateNormals();
  collider_ = Collider(faceBox, faceMorton);
}

/**
 * Does a full recalculation of the face bounding boxes, including updating the
 * collider, but does not resort the faces.
 */
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

/**
 * Bake the manifold's transform into its vertices. This function allows lazy
 * evaluation, which is important because often several transforms are applied
 * between operations.
 */
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

/**
 * This fills in the nextHalfedge_ vector indicating how the halfedges connect
 * to each other going CCW around a face. This data cannot be stored by simply
 * sorting the halfedges, as the faces may be polygons with holes.
 *
 * TODO: This function is slow and should be moved from CPU to GPU.
 */
VecH<int> Manifold::Impl::AssembleFaces(const VecH<int>& faceEdge) const {
  VecH<int> nextHalfedge(halfedge_.size());
  const VecH<Halfedge>& halfedge = halfedge_.H();

  for (int face = 0; face < NumFace(); ++face) {
    int edge = faceEdge[face];
    const int nEdge = faceEdge[face + 1] - edge;
    ALWAYS_ASSERT(nEdge >= 3, runtimeErr, "face has less than three edges.");
    if (nEdge == 3) {
      const bool forward =
          halfedge[edge].endVert == halfedge[edge + 1].startVert;
      const int edge1 = edge + (forward ? 1 : 2);
      const int edge2 = edge + (forward ? 2 : 1);
      ALWAYS_ASSERT(halfedge[edge].endVert == halfedge[edge1].startVert &&
                        halfedge[edge1].endVert == halfedge[edge2].startVert &&
                        halfedge[edge2].endVert == halfedge[edge].startVert,
                    runtimeErr, "triangle does not assemble.");
      nextHalfedge[edge] = edge1;
      nextHalfedge[edge1] = edge2;
      nextHalfedge[edge2] = edge;
      continue;
    }
    std::map<int, int> vert_edge;
    for (; edge < faceEdge[face + 1]; ++edge) {
      ALWAYS_ASSERT(
          vert_edge.emplace(std::make_pair(halfedge[edge].startVert, edge))
              .second,
          runtimeErr, "face has duplicate vertices.");
    }

    int startEdge = 0;
    int thisEdge = startEdge;
    while (1) {
      if (thisEdge == startEdge) {
        if (vert_edge.empty()) break;
        startEdge = vert_edge.begin()->second;
        thisEdge = startEdge;
      }
      const auto result = vert_edge.find(halfedge[thisEdge].endVert);
      ALWAYS_ASSERT(result != vert_edge.end(), runtimeErr, "nonmanifold edge");
      nextHalfedge[thisEdge] = result->second;
      thisEdge = result->second;
      vert_edge.erase(result);
    }
  }
  return nextHalfedge;
}

bool Manifold::Impl::Tri2Face() const {
  // This const_cast is here because this operation tweaks the internal data
  // structure, but does not change what it represents.
  return const_cast<Impl*>(this)->Tri2Face();
}

/**
 * Fills in the faceEdge_ structure for the situation where the halfedges
 * correspond to triVerts entries.
 */
bool Manifold::Impl::Tri2Face() {
  if (faceEdge_.size() != 0 || halfedge_.size() % 3 != 0) return false;
  faceEdge_.resize(halfedge_.size() / 3 + 1);
  thrust::sequence(faceEdge_.beginD(), faceEdge_.endD(), 0, 3);
  return true;
}

/**
 * Triangulates the faces. It is possible, but rare, that this function can
 * also add vertices. This never happens for geometrically valid manifolds.
 */
bool Manifold::Impl::Face2Tri(const VecDH<int>& faceEdge) {
  if (faceEdge.size() == 0 && halfedge_.size() % 3 == 0) return false;
  VecDH<glm::ivec3> triVertsOut;
  VecDH<glm::vec3> triNormalOut;

  VecH<glm::ivec3>& triVerts = triVertsOut.H();
  VecH<glm::vec3>& triNormal = triNormalOut.H();
  VecH<glm::vec3>& vertPos = vertPos_.H();
  const VecH<int>& face = faceEdge.H();
  const VecH<Halfedge>& halfedge = halfedge_.H();
  const VecH<glm::vec3>& faceNormal = faceNormal_.H();
  const VecH<int> nextHalfedge = AssembleFaces(face);

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
      Polygons polys = Face2Polygons(i, projection, face, nextHalfedge);

      std::vector<glm::ivec3> newTris = Triangulate(polys);

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

/**
 * Triangulate the manifold, then split each resulting edge into n pieces and
 * sub-triangulate each triangle accordingly. This function doesn't run
 * Finish(), as that is expensive and it'll need to be run after the new
 * vertices have moved, which is a likely scenario after refinement (smoothing).
 */
void Manifold::Impl::Refine(int n) {
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

/**
 * Returns true if this manifold is in fact an oriented 2-manifold and all of
 * the data structures are consistent.
 */
bool Manifold::Impl::IsManifold() const {
  if (halfedge_.size() == 0) return true;
  bool isManifold = thrust::all_of(thrust::make_counting_iterator(0),
                                   thrust::make_counting_iterator(NumFace()),
                                   CheckManifold({halfedge_.cptrD()}));

  VecDH<Halfedge> halfedge(halfedge_);
  thrust::sort(halfedge.beginD(), halfedge.endD());
  isManifold &=
      thrust::all_of(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(2 * NumEdge() - 1),
                     NoDuplicates({halfedge.cptrD()}));
  return isManifold;
}

/**
 * Returns the surface area and volume of the manifold in a Properties
 * structure. These properties are clamped to zero for a given face if they are
 * within rounding tolerance. This means degenerate manifolds can by identified
 * by testing these properties as == 0.
 */
Manifold::Properties Manifold::Impl::GetProperties() const {
  if (halfedge_.size() == 0) return {0, 0};
  ApplyTransform();
  thrust::pair<float, float> areaVolume = thrust::transform_reduce(
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(NumFace()),
      FaceAreaVolume({halfedge_.cptrD(), vertPos_.cptrD()}),
      thrust::make_pair(0.0f, 0.0f), SumPair());
  return {areaVolume.first, areaVolume.second};
}

/**
 * Calculates the bounding box of the entire manifold, which is stored
 * internally to short-cut Boolean operations and to serve as the precision
 * range for Morton code calulation.
 */
void Manifold::Impl::CalculateBBox() {
  bBox_.min = thrust::reduce(vertPos_.begin(), vertPos_.end(),
                             glm::vec3(1 / 0.0f), PosMin());
  bBox_.max = thrust::reduce(vertPos_.begin(), vertPos_.end(),
                             glm::vec3(-1 / 0.0f), PosMax());
  ALWAYS_ASSERT(bBox_.isFinite(), runtimeErr,
                "Input vertices are not all finite!");
}

/**
 * Sorts the vertices according to their Morton code.
 */
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

/**
 * Updates the halfedges to point to new vert indices based on a mapping,
 * vertNew2Old. This may be a subset, so the total number of original verts is
 * also given.
 */
void Manifold::Impl::ReindexVerts(const VecDH<int>& vertNew2Old,
                                  int oldNumVert) {
  VecDH<int> vertOld2New(oldNumVert);
  thrust::scatter(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(NumVert()),
                  vertNew2Old.beginD(), vertOld2New.beginD());
  thrust::for_each(halfedge_.beginD(), halfedge_.endD(),
                   Reindex({vertOld2New.cptrD()}));
}

/**
 * Fills the faceBox and faceMorton input with the bounding boxes and Morton
 * codes of the faces, respectively. The Morton code is based on the center of
 * the bounding box.
 */
void Manifold::Impl::GetFaceBoxMorton(VecDH<Box>& faceBox,
                                      VecDH<uint32_t>& faceMorton) const {
  faceBox.resize(NumFace());
  faceMorton.resize(NumFace());
  thrust::for_each_n(
      zip(faceMorton.beginD(), faceBox.beginD(),
          thrust::make_counting_iterator(0)),
      NumFace(), FaceMortonBox({halfedge_.cptrD(), vertPos_.cptrD(), bBox_}));
}

/**
 * Sorts the faces of this manifold according to their input Morton code. The
 * bounding box and Morton code arrays are also sorted accordingly.
 */
void Manifold::Impl::SortFaces(VecDH<Box>& faceBox,
                               VecDH<uint32_t>& faceMorton) {
  VecDH<int> faceNew2Old(NumFace());
  thrust::sequence(faceNew2Old.beginD(), faceNew2Old.endD());

  VecDH<int> faceSize = FaceSize();

  if (faceNormal_.size() == NumFace()) {
    thrust::sort_by_key(faceMorton.beginD(), faceMorton.endD(),
                        zip(faceBox.beginD(), faceNew2Old.beginD(),
                            faceSize.beginD() + 1, faceNormal_.beginD()));
  } else {
    thrust::sort_by_key(
        faceMorton.beginD(), faceMorton.endD(),
        zip(faceBox.beginD(), faceNew2Old.beginD(), faceSize.beginD() + 1));
  }

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

/**
 * Creates the halfedge_ and faceEdge_ vectors for this manifold by copying a
 * set of faces from another manifold, given by oldHalfedge and oldFaceEdge.
 * Input faceNew2Old defines the old faces to gather into this, while
 * newFaceSize is the same length as faceNew2Old and contains the sizes of the
 * faces to be copied.
 */
void Manifold::Impl::GatherFaces(const VecDH<Halfedge>& oldHalfedge,
                                 const VecDH<int>& oldFaceEdge,
                                 const VecDH<int>& faceNew2Old,
                                 const VecDH<int>& newFaceSize) {
  faceEdge_.resize(faceNew2Old.size() + 1);

  VecDH<int> faceOld2New(oldFaceEdge.size() - 1);
  thrust::scatter(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(NumFace()),
                  faceNew2Old.beginD(), faceOld2New.beginD());

  thrust::inclusive_scan(newFaceSize.beginD() + 1, newFaceSize.endD(),
                         faceEdge_.beginD() + 1);

  halfedge_.resize(faceEdge_.H().back());
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), faceEdge_.beginD()),
                     NumFace(),
                     ReindexFace({halfedge_.ptrD(), faceEdge_.cptrD(),
                                  oldHalfedge.cptrD(), oldFaceEdge.cptrD(),
                                  faceNew2Old.cptrD(), faceOld2New.cptrD()}));
}

/**
 * If face normals are already present, this function uses them to compute
 * vertex normals (angle-weighted pseudo-normals); otherwise it also computes
 * the face normals as well. Face normals are only calculated when needed
 * because nearly degenerate faces will accrue rounding error, while the Boolean
 * can retain their original normal, which is more accurate and can help with
 * merging coplanar faces.
 *
 * If the face normals have been invalidated by an operation like Warp(), ensure
 * you do faceNormal_.resize(0) before calling this function to force
 * recalculation.
 */
void Manifold::Impl::CalculateNormals() {
  vertNormal_.resize(NumVert(), glm::vec3(0.0f));
  bool calculateTriNormal = false;
  if (faceNormal_.size() != NumFace()) {
    faceNormal_.resize(NumFace());
    calculateTriNormal = true;
  }
  thrust::for_each_n(
      zip(faceNormal_.beginD(), thrust::make_counting_iterator(0)), NumFace(),
      AssignNormals({vertNormal_.ptrD(), vertPos_.cptrD(), halfedge_.cptrD(),
                     calculateTriNormal}));
  thrust::for_each(vertNormal_.begin(), vertNormal_.end(), NormalizeTo({1.0}));
}

/**
 * Returns a sparse array of the bounding box overlaps between the edges of the
 * input manifold, Q and the faces of this manifold. Returned indices only
 * point to forward halfedges.
 */
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

/**
 * Returns a sparse array of the input vertices that project inside the XY
 * bounding boxes of the faces of this manifold.
 */
SparseIndices Manifold::Impl::VertexCollisionsZ(
    const VecDH<glm::vec3>& vertsIn) const {
  return collider_.Collisions(vertsIn);
}

/**
 * For the input face index, return a set of 2D polygons formed by the input
 * projection of the vertices.
 */
Polygons Manifold::Impl::Face2Polygons(int face, glm::mat3x2 projection,
                                       const VecH<int>& faceEdge,
                                       const VecH<int>& nextHalfedge) const {
  const VecH<Halfedge>& halfedge = halfedge_.H();
  const VecH<glm::vec3>& vertPos = vertPos_.H();
  const int firstEdge = faceEdge[face];
  const int lastEdge = faceEdge[face + 1];

  Polygons polys;
  std::vector<bool> visited(lastEdge - firstEdge, false);
  int startEdge = firstEdge;
  int thisEdge = firstEdge;
  while (1) {
    if (thisEdge == startEdge) {
      auto next = std::find(visited.begin(), visited.end(), false);
      if (next == visited.end()) break;
      startEdge = next - visited.begin() + firstEdge;
      thisEdge = startEdge;
      polys.push_back({});
    }
    int vert = halfedge[thisEdge].startVert;
    polys.back().push_back({projection * vertPos[vert], vert,
                            halfedge[halfedge[thisEdge].pairedHalfedge].face});
    visited[thisEdge - firstEdge] = true;
    thisEdge = nextHalfedge[thisEdge];
  }
  return polys;
}
}  // namespace manifold