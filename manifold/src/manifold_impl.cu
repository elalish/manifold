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

#include <thrust/adjacent_difference.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <map>
#include <set>

#include "connected_components.cuh"
#include "manifold_impl.cuh"
#include "polygon.h"

namespace {
using namespace manifold;

constexpr uint32_t kNoCode = 0xFFFFFFFFu;

__host__ __device__ glm::vec3 SafeNormalize(glm::vec3 v) {
  v = glm::normalize(v);
  return isfinite(v.x) ? v : glm::vec3(0);
}

__host__ __device__ glm::vec3 OrthogonalTo(glm::vec3 in, glm::vec3 ref) {
  in -= glm::dot(in, ref) * ref;
  return in;
}

__host__ __device__ int NextHalfedge(int current) {
  ++current;
  if (current % 3 == 0) current -= 3;
  return current;
}

/**
 * The total number of verts if a triangle is subdivided naturally such that
 * each edge has edgeVerts verts along it (edgeVerts >= -1).
 */
__host__ __device__ int VertsPerTri(int edgeVerts) {
  return (edgeVerts * edgeVerts + edgeVerts) / 2;
}

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

/**
 * By using the closest axis-aligned projection to the normal instead of a
 * projection along the normal, we avoid introducing any rounding error.
 */
__host__ __device__ glm::mat3x2 GetAxisAlignedProjection(glm::vec3 normal) {
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

struct Barycentric {
  int tri;
  glm::vec3 uvw;
};

struct Normalize {
  __host__ __device__ void operator()(glm::vec3& v) {
    v = glm::normalize(v);
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
  thrust::for_each_n(zip(edges.beginD(), halfedge.beginD(), countAt(0)),
                     edges.size(), Halfedge2Tmp());
  int numEdge = thrust::remove_if(edges.beginD(), edges.endD(), TmpInvalid()) -
                edges.beginD();
  ALWAYS_ASSERT(numEdge == halfedge.size() / 2, topologyErr, "Not oriented!");
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

struct EdgeVerts {
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
  glm::vec3* uvw;
  BaryRef* triBary;
  const int startIdx;
  const int n;
  const Halfedge* halfedge;

  __host__ __device__ void operator()(int tri) {
    const float invTotal = 1.0f / n;
    int posTri = tri * n * n;
    int posBary = tri * VertsPerTri(n + 1);
    int pos = startIdx + tri * VertsPerTri(n - 2);
    for (int i = 0; i <= n; ++i)
      for (int j = 0; j <= n - i; ++j) {
        const int k = n - i - j;
        const float u = invTotal * j;
        const float v = invTotal * k;
        const float w = invTotal * i;
        const int first = posBary;
        uvw[posBary++] = {u, v, w};
        if (j == n - i) continue;

        // The three retained verts are denoted -1. uvw entries are added for
        // them out of laziness of indexing only.
        const int a = (k == n) ? -1 : first;
        const int b = (i == n - 1) ? -1 : first + n - i + 1;
        const int c = (j == n - 1) ? -1 : first + 1;
        triBary[posTri++] = {tri, {c, a, b}};
        if (j < n - 1 - i) {
          int d = b + 1;
          triBary[posTri++] = {tri, {b, d, c}};
        }

        if (i == 0 || j == 0 || k == 0) continue;

        vertPos[pos++] = u * vertPos[halfedge[3 * tri].startVert] +      //
                         v * vertPos[halfedge[3 * tri + 1].startVert] +  //
                         w * vertPos[halfedge[3 * tri + 2].startVert];
      }
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
        triVerts[pos++] = glm::ivec3(c, a, b);
        if (j < n - 1 - i) {
          int d = Vert(i + 1, j + 1, tri);
          triVerts[pos++] = glm::ivec3(b, d, c);
        }
      }
    }
  }
};

struct SmoothBezier {
  const glm::vec3* vertPos;
  const glm::vec3* triNormal;
  const glm::vec3* vertNormal;
  const Halfedge* halfedge;

  __host__ __device__ void operator()(
      thrust::tuple<glm::vec4&, Halfedge> inOut) {
    glm::vec4& tangent = thrust::get<0>(inOut);
    const Halfedge edge = thrust::get<1>(inOut);

    const glm::vec3 startV = vertPos[edge.startVert];
    const glm::vec3 edgeVec = vertPos[edge.endVert] - startV;
    const glm::vec3 edgeNormal =
        (triNormal[edge.face] + triNormal[halfedge[edge.pairedHalfedge].face]) /
        2.0f;
    glm::vec3 dir = glm::normalize(glm::cross(glm::cross(edgeNormal, edgeVec),
                                              vertNormal[edge.startVert]));

    const float weight = glm::abs(glm::dot(dir, glm::normalize(edgeVec)));
    // Quadratic weighted bezier for circular interpolation
    const glm::vec4 bz2 =
        weight *
        glm::vec4(startV + dir * glm::length(edgeVec) / (2 * weight), 1.0f);
    // Equivalent cubic weighted bezier
    const glm::vec4 bz3 = glm::mix(glm::vec4(startV, 1.0f), bz2, 2 / 3.0f);
    // Convert from homogeneous form to geometric form
    tangent = glm::vec4(glm::vec3(bz3) / bz3.w - startV, bz3.w);
  }
};

struct TriBary2Vert {
  Barycentric* vertBary;
  int* lock;
  const glm::vec3* uvw;
  const Halfedge* halfedge;

  __host__ __device__ void operator()(thrust::tuple<BaryRef, int> in) {
    const BaryRef baryRef = thrust::get<0>(in);
    const int tri = thrust::get<1>(in);

    for (int i : {0, 1, 2}) {
      int vert = halfedge[3 * tri + i].startVert;
      if (AtomicAdd(lock[vert], 1) != 0) continue;

      const int idx = baryRef.vertBary[i];
      glm::vec3 bary(0);
      if (idx < 0)
        bary[i] = 1;
      else
        bary = uvw[idx];
      vertBary[vert] = {baryRef.tri, bary};
    }
  }
};

struct InterpTri {
  const Halfedge* halfedge;
  const glm::vec4* halfedgeTangent;
  const glm::vec3* vertPos;

  __host__ __device__ glm::vec4 Homogeneous(glm::vec4 v) const {
    v.x *= v.w;
    v.y *= v.w;
    v.z *= v.w;
    return v;
  }

  __host__ __device__ glm::vec4 Homogeneous(glm::vec3 v) const {
    return glm::vec4(v, 1.0f);
  }

  __host__ __device__ glm::vec3 HNormalize(glm::vec4 v) const {
    return glm::vec3(v) / v.w;
  }

  __host__ __device__ glm::vec4 Bezier(glm::vec3 point,
                                       glm::vec4 tangent) const {
    return Homogeneous(glm::vec4(point, 0) + tangent);
  }

  __host__ __device__ glm::mat2x4 CubicBezier2Linear(glm::vec4 p0, glm::vec4 p1,
                                                     glm::vec4 p2, glm::vec4 p3,
                                                     float x) const {
    glm::mat2x4 out;
    glm::vec4 p12 = glm::mix(p1, p2, x);
    out[0] = glm::mix(glm::mix(p0, p1, x), p12, x);
    out[1] = glm::mix(p12, glm::mix(p2, p3, x), x);
    return out;
  }

  __host__ __device__ glm::vec3 BezierPoint(glm::mat2x4 points, float x) const {
    return HNormalize(glm::mix(points[0], points[1], x));
  }

  __host__ __device__ glm::vec3 BezierTangent(glm::mat2x4 points) const {
    return glm::normalize(HNormalize(points[1]) - HNormalize(points[0]));
  }

  __host__ __device__ void operator()(
      thrust::tuple<glm::vec3&, Barycentric> inOut) {
    glm::vec3& pos = thrust::get<0>(inOut);
    const int tri = thrust::get<1>(inOut).tri;
    const glm::vec3 uvw = thrust::get<1>(inOut).uvw;

    glm::vec4 posH(0);
    const glm::mat3 corners = {vertPos[halfedge[3 * tri].startVert],
                               vertPos[halfedge[3 * tri + 1].startVert],
                               vertPos[halfedge[3 * tri + 2].startVert]};

    for (const int i : {0, 1, 2}) {
      if (uvw[i] == 1) {
        pos = glm::vec3(corners[i]);
        return;
      }
    }

    const glm::mat3x4 tangentR = {halfedgeTangent[3 * tri],
                                  halfedgeTangent[3 * tri + 1],
                                  halfedgeTangent[3 * tri + 2]};
    const glm::mat3x4 tangentL = {
        halfedgeTangent[halfedge[3 * tri + 2].pairedHalfedge],
        halfedgeTangent[halfedge[3 * tri].pairedHalfedge],
        halfedgeTangent[halfedge[3 * tri + 1].pairedHalfedge]};

    for (const int i : {0, 1, 2}) {
      const int j = (i + 1) % 3;
      const int k = (i + 2) % 3;
      const float x = uvw[k] / (1 - uvw[i]);

      const glm::mat2x4 bez = CubicBezier2Linear(
          Homogeneous(corners[j]), Bezier(corners[j], tangentR[j]),
          Bezier(corners[k], tangentL[k]), Homogeneous(corners[k]), x);
      const glm::vec3 end = BezierPoint(bez, x);
      const glm::vec3 tangent = BezierTangent(bez);

      const glm::vec3 jBitangent = SafeNormalize(OrthogonalTo(
          glm::vec3(tangentL[j]), SafeNormalize(glm::vec3(tangentR[j]))));
      const glm::vec3 kBitangent = SafeNormalize(OrthogonalTo(
          glm::vec3(tangentR[k]), -SafeNormalize(glm::vec3(tangentL[k]))));
      const glm::vec3 normal = SafeNormalize(
          glm::cross(glm::mix(jBitangent, kBitangent, x), tangent));
      const glm::vec3 delta = OrthogonalTo(
          glm::mix(glm::vec3(tangentL[j]), glm::vec3(tangentR[k]), x), normal);
      const float deltaW = glm::mix(tangentL[j].w, tangentR[k].w, x);

      const glm::mat2x4 bez1 = CubicBezier2Linear(
          Homogeneous(end), Homogeneous(glm::vec4(end + delta, deltaW)),
          Bezier(corners[i], glm::mix(tangentR[i], tangentL[i], x)),
          Homogeneous(corners[i]), uvw[i]);
      const glm::vec3 p = BezierPoint(bez1, uvw[i]);
      float w = uvw[j] * uvw[j] * uvw[k] * uvw[k];
      posH += Homogeneous(glm::vec4(p, w));
    }
    pos = HNormalize(posH);
  }
};

struct FaceAreaVolume {
  const Halfedge* halfedges;
  const glm::vec3* vertPos;
  const float precision;

  __host__ __device__ thrust::pair<float, float> operator()(int face) {
    float perimeter = 0;
    glm::vec3 edge[3];
    for (int i : {0, 1, 2}) {
      const int j = (i + 1) % 3;
      edge[i] = vertPos[halfedges[3 * face + j].startVert] -
                vertPos[halfedges[3 * face + i].startVert];
      perimeter += glm::length(edge[i]);
    }
    glm::vec3 crossP = glm::cross(edge[0], edge[1]);

    float area = glm::length(crossP);
    float volume = glm::dot(crossP, vertPos[halfedges[3 * face].startVert]);

    return area > perimeter * precision
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
    if (isnan(a.x)) return b;
    if (isnan(b.x)) return a;
    return glm::min(a, b);
  }
};

struct PosMax
    : public thrust::binary_function<glm::vec3, glm::vec3, glm::vec3> {
  __host__ __device__ glm::vec3 operator()(glm::vec3 a, glm::vec3 b) {
    if (isnan(a.x)) return b;
    if (isnan(b.x)) return a;
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

struct Transform4x3 {
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
  // Unreferenced vertices are marked NaN, and this will sort them to the end
  // (the Morton code only uses the first 30 of 32 bits).
  if (isnan(position.x)) return kNoCode;

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

    // Removed tris are marked by all halfedges having pairedHalfedge = -1, and
    // this will sort them to the end (the Morton code only uses the first 30 of
    // 32 bits).
    if (halfedge[3 * face].pairedHalfedge < 0) {
      mortonCode = kNoCode;
      return;
    }

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
    if (edge.startVert < 0) return;
    edge.startVert = indexInv[edge.startVert];
    edge.endVert = indexInv[edge.endVert];
  }
};

template <typename T>
void Permute(VecDH<T>& inOut, const VecDH<int>& new2Old) {
  VecDH<T> tmp(inOut);
  inOut.resize(new2Old.size());
  thrust::gather(new2Old.beginD(), new2Old.endD(), tmp.beginD(),
                 inOut.beginD());
}

template void Permute<BaryRef>(VecDH<BaryRef>&, const VecDH<int>&);
template void Permute<glm::vec3>(VecDH<glm::vec3>&, const VecDH<int>&);

struct ReindexFace {
  Halfedge* halfedge;
  glm::vec4* halfedgeTangent;
  const Halfedge* oldHalfedge;
  const glm::vec4* oldHalfedgeTangent;
  const int* faceNew2Old;
  const int* faceOld2New;

  __host__ __device__ void operator()(int newFace) {
    const int oldFace = faceNew2Old[newFace];
    for (const int i : {0, 1, 2}) {
      const int oldEdge = 3 * oldFace + i;
      Halfedge edge = oldHalfedge[oldEdge];
      edge.face = newFace;
      const int pairedFace = oldHalfedge[edge.pairedHalfedge].face;
      const int offset = edge.pairedHalfedge - 3 * pairedFace;
      edge.pairedHalfedge = 3 * faceOld2New[pairedFace] + offset;
      const int newEdge = 3 * newFace + i;
      halfedge[newEdge] = edge;
      if (oldHalfedgeTangent != nullptr) {
        halfedgeTangent[newEdge] = oldHalfedgeTangent[oldEdge];
      }
    }
  }
};

struct AssignNormals {
  glm::vec3* vertNormal;
  const glm::vec3* vertPos;
  const Halfedge* halfedges;
  const float precision;
  const bool calculateTriNormal;

  __host__ __device__ void operator()(thrust::tuple<glm::vec3&, int> in) {
    glm::vec3& triNormal = thrust::get<0>(in);
    const int face = thrust::get<1>(in);

    glm::ivec3 triVerts;
    for (int i : {0, 1, 2}) triVerts[i] = halfedges[3 * face + i].startVert;

    glm::vec3 edge[3];
    glm::vec3 edgeLength;
    float perimeter = 0;
    for (int i : {0, 1, 2}) {
      const int j = (i + 1) % 3;
      edge[i] = vertPos[triVerts[j]] - vertPos[triVerts[i]];
      edgeLength[i] = glm::length(edge[i]);
      perimeter += edgeLength[i];
    }

    if (calculateTriNormal) {
      triNormal = glm::normalize(glm::cross(edge[0], edge[1]));
      if (isnan(triNormal.x)) triNormal = glm::vec3(0, 0, 1);
    }

    // corner angles
    glm::vec3 phi;
    for (int i : {0, 1, 2}) edge[i] /= edgeLength[i];
    phi[0] = glm::acos(-glm::dot(edge[2], edge[0]));
    phi[1] = glm::acos(-glm::dot(edge[0], edge[1]));
    phi[2] = glm::pi<float>() - phi[0] - phi[1];

    // assign weighted sum
    for (int i : {0, 1, 2}) {
      AtomicAddVec3(vertNormal[triVerts[i]], phi[i] * triNormal);
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

struct SwapHalfedges {
  Halfedge* halfedges;
  const TmpEdge* edges;

  __host__ void operator()(int k) {
    const int i = 2 * k;
    const int j = i - 2;
    const TmpEdge thisEdge = edges[i];
    const TmpEdge lastEdge = edges[j];
    if (thisEdge.first == lastEdge.first &&
        thisEdge.second == lastEdge.second) {
      const int swap0idx = thisEdge.halfedgeIdx;
      Halfedge& swap0 = halfedges[swap0idx];
      const int swap1idx = swap0.pairedHalfedge;
      Halfedge& swap1 = halfedges[swap1idx];

      const int next0idx = swap0idx + ((swap0idx + 1) % 3 == 0 ? -2 : 1);
      const int next1idx = swap1idx + ((swap1idx + 1) % 3 == 0 ? -2 : 1);
      Halfedge& next0 = halfedges[next0idx];
      Halfedge& next1 = halfedges[next1idx];

      next0.startVert = swap0.endVert = next1.endVert;
      swap0.pairedHalfedge = next1.pairedHalfedge;
      halfedges[swap0.pairedHalfedge].pairedHalfedge = swap0idx;

      next1.startVert = swap1.endVert = next0.endVert;
      swap1.pairedHalfedge = next0.pairedHalfedge;
      halfedges[swap1.pairedHalfedge].pairedHalfedge = swap1idx;

      next0.pairedHalfedge = next1idx;
      next1.pairedHalfedge = next0idx;
    }
  }
};

struct InitializeBaryRef {
  __host__ __device__ void operator()(thrust::tuple<BaryRef&, int> inOut) {
    BaryRef& baryRef = thrust::get<0>(inOut);
    int tri = thrust::get<1>(inOut);

    baryRef.tri = tri;
    baryRef.vertBary = {-1, -1, -1};
  }
};

struct MarkShortEdge {
  const glm::vec3* vertPos;
  const float precision;

  __host__ __device__ void operator()(Halfedge& edge) {
    const glm::vec3 delta = vertPos[edge.endVert] - vertPos[edge.startVert];
    if (glm::dot(delta, delta) < precision * precision) edge.face = -1;
  }
};

struct MarkColinearEdge {
  bool* marked;
  const glm::vec3* vertPos;
  const glm::vec3* faceNormal;
  const Halfedge* halfedge;
  const float precision;

  __host__ __device__ void operator()(Halfedge& edge) {
    if (edge.pairedHalfedge < 0) return;
    const glm::vec3 delta = vertPos[edge.endVert] - vertPos[edge.startVert];

    // If all triangles touching startVert are coplanar with the edge, then
    // startVert can be moved to endVert and merged without altering the
    // geometry.
    int start = edge.pairedHalfedge;
    int current = NextHalfedge(start);
    current = halfedge[current].pairedHalfedge;
    while (current != start) {
      current = NextHalfedge(current);
      if (glm::abs(glm::dot(delta, faceNormal[current / 3])) > precision)
        return;
      current = halfedge[current].pairedHalfedge;
    }
    edge.face = -1;
    *marked = true;
  }
};

struct CollapseEdge {
  bool& collapsed;
  VecH<Halfedge>& halfedge;
  VecH<glm::vec3>& vertPos;
  const VecH<glm::vec3>& triNormal;
  const bool shortEdge;

  __host__ glm::ivec3 TriOf(int edge) const {
    glm::ivec3 triEdge;
    triEdge[0] = edge;
    triEdge[1] = edge + ((edge + 1) % 3 == 0 ? -2 : 1);
    triEdge[2] = triEdge[1] + ((triEdge[1] + 1) % 3 == 0 ? -2 : 1);
    return triEdge;
  }

  __host__ void UnmarkEdge(int edge) {
    if (!shortEdge) halfedge[edge].face = edge / 3;
  }

  __host__ bool CCW2Normal(const glm::vec3& a, const glm::vec3& b,
                           const glm::vec3& normal) const {
    return glm::dot(glm::cross(a, b), normal) >= 0;
  }

  // Traverses CW around startEdge.endVert from startEdge to endEdge
  // (edgeEdge.endVert must == startEdge.endVert), updating each edge to point
  // to vert instead.
  __host__ void UpdateVert(int vert, int startEdge, int endEdge) {
    while (startEdge != endEdge) {
      halfedge[startEdge].endVert = vert;
      startEdge = NextHalfedge(startEdge);
      halfedge[startEdge].startVert = vert;
      startEdge = halfedge[startEdge].pairedHalfedge;
    }
  }

  // In the event that the edge collapse would create a non-manifold edge,
  // instead we duplicate the two verts and attach the manifolds the other way
  // across this edge.
  __host__ void FormLoop(int current, int end) {
    int startVert = vertPos.size();
    vertPos.push_back(vertPos[halfedge[current].startVert]);
    int endVert = vertPos.size();
    vertPos.push_back(vertPos[halfedge[current].endVert]);

    int oldMatch = halfedge[current].pairedHalfedge;
    int newMatch = halfedge[end].pairedHalfedge;

    UpdateVert(startVert, oldMatch, newMatch);
    UpdateVert(endVert, end, current);

    halfedge[current].pairedHalfedge = newMatch;
    halfedge[newMatch].pairedHalfedge = current;
    halfedge[end].pairedHalfedge = oldMatch;
    halfedge[oldMatch].pairedHalfedge = end;
  }

  __host__ void CollapseTri(const glm::ivec3& triEdge) {
    int pair1 = halfedge[triEdge[1]].pairedHalfedge;
    int pair2 = halfedge[triEdge[2]].pairedHalfedge;
    halfedge[pair1].pairedHalfedge = pair2;
    halfedge[pair2].pairedHalfedge = pair1;
    for (int i : {0, 1, 2}) {
      halfedge[triEdge[i]] = {-1, -1, -1, -1};
    }
  }

  __host__ void operator()(int edge) {
    const Halfedge toRemove = halfedge[edge];
    if (toRemove.face >= 0 || toRemove.pairedHalfedge < 0) return;

    const glm::ivec3 tri0edge = TriOf(edge);
    const glm::ivec3 tri1edge = TriOf(toRemove.pairedHalfedge);

    if (halfedge[tri0edge[1]].endVert == halfedge[tri1edge[1]].endVert) {
      // Remove disconnected triangles
      for (int i : {0, 1, 2}) {
        vertPos[halfedge[tri0edge[i]].startVert] = glm::vec3(0.0f / 0.0f);
        halfedge[tri0edge[i]] = {-1, -1, -1, -1};
        halfedge[tri1edge[i]] = {-1, -1, -1, -1};
      }
      return;
    }

    std::vector<int> edges;
    int current = halfedge[tri0edge[1]].pairedHalfedge;
    while (current != tri1edge[2]) {
      UnmarkEdge(current);
      current = NextHalfedge(current);
      edges.push_back(current);
      UnmarkEdge(current);
      current = halfedge[current].pairedHalfedge;
    }

    const int endVert = toRemove.endVert;
    int start = halfedge[tri1edge[1]].pairedHalfedge;
    if (!shortEdge) {
      current = start;
      glm::vec3 lastEdge =
          vertPos[halfedge[tri1edge[1]].endVert] - vertPos[endVert];
      while (current != tri0edge[2]) {
        current = NextHalfedge(current);
        glm::vec3 thisEdge =
            vertPos[halfedge[current].endVert] - vertPos[endVert];
        if (!CCW2Normal(thisEdge, lastEdge, triNormal[current / 3])) {
          UnmarkEdge(edge);
          return;
        }
        lastEdge = thisEdge;
        current = halfedge[current].pairedHalfedge;
      }
    }

    // Remove toRemove.startVert and replace with endVert.
    vertPos[toRemove.startVert] = glm::vec3(0.0f / 0.0f);
    CollapseTri(tri1edge);

    current = start;
    while (current != tri0edge[2]) {
      UnmarkEdge(current);
      current = NextHalfedge(current);
      UnmarkEdge(current);
      const int vert = halfedge[current].endVert;
      const int next = halfedge[current].pairedHalfedge;
      for (int i = 0; i < edges.size(); ++i) {
        if (vert == halfedge[edges[i]].endVert) {
          FormLoop(edges[i], current);
          start = next;
          edges.resize(i);
          break;
        }
      }
      current = next;
    }

    UpdateVert(endVert, start, tri0edge[2]);
    CollapseTri(tri0edge);
    collapsed = true;
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

  __host__ __device__ bool operator()(int edge) {
    bool good = true;
    const Halfedge halfedge = halfedges[edge];
    const Halfedge paired = halfedges[halfedge.pairedHalfedge];
    good &= paired.pairedHalfedge == edge;
    good &= halfedge.startVert != halfedge.endVert;
    good &= halfedge.startVert == paired.endVert;
    good &= halfedge.endVert == paired.startVert;
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

struct CheckCCW {
  const Halfedge* halfedges;
  const glm::vec3* vertPos;
  const glm::vec3* triNormal;
  const float precision;

  __host__ __device__ bool operator()(int face) {
    const glm::mat3x2 projection = GetAxisAlignedProjection(triNormal[face]);
    glm::vec2 v[3];
    for (int i : {0, 1, 2})
      v[i] = projection * vertPos[halfedges[3 * face + i].startVert];
    int ccw = CCW(v[0], v[1], v[2], 2 * precision);
    if (ccw < 0) {
      glm::vec2 v1 = v[1] - v[0];
      glm::vec2 v2 = v[2] - v[0];
      float area = v1.x * v2.y - v1.y * v2.x;
      float base = glm::sqrt(glm::max(glm::dot(v1, v1), glm::dot(v2, v2)));
      printf(
          "Tri %d does not match normal, height = %g, base = %g\n"
          "normal = %g, %g, %g\n",
          face, area / base, base, triNormal[face].x, triNormal[face].y,
          triNormal[face].z);
    }
    return ccw >= 0;
  }
};

}  // namespace

namespace manifold {

/**
 * Create a manifold from an input triangle Mesh. Will throw if the Mesh is not
 * manifold. TODO: update halfedgeTangent during CollapseDegenerates.
 */
Manifold::Impl::Impl(const Mesh& mesh)
    : vertPos_(mesh.vertPos), halfedgeTangent_(mesh.halfedgeTangent) {
  CheckDevice();
  CalculateBBox();
  SetPrecision();
  CreateAndFixHalfedges(mesh.triVerts);
  CalculateNormals();
  CollapseDegenerates();
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
      throw userErr("Unrecognized shape!");
  }
  vertPos_ = vertPos;
  CreateAndFixHalfedges(triVerts);
  Finish();
}

/**
 * Create the halfedge_ data structure from an input triVerts array like Mesh.
 */
void Manifold::Impl::CreateHalfedges(const VecDH<glm::ivec3>& triVerts) {
  const int numTri = triVerts.size();
  halfedge_.resize(3 * numTri);
  VecDH<TmpEdge> edge(3 * numTri);
  thrust::for_each_n(zip(countAt(0), triVerts.beginD()), numTri,
                     Tri2Halfedges({halfedge_.ptrD(), edge.ptrD()}));
  thrust::sort(edge.beginD(), edge.endD());
  thrust::for_each_n(countAt(0), halfedge_.size() / 2,
                     LinkHalfedges({halfedge_.ptrD(), edge.cptrD()}));
  if (meshRelation_.triBary.size() != numTri) {
    meshRelation_.triBary.resize(numTri);
    thrust::for_each_n(zip(meshRelation_.triBary.beginD(), countAt(0)), numTri,
                       InitializeBaryRef());
  }
}

/**
 * Create the halfedge_ data structure from an input triVerts array like Mesh.
 * Check that the input is an even-manifold, and if it is not 2-manifold,
 * perform edge swaps until it is. This is a host function.
 */
void Manifold::Impl::CreateAndFixHalfedges(const VecDH<glm::ivec3>& triVerts) {
  const int numTri = triVerts.size();
  halfedge_.resize(3 * numTri);
  VecDH<TmpEdge> edge(3 * numTri);
  thrust::for_each_n(zip(countAt(0), triVerts.begin()), numTri,
                     Tri2Halfedges({halfedge_.ptrH(), edge.ptrH()}));
  // Stable sort is required here so that halfedges from the same face are
  // paired together (the triangles were created in face order). In some
  // degenerate situations the triangulator can add the same internal edge in
  // two different faces, causing this edge to not be 2-manifold. We detect this
  // and fix it by swapping one of the identical edges, so it is important that
  // we have the edges paired according to their face.
  std::stable_sort(edge.begin(), edge.end());
  thrust::for_each_n(thrust::host, countAt(0), halfedge_.size() / 2,
                     LinkHalfedges({halfedge_.ptrH(), edge.cptrH()}));
  thrust::for_each(thrust::host, countAt(1), countAt(halfedge_.size() / 2),
                   SwapHalfedges({halfedge_.ptrH(), edge.cptrH()}));
  meshRelation_.triBary.resize(numTri);
  thrust::for_each_n(zip(meshRelation_.triBary.begin(), countAt(0)), numTri,
                     InitializeBaryRef());
}

/**
 * Collapses degenerate triangles by removing edges shorter than precision_ and
 * edges that are colinear whose collapse does not generate a geometric change.
 * Rather than actually removing them, this step merely marks them for removal,
 * by setting vertPos to NaN and halfedge to -1.
 *
 * TODO: remove colinear edge collapse, since this could easily conflict with
 * mesh property junctions; consider test Boolean.FaceUnion if the two cubes
 * were different colors - we are losing the verts that denote the boundary.
 */
void Manifold::Impl::CollapseDegenerates() {
  // Short edge collapse is commented out because it was causing a test to fail,
  // but only on the GPU. Colinear removal is broader, so it should still
  // accomplish the same thing, but with some additional computation. I'm
  // leaving this here because I'd like to understand why it gets the wrong
  // result and why it's not any faster.

  // thrust::for_each(halfedge_.beginD(), halfedge_.endD(),
  //                  MarkShortEdge({vertPos_.cptrD(), precision_}));
  bool collapsed = false;
  // thrust::for_each_n(thrust::host, countAt(0), halfedge_.size(),
  //                    CollapseEdge({collapsed, halfedge_.H(), vertPos_.H(),
  //                                  faceNormal_.H(), true}));

  VecDH<bool> marked(1);
  while (1) {
    marked.H()[0] = false;
    thrust::for_each(
        halfedge_.beginD(), halfedge_.endD(),
        MarkColinearEdge({marked.ptrD(), vertPos_.cptrD(), faceNormal_.cptrD(),
                          halfedge_.cptrD(), precision_}));
    if (!marked.H()[0]) break;

    collapsed = false;
    thrust::for_each_n(thrust::host, countAt(0), halfedge_.size(),
                       CollapseEdge({collapsed, halfedge_.H(), vertPos_.H(),
                                     faceNormal_.H(), false}));
    if (!collapsed) break;
  }
}

/**
 * Once halfedge_ has been filled in, this function can be called to create the
 * rest of the internal data structures. This function also removes the verts
 * and halfedges flagged for removal (NaN verts and -1 halfedges).
 */
void Manifold::Impl::Finish() {
  if (halfedge_.size() == 0) return;

  CalculateBBox();
  SetPrecision(precision_);
  if (!bBox_.isFinite()) {
    vertPos_.resize(0);
    halfedge_.resize(0);
    faceNormal_.resize(0);
    return;
  }

  SortVerts();
  VecDH<Box> faceBox;
  VecDH<uint32_t> faceMorton;
  GetFaceBoxMorton(faceBox, faceMorton);
  SortFaces(faceBox, faceMorton);
  if (halfedge_.size() == 0) return;

  ALWAYS_ASSERT(halfedge_.size() % 6 == 0, topologyErr,
                "Not an even number of faces after sorting faces!");
  Halfedge extrema = {0, 0, 0, 0};
  extrema =
      thrust::reduce(halfedge_.beginD(), halfedge_.endD(), extrema, Extrema());

  ALWAYS_ASSERT(extrema.startVert >= 0, topologyErr,
                "Vertex index is negative!");
  ALWAYS_ASSERT(extrema.endVert < NumVert(), topologyErr,
                "Vertex index exceeds number of verts!");
  ALWAYS_ASSERT(extrema.face >= 0, topologyErr, "Face index is negative!");
  ALWAYS_ASSERT(extrema.face < NumTri(), topologyErr,
                "Face index exceeds number of faces!");
  ALWAYS_ASSERT(extrema.pairedHalfedge >= 0, topologyErr,
                "Halfedge index is negative!");
  ALWAYS_ASSERT(extrema.pairedHalfedge < 2 * NumEdge(), topologyErr,
                "Halfedge index exceeds number of halfedges!");

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
  thrust::for_each(vertPos_.beginD(), vertPos_.endD(),
                   Transform4x3({transform_}));

  glm::mat3 normalTransform =
      glm::inverse(glm::transpose(glm::mat3(transform_)));
  thrust::for_each(faceNormal_.beginD(), faceNormal_.endD(),
                   TransformNormals({normalTransform}));
  thrust::for_each(vertNormal_.beginD(), vertNormal_.endD(),
                   TransformNormals({normalTransform}));
  // This optimization does a cheap collider update if the transform is
  // axis-aligned.
  if (!collider_.Transform(transform_)) Update();

  const float oldScale = bBox_.Scale();
  transform_ = glm::mat4x3(1.0f);
  CalculateBBox();

  const float newScale = bBox_.Scale();
  precision_ *= glm::max(1.0f, newScale / oldScale) *
                glm::max(glm::length(transform_[0]),
                         glm::max(glm::length(transform_[1]),
                                  glm::length(transform_[2])));

  // Maximum of inherited precision loss and translational precision loss.
  SetPrecision(precision_);
}

/**
 * Triangulates the faces. In this case, the halfedge_ vector is not yet a set
 * of triangles as required by this data structure, but is instead a set of
 * general faces with the input faceEdge vector having length of the number of
 * faces + 1. The values are indicies into the halfedge_ vector for the first
 * edge of each face, with the final value being the length of the halfedge_
 * vector itself. Upon return, halfedge_ has been lengthened and properly
 * represents the mesh as a set of triangles as usual. In this process the
 * faceNormal_ values are retained, repeated as necessary.
 */
void Manifold::Impl::Face2Tri(const VecDH<int>& faceEdge) {
  VecDH<glm::ivec3> triVertsOut;
  VecDH<glm::vec3> triNormalOut;

  VecH<glm::ivec3>& triVerts = triVertsOut.H();
  VecH<glm::vec3>& triNormal = triNormalOut.H();
  const VecH<glm::vec3>& vertPos = vertPos_.H();
  const VecH<int>& face = faceEdge.H();
  const VecH<Halfedge>& halfedge = halfedge_.H();
  const VecH<glm::vec3>& faceNormal = faceNormal_.H();

  for (int i = 0; i < face.size() - 1; ++i) {
    const int edge = face[i];
    const int lastEdge = face[i + 1];
    const int numEdge = lastEdge - edge;
    ALWAYS_ASSERT(numEdge >= 3, topologyErr, "face has less than three edges.");
    const glm::vec3 normal = faceNormal[i];

    if (numEdge == 3) {  // Single triangle
      glm::ivec3 tri(halfedge[edge].startVert, halfedge[edge + 1].startVert,
                     halfedge[edge + 2].startVert);
      glm::ivec3 ends(halfedge[edge].endVert, halfedge[edge + 1].endVert,
                      halfedge[edge + 2].endVert);
      if (ends[0] == tri[2]) {
        std::swap(tri[1], tri[2]);
        std::swap(ends[1], ends[2]);
      }
      ALWAYS_ASSERT(ends[0] == tri[1] && ends[1] == tri[2] && ends[2] == tri[0],
                    topologyErr, "These 3 edges do not form a triangle!");

      triVerts.push_back(tri);
      triNormal.push_back(normal);
    } else if (numEdge == 4) {  // Pair of triangles
      const glm::mat3x2 projection = GetAxisAlignedProjection(normal);
      auto triCCW = [&projection, &vertPos, this](const glm::ivec3 tri) {
        return CCW(projection * vertPos[tri[0]], projection * vertPos[tri[1]],
                   projection * vertPos[tri[2]], precision_) >= 0;
      };

      glm::ivec3 tri0(halfedge[edge].startVert, halfedge[edge].endVert, -1);
      glm::ivec3 tri1(-1, -1, tri0[0]);
      for (const int i : {1, 2, 3}) {
        if (halfedge[edge + i].startVert == tri0[1]) {
          tri0[2] = halfedge[edge + i].endVert;
          tri1[0] = tri0[2];
        }
        if (halfedge[edge + i].endVert == tri0[0]) {
          tri1[1] = halfedge[edge + i].startVert;
        }
      }
      ALWAYS_ASSERT(glm::all(glm::greaterThanEqual(tri0, glm::ivec3(0))) &&
                        glm::all(glm::greaterThanEqual(tri1, glm::ivec3(0))),
                    topologyErr, "non-manifold quad!");
      bool firstValid = triCCW(tri0) && triCCW(tri1);
      tri0[2] = tri1[1];
      tri1[2] = tri0[1];
      bool secondValid = triCCW(tri0) && triCCW(tri1);

      if (!secondValid) {
        tri0[2] = tri1[0];
        tri1[2] = tri0[0];
      } else if (firstValid) {
        glm::vec3 firstCross = vertPos[tri0[0]] - vertPos[tri1[0]];
        glm::vec3 secondCross = vertPos[tri0[1]] - vertPos[tri1[1]];
        if (glm::dot(firstCross, firstCross) <
            glm::dot(secondCross, secondCross)) {
          tri0[2] = tri1[0];
          tri1[2] = tri0[0];
        }
      }

      triVerts.push_back(tri0);
      triNormal.push_back(normal);
      triVerts.push_back(tri1);
      triNormal.push_back(normal);
    } else {  // General triangulation
      const glm::mat3x2 projection = GetAxisAlignedProjection(normal);

      Polygons polys;
      try {
        polys = Face2Polygons(i, projection, face);
      } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        for (int edge = face[i]; edge < face[i + 1]; ++edge)
          std::cout << "halfedge: " << edge << ", " << halfedge[edge]
                    << std::endl;
        throw;
      }

      std::vector<glm::ivec3> newTris = Triangulate(polys, precision_);

      for (auto tri : newTris) {
        triVerts.push_back(tri);
        triNormal.push_back(normal);
      }
    }
  }
  faceNormal_ = triNormalOut;
  CreateAndFixHalfedges(triVertsOut);
}

void Manifold::Impl::CreateTangents(
    const std::vector<Smoothness>& sharpenedEdges) {
  const int numHalfedge = halfedge_.size();
  halfedgeTangent_.resize(numHalfedge);

  thrust::for_each_n(zip(halfedgeTangent_.beginD(), halfedge_.cbeginD()),
                     numHalfedge,
                     SmoothBezier({vertPos_.cptrD(), faceNormal_.cptrD(),
                                   vertNormal_.cptrD(), halfedge_.cptrD()}));

  if (!sharpenedEdges.empty()) {
    const VecH<Halfedge>& halfedge = halfedge_.H();
    const VecH<BaryRef>& triBary = meshRelation_.triBary.H();

    std::vector<int> oldHalfedge2New(halfedge.size());
    for (int tri = 0; tri < NumTri(); ++tri) {
      int oldTri = triBary[tri].tri;
      for (int i : {0, 1, 2}) oldHalfedge2New[3 * oldTri + i] = 3 * tri + i;
    }

    using Pair = std::pair<Smoothness, Smoothness>;
    // Fill in missing pairs with default smoothness = 1.
    std::map<int, Pair> edges;
    for (Smoothness edge : sharpenedEdges) {
      if (edge.smoothness == 1) continue;
      edge.halfedge = oldHalfedge2New[edge.halfedge];
      int pair = halfedge[edge.halfedge].pairedHalfedge;
      if (edges.find(pair) == edges.end()) {
        edges[edge.halfedge] = {edge, {pair, 1}};
      } else {
        edges[pair].second = edge;
      }
    }

    std::map<int, std::vector<Pair>> vertTangents;
    for (const auto value : edges) {
      const Pair edge = value.second;
      vertTangents[halfedge[edge.first.halfedge].startVert].push_back(edge);
      vertTangents[halfedge[edge.second.halfedge].startVert].push_back(
          {edge.second, edge.first});
    }

    VecH<glm::vec4>& tangent = halfedgeTangent_.H();
    for (const auto& value : vertTangents) {
      const std::vector<Pair>& vert = value.second;
      // Sharp edges that end are smooth at their terminal vert.
      if (vert.size() == 1) continue;
      if (vert.size() == 2) {  // Make continuous edge
        const int first = vert[0].first.halfedge;
        const int second = vert[1].first.halfedge;
        const glm::vec3 newTangent = glm::normalize(glm::vec3(tangent[first]) -
                                                    glm::vec3(tangent[second]));
        tangent[first] =
            glm::vec4(glm::length(glm::vec3(tangent[first])) * newTangent,
                      tangent[first].w);
        tangent[second] =
            glm::vec4(-glm::length(glm::vec3(tangent[second])) * newTangent,
                      tangent[second].w);

        auto SmoothHalf = [&](int first, int last, float smoothness) {
          int current = NextHalfedge(halfedge[first].pairedHalfedge);
          while (current != last) {
            const float cosBeta = glm::dot(
                newTangent, glm::normalize(glm::vec3(tangent[current])));
            const float factor =
                (1 - smoothness) * cosBeta * cosBeta + smoothness;
            tangent[current] = glm::vec4(factor * glm::vec3(tangent[current]),
                                         tangent[current].w);
            current = NextHalfedge(halfedge[current].pairedHalfedge);
          }
        };

        SmoothHalf(first, second,
                   (vert[0].second.smoothness + vert[1].first.smoothness) / 2);
        SmoothHalf(second, first,
                   (vert[1].second.smoothness + vert[0].first.smoothness) / 2);

      } else {  // Sharpen vertex uniformly
        float smoothness = 0;
        for (const Pair pair : vert) {
          smoothness += pair.first.smoothness;
          smoothness += pair.second.smoothness;
        }
        smoothness /= 2 * vert.size();

        const int start = vert[0].first.halfedge;
        int current = start;
        do {
          tangent[current] = glm::vec4(smoothness * glm::vec3(tangent[current]),
                                       tangent[current].w);
          current = NextHalfedge(halfedge[current].pairedHalfedge);
        } while (current != start);
      }
    }
  }
}

/**
 * Split each edge into n pieces and sub-triangulate each triangle accordingly.
 * This function doesn't run Finish(), as that is expensive and it'll need to be
 * run after the new vertices have moved, which is a likely scenario after
 * refinement (smoothing).
 */
void Manifold::Impl::Subdivide(int n) {
  int numVert = NumVert();
  int numEdge = NumEdge();
  int numTri = NumTri();
  // Append new verts
  int vertsPerEdge = n - 1;
  int triVertStart = numVert + numEdge * vertsPerEdge;
  vertPos_.resize(triVertStart + numTri * VertsPerTri(n - 2));
  meshRelation_.barycentric.resize(numTri * VertsPerTri(n + 1));
  meshRelation_.triBary.resize(n * n * numTri);
  VecDH<TmpEdge> edges = CreateTmpEdges(halfedge_);
  VecDH<int> half2Edge(2 * numEdge);
  thrust::for_each_n(zip(countAt(0), edges.beginD()), numEdge,
                     ReindexHalfedge({half2Edge.ptrD()}));
  thrust::for_each_n(zip(countAt(0), edges.beginD()), numEdge,
                     EdgeVerts({vertPos_.ptrD(), numVert, n}));
  thrust::for_each_n(
      countAt(0), numTri,
      InteriorVerts({vertPos_.ptrD(), meshRelation_.barycentric.ptrD(),
                     meshRelation_.triBary.ptrD(), triVertStart, n,
                     halfedge_.ptrD()}));
  // Create subtriangles
  VecDH<glm::ivec3> triVerts(n * n * numTri);
  thrust::for_each_n(countAt(0), numTri,
                     SplitTris({triVerts.ptrD(), halfedge_.cptrD(),
                                half2Edge.cptrD(), numVert, triVertStart, n}));
  CreateHalfedges(triVerts);
}

void Manifold::Impl::Refine(int n) {
  Manifold::Impl old = *this;
  Subdivide(n);

  if (old.halfedgeTangent_.size() == old.halfedge_.size()) {
    VecDH<Barycentric> vertBary(NumVert());
    VecDH<int> lock(NumVert(), 0);
    thrust::for_each_n(
        zip(meshRelation_.triBary.beginD(), countAt(0)), NumTri(),
        TriBary2Vert({vertBary.ptrD(), lock.ptrD(),
                      meshRelation_.barycentric.cptrD(), halfedge_.cptrD()}));

    thrust::for_each_n(
        zip(vertPos_.beginD(), vertBary.beginD()), NumVert(),
        InterpTri({old.halfedge_.cptrD(), old.halfedgeTangent_.cptrD(),
                   old.vertPos_.cptrD()}));
  }

  halfedgeTangent_.resize(0);
  Finish();
}

/**
 * Returns true if this manifold is in fact an oriented 2-manifold and all of
 * the data structures are consistent.
 */
bool Manifold::Impl::IsManifold() const {
  if (halfedge_.size() == 0) return true;
  bool isManifold = thrust::all_of(countAt(0), countAt(halfedge_.size()),
                                   CheckManifold({halfedge_.cptrD()}));
  if (!isManifold) std::cout << "not manifold!" << std::endl;
  VecDH<Halfedge> halfedge(halfedge_);
  thrust::sort(halfedge.beginD(), halfedge.endD());
  isManifold &= thrust::all_of(countAt(0), countAt(2 * NumEdge() - 1),
                               NoDuplicates({halfedge.cptrD()}));
  if (!isManifold) std::cout << "not 2-manifold!" << std::endl;
  return isManifold;
}

/**
 * Returns true if all triangles are CCW relative to their triNormals_.
 */
bool Manifold::Impl::MatchesTriNormals() const {
  if (halfedge_.size() == 0 || faceNormal_.size() != NumTri()) return true;
  return thrust::all_of(thrust::device, countAt(0), countAt(NumTri()),
                        CheckCCW({halfedge_.cptrD(), vertPos_.cptrD(),
                                  faceNormal_.cptrD(), precision_}));
}

/**
 * Returns the surface area and volume of the manifold in a Properties
 * structure. These properties are clamped to zero for a given face if they are
 * within rounding tolerance. This means degenerate manifolds can by identified
 * by testing these properties as == 0.
 */
Properties Manifold::Impl::GetProperties() const {
  if (halfedge_.size() == 0) return {0, 0};
  ApplyTransform();
  thrust::pair<float, float> areaVolume = thrust::transform_reduce(
      countAt(0), countAt(NumTri()),
      FaceAreaVolume({halfedge_.cptrD(), vertPos_.cptrD(), precision_}),
      thrust::make_pair(0.0f, 0.0f), SumPair());
  return {areaVolume.first, areaVolume.second};
}

/**
 * Calculates the bounding box of the entire manifold, which is stored
 * internally to short-cut Boolean operations and to serve as the precision
 * range for Morton code calculation.
 */
void Manifold::Impl::CalculateBBox() {
  bBox_.min = thrust::reduce(vertPos_.beginD(), vertPos_.endD(),
                             glm::vec3(1 / 0.0f), PosMin());
  bBox_.max = thrust::reduce(vertPos_.beginD(), vertPos_.endD(),
                             glm::vec3(-1 / 0.0f), PosMax());
}

/**
 * Sets the precision based on the bounding box, and limits its minimum value by
 * the optional input.
 */
void Manifold::Impl::SetPrecision(float minPrecision) {
  precision_ = glm::max(minPrecision, kTolerance * bBox_.Scale());
  if (!glm::isfinite(precision_)) precision_ = -1;
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
  thrust::sort_by_key(vertMorton.beginD(), vertMorton.endD(),
                      zip(vertPos_.beginD(), vertNew2Old.beginD()));

  ReindexVerts(vertNew2Old, NumVert());

  // Verts were flagged for removal with NaNs and assigned kNoCode to sort them
  // to the end, which allows them to be removed.
  const int newNumVert =
      thrust::find(vertMorton.beginD(), vertMorton.endD(), kNoCode) -
      vertMorton.beginD();
  vertPos_.resize(newNumVert);
}

/**
 * Updates the halfedges to point to new vert indices based on a mapping,
 * vertNew2Old. This may be a subset, so the total number of original verts is
 * also given.
 */
void Manifold::Impl::ReindexVerts(const VecDH<int>& vertNew2Old,
                                  int oldNumVert) {
  VecDH<int> vertOld2New(oldNumVert);
  thrust::scatter(countAt(0), countAt(NumVert()), vertNew2Old.beginD(),
                  vertOld2New.beginD());
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
  faceBox.resize(NumTri());
  faceMorton.resize(NumTri());
  thrust::for_each_n(
      zip(faceMorton.beginD(), faceBox.beginD(), countAt(0)), NumTri(),
      FaceMortonBox({halfedge_.cptrD(), vertPos_.cptrD(), bBox_}));
}

/**
 * Sorts the faces of this manifold according to their input Morton code. The
 * bounding box and Morton code arrays are also sorted accordingly.
 */
void Manifold::Impl::SortFaces(VecDH<Box>& faceBox,
                               VecDH<uint32_t>& faceMorton) {
  VecDH<int> faceNew2Old(NumTri());
  thrust::sequence(faceNew2Old.beginD(), faceNew2Old.endD());

  thrust::sort_by_key(faceMorton.beginD(), faceMorton.endD(),
                      zip(faceBox.beginD(), faceNew2Old.beginD()));

  // Tris were flagged for removal with pairedHalfedge = -1 and assigned kNoCode
  // to sort them to the end, which allows them to be removed.
  const int newNumTri =
      thrust::find(faceMorton.beginD(), faceMorton.endD(), kNoCode) -
      faceMorton.beginD();
  faceBox.resize(newNumTri);
  faceMorton.resize(newNumTri);
  faceNew2Old.resize(newNumTri);

  GatherFaces(faceNew2Old);
}

/**
 * Creates the halfedge_ vector for this manifold by copying a set of faces from
 * another manifold, given by oldHalfedge. Input faceNew2Old defines the old
 * faces to gather into this.
 */
void Manifold::Impl::GatherFaces(const VecDH<int>& faceNew2Old) {
  const int numTri = faceNew2Old.size();
  Permute(meshRelation_.triBary, faceNew2Old);

  if (faceNormal_.size() == NumTri()) Permute(faceNormal_, faceNew2Old);

  VecDH<Halfedge> oldHalfedge(halfedge_);
  VecDH<glm::vec4> oldHalfedgeTangent(halfedgeTangent_);
  VecDH<int> faceOld2New(oldHalfedge.size() / 3);
  thrust::scatter(countAt(0), countAt(numTri), faceNew2Old.beginD(),
                  faceOld2New.beginD());

  halfedge_.resize(3 * numTri);
  if (oldHalfedgeTangent.size() != 0) halfedgeTangent_.resize(3 * numTri);
  thrust::for_each_n(
      countAt(0), numTri,
      ReindexFace({halfedge_.ptrD(), halfedgeTangent_.ptrD(),
                   oldHalfedge.cptrD(), oldHalfedgeTangent.cptrD(),
                   faceNew2Old.cptrD(), faceOld2New.cptrD()}));
}

void Manifold::Impl::GatherFaces(const Impl& old,
                                 const VecDH<int>& faceNew2Old) {
  const int numTri = faceNew2Old.size();
  meshRelation_.triBary.resize(numTri);
  thrust::gather(faceNew2Old.beginD(), faceNew2Old.endD(),
                 old.meshRelation_.triBary.beginD(),
                 meshRelation_.triBary.beginD());

  if (old.faceNormal_.size() == old.NumTri()) {
    faceNormal_.resize(numTri);
    thrust::gather(faceNew2Old.beginD(), faceNew2Old.endD(),
                   old.faceNormal_.beginD(), faceNormal_.beginD());
  }

  VecDH<int> faceOld2New(old.NumTri());
  thrust::scatter(countAt(0), countAt(numTri), faceNew2Old.beginD(),
                  faceOld2New.beginD());

  halfedge_.resize(3 * numTri);
  if (old.halfedgeTangent_.size() != 0) halfedgeTangent_.resize(3 * numTri);
  thrust::for_each_n(
      countAt(0), numTri,
      ReindexFace({halfedge_.ptrD(), halfedgeTangent_.ptrD(),
                   old.halfedge_.cptrD(), old.halfedgeTangent_.cptrD(),
                   faceNew2Old.cptrD(), faceOld2New.cptrD()}));
}

/**
 * If face normals are already present, this function uses them to compute
 * vertex normals (angle-weighted pseudo-normals); otherwise it also computes
 * the face normals. Face normals are only calculated when needed because nearly
 * degenerate faces will accrue rounding error, while the Boolean can retain
 * their original normal, which is more accurate and can help with merging
 * coplanar faces.
 *
 * If the face normals have been invalidated by an operation like Warp(), ensure
 * you do faceNormal_.resize(0) before calling this function to force
 * recalculation.
 */
void Manifold::Impl::CalculateNormals() {
  vertNormal_.resize(NumVert());
  thrust::fill(vertNormal_.beginD(), vertNormal_.endD(), glm::vec3(0));
  bool calculateTriNormal = false;
  if (faceNormal_.size() != NumTri()) {
    faceNormal_.resize(NumTri());
    calculateTriNormal = true;
  }
  thrust::for_each_n(
      zip(faceNormal_.beginD(), countAt(0)), NumTri(),
      AssignNormals({vertNormal_.ptrD(), vertPos_.cptrD(), halfedge_.cptrD(),
                     precision_, calculateTriNormal}));
  thrust::for_each(vertNormal_.beginD(), vertNormal_.endD(), Normalize());
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
                                       const VecH<int>& faceEdge) const {
  const VecH<glm::vec3>& vertPos = vertPos_.H();
  const VecH<Halfedge>& halfedge = halfedge_.H();
  const int firstEdge = faceEdge[face];
  const int lastEdge = faceEdge[face + 1];

  std::map<int, int> vert_edge;
  for (int edge = firstEdge; edge < lastEdge; ++edge) {
    ALWAYS_ASSERT(
        vert_edge.emplace(std::make_pair(halfedge[edge].startVert, edge))
            .second,
        topologyErr, "face has duplicate vertices.");
  }

  Polygons polys;
  int startEdge = 0;
  int thisEdge = startEdge;
  while (1) {
    if (thisEdge == startEdge) {
      if (vert_edge.empty()) break;
      startEdge = vert_edge.begin()->second;
      thisEdge = startEdge;
      polys.push_back({});
    }
    int vert = halfedge[thisEdge].startVert;
    polys.back().push_back({projection * vertPos[vert], vert});
    const auto result = vert_edge.find(halfedge[thisEdge].endVert);
    ALWAYS_ASSERT(result != vert_edge.end(), topologyErr, "nonmanifold edge");
    thisEdge = result->second;
    vert_edge.erase(result);
  }
  return polys;
}
}  // namespace manifold