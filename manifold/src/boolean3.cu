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

#include "boolean3.cuh"
#include "connected_components.cuh"
#include "polygon.h"

#include <math_constants.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>
#include <algorithm>

constexpr bool kVerbose = false;

using namespace thrust::placeholders;

namespace {
using namespace manifold;

// These two functions (Interpolate and Intersect) are the only places where
// floating-point operations take place in the whole Boolean function. These are
// carefully designed to minimize rounding error and to eliminate it at edge
// cases to ensure consistency.

__host__ __device__ glm::vec2 Interpolate(glm::vec3 pL, glm::vec3 pR, float x) {
  float dxL = x - pL.x;
  float dxR = x - pR.x;
  // assert dxL and dxR have opposite signs, cannot both be zero
  // if (dxL * dxR > 0 || (dxL == 0 && dxR == 0))
  //   printf("dxL = %f, dxR = %f\n", dxL, dxR);
  bool useL = fabs(dxL) < fabs(dxR);
  float lambda = (useL ? dxL : dxR) / (pR.x - pL.x);
  glm::vec2 yz;
  yz[0] = (useL ? pL.y : pR.y) + lambda * (pR.y - pL.y);
  yz[1] = (useL ? pL.z : pR.z) + lambda * (pR.z - pL.z);
  return yz;
}

__host__ __device__ glm::vec4 Intersect(const glm::vec3 &pL,
                                        const glm::vec3 &pR,
                                        const glm::vec3 &qL,
                                        const glm::vec3 &qR) {
  // assert pL.x == qL.x, pR.x == qR.x
  // if (pL.x != qL.x || pR.x != qR.x)
  //   printf("pL.x = %f, qL.x = %f, pR.x = %f, qR.x = %f\n", pL.x, qL.x, pR.x,
  //          qR.x);
  float dyL = qL.y - pL.y;
  float dyR = qR.y - pR.y;
  // assert dyL and dyR have opposite signs, cannot both be zero
  // if (dyL * dyR > 0 || (dyL == 0 && dyR == 0))
  //   printf("dyL = %f, dyR = %f\n", dyL, dyR);
  bool useL = fabs(dyL) < fabs(dyR);
  float dx = pR.x - pL.x;
  float lambda = (useL ? dyL : dyR) / (dyL - dyR);
  glm::vec4 xyzz;
  xyzz.x = (useL ? pL.x : pR.x) + lambda * dx;
  float pDy = pR.y - pL.y;
  float qDy = qR.y - qL.y;
  bool useP = fabs(pDy) < fabs(qDy);
  xyzz.y = (useL ? (useP ? pL.y : qL.y) : (useP ? pR.y : qR.y)) +
           lambda * (useP ? pDy : qDy);
  xyzz.z = (useL ? pL.z : pR.z) + lambda * (pR.z - pL.z);
  xyzz.w = (useL ? qL.z : qR.z) + lambda * (qR.z - qL.z);
  return xyzz;
}

struct CopyEdgeVerts {
  int *verts;
  const EdgeVertsD *edgeVerts;

  __host__ __device__ void operator()(thrust::tuple<int, int> in) {
    int idx = 2 * thrust::get<0>(in);
    int edge = thrust::get<1>(in);

    verts[idx] = edgeVerts[edge].first;
    verts[idx + 1] = edgeVerts[edge].second;
  }
};

struct CopyTriVerts {
  int *verts;
  const glm::ivec3 *triVerts;

  __host__ __device__ void operator()(thrust::tuple<int, int> in) {
    int idx = 3 * thrust::get<0>(in);
    int tri = thrust::get<1>(in);

    for (int i : {0, 1, 2}) verts[idx + i] = triVerts[tri][i];
  }
};

struct CopyTriEdges {
  int *edges;
  const TriEdges *triEdges;

  __host__ __device__ void operator()(thrust::tuple<int, int> in) {
    int idx = 3 * thrust::get<0>(in);
    int tri = thrust::get<1>(in);

    for (int i : {0, 1, 2}) {
      edges[idx + i] = triEdges[tri][i].Idx();
    }
  }
};

SparseIndices Filter02(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                       const VecDH<int> &edges, const VecDH<int> &tris) {
  // find one vertex from each connected component of inP (in case it has no
  // intersections)
  VecDH<int> vLabels;
  int n_comp = ConnectedComponents(vLabels, inP.NumVert(), inP.edgeVerts_);
  thrust::sort(vLabels.beginD(), vLabels.endD());
  thrust::unique(vLabels.beginD(), vLabels.endD());
  // find inP's involved vertices from edges & tris
  VecDH<int> A0(2 * edges.size() + 3 * tris.size() + n_comp);
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), edges.beginD()),
                     edges.size(),
                     CopyEdgeVerts({A0.ptrD(), inP.edgeVerts_.cptrD()}));
  thrust::for_each_n(
      zip(thrust::make_counting_iterator(0), tris.beginD()), tris.size(),
      CopyTriVerts({A0.ptrD() + 2 * edges.size(), inP.triVerts_.cptrD()}));
  thrust::copy(vLabels.beginD(), vLabels.beginD() + n_comp, A0.endD() - n_comp);
  thrust::sort(A0.beginD(), A0.endD());
  A0.resize(thrust::unique(A0.beginD(), A0.endD()) - A0.beginD());
  // find which inQ faces shadow these vertices
  VecDH<glm::vec3> AV(A0.size());
  thrust::gather(A0.beginD(), A0.endD(), inP.vertPos_.cbeginD(), AV.beginD());
  SparseIndices p0q2 = inQ.VertexCollisionsZ(AV);
  VecDH<int> i02temp(p0q2.size());
  thrust::copy(p0q2.beginD(0), p0q2.endD(0), i02temp.beginD());
  thrust::gather(i02temp.beginD(), i02temp.endD(), A0.beginD(), p0q2.beginD(0));
  return p0q2;
}

struct Fill11 {
  thrust::pair<int *, int *> p1q1;
  const TriEdges *triEdgesQ;

  __host__ __device__ void operator()(thrust::tuple<int, int, int> in) {
    int idx = 3 * thrust::get<0>(in);
    int edgeP = thrust::get<1>(in);
    int triQ = thrust::get<2>(in);

    for (int i : {0, 1, 2}) {
      p1q1.first[idx + i] = edgeP;
      p1q1.second[idx + i] = triEdgesQ[triQ][i].Idx();
    }
  }
};

SparseIndices Filter11(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                       const SparseIndices &p1q2, const SparseIndices &p2q1) {
  SparseIndices p1q1(3 * p1q2.size() + 3 * p2q1.size());
  thrust::for_each_n(
      zip(thrust::make_counting_iterator(0), p1q2.beginD(0), p1q2.beginD(1)),
      p1q2.size(), Fill11({p1q1.ptrDpq(), inQ.triEdges_.cptrD()}));
  p1q1.Swap();
  thrust::for_each_n(zip(thrust::make_counting_iterator(p1q2.size()),
                         p2q1.beginD(1), p2q1.beginD(0)),
                     p2q1.size(),
                     Fill11({p1q1.ptrDpq(), inP.triEdges_.cptrD()}));
  p1q1.Swap();
  p1q1.Unique();
  return p1q1;
}

SparseIndices Filter01(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                       const SparseIndices &p0q2, const SparseIndices &p1q1) {
  SparseIndices p0q1(3 * p0q2.size() + 2 * p1q1.size());
  // Copy vertices
  for (int i : {0, 1, 2}) {
    auto verts3 =
        strided_range<VecDH<int>::IterD>(p0q1.beginD(0) + i, p0q1.endD(0), 3);
    thrust::copy(p0q2.beginD(0), p0q2.endD(0), verts3.begin());
  }
  thrust::for_each_n(
      zip(thrust::make_counting_iterator(0), p1q1.beginD(0)), p1q1.size(),
      CopyEdgeVerts({p0q1.ptrD(0) + 3 * p0q2.size(), inP.edgeVerts_.cptrD()}));
  // Copy edges
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), p0q2.beginD(1)),
                     p0q2.size(),
                     CopyTriEdges({p0q1.ptrD(1), inQ.triEdges_.cptrD()}));
  for (int i : {0, 1}) {
    auto edges2 = strided_range<VecDH<int>::IterD>(
        p0q1.beginD(1) + 3 * p0q2.size() + i, p0q1.endD(1), 2);
    thrust::copy(p1q1.beginD(1), p1q1.endD(1), edges2.begin());
  }
  return p0q1;
}

struct Not_zero {
  __host__ __device__ bool operator()(const int x) { return x != 0; }
};

struct Right : public thrust::unary_function<EdgeTrisD, int> {
  __host__ __device__ int operator()(EdgeTrisD edge) { return edge.right; }
};

struct Left : public thrust::unary_function<EdgeTrisD, int> {
  __host__ __device__ int operator()(EdgeTrisD edge) { return edge.left; }
};

struct AbsSum : public thrust::binary_function<int, int, int> {
  __host__ __device__ int operator()(int a, int b) { return abs(a) + abs(b); }
};

struct ShadowKernel01 {
  const bool reverse;
  const glm::vec3 *vertPosP;
  const glm::vec3 *vertPosQ;

  __host__ __device__ void operator()(
      thrust::tuple<int &, int, EdgeVertsD> inout) {
    int &s01 = thrust::get<0>(inout);
    int vertP = thrust::get<1>(inout);
    EdgeVertsD edgeVertsQ = thrust::get<2>(inout);

    s01 = reverse ? (vertPosQ[edgeVertsQ.first].x <= vertPosP[vertP].x) -
                        (vertPosQ[edgeVertsQ.second].x <= vertPosP[vertP].x)
                  : (vertPosQ[edgeVertsQ.second].x >= vertPosP[vertP].x) -
                        (vertPosQ[edgeVertsQ.first].x >= vertPosP[vertP].x);
  }
};

struct Kernel01 {
  const bool reverse;
  const glm::vec3 *vertPosP;
  const glm::vec3 *vertPosQ;

  __host__ __device__ void operator()(
      thrust::tuple<glm::vec2 &, int &, EdgeVertsD, int> inout) {
    glm::vec2 &yz01 = thrust::get<0>(inout);
    int &s01 = thrust::get<1>(inout);
    EdgeVertsD edgeVertsQ = thrust::get<2>(inout);
    int vertP = thrust::get<3>(inout);

    glm::vec3 vertPos0 = vertPosQ[edgeVertsQ.first];
    glm::vec3 vertPos1 = vertPosQ[edgeVertsQ.second];
    yz01 = Interpolate(vertPos0, vertPos1, vertPosP[vertP].x);
    if (reverse) {
      if (yz01[0] > vertPosP[vertP].y) s01 = 0;
    } else {
      if (yz01[0] < vertPosP[vertP].y) s01 = 0;
    }
  }
};

std::tuple<VecDH<int>, VecDH<glm::vec2>> Shadow01(SparseIndices &p0q1,
                                                  const Manifold::Impl &inP,
                                                  const Manifold::Impl &inQ,
                                                  bool reverse) {
  // TODO: remove reverse once symbolic perturbation is updated to be per-vertex
  VecDH<int> s01(p0q1.size());
  if (reverse) p0q1.Swap();
  auto BedgeV = perm(inQ.edgeVerts_.beginD(), p0q1.beginD(1));
  thrust::for_each_n(
      zip(s01.beginD(), p0q1.beginD(0), BedgeV), p0q1.size(),
      ShadowKernel01({reverse, inP.vertPos_.cptrD(), inQ.vertPos_.cptrD()}));
  size_t size = p0q1.RemoveZeros(s01);
  VecDH<glm::vec2> yz01(size);
  BedgeV = perm(inQ.edgeVerts_.beginD(), p0q1.beginD(1));
  thrust::for_each_n(
      zip(yz01.beginD(), s01.beginD(), BedgeV, p0q1.beginD(0)), size,
      Kernel01({reverse, inP.vertPos_.cptrD(), inQ.vertPos_.cptrD()}));
  if (reverse) p0q1.Swap();
  return std::make_tuple(s01, yz01);
}

template <typename Val>
__host__ __device__ Val BinarySearchByKey(
    const thrust::pair<const int *, const int *> keys, const Val *vals,
    const int size, const thrust::pair<int, int> key, const Val missingVal) {
  if (size <= 0) return missingVal;
  int left = 0;
  int right = size - 1;
  int m;
  thrust::pair<int, int> keyM;
  for (;;) {
    m = right - (right - left) / 2;
    keyM = thrust::make_pair(keys.first[m], keys.second[m]);
    if (left == right) break;
    if (keyM > key)
      right = m - 1;
    else
      left = m;
  }
  if (keyM == key)
    return vals[m];
  else
    return missingVal;
}

struct Gather01 {
  const thrust::pair<const int *, const int *> p0q1;
  const int *s01;
  const int size;
  const EdgeVertsD *edgeVertsP;
  const bool reverse;

  __host__ __device__ void operator()(thrust::tuple<int &, int, int> inout) {
    int &s11 = thrust::get<0>(inout);
    const int p1 = thrust::get<1>(inout);
    const int q1 = thrust::get<2>(inout);

    int p0 = edgeVertsP[p1].second;
    auto key = reverse ? thrust::make_pair(q1, p0) : thrust::make_pair(p0, q1);
    s11 += BinarySearchByKey(p0q1, s01, size, key, 0);
    p0 = edgeVertsP[p1].first;
    key = reverse ? thrust::make_pair(q1, p0) : thrust::make_pair(p0, q1);
    s11 -= BinarySearchByKey(p0q1, s01, size, key, 0);
  }
};

__host__ __device__ glm::ivec2 Middle2of4(glm::vec4 in) {
  glm::ivec4 idx(0, 1, 2, 3);
#define SWAP(a, b)     \
  if (in[b] < in[a]) { \
    float x1 = in[a];  \
    in[a] = in[b];     \
    in[b] = x1;        \
    int idx1 = idx[a]; \
    idx[a] = idx[b];   \
    idx[b] = idx1;     \
  }
  SWAP(0, 1);
  SWAP(2, 3);
  SWAP(0, 2);
  SWAP(1, 3);
  SWAP(1, 2);
#undef SWAP
  return glm::ivec2(idx[1], idx[2]);
}

struct Kernel11 {
  const glm::vec3 *vertPosP;
  const glm::vec3 *vertPosQ;
  const EdgeVertsD *edgeVertsP;
  const EdgeVertsD *edgeVertsQ;
  thrust::pair<const int *, const int *> p0q1;
  const glm::vec2 *yz01;
  int size01;
  thrust::pair<const int *, const int *> p1q0;
  const glm::vec2 *yz10;
  int size10;

  __host__ __device__ void operator()(
      thrust::tuple<glm::vec4 &, int &, int, int> inout) {
    glm::vec4 &xyzz11 = thrust::get<0>(inout);
    int &s11 = thrust::get<1>(inout);
    const int p1 = thrust::get<2>(inout);
    const int q1 = thrust::get<3>(inout);

    thrust::pair<int, int> key4[4];
    key4[0] = thrust::make_pair(edgeVertsP[p1].first, q1);
    key4[1] = thrust::make_pair(edgeVertsP[p1].second, q1);
    key4[2] = thrust::make_pair(p1, edgeVertsQ[q1].first);
    key4[3] = thrust::make_pair(p1, edgeVertsQ[q1].second);

    glm::vec3 vertPos4[4];
    glm::vec4 x;
    for (int i : {0, 1, 2, 3}) {
      vertPos4[i] = i < 2 ? vertPosP[key4[i].first] : vertPosQ[key4[i].second];
      x[i] = vertPos4[i].x;
    }
    const glm::ivec2 tmpLR = Middle2of4(x);
    const int idxL = tmpLR.x;
    const int idxR = tmpLR.y;

    auto pq = idxL < 2 ? p0q1 : p1q0;
    auto yz = idxL < 2 ? yz01 : yz10;
    int size = idxL < 2 ? size01 : size10;
    const glm::vec2 yzL =
        BinarySearchByKey(pq, yz, size, key4[idxL], glm::vec2(0.0f / 0.0f));
    pq = idxR < 2 ? p0q1 : p1q0;
    yz = idxR < 2 ? yz01 : yz10;
    size = idxR < 2 ? size01 : size10;
    const glm::vec2 yzR =
        BinarySearchByKey(pq, yz, size, key4[idxR], glm::vec2(0.0f / 0.0f));

    glm::vec3 pL, pR, qL, qR;
    pL = vertPos4[idxL];
    qL.x = pL.x;
    qL.y = yzL[0];
    qL.z = yzL[1];
    if (idxL > 1) thrust::swap(pL, qL);  // idxL is an endpoint of Q
    pR = vertPos4[idxR];
    qR.x = pR.x;
    qR.y = yzR[0];
    qR.z = yzR[1];
    if (idxR > 1) thrust::swap(pR, qR);  // idxR is an endpoint of Q
    xyzz11 = Intersect(pL, pR, qL, qR);
    if (xyzz11.z > xyzz11.w) s11 = 0;
  }
};

std::tuple<VecDH<int>, VecDH<glm::vec4>> Shadow11(
    SparseIndices &p1q1, const Manifold::Impl &inP, const Manifold::Impl &inQ,
    const SparseIndices &p0q1, const VecDH<int> &s01,
    const VecDH<glm::vec2> &yz01, const SparseIndices &p1q0,
    const VecDH<int> &s10, const VecDH<glm::vec2> &yz10) {
  VecDH<int> s11(p1q1.size(), 0);

  thrust::for_each_n(zip(s11.beginD(), p1q1.beginD(0), p1q1.beginD(1)),
                     p1q1.size(),
                     Gather01({p0q1.ptrDpq(), s01.ptrD(), p0q1.size(),
                               inP.edgeVerts_.ptrD(), false}));
  thrust::for_each_n(zip(s11.beginD(), p1q1.beginD(1), p1q1.beginD(0)),
                     p1q1.size(),
                     Gather01({p1q0.ptrDpq(), s10.ptrD(), p1q0.size(),
                               inQ.edgeVerts_.ptrD(), true}));

  size_t size = p1q1.RemoveZeros(s11);
  VecDH<glm::vec4> xyzz11(size);

  thrust::for_each_n(
      zip(xyzz11.beginD(), s11.beginD(), p1q1.beginD(0), p1q1.beginD(1)),
      p1q1.size(),
      Kernel11({inP.vertPos_.ptrD(), inQ.vertPos_.ptrD(), inP.edgeVerts_.ptrD(),
                inQ.edgeVerts_.ptrD(), p0q1.ptrDpq(), yz01.ptrD(), p0q1.size(),
                p1q0.ptrDpq(), yz10.ptrD(), p1q0.size()}));

  return std::make_tuple(s11, xyzz11);
};

struct Gather02 {
  const thrust::pair<const int *, const int *> p0q1;
  const int *s01;
  const int size;
  const TriEdges *triEdgesQ;
  const bool forward;

  __host__ __device__ void operator()(thrust::tuple<int &, int, int> inout) {
    int &s02 = thrust::get<0>(inout);
    const int p0 = thrust::get<1>(inout);
    const int q2 = thrust::get<2>(inout);

    const TriEdges triEdges = triEdgesQ[q2];
    for (int i : {0, 1, 2}) {
      auto key = forward ? thrust::make_pair(p0, triEdges[i].Idx())
                         : thrust::make_pair(triEdges[i].Idx(), p0);
      s02 += (forward ? -1 : 1) * triEdges[i].Dir() *
             BinarySearchByKey(p0q1, s01, size, key, 0);
    }
  }
};

struct Kernel02 {
  const glm::vec3 *vertPosP;
  const thrust::pair<const int *, const int *> p0q1;
  const glm::vec2 *yz01;
  const int size;
  const TriEdges *triEdgesQ;
  const bool forward;

  __host__ __device__ void operator()(
      thrust::tuple<float &, int &, int, int> inout) {
    float &z02 = thrust::get<0>(inout);
    int &s02 = thrust::get<1>(inout);
    const int p0 = thrust::get<2>(inout);
    const int q2 = thrust::get<3>(inout);

    const TriEdges triEdges = triEdgesQ[q2];
    glm::vec2 yz3[3];
    for (int i : {0, 1, 2}) {
      auto key = forward ? thrust::make_pair(p0, triEdges[i].Idx())
                         : thrust::make_pair(triEdges[i].Idx(), p0);
      yz3[i] = BinarySearchByKey(p0q1, yz01, size, key, glm::vec2(0.0f / 0.0f));
    }

    // assert exactly 2 of yz3 are found
    // if (isnan(yz3[0][0]) + isnan(yz3[1][0]) + isnan(yz3[2][0]) != 1)
    //   printf("yz0 = %f, yz1 = %f, yz2 = %f\n", yz3[0][0], yz3[1][0],
    //   yz3[2][0]);
    if (isnan(yz3[0][0])) yz3[0] = yz3[2];
    if (isnan(yz3[1][0])) yz3[1] = yz3[2];
    glm::vec3 pL, pR;
    pL.x = yz3[0][0];
    pL.y = yz3[0][1];
    pL.z = yz3[0][1];
    pR.x = yz3[1][0];
    pR.y = yz3[1][1];
    pR.z = yz3[1][1];
    glm::vec3 vertPos = vertPosP[p0];
    z02 = Interpolate(pL, pR, vertPos.y)[1];
    if (forward) {
      if (z02 < vertPos.z) s02 = 0;
    } else {
      if (z02 > vertPos.z) s02 = 0;
    }
  }
};

std::tuple<VecDH<int>, VecDH<float>> Shadow02(
    const VecDH<glm::vec3> &vertPosP, const Manifold::Impl &inQ,
    const VecDH<int> &s01, const SparseIndices &p0q1,
    const VecDH<glm::vec2> &yz01, SparseIndices &p0q2, bool forward) {
  VecDH<int> s02(p0q2.size(), 0);

  thrust::for_each_n(
      zip(s02.beginD(), p0q2.beginD(!forward), p0q2.beginD(forward)),
      p0q2.size(),
      Gather02({p0q1.ptrDpq(), s01.ptrD(), p0q1.size(), inQ.triEdges_.ptrD(),
                forward}));

  size_t size = p0q2.RemoveZeros(s02);
  VecDH<float> z02(size);

  thrust::for_each_n(zip(z02.beginD(), s02.beginD(), p0q2.beginD(!forward),
                         p0q2.beginD(forward)),
                     size,
                     Kernel02({vertPosP.ptrD(), p0q1.ptrDpq(), yz01.ptrD(),
                               p0q1.size(), inQ.triEdges_.ptrD(), forward}));

  return std::make_tuple(s02, z02);
};

struct Gather12 {
  const thrust::pair<const int *, const int *> p0q2;
  const int *s02;
  const int size02;
  const thrust::pair<const int *, const int *> p1q1;
  const int *s11;
  const int size11;
  const EdgeVertsD *edgeVertsP;
  const TriEdges *triEdgesQ;
  const bool forward;

  __host__ __device__ void operator()(thrust::tuple<int &, int, int> inout) {
    int &x12 = thrust::get<0>(inout);
    const int p1 = thrust::get<1>(inout);
    const int q2 = thrust::get<2>(inout);

    const EdgeVertsD edgeVerts = edgeVertsP[p1];
    auto key = forward ? thrust::make_pair(edgeVerts.first, q2)
                       : thrust::make_pair(q2, edgeVerts.second);
    x12 = BinarySearchByKey(p0q2, s02, size02, key, 0);
    key = forward ? thrust::make_pair(edgeVerts.second, q2)
                  : thrust::make_pair(q2, edgeVerts.first);
    x12 -= BinarySearchByKey(p0q2, s02, size02, key, 0);

    const TriEdges triEdges = triEdgesQ[q2];
    for (int i : {0, 1, 2}) {
      key = forward ? thrust::make_pair(p1, triEdges[i].Idx())
                    : thrust::make_pair(triEdges[i].Idx(), p1);
      x12 -= triEdges[i].Dir() * BinarySearchByKey(p1q1, s11, size11, key, 0);
    }
  }
};

struct Kernel12 {
  const thrust::pair<const int *, const int *> p0q2;
  const float *z02;
  const int size02;
  const thrust::pair<const int *, const int *> p1q1;
  const glm::vec4 *xyzz11;
  const int size11;
  const EdgeVertsD *edgeVertsP;
  const TriEdges *triEdgesQ;
  const glm::vec3 *vertPosP;
  const bool forward;

  __host__ __device__ void operator()(
      thrust::tuple<glm::vec3 &, int, int> inout) {
    glm::vec3 &v12 = thrust::get<0>(inout);
    const int p1 = thrust::get<1>(inout);
    const int q2 = thrust::get<2>(inout);

    const EdgeVertsD edgeVerts = edgeVertsP[p1];
    auto key = forward ? thrust::make_pair(edgeVerts.first, q2)
                       : thrust::make_pair(q2, edgeVerts.first);
    float z0 = BinarySearchByKey(p0q2, z02, size02, key, 0.0f / 0.0f);
    key = forward ? thrust::make_pair(edgeVerts.second, q2)
                  : thrust::make_pair(q2, edgeVerts.second);
    float z1 = BinarySearchByKey(p0q2, z02, size02, key, 0.0f / 0.0f);

    const TriEdges triEdges = triEdgesQ[q2];
    glm::vec4 xyzz3[3];
    for (int i : {0, 1, 2}) {
      key = forward ? thrust::make_pair(p1, triEdges[i].Idx())
                    : thrust::make_pair(triEdges[i].Idx(), p1);
      xyzz3[i] =
          BinarySearchByKey(p1q1, xyzz11, size11, key, glm::vec4(0.0f / 0.0f));
    }

    glm::vec3 xzyLR0[2];
    glm::vec3 xzyLR1[2];
    int k = 0;
    if (!isnan(z0)) {
      xzyLR0[k] = vertPosP[edgeVerts.first];
      thrust::swap(xzyLR0[k].y, xzyLR0[k].z);
      xzyLR1[k] = xzyLR0[k];
      xzyLR1[k][1] = z0;
      k++;
    }
    if (!isnan(z1)) {
      xzyLR0[k] = vertPosP[edgeVerts.second];
      thrust::swap(xzyLR0[k].y, xzyLR0[k].z);
      xzyLR1[k] = xzyLR0[k];
      xzyLR1[k][1] = z1;
      k++;
    }
    for (int i : {0, 1, 2}) {
      if (!isnan(xyzz3[i].x)) {
        xzyLR0[k][0] = xyzz3[i].x;
        xzyLR0[k][1] = xyzz3[i].z;
        xzyLR0[k][2] = xyzz3[i].y;
        xzyLR1[k] = xzyLR0[k];
        xzyLR1[k][1] = xyzz3[i].w;
        if (!forward) thrust::swap(xzyLR0[k][1], xzyLR1[k][1]);
        k++;
      }
    }
    // assert exactly two of these five were found
    // if (k != 2) printf("k = %d\n", k);

    glm::vec4 xzyy = Intersect(xzyLR0[0], xzyLR0[1], xzyLR1[0], xzyLR1[1]);
    v12.x = xzyy[0];
    v12.y = xzyy[2];
    v12.z = xzyy[1];
  }
};

std::tuple<VecDH<int>, VecDH<glm::vec3>> Intersect12(
    const Manifold::Impl &inP, const Manifold::Impl &inQ, const VecDH<int> &s02,
    const SparseIndices &p0q2, const VecDH<int> &s11, const SparseIndices &p1q1,
    const VecDH<float> &z02, const VecDH<glm::vec4> &xyzz11,
    SparseIndices &p1q2, bool forward) {
  VecDH<int> x12(p1q2.size());
  VecDH<glm::vec3> v12;

  auto edgeVertsPtr = forward ? inP.edgeVerts_.ptrD() : inQ.edgeVerts_.ptrD();
  auto triEdgesPtr = forward ? inQ.triEdges_.ptrD() : inP.triEdges_.ptrD();
  thrust::for_each_n(
      zip(x12.beginD(), p1q2.beginD(!forward), p1q2.beginD(forward)),
      p1q2.size(),
      Gather12({p0q2.ptrDpq(), s02.ptrD(), p0q2.size(), p1q1.ptrDpq(),
                s11.ptrD(), p1q1.size(), edgeVertsPtr, triEdgesPtr, forward}));

  size_t size = p1q2.RemoveZeros(x12);
  v12.resize(size);

  auto vertPosPtr = forward ? inP.vertPos_.ptrD() : inQ.vertPos_.ptrD();
  thrust::for_each_n(
      zip(v12.beginD(), p1q2.beginD(!forward), p1q2.beginD(forward)),
      p1q2.size(),
      Kernel12({p0q2.ptrDpq(), z02.ptrD(), p0q2.size(), p1q1.ptrDpq(),
                xyzz11.ptrD(), p1q1.size(), edgeVertsPtr, triEdgesPtr,
                vertPosPtr, forward}));
  return std::make_tuple(x12, v12);
};

VecDH<int> Winding03(const Manifold::Impl &inP, SparseIndices &p0q2,
                     VecDH<int> &s02, const SparseIndices &p1q2, bool reverse) {
  VecDH<int> w03(inP.NumVert(), kInvalidInt);
  // keepEdgesP is the set of edges that connect regions of the manifold with
  // the same winding number, so we remove any edges associated with
  // intersections.
  VecDH<bool> keepEdgesP(inP.NumEdge(), true);
  thrust::scatter(thrust::make_constant_iterator(false, 0),
                  thrust::make_constant_iterator(false, p1q2.size()),
                  p1q2.beginD(reverse), keepEdgesP.beginD());

  if (!thrust::is_sorted(p0q2.beginD(reverse), p0q2.endD(reverse)))
    thrust::sort_by_key(p0q2.beginD(reverse), p0q2.endD(reverse), s02.beginD());
  VecDH<int> w03val(w03.size());
  VecDH<int> w03vert(w03.size());
  // sum known s02 values into w03 (winding number)
  auto endPair =
      thrust::reduce_by_key(p0q2.beginD(reverse), p0q2.endD(reverse),
                            s02.beginD(), w03vert.beginD(), w03val.beginD());
  thrust::scatter(w03val.beginD(), endPair.second, w03vert.beginD(),
                  w03.beginD());

  // find connected regions (separated by intersections)
  VecDH<int> vLabels;
  int n_comp =
      ConnectedComponents(vLabels, inP.NumVert(), inP.edgeVerts_, keepEdgesP);
  // flood the w03 values throughout their connected components (they are
  // consistent)
  FloodComponents(w03, vLabels, n_comp);

  if (kVerbose) std::cout << n_comp << " components" << std::endl;

  if (reverse)
    thrust::transform(w03.beginD(), w03.endD(), w03.beginD(),
                      thrust::negate<int>());
  return w03;
};

SparseIndices Intersect22(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                          const SparseIndices &p1q2, const SparseIndices &p2q1,
                          const VecDH<int> &dir12, const VecDH<int> &dir21) {
  SparseIndices p2q2(2 * (p1q2.size() + p2q1.size()));
  auto f22ptr = p2q2.beginDpq();
  auto rightTrisA =
      zip(perm(thrust::make_transform_iterator(inP.edgeTris_.beginD(), Right()),
               p1q2.beginD(0)),
          p1q2.beginD(1));
  f22ptr = thrust::copy_if(rightTrisA, rightTrisA + p1q2.size(), dir12.beginD(),
                           f22ptr, Not_zero());
  auto leftTrisA =
      zip(perm(thrust::make_transform_iterator(inP.edgeTris_.beginD(), Left()),
               p1q2.beginD(0)),
          p1q2.beginD(1));
  f22ptr = thrust::copy_if(leftTrisA, leftTrisA + p1q2.size(), dir12.beginD(),
                           f22ptr, Not_zero());
  auto rightTrisB =
      zip(p2q1.beginD(0),
          perm(thrust::make_transform_iterator(inQ.edgeTris_.beginD(), Right()),
               p2q1.beginD(1)));
  f22ptr = thrust::copy_if(rightTrisB, rightTrisB + p2q1.size(), dir21.beginD(),
                           f22ptr, Not_zero());
  auto leftTrisB =
      zip(p2q1.beginD(0),
          perm(thrust::make_transform_iterator(inQ.edgeTris_.beginD(), Left()),
               p2q1.beginD(1)));
  f22ptr = thrust::copy_if(leftTrisB, leftTrisB + p2q1.size(), dir21.beginD(),
                           f22ptr, Not_zero());
  p2q2.Unique();
  return p2q2;
}

struct AssignOnes {
  bool *keepTrisP;
  bool *keepTrisQ;

  __host__ __device__ void operator()(thrust::tuple<int, int> p2q2) {
    keepTrisP[thrust::get<0>(p2q2)] = true;
    keepTrisQ[thrust::get<1>(p2q2)] = true;
  }
};

struct DuplicateVerts {
  glm::vec3 *vertPosR;
  int *vertR2P;

  __host__ __device__ void operator()(
      thrust::tuple<int, int, int, glm::vec3> in) {
    int vertP = thrust::get<0>(in);
    int inclusion = abs(thrust::get<1>(in));
    int vertR = thrust::get<2>(in);
    glm::vec3 vertPosP = thrust::get<3>(in);

    for (int i = 0; i < inclusion; ++i) {
      vertPosR[vertR + i] = vertPosP;
      vertR2P[vertR + i] = vertP;
    }
  }
};

void AppendRetainedFaces(VecDH<glm::ivec3> &triVerts, VecDH<int> p12,
                         VecDH<int> p21, const VecDH<int> &i03,
                         const VecDH<int> &vP2R, const Manifold::Impl &inP) {
  // keepTriP is a list of the triangle indicies of P which does not include the
  // triangles that intersect edges of Q.
  VecDH<int> keepTriP(inP.NumTri());
  if (!thrust::is_sorted(p21.beginD(), p21.endD()))
    thrust::sort(p21.beginD(), p21.endD());
  auto fA2end = thrust::unique(p21.beginD(), p21.endD());
  keepTriP.resize(
      thrust::set_difference(thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(inP.NumTri()),
                             p21.beginD(), fA2end, keepTriP.beginD()) -
      keepTriP.beginD());

  if (!thrust::is_sorted(p12.beginD(), p12.endD()))
    thrust::sort(p12.beginD(), p12.endD());
  p12.resize(thrust::unique(p12.beginD(), p12.endD()) - p12.beginD());

  for (int i : keepTriP.H()) {
    // Discard any triangles whose edges intersect triangles of Q.
    thrust::host_vector<int> edges(3);
    thrust::host_vector<bool> bad(3);
    for (int j : {0, 1, 2}) edges[j] = inP.triEdges_.H()[i][j].Idx();
    thrust::binary_search(p12.begin(), p12.end(), edges.begin(), edges.end(),
                          bad.begin());
    if (thrust::any_of(bad.begin(), bad.end(), thrust::identity<bool>()))
      continue;
    // Check the inclusion number of a single vertex of a triangle, since
    // non-intersecting triangles must have all identical inclusion numbers.
    glm::ivec3 triVertsP = inP.triVerts_.H()[i];
    int inclusion = i03.H()[triVertsP[0]];
    glm::ivec3 triVertsR(vP2R.H()[triVertsP[0]], vP2R.H()[triVertsP[1]],
                         vP2R.H()[triVertsP[2]]);
    if (inclusion < 0) std::swap(triVertsR[1], triVertsR[2]);
    for (int i = 0; i < abs(inclusion); ++i)
      triVerts.H().push_back(triVertsR + i);
  }
}

struct GetRetainedEdges {
  bool *retainedEdges;

  __host__ __device__ void operator()(thrust::tuple<TriEdges, bool> in) {
    TriEdges triEdges = thrust::get<0>(in);
    bool isIntersecting = thrust::get<1>(in);

    if (isIntersecting)
      for (int i : {0, 1, 2}) retainedEdges[triEdges[i].Idx()] = true;
  }
};

struct VertsPos {
  int vidx;
  int dir;
  float edgePos;
};

std::vector<EdgeVerts> PairUp(std::vector<VertsPos> &vertsPos, int edge) {
  // Pair start vertices with end vertices to form edges. The choice of pairing
  // is arbitrary for the manifoldness guarantee, but must be ordered to be
  // geometrically valid. If the order does not go start-end-start-end... then
  // the input and output are not geometrically valid and this algorithm becomes
  // a heuristic.
  ALWAYS_ASSERT(vertsPos.size() % 2 == 0, logicErr,
                "Non-manifold edge! Not an even number of points.");
  int nEdges = vertsPos.size() / 2;
  auto middle = std::partition(vertsPos.begin(), vertsPos.end(),
                               [](VertsPos x) { return x.dir < 0; });
  ALWAYS_ASSERT(middle - vertsPos.begin() == nEdges, logicErr,
                "Non-manifold edge!");
  auto cmp = [](VertsPos a, VertsPos b) { return a.edgePos < b.edgePos; };
  std::sort(vertsPos.begin(), middle, cmp);
  std::sort(middle, vertsPos.end(), cmp);
  std::vector<EdgeVerts> edges;
  for (int i = 0; i < nEdges; ++i)
    edges.push_back({vertsPos[i].vidx, vertsPos[i + nEdges].vidx, edge});
  return edges;
}

void AppendRetainedEdges(std::vector<std::vector<EdgeVerts>> &faces,
                         const Manifold::Impl &inP, const VecDH<int> &i03,
                         const VecDH<int> &i12, const VecDH<int> &p12,
                         const VecDH<bool> &intersectedTriP,
                         const VecDH<glm::vec3> &vertPos,
                         const VecDH<int> &vP2R, int start12) {
  VecDH<bool> retainedEdgesBool(inP.NumEdge(), false);
  thrust::for_each_n(zip(inP.triEdges_.beginD(), intersectedTriP.beginD()),
                     inP.NumTri(),
                     GetRetainedEdges({retainedEdgesBool.ptrD()}));
  VecDH<int> retainedEdges(inP.NumEdge());
  int length = thrust::copy_if(thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(inP.NumEdge()),
                               retainedEdgesBool.beginD(),
                               retainedEdges.beginD(), Not_zero()) -
               retainedEdges.beginD();
  retainedEdges.resize(length);
  for (int edge : retainedEdges.H()) {
    std::vector<VertsPos> vertsPos;
    glm::vec3 edgeVec = inP.vertPos_.H()[inP.edgeVerts_.H()[edge].second] -
                        inP.vertPos_.H()[inP.edgeVerts_.H()[edge].first];
    // Get inP's vertices
    int v = inP.edgeVerts_.H()[edge].first;
    int inclusion = i03.H()[v];
    for (int i = 0; i < abs(inclusion); ++i)
      vertsPos.push_back({vP2R.H()[v] + i, -Signum(inclusion),
                          glm::dot(inP.vertPos_.H()[v], edgeVec)});
    v = inP.edgeVerts_.H()[edge].second;
    inclusion = i03.H()[v];
    for (int i = 0; i < abs(inclusion); ++i)
      vertsPos.push_back({vP2R.H()[v] + i, Signum(inclusion),
                          glm::dot(inP.vertPos_.H()[v], edgeVec)});
    // Get new vertices
    for (int i = 0; i < p12.size(); ++i) {
      if (p12.H()[i] != edge)
        continue;  // speed this up by splitting up the loops
      v = start12 + i;
      vertsPos.push_back(
          {v, Signum(i12.H()[i]), glm::dot(vertPos.H()[v], edgeVec)});
    }
    // sort edges into start/end pairs along length
    std::vector<EdgeVerts> edges = PairUp(vertsPos, edge);
    // add edges to face lists
    int faceidx = inP.edgeTris_.H()[edge].left;
    if (intersectedTriP.H()[faceidx]) {
      auto &face = faces[faceidx];
      face.insert(face.end(), edges.begin(), edges.end());
    }
    faceidx = inP.edgeTris_.H()[edge].right;
    if (intersectedTriP.H()[faceidx]) {
      auto &face = faces[faceidx];
      // reverse edges
      for (auto &e : edges) std::swap(e.first, e.second);
      face.insert(face.end(), edges.begin(), edges.end());
    }
  }
}

void AppendNewEdges(std::vector<std::vector<EdgeVerts>> &facesP,
                    std::vector<std::vector<EdgeVerts>> &facesQ,
                    const VecDH<TriEdges> &triEdgesP,
                    const VecDH<TriEdges> &triEdgesQ, const SparseIndices &p1q2,
                    const SparseIndices &p2q1, const SparseIndices &p2q2,
                    const VecDH<int> &i12, const VecDH<int> &i21, int start12) {
  for (int k = 0; k < p2q2.size(); ++k) {
    int triP = p2q2.Get(0).H()[k];
    int triQ = p2q2.Get(1).H()[k];
    std::vector<int> edge, dir;
    thrust::host_vector<int> edges(3), found(3), idx(3);
    // edges of inP - face of inQ
    for (int i : {0, 1, 2}) edges[i] = triEdgesP.H()[triP][i].Idx();
    // TODO: simplify this logic
    thrust::binary_search(
        p1q2.beginHpq(), p1q2.endHpq(),
        zip(edges.begin(), thrust::make_constant_iterator(triQ, 0)),
        zip(edges.end(), thrust::make_constant_iterator(triQ, 3)),
        found.begin());
    thrust::lower_bound(
        p1q2.beginHpq(), p1q2.endHpq(),
        zip(edges.begin(), thrust::make_constant_iterator(triQ, 0)),
        zip(edges.end(), thrust::make_constant_iterator(triQ, 3)), idx.begin());
    for (int i : {0, 1, 2}) {
      if (found[i]) {
        edge.push_back(idx[i] + start12);
        dir.push_back(-i12.H()[idx[i]] * triEdgesP.H()[triP][i].Dir());
      }
    }
    // face of inP - edges of inQ
    for (int i : {0, 1, 2}) edges[i] = triEdgesQ.H()[triQ][i].Idx();
    thrust::binary_search(
        p2q1.beginHpq(), p2q1.endHpq(),
        zip(thrust::make_constant_iterator(triP, 0), edges.begin()),
        zip(thrust::make_constant_iterator(triP, 3), edges.end()),
        found.begin());
    thrust::lower_bound(
        p2q1.beginHpq(), p2q1.endHpq(),
        zip(thrust::make_constant_iterator(triP, 0), edges.begin()),
        zip(thrust::make_constant_iterator(triP, 3), edges.end()), idx.begin());
    for (int i : {0, 1, 2}) {
      if (found[i]) {
        edge.push_back(idx[i] + start12 + p1q2.size());
        dir.push_back(i21.H()[idx[i]] * triEdgesQ.H()[triQ][i].Dir());
      }
    }
    ALWAYS_ASSERT(
        edge.size() == 2, logicErr,
        "Number of points in intersection of two triangles did not equal 2!");
    ALWAYS_ASSERT(dir[0] == -dir[1], logicErr,
                  "Intersection points do not have opposite directions!");
    if (dir[0] > 0) std::swap(edge[0], edge[1]);
    // Since these are not input edges, their index is undefined, so set to -1.
    facesP[triP].push_back({edge[0], edge[1], Edge::kNoIdx});
    facesQ[triQ].push_back({edge[1], edge[0], Edge::kNoIdx});
  }
}

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

void AppendIntersectedFaces(VecDH<glm::ivec3> &triVerts,
                            VecDH<glm::vec3> &vertPos,
                            const std::vector<std::vector<EdgeVerts>> &facesP,
                            const Manifold::Impl &inP, bool invertNormals) {
  for (int i = 0; i < facesP.size(); ++i) {
    auto &face = facesP[i];
    switch (face.size()) {
      case 0:
        break;
      case 1:
      case 2:
        ALWAYS_ASSERT(face.size() >= 3, logicErr,
                      "face has less than three edges.");
      case 3: {  // triangle
        auto tri = face;
        if (tri[0].second == tri[2].first) std::swap(tri[1], tri[2]);
        ALWAYS_ASSERT(tri[0].second == tri[1].first &&
                          tri[1].second == tri[2].first &&
                          tri[2].second == tri[0].first,
                      runtimeErr, "These 3 edges do not form a triangle!");
        glm::ivec3 triangle(tri[0].first, tri[1].first, tri[2].first);
        triVerts.H().push_back(triangle);
        break;
      }
      default: {  // requires triangulation
        Polygons polys = Assemble(face);
        glm::vec3 normal = inP.GetTriNormal(i);
        if (invertNormals) normal *= -1.0f;
        glm::mat3x2 projection = GetAxisAlignedProjection(normal);
        for (auto &poly : polys) {
          for (PolyVert &v : poly) {
            v.pos = projection * vertPos.H()[v.idx];
          }
        }
        std::vector<glm::ivec3> newTris;
        try {
          newTris = Triangulate(polys);
        } catch (const std::exception &e) {
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
          for (const auto &poly : polys) {
            glm::vec3 centroid = thrust::transform_reduce(
                poly.begin(), poly.end(),
                [&vertPos](PolyVert v) { return vertPos.H()[v.idx]; },
                glm::vec3(0.0f),
                [](glm::vec3 a, glm::vec3 b) { return a + b; });
            centroid /= poly.size();
            int newVert = vertPos.size();
            vertPos.H().push_back(centroid);
            newTris.push_back({poly.back().idx, poly.front().idx, newVert});
            for (int i = 1; i < poly.size(); ++i)
              newTris.push_back({poly[i - 1].idx, poly[i].idx, newVert});
          }
        }
        for (auto tri : newTris) triVerts.H().push_back(tri);
      }
    }
  }
}

void CheckPreTriangulationManfold(
    const VecDH<glm::ivec3> &triVerts,
    const std::vector<std::vector<EdgeVerts>> &facesP,
    const std::vector<std::vector<EdgeVerts>> &facesQ) {
  std::vector<glm::ivec3> triVertsH;
  for (auto tri : triVerts) triVertsH.push_back(tri);
  std::vector<EdgeVerts> edges = Triangles2Edges(triVertsH);
  for (const auto &face : facesP) {
    for (const EdgeVerts &edge : face) {
      edges.push_back(edge);
    }
  }
  for (const auto &face : facesQ) {
    for (const EdgeVerts &edge : face) {
      edges.push_back(edge);
    }
  }
  // Edge data creates an extra manifoldness constraint that is only needed for
  // checking polygon triangulation.
  for (auto &edge : edges) edge.edge = Edge::kNoIdx;
  CheckTopology(edges);
}
}  // namespace

namespace manifold {

Boolean3::Boolean3(const Manifold::Impl &inP, const Manifold::Impl &inQ)
    : inP_(inP), inQ_(inQ) {
  // TODO: new symbolic perturbation:
  // Union -> expand inP
  // Difference, Intersection -> contract inP

  // Level 3
  // Find edge-triangle overlaps (broad phase)
  p1q2_ = inQ_.EdgeCollisions(inP_);
  p1q2_.Sort();
  if (kVerbose) std::cout << "p1q2 size = " << p1q2_.size() << std::endl;

  p2q1_ = inP_.EdgeCollisions(inQ_);
  p2q1_.Swap();
  p2q1_.Sort();
  if (kVerbose) std::cout << "p2q1 size = " << p2q1_.size() << std::endl;

  // Level 2
  // Find vertices from Level 3 that overlap triangles in XY-projection
  SparseIndices p0q2 = Filter02(inP_, inQ_, p1q2_.Get(0), p2q1_.Get(0));
  p0q2.Sort();
  if (kVerbose) std::cout << "p0q2 size = " << p0q2.size() << std::endl;

  SparseIndices p2q0 = Filter02(inQ_, inP_, p2q1_.Get(1), p1q2_.Get(1));
  p2q0.Swap();
  p2q0.Sort();
  if (kVerbose) std::cout << "p2q0 size = " << p2q0.size() << std::endl;

  // Find involved edge pairs from Level 3
  SparseIndices p1q1 = Filter11(inP_, inQ_, p1q2_, p2q1_);
  if (kVerbose) std::cout << "p1q1 size = " << p1q1.size() << std::endl;

  // Level 1
  // Find involved vertex-edge pairs Level 2
  SparseIndices p0q1 = Filter01(inP_, inQ_, p0q2, p1q1);
  p0q1.Unique();
  if (kVerbose) std::cout << "p0q1 size = " << p0q1.size() << std::endl;

  p2q0.Swap();
  p1q1.Swap();
  SparseIndices p1q0 = Filter01(inQ_, inP_, p2q0, p1q1);
  p2q0.Swap();
  p1q1.Swap();
  p1q0.Swap();
  p1q0.Unique();
  if (kVerbose) std::cout << "p1q0 size = " << p1q0.size() << std::endl;

  // Level 1
  // Find X-projections of vertices onto edges, keeping only those that actually
  // fall inside the edge.
  VecDH<int> s01;
  VecDH<glm::vec2> yz01;
  std::tie(s01, yz01) = Shadow01(p0q1, inP, inQ, false);
  if (kVerbose) std::cout << "s01 size = " << s01.size() << std::endl;

  VecDH<int> s10;
  VecDH<glm::vec2> yz10;
  std::tie(s10, yz10) = Shadow01(p1q0, inQ, inP, true);
  if (kVerbose) std::cout << "s10 size = " << s10.size() << std::endl;

  // Level 2
  // Build up XY-projection intersection of two edges, including the z-value for
  // each edge, keeping only those whose intersection exists.
  VecDH<int> s11;
  VecDH<glm::vec4> xyzz11;
  std::tie(s11, xyzz11) =
      Shadow11(p1q1, inP, inQ, p0q1, s01, yz01, p1q0, s10, yz10);
  if (kVerbose) std::cout << "s11 size = " << s11.size() << std::endl;

  // Build up Z-projection of vertices onto triangles, keeping only those that
  // fall inside the triangle.
  VecDH<int> s02;
  VecDH<float> z02;
  std::tie(s02, z02) =
      Shadow02(inP_.vertPos_, inQ, s01, p0q1, yz01, p0q2, true);
  if (kVerbose) std::cout << "s02 size = " << s02.size() << std::endl;

  VecDH<int> s20;
  VecDH<float> z20;
  std::tie(s20, z20) =
      Shadow02(inQ_.vertPos_, inP, s10, p1q0, yz10, p2q0, false);
  if (kVerbose) std::cout << "s20 size = " << s20.size() << std::endl;

  // Level 3
  // Build up the intersection of the edges and triangles, keeping only those
  // that intersect, and record the direction the edge is passing through the
  // triangle.
  std::tie(dir12_, v12_) =
      Intersect12(inP, inQ, s02, p0q2, s11, p1q1, z02, xyzz11, p1q2_, true);
  if (kVerbose) std::cout << "dir12 size = " << dir12_.size() << std::endl;

  std::tie(dir21_, v21_) =
      Intersect12(inP, inQ, s20, p2q0, s11, p1q1, z20, xyzz11, p2q1_, false);
  if (kVerbose) std::cout << "dir21 size = " << dir21_.size() << std::endl;

  // Build up the winding numbers of all vertices. The involved vertices are
  // calculated from Level 2, while the rest are assigned consistently with
  // connected-components flooding.
  w03_ = Winding03(inP, p0q2, s02, p1q2_, false);

  w30_ = Winding03(inQ, p2q0, s20, p2q1_, true);

  // Level 4
  // Record all intersecting triangle pairs.
  p2q2_ = Intersect22(inP_, inQ_, p1q2_, p2q1_, dir12_, dir21_);
}

Manifold::Impl Boolean3::Result(Manifold::OpType op) const {
  int c1, c2, c3;
  switch (op) {
    case Manifold::OpType::ADD:
      c1 = 1;
      c2 = 1;
      c3 = -1;
      break;
    case Manifold::OpType::SUBTRACT:
      c1 = 1;
      c2 = 0;
      c3 = -1;
      break;
    case Manifold::OpType::INTERSECT:
      c1 = 0;
      c2 = 0;
      c3 = 1;
      break;
    default:
      throw std::invalid_argument("invalid enum: OpType.");
  }

  // Convert winding numbers to inclusion values based on operation type.
  VecDH<int> i12(dir12_.size());
  VecDH<int> i21(dir21_.size());
  VecDH<int> i03(w03_.size());
  VecDH<int> i30(w30_.size());
  thrust::transform(dir12_.beginD(), dir12_.endD(), i12.beginD(), c3 * _1);
  thrust::transform(dir21_.beginD(), dir21_.endD(), i21.beginD(), c3 * _1);
  thrust::transform(w03_.beginD(), w03_.endD(), i03.beginD(), c1 + c3 * _1);
  thrust::transform(w30_.beginD(), w30_.endD(), i30.beginD(), c2 + c3 * _1);

  // Calculate some internal indexing vectors
  VecDH<bool> intersectedTriP(inP_.NumTri(), false);
  VecDH<bool> intersectedTriQ(inQ_.NumTri(), false);
  thrust::for_each_n(
      p2q2_.beginDpq(), p2q2_.size(),
      AssignOnes({intersectedTriP.ptrD(), intersectedTriQ.ptrD()}));

  VecDH<int> vP2R(inP_.NumVert() + 1, 0);
  VecDH<int> vQ2R(inQ_.NumVert() + 1, 0);
  thrust::inclusive_scan(i03.beginD(), i03.endD(), vP2R.beginD() + 1, AbsSum());
  thrust::inclusive_scan(i30.beginD(), i30.endD(), vQ2R.beginD() + 1, AbsSum());
  const int nPv = vP2R.H()[inP_.NumVert()];
  const int nQv = vQ2R.H()[inQ_.NumVert()];
  thrust::transform(vQ2R.beginD(), vQ2R.endD(), vQ2R.beginD(), _1 + nPv);

  int n12 = v12_.size();
  int n21 = v21_.size();

  // Create the output Manifold
  Manifold::Impl outR;

  int totalVerts = nPv + nQv + n12 + n21;
  if (totalVerts == 0) return outR;

  outR.vertPos_.resize(totalVerts);
  // Add retained vertices, duplicating for inclusion numbers not in [-1, 1].
  VecDH<int> vertR2PQ(nPv + nQv);
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), i03.beginD(),
                         vP2R.beginD(), inP_.vertPos_.beginD()),
                     inP_.NumVert(),
                     DuplicateVerts({outR.vertPos_.ptrD(), vertR2PQ.ptrD()}));
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), i30.beginD(),
                         vQ2R.beginD(), inQ_.vertPos_.beginD()),
                     inQ_.NumVert(),
                     DuplicateVerts({outR.vertPos_.ptrD(), vertR2PQ.ptrD()}));
  // Add new vertices created from intersections.
  thrust::copy(v12_.beginD(), v12_.endD(), outR.vertPos_.beginD() + nPv + nQv);
  thrust::copy(v21_.beginD(), v21_.endD(),
               outR.vertPos_.beginD() + nPv + nQv + n12);
  // Duplicate retained faces as above.
  AppendRetainedFaces(outR.triVerts_, p1q2_.Copy(0), p2q1_.Copy(0), i03, vP2R,
                      inP_);
  AppendRetainedFaces(outR.triVerts_, p2q1_.Copy(1), p1q2_.Copy(1), i30, vQ2R,
                      inQ_);

  if (kVerbose) {
    std::cout << nPv << " verts from inP, including duplcations" << std::endl;
    std::cout << nQv << " verts from inQ, including duplcations" << std::endl;
    std::cout << n12 << " new verts from edgesP -> facesQ" << std::endl;
    std::cout << n21 << " new verts from facesP -> edgesQ" << std::endl;
  }

  // Build up new polygonal faces from triangle intersections. At this point the
  // calculation switches from parallel to serial.
  std::vector<std::vector<EdgeVerts>> facesP(inP_.NumTri()),
      facesQ(inQ_.NumTri());
  AppendRetainedEdges(facesP, inP_, i03, i12, p1q2_.Get(0), intersectedTriP,
                      outR.vertPos_, vP2R, nPv + nQv);
  AppendRetainedEdges(facesQ, inQ_, i30, i21, p2q1_.Get(1), intersectedTriQ,
                      outR.vertPos_, vQ2R, nPv + nQv + n12);
  AppendNewEdges(facesP, facesQ, inP_.triEdges_, inQ_.triEdges_, p1q2_, p2q1_,
                 p2q2_, i12, i21, nPv + nQv);

  if (PolygonParams().intermediateChecks)
    CheckPreTriangulationManfold(outR.triVerts_, facesP, facesQ);

  // Triangulate the faces and add them to the manifold.
  if (kVerbose) std::cout << "Adding intersected faces of inP" << std::endl;
  AppendIntersectedFaces(outR.triVerts_, outR.vertPos_, facesP, inP_, false);
  if (kVerbose) std::cout << "Adding intersected faces of inQ" << std::endl;
  AppendIntersectedFaces(outR.triVerts_, outR.vertPos_, facesQ, inQ_,
                         op == Manifold::OpType::SUBTRACT);

  // Create the manifold's data structures and verify manifoldness.
  outR.Finish();

  return outR;
}

}  // namespace manifold