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
#include <map>

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
  // if (!(dxL * dxR <= 0 && (dxL != 0 || dxR != 0)))
  //   printf("dxL = %f, dxR = %f\n", dxL, dxR);
  bool useL = fabs(dxL) < fabs(dxR);
  float lambda = (useL ? dxL : dxR) / (pR.x - pL.x);
  if (isnan(lambda)) return glm::vec2(pL.y, pL.z);
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
  // if (!(pL.x == qL.x && pR.x == qR.x))
  //   printf("pL.x = %f, qL.x = %f, pR.x = %f, qR.x = %f\n", pL.x, qL.x, pR.x,
  //          qR.x);
  float dyL = qL.y - pL.y;
  float dyR = qR.y - pR.y;
  // assert dyL and dyR have opposite signs, cannot both be zero
  // if (!(dyL * dyR <= 0 && (dyL != 0 || dyR != 0)))
  //   printf("dyL = %f, dyR = %f\n", dyL, dyR);
  bool useL = fabs(dyL) < fabs(dyR);
  float dx = pR.x - pL.x;
  float lambda = (useL ? dyL : dyR) / (dyL - dyR);
  if (isnan(lambda)) lambda = 0.0f;
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
  // find inP's involved vertices from edges & tris
  VecDH<int> A0(2 * edges.size() + 3 * tris.size() + inP.numLabel_);
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), edges.beginD()),
                     edges.size(),
                     CopyEdgeVerts({A0.ptrD(), inP.edgeVerts_.cptrD()}));
  thrust::for_each_n(
      zip(thrust::make_counting_iterator(0), tris.beginD()), tris.size(),
      CopyTriVerts({A0.ptrD() + 2 * edges.size(), inP.triVerts_.cptrD()}));
  // find one vertex from each connected component of inP (in case it has no
  // intersections)
  thrust::lower_bound(inP.vertLabel_.beginD(), inP.vertLabel_.endD(),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(inP.numLabel_),
                      A0.endD() - inP.numLabel_);

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
  p1q1.SwapPQ();
  thrust::for_each_n(zip(thrust::make_counting_iterator(p1q2.size()),
                         p2q1.beginD(1), p2q1.beginD(0)),
                     p2q1.size(),
                     Fill11({p1q1.ptrDpq(), inP.triEdges_.cptrD()}));
  p1q1.SwapPQ();
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

__host__ __device__ bool Shadows(float p, float q, float dir) {
  return p == q ? dir < 0 : p < q;
}

struct ShadowKernel01 {
  const bool reverse;
  const glm::vec3 *vertPosP;
  const glm::vec3 *vertPosQ;
  const EdgeVertsD *edgeVertsQ;
  const float expandP;
  const glm::vec3 *normalP;

  __host__ __device__ void operator()(thrust::tuple<int &, int, int> inout) {
    int &s01 = thrust::get<0>(inout);
    int vertP = thrust::get<1>(inout);
    int edgeQ = thrust::get<2>(inout);

    int vertQ0 = edgeVertsQ[edgeQ].first;
    int vertQ1 = edgeVertsQ[edgeQ].second;
    float p = vertPosP[vertP].x;
    float q0 = vertPosQ[vertQ0].x;
    float q1 = vertPosQ[vertQ1].x;
    s01 = reverse
              ? Shadows(q0, p, expandP * normalP[vertQ0].x) -
                    Shadows(q1, p, expandP * normalP[vertQ1].x)
              : Shadows(p, q1, expandP * normalP[vertP].x) -
                    Shadows(p, q0, expandP * normalP[vertP].x);
  }
};

struct Kernel01 {
  const bool reverse;
  const glm::vec3 *vertPosP;
  const glm::vec3 *vertPosQ;
  const EdgeVertsD *edgeVertsQ;
  const float expandP;
  const glm::vec3 *normalP;

  __host__ __device__ void operator()(
      thrust::tuple<glm::vec2 &, int &, int, int> inout) {
    glm::vec2 &yz01 = thrust::get<0>(inout);
    int &s01 = thrust::get<1>(inout);
    int vertP = thrust::get<2>(inout);
    int edgeQ = thrust::get<3>(inout);

    int vertQ0 = edgeVertsQ[edgeQ].first;
    int vertQ1 = edgeVertsQ[edgeQ].second;
    glm::vec3 vertPos0 = vertPosQ[vertQ0];
    glm::vec3 vertPos1 = vertPosQ[vertQ1];
    yz01 = Interpolate(vertPos0, vertPos1, vertPosP[vertP].x);
    if (reverse) {
      if (!Shadows(yz01[0], vertPosP[vertP].y, expandP * normalP[vertQ0].y))
        s01 = 0;
    } else {
      if (!Shadows(vertPosP[vertP].y, yz01[0], expandP * normalP[vertP].y))
        s01 = 0;
    }
  }
};

std::tuple<VecDH<int>, VecDH<glm::vec2>> Shadow01(SparseIndices &p0q1,
                                                  const Manifold::Impl &inP,
                                                  const Manifold::Impl &inQ,
                                                  bool reverse, float expandP) {
  VecDH<int> s01(p0q1.size());
  if (reverse) p0q1.SwapPQ();
  auto normalP = reverse ? inQ.vertNormal_.cptrD() : inP.vertNormal_.cptrD();
  thrust::for_each_n(
      zip(s01.beginD(), p0q1.beginD(0), p0q1.beginD(1)), p0q1.size(),
      ShadowKernel01({reverse, inP.vertPos_.cptrD(), inQ.vertPos_.cptrD(),
                      inQ.edgeVerts_.cptrD(), expandP, normalP}));
  size_t size = p0q1.RemoveZeros(s01);
  VecDH<glm::vec2> yz01(size);

  normalP = reverse ? inQ.vertNormal_.cptrD() : inP.vertNormal_.cptrD();
  thrust::for_each_n(
      zip(yz01.beginD(), s01.beginD(), p0q1.beginD(0), p0q1.beginD(1)), size,
      Kernel01({reverse, inP.vertPos_.cptrD(), inQ.vertPos_.cptrD(),
                inQ.edgeVerts_.cptrD(), expandP, normalP}));
  if (reverse) p0q1.SwapPQ();
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
  while (1) {
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

struct Gather11 {
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
  float expandP;
  const glm::vec3 *normalP;

  __host__ __device__ void operator()(
      thrust::tuple<glm::vec4 &, int &, int, int> inout) {
    glm::vec4 &xyzz11 = thrust::get<0>(inout);
    int &s11 = thrust::get<1>(inout);
    const int p1 = thrust::get<2>(inout);
    const int q1 = thrust::get<3>(inout);

    glm::vec3 p2[2], q2[2];
    int k = 0;
    thrust::pair<int, int> key2[2];

    key2[0] = thrust::make_pair(edgeVertsP[p1].first, q1);
    key2[1] = thrust::make_pair(edgeVertsP[p1].second, q1);
    for (int i : {0, 1}) {
      p2[k] = vertPosP[key2[i].first];
      q2[k] = glm::vec3(p2[k].x, BinarySearchByKey(p0q1, yz01, size01, key2[i],
                                                   glm::vec2(0.0f / 0.0f)));
      if (!isnan(q2[k].y)) k++;
    }

    key2[0] = thrust::make_pair(p1, edgeVertsQ[q1].first);
    key2[1] = thrust::make_pair(p1, edgeVertsQ[q1].second);
    for (int i : {0, 1}) {
      if (k > 1) break;
      q2[k] = vertPosQ[key2[i].second];
      p2[k] = glm::vec3(q2[k].x, BinarySearchByKey(p1q0, yz10, size10, key2[i],
                                                   glm::vec2(0.0f / 0.0f)));
      if (!isnan(p2[k].y)) k++;
    }

    // assert two of these four were found
    if (k != 2) printf("k = %d\n", k);

    xyzz11 = Intersect(p2[0], p2[1], q2[0], q2[1]);
    if (!Shadows(xyzz11.z, xyzz11.w, expandP * normalP[edgeVertsP[p1].first].z))
      s11 = 0;
  }
};

std::tuple<VecDH<int>, VecDH<glm::vec4>> Shadow11(
    SparseIndices &p1q1, const Manifold::Impl &inP, const Manifold::Impl &inQ,
    const SparseIndices &p0q1, const VecDH<int> &s01,
    const VecDH<glm::vec2> &yz01, const SparseIndices &p1q0,
    const VecDH<int> &s10, const VecDH<glm::vec2> &yz10, float expandP) {
  VecDH<int> s11(p1q1.size(), 0);

  thrust::for_each_n(zip(s11.beginD(), p1q1.beginD(0), p1q1.beginD(1)),
                     p1q1.size(),
                     Gather11({p0q1.ptrDpq(), s01.ptrD(), p0q1.size(),
                               inP.edgeVerts_.ptrD(), false}));
  thrust::for_each_n(zip(s11.beginD(), p1q1.beginD(1), p1q1.beginD(0)),
                     p1q1.size(),
                     Gather11({p1q0.ptrDpq(), s10.ptrD(), p1q0.size(),
                               inQ.edgeVerts_.ptrD(), true}));

  size_t size = p1q1.RemoveZeros(s11);
  VecDH<glm::vec4> xyzz11(size);

  thrust::for_each_n(
      zip(xyzz11.beginD(), s11.beginD(), p1q1.beginD(0), p1q1.beginD(1)),
      p1q1.size(),
      Kernel11({inP.vertPos_.cptrD(), inQ.vertPos_.cptrD(),
                inP.edgeVerts_.cptrD(), inQ.edgeVerts_.cptrD(), p0q1.ptrDpq(),
                yz01.cptrD(), p0q1.size(), p1q0.ptrDpq(), yz10.cptrD(),
                p1q0.size(), expandP, inP.vertNormal_.cptrD()}));

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
  const float expandP;
  const glm::vec3 *normalP;

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
    if (isnan(yz3[0][0]) + isnan(yz3[1][0]) + isnan(yz3[2][0]) != 1)
      printf("yz0 = %f, yz1 = %f, yz2 = %f\n", yz3[0][0], yz3[1][0], yz3[2][0]);
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
      if (!Shadows(vertPos.z, z02, expandP * normalP[p0].z)) s02 = 0;
    } else {
      if (!Shadows(z02, vertPos.z, expandP * normalP[q2].z)) s02 = 0;
    }
  }
};

std::tuple<VecDH<int>, VecDH<float>> Shadow02(
    const Manifold::Impl &inP, const Manifold::Impl &inQ, const VecDH<int> &s01,
    const SparseIndices &p0q1, const VecDH<glm::vec2> &yz01,
    SparseIndices &p0q2, bool forward, float expandP) {
  VecDH<int> s02(p0q2.size(), 0);

  thrust::for_each_n(
      zip(s02.beginD(), p0q2.beginD(!forward), p0q2.beginD(forward)),
      p0q2.size(), Gather02({p0q1.ptrDpq(), s01.ptrD(), p0q1.size(),
                             inQ.triEdges_.ptrD(), forward}));

  size_t size = p0q2.RemoveZeros(s02);
  VecDH<float> z02(size);

  auto normalP = forward ? inP.vertNormal_.cptrD() : inQ.triNormal_.cptrD();
  thrust::for_each_n(
      zip(z02.beginD(), s02.beginD(), p0q2.beginD(!forward),
          p0q2.beginD(forward)),
      size,
      Kernel02({inP.vertPos_.cptrD(), p0q1.ptrDpq(), yz01.cptrD(), p0q1.size(),
                inQ.triEdges_.cptrD(), forward, expandP, normalP}));

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
      if (k > 1) break;
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
    // assert two of these five were found
    if (k != 2) printf("k = %d\n", k);

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
      p1q2.size(), Kernel12({p0q2.ptrDpq(), z02.ptrD(), p0q2.size(),
                             p1q1.ptrDpq(), xyzz11.ptrD(), p1q1.size(),
                             edgeVertsPtr, triEdgesPtr, vertPosPtr, forward}));
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
  VecDH<int> vertLabels;
  int n_comp = ConnectedComponents(vertLabels, inP.NumVert(), inP.edgeVerts_,
                                   keepEdgesP);
  // flood the w03 values throughout their connected components (they are
  // consistent)
  FloodComponents(w03, vertLabels, n_comp);

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

struct DuplicateVerts {
  glm::vec3 *vertPosR;

  __host__ __device__ void operator()(
      thrust::tuple<int, int, int, glm::vec3> in) {
    int vertP = thrust::get<0>(in);
    int inclusion = abs(thrust::get<1>(in));
    int vertR = thrust::get<2>(in);
    glm::vec3 vertPosP = thrust::get<3>(in);

    for (int i = 0; i < inclusion; ++i) {
      vertPosR[vertR + i] = vertPosP;
    }
  }
};

struct EdgePos {
  int vert;
  float edgePos;
  bool isStart;
};

void AddNewEdgeVerts(
    std::map<int, std::vector<EdgePos>> &edgesP,
    std::map<std::pair<int, int>, std::vector<EdgePos>> &edgesNew,
    const SparseIndices &p1q2, const VecH<int> &i12, const VecH<int> &v12R,
    const VecH<EdgeTrisD> &edgeTrisP, bool forward) {
  // For each edge of P that intersects a face of Q (p1q2), add this vertex to
  // P's corresponding edge vector and to the two new edges, which are
  // intersections between the face of Q and the two faces of P attached to the
  // edge. The direction and duplicity are given by i12, while v12R remaps to
  // the output vert index. When forward is false, all is reversed.
  const VecH<int> &p1 = p1q2.Get(!forward).H();
  const VecH<int> &q2 = p1q2.Get(forward).H();
  for (int i = 0; i < p1q2.size(); ++i) {
    const int edgeP = p1[i];
    const int faceQ = q2[i];
    const int vert = v12R[i];
    const int inclusion = i12[i];

    const auto edgePosP = edgesP.insert({edgeP, {}});

    EdgeTrisD edgePfaces = edgeTrisP[edgeP];
    std::pair<int, int> key = {edgePfaces.right, faceQ};
    if (!forward) std::swap(key.first, key.second);
    const auto edgePosRight = edgesNew.insert({key, {}});

    key = {edgePfaces.left, faceQ};
    if (!forward) std::swap(key.first, key.second);
    const auto edgePosLeft = edgesNew.insert({key, {}});

    EdgePos edgePos = {vert, 0.0f, inclusion < 0};
    EdgePos edgePosRev = edgePos;
    edgePosRev.isStart = !edgePos.isStart;

    for (int j = 0; j < glm::abs(inclusion); ++j) {
      edgePosP.first->second.push_back(edgePos);
      edgePosRight.first->second.push_back(forward ? edgePos : edgePosRev);
      edgePosLeft.first->second.push_back(forward ? edgePosRev : edgePos);
      ++edgePos.vert;
      ++edgePosRev.vert;
    }
  }
}

std::vector<EdgeVerts> PairUp(std::vector<EdgePos> &edgePos, int edge) {
  // Pair start vertices with end vertices to form edges. The choice of pairing
  // is arbitrary for the manifoldness guarantee, but must be ordered to be
  // geometrically valid. If the order does not go start-end-start-end... then
  // the input and output are not geometrically valid and this algorithm becomes
  // a heuristic.
  ALWAYS_ASSERT(edgePos.size() % 2 == 0, logicErr,
                "Non-manifold edge! Not an even number of points.");
  int nEdges = edgePos.size() / 2;
  auto middle = std::partition(edgePos.begin(), edgePos.end(),
                               [](EdgePos x) { return x.isStart; });
  ALWAYS_ASSERT(middle - edgePos.begin() == nEdges, logicErr,
                "Non-manifold edge!");
  auto cmp = [](EdgePos a, EdgePos b) { return a.edgePos < b.edgePos; };
  std::sort(edgePos.begin(), middle, cmp);
  std::sort(middle, edgePos.end(), cmp);
  std::vector<EdgeVerts> edges;
  for (int i = 0; i < nEdges; ++i)
    edges.push_back({edgePos[i].vert, edgePos[i + nEdges].vert, edge});
  return edges;
}

void AppendRetainedEdges(std::map<int, std::vector<EdgeVerts>> &facesP,
                         std::map<int, std::vector<EdgePos>> &edgesP,
                         const Manifold::Impl &inP, const VecH<int> &i03,
                         const VecH<int> &vP2R,
                         const VecH<glm::vec3> &vertPos) {
  // Each edge in the map is partially retained; for each of these, look up
  // their original verts and include them based on their winding number (i03),
  // while remaping them to the output using vP2R. Use the verts position
  // projected along the edge vector to pair them up, then distribute these
  // edges to their faces. Copy any original edges of each face in that are not
  // in the retained edge map.
  const VecH<glm::vec3> &vertPosP = inP.vertPos_.H();
  const VecH<EdgeVertsD> &edgeVertsP = inP.edgeVerts_.H();
  const VecH<EdgeTrisD> &edgeTrisP = inP.edgeTris_.H();

  for (auto &value : edgesP) {
    const int edgeP = value.first;
    std::vector<EdgePos> &edgePosP = value.second;

    const int vStart = edgeVertsP[edgeP].first;
    const int vEnd = edgeVertsP[edgeP].second;
    const glm::vec3 edgeVec = vertPosP[vEnd] - vertPosP[vStart];
    // Fill in the edge positions of the old points.
    for (EdgePos &edge : edgePosP) {
      edge.edgePos = glm::dot(vertPos[edge.vert], edgeVec);
    }

    int inclusion = i03[vStart];
    EdgePos edgePos = {vP2R[vStart], -1.0f / 0.0f, inclusion > 0};
    for (int j = 0; j < glm::abs(inclusion); ++j) {
      edgePosP.push_back(edgePos);
      ++edgePos.vert;
    }

    inclusion = i03[vEnd];
    edgePos = {vP2R[vEnd], 1.0f / 0.0f, inclusion < 0};
    for (int j = 0; j < glm::abs(inclusion); ++j) {
      edgePosP.push_back(edgePos);
      ++edgePos.vert;
    }

    // sort edges into start/end pairs along length
    std::vector<EdgeVerts> edges = PairUp(edgePosP, edgeP);

    // add edges to left face
    const int faceLeft = edgeTrisP[edgeP].left;
    auto result = facesP.insert({faceLeft, edges});
    if (!result.second) {
      auto &vec = result.first->second;
      vec.insert(vec.end(), edges.begin(), edges.end());
    }
    // reverse edges and add to right face
    for (auto &e : edges) std::swap(e.first, e.second);
    const int faceRight = edgeTrisP[edgeP].right;
    result = facesP.insert({faceRight, edges});
    if (!result.second) {
      auto &vec = result.first->second;
      vec.insert(vec.end(), edges.begin(), edges.end());
    }
  }
}

void AppendNewEdges(
    std::map<int, std::vector<EdgeVerts>> &facesP,
    std::map<int, std::vector<EdgeVerts>> &facesQ,
    std::map<std::pair<int, int>, std::vector<EdgePos>> &edgesNew) {
  // Pair up each edge's verts and distribute to faces based on indices in key.
  // Usually only two verts are in each edge, and if not, they are degenerate
  // anyway, so pair arbitrarily without bothering with vertex projections.
  int edgeID = std::numeric_limits<int>::max();
  for (auto &value : edgesNew) {
    const int faceP = value.first.first;
    const int faceQ = value.first.second;
    std::vector<EdgePos> &edgePos = value.second;

    // sort edges into start/end pairs along length
    // Since these are not input edges, their index is undefined.
    std::vector<EdgeVerts> edges = PairUp(edgePos, edgeID--);

    auto result = facesP.insert({faceP, edges});
    if (!result.second) {
      auto &vec = result.first->second;
      vec.insert(vec.end(), edges.begin(), edges.end());
    }
    // reverse edges and add to right face
    for (auto &e : edges) std::swap(e.first, e.second);
    result = facesQ.insert({faceQ, edges});
    if (!result.second) {
      auto &vec = result.first->second;
      vec.insert(vec.end(), edges.begin(), edges.end());
    }
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

void AppendFaces(Manifold::Impl &outR,
                 std::map<int, std::vector<EdgeVerts>> &facesP,
                 const std::map<int, std::vector<EdgePos>> &edgesP,
                 const VecH<int> &i03, const Manifold::Impl &inP,
                 const VecH<int> &vP2R, bool invertNormals) {
  // Proceed through the map, triangulating each face into the result. For each
  // face not included as a map index, copy it from the original mesh,
  // duplicating according to its inclusion number (i03).
  const VecH<glm::ivec3> &triVertsP = inP.triVerts_.H();
  const VecH<TriEdges> &triEdgesP = inP.triEdges_.H();
  const VecH<glm::vec3> &triNormalP = inP.triNormal_.H();
  const VecH<EdgeVertsD> &edgeVertsP = inP.edgeVerts_.H();
  VecH<glm::ivec3> &triVertsR = outR.triVerts_.H();
  VecH<glm::vec3> &triNormalR = outR.triNormal_.H();
  VecH<glm::vec3> &vertPosR = outR.vertPos_.H();

  auto nextIntersectedFace = facesP.begin();
  for (int triP = 0; triP < inP.NumTri(); ++triP) {
    const int faceP = facesP.empty() ? -1 : nextIntersectedFace->first;
    if (faceP != triP) {  // Non-intersecting face
      // Copy triangle from inP
      glm::ivec3 triVerts = triVertsP[triP];
      glm::vec3 normal = triNormalP[triP];
      // Check the inclusion number of a single vertex of a triangle, since
      // non-intersecting triangles must have all identical inclusion numbers.
      int inclusion = i03[triVerts[0]];
      glm::ivec3 outTri(vP2R[triVerts[0]], vP2R[triVerts[1]],
                        vP2R[triVerts[2]]);
      if (inclusion < 0) {
        std::swap(outTri[1], outTri[2]);
        normal *= -1.0f;
      }
      for (int j = 0; j < abs(inclusion); ++j) {
        triVertsR.push_back(outTri + j);
        triNormalR.push_back(normal);
      }
    } else {  // intersecting face
      std::vector<EdgeVerts> &faceEdges = nextIntersectedFace->second;
      if (std::next(nextIntersectedFace) != facesP.end()) ++nextIntersectedFace;

      // Copy in non-intersecting edges of intersected face
      for (int i : {0, 1, 2}) {
        EdgeIdx edge = triEdgesP[faceP][i];
        if (edgesP.find(edge.Idx()) == edgesP.end()) {
          EdgeVertsD oldEdgeVerts = edgeVertsP[edge.Idx()];
          // Non-intersecting edge has the same inclusion number at both ends.
          const int inclusion = i03[oldEdgeVerts.first];
          int vStart = vP2R[oldEdgeVerts.first];
          int vEnd = vP2R[oldEdgeVerts.second];
          if ((inclusion > 0) != (edge.Dir() > 0)) std::swap(vStart, vEnd);
          for (int j = 0; j < std::abs(inclusion); ++j) {
            faceEdges.push_back({vStart + j, vEnd + j, edge.Idx()});
          }
        }
      }

      // Triangulate intersected face
      ALWAYS_ASSERT(faceEdges.size() >= 3, logicErr,
                    "face has less than three edges.");
      const glm::vec3 normal =
          (invertNormals ? -1.0f : 1.0f) * triNormalP[faceP];

      if (faceEdges.size() == 3) {  // Special case to increase performance
        auto tri = faceEdges;
        if (tri[0].second == tri[2].first) std::swap(tri[1], tri[2]);
        ALWAYS_ASSERT(tri[0].second == tri[1].first &&
                          tri[1].second == tri[2].first &&
                          tri[2].second == tri[0].first,
                      runtimeErr, "These 3 edges do not form a triangle!");
        glm::ivec3 triangle(tri[0].first, tri[1].first, tri[2].first);
        triVertsR.push_back(triangle);
        triNormalR.push_back(normal);
      } else {  // General triangulation
        const glm::mat3x2 projection = GetAxisAlignedProjection(normal);
        Polygons polys =
            Assemble(faceEdges, [&vertPosR, &projection](int vert) {
              return projection * vertPosR[vert];
            });
        std::vector<glm::ivec3> newTris;
        try {
          newTris = Triangulate(polys);
        } catch (const runtimeErr &e) {
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
                [&vertPosR](PolyVert v) { return vertPosR[v.idx]; },
                glm::vec3(0.0f),
                [](glm::vec3 a, glm::vec3 b) { return a + b; });
            centroid /= poly.size();
            int newVert = vertPosR.size();
            vertPosR.push_back(centroid);
            newTris.push_back({poly.back().idx, poly.front().idx, newVert});
            for (int j = 1; j < poly.size(); ++j)
              newTris.push_back({poly[j - 1].idx, poly[j].idx, newVert});
          }
        }
        for (auto tri : newTris) {
          triVertsR.push_back(tri);
          triNormalR.push_back(normal);
        }
      }
    }
  }
}
}  // namespace

namespace manifold {

Boolean3::Boolean3(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                   Manifold::OpType op)
    : inP_(inP), inQ_(inQ), expandP_(op == Manifold::OpType::ADD ? 1.0 : -1.0) {
  // Symbolic perturbation:
  // Union -> expand inP
  // Difference, Intersection -> contract inP

  Time t0 = NOW();
  Time t1;
  // Level 3
  // Find edge-triangle overlaps (broad phase)
  p1q2_ = inQ_.EdgeCollisions(inP_);
  p1q2_.Sort();
  if (kVerbose) std::cout << "p1q2 size = " << p1q2_.size() << std::endl;

  p2q1_ = inP_.EdgeCollisions(inQ_);
  p2q1_.SwapPQ();
  p2q1_.Sort();
  if (kVerbose) std::cout << "p2q1 size = " << p2q1_.size() << std::endl;

  // Level 2
  // Find vertices from Level 3 that overlap triangles in XY-projection
  SparseIndices p0q2 = Filter02(inP_, inQ_, p1q2_.Get(0), p2q1_.Get(0));
  p0q2.Sort();
  if (kVerbose) std::cout << "p0q2 size = " << p0q2.size() << std::endl;

  SparseIndices p2q0 = Filter02(inQ_, inP_, p2q1_.Get(1), p1q2_.Get(1));
  p2q0.SwapPQ();
  p2q0.Sort();
  if (kVerbose) std::cout << "p2q0 size = " << p2q0.size() << std::endl;

  // Find involved edge pairs from Level 3
  SparseIndices p1q1 = Filter11(inP_, inQ_, p1q2_, p2q1_);
  if (kVerbose) std::cout << "p1q1 size = " << p1q1.size() << std::endl;

  // Level 1
  // Find involved vertex-edge pairs from Level 2
  SparseIndices p0q1 = Filter01(inP_, inQ_, p0q2, p1q1);
  p0q1.Unique();
  if (kVerbose) std::cout << "p0q1 size = " << p0q1.size() << std::endl;

  p2q0.SwapPQ();
  p1q1.SwapPQ();
  SparseIndices p1q0 = Filter01(inQ_, inP_, p2q0, p1q1);
  p2q0.SwapPQ();
  p1q1.SwapPQ();
  p1q0.SwapPQ();
  p1q0.Unique();
  if (kVerbose) std::cout << "p1q0 size = " << p1q0.size() << std::endl;

  if (kVerbose) {
    std::cout << "Time for Filter";
    t1 = NOW();
    PrintDuration(t1 - t0);
    t0 = t1;
  }

  // Level 1
  // Find X-projections of vertices onto edges, keeping only those that actually
  // fall inside the edge.
  VecDH<int> s01;
  VecDH<glm::vec2> yz01;
  std::tie(s01, yz01) = Shadow01(p0q1, inP, inQ, false, expandP_);
  if (kVerbose) std::cout << "s01 size = " << s01.size() << std::endl;

  VecDH<int> s10;
  VecDH<glm::vec2> yz10;
  std::tie(s10, yz10) = Shadow01(p1q0, inQ, inP, true, expandP_);
  if (kVerbose) std::cout << "s10 size = " << s10.size() << std::endl;

  // Level 2
  // Build up XY-projection intersection of two edges, including the z-value for
  // each edge, keeping only those whose intersection exists.
  VecDH<int> s11;
  VecDH<glm::vec4> xyzz11;
  std::tie(s11, xyzz11) =
      Shadow11(p1q1, inP, inQ, p0q1, s01, yz01, p1q0, s10, yz10, expandP_);
  if (kVerbose) std::cout << "s11 size = " << s11.size() << std::endl;

  // Build up Z-projection of vertices onto triangles, keeping only those that
  // fall inside the triangle.
  VecDH<int> s02;
  VecDH<float> z02;
  std::tie(s02, z02) =
      Shadow02(inP, inQ, s01, p0q1, yz01, p0q2, true, expandP_);
  if (kVerbose) std::cout << "s02 size = " << s02.size() << std::endl;

  VecDH<int> s20;
  VecDH<float> z20;
  std::tie(s20, z20) =
      Shadow02(inQ, inP, s10, p1q0, yz10, p2q0, false, expandP_);
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

  if (kVerbose) {
    std::cout << "Time for Levels 1-3";
    t1 = NOW();
    PrintDuration(t1 - t0);
    t0 = t1;
  }

  // Build up the winding numbers of all vertices. The involved vertices are
  // calculated from Level 2, while the rest are assigned consistently with
  // connected-components flooding.
  w03_ = Winding03(inP, p0q2, s02, p1q2_, false);

  w30_ = Winding03(inQ, p2q0, s20, p2q1_, true);

  // Level 4
  // Record all intersecting triangle pairs.
  p2q2_ = Intersect22(inP_, inQ_, p1q2_, p2q1_, dir12_, dir21_);

  if (kVerbose) {
    std::cout << "Time for rest of first stage";
    t1 = NOW();
    PrintDuration(t1 - t0);
    t0 = t1;
  }
}

Manifold::Impl Boolean3::Result(Manifold::OpType op) const {
  if ((expandP_ > 0) != (op == Manifold::OpType::ADD))
    std::cout << "Warning! Result op type not compatible with constructor op "
                 "type: coplanar faces may have incorrect results."
              << std::endl;
  int c1, c2, c3;
  switch (op) {
    case Manifold::OpType::ADD:
      c1 = 1;
      c2 = 1;
      c3 = -1;
      if (kVerbose) std::cout << "ADD" << std::endl;
      break;
    case Manifold::OpType::SUBTRACT:
      c1 = 1;
      c2 = 0;
      c3 = -1;
      if (kVerbose) std::cout << "SUBTRACT" << std::endl;
      break;
    case Manifold::OpType::INTERSECT:
      c1 = 0;
      c2 = 0;
      c3 = 1;
      if (kVerbose) std::cout << "INTERSECT" << std::endl;
      break;
    default:
      throw std::invalid_argument("invalid enum: OpType.");
  }

  Time t0 = NOW();
  Time t1;

  // Convert winding numbers to inclusion values based on operation type.
  VecDH<int> i12(dir12_.size());
  VecDH<int> i21(dir21_.size());
  VecDH<int> i03(w03_.size());
  VecDH<int> i30(w30_.size());
  thrust::transform(dir12_.beginD(), dir12_.endD(), i12.beginD(), c3 * _1);
  thrust::transform(dir21_.beginD(), dir21_.endD(), i21.beginD(), c3 * _1);
  thrust::transform(w03_.beginD(), w03_.endD(), i03.beginD(), c1 + c3 * _1);
  thrust::transform(w30_.beginD(), w30_.endD(), i30.beginD(), c2 + c3 * _1);

  VecDH<int> vP2R(inP_.NumVert());
  thrust::exclusive_scan(i03.beginD(), i03.endD(), vP2R.beginD(), 0, AbsSum());
  int numVertR = AbsSum()(vP2R.H().back(), i03.H().back());
  const int nPv = numVertR;

  VecDH<int> vQ2R(inQ_.NumVert());
  thrust::exclusive_scan(i30.beginD(), i30.endD(), vQ2R.beginD(), numVertR,
                         AbsSum());
  numVertR = AbsSum()(vQ2R.H().back(), i30.H().back());
  const int nQv = numVertR - nPv;

  VecDH<int> v12R(v12_.size());
  if (v12_.size() > 0) {
    thrust::exclusive_scan(i12.beginD(), i12.endD(), v12R.beginD(), numVertR,
                           AbsSum());
    numVertR = AbsSum()(v12R.H().back(), i12.H().back());
  }
  const int n12 = numVertR - nPv - nQv;

  VecDH<int> v21R(v21_.size());
  if (v21_.size() > 0) {
    thrust::exclusive_scan(i21.beginD(), i21.endD(), v21R.beginD(), numVertR,
                           AbsSum());
    numVertR = AbsSum()(v21R.H().back(), i21.H().back());
  }
  const int n21 = numVertR - nPv - nQv - n12;

  // Create the output Manifold
  Manifold::Impl outR;

  if (numVertR == 0) return outR;

  outR.vertPos_.resize(numVertR);
  // Add vertices, duplicating for inclusion numbers not in [-1, 1].
  // Retained vertices from P and Q:
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), i03.beginD(),
                         vP2R.beginD(), inP_.vertPos_.beginD()),
                     inP_.NumVert(), DuplicateVerts({outR.vertPos_.ptrD()}));
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), i30.beginD(),
                         vQ2R.beginD(), inQ_.vertPos_.beginD()),
                     inQ_.NumVert(), DuplicateVerts({outR.vertPos_.ptrD()}));
  // New vertices created from intersections:
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), i12.beginD(),
                         v12R.beginD(), v12_.beginD()),
                     i12.size(), DuplicateVerts({outR.vertPos_.ptrD()}));
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), i21.beginD(),
                         v21R.beginD(), v21_.beginD()),
                     i21.size(), DuplicateVerts({outR.vertPos_.ptrD()}));

  if (kVerbose) {
    std::cout << nPv << " verts from inP" << std::endl;
    std::cout << nQv << " verts from inQ" << std::endl;
    std::cout << n12 << " new verts from edgesP -> facesQ" << std::endl;
    std::cout << n21 << " new verts from facesP -> edgesQ" << std::endl;
  }

  if (kVerbose) {
    std::cout << "Time for GPU part of result";
    t1 = NOW();
    PrintDuration(t1 - t0);
    t0 = t1;
  }

  // Build up new polygonal faces from triangle intersections. At this point the
  // calculation switches from parallel to serial.

  // This key is the edge index of P or Q. Only includes intersected edges.
  std::map<int, std::vector<EdgePos>> edgesP, edgesQ;
  // This key is the tri index of <P, Q>
  std::map<std::pair<int, int>, std::vector<EdgePos>> edgesNew;

  AddNewEdgeVerts(edgesP, edgesNew, p1q2_, i12.H(), v12R.H(),
                  inP_.edgeTris_.H(), true);
  AddNewEdgeVerts(edgesQ, edgesNew, p2q1_, i21.H(), v21R.H(),
                  inQ_.edgeTris_.H(), false);

  // This key is the tri index of P or Q. Only includes intersected faces.
  std::map<int, std::vector<EdgeVerts>> facesP, facesQ;

  AppendRetainedEdges(facesP, edgesP, inP_, i03.H(), vP2R.H(),
                      outR.vertPos_.H());
  AppendRetainedEdges(facesQ, edgesQ, inQ_, i30.H(), vQ2R.H(),
                      outR.vertPos_.H());
  AppendNewEdges(facesP, facesQ, edgesNew);

  if (kVerbose) {
    std::cout << "Time for CPU part of result";
    t1 = NOW();
    PrintDuration(t1 - t0);
    t0 = t1;
  }

  // Copy retained triangles and triangulate the intersected faces and add them
  // to the manifold.
  if (kVerbose) std::cout << "Adding faces of inP" << std::endl;
  AppendFaces(outR, facesP, edgesP, i03.H(), inP_, vP2R.H(), false);
  if (kVerbose) std::cout << "Adding faces of inQ" << std::endl;
  AppendFaces(outR, facesQ, edgesQ, i30.H(), inQ_, vQ2R.H(),
              op == Manifold::OpType::SUBTRACT);

  // outR.triVerts_.Dump();

  if (kVerbose) {
    std::cout << "Time for triangulation";
    t1 = NOW();
    PrintDuration(t1 - t0);
    t0 = t1;
  }

  // Create the manifold's data structures and verify manifoldness.
  outR.RemoveChaff();
  outR.Finish();

  if (kVerbose) {
    std::cout << "Time for manifold finishing";
    t1 = NOW();
    PrintDuration(t1 - t0);
    t0 = t1;
  }

  return outR;
}

}  // namespace manifold