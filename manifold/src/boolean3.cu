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

#include "boolean3.cuh"
#include "polygon.h"

/**
 * The notation in this file is abbreviated due to the complexity of the
 * functions involved. The key is that the input manifolds are P and Q, while
 * the output is R, and these letters in both upper and lower case refer to
 * these objects. Operations are based on dimensionality: vert: 0, edge: 1,
 * face: 2, solid: 3. X denotes a winding-number type quantity from the source
 * paper of this algorithm, while S is closely related but includes only the
 * subset of X values which "shadow" (are on the correct side of).
 *
 * Nearly everything here are sparse arrays, where for instance each pair in
 * p2q1 refers to a face index of P interacting with a halfedge index of Q.
 * Adjacent arrays like x21 refer to the values of X corresponding to each
 * sparse index pair.
 *
 * Note many functions are designed to work symmetrically, for instance for both
 * p2q1 and p1q2. Inside of these functions P and Q are marked as though the
 * funtion is forwards, but it may include a Boolean "reverse" that indicates P
 * and Q have been swapped.
 */

// TODO: make this runtime configurable for quicker debug
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
  if (dxL * dxR > 0) printf("Not in domain!\n");
  bool useL = fabs(dxL) < fabs(dxR);
  float lambda = (useL ? dxL : dxR) / (pR.x - pL.x);
  if (!isfinite(lambda)) return glm::vec2(pL.y, pL.z);
  glm::vec2 yz;
  yz[0] = (useL ? pL.y : pR.y) + lambda * (pR.y - pL.y);
  yz[1] = (useL ? pL.z : pR.z) + lambda * (pR.z - pL.z);
  return yz;
}

__host__ __device__ glm::vec4 Intersect(const glm::vec3 &pL,
                                        const glm::vec3 &pR,
                                        const glm::vec3 &qL,
                                        const glm::vec3 &qR) {
  float dyL = qL.y - pL.y;
  float dyR = qR.y - pR.y;
  if (dyL * dyR > 0) printf("No intersection!\n");
  bool useL = fabs(dyL) < fabs(dyR);
  float dx = pR.x - pL.x;
  float lambda = (useL ? dyL : dyR) / (dyL - dyR);
  if (!isfinite(lambda)) lambda = 0.0f;
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

struct CopyFaceEdges {
  // x can be either vert or edge (0 or 1).
  thrust::pair<int *, int *> pXq1;
  const Halfedge *halfedgesQ;

  __host__ __device__ void operator()(thrust::tuple<int, int, int> in) {
    int idx = 3 * thrust::get<0>(in);
    const int pX = thrust::get<1>(in);
    const int q2 = thrust::get<2>(in);

    for (const int i : {0, 1, 2}) {
      pXq1.first[idx + i] = pX;
      const int q1 = 3 * q2 + i;
      const Halfedge edge = halfedgesQ[q1];
      pXq1.second[idx + i] = edge.IsForward() ? q1 : edge.pairedHalfedge;
    }
  }
};

SparseIndices Filter11(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                       const SparseIndices &p1q2, const SparseIndices &p2q1) {
  SparseIndices p1q1(3 * p1q2.size() + 3 * p2q1.size());
  thrust::for_each_n(zip(countAt(0), p1q2.beginD(0), p1q2.beginD(1)),
                     p1q2.size(),
                     CopyFaceEdges({p1q1.ptrDpq(), inQ.halfedge_.cptrD()}));

  p1q1.SwapPQ();
  thrust::for_each_n(zip(countAt(p1q2.size()), p2q1.beginD(1), p2q1.beginD(0)),
                     p2q1.size(),
                     CopyFaceEdges({p1q1.ptrDpq(), inP.halfedge_.cptrD()}));
  p1q1.SwapPQ();
  p1q1.Unique();
  return p1q1;
}

struct AbsSum : public thrust::binary_function<int, int, int> {
  __host__ __device__ int operator()(int a, int b) { return abs(a) + abs(b); }
};

__host__ __device__ bool Shadows(float p, float q, float dir) {
  return p == q ? dir < 0 : p < q;
}

__host__ __device__ thrust::pair<int, glm::vec2> Shadow01(
    const int p0, const int q1, const glm::vec3 *vertPosP,
    const glm::vec3 *vertPosQ, const Halfedge *halfedgeQ, const float expandP,
    const glm::vec3 *normalP, const bool reverse) {
  const int q1s = halfedgeQ[q1].startVert;
  const int q1e = halfedgeQ[q1].endVert;
  const float p0x = vertPosP[p0].x;
  const float q1sx = vertPosQ[q1s].x;
  const float q1ex = vertPosQ[q1e].x;
  int s01 = reverse ? Shadows(q1sx, p0x, expandP * normalP[q1s].x) -
                          Shadows(q1ex, p0x, expandP * normalP[q1e].x)
                    : Shadows(p0x, q1ex, expandP * normalP[p0].x) -
                          Shadows(p0x, q1sx, expandP * normalP[p0].x);
  glm::vec2 yz01(0.0f / 0.0f);

  if (s01 != 0) {
    yz01 = Interpolate(vertPosQ[q1s], vertPosQ[q1e], vertPosP[p0].x);
    if (reverse) {
      glm::vec3 diff = vertPosQ[q1s] - vertPosP[p0];
      const float start2 = glm::dot(diff, diff);
      diff = vertPosQ[q1e] - vertPosP[p0];
      const float end2 = glm::dot(diff, diff);
      const float dir = start2 < end2 ? normalP[q1s].y : normalP[q1e].y;
      if (!Shadows(yz01[0], vertPosP[p0].y, expandP * dir)) s01 = 0;
    } else {
      if (!Shadows(vertPosP[p0].y, yz01[0], expandP * normalP[p0].y)) s01 = 0;
    }
  }
  return thrust::make_pair(s01, yz01);
}

__host__ __device__ int BinarySearch(
    const thrust::pair<const int *, const int *> keys, const int size,
    const thrust::pair<int, int> key) {
  if (size <= 0) return -1;
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
    return m;
  else
    return -1;
}

struct Kernel11 {
  const glm::vec3 *vertPosP;
  const glm::vec3 *vertPosQ;
  const Halfedge *halfedgeP;
  const Halfedge *halfedgeQ;
  float expandP;
  const glm::vec3 *normalP;

  __host__ __device__ void operator()(
      thrust::tuple<glm::vec4 &, int &, int, int> inout) {
    glm::vec4 &xyzz11 = thrust::get<0>(inout);
    int &s11 = thrust::get<1>(inout);
    const int p1 = thrust::get<2>(inout);
    const int q1 = thrust::get<3>(inout);

    // For pRL[k], qRL[k], k==0 is the left and k==1 is the right.
    int k = 0;
    glm::vec3 pRL[2], qRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows;
    s11 = 0;

    const int p0[2] = {halfedgeP[p1].startVert, halfedgeP[p1].endVert};
    for (int i : {0, 1}) {
      const auto syz01 = Shadow01(p0[i], q1, vertPosP, vertPosQ, halfedgeQ,
                                  expandP, normalP, false);
      const int s01 = syz01.first;
      const glm::vec2 yz01 = syz01.second;
      // If the value is NaN, then these do not overlap.
      if (isfinite(yz01[0])) {
        s11 += s01 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          pRL[k] = vertPosP[p0[i]];
          qRL[k] = glm::vec3(pRL[k].x, yz01);
          ++k;
        }
      }
    }

    const int q0[2] = {halfedgeQ[q1].startVert, halfedgeQ[q1].endVert};
    for (int i : {0, 1}) {
      const auto syz10 = Shadow01(q0[i], p1, vertPosQ, vertPosP, halfedgeP,
                                  expandP, normalP, true);
      const int s10 = syz10.first;
      const glm::vec2 yz10 = syz10.second;
      // If the value is NaN, then these do not overlap.
      if (isfinite(yz10[0])) {
        s11 += s10 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s10 != 0) != shadows)) {
          shadows = s10 != 0;
          qRL[k] = vertPosQ[q0[i]];
          pRL[k] = glm::vec3(qRL[k].x, yz10);
          ++k;
        }
      }
    }

    if (s11 == 0) {  // No intersection
      xyzz11 = glm::vec4(0.0f / 0.0f);
    } else {
      // Assert left and right were both found
      if (k != 2) {
        printf("k = %d\n", k);
      }

      xyzz11 = Intersect(pRL[0], pRL[1], qRL[0], qRL[1]);

      const int p1s = halfedgeP[p1].startVert;
      const int p1e = halfedgeP[p1].endVert;
      glm::vec3 diff = vertPosP[p1s] - glm::vec3(xyzz11);
      const float start2 = glm::dot(diff, diff);
      diff = vertPosP[p1e] - glm::vec3(xyzz11);
      const float end2 = glm::dot(diff, diff);
      const float dir = start2 < end2 ? normalP[p1s].z : normalP[p1e].z;

      if (!Shadows(xyzz11.z, xyzz11.w, expandP * dir)) s11 = 0;
    }
  }
};

std::tuple<VecDH<int>, VecDH<glm::vec4>> Shadow11(SparseIndices &p1q1,
                                                  const Manifold::Impl &inP,
                                                  const Manifold::Impl &inQ,
                                                  float expandP) {
  VecDH<int> s11(p1q1.size());
  VecDH<glm::vec4> xyzz11(p1q1.size());

  thrust::for_each_n(
      zip(xyzz11.beginD(), s11.beginD(), p1q1.beginD(0), p1q1.beginD(1)),
      p1q1.size(),
      Kernel11({inP.vertPos_.cptrD(), inQ.vertPos_.cptrD(),
                inP.halfedge_.cptrD(), inQ.halfedge_.cptrD(), expandP,
                inP.vertNormal_.cptrD()}));

  p1q1.KeepFinite(xyzz11, s11);

  return std::make_tuple(s11, xyzz11);
};

struct Kernel02 {
  const glm::vec3 *vertPosP;
  const Halfedge *halfedgeQ;
  const glm::vec3 *vertPosQ;
  const bool forward;
  const float expandP;
  const glm::vec3 *vertNormalP;
  const glm::vec3 *faceNormalP;

  __host__ __device__ void operator()(
      thrust::tuple<int &, float &, int, int> inout) {
    int &s02 = thrust::get<0>(inout);
    float &z02 = thrust::get<1>(inout);
    const int p0 = thrust::get<2>(inout);
    const int q2 = thrust::get<3>(inout);

    // For yzzLR[k], k==0 is the left and k==1 is the right.
    int k = 0;
    glm::vec3 yzzRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows;
    int closestVert;
    float minMetric = 1.0f / 0.0f;
    s02 = 0;

    const glm::vec3 posP = vertPosP[p0];
    for (const int i : {0, 1, 2}) {
      const int q1 = 3 * q2 + i;
      const Halfedge edge = halfedgeQ[q1];
      const int q1F = edge.IsForward() ? q1 : edge.pairedHalfedge;

      if (!forward) {
        const int qVert = halfedgeQ[q1F].startVert;
        const glm::vec3 diff = posP - vertPosQ[qVert];
        const float metric = glm::dot(diff, diff);
        if (metric < minMetric) {
          minMetric = metric;
          closestVert = qVert;
        }
      }

      const auto syz01 = Shadow01(p0, q1F, vertPosP, vertPosQ, halfedgeQ,
                                  expandP, vertNormalP, !forward);
      const int s01 = syz01.first;
      const glm::vec2 yz01 = syz01.second;
      // If the value is NaN, then these do not overlap.
      if (isfinite(yz01[0])) {
        s02 += s01 * (forward == edge.IsForward() ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          yzzRL[k++] = glm::vec3(yz01[0], yz01[1], yz01[1]);
        }
      }
    }

    if (s02 == 0) {  // No intersection
      z02 = 0.0f / 0.0f;
    } else {
      // Assert left and right were both found
      if (k != 2) {
        printf("k = %d\n", k);
      }

      glm::vec3 vertPos = vertPosP[p0];
      z02 = Interpolate(yzzRL[0], yzzRL[1], vertPos.y)[1];
      if (forward) {
        if (!Shadows(vertPos.z, z02, expandP * vertNormalP[p0].z)) s02 = 0;
      } else {
        if (!Shadows(z02, vertPos.z, expandP * vertNormalP[closestVert].z))
          s02 = 0;
      }
    }
  }
};

std::tuple<VecDH<int>, VecDH<float>> Shadow02(const Manifold::Impl &inP,
                                              const Manifold::Impl &inQ,
                                              SparseIndices &p0q2, bool forward,
                                              float expandP) {
  VecDH<int> s02(p0q2.size());
  VecDH<float> z02(p0q2.size());

  auto vertNormalP =
      forward ? inP.vertNormal_.cptrD() : inQ.vertNormal_.cptrD();
  auto faceNormalP =
      forward ? inP.faceNormal_.cptrD() : inQ.faceNormal_.cptrD();
  thrust::for_each_n(zip(s02.beginD(), z02.beginD(), p0q2.beginD(!forward),
                         p0q2.beginD(forward)),
                     p0q2.size(),
                     Kernel02({inP.vertPos_.cptrD(), inQ.halfedge_.cptrD(),
                               inQ.vertPos_.cptrD(), forward, expandP,
                               vertNormalP, faceNormalP}));

  p0q2.KeepFinite(z02, s02);

  return std::make_tuple(s02, z02);
};

struct Kernel12 {
  const thrust::pair<const int *, const int *> p0q2;
  const int *s02;
  const float *z02;
  const int size02;
  const thrust::pair<const int *, const int *> p1q1;
  const int *s11;
  const glm::vec4 *xyzz11;
  const int size11;
  const Halfedge *halfedgesP;
  const Halfedge *halfedgesQ;
  const glm::vec3 *vertPosP;
  const bool forward;

  __host__ __device__ void operator()(
      thrust::tuple<int &, glm::vec3 &, int, int> inout) {
    int &x12 = thrust::get<0>(inout);
    glm::vec3 &v12 = thrust::get<1>(inout);
    const int p1 = thrust::get<2>(inout);
    const int q2 = thrust::get<3>(inout);

    // For xzyLR-[k], k==0 is the left and k==1 is the right.
    int k = 0;
    glm::vec3 xzyLR0[2];
    glm::vec3 xzyLR1[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows;
    x12 = 0;

    const Halfedge edge = halfedgesP[p1];

    for (int vert : {edge.startVert, edge.endVert}) {
      const auto key =
          forward ? thrust::make_pair(vert, q2) : thrust::make_pair(q2, vert);
      const int idx = BinarySearch(p0q2, size02, key);
      if (idx != -1) {
        const int s = s02[idx];
        x12 += s * ((vert == edge.startVert) == forward ? 1 : -1);
        if (k < 2 && (k == 0 || (s != 0) != shadows)) {
          shadows = s != 0;
          xzyLR0[k] = vertPosP[vert];
          thrust::swap(xzyLR0[k].y, xzyLR0[k].z);
          xzyLR1[k] = xzyLR0[k];
          xzyLR1[k][1] = z02[idx];
          k++;
        }
      }
    }

    for (const int i : {0, 1, 2}) {
      const int q1 = 3 * q2 + i;
      const Halfedge edge = halfedgesQ[q1];
      const int q1F = edge.IsForward() ? q1 : edge.pairedHalfedge;
      const auto key =
          forward ? thrust::make_pair(p1, q1F) : thrust::make_pair(q1F, p1);
      const int idx = BinarySearch(p1q1, size11, key);
      if (idx != -1) {  // s is implicitly zero for anything not found
        const int s = s11[idx];
        x12 -= s * (edge.IsForward() ? 1 : -1);
        if (k < 2 && (k == 0 || (s != 0) != shadows)) {
          shadows = s != 0;
          const glm::vec4 xyzz = xyzz11[idx];
          xzyLR0[k][0] = xyzz.x;
          xzyLR0[k][1] = xyzz.z;
          xzyLR0[k][2] = xyzz.y;
          xzyLR1[k] = xzyLR0[k];
          xzyLR1[k][1] = xyzz.w;
          if (!forward) thrust::swap(xzyLR0[k][1], xzyLR1[k][1]);
          k++;
        }
      }
    }

    if (x12 == 0) {  // No intersection
      v12 = glm::vec3(0.0f / 0.0f);
    } else {
      // Assert left and right were both found
      if (k != 2) {
        printf("k = %d\n", k);
      }
      const glm::vec4 xzyy =
          Intersect(xzyLR0[0], xzyLR0[1], xzyLR1[0], xzyLR1[1]);
      v12.x = xzyy[0];
      v12.y = xzyy[2];
      v12.z = xzyy[1];
    }
  }
};

std::tuple<VecDH<int>, VecDH<glm::vec3>> Intersect12(
    const Manifold::Impl &inP, const Manifold::Impl &inQ, const VecDH<int> &s02,
    const SparseIndices &p0q2, const VecDH<int> &s11, const SparseIndices &p1q1,
    const VecDH<float> &z02, const VecDH<glm::vec4> &xyzz11,
    SparseIndices &p1q2, bool forward) {
  VecDH<int> x12(p1q2.size());
  VecDH<glm::vec3> v12(p1q2.size());

  thrust::for_each_n(
      zip(x12.beginD(), v12.beginD(), p1q2.beginD(!forward),
          p1q2.beginD(forward)),
      p1q2.size(),
      Kernel12({p0q2.ptrDpq(), s02.ptrD(), z02.cptrD(), p0q2.size(),
                p1q1.ptrDpq(), s11.ptrD(), xyzz11.cptrD(), p1q1.size(),
                inP.halfedge_.cptrD(), inQ.halfedge_.cptrD(),
                inP.vertPos_.cptrD(), forward}));

  p1q2.KeepFinite(v12, x12);

  return std::make_tuple(x12, v12);
};

VecDH<int> Winding03(const Manifold::Impl &inP, SparseIndices &p0q2,
                     VecDH<int> &s02, const SparseIndices &p1q2, bool reverse) {
  // verts that are not shadowed (not in p0q2) have winding number zero.
  VecDH<int> w03(inP.NumVert(), 0);

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

  if (reverse)
    thrust::transform(w03.beginD(), w03.endD(), w03.beginD(),
                      thrust::negate<int>());
  return w03;
};

struct DuplicateVerts {
  glm::vec3 *vertPosR;

  __host__ __device__ void operator()(thrust::tuple<int, int, glm::vec3> in) {
    int inclusion = abs(thrust::get<0>(in));
    int vertR = thrust::get<1>(in);
    glm::vec3 vertPosP = thrust::get<2>(in);

    for (int i = 0; i < inclusion; ++i) {
      vertPosR[vertR + i] = vertPosP;
    }
  }
};

__host__ __device__ int AtomicAddInt(int &target, int add) {
#ifdef __CUDA_ARCH__
  return atomicAdd(&target, add);
#else
  int out;
#pragma omp atomic capture
  {
    out = target;
    target += add;
  }
  return out;
#endif
}

struct CountVerts {
  int *count;
  const int *inclusion;

  __host__ __device__ void operator()(const Halfedge &edge) {
    AtomicAddInt(count[edge.face], glm::abs(inclusion[edge.startVert]));
  }
};

struct CountNewVerts {
  int *countP;
  int *countQ;
  const Halfedge *halfedges;

  __host__ __device__ void operator()(thrust::tuple<int, int, int> in) {
    int edgeP = thrust::get<0>(in);
    int faceQ = thrust::get<1>(in);
    int inclusion = glm::abs(thrust::get<2>(in));

    AtomicAddInt(countQ[faceQ], inclusion);
    const Halfedge half = halfedges[edgeP];
    AtomicAddInt(countP[half.face], inclusion);
    AtomicAddInt(countP[halfedges[half.pairedHalfedge].face], inclusion);
  }
};

struct NotZero : public thrust::unary_function<int, int> {
  __host__ __device__ int operator()(int x) const { return x > 0 ? 1 : 0; }
};

std::tuple<VecDH<int>, VecDH<int>> SizeOutput(
    Manifold::Impl &outR, const Manifold::Impl &inP, const Manifold::Impl &inQ,
    const VecDH<int> &i03, const VecDH<int> &i30, const VecDH<int> &i12,
    const VecDH<int> &i21, const SparseIndices &p1q2, const SparseIndices &p2q1,
    bool invertQ) {
  VecDH<int> sidesPerFacePQ(inP.NumTri() + inQ.NumTri());
  auto sidesPerFaceP = sidesPerFacePQ.ptrD();
  auto sidesPerFaceQ = sidesPerFacePQ.ptrD() + inP.NumTri();

  thrust::for_each(inP.halfedge_.beginD(), inP.halfedge_.endD(),
                   CountVerts({sidesPerFaceP, i03.cptrD()}));
  thrust::for_each(inQ.halfedge_.beginD(), inQ.halfedge_.endD(),
                   CountVerts({sidesPerFaceQ, i30.cptrD()}));
  thrust::for_each_n(
      zip(p1q2.beginD(0), p1q2.beginD(1), i12.beginD()), i12.size(),
      CountNewVerts({sidesPerFaceP, sidesPerFaceQ, inP.halfedge_.cptrD()}));
  thrust::for_each_n(
      zip(p2q1.beginD(1), p2q1.beginD(0), i21.beginD()), i21.size(),
      CountNewVerts({sidesPerFaceQ, sidesPerFaceP, inQ.halfedge_.cptrD()}));

  VecDH<int> facePQ2R(inP.NumTri() + inQ.NumTri() + 1);
  auto keepFace =
      thrust::make_transform_iterator(sidesPerFacePQ.beginD(), NotZero());
  thrust::inclusive_scan(keepFace, keepFace + sidesPerFacePQ.size(),
                         facePQ2R.beginD() + 1);
  int numFaceR = facePQ2R.H().back();
  facePQ2R.resize(inP.NumTri() + inQ.NumTri());

  outR.faceNormal_.resize(numFaceR);
  auto next = thrust::copy_if(inP.faceNormal_.beginD(), inP.faceNormal_.endD(),
                              keepFace, outR.faceNormal_.beginD(),
                              thrust::identity<bool>());
  if (invertQ) {
    auto start = thrust::make_transform_iterator(inQ.faceNormal_.beginD(),
                                                 thrust::negate<glm::vec3>());
    auto end = thrust::make_transform_iterator(inQ.faceNormal_.endD(),
                                               thrust::negate<glm::vec3>());
    thrust::copy_if(start, end, keepFace + inP.NumTri(), next,
                    thrust::identity<bool>());
  } else {
    thrust::copy_if(inQ.faceNormal_.beginD(), inQ.faceNormal_.endD(),
                    keepFace + inP.NumTri(), next, thrust::identity<bool>());
  }

  auto newEnd =
      thrust::remove(sidesPerFacePQ.beginD(), sidesPerFacePQ.endD(), 0);
  VecDH<int> faceEdge(newEnd - sidesPerFacePQ.beginD() + 1);
  thrust::inclusive_scan(sidesPerFacePQ.beginD(), newEnd,
                         faceEdge.beginD() + 1);
  outR.halfedge_.resize(faceEdge.H().back());

  return std::make_tuple(faceEdge, facePQ2R);
}

struct DuplicateHalfedges {
  Halfedge *halfedgesR;
  int *facePtr;
  const Halfedge *halfedgesP;
  const int *i03;
  const int *vP2R;
  const int *faceP2R;

  __host__ __device__ void operator()(thrust::tuple<bool, Halfedge> in) {
    if (!thrust::get<0>(in)) return;
    Halfedge halfedge = thrust::get<1>(in);
    if (!halfedge.IsForward()) return;

    const int inclusion = i03[halfedge.startVert];
    if (inclusion == 0) return;
    if (inclusion < 0) {  // reverse
      int tmp = halfedge.startVert;
      halfedge.startVert = halfedge.endVert;
      halfedge.endVert = tmp;
    }
    halfedge.startVert = vP2R[halfedge.startVert];
    halfedge.endVert = vP2R[halfedge.endVert];
    halfedge.face = faceP2R[halfedge.face];
    int faceRight = faceP2R[halfedgesP[halfedge.pairedHalfedge].face];

    for (int i = 0; i < glm::abs(inclusion); ++i) {
      int forwardIdx = AtomicAddInt(facePtr[halfedge.face], 1);
      int backwardIdx = AtomicAddInt(facePtr[faceRight], 1);
      halfedge.pairedHalfedge = backwardIdx;

      halfedgesR[forwardIdx] = halfedge;
      halfedgesR[backwardIdx] = {halfedge.endVert, halfedge.startVert,
                                 forwardIdx, faceRight};

      ++halfedge.startVert;
      ++halfedge.endVert;
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
    const VecH<Halfedge> &halfedgeP, bool forward) {
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

    auto &edgePosP = edgesP[edgeP];

    Halfedge halfedge = halfedgeP[edgeP];
    std::pair<int, int> key = {halfedgeP[halfedge.pairedHalfedge].face, faceQ};
    if (!forward) std::swap(key.first, key.second);
    auto &edgePosRight = edgesNew[key];

    key = {halfedge.face, faceQ};
    if (!forward) std::swap(key.first, key.second);
    auto &edgePosLeft = edgesNew[key];

    EdgePos edgePos = {vert, 0.0f, inclusion < 0};
    EdgePos edgePosRev = edgePos;
    edgePosRev.isStart = !edgePos.isStart;

    for (int j = 0; j < glm::abs(inclusion); ++j) {
      edgePosP.push_back(edgePos);
      edgePosRight.push_back(forward ? edgePos : edgePosRev);
      edgePosLeft.push_back(forward ? edgePosRev : edgePos);
      ++edgePos.vert;
      ++edgePosRev.vert;
    }
  }
}

std::vector<Halfedge> PairUp(std::vector<EdgePos> &edgePos) {
  // Pair start vertices with end vertices to form edges. The choice of pairing
  // is arbitrary for the manifoldness guarantee, but must be ordered to be
  // geometrically valid. If the order does not go start-end-start-end... then
  // the input and output are not geometrically valid and this algorithm becomes
  // a heuristic.
  ALWAYS_ASSERT(edgePos.size() % 2 == 0, topologyErr,
                "Non-manifold edge! Not an even number of points.");
  int nEdges = edgePos.size() / 2;
  auto middle = std::partition(edgePos.begin(), edgePos.end(),
                               [](EdgePos x) { return x.isStart; });
  ALWAYS_ASSERT(middle - edgePos.begin() == nEdges, topologyErr,
                "Non-manifold edge!");
  auto cmp = [](EdgePos a, EdgePos b) { return a.edgePos < b.edgePos; };
  std::sort(edgePos.begin(), middle, cmp);
  std::sort(middle, edgePos.end(), cmp);
  std::vector<Halfedge> edges;
  for (int i = 0; i < nEdges; ++i)
    edges.push_back({edgePos[i].vert, edgePos[i + nEdges].vert, -1, -1});
  return edges;
}

void AppendPartialEdges(
    Manifold::Impl &outR, VecH<bool> &wholeHalfedgeP, VecH<int> &facePtrR,
    std::map<int, std::vector<EdgePos>> &edgesP, const Manifold::Impl &inP,
    const VecH<int> &i03, const VecH<int> &vP2R,
    const thrust::host_vector<int>::const_iterator faceP2R) {
  // Each edge in the map is partially retained; for each of these, look up
  // their original verts and include them based on their winding number (i03),
  // while remaping them to the output using vP2R. Use the verts position
  // projected along the edge vector to pair them up, then distribute these
  // edges to their faces.
  VecH<Halfedge> &halfedgeR = outR.halfedge_.H();
  const VecH<glm::vec3> &vertPosP = inP.vertPos_.H();
  const VecH<Halfedge> &halfedgeP = inP.halfedge_.H();

  for (auto &value : edgesP) {
    const int edgeP = value.first;
    std::vector<EdgePos> &edgePosP = value.second;

    const Halfedge &halfedge = halfedgeP[edgeP];
    wholeHalfedgeP[edgeP] = false;
    wholeHalfedgeP[halfedge.pairedHalfedge] = false;

    const int vStart = halfedge.startVert;
    const int vEnd = halfedge.endVert;
    const glm::vec3 edgeVec = vertPosP[vEnd] - vertPosP[vStart];
    // Fill in the edge positions of the old points.
    for (EdgePos &edge : edgePosP) {
      edge.edgePos = glm::dot(outR.vertPos_.H()[edge.vert], edgeVec);
    }

    int inclusion = i03[vStart];
    EdgePos edgePos = {vP2R[vStart],
                       glm::dot(outR.vertPos_.H()[vP2R[vStart]], edgeVec),
                       inclusion > 0};
    for (int j = 0; j < glm::abs(inclusion); ++j) {
      edgePosP.push_back(edgePos);
      ++edgePos.vert;
    }

    inclusion = i03[vEnd];
    edgePos = {vP2R[vEnd], glm::dot(outR.vertPos_.H()[vP2R[vEnd]], edgeVec),
               inclusion < 0};
    for (int j = 0; j < glm::abs(inclusion); ++j) {
      edgePosP.push_back(edgePos);
      ++edgePos.vert;
    }

    // sort edges into start/end pairs along length
    std::vector<Halfedge> edges = PairUp(edgePosP);

    // add halfedges to result
    const int faceLeft = faceP2R[halfedge.face];
    const int faceRight = faceP2R[halfedgeP[halfedge.pairedHalfedge].face];
    for (Halfedge e : edges) {
      const int forwardEdge = facePtrR[faceLeft]++;
      const int backwardEdge = facePtrR[faceRight]++;

      e.face = faceLeft;
      e.pairedHalfedge = backwardEdge;
      halfedgeR[forwardEdge] = e;

      std::swap(e.startVert, e.endVert);
      e.face = faceRight;
      e.pairedHalfedge = forwardEdge;
      halfedgeR[backwardEdge] = e;
    }
  }
}

void AppendNewEdges(
    Manifold::Impl &outR, VecH<int> &facePtrR,
    std::map<std::pair<int, int>, std::vector<EdgePos>> &edgesNew,
    const VecH<int> &facePQ2R, const int numFaceP) {
  // Pair up each edge's verts and distribute to faces based on indices in key.
  VecH<Halfedge> &halfedgeR = outR.halfedge_.H();
  VecH<glm::vec3> &vertPosR = outR.vertPos_.H();

  for (auto &value : edgesNew) {
    const int faceP = value.first.first;
    const int faceQ = value.first.second;
    std::vector<EdgePos> &edgePos = value.second;

    Box bbox;
    for (auto edge : edgePos) {
      bbox.Union(vertPosR[edge.vert]);
    }
    const glm::vec3 size = bbox.Size();
    // Order the points along their longest dimension.
    const int i = (size.x > size.y && size.x > size.z) ? 0
                  : size.y > size.z                    ? 1
                                                       : 2;
    for (auto &edge : edgePos) {
      edge.edgePos = vertPosR[edge.vert][i];
    }

    // sort edges into start/end pairs along length.
    std::vector<Halfedge> edges = PairUp(edgePos);

    // add halfedges to result
    const int faceLeft = facePQ2R[faceP];
    const int faceRight = facePQ2R[numFaceP + faceQ];
    for (Halfedge e : edges) {
      const int forwardEdge = facePtrR[faceLeft]++;
      const int backwardEdge = facePtrR[faceRight]++;

      e.face = faceLeft;
      e.pairedHalfedge = backwardEdge;
      halfedgeR[forwardEdge] = e;

      std::swap(e.startVert, e.endVert);
      e.face = faceRight;
      e.pairedHalfedge = forwardEdge;
      halfedgeR[backwardEdge] = e;
    }
  }
}

void AppendWholeEdges(Manifold::Impl &outR, VecDH<int> &facePtrR,
                      const Manifold::Impl &inP, VecDH<bool> wholeHalfedgeP,
                      const VecDH<int> &i03, const VecDH<int> &vP2R,
                      const int *faceP2R) {
  thrust::for_each_n(zip(wholeHalfedgeP.beginD(), inP.halfedge_.beginD()),
                     inP.halfedge_.size(),
                     DuplicateHalfedges({outR.halfedge_.ptrD(), facePtrR.ptrD(),
                                         inP.halfedge_.cptrD(), i03.cptrD(),
                                         vP2R.cptrD(), faceP2R}));
}
}  // namespace

namespace manifold {
Boolean3::Boolean3(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                   Manifold::OpType op)
    : inP_(inP), inQ_(inQ), expandP_(op == Manifold::OpType::ADD ? 1.0 : -1.0) {
  // Symbolic perturbation:
  // Union -> expand inP
  // Difference, Intersection -> contract inP

  Timer filter;
  filter.Start();

  if (inP.IsEmpty() || inQ.IsEmpty() || !inP.bBox_.DoesOverlap(inQ.bBox_)) {
    if (kVerbose) std::cout << "No overlap, early out" << std::endl;
    w03_.resize(inP.NumVert(), 0);
    w30_.resize(inQ.NumVert(), 0);
    return;
  }

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
  // Find vertices that overlap faces in XY-projection
  SparseIndices p0q2 = inQ.VertexCollisionsZ(inP.vertPos_);
  p0q2.Sort();
  if (kVerbose) std::cout << "p0q2 size = " << p0q2.size() << std::endl;

  SparseIndices p2q0 = inP.VertexCollisionsZ(inQ.vertPos_);
  p2q0.SwapPQ();
  p2q0.Sort();
  if (kVerbose) std::cout << "p2q0 size = " << p2q0.size() << std::endl;

  // Find involved edge pairs from Level 3
  SparseIndices p1q1 = Filter11(inP_, inQ_, p1q2_, p2q1_);
  if (kVerbose) std::cout << "p1q1 size = " << p1q1.size() << std::endl;

  filter.Stop();
  Timer levels;
  levels.Start();

  // Level 2
  // Build up XY-projection intersection of two edges, including the z-value for
  // each edge, keeping only those whose intersection exists.
  VecDH<int> s11;
  VecDH<glm::vec4> xyzz11;
  std::tie(s11, xyzz11) = Shadow11(p1q1, inP, inQ, expandP_);
  if (kVerbose) std::cout << "s11 size = " << s11.size() << std::endl;

  // Build up Z-projection of vertices onto triangles, keeping only those that
  // fall inside the triangle.
  VecDH<int> s02;
  VecDH<float> z02;
  std::tie(s02, z02) = Shadow02(inP, inQ, p0q2, true, expandP_);
  if (kVerbose) std::cout << "s02 size = " << s02.size() << std::endl;

  VecDH<int> s20;
  VecDH<float> z20;
  std::tie(s20, z20) = Shadow02(inQ, inP, p2q0, false, expandP_);
  if (kVerbose) std::cout << "s20 size = " << s20.size() << std::endl;

  // Level 3
  // Build up the intersection of the edges and triangles, keeping only those
  // that intersect, and record the direction the edge is passing through the
  // triangle.
  std::tie(x12_, v12_) =
      Intersect12(inP, inQ, s02, p0q2, s11, p1q1, z02, xyzz11, p1q2_, true);
  if (kVerbose) std::cout << "x12 size = " << x12_.size() << std::endl;

  std::tie(x21_, v21_) =
      Intersect12(inQ, inP, s20, p2q0, s11, p1q1, z20, xyzz11, p2q1_, false);
  if (kVerbose) std::cout << "x21 size = " << x21_.size() << std::endl;

  // Sum up the winding numbers of all vertices.
  w03_ = Winding03(inP, p0q2, s02, p1q2_, false);

  w30_ = Winding03(inQ, p2q0, s20, p2q1_, true);

  levels.Stop();

  if (kVerbose) {
    filter.Print("Filter");
    levels.Print("Levels 1-3");
    MemUsage();
  }
}

Manifold::Impl Boolean3::Result(Manifold::OpType op) const {
  Timer assemble;
  assemble.Start();

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

  if (w03_.size() == 0) {
    if (w30_.size() != 0 && op == Manifold::OpType::ADD) {
      return inQ_;
    }
    return Manifold::Impl();
  } else if (w30_.size() == 0) {
    if (op == Manifold::OpType::INTERSECT) {
      return Manifold::Impl();
    }
    return inP_;
  }

  // Convert winding numbers to inclusion values based on operation type.
  VecDH<int> i12(x12_.size());
  VecDH<int> i21(x21_.size());
  VecDH<int> i03(w03_.size());
  VecDH<int> i30(w30_.size());
  thrust::transform(x12_.beginD(), x12_.endD(), i12.beginD(), c3 * _1);
  thrust::transform(x21_.beginD(), x21_.endD(), i21.beginD(), c3 * _1);
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
  thrust::for_each_n(zip(i03.beginD(), vP2R.beginD(), inP_.vertPos_.beginD()),
                     inP_.NumVert(), DuplicateVerts({outR.vertPos_.ptrD()}));
  thrust::for_each_n(zip(i30.beginD(), vQ2R.beginD(), inQ_.vertPos_.beginD()),
                     inQ_.NumVert(), DuplicateVerts({outR.vertPos_.ptrD()}));
  // New vertices created from intersections:
  thrust::for_each_n(zip(i12.beginD(), v12R.beginD(), v12_.beginD()),
                     i12.size(), DuplicateVerts({outR.vertPos_.ptrD()}));
  thrust::for_each_n(zip(i21.beginD(), v21R.beginD(), v21_.beginD()),
                     i21.size(), DuplicateVerts({outR.vertPos_.ptrD()}));

  if (kVerbose) {
    std::cout << nPv << " verts from inP" << std::endl;
    std::cout << nQv << " verts from inQ" << std::endl;
    std::cout << n12 << " new verts from edgesP -> facesQ" << std::endl;
    std::cout << n21 << " new verts from facesP -> edgesQ" << std::endl;
  }

  // Build up new polygonal faces from triangle intersections. At this point the
  // calculation switches from parallel to serial.

  // Level 3

  // This key is the forward halfedge index of P or Q. Only includes intersected
  // edges.
  std::map<int, std::vector<EdgePos>> edgesP, edgesQ;
  // This key is the face index of <P, Q>
  std::map<std::pair<int, int>, std::vector<EdgePos>> edgesNew;

  AddNewEdgeVerts(edgesP, edgesNew, p1q2_, i12.H(), v12R.H(),
                  inP_.halfedge_.H(), true);
  AddNewEdgeVerts(edgesQ, edgesNew, p2q1_, i21.H(), v21R.H(),
                  inQ_.halfedge_.H(), false);

  // Level 4
  VecDH<int> faceEdge;
  VecDH<int> facePQ2R;
  std::tie(faceEdge, facePQ2R) =
      SizeOutput(outR, inP_, inQ_, i03, i30, i12, i21, p1q2_, p2q1_,
                 op == Manifold::OpType::SUBTRACT);

  // This gets incremented for each halfedge that's added to a face so that the
  // next one knows where to slot in.
  VecDH<int> facePtrR = faceEdge;
  // Intersected halfedges are marked false.
  VecDH<bool> wholeHalfedgeP(inP_.halfedge_.size(), true);
  VecDH<bool> wholeHalfedgeQ(inQ_.halfedge_.size(), true);

  AppendPartialEdges(outR, wholeHalfedgeP.H(), facePtrR.H(), edgesP, inP_,
                     i03.H(), vP2R.H(), facePQ2R.begin());
  AppendPartialEdges(outR, wholeHalfedgeQ.H(), facePtrR.H(), edgesQ, inQ_,
                     i30.H(), vQ2R.H(), facePQ2R.begin() + inP_.NumTri());

  AppendNewEdges(outR, facePtrR.H(), edgesNew, facePQ2R.H(), inP_.NumTri());

  AppendWholeEdges(outR, facePtrR, inP_, wholeHalfedgeP, i03, vP2R,
                   facePQ2R.cptrD());
  AppendWholeEdges(outR, facePtrR, inQ_, wholeHalfedgeQ, i30, vQ2R,
                   facePQ2R.cptrD() + inP_.NumTri());

  assemble.Stop();
  Timer triangulate;
  triangulate.Start();

  // Level 6

  // Create the manifold's data structures.
  outR.precision_ = glm::max(inP_.precision_, inQ_.precision_);

  outR.Face2Tri(faceEdge);

  // int chi = outR.NumVert() - outR.NumEdge() + outR.NumTri();
  // std::cout << "triangle Genus = " << 1 - chi / 2 << std::endl;

  // outR.SplitNonmanifoldVerts();

  outR.CollapseDegenerates();

  triangulate.Stop();
  Timer finish;
  finish.Start();

  outR.Finish();

  finish.Stop();
  if (kVerbose) {
    assemble.Print("Assembly");
    triangulate.Print("Triangulation");
    finish.Print("Finishing the manifold");
  }

  return outR;
}

}  // namespace manifold