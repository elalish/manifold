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

#include "./boolean3.h"

#include <limits>

#include "./parallel.h"

#if (MANIFOLD_PAR == 1)
#include <tbb/combinable.h>
#endif

using namespace manifold;

namespace {

// These two functions (Interpolate and Intersect) are the only places where
// floating-point operations take place in the whole Boolean function. These
// are carefully designed to minimize rounding error and to eliminate it at edge
// cases to ensure consistency.

vec2 Interpolate(vec3 pL, vec3 pR, double x) {
  const double dxL = x - pL.x;
  const double dxR = x - pR.x;
  DEBUG_ASSERT(dxL * dxR <= 0, logicErr,
               "Boolean manifold error: not in domain");
  const bool useL = fabs(dxL) < fabs(dxR);
  const vec3 dLR = pR - pL;
  const double lambda = (useL ? dxL : dxR) / dLR.x;
  if (!std::isfinite(lambda) || !std::isfinite(dLR.y) || !std::isfinite(dLR.z))
    return vec2(pL.y, pL.z);
  vec2 yz;
  yz[0] = fma(lambda, dLR.y, useL ? pL.y : pR.y);
  yz[1] = fma(lambda, dLR.z, useL ? pL.z : pR.z);
  return yz;
}

vec4 Intersect(const vec3 &pL, const vec3 &pR, const vec3 &qL, const vec3 &qR) {
  const double dyL = qL.y - pL.y;
  const double dyR = qR.y - pR.y;
  DEBUG_ASSERT(dyL * dyR <= 0, logicErr,
               "Boolean manifold error: no intersection");
  const bool useL = fabs(dyL) < fabs(dyR);
  const double dx = pR.x - pL.x;
  double lambda = (useL ? dyL : dyR) / (dyL - dyR);
  if (!std::isfinite(lambda)) lambda = 0.0;
  vec4 xyzz;
  xyzz.x = fma(lambda, dx, useL ? pL.x : pR.x);
  const double pDy = pR.y - pL.y;
  const double qDy = qR.y - qL.y;
  const bool useP = fabs(pDy) < fabs(qDy);
  xyzz.y = fma(lambda, useP ? pDy : qDy,
               useL ? (useP ? pL.y : qL.y) : (useP ? pR.y : qR.y));
  xyzz.z = fma(lambda, pR.z - pL.z, useL ? pL.z : pR.z);
  xyzz.w = fma(lambda, qR.z - qL.z, useL ? qL.z : qR.z);
  return xyzz;
}

inline bool Shadows(double a, double b, double dir) {
  return a == b ? dir < 0 : a < b;
}

inline std::pair<int, vec2> Shadow01(
    const int a0, const int b1, VecView<const vec3> vertPosA,
    VecView<const vec3> vertPosB, VecView<const Halfedge> halfedgeB,
    const double expandP, VecView<const vec3> normalP, const bool reverse) {
  const int b1s = halfedgeB[b1].startVert;
  const int b1e = halfedgeB[b1].endVert;
  const double a0x = vertPosA[a0].x;
  const double b1sx = vertPosB[b1s].x;
  const double b1ex = vertPosB[b1e].x;
  int s01 = reverse ? Shadows(b1sx, a0x, expandP * normalP[b1s].x) -
                          Shadows(b1ex, a0x, expandP * normalP[b1e].x)
                    : Shadows(a0x, b1ex, expandP * normalP[a0].x) -
                          Shadows(a0x, b1sx, expandP * normalP[a0].x);
  vec2 yz01(NAN);

  if (s01 != 0) {
    yz01 = Interpolate(vertPosB[b1s], vertPosB[b1e], vertPosA[a0].x);
    if (reverse) {
      vec3 diff = vertPosB[b1s] - vertPosA[a0];
      const double start2 = la::dot(diff, diff);
      diff = vertPosB[b1e] - vertPosA[a0];
      const double end2 = la::dot(diff, diff);
      const double dir = start2 < end2 ? normalP[b1s].y : normalP[b1e].y;
      if (!Shadows(yz01[0], vertPosA[a0].y, expandP * dir)) s01 = 0;
    } else {
      if (!Shadows(vertPosA[a0].y, yz01[0], expandP * normalP[a0].y)) s01 = 0;
    }
  }
  return std::make_pair(s01, yz01);
}

struct Kernel11 {
  VecView<const vec3> vertPosP;
  VecView<const vec3> vertPosQ;
  VecView<const Halfedge> halfedgeP;
  VecView<const Halfedge> halfedgeQ;
  const double expandP;
  VecView<const vec3> normalP;

  std::pair<int, vec4> operator()(int p1, int q1) {
    vec4 xyzz11 = vec4(NAN);
    int s11 = 0;

    // For pRL[k], qRL[k], k==0 is the left and k==1 is the right.
    int k = 0;
    vec3 pRL[2], qRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    s11 = 0;

    const int p0[2] = {halfedgeP[p1].startVert, halfedgeP[p1].endVert};
    for (int i : {0, 1}) {
      const auto [s01, yz01] = Shadow01(p0[i], q1, vertPosP, vertPosQ,
                                        halfedgeQ, expandP, normalP, false);
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz01[0])) {
        s11 += s01 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          pRL[k] = vertPosP[p0[i]];
          qRL[k] = vec3(pRL[k].x, yz01.x, yz01.y);
          ++k;
        }
      }
    }

    const int q0[2] = {halfedgeQ[q1].startVert, halfedgeQ[q1].endVert};
    for (int i : {0, 1}) {
      const auto [s10, yz10] = Shadow01(q0[i], p1, vertPosQ, vertPosP,
                                        halfedgeP, expandP, normalP, true);
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz10[0])) {
        s11 += s10 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s10 != 0) != shadows)) {
          shadows = s10 != 0;
          qRL[k] = vertPosQ[q0[i]];
          pRL[k] = vec3(qRL[k].x, yz10.x, yz10.y);
          ++k;
        }
      }
    }

    if (s11 == 0) {  // No intersection
      xyzz11 = vec4(NAN);
    } else {
      DEBUG_ASSERT(k == 2, logicErr, "Boolean manifold error: s11");
      xyzz11 = Intersect(pRL[0], pRL[1], qRL[0], qRL[1]);

      const int p1s = halfedgeP[p1].startVert;
      const int p1e = halfedgeP[p1].endVert;
      vec3 diff = vertPosP[p1s] - vec3(xyzz11);
      const double start2 = la::dot(diff, diff);
      diff = vertPosP[p1e] - vec3(xyzz11);
      const double end2 = la::dot(diff, diff);
      const double dir = start2 < end2 ? normalP[p1s].z : normalP[p1e].z;

      if (!Shadows(xyzz11.z, xyzz11.w, expandP * dir)) s11 = 0;
    }

    return std::make_pair(s11, xyzz11);
  }
};

struct Kernel02 {
  VecView<const vec3> vertPosA;
  VecView<const Halfedge> halfedgeB;
  VecView<const vec3> vertPosB;
  const double expandP;
  VecView<const vec3> vertNormalP;
  const bool forward;

  std::pair<int, double> operator()(int a0, int b2) {
    int s02 = 0;
    double z02 = 0.0;

    // For yzzLR[k], k==0 is the left and k==1 is the right.
    int k = 0;
    vec3 yzzRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    int closestVert = -1;
    double minMetric = std::numeric_limits<double>::infinity();
    s02 = 0;

    const vec3 posA = vertPosA[a0];
    for (const int i : {0, 1, 2}) {
      const int b1 = 3 * b2 + i;
      const Halfedge edge = halfedgeB[b1];
      const int b1F = edge.IsForward() ? b1 : edge.pairedHalfedge;

      if (!forward) {
        const int b0 = halfedgeB[b1F].startVert;
        const vec3 diff = posA - vertPosB[b0];
        const double metric = la::dot(diff, diff);
        if (metric < minMetric) {
          minMetric = metric;
          closestVert = b0;
        }
      }

      const auto syz01 = Shadow01(a0, b1F, vertPosA, vertPosB, halfedgeB,
                                  expandP, vertNormalP, !forward);
      const int s01 = syz01.first;
      const vec2 yz01 = syz01.second;
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz01[0])) {
        s02 += s01 * (forward == edge.IsForward() ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          yzzRL[k++] = vec3(yz01[0], yz01[1], yz01[1]);
        }
      }
    }

    if (s02 == 0) {  // No intersection
      z02 = NAN;
    } else {
      DEBUG_ASSERT(k == 2, logicErr, "Boolean manifold error: s02");
      vec3 vertPos = vertPosA[a0];
      z02 = Interpolate(yzzRL[0], yzzRL[1], vertPos.y)[1];
      if (forward) {
        if (!Shadows(vertPos.z, z02, expandP * vertNormalP[a0].z)) s02 = 0;
      } else {
        // DEBUG_ASSERT(closestVert != -1, topologyErr, "No closest vert");
        if (!Shadows(z02, vertPos.z, expandP * vertNormalP[closestVert].z))
          s02 = 0;
      }
    }
    return std::make_pair(s02, z02);
  }
};

struct Kernel12 {
  VecView<const Halfedge> halfedgeA;
  VecView<const Halfedge> halfedgeB;
  VecView<const vec3> vertPosA;
  const bool forward;
  Kernel02 k02;
  Kernel11 k11;

  std::pair<int, vec3> operator()(int a1, int b2) {
    int x12 = 0;
    vec3 v12 = vec3(NAN);

    // For xzyLR-[k], k==0 is the left and k==1 is the right.
    int k = 0;
    vec3 xzyLR0[2];
    vec3 xzyLR1[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    x12 = 0;

    const Halfedge edge = halfedgeA[a1];

    for (int vert : {edge.startVert, edge.endVert}) {
      const auto [s, z] = k02(vert, b2);
      if (std::isfinite(z)) {
        x12 += s * ((vert == edge.startVert) == forward ? 1 : -1);
        if (k < 2 && (k == 0 || (s != 0) != shadows)) {
          shadows = s != 0;
          xzyLR0[k] = vertPosA[vert];
          std::swap(xzyLR0[k].y, xzyLR0[k].z);
          xzyLR1[k] = xzyLR0[k];
          xzyLR1[k][1] = z;
          k++;
        }
      }
    }

    for (const int i : {0, 1, 2}) {
      const int b1 = 3 * b2 + i;
      const Halfedge edge = halfedgeB[b1];
      const int b1F = edge.IsForward() ? b1 : edge.pairedHalfedge;
      const auto [s, xyzz] = forward ? k11(a1, b1F) : k11(b1F, a1);
      if (std::isfinite(xyzz[0])) {
        x12 -= s * (edge.IsForward() ? 1 : -1);
        if (k < 2 && (k == 0 || (s != 0) != shadows)) {
          shadows = s != 0;
          xzyLR0[k][0] = xyzz.x;
          xzyLR0[k][1] = xyzz.z;
          xzyLR0[k][2] = xyzz.y;
          xzyLR1[k] = xzyLR0[k];
          xzyLR1[k][1] = xyzz.w;
          if (!forward) std::swap(xzyLR0[k][1], xzyLR1[k][1]);
          k++;
        }
      }
    }

    if (x12 == 0) {  // No intersection
      v12 = vec3(NAN);
    } else {
      DEBUG_ASSERT(k == 2, logicErr, "Boolean manifold error: v12");
      const vec4 xzyy = Intersect(xzyLR0[0], xzyLR0[1], xzyLR1[0], xzyLR1[1]);
      v12.x = xzyy[0];
      v12.y = xzyy[2];
      v12.z = xzyy[1];
    }
    return std::make_pair(x12, v12);
  }
};

struct Kernel12Tmp {
  Vec<std::array<int, 2>> a1q2_;
  Vec<int> x12_;
  Vec<vec3> v12_;
};

struct Kernel12Recorder {
  using Local = Kernel12Tmp;
  Kernel12 &k12;
  VecView<const TmpEdge> tmpedges;
  bool forward;

#if MANIFOLD_PAR == 1
  tbb::combinable<Kernel12Tmp> store;
  Local &local() { return store.local(); }
#else
  Kernel12Tmp localStore;
  Local &local() { return localStore; }
#endif

  void record(int queryIdx, int leafIdx, Local &tmp) {
    queryIdx = tmpedges[queryIdx].halfedgeIdx;
    const auto [x12, v12] = k12(queryIdx, leafIdx);
    if (std::isfinite(v12[0])) {
      if (forward)
        tmp.a1q2_.push_back({queryIdx, leafIdx});
      else
        tmp.a1q2_.push_back({leafIdx, queryIdx});
      tmp.x12_.push_back(x12);
      tmp.v12_.push_back(v12);
    }
  }

  Kernel12Tmp get() {
#if MANIFOLD_PAR == 1
    Kernel12Tmp result;
    std::vector<Kernel12Tmp> tmps;
    store.combine_each(
        [&](Kernel12Tmp &data) { tmps.emplace_back(std::move(data)); });
    std::vector<size_t> sizes;
    size_t total_size = 0;
    for (const auto &tmp : tmps) {
      sizes.push_back(total_size);
      total_size += tmp.x12_.size();
    }
    result.a1q2_.resize(total_size);
    result.x12_.resize(total_size);
    result.v12_.resize(total_size);
    for_each_n(ExecutionPolicy::Seq, countAt(0), tmps.size(), [&](size_t i) {
      std::copy(tmps[i].a1q2_.begin(), tmps[i].a1q2_.end(),
                result.a1q2_.begin() + sizes[i]);
      std::copy(tmps[i].x12_.begin(), tmps[i].x12_.end(),
                result.x12_.begin() + sizes[i]);
      std::copy(tmps[i].v12_.begin(), tmps[i].v12_.end(),
                result.v12_.begin() + sizes[i]);
    });
    return result;
#else
    return localStore;
#endif
  }
};

std::tuple<Vec<int>, Vec<vec3>> Intersect12(const Manifold::Impl &inP,
                                            const Manifold::Impl &inQ,
                                            Vec<std::array<int, 2>> &a1b2,
                                            double expandP, bool forward) {
  ZoneScoped;
  const Manifold::Impl &a1 = forward ? inP : inQ;
  const Manifold::Impl &b2 = forward ? inQ : inP;

  Kernel02 k02{a1.vertPos_, b2.halfedge_,    b2.vertPos_,
               expandP,     inP.vertNormal_, forward};
  Kernel11 k11{inP.vertPos_,  inQ.vertPos_, inP.halfedge_,
               inQ.halfedge_, expandP,      inP.vertNormal_};

  Vec<TmpEdge> a1TmpEdges = CreateTmpEdges(a1.halfedge_);
  Vec<Box> a1EdgeBB(a1TmpEdges.size());
  for_each_n(autoPolicy(a1TmpEdges.size(), 1e5), countAt(0), a1TmpEdges.size(),
             [&](const int e) {
               a1EdgeBB[e] = Box(a1.vertPos_[a1TmpEdges[e].first],
                                 a1.vertPos_[a1TmpEdges[e].second]);
             });
  Kernel12 k12{a1.halfedge_, b2.halfedge_, a1.vertPos_, forward, k02, k11};
  Kernel12Recorder recorder{k12, a1TmpEdges, forward};

  b2.collider_.Collisions<false, Box, Kernel12Recorder>(a1EdgeBB.cview(),
                                                        recorder);

  Kernel12Tmp result = recorder.get();
  a1b2 = std::move(result.a1q2_);
  auto x12 = std::move(result.x12_);
  auto v12 = std::move(result.v12_);
  // sort p1q2
  Vec<size_t> i12(a1b2.size());
  sequence(i12.begin(), i12.end());
  stable_sort(i12.begin(), i12.end(), [&](int a, int b) {
    return a1b2[a][0] < a1b2[b][0] ||
           (a1b2[a][0] == a1b2[b][0] && a1b2[a][1] < a1b2[b][1]);
  });
  Permute(a1b2, i12);
  Permute(x12, i12);
  Permute(v12, i12);
  return std::make_tuple(x12, v12);
};

Vec<int> Winding03(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                   double expandP, bool forward) {
  ZoneScoped;
  // verts that are not shadowed (not in p0q2) have winding number zero.
  const Manifold::Impl &a0 = forward ? inP : inQ;
  const Manifold::Impl &b2 = forward ? inQ : inP;
  Vec<int> w03(a0.NumVert(), 0);
  Kernel02 k02{a0.vertPos_, b2.halfedge_,    b2.vertPos_,
               expandP,     inP.vertNormal_, forward};
  auto f = [&](int a, int b) {
    const auto [s02, z02] = k02(a, b);
    if (std::isfinite(z02)) AtomicAdd(w03[a], s02 * (!forward ? -1 : 1));
  };
  auto recorder = MakeSimpleRecorder(f);
  b2.collider_.Collisions<false>(a0.vertPos_.cview(), recorder);
  return w03;
};
}  // namespace

namespace manifold {
Boolean3::Boolean3(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                   OpType op)
    : inP_(inP), inQ_(inQ), expandP_(op == OpType::Add ? 1.0 : -1.0) {
  // Symbolic perturbation:
  // Union -> expand inP
  // Difference, Intersection -> contract inP
  constexpr size_t INT_MAX_SZ =
      static_cast<size_t>(std::numeric_limits<int>::max());

  if (inP.IsEmpty() || inQ.IsEmpty() || !inP.bBox_.DoesOverlap(inQ.bBox_)) {
    PRINT("No overlap, early out");
    w03_.resize(inP.NumVert(), 0);
    w30_.resize(inQ.NumVert(), 0);
    return;
  }

#ifdef MANIFOLD_DEBUG
  Timer intersections;
  intersections.Start();
#endif

  // Level 3
  // Build up the intersection of the edges and triangles, keeping only those
  // that intersect, and record the direction the edge is passing through the
  // triangle.
  std::tie(x12_, v12_) = Intersect12(inP, inQ, p1q2_, expandP_, true);
  PRINT("x12 size = " << x12_.size());

  std::tie(x21_, v21_) = Intersect12(inP, inQ, p2q1_, expandP_, false);
  PRINT("x21 size = " << x21_.size());

  if (x12_.size() > INT_MAX_SZ || x21_.size() > INT_MAX_SZ) {
    valid = false;
    return;
  }

  // Sum up the winding numbers of all vertices.
  w03_ = Winding03(inP, inQ, expandP_, true);
  w30_ = Winding03(inP, inQ, expandP_, false);

#ifdef MANIFOLD_DEBUG
  intersections.Stop();

  if (ManifoldParams().verbose) {
    intersections.Print("Intersections");
  }
#endif
}
}  // namespace manifold
