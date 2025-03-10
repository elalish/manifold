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

template <const bool inverted>
struct CopyFaceEdges {
  const SparseIndices &p1q1;
  // x can be either vert or edge (0 or 1).
  SparseIndices &pXq1;
  VecView<const Halfedge> halfedgesQ;
  const size_t offset;

  void operator()(const size_t i) {
    int idx = 3 * (i + offset);
    int pX = p1q1.Get(i, inverted);
    int q2 = p1q1.Get(i, !inverted);

    for (const int j : {0, 1, 2}) {
      const int q1 = 3 * q2 + j;
      const Halfedge edge = halfedgesQ[q1];
      int a = pX;
      int b = edge.IsForward() ? q1 : edge.pairedHalfedge;
      if (inverted) std::swap(a, b);
      pXq1.Set(idx + static_cast<size_t>(j), a, b);
    }
  }
};

inline bool Shadows(double p, double q, double dir) {
  return p == q ? dir < 0 : p < q;
}

inline std::pair<int, vec2> Shadow01(
    const int p0, const int q1, VecView<const vec3> vertPosP,
    VecView<const vec3> vertPosQ, VecView<const Halfedge> halfedgeQ,
    const double expandP, VecView<const vec3> normalP, const bool reverse) {
  const int q1s = halfedgeQ[q1].startVert;
  const int q1e = halfedgeQ[q1].endVert;
  const double p0x = vertPosP[p0].x;
  const double q1sx = vertPosQ[q1s].x;
  const double q1ex = vertPosQ[q1e].x;
  int s01 = reverse ? Shadows(q1sx, p0x, expandP * normalP[q1s].x) -
                          Shadows(q1ex, p0x, expandP * normalP[q1e].x)
                    : Shadows(p0x, q1ex, expandP * normalP[p0].x) -
                          Shadows(p0x, q1sx, expandP * normalP[p0].x);
  vec2 yz01(NAN);

  if (s01 != 0) {
    yz01 = Interpolate(vertPosQ[q1s], vertPosQ[q1e], vertPosP[p0].x);
    if (reverse) {
      vec3 diff = vertPosQ[q1s] - vertPosP[p0];
      const double start2 = la::dot(diff, diff);
      diff = vertPosQ[q1e] - vertPosP[p0];
      const double end2 = la::dot(diff, diff);
      const double dir = start2 < end2 ? normalP[q1s].y : normalP[q1e].y;
      if (!Shadows(yz01[0], vertPosP[p0].y, expandP * dir)) s01 = 0;
    } else {
      if (!Shadows(vertPosP[p0].y, yz01[0], expandP * normalP[p0].y)) s01 = 0;
    }
  }
  return std::make_pair(s01, yz01);
}

struct F11 {
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

struct F02 {
  VecView<const vec3> vertPosP;
  VecView<const Halfedge> halfedgeQ;
  VecView<const vec3> vertPosQ;
  const double expandP;
  VecView<const vec3> vertNormalP;
  const bool forward;

  std::pair<int, double> operator()(int p0, int q2) {
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

    const vec3 posP = vertPosP[p0];
    for (const int i : {0, 1, 2}) {
      const int q1 = 3 * q2 + i;
      const Halfedge edge = halfedgeQ[q1];
      const int q1F = edge.IsForward() ? q1 : edge.pairedHalfedge;

      if (!forward) {
        const int qVert = halfedgeQ[q1F].startVert;
        const vec3 diff = posP - vertPosQ[qVert];
        const double metric = la::dot(diff, diff);
        if (metric < minMetric) {
          minMetric = metric;
          closestVert = qVert;
        }
      }

      const auto syz01 = Shadow01(p0, q1F, vertPosP, vertPosQ, halfedgeQ,
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
      vec3 vertPos = vertPosP[p0];
      z02 = Interpolate(yzzRL[0], yzzRL[1], vertPos.y)[1];
      if (forward) {
        if (!Shadows(vertPos.z, z02, expandP * vertNormalP[p0].z)) s02 = 0;
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
  VecView<int> x;
  VecView<vec3> v;
  VecView<const Halfedge> halfedgesP;
  VecView<const Halfedge> halfedgesQ;
  VecView<const vec3> vertPosP;
  const bool forward;
  const SparseIndices &p1q2;
  F02 f02;
  F11 f11;

  void operator()(const size_t idx) {
    int p1 = p1q2.Get(idx, !forward);
    int q2 = p1q2.Get(idx, forward);
    int &x12 = x[idx];
    vec3 &v12 = v[idx];

    // For xzyLR-[k], k==0 is the left and k==1 is the right.
    int k = 0;
    vec3 xzyLR0[2];
    vec3 xzyLR1[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    x12 = 0;

    const Halfedge edge = halfedgesP[p1];

    for (int vert : {edge.startVert, edge.endVert}) {
      const auto [s, z] = f02(vert, q2);
      if (std::isfinite(z)) {
        x12 += s * ((vert == edge.startVert) == forward ? 1 : -1);
        if (k < 2 && (k == 0 || (s != 0) != shadows)) {
          shadows = s != 0;
          xzyLR0[k] = vertPosP[vert];
          std::swap(xzyLR0[k].y, xzyLR0[k].z);
          xzyLR1[k] = xzyLR0[k];
          xzyLR1[k][1] = z;
          k++;
        }
      }
    }

    for (const int i : {0, 1, 2}) {
      const int q1 = 3 * q2 + i;
      const Halfedge edge = halfedgesQ[q1];
      const int q1F = edge.IsForward() ? q1 : edge.pairedHalfedge;
      const auto [s, xyzz] = forward ? f11(p1, q1F) : f11(q1F, p1);
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
  }
};

std::tuple<Vec<int>, Vec<vec3>> Intersect12(const Manifold::Impl &inP,
                                            const Manifold::Impl &inQ,
                                            SparseIndices &p1q2, double expandP,
                                            bool forward) {
  ZoneScoped;
  Vec<int> x12(p1q2.size());
  Vec<vec3> v12(p1q2.size());

  F02 f02{inP.vertPos_,
          inQ.halfedge_,
          inQ.vertPos_,
          expandP,
          forward ? inP.vertNormal_ : inQ.vertNormal_,
          forward};
  F11 f11{forward ? inP.vertPos_ : inQ.vertPos_,
          forward ? inQ.vertPos_ : inP.vertPos_,
          forward ? inP.halfedge_ : inQ.halfedge_,
          forward ? inQ.halfedge_ : inP.halfedge_,
          expandP,
          forward ? inP.vertNormal_ : inQ.vertNormal_};

  for_each_n(autoPolicy(p1q2.size(), 1e4), countAt(0_uz), p1q2.size(),
             Kernel12({x12, v12, inP.halfedge_, inQ.halfedge_, inP.vertPos_,
                       forward, p1q2, f02, f11}));

  p1q2.KeepFinite(v12, x12);

  return std::make_tuple(x12, v12);
};

struct Winding03Recorder {
  using LocalT = VecView<int>;
  VecView<int> w03;
  F02 &f02;
  bool forward;

  void record(int queryIdx, int leafIdx, VecView<int> &w03) const {
    const auto [s02, z02] = f02(queryIdx, leafIdx);
    if (std::isfinite(z02)) {
      AtomicAdd(w03[queryIdx], s02 * (!forward ? -1 : 1));
    }
  }

  LocalT &local() { return w03; }
};

Vec<int> Winding03(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                   double expandP, bool forward) {
  ZoneScoped;
  // verts that are not shadowed (not in p0q2) have winding number zero.
  Vec<int> w03(inP.NumVert(), 0);
  F02 f02{inP.vertPos_,
          inQ.halfedge_,
          inQ.vertPos_,
          expandP,
          forward ? inP.vertNormal_ : inQ.vertNormal_,
          forward};
  Winding03Recorder recorder{w03.view(), f02, forward};

  inQ.collider_.Collisions<false, const vec3, Winding03Recorder>(inP.vertPos_,
                                                                 recorder);
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

#ifdef MANIFOLD_DEBUG
  Timer broad;
  broad.Start();
#endif

  if (inP.IsEmpty() || inQ.IsEmpty() || !inP.bBox_.DoesOverlap(inQ.bBox_)) {
    PRINT("No overlap, early out");
    w03_.resize(inP.NumVert(), 0);
    w30_.resize(inQ.NumVert(), 0);
    return;
  }

  // Level 3
  // Find edge-triangle overlaps (broad phase)
  p1q2_ = inQ_.EdgeCollisions(inP_);
  p2q1_ = inP_.EdgeCollisions(inQ_, true);  // inverted

#ifdef MANIFOLD_DEBUG
  broad.Stop();
  Timer intersections;
  intersections.Start();
#endif

  // Level 3
  // Build up the intersection of the edges and triangles, keeping only those
  // that intersect, and record the direction the edge is passing through the
  // triangle.
  std::tie(x12_, v12_) = Intersect12(inP, inQ, p1q2_, expandP_, true);
  PRINT("x12 size = " << x12_.size());

  std::tie(x21_, v21_) = Intersect12(inQ, inP, p2q1_, expandP_, false);
  PRINT("x21 size = " << x21_.size());

  if (x12_.size() > INT_MAX_SZ || x21_.size() > INT_MAX_SZ) {
    valid = false;
    return;
  }

  // Sum up the winding numbers of all vertices.
  w03_ = Winding03(inP, inQ, expandP_, true);
  w30_ = Winding03(inQ, inP, expandP_, false);

#ifdef MANIFOLD_DEBUG
  intersections.Stop();

  if (ManifoldParams().verbose) {
    broad.Print("Broad phase");
    intersections.Print("Intersections");
  }
#endif
}
}  // namespace manifold
