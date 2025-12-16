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

#include "boolean3.h"

#include <limits>
#include <unordered_set>

#include "disjoint_sets.h"
#include "parallel.h"

#if (MANIFOLD_PAR == 1)
#include <tbb/combinable.h>
#endif

using namespace manifold;

namespace {

// These two functions (Interpolate and Intersect) are the only places where
// floating-point operations take place in the whole Boolean function. These
// are carefully designed to minimize rounding error and to eliminate it at edge
// cases to ensure consistency.

inline double withSign(bool pos, double v) { return pos ? v : -v; }

inline vec2 Interpolate(vec3 aL, vec3 aR, double x) {
  const double dxL = x - aL.x;
  const double dxR = x - aR.x;
  DEBUG_ASSERT(dxL * dxR <= 0, logicErr,
               "Boolean manifold error: not in domain");
  const bool useL = fabs(dxL) < fabs(dxR);
  const vec3 dLR = aR - aL;
  const double lambda = (useL ? dxL : dxR) / dLR.x;
  if (!std::isfinite(lambda) || !std::isfinite(dLR.y) || !std::isfinite(dLR.z))
    return vec2(aL.y, aL.z);
  vec2 yz;
  yz[0] = lambda * dLR.y + (useL ? aL.y : aR.y);
  yz[1] = lambda * dLR.z + (useL ? aL.z : aR.z);
  return yz;
}

vec4 Intersect(const vec3& aL, const vec3& aR, const vec3& bL, const vec3& bR) {
  const double dyL = bL.y - aL.y;
  const double dyR = bR.y - aR.y;
  DEBUG_ASSERT(dyL * dyR <= 0, logicErr,
               "Boolean manifold error: no intersection");
  const bool useL = fabs(dyL) < fabs(dyR);
  const double dx = aR.x - aL.x;
  double lambda = (useL ? dyL : dyR) / (dyL - dyR);
  if (!std::isfinite(lambda)) lambda = 0.0;
  vec4 xyzz;
  xyzz.x = lambda * dx + (useL ? aL.x : aR.x);
  const double aDy = aR.y - aL.y;
  const double bDy = bR.y - bL.y;
  const bool useA = fabs(aDy) < fabs(bDy);
  xyzz.y = lambda * (useA ? aDy : bDy) +
           (useL ? (useA ? aL.y : bL.y) : (useA ? aR.y : bR.y));
  xyzz.z = lambda * (aR.z - aL.z) + (useL ? aL.z : aR.z);
  xyzz.w = lambda * (bR.z - bL.z) + (useL ? bL.z : bR.z);
  return xyzz;
}

inline bool Shadows(double p, double q, double dir) {
  return p == q ? dir < 0 : p < q;
}

template <bool expandP, bool forward>
inline std::pair<int, vec2> Shadow01(const int a0, const int b1,
                                     const Manifold::Impl& inA,
                                     const Manifold::Impl& inB) {
  const int b1s = inB.halfedge_[b1].startVert;
  const int b1e = inB.halfedge_[b1].endVert;
  const double a0x = inA.vertPos_[a0].x;
  const double b1sx = inB.vertPos_[b1s].x;
  const double b1ex = inB.vertPos_[b1e].x;
  const double a0xp = inA.vertNormal_[a0].x;
  const double b1sxp = inB.vertNormal_[b1s].x;
  const double b1exp = inB.vertNormal_[b1e].x;
  int s01 = forward ? Shadows(a0x, b1ex, withSign(expandP, a0xp) - b1exp) -
                          Shadows(a0x, b1sx, withSign(expandP, a0xp) - b1sxp)
                    : Shadows(b1sx, a0x, withSign(expandP, b1sxp) - a0xp) -
                          Shadows(b1ex, a0x, withSign(expandP, b1exp) - a0xp);
  vec2 yz01(NAN);

  if (s01 != 0) {
    yz01 =
        Interpolate(inB.vertPos_[b1s], inB.vertPos_[b1e], inA.vertPos_[a0].x);
    const int b1pair = inB.halfedge_[b1].pairedHalfedge;
    const double dir =
        inB.faceNormal_[b1 / 3].y + inB.faceNormal_[b1pair / 3].y;
    if (forward) {
      if (!Shadows(inA.vertPos_[a0].y, yz01[0], -dir)) s01 = 0;
    } else {
      if (!Shadows(yz01[0], inA.vertPos_[a0].y, withSign(expandP, dir)))
        s01 = 0;
    }
  }
  return std::make_pair(s01, yz01);
}

template <bool expandP>
struct Kernel11 {
  const Manifold::Impl& inP;
  const Manifold::Impl& inQ;

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

    const int p0[2] = {inP.halfedge_[p1].startVert, inP.halfedge_[p1].endVert};
    for (int i : {0, 1}) {
      const auto [s01, yz01] = Shadow01<expandP, true>(p0[i], q1, inP, inQ);
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz01[0])) {
        s11 += s01 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          pRL[k] = inP.vertPos_[p0[i]];
          qRL[k] = vec3(pRL[k].x, yz01.x, yz01.y);
          ++k;
        }
      }
    }

    const int q0[2] = {inQ.halfedge_[q1].startVert, inQ.halfedge_[q1].endVert};
    for (int i : {0, 1}) {
      const auto [s10, yz10] = Shadow01<expandP, false>(q0[i], p1, inQ, inP);
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz10[0])) {
        s11 += s10 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s10 != 0) != shadows)) {
          shadows = s10 != 0;
          qRL[k] = inQ.vertPos_[q0[i]];
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

      const int p1pair = inP.halfedge_[p1].pairedHalfedge;
      const double dirP =
          inP.faceNormal_[p1 / 3].z + inP.faceNormal_[p1pair / 3].z;
      const int q1pair = inQ.halfedge_[q1].pairedHalfedge;
      const double dirQ =
          inQ.faceNormal_[q1 / 3].z + inQ.faceNormal_[q1pair / 3].z;
      if (!Shadows(xyzz11.z, xyzz11.w, withSign(expandP, dirP) - dirQ)) s11 = 0;
    }

    return std::make_pair(s11, xyzz11);
  }
};

template <bool expandP, bool forward>
struct Kernel02 {
  const Manifold::Impl& inA;
  const Manifold::Impl& inB;

  std::pair<int, double> operator()(int a0, int b2) {
    int s02 = 0;
    double z02 = 0.0;

    // For yzzLR[k], k==0 is the left and k==1 is the right.
    int k = 0;
    vec3 yzzRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;

    for (const int i : {0, 1, 2}) {
      const int b1 = 3 * b2 + i;
      const Halfedge edgeB = inB.halfedge_[b1];
      const int b1F = edgeB.IsForward() ? b1 : edgeB.pairedHalfedge;

      const auto syz01 = Shadow01<expandP, forward>(a0, b1F, inA, inB);
      const int s01 = syz01.first;
      const vec2 yz01 = syz01.second;
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz01[0])) {
        s02 += s01 * (forward == edgeB.IsForward() ? -1 : 1);
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
      vec3 vertPosA = inA.vertPos_[a0];
      z02 = Interpolate(yzzRL[0], yzzRL[1], vertPosA.y)[1];
      if (forward) {
        if (!Shadows(vertPosA.z, z02, -inB.faceNormal_[b2].z)) s02 = 0;
      } else {
        // DEBUG_ASSERT(closestVert != -1, topologyErr, "No closest vert");
        if (!Shadows(z02, vertPosA.z, withSign(expandP, inB.faceNormal_[b2].z)))
          s02 = 0;
      }
    }
    return std::make_pair(s02, z02);
  }
};

template <bool expandP, bool forward>
struct Kernel12 {
  const Manifold::Impl& inA;
  const Manifold::Impl& inB;
  Kernel02<expandP, forward> k02;
  Kernel11<expandP> k11;

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

    const Halfedge edgeA = inA.halfedge_[a1];

    for (int vertA : {edgeA.startVert, edgeA.endVert}) {
      const auto [s, z] = k02(vertA, b2);
      if (std::isfinite(z)) {
        x12 += s * ((vertA == edgeA.startVert) == forward ? 1 : -1);
        if (k < 2 && (k == 0 || (s != 0) != shadows)) {
          shadows = s != 0;
          xzyLR0[k] = inA.vertPos_[vertA];
          std::swap(xzyLR0[k].y, xzyLR0[k].z);
          xzyLR1[k] = xzyLR0[k];
          xzyLR1[k][1] = z;
          k++;
        }
      }
    }

    for (const int i : {0, 1, 2}) {
      const int b1 = 3 * b2 + i;
      const Halfedge edgeB = inB.halfedge_[b1];
      const int b1F = edgeB.IsForward() ? b1 : edgeB.pairedHalfedge;
      const auto [s, xyzz] = forward ? k11(a1, b1F) : k11(b1F, a1);
      if (std::isfinite(xyzz[0])) {
        x12 -= s * (edgeB.IsForward() ? 1 : -1);
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

template <bool expandP, bool forward>
struct Kernel12Recorder {
  using Local = Intersections;
  Kernel12<expandP, forward>& k12;

#if MANIFOLD_PAR == 1
  tbb::combinable<Intersections> store;
  Local& local() { return store.local(); }
#else
  Intersections localStore;
  Local& local() { return localStore; }
#endif

  void record(int queryIdx, int leafIdx, Local& tmp) {
    const auto [x12, v12] = k12(queryIdx, leafIdx);
    if (std::isfinite(v12[0])) {
      if (forward)
        tmp.p1q2.push_back({queryIdx, leafIdx});
      else
        tmp.p1q2.push_back({leafIdx, queryIdx});
      tmp.x12.push_back(x12);
      tmp.v12.push_back(v12);
    }
  }

  Intersections get() {
#if MANIFOLD_PAR == 1
    Intersections result;
    std::vector<Intersections> tmps;
    store.combine_each(
        [&](Intersections& data) { tmps.emplace_back(std::move(data)); });
    std::vector<size_t> sizes;
    size_t total_size = 0;
    for (const auto& tmp : tmps) {
      sizes.push_back(total_size);
      total_size += tmp.x12.size();
    }
    result.p1q2.resize(total_size);
    result.x12.resize(total_size);
    result.v12.resize(total_size);
    for_each_n(ExecutionPolicy::Seq, countAt(0), tmps.size(), [&](size_t i) {
      std::copy(tmps[i].p1q2.begin(), tmps[i].p1q2.end(),
                result.p1q2.begin() + sizes[i]);
      std::copy(tmps[i].x12.begin(), tmps[i].x12.end(),
                result.x12.begin() + sizes[i]);
      std::copy(tmps[i].v12.begin(), tmps[i].v12.end(),
                result.v12.begin() + sizes[i]);
    });
    return result;
#else
    return localStore;
#endif
  }
};

template <bool expandP, bool forward>
Intersections Intersect12_(const Manifold::Impl& inP,
                           const Manifold::Impl& inQ) {
  ZoneScoped;
  // a: 1 (edge), b: 2 (face)
  const Manifold::Impl& a = forward ? inP : inQ;
  const Manifold::Impl& b = forward ? inQ : inP;

  Kernel02<expandP, forward> k02{a, b};
  Kernel11<expandP> k11{inP, inQ};

  Kernel12<expandP, forward> k12{a, b, k02, k11};
  Kernel12Recorder<expandP, forward> recorder{k12, {}};
  auto f = [&a](int i) {
    return a.halfedge_[i].IsForward()
               ? Box(a.vertPos_[a.halfedge_[i].startVert],
                     a.vertPos_[a.halfedge_[i].endVert])
               : Box();
  };
  b.collider_.Collisions<false>(f, a.halfedge_.size(), recorder);

  Intersections result = recorder.get();
  auto& p1q2 = result.p1q2;
  // sort p1q2 according to edges
  Vec<size_t> i12(p1q2.size());
  sequence(i12.begin(), i12.end());

  int index = forward ? 0 : 1;
  stable_sort(i12.begin(), i12.end(), [&](int a, int b) {
    return p1q2[a][index] < p1q2[b][index] ||
           (p1q2[a][index] == p1q2[b][index] &&
            p1q2[a][1 - index] < p1q2[b][1 - index]);
  });
  Permute(p1q2, i12);
  Permute(result.x12, i12);
  Permute(result.v12, i12);
  return result;
};

template <bool forward>
Intersections Intersect12(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                          bool expandP) {
  if (expandP)
    return Intersect12_<true, forward>(inP, inQ);
  else
    return Intersect12_<false, forward>(inP, inQ);
}

template <bool expandP, bool forward>
Vec<int> Winding03_(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                    const VecView<std::array<int, 2>> p1q2) {
  ZoneScoped;
  // a: 0 (vert), b: 2 (face)
  const Manifold::Impl& a = forward ? inP : inQ;
  const Manifold::Impl& b = forward ? inQ : inP;
  Vec<int> brokenHalfedges;
  int index = forward ? 0 : 1;

  DisjointSets uA(a.vertPos_.size());
  for_each(autoPolicy(a.halfedge_.size()), countAt(0),
           countAt(a.halfedge_.size()), [&](int edge) {
             const Halfedge& he = a.halfedge_[edge];
             if (!he.IsForward()) return;
             // check if the edge is broken
             auto it = std::lower_bound(
                 p1q2.begin(), p1q2.end(), edge,
                 [index](const std::array<int, 2>& collisionPair, int e) {
                   return collisionPair[index] < e;
                 });
             if (it == p1q2.end() || (*it)[index] != edge)
               uA.unite(he.startVert, he.endVert);
           });

  // find components, the hope is the number of components should be small
  std::unordered_set<int> components;
#if (MANIFOLD_PAR == 1)
  if (a.vertPos_.size() > 1e5) {
    tbb::combinable<std::unordered_set<int>> componentsShared;
    for_each(autoPolicy(a.vertPos_.size()), countAt(0),
             countAt(a.vertPos_.size()),
             [&](int v) { componentsShared.local().insert(uA.find(v)); });
    componentsShared.combine_each([&](const std::unordered_set<int>& data) {
      components.insert(data.begin(), data.end());
    });
  } else
#endif
  {
    for (size_t v = 0; v < a.vertPos_.size(); v++)
      components.insert(uA.find(v));
  }
  Vec<int> verts;
  verts.reserve(components.size());
  for (int c : components) verts.push_back(c);

  Vec<int> w03(a.NumVert(), 0);
  Kernel02<expandP, forward> k02{a, b};
  auto recorderf = [&](int i, int b) {
    const auto [s02, z02] = k02(verts[i], b);
    if (std::isfinite(z02)) w03[verts[i]] += s02 * (forward ? 1 : -1);
  };
  auto recorder = MakeSimpleRecorder(recorderf);
  auto f = [&](int i) { return a.vertPos_[verts[i]]; };
  b.collider_.Collisions<false, decltype(f), decltype(recorder)>(
      f, verts.size(), recorder);
  // flood fill
  for_each(autoPolicy(w03.size()), countAt(0), countAt(w03.size()),
           [&](size_t i) {
             size_t root = uA.find(i);
             if (root == i) return;
             w03[i] = w03[root];
           });
  return w03;
}

template <bool forward>
Vec<int> Winding03(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                   const VecView<std::array<int, 2>> p1q2, bool expandP) {
  if (expandP)
    return Winding03_<true, forward>(inP, inQ, p1q2);
  else
    return Winding03_<false, forward>(inP, inQ, p1q2);
}
}  // namespace

namespace manifold {
Boolean3::Boolean3(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                   OpType op)
    : inP_(inP), inQ_(inQ), expandP_(op == OpType::Add) {
  // Symbolic perturbation:
  // Union -> expand inP, expand inQ
  // Difference, Intersection -> contract inP, expand inQ
  // Technically Intersection should contract inQ, but doing it this way makes
  // Split faster and any suboptimal cases seem pretty rare.

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
  xv12_ = Intersect12<true>(inP, inQ, expandP_);
  xv21_ = Intersect12<false>(inP, inQ, expandP_);

  if (xv12_.x12.size() > INT_MAX_SZ || xv21_.x12.size() > INT_MAX_SZ) {
    valid = false;
    return;
  }

  // Compute winding numbers of all vertices using flood fill
  // Vertices on the same connected component have the same winding number
  w03_ = Winding03<true>(inP, inQ, xv12_.p1q2, expandP_);
  w30_ = Winding03<false>(inP, inQ, xv21_.p1q2, expandP_);

#ifdef MANIFOLD_DEBUG
  intersections.Stop();

  if (ManifoldParams().verbose) {
    intersections.Print("Intersections");
  }
#endif
}
}  // namespace manifold
