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
#include "shared.h"

#if (MANIFOLD_PAR == 1)
#include <tbb/combinable.h>
#endif

using namespace manifold;

namespace {

// Coordinates within the smaller mesh's epsilon are ties broken by the
// symbolic perturbation, so surfaces that are coincident up to rounding are
// treated as coincident.
inline double TieTol(const Manifold::Impl& a, const Manifold::Impl& b) {
  return std::min(a.epsilon_, b.epsilon_);
}

inline Box VerticalQueryBox(const vec3& point, double tol) {
  const double inf = std::numeric_limits<double>::infinity();
  return Box(vec3(point.x - tol, point.y - tol, -inf),
             vec3(point.x + tol, point.y + tol, inf));
}

constexpr double kPlaneTolerance = 1e-9;

// The direction each vertex moves to decide ties: where it touches at most
// three distinct face planes, the one that moves each of them outward by
// exactly the same distance, so that a face stays parallel to itself;
// elsewhere the vertex normal. At most twice the length of a normal, so that
// a sharp wedge doesn't move a vertex by more than the tolerance.
Vec<vec3> Displacements(const Manifold::Impl& m) {
  const int numVert = m.NumVert();
  Vec<vec3> disp(m.vertNormal_);
  std::vector<std::array<vec3, 3>> planes(numVert);
  std::vector<int> numPlanes(numVert, 0);
  for (size_t tri = 0; tri < m.NumTri(); ++tri) {
    const vec3 n = m.faceNormal_[tri];
    for (const int i : {0, 1, 2}) {
      const int v = m.halfedge_.Start(3 * tri + i);
      if (v < 0 || numPlanes[v] < 0) continue;
      bool seen = false;
      for (int k = 0; k < numPlanes[v]; ++k)
        if (la::dot(planes[v][k], n) > 1 - kPlaneTolerance) seen = true;
      if (seen) continue;
      if (numPlanes[v] == 3) {
        numPlanes[v] = -1;
        continue;
      }
      planes[v][numPlanes[v]++] = n;
    }
  }
  for (int v = 0; v < numVert; ++v) {
    const std::array<vec3, 3>& n = planes[v];
    if (numPlanes[v] == 1) {
      disp[v] = n[0];
    } else if (numPlanes[v] == 2) {
      disp[v] = (n[0] + n[1]) / (1 + la::dot(n[0], n[1]));
    } else if (numPlanes[v] == 3) {
      const double det = la::dot(n[0], la::cross(n[1], n[2]));
      // three planes through a common line have no common offset
      if (std::abs(det) > kPlaneTolerance)
        disp[v] = (la::cross(n[1], n[2]) + la::cross(n[2], n[0]) +
                   la::cross(n[0], n[1])) /
                  det;
    }
  }
  for (vec3& d : disp)
    if (la::length(d) > 2) d = 2.0 * la::normalize(d);
  return disp;
}

// The verts of a mesh as moved to decide ties: each position plus scale times
// its direction plus shift. At a tie (two values within TieTol) the two are
// taken as equal, and the comparison is decided by how much the moves change
// each. The moves are one real configuration, so all the ties are decided
// alike, and they are less than half the tolerance, far above the rounding
// that makes values which should be equal into ties.
class Moved {
 public:
  // The operands of a boolean, along their Displacements by a quarter of the
  // tolerance: the verts of Q outward, those of P outward when expanding and
  // inward otherwise and an eighth further, so that where faces of P and Q
  // coincide, P's is the outer one; and P shifted a little more, for the ties
  // the displacements leave exact: along an axis where neither vert's
  // displacement has a component, the moves are equal. The shift differs per
  // axis so that a slope of 1, common on grids, can't cancel it.
  static Moved P(const Manifold::Impl& m, double tol, bool expandP) {
    const double step = tol / 4;
    return Moved(m, Displacements(m), (expandP ? 1.125 : -1.125) * step,
                 step * vec3(1.0 / 61, 1.0 / 89, 1.0 / 113));
  }
  static Moved Q(const Manifold::Impl& m, double tol) {
    return Moved(m, Displacements(m), tol / 4, vec3(0.0));
  }
  // A mesh queried by points or a ray, outward along its vertex normals.
  static Moved Queried(const Manifold::Impl& m, double tol) {
    return Moved(m, {}, tol / 4, vec3(0.0));
  }
  // The points or the ray of a query, which don't move.
  static Moved Still(const Manifold::Impl& m) {
    return Moved(m, {}, 0, vec3(0.0));
  }

  vec3 operator[](int vert) const {
    if (scale_ == 0) return mesh_.vertPos_[vert] + shift_;
    const vec3 dir = disp_.empty() ? mesh_.vertNormal_[vert] : disp_[vert];
    return mesh_.vertPos_[vert] + scale_ * dir + shift_;
  }

 private:
  Moved(const Manifold::Impl& mesh, Vec<vec3> disp, double scale, vec3 shift)
      : mesh_(mesh), disp_(std::move(disp)), scale_(scale), shift_(shift) {}

  const Manifold::Impl& mesh_;
  Vec<vec3> disp_;  // empty: along the vertex normals
  double scale_;
  vec3 shift_;
};

struct FaceEdge {
  int edge;
  int start;
  int end;
  bool isForward;
};

inline void LoadFaceEdges(const Halfedges& halfedges, int tri,
                          FaceEdge edge[3]) {
  for (const int i : {0, 1, 2}) {
    const int halfedge = 3 * tri + i;
    const int start = halfedges.Start(halfedge);
    const int end = halfedges.Start(3 * tri + Next3(i));
    if (start < end) {
      edge[i] = {halfedge, start, end, true};
    } else {
      edge[i] = {halfedges.Pair(halfedge), end, start, false};
    }
  }
}

// p < q, where p belongs to P and q to Q. At a tie the two are taken as
// equal, and the moves decide: by how much each changes. Not movedP < movedQ:
// the moves are a fraction of tol, so a real gap inside tol could outweigh
// them, and values within tol would no longer count as equal, which breaks
// Precision2 and PointWithinToleranceOfCoincidentEdge among others.
inline bool Shadows(double p, double q, double movedP, double movedQ,
                    double tol) {
  return std::abs(p - q) > tol ? p < q : movedP - p < movedQ - q;
}

template <bool forward>
inline std::pair<int, vec2> Shadow01(const int a0, const int b1, const int b1s,
                                     const int b1e, const Manifold::Impl& inA,
                                     const Manifold::Impl& inB, const Moved& mA,
                                     const Moved& mB) {
  const double a0x = inA.vertPos_[a0].x;
  const double b1sx = inB.vertPos_[b1s].x;
  const double b1ex = inB.vertPos_[b1e].x;
  const double tol = TieTol(inA, inB);
  const double ma0x = mA[a0].x;
  int s01 = forward ? Shadows(a0x, b1ex, ma0x, mB[b1e].x, tol) -
                          Shadows(a0x, b1sx, ma0x, mB[b1s].x, tol)
                    : Shadows(b1sx, a0x, mB[b1s].x, ma0x, tol) -
                          Shadows(b1ex, a0x, mB[b1e].x, ma0x, tol);
  vec2 yz01(NAN);

  if (s01 != 0) {
    yz01 = Interpolate(inB.vertPos_[b1s], inB.vertPos_[b1e], a0x, tol);
    const double ay = inA.vertPos_[a0].y;
    double y = yz01[0], may = ay, myz = yz01[0];
    if (std::abs(ay - yz01[0]) <= tol) {
      y = InterpolateLine(inB.vertPos_[b1s], inB.vertPos_[b1e], a0x)[0];
      may = mA[a0].y;
      myz = InterpolateLine(mB[b1s], mB[b1e], ma0x)[0];
    }
    if (forward ? !Shadows(ay, y, may, myz, tol)
                : !Shadows(y, ay, myz, may, tol))
      s01 = 0;
  }
  return std::make_pair(s01, yz01);
}

template <bool expandP>
struct Kernel11 {
  const Manifold::Impl& inP;
  const Manifold::Impl& inQ;
  const Moved& mP;
  const Moved& mQ;

  std::pair<int, vec4> operator()(int p1, int p1s, int p1e, int q1, int q1s,
                                  int q1e) {
    vec4 xyzz11 = vec4(NAN);
    int s11 = 0;

    // For pRL[k], qRL[k], k==0 is the left and k==1 is the right.
    int k = 0;
    vec3 pRL[2], qRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    s11 = 0;
    // which vertex each of pRL/qRL came from, P's (true) or Q's
    int vertRL[2];
    bool ofPRL[2];

    const int p0[2] = {p1s, p1e};
    for (int i : {0, 1}) {
      const auto [s01, yz01] =
          Shadow01<true>(p0[i], q1, q1s, q1e, inP, inQ, mP, mQ);
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz01[0])) {
        s11 += s01 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          pRL[k] = inP.vertPos_[p0[i]];
          qRL[k] = vec3(pRL[k].x, yz01.x, yz01.y);
          vertRL[k] = p0[i];
          ofPRL[k] = true;
          ++k;
        }
      }
    }

    const int q0[2] = {q1s, q1e};
    for (int i : {0, 1}) {
      const auto [s10, yz10] =
          Shadow01<false>(q0[i], p1, p1s, p1e, inQ, inP, mQ, mP);
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz10[0])) {
        s11 += s10 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s10 != 0) != shadows)) {
          shadows = s10 != 0;
          qRL[k] = inQ.vertPos_[q0[i]];
          pRL[k] = vec3(qRL[k].x, yz10.x, yz10.y);
          vertRL[k] = q0[i];
          ofPRL[k] = false;
          ++k;
        }
      }
    }

    if (s11 == 0) {  // No intersection
      xyzz11 = vec4(NAN);
    } else {
      DEBUG_ASSERT(k == 2, logicErr, "Boolean manifold error: s11");
      const double tol = TieTol(inP, inQ);
      xyzz11 = Intersect(pRL[0], pRL[1], qRL[0], qRL[1], tol);
      double z = xyzz11.z, w = xyzz11.w, mz = z, mw = w;
      if (std::abs(z - w) <= tol) {
        // the crossing as it is and as moved, by the same formula
        vec3 mpRL[2], mqRL[2];
        for (const int j : {0, 1}) {
          if (ofPRL[j]) {
            mpRL[j] = mP[vertRL[j]];
            const vec2 yz = InterpolateLine(mQ[q1s], mQ[q1e], mpRL[j].x);
            mqRL[j] = vec3(mpRL[j].x, yz.x, yz.y);
          } else {
            mqRL[j] = mQ[vertRL[j]];
            const vec2 yz = InterpolateLine(mP[p1s], mP[p1e], mqRL[j].x);
            mpRL[j] = vec3(mqRL[j].x, yz.x, yz.y);
          }
        }
        const vec2 zw = IntersectZ(pRL[0], pRL[1], qRL[0], qRL[1]);
        const vec2 mzw = IntersectZ(mpRL[0], mpRL[1], mqRL[0], mqRL[1]);
        z = zw[0];
        w = zw[1];
        mz = mzw[0];
        mw = mzw[1];
      }
      if (!Shadows(z, w, mz, mw, tol)) s11 = 0;
    }

    return std::make_pair(s11, xyzz11);
  }
};

template <bool expandP, bool forward>
struct Kernel02 {
  const Manifold::Impl& inA;
  const Manifold::Impl& inB;
  const Moved& mA;
  const Moved& mB;

  std::pair<int, double> operator()(int a0, int b2) {
    FaceEdge edgeB[3];
    LoadFaceEdges(inB.halfedge_, b2, edgeB);
    return (*this)(a0, b2, edgeB);
  }

  std::pair<int, double> operator()(int a0, int b2, const FaceEdge edgeB[3]) {
    int s02 = 0;
    double z02 = 0.0;

    // For yzzLR[k], k==0 is the left and k==1 is the right.
    int k = 0;
    vec3 yzzRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    // which of the face's edges each of yzzRL came from
    int edgeRL[2];

    for (const int i : {0, 1, 2}) {
      const auto syz01 = Shadow01<forward>(a0, edgeB[i].edge, edgeB[i].start,
                                           edgeB[i].end, inA, inB, mA, mB);
      const int s01 = syz01.first;
      const vec2 yz01 = syz01.second;
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz01[0])) {
        s02 += s01 * (forward == edgeB[i].isForward ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          edgeRL[k] = i;
          yzzRL[k++] = vec3(yz01[0], yz01[1], yz01[1]);
        }
      }
    }

    if (s02 == 0) {  // No intersection
      z02 = NAN;
    } else {
      DEBUG_ASSERT(k == 2, logicErr, "Boolean manifold error: s02");
      vec3 vertPosA = inA.vertPos_[a0];
      const double tol = TieTol(inA, inB);
      z02 = Interpolate(yzzRL[0], yzzRL[1], vertPosA.y, tol)[1];
      double z = z02, maz = vertPosA.z, mz02 = z02;
      if (std::abs(vertPosA.z - z02) <= tol) {
        // the face's z under the vertex as it is and as moved, by the same
        // formula
        const vec3 ma = mA[a0];
        vec3 zRL[2], mzRL[2];
        for (const int j : {0, 1}) {
          const FaceEdge& e = edgeB[edgeRL[j]];
          const vec2 yz = InterpolateLine(inB.vertPos_[e.start],
                                          inB.vertPos_[e.end], vertPosA.x);
          const vec2 myz = InterpolateLine(mB[e.start], mB[e.end], ma.x);
          zRL[j] = vec3(yz[0], yz[1], yz[1]);
          mzRL[j] = vec3(myz[0], myz[1], myz[1]);
        }
        z = InterpolateLine(zRL[0], zRL[1], vertPosA.y)[1];
        maz = ma.z;
        mz02 = InterpolateLine(mzRL[0], mzRL[1], ma.y)[1];
      }
      if (forward ? !Shadows(vertPosA.z, z, maz, mz02, tol)
                  : !Shadows(z, vertPosA.z, mz02, maz, tol))
        s02 = 0;
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

    const int edgeAStart = inA.halfedge_.Start(a1);
    const int edgeAEnd = inA.halfedge_.End(a1);
    FaceEdge edgeB[3];
    LoadFaceEdges(inB.halfedge_, b2, edgeB);

    for (int vertA : {edgeAStart, edgeAEnd}) {
      const auto [s, z] = k02(vertA, b2, edgeB);
      if (std::isfinite(z)) {
        x12 += s * ((vertA == edgeAStart) == forward ? 1 : -1);
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
      const auto [s, xyzz] = forward
                                 ? k11(a1, edgeAStart, edgeAEnd, edgeB[i].edge,
                                       edgeB[i].start, edgeB[i].end)
                                 : k11(edgeB[i].edge, edgeB[i].start,
                                       edgeB[i].end, a1, edgeAStart, edgeAEnd);
      if (std::isfinite(xyzz[0])) {
        x12 -= s * (edgeB[i].isForward ? 1 : -1);
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
      const double tol = TieTol(inA, inB);
      const vec4 xzyy =
          Intersect(xzyLR0[0], xzyLR0[1], xzyLR1[0], xzyLR1[1], tol);
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
Intersections Intersect12_(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                           const Moved& mP, const Moved& mQ,
                           ExecutionContext::Impl* ctx) {
  ZoneScoped;
  // Invariant: every ctx-passing parallel op is followed by IsCancelled to
  // keep partial output from feeding unconditional downstream consumers.
  // a: 1 (edge), b: 2 (face)
  const Manifold::Impl& a = forward ? inP : inQ;
  const Manifold::Impl& b = forward ? inQ : inP;

  Kernel02<expandP, forward> k02{a, b, forward ? mP : mQ, forward ? mQ : mP};
  Kernel11<expandP> k11{inP, inQ, mP, mQ};

  Kernel12<expandP, forward> k12{a, b, k02, k11};
  Kernel12Recorder<expandP, forward> recorder{k12, {}};
  const double tol = TieTol(inP, inQ);
  // Ties shift comparisons by up to tol, so a pair can interact a little
  // beyond tol once rounding is counted. Twice tol strictly contains that.
  auto f = [&a, tol](int i) {
    const int start = a.halfedge_.Start(i);
    const int end = a.halfedge_.End(i);
    return start < end
               ? Box(la::min(a.vertPos_[start], a.vertPos_[end]) - 2 * tol,
                     la::max(a.vertPos_[start], a.vertPos_[end]) + 2 * tol)
               : Box();
  };
  b.collider_.Collisions<false>(recorder, f, a.halfedge_.size(), true, ctx);
  if (IsCancelled(ctx)) return Intersections{};

  Intersections result = recorder.get();
  auto& p1q2 = result.p1q2;
  // sort p1q2 according to edges
  Vec<size_t> i12(p1q2.size());
  sequence(i12.begin(), i12.end());

  int index = forward ? 0 : 1;
  stable_sort(i12.begin(), i12.end(), [&](auto a, auto b) {
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
                          const Moved& mP, const Moved& mQ, bool expandP,
                          ExecutionContext::Impl* ctx) {
  if (expandP)
    return Intersect12_<true, forward>(inP, inQ, mP, mQ, ctx);
  else
    return Intersect12_<false, forward>(inP, inQ, mP, mQ, ctx);
}

template <bool expandP, bool forward>
Vec<int> Winding03_(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                    const Moved& mP, const Moved& mQ,
                    const VecView<std::array<int, 2>> p1q2,
                    ExecutionContext::Impl* ctx) {
  ZoneScoped;
  // a: 0 (vert), b: 2 (face)
  const Manifold::Impl& a = forward ? inP : inQ;
  const Manifold::Impl& b = forward ? inQ : inP;
  Vec<int> brokenHalfedges;
  int index = forward ? 0 : 1;

  // Invariant: every ctx-passing parallel op is followed by IsCancelled to
  // keep partial output from feeding unconditional downstream consumers.
  DisjointSets uA(a.vertPos_.size());
  for_each(autoPolicy(a.halfedge_.size()), countAt(0),
           countAt(static_cast<int>(a.halfedge_.size())), ctx, [&](int edge) {
             const int start = a.halfedge_.Start(edge);
             const int end = a.halfedge_.End(edge);
             if (start >= end) return;
             // check if the edge is broken
             auto it = std::lower_bound(
                 p1q2.begin(), p1q2.end(), edge,
                 [index](const std::array<int, 2>& collisionPair, int e) {
                   return collisionPair[index] < e;
                 });
             if (it == p1q2.end() || (*it)[index] != edge) uA.unite(start, end);
           });
  if (IsCancelled(ctx)) return Vec<int>{};

  // find components, the hope is the number of components should be small
  std::unordered_set<size_t> components;
#if (MANIFOLD_PAR == 1)
  if (a.vertPos_.size() > 1e5) {
    tbb::combinable<std::unordered_set<size_t>> componentsShared;
    for_each(autoPolicy(a.vertPos_.size()), countAt(0_uz),
             countAt(a.vertPos_.size()), ctx,
             [&](size_t v) { componentsShared.local().insert(uA.find(v)); });
    componentsShared.combine_each([&](const std::unordered_set<size_t>& data) {
      components.insert(data.begin(), data.end());
    });
  } else
#endif
  {
    for (size_t v = 0; v < a.vertPos_.size(); v++)
      components.insert(uA.find(v));
  }
  if (IsCancelled(ctx)) return Vec<int>{};
  Vec<int> verts;
  verts.reserve(components.size());
  for (size_t c : components) verts.push_back(static_cast<int>(c));

  Vec<int> w03(a.NumVert(), 0);
  Kernel02<expandP, forward> k02{a, b, forward ? mP : mQ, forward ? mQ : mP};
  auto recorderf = [&](int i, int b) {
    const auto [s02, z02] = k02(verts[i], b);
    // note that i is distinct on each thread, and verts contains unique
    // elements, so this does not require atomics
    if (std::isfinite(z02)) w03[verts[i]] += s02 * (forward ? 1 : -1);
  };
  auto recorder = MakeSimpleRecorder(recorderf);
  const double tol = TieTol(inP, inQ);
  auto f = [&](int i) {
    return VerticalQueryBox(a.vertPos_[verts[i]], 2 * tol);
  };
  b.collider_.Collisions<false>(recorder, f, verts.size(), true, ctx);
  if (IsCancelled(ctx)) return Vec<int>{};
  // flood fill
  for_each(autoPolicy(w03.size()), countAt(0_uz), countAt(w03.size()), ctx,
           [&](size_t i) {
             size_t root = uA.find(i);
             if (root == i) return;
             w03[i] = w03[root];
           });
  if (IsCancelled(ctx)) return Vec<int>{};
  return w03;
}

template <bool forward>
Vec<int> Winding03(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                   const Moved& mP, const Moved& mQ,
                   const VecView<std::array<int, 2>> p1q2, bool expandP,
                   ExecutionContext::Impl* ctx) {
  if (expandP)
    return Winding03_<true, forward>(inP, inQ, mP, mQ, p1q2, ctx);
  else
    return Winding03_<false, forward>(inP, inQ, mP, mQ, p1q2, ctx);
}
}  // namespace

namespace manifold {
Boolean3::Boolean3(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                   OpType op, ExecutionContext::Impl* ctx)
    : inP_(inP),
      inQ_(inQ),
      expandP_(op == OpType::Add),
      tol_(TieTol(inP, inQ)),
      ctx_(ctx) {
  ZoneScoped;
  // Symbolic perturbation:
  // Union -> expand inP, expand inQ
  // Difference, Intersection -> contract inP, expand inQ
  // Technically Intersection should contract inQ, but doing it this way makes
  // Split faster and any suboptimal cases seem pretty rare.

  constexpr size_t INT_MAX_SZ =
      static_cast<size_t>(std::numeric_limits<int>::max());

  const double tol = TieTol(inP, inQ);
  if (inP.IsEmpty() || inQ.IsEmpty() ||
      !Box(inP.bBox_.min - 2 * tol, inP.bBox_.max + 2 * tol)
           .DoesOverlap(inQ.bBox_)) {
    PRINT("No overlap, early out");
    w03_.resize(inP.NumVert(), 0);
    w30_.resize(inQ.NumVert(), 0);
    return;
  }

  // Phase-boundary fast-path: skip launching the next phase if cancel fired
  // between phases. The per-phase invariant is enforced inside the called
  // functions (Intersect12_, Winding03_).
#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
  Timer intersections;
  intersections.Start();
  Timer intersect12P, intersect12Q, winding03P, winding03Q;
  intersect12P.Start();
#endif

  // the moved verts that decide ties
  const Moved mP = Moved::P(inP, tol, expandP_);
  const Moved mQ = Moved::Q(inQ, tol);

  // Level 3
  // Build up the intersection of the edges and triangles, keeping only those
  // that intersect, and record the direction the edge is passing through the
  // triangle.
  if (IsCancelled(ctx_)) return;
  xv12_ = Intersect12<true>(inP, inQ, mP, mQ, expandP_, ctx_);
#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
  intersect12P.Stop();
  intersect12Q.Start();
#endif
  if (IsCancelled(ctx_)) return;
  xv21_ = Intersect12<false>(inP, inQ, mP, mQ, expandP_, ctx_);
#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
  intersect12Q.Stop();
#endif

  if (xv12_.x12.size() > INT_MAX_SZ || xv21_.x12.size() > INT_MAX_SZ) {
    valid = false;
    return;
  }

  // Compute winding numbers of all vertices using flood fill
  // Vertices on the same connected component have the same winding number
#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
  winding03P.Start();
#endif
  if (IsCancelled(ctx_)) return;
  w03_ = Winding03<true>(inP, inQ, mP, mQ, xv12_.p1q2, expandP_, ctx_);
#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
  winding03P.Stop();
  winding03Q.Start();
#endif
  if (IsCancelled(ctx_)) return;
  w30_ = Winding03<false>(inP, inQ, mP, mQ, xv21_.p1q2, expandP_, ctx_);
  // No trailing check: Winding03_ already returns empty on cancel and
  // Boolean3::Result re-checks on entry.

#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
  winding03Q.Stop();
  intersections.Stop();

  if (ManifoldParams().verbose >= 2) {
    intersect12P.Print("  Intersect12 P->Q");
    intersect12Q.Print("  Intersect12 Q->P");
    winding03P.Print("  Winding03 P");
    winding03Q.Print("  Winding03 Q");
    intersections.Print("Intersections (total)");
  }
#endif
}

Vec<int> Manifold::Impl::PointWinding(VecView<const vec3> points) const {
  ZoneScoped;
  Vec<int> winding(points.size(), 0);
  if (points.empty() || IsEmpty()) return winding;

  // Points outside the bounding box have winding 0 for any closed manifold.
  const Box tolBox(bBox_.min - epsilon_, bBox_.max + epsilon_);
  Vec<int> query2input;
  query2input.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i)
    if (tolBox.Contains(points[i])) query2input.push_back(static_cast<int>(i));
  if (query2input.empty()) return winding;

  // Build a minimal Impl for the query points. Kernel02 only reads vertPos_
  // and vertNormal_ from inA; halfedges and faces are not accessed.
  Impl pointImpl;
  pointImpl.epsilon_ = epsilon_;  // same tie tolerance as the booleans
  pointImpl.vertPos_.resize(query2input.size());
  pointImpl.vertNormal_.resize(query2input.size(), vec3(0.0));
  for (size_t i = 0; i < query2input.size(); ++i)
    pointImpl.vertPos_[i] = points[query2input[i]];

  // expandP=false: no symbolic perturbation from the query side (zero normals).
  // forward=true: project along +Z to count signed face crossings above each
  // point. The query box spans all Z and is widened by epsilon_ in XY,
  // so all faces with XY overlap within tolerance are returned regardless of Z.
  const double tol = epsilon_;
  const Moved still = Moved::Still(pointImpl);
  const Moved moved = Moved::Queried(*this, tol);
  Kernel02<false, true> k02{pointImpl, *this, still, moved};
  auto recorderf = [&](int localIdx, int tri) {
    const auto [s02, z02] = k02(localIdx, tri);
    if (std::isfinite(z02)) winding[query2input[localIdx]] += s02;
  };
  auto recorder = MakeSimpleRecorder(recorderf);
  auto f = [&](int i) {
    return VerticalQueryBox(pointImpl.vertPos_[i], epsilon_);
  };
  // Each queryIdx is processed by at most one thread (for_each_n), so the
  // per-queryIdx accumulation into winding[] is race-free without atomics.
  collider_.Collisions<false>(recorder, f, static_cast<int>(query2input.size()),
                              true);
  return winding;
}

std::vector<RayHit> Manifold::Impl::RayCast(vec3 origin, vec3 endpoint) const {
  ZoneScoped;
  if (IsEmpty()) return {};
  const vec3 dir = endpoint - origin;
  if (la::dot(dir, dir) == 0.0) return {};

  // Build a minimal single-edge Impl representing the ray segment.
  // Kernel12 treats inA as an edge mesh. Halfedges derives endVert from the
  // next edge in a triangle, so this helper uses a padded degenerate face:
  // edge 0 is the forward ray, edge 1 is its reverse, and edge 2 only closes
  // the local storage loop. Zero vertex normals and face normal mean the ray
  // contributes nothing to perturbation tiebreakers; consistency at shared
  // edges/vertices depends entirely on the mesh's own normals.
  Impl rayImpl;
  rayImpl.epsilon_ = epsilon_;  // same tie tolerance as the booleans
  rayImpl.vertPos_.resize(2);
  rayImpl.vertPos_[0] = origin;
  rayImpl.vertPos_[1] = endpoint;
  rayImpl.vertNormal_.resize(2);
  rayImpl.vertNormal_[0] = vec3(0.0);
  rayImpl.vertNormal_[1] = vec3(0.0);
  rayImpl.halfedge_.resize(3);
  rayImpl.halfedge_.Set(0, 0, 1, 0);  // forward: vert 0 → 1
  rayImpl.halfedge_.Set(1, 1, 0, 0);  // backward: vert 1 → 0
  rayImpl.halfedge_.Set(2, -1, -1, 0);
  rayImpl.faceNormal_.resize(1);
  rayImpl.faceNormal_[0] = vec3(0.0);

  // expandP=false with zero vertNormal means the ray-side perturbation is
  // zero. forward=true means we project along +Z for the lower-dimensional
  // kernel cascade (Shadow01 → Kernel02 → Kernel11 → Kernel12).
  const Moved still = Moved::Still(rayImpl);
  const Moved moved = Moved::Queried(*this, epsilon_);
  Kernel02<false, true> k02{rayImpl, *this, still, moved};
  Kernel11<false> k11{rayImpl, *this, still, moved};
  Kernel12<false, true> k12{rayImpl, *this, k02, k11};

  // Use the component with largest magnitude for stable t computation.
  const vec3 absDir = la::abs(dir);
  const int tAxis = absDir.x > absDir.y && absDir.x > absDir.z ? 0
                    : absDir.y > absDir.z                      ? 1
                                                               : 2;

  std::vector<RayHit> hits;
  // Query the BVH with the ray's AABB.
  const Box rayBox(la::min(origin, endpoint) - epsilon_,
                   la::max(origin, endpoint) + epsilon_);
  auto recorderf = [&](int /*queryIdx*/, int tri) {
    const auto [s, v] = k12(0, tri);  // halfedge 0 vs triangle tri
    if (s != 0 && std::isfinite(v.x)) {
      // v is the 3D intersection point computed by Kernel12.
      // Compute parametric t ∈ [0,1] along the ray segment.
      const double t = (v[tAxis] - origin[tAxis]) / dir[tAxis];
      if (t >= 0.0 && t <= 1.0) {
        // Kernel12's sign is negative where the ray enters the solid.
        hits.push_back(
            {static_cast<uint64_t>(tri), t, v, faceNormal_[tri], -s});
      }
    }
  };
  auto recorder = MakeSimpleRecorder(recorderf);
  auto f = [&rayBox](int) { return rayBox; };
  collider_.Collisions<false>(recorder, f, 1, false);

  // Break distance ties by faceID so the order doesn't depend on the
  // std::sort implementation, e.g. for canceling hits at a grazed vertex.
  std::sort(hits.begin(), hits.end(), [](const RayHit& a, const RayHit& b) {
    return a.distance != b.distance ? a.distance < b.distance
                                    : a.faceID < b.faceID;
  });
  return hits;
}

}  // namespace manifold
