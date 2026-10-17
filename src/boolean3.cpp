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

#include <array>
#include <limits>
#include <unordered_set>
#include <vector>

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

// Symbolic perturbation. Every vertex is displaced by an infinitesimal eps
// times its displacement - the verts of Q outward, the verts of P outward
// when expanding and inward otherwise; P is displaced once more by a smaller
// infinitesimal g, so that where P's and Q's faces coincide P's is the outer
// one alike on every side; and then P is translated along +x, +y and +z by
// infinitesimals tx >> ty >> tz. Each is infinitely smaller than anything
// before it. Every comparison between P and Q is decided as it would come out
// in that perturbed configuration, so all of them agree with one another.
//
// Two values within the tie tolerance (TieTol) are a tie: they are taken as
// equal, and the comparison is decided by the perturbation alone. The
// comparisons at a tie are signs of 2D cross products and 3x3
// determinants of vertex differences, so polynomials of degree at most 3 in
// (eps, g, tx, ty, tz). Their sign is that of the most significant nonzero
// coefficient: the fewest tz first, then ty, tx, g, then eps. Coefficients
// are computed one at a time in that order from the multilinear expansion,
// and the first nonzero one, nearly always the first, settles it.
constexpr int kVars = 6;  // the constant, then eps, g, tx, ty, tz

struct SymTables {
  // for each nonconstant monomial of degree <= 3, in order of significance,
  // the ways to pick one variable from each of 2 (resp. 3) factors to make it
  std::vector<std::vector<std::array<int, 2>>> picks2;
  std::vector<std::vector<std::array<int, 3>>> picks3;
  SymTables() {
    for (int tz = 0; tz <= 3; ++tz)
      for (int ty = 0; ty + tz <= 3; ++ty)
        for (int tx = 0; tx + ty + tz <= 3; ++tx)
          for (int g = 0; g + tx + ty + tz <= 3; ++g)
            for (int e = 0; e + g + tx + ty + tz <= 3; ++e) {
              const std::array<int, kVars> want = {0, e, g, tx, ty, tz};
              const int degree = e + g + tx + ty + tz;
              if (degree == 0) continue;
              auto matches = [&](std::initializer_list<int> pick) {
                std::array<int, kVars> n{};
                for (int v : pick) ++n[v];
                for (int k = 1; k < kVars; ++k)
                  if (n[k] != want[k]) return false;
                return true;
              };
              std::vector<std::array<int, 2>> p2;
              std::vector<std::array<int, 3>> p3;
              for (int a = 0; a < kVars; ++a)
                for (int b = 0; b < kVars; ++b) {
                  if (degree <= 2 && matches({a, b})) p2.push_back({a, b});
                  for (int c = 0; c < kVars; ++c)
                    if (matches({a, b, c})) p3.push_back({a, b, c});
                }
              if (degree <= 2) picks2.push_back(p2);
              picks3.push_back(p3);
            }
  }
};

inline const SymTables& Tables() {
  static const SymTables tables;
  return tables;
}

// A point or vector of the perturbed configuration: its value and its
// derivative along each infinitesimal, each with a bound on its magnitude.
// The displacements come from normals and the arithmetic rounds, so a
// coefficient within that error of zero is taken as zero - which is what it
// is whenever the configuration is symmetric, e.g. two edges in a plane that
// all the displacements involved lie in, where noise of either sign would
// otherwise decide.
struct SymVec {
  std::array<vec3, kVars> t{};
  std::array<double, kVars> mag{};
};

constexpr double kNoise = 64 * std::numeric_limits<double>::epsilon();

inline SymVec operator-(const SymVec& a, const SymVec& b) {
  SymVec r;
  for (int k = 0; k < kVars; ++k) {
    r.t[k] = a.t[k] - b.t[k];
    r.mag[k] = a.mag[k] + b.mag[k];
  }
  return r;
}

inline int Sign(double c, double bound) {
  return std::abs(c) <= kNoise * bound ? 0 : c > 0 ? 1 : -1;
}

// The sign of the most significant nonconstant term of component k of a, or 0.
inline int LeadSign1(const SymVec& a, int k) {
  for (int v = 1; v < kVars; ++v)
    if (const int s = Sign(a.t[v][k], a.mag[v])) return s;
  return 0;
}

// The sign of the most significant nonconstant term of Cross2(a, b), or 0.
inline int LeadSign2(const SymVec& a, const SymVec& b) {
  for (const auto& picks : Tables().picks2) {
    double c = 0, bound = 0;
    for (const auto& p : picks) {
      const vec3 &u = a.t[p[0]], &v = b.t[p[1]];
      c += u.x * v.y - u.y * v.x;
      bound += a.mag[p[0]] * b.mag[p[1]];
    }
    if (const int s = Sign(c, bound)) return s;
  }
  return 0;
}

// The sign of the most significant nonconstant term of Det3(a, b, c), or 0.
inline int LeadSign3(const SymVec& a, const SymVec& b, const SymVec& c) {
  for (const auto& picks : Tables().picks3) {
    double d = 0, bound = 0;
    for (const auto& p : picks) {
      d += la::dot(la::cross(a.t[p[0]], b.t[p[1]]), c.t[p[2]]);
      bound += a.mag[p[0]] * b.mag[p[1]] * c.mag[p[2]];
    }
    if (const int s = Sign(d, bound)) return s;
  }
  return 0;
}

using Disp = const vec3*;

constexpr double kPlaneTolerance = 1e-9;

inline SymVec SymVert(const Manifold::Impl& m, Disp d, int vert, bool isP,
                      bool expandP) {
  const vec3 pos = m.vertPos_[vert];
  const vec3 disp = isP && !expandP ? -d[vert] : d[vert];
  SymVec v;
  v.t[0] = pos;
  v.mag[0] = la::length(pos);
  v.t[1] = disp;
  v.mag[1] = std::max(1.0, la::length(disp));
  if (isP) {
    v.t[2] = disp;
    v.mag[2] = v.mag[1];
    for (const int i : {0, 1, 2}) {
      v.t[3 + i][i] = 1;
      v.mag[3 + i] = 1;
    }
  }
  return v;
}

// The displacement of each vertex: where it touches at most three distinct
// face planes, the one that moves each of them outward by exactly the same
// distance, so that a face stays parallel to itself and faces of P and Q
// that coincide stay tied, to be settled by the translation alike all over;
// elsewhere the vertex normal.
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
  return disp;
}

// Ties are rare, so their handling is kept out of line, leaving the hot
// comparisons as small as they were.
#if defined(_MSC_VER)
#define BOOLEAN3_COLD __declspec(noinline)
#define BOOLEAN3_HOT __forceinline
#else
#define BOOLEAN3_COLD __attribute__((noinline))
#define BOOLEAN3_HOT inline __attribute__((always_inline))
#endif
#if defined(__GNUC__) || defined(__clang__)
#define BOOLEAN3_TIE(x) __builtin_expect(!!(x), 0)
#else
#define BOOLEAN3_TIE(x) (x)
#endif

// P before Q in x, at a tie: by P minus Q along eps, then along g
// (P's own displacement), each taken as zero within the displacements'
// rounding error like every other term; then +x translation of P decides,
// putting P after Q.
template <bool expandP>
BOOLEAN3_COLD bool BeforeAtTie(vec3 dP, vec3 dQ) {
  const double dPx = withSign(expandP, dP.x);
  const double magP = std::max(1.0, la::length(dP));
  const double magQ = std::max(1.0, la::length(dQ));
  const double d = dPx - dQ.x;
  if (std::abs(d) > kNoise * (magP + magQ)) return d < 0;
  return std::abs(dPx) > kNoise * magP && dPx < 0;
}

template <bool expandP>
inline bool Before(double pX, double qX, Disp dP, int p, Disp dQ, int q,
                   double tol) {
  if (!BOOLEAN3_TIE(std::abs(pX - qX) <= tol)) return pX < qX;
  return BeforeAtTie<expandP>(dP[p], dQ[q]);
}

// Whether P is below Q in y where vertex a0 of A meets edge b1 of B, at a
// tie.
template <bool expandP, bool forward>
BOOLEAN3_COLD bool BelowAtTie01(const int a0, const int b1, const int b1s,
                                const int b1e, const Manifold::Impl& inA,
                                const Manifold::Impl& inB, Disp dA, Disp dB) {
  const vec3 s = inB.vertPos_[b1s];
  const vec3 e = inB.vertPos_[b1e];
  // a.y - (the edge's y at a.x) = Cross2(e - s, a - s) / (e.x - s.x);
  // P minus Q is that forward and its negation backward
  const SymVec sa = SymVert(inA, dA, a0, forward, expandP);
  const SymVec ss = SymVert(inB, dB, b1s, !forward, expandP);
  const SymVec se = SymVert(inB, dB, b1e, !forward, expandP);
  // e.x - s.x, or for an edge perpendicular to x, as the perturbation turns it
  const int dx = e.x > s.x ? 1 : e.x < s.x ? -1 : LeadSign1(se - ss, 0);
  if (dx == 0) {
    // Perpendicular to x even perturbed: the earlier, per-edge rule.
    const int b1pair = inB.halfedge_.Pair(b1);
    const double dir =
        inB.faceNormal_[b1 / 3].y + inB.faceNormal_[b1pair / 3].y;
    return (forward ? -dir : withSign(expandP, dir)) < 0;
  }
  return LeadSign2(se - ss, sa - ss) * dx * (forward ? 1 : -1) < 0;
}

template <bool expandP, bool forward>
BOOLEAN3_HOT std::pair<int, vec2> Shadow01(const int a0, const int b1,
                                           const int b1s, const int b1e,
                                           const Manifold::Impl& inA,
                                           const Manifold::Impl& inB, Disp dA,
                                           Disp dB) {
  const double a0x = inA.vertPos_[a0].x;
  const double b1sx = inB.vertPos_[b1s].x;
  const double b1ex = inB.vertPos_[b1e].x;
  const double tol = TieTol(inA, inB);
  int s01 = forward ? Before<expandP>(a0x, b1ex, dA, a0, dB, b1e, tol) -
                          Before<expandP>(a0x, b1sx, dA, a0, dB, b1s, tol)
                    : Before<expandP>(b1sx, a0x, dB, b1s, dA, a0, tol) -
                          Before<expandP>(b1ex, a0x, dB, b1e, dA, a0, tol);
  vec2 yz01(NAN);

  if (s01 != 0) {
    yz01 = Interpolate(inB.vertPos_[b1s], inB.vertPos_[b1e], a0x, tol);
    const double ay = inA.vertPos_[a0].y;
    if (!BOOLEAN3_TIE(std::abs(ay - yz01[0]) <= tol)) {
      if (forward ? !(ay < yz01[0]) : !(yz01[0] < ay)) s01 = 0;
    } else if (!BelowAtTie01<expandP, forward>(a0, b1, b1s, b1e, inA, inB, dA,
                                               dB)) {
      s01 = 0;
    }
  }
  return std::make_pair(s01, yz01);
}

// Whether edge p1 of P is below edge q1 of Q where they cross, at a tie.
template <bool expandP>
BOOLEAN3_COLD bool BelowAtTie11(int p1, int p1s, int p1e, int q1, int q1s,
                                int q1e, const Manifold::Impl& inP,
                                const Manifold::Impl& inQ, Disp dP, Disp dQ) {
  const vec3 p0 = inP.vertPos_[p1s], q0 = inQ.vertPos_[q1s];
  const vec3 dp = inP.vertPos_[p1e] - p0, dq = inQ.vertPos_[q1e] - q0;
  const double cross = dp.x * dq.y - dp.y * dq.x;
  // zP - zQ at the crossing = -Det3(dp, dq, q0 - p0) / Cross2(dp, dq)
  const SymVec sp0 = SymVert(inP, dP, p1s, true, expandP);
  const SymVec sq0 = SymVert(inQ, dQ, q1s, false, expandP);
  const SymVec sdp = SymVert(inP, dP, p1e, true, expandP) - sp0;
  const SymVec sdq = SymVert(inQ, dQ, q1e, false, expandP) - sq0;
  // Cross2(dp, dq), or for edges parallel in projection, as the perturbation
  // turns them
  const int c = cross > 0 ? 1 : cross < 0 ? -1 : LeadSign2(sdp, sdq);
  if (c == 0) {
    // Parallel in projection even perturbed: the earlier, per-edge rule.
    const int p1pair = inP.halfedge_.Pair(p1);
    const double dirP =
        inP.faceNormal_[p1 / 3].z + inP.faceNormal_[p1pair / 3].z;
    const int q1pair = inQ.halfedge_.Pair(q1);
    const double dirQ =
        inQ.faceNormal_[q1 / 3].z + inQ.faceNormal_[q1pair / 3].z;
    return withSign(expandP, dirP) - dirQ < 0;
  }
  return -LeadSign3(sdp, sdq, sq0 - sp0) * c < 0;
}

// Whether P is below Q in z where vertex a0 of A meets face b2 of B, at a
// tie.
template <bool expandP, bool forward>
BOOLEAN3_COLD bool BelowAtTie02(int a0, int b2, const Manifold::Impl& inA,
                                const Manifold::Impl& inB, Disp dA, Disp dB) {
  const int vb = inB.halfedge_.Start(3 * b2);
  const int vc = inB.halfedge_.Start(3 * b2 + 1);
  const int vd = inB.halfedge_.Start(3 * b2 + 2);
  const vec3 cb = inB.vertPos_[vc] - inB.vertPos_[vb];
  const vec3 db = inB.vertPos_[vd] - inB.vertPos_[vb];
  const double nz = cb.x * db.y - cb.y * db.x;
  // a.z - (the face's z at a) = Det3(c - b, d - b, a - b) / n.z;
  // P minus Q is that forward and its negation backward
  const SymVec sa = SymVert(inA, dA, a0, forward, expandP);
  const SymVec sb = SymVert(inB, dB, vb, !forward, expandP);
  const SymVec scb = SymVert(inB, dB, vc, !forward, expandP) - sb;
  const SymVec sdb = SymVert(inB, dB, vd, !forward, expandP) - sb;
  // n.z, or for a face parallel to z, as the perturbation turns it
  const int n = nz > 0 ? 1 : nz < 0 ? -1 : LeadSign2(scb, sdb);
  if (n == 0) {
    // Parallel to z even perturbed: the earlier, per-face rule.
    const double nBz = inB.faceNormal_[b2].z;
    return (forward ? -nBz : withSign(expandP, nBz)) < 0;
  }
  return LeadSign3(scb, sdb, sa - sb) * n * (forward ? 1 : -1) < 0;
}

template <bool expandP>
struct Kernel11 {
  const Manifold::Impl& inP;
  const Manifold::Impl& inQ;
  Disp dP;
  Disp dQ;

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

    const int p0[2] = {p1s, p1e};
    for (int i : {0, 1}) {
      const auto [s01, yz01] =
          Shadow01<expandP, true>(p0[i], q1, q1s, q1e, inP, inQ, dP, dQ);
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

    const int q0[2] = {q1s, q1e};
    for (int i : {0, 1}) {
      const auto [s10, yz10] =
          Shadow01<expandP, false>(q0[i], p1, p1s, p1e, inQ, inP, dQ, dP);
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
      const double tol = TieTol(inP, inQ);
      xyzz11 = Intersect(pRL[0], pRL[1], qRL[0], qRL[1], tol);
      if (!BOOLEAN3_TIE(std::abs(xyzz11.z - xyzz11.w) <= tol)) {
        if (!(xyzz11.z < xyzz11.w)) s11 = 0;
      } else if (!BelowAtTie11<expandP>(p1, p1s, p1e, q1, q1s, q1e, inP, inQ,
                                        dP, dQ)) {
        s11 = 0;
      }
    }

    return std::make_pair(s11, xyzz11);
  }
};

template <bool expandP, bool forward>
struct Kernel02 {
  const Manifold::Impl& inA;
  const Manifold::Impl& inB;
  Disp dA;
  Disp dB;

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

    for (const int i : {0, 1, 2}) {
      const auto syz01 = Shadow01<expandP, forward>(
          a0, edgeB[i].edge, edgeB[i].start, edgeB[i].end, inA, inB, dA, dB);
      const int s01 = syz01.first;
      const vec2 yz01 = syz01.second;
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz01[0])) {
        s02 += s01 * (forward == edgeB[i].isForward ? -1 : 1);
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
      const double tol = TieTol(inA, inB);
      z02 = Interpolate(yzzRL[0], yzzRL[1], vertPosA.y, tol)[1];
      if (!BOOLEAN3_TIE(std::abs(vertPosA.z - z02) <= tol)) {
        if (forward ? !(vertPosA.z < z02) : !(z02 < vertPosA.z)) s02 = 0;
      } else if (!BelowAtTie02<expandP, forward>(a0, b2, inA, inB, dA, dB)) {
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
                           Disp dP, Disp dQ, ExecutionContext::Impl* ctx) {
  ZoneScoped;
  // Invariant: every ctx-passing parallel op is followed by IsCancelled to
  // keep partial output from feeding unconditional downstream consumers.
  // a: 1 (edge), b: 2 (face)
  const Manifold::Impl& a = forward ? inP : inQ;
  const Manifold::Impl& b = forward ? inQ : inP;

  Kernel02<expandP, forward> k02{a, b, forward ? dP : dQ, forward ? dQ : dP};
  Kernel11<expandP> k11{inP, inQ, dP, dQ};

  Kernel12<expandP, forward> k12{a, b, k02, k11};
  Kernel12Recorder<expandP, forward> recorder{k12, {}};
  const double tol = TieTol(inP, inQ);
  auto f = [&a, tol](int i) {
    const int start = a.halfedge_.Start(i);
    const int end = a.halfedge_.End(i);
    return start < end ? Box(la::min(a.vertPos_[start], a.vertPos_[end]) - tol,
                             la::max(a.vertPos_[start], a.vertPos_[end]) + tol)
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
                          Disp dP, Disp dQ, bool expandP,
                          ExecutionContext::Impl* ctx) {
  if (expandP)
    return Intersect12_<true, forward>(inP, inQ, dP, dQ, ctx);
  else
    return Intersect12_<false, forward>(inP, inQ, dP, dQ, ctx);
}

template <bool expandP, bool forward>
Vec<int> Winding03_(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                    Disp dP, Disp dQ, const VecView<std::array<int, 2>> p1q2,
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
  Kernel02<expandP, forward> k02{a, b, forward ? dP : dQ, forward ? dQ : dP};
  auto recorderf = [&](int i, int b) {
    const auto [s02, z02] = k02(verts[i], b);
    // note that i is distinct on each thread, and verts contains unique
    // elements, so this does not require atomics
    if (std::isfinite(z02)) w03[verts[i]] += s02 * (forward ? 1 : -1);
  };
  auto recorder = MakeSimpleRecorder(recorderf);
  const double tol = TieTol(inP, inQ);
  auto f = [&](int i) { return VerticalQueryBox(a.vertPos_[verts[i]], tol); };
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
                   Disp dP, Disp dQ, const VecView<std::array<int, 2>> p1q2,
                   bool expandP, ExecutionContext::Impl* ctx) {
  if (expandP)
    return Winding03_<true, forward>(inP, inQ, dP, dQ, p1q2, ctx);
  else
    return Winding03_<false, forward>(inP, inQ, dP, dQ, p1q2, ctx);
}
}  // namespace

namespace manifold {
Boolean3::Boolean3(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                   OpType op, ExecutionContext::Impl* ctx)
    : inP_(inP), inQ_(inQ), expandP_(op == OpType::Add), ctx_(ctx) {
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
      !Box(inP.bBox_.min - tol, inP.bBox_.max + tol).DoesOverlap(inQ.bBox_)) {
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

  const Vec<vec3> dispP = Displacements(inP);
  const Vec<vec3> dispQ = Displacements(inQ);

  // Level 3
  // Build up the intersection of the edges and triangles, keeping only those
  // that intersect, and record the direction the edge is passing through the
  // triangle.
  if (IsCancelled(ctx_)) return;
  xv12_ =
      Intersect12<true>(inP, inQ, dispP.data(), dispQ.data(), expandP_, ctx_);
#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
  intersect12P.Stop();
  intersect12Q.Start();
#endif
  if (IsCancelled(ctx_)) return;
  xv21_ =
      Intersect12<false>(inP, inQ, dispP.data(), dispQ.data(), expandP_, ctx_);
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
  w03_ = Winding03<true>(inP, inQ, dispP.data(), dispQ.data(), xv12_.p1q2,
                         expandP_, ctx_);
#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
  winding03P.Stop();
  winding03Q.Start();
#endif
  if (IsCancelled(ctx_)) return;
  w30_ = Winding03<false>(inP, inQ, dispP.data(), dispQ.data(), xv21_.p1q2,
                          expandP_, ctx_);
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
  Kernel02<false, true> k02{pointImpl, *this, pointImpl.vertNormal_.data(),
                            vertNormal_.data()};
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
  Kernel02<false, true> k02{rayImpl, *this, rayImpl.vertNormal_.data(),
                            vertNormal_.data()};
  Kernel11<false> k11{rayImpl, *this, rayImpl.vertNormal_.data(),
                      vertNormal_.data()};
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
        hits.push_back({static_cast<uint64_t>(tri), t, v, faceNormal_[tri]});
      }
    }
  };
  auto recorder = MakeSimpleRecorder(recorderf);
  auto f = [&rayBox](int) { return rayBox; };
  collider_.Collisions<false>(recorder, f, 1, false);

  std::sort(hits.begin(), hits.end(), [](const RayHit& a, const RayHit& b) {
    return a.distance < b.distance;
  });
  return hits;
}

}  // namespace manifold
