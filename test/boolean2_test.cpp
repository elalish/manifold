// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../src/boolean2.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "manifold/common.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"

using namespace manifold;

namespace {

SimplePolygon RandomTopologicalRing(int n, uint64_t seed) {
  // std::mt19937_64 is portable, but std::uniform_real_distribution and
  // std::shuffle are not, and std::cos/std::sin vary by a ULP across libm
  // implementations. Build the ring from raw engine output with portable
  // integer/float arithmetic only, so the points are bit-identical on every
  // platform; with boolean2's exact predicates the resulting topology is then
  // portable too.
  std::mt19937_64 rng(seed);
  auto unit = [&rng]() {
    return static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);
  };
  std::vector<vec2> verts(n);
  for (int i = 0; i < n; ++i) verts[i] = {unit(), unit()};

  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  for (int i = n - 1; i > 0; --i) {
    const int j = static_cast<int>(rng() % static_cast<uint64_t>(i + 1));
    std::swap(order[i], order[j]);
  }

  SimplePolygon ring;
  ring.reserve(n);
  for (int i : order) ring.push_back(verts[i]);
  return ring;
}

template <typename Edge>
std::map<int, int> ComputeBalance(const std::vector<Edge>& edges) {
  std::map<int, int> balance;
  for (const auto& edge : edges) {
    balance[edge.v0] += edge.mult;
    balance[edge.v1] -= edge.mult;
  }
  return balance;
}

// The predicates below are an independent oracle for validating engine
// output, deliberately not reusing the engine's IntersectSegments (that
// would be circular) or the house CCW (whose zero band normalizes by the
// longer edge from p0, not by perpendicular distance). Each uses an exact
// distance-from-line band: |cross(d, p - a)| <= eps * |d|. They run only on
// retained engine output, where the validator has already rejected eps-zero
// edges, so |b - a| > eps holds and the band is well-defined - this is not a
// general collinearity test for sub-eps segments.
bool ValidVertId(int v, const std::vector<vec2>& verts) {
  return 0 <= v && v < static_cast<int>(verts.size());
}

bool PointsCollinearWithinEps(vec2 a, vec2 b, vec2 p, double eps) {
  const vec2 d = b - a;
  const double len2 = dot(d, d);
  if (len2 == 0.0) return false;
  return std::fabs(la::cross(d, p - a)) <= eps * la::length(d);
}

bool SegmentsHavePositiveCollinearOverlap(vec2 a0, vec2 a1, vec2 b0, vec2 b1,
                                          double eps) {
  if (!PointsCollinearWithinEps(a0, a1, b0, eps) ||
      !PointsCollinearWithinEps(a0, a1, b1, eps)) {
    return false;
  }
  const vec2 spread = la::abs(a1 - a0);
  const int axis = spread.x >= spread.y ? 0 : 1;
  const double aMin = std::min(Coord(a0, axis), Coord(a1, axis));
  const double aMax = std::max(Coord(a0, axis), Coord(a1, axis));
  const double bMin = std::min(Coord(b0, axis), Coord(b1, axis));
  const double bMax = std::max(Coord(b0, axis), Coord(b1, axis));
  return std::min(aMax, bMax) - std::max(aMin, bMin) > eps;
}

int StrictSide(vec2 a, vec2 b, vec2 p, double eps) {
  const vec2 d = b - a;
  const double threshold = eps * la::length(d);
  const double area = la::cross(d, p - a);
  return (area > threshold) - (area < -threshold);
}

bool SegmentsHaveStrictCrossing(vec2 a0, vec2 a1, vec2 b0, vec2 b1,
                                double eps) {
  const int b0Side = StrictSide(a0, a1, b0, eps);
  const int b1Side = StrictSide(a0, a1, b1, eps);
  const int a0Side = StrictSide(b0, b1, a0, eps);
  const int a1Side = StrictSide(b0, b1, a1, eps);
  return b0Side * b1Side < 0 && a0Side * a1Side < 0;
}

bool PointInSegmentInteriorBand(vec2 p, vec2 a, vec2 b, double eps) {
  const vec2 d = b - a;
  const double len2 = dot(d, d);
  if (len2 == 0.0) return false;
  const double along = dot(p - a, d);
  if (along <= 0.0 || along >= len2) return false;
  return std::fabs(la::cross(d, p - a)) <= eps * la::length(d);
}

// Independent oracle for the arrangement RemoveOverlaps2D retains - its
// predicates are not reused from the engine, so it can't rubber-stamp a bug.
// Checks: retained verts finite and >eps apart; edges valid and non-eps-zero;
// edge balance conserved (per-vertex signed multiplicity matches the input's,
// zero for introduced verts); no two non-adjacent edges strictly cross.
::testing::AssertionResult CheckRetainedGraphValidity(
    const OverlapResult& result, const std::vector<EdgeM>& inputEdges,
    const std::vector<int>& inputVert2Merged, int numMergedVerts, double eps) {
  auto fail = [](const std::string& msg) {
    return ::testing::AssertionFailure() << msg;
  };

  for (int v = 0; v < static_cast<int>(result.verts.size()); ++v) {
    if (!la::all(la::isfinite(result.verts[v]))) {
      std::ostringstream out;
      out << "non-finite retained vertex " << v << " = (" << result.verts[v].x
          << ", " << result.verts[v].y << ")";
      return fail(out.str());
    }
  }

  const double eps2 = eps * eps;
  for (int a = 0; a < static_cast<int>(result.verts.size()); ++a) {
    for (int b = a + 1; b < static_cast<int>(result.verts.size()); ++b) {
      if (dot(result.verts[b] - result.verts[a],
              result.verts[b] - result.verts[a]) <= eps2) {
        std::ostringstream out;
        out << "retained vertices " << a << " and " << b
            << " remain within epsilon";
        return fail(out.str());
      }
    }
  }

  for (int i = 0; i < static_cast<int>(result.edges.size()); ++i) {
    const auto& edge = result.edges[i];
    if (!ValidVertId(edge.v0, result.verts) ||
        !ValidVertId(edge.v1, result.verts)) {
      std::ostringstream out;
      out << "retained edge " << i << " has invalid verts " << edge.v0 << " -> "
          << edge.v1;
      return fail(out.str());
    }
    if (dot(result.verts[edge.v1] - result.verts[edge.v0],
            result.verts[edge.v1] - result.verts[edge.v0]) <= eps2) {
      std::ostringstream out;
      out << "retained edge " << i << " is epsilon-zero: " << edge.v0 << " -> "
          << edge.v1;
      return fail(out.str());
    }
  }

  std::vector<EdgeM> remapped;
  remapped.reserve(inputEdges.size());
  for (const auto& edge : inputEdges) {
    if (edge.v0 < 0 || edge.v0 >= static_cast<int>(inputVert2Merged.size()) ||
        edge.v1 < 0 || edge.v1 >= static_cast<int>(inputVert2Merged.size())) {
      std::ostringstream out;
      out << "input edge has verts outside inputVert2Merged: " << edge.v0
          << " -> " << edge.v1;
      return fail(out.str());
    }
    const int a = inputVert2Merged[edge.v0];
    const int b = inputVert2Merged[edge.v1];
    if (a != b) remapped.push_back({a, b, edge.mult});
  }

  const auto expected = ComputeBalance(remapped);
  const auto actual = ComputeBalance(result.edges);
  for (const auto& [v, actualBalance] : actual) {
    if (v < 0 || v >= static_cast<int>(result.verts.size())) {
      std::ostringstream out;
      out << "retained balance references invalid vertex " << v;
      return fail(out.str());
    }
    (void)actualBalance;
  }
  for (int v = 0; v < static_cast<int>(result.verts.size()); ++v) {
    const int expectedBalance =
        expected.count(v) ? expected.find(v)->second : 0;
    const int actualBalance = actual.count(v) ? actual.find(v)->second : 0;
    const int target = (v < numMergedVerts) ? expectedBalance : 0;
    if (actualBalance != target) {
      std::ostringstream out;
      out << "retained balance mismatch at vertex " << v << ": expected "
          << target << ", got " << actualBalance;
      return fail(out.str());
    }
  }

  for (int i = 0; i < static_cast<int>(result.edges.size()); ++i) {
    const auto& a = result.edges[i];
    for (int j = i + 1; j < static_cast<int>(result.edges.size()); ++j) {
      const auto& b = result.edges[j];
      if (a.v0 == b.v0 || a.v0 == b.v1 || a.v1 == b.v0 || a.v1 == b.v1) {
        continue;
      }
      const vec2 a0 = result.verts[a.v0];
      const vec2 a1 = result.verts[a.v1];
      const vec2 b0 = result.verts[b.v0];
      const vec2 b1 = result.verts[b.v1];
      if (SegmentsHaveStrictCrossing(a0, a1, b0, b1, eps)) {
        std::ostringstream out;
        out << "retained edges " << i << " and " << j
            << " still have a strict crossing";
        return fail(out.str());
      }
      if (SegmentsHavePositiveCollinearOverlap(a0, a1, b0, b1, eps)) {
        std::ostringstream out;
        out << "retained edges " << i << " and " << j
            << " still have positive collinear overlap";
        return fail(out.str());
      }
    }
  }

  for (int e = 0; e < static_cast<int>(result.edges.size()); ++e) {
    const auto& edge = result.edges[e];
    const vec2 a = result.verts[edge.v0];
    const vec2 b = result.verts[edge.v1];
    for (int v = 0; v < static_cast<int>(result.verts.size()); ++v) {
      if (v == edge.v0 || v == edge.v1) continue;
      if (PointInSegmentInteriorBand(result.verts[v], a, b, eps)) {
        std::ostringstream out;
        out << "retained vertex " << v << " lies in the interior band of edge "
            << e << " (" << edge.v0 << " -> " << edge.v1 << ")";
        return fail(out.str());
      }
    }
  }

  return ::testing::AssertionSuccess();
}

std::vector<EdgeM> EdgesFromOverlapResult(const OverlapResult& result) {
  std::vector<EdgeM> edges;
  edges.reserve(result.edges.size());
  for (const auto& edge : result.edges) {
    edges.push_back({edge.v0, edge.v1, edge.mult});
  }
  return edges;
}

OverlapResult CleanupPassLikeIterate(const OverlapResult& result, double eps) {
  return RemoveOverlaps2D(result.verts, EdgesFromOverlapResult(result), eps,
                          /*debug=*/false, WindRule::Add);
}

// Determinism/idempotence check: two arrangements equal up to eps-quantization,
// comparing sorted per-edge keys (endpoints rounded to eps*0.01 and canonically
// ordered, with signed multiplicity).
void ExpectSameFingerprint(const OverlapResult& a, const OverlapResult& b,
                           double eps) {
  using Fingerprint =
      std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int>>;
  auto fingerprint = [eps](const OverlapResult& r) {
    const double quantum = eps * 0.01;
    auto q = [quantum](double x) {
      return static_cast<int64_t>(std::round(x / quantum));
    };
    Fingerprint fp;
    fp.reserve(r.edges.size());
    for (const auto& edge : r.edges) {
      vec2 p0 = r.verts[edge.v0];
      vec2 p1 = r.verts[edge.v1];
      auto k0 = std::make_pair(q(p0.x), q(p0.y));
      auto k1 = std::make_pair(q(p1.x), q(p1.y));
      int mult = edge.mult;
      if (k1 < k0) {
        std::swap(k0, k1);
        mult = -mult;
      }
      fp.emplace_back(k0.first, k0.second, k1.first, k1.second, mult);
    }
    manifold::stable_sort(fp.begin(), fp.end());
    return fp;
  };

  const auto fpA = fingerprint(a);
  const auto fpB = fingerprint(b);
  EXPECT_EQ(fpA, fpB) << "fingerprints differ: lhs edges=" << a.edges.size()
                      << " rhs edges=" << b.edges.size();
}

std::pair<std::vector<vec2>, std::vector<EdgeM>> CombinedInput(
    const Polygons& a, const Polygons& b, int bMult) {
  auto [verts, edges] = PolygonsToInput(a);
  auto [bVerts, bEdges] = PolygonsToInput(b);
  const int base = static_cast<int>(verts.size());
  verts.insert(verts.end(), bVerts.begin(), bVerts.end());
  for (auto edge : bEdges) {
    edge.v0 += base;
    edge.v1 += base;
    edge.mult *= bMult;
    edges.push_back(edge);
  }
  return {std::move(verts), std::move(edges)};
}

}  // namespace

TEST(Boolean2, PropagatesShallowIndependentIntersections) {
  constexpr double eps = 1e-3;
  std::vector<vec2> verts = {{-1., 0.},
                             {1., 0.},
                             {0., -1.},
                             {0., 1.},
                             {-1., 5. * eps},
                             {1., 5. * eps},
                             {0., -1. + 5. * eps},
                             {0., 1. + 5. * eps},
                             {0., -1.},
                             {0., 1.}};
  std::vector<EdgeM> edges = {
      {0, 1, 1}, {2, 3, 1}, {4, 5, 1}, {6, 7, 1}, {8, 9, 1}};
  std::vector<Box2> edgeBoxes;
  edgeBoxes.reserve(edges.size());
  for (const EdgeM& edge : edges) {
    edgeBoxes.push_back(BoxOf2DEdge(verts[edge.v0], verts[edge.v1], eps));
  }
  const std::vector<std::pair<int, int>> pairs = {{0, 1}, {2, 3}};

  NarrowPhaseResult narrow =
      BuildListsAndFindIntersections(edges, verts, eps, pairs);
  IntersectionInsertion inserted = FindAndInsertIntersections(
      edges, std::move(verts), std::move(narrow.lists), eps, edgeBoxes, BVH{},
      narrow.intersections);

  ASSERT_EQ(inserted.verts.size(), 12);
  EXPECT_EQ(inserted.lists[4].size(), 2);
}

// Edge balance at a vertex is the signed sum of incident edge multiplicities
// (outgoing minus incoming); a valid arrangement conserves it. Asserts
// RemoveOverlaps2D does so through a boolean Add on a hard mixed-scale input
// (1e-6 alongside 1024). Replicated inline (CheckRetainedGraphValidity isn't
// public).
TEST(Boolean2, RemoveOverlaps2DTopologyMixedScale) {
  // Inputs A and B exactly as the BooleanRobustness fuzz target
  // constructs them from the counterexample raw polygons. The
  // topology check is on the OUTPUT of the boolean Add, not the
  // raw inputs.
  const Polygons polysA = {
      {{-1e-6, 1e-6}, {-1e-6, -1e-6}, {-0.0, 0.0}, {1.0, 1024.0}},
      {{1024.0, 1024.0}, {1.0, 1.0}, {1e-6, -1e-6}}};
  const Polygons polysB = {
      {{0.0, 1.0},
       {1e-6, 1024.0},
       {-1.0, -1024.0},
       {-0.0, 1024.0},
       {1.0, -1.0}},
      {{-260.66988565137592, -0.0},
       {0.0, -414.46576455279967},
       {-1.0, -1024.0}},
      {{1.0, 1.0}, {-1e-6, -0.0}, {1.0, 1024.0}, {1e-6, 0.0}},
      {{-111.03407576117854, 0.0},
       {560.59976308273758, 2.9313131393310714},
       {-1024.0, 1.0},
       {89.678187020143696, 0.0},
       {1024.0, 1.0}}};
  const CrossSection a(polysA);
  const CrossSection b(polysB);
  const auto result = a + b;
  const auto resultPolys = result.ToPolygons();

  // Run RemoveOverlaps2D on the Boolean's output and check that the
  // per-vertex edge balance is preserved (sum of mult on outgoing
  // minus incoming) for surviving verts, zero for newly-introduced.
  const auto [verts, edges] = manifold::PolygonsToInput(resultPolys);
  if (verts.empty()) GTEST_SKIP() << "Boolean output collapsed to empty";
  const double eps = manifold::InferEps(resultPolys, {});
  const auto overlapResult = manifold::RemoveOverlaps2D(verts, edges, eps);

  std::vector<manifold::EdgeM> remapped;
  for (const auto& edge : edges) {
    const int aIdx = overlapResult.inputVert2Merged[edge.v0];
    const int bIdx = overlapResult.inputVert2Merged[edge.v1];
    if (aIdx != bIdx) remapped.push_back({aIdx, bIdx, edge.mult});
  }
  const auto expected = ComputeBalance(remapped);
  const auto actual = ComputeBalance(overlapResult.edges);
  for (int v = 0; v < static_cast<int>(overlapResult.verts.size()); ++v) {
    const int expectedBalance =
        expected.count(v) ? expected.find(v)->second : 0;
    const int actualBalance = actual.count(v) ? actual.find(v)->second : 0;
    const int target = (v < overlapResult.numMergedVerts) ? expectedBalance : 0;
    EXPECT_EQ(actualBalance, target)
        << "Vertex " << v
        << " edge-balance mismatch after RemoveOverlaps2D on boolean "
        << "Add result: expected=" << target << " actual=" << actualBalance;
  }
}

TEST(Boolean2, MergeVertsTransitiveChainCanDriftPastEps) {
  const double eps = 1.0;
  const std::vector<vec2> verts = {{0.0, 0.0},  {0.99, 0.0}, {1.98, 0.0},
                                   {2.97, 0.0}, {3.96, 0.0}, {10.0, 0.0},
                                   {20.0, 0.0}};

  const VertexMerge merged = MergeVerts(verts, eps);

  ASSERT_EQ(merged.inputVert2Merged.size(), verts.size());
  ASSERT_EQ(merged.verts.size(), 3);
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(merged.inputVert2Merged[i], merged.inputVert2Merged[0]);
  EXPECT_NE(merged.inputVert2Merged[5], merged.inputVert2Merged[0]);
  EXPECT_NE(merged.inputVert2Merged[6], merged.inputVert2Merged[0]);

  EXPECT_NEAR(merged.verts[0].x, 1.98, 1e-12);
  EXPECT_NEAR(merged.verts[0].y, 0.0, 1e-12);
  const vec2 endpointDrift = verts.front() - merged.verts[0];
  EXPECT_GT(std::hypot(endpointDrift.x, endpointDrift.y), eps);
}

TEST(Boolean2, VertexMergeIdempotenceTightCluster) {
  const std::vector<vec2> verts = {{-564.24299726366871, -1.},
                                   {-564.25684749526681, -1.},
                                   {-564.25684749526681, -1.},
                                   {-564.25684749526681, 1.3146737456995536},
                                   {-564.25684749526681, 3.334547678102286},
                                   {-564.25684749526681, 924.18159456764056},
                                   {-564.25684749526681, -5.0212043784034019},
                                   {-564.2434248982521, -3.442978883496882},
                                   {-564.25684749526681, -882.0023276178074},
                                   {-564.25684749526681, -3.0103861859513601},
                                   {-564.25684749526681, -218.03838880073647},
                                   {-560.99098110791851, 1.1654515551817912},
                                   {-564.24548426671515, -361.18598495388369},
                                   {-564.25684749526681, -0.040195183151324976},
                                   {-564.25684749526681, -229.62237726036881},
                                   {-564.24797363468235, -3.4005396935541334},
                                   {-564.25105014358735, -0.37374103257085878},
                                   {-564.25684749526681, 3.993690857296496},
                                   {-564.25684749526681, 1.1420137899457909},
                                   {-564.2650553505373, 210.83410884574982},
                                   {-564.25684749526681, -3.4038895533070526},
                                   {-564.25684749526681, -4.3056143084054153},
                                   {-564.25684749526681, -2.8446598710193314},
                                   {-564.25684749526681, 0.90992952485448519},
                                   {-564.25684749526681, -5.1663919232428848},
                                   {-564.25684749526681, -502.42923749613101},
                                   {-564.25684749526681, -2.3459351240317718},
                                   {-564.2565950636108, 223.16662739130152},
                                   {-564.25684749526681, 810.70082359638945},
                                   {-564.25684749526681, -3.395314716558568},
                                   {-564.25684749526681, 0.055697227037768471},
                                   {-564.25684749526681, -1.6836095819963903}};
  const double eps = std::pow(10.0, -2.0433270966230594);
  const auto m1 = MergeVerts(verts, eps);
  ASSERT_FALSE(m1.verts.empty());
  const auto m2 = MergeVerts(m1.verts, eps);
  EXPECT_EQ(m2.verts.size(), m1.verts.size())
      << "MergeVerts not idempotent: pass1=" << m1.verts.size()
      << " pass2=" << m2.verts.size() << " (n=" << verts.size()
      << ", eps=" << eps << ")";
}

TEST(Boolean2, ValidatorRejectsRetainedVertsWithinEps) {
  const std::vector<vec2> verts = {{0.0, 0.0}, {10.0, 0.0}, {0.0, 10.0},
                                   {0.5, 0.5}, {20.0, 0.0}, {20.0, 10.0}};
  const std::vector<EdgeM> edges = {{0, 1, 1}, {1, 2, 1}, {2, 0, 1},
                                    {3, 4, 1}, {4, 5, 1}, {5, 3, 1}};
  const OverlapResult result{verts, edges, {0, 1, 2, 3, 4, 5}, 6};

  EXPECT_FALSE(CheckRetainedGraphValidity(
      result, edges, result.inputVert2Merged, result.numMergedVerts, 1.0));
}

TEST(Boolean2, ValidatorRejectsNearEndpointTJunction) {
  const std::vector<vec2> verts = {{0.0, 0.0},   {10.0, 0.0}, {0.5, 0.9},
                                   {10.0, 10.0}, {20.0, 5.0}, {20.0, 15.0}};
  const std::vector<EdgeM> edges = {{0, 1, 1}, {1, 3, 1}, {3, 0, 1},
                                    {2, 4, 1}, {4, 5, 1}, {5, 2, 1}};
  const OverlapResult result{verts, edges, {0, 1, 2, 3, 4, 5}, 6};

  EXPECT_FALSE(CheckRetainedGraphValidity(
      result, edges, result.inputVert2Merged, result.numMergedVerts, 1.0));
}

TEST(Boolean2, ValidatorRejectsRetainedStrictCrossing) {
  const std::vector<vec2> verts = {{0.0, 0.0},  {10.0, 10.0}, {0.0, 10.0},
                                   {10.0, 0.0}, {-10.0, 5.0}, {20.0, 5.0}};
  const std::vector<EdgeM> edges = {{0, 1, 1}, {1, 4, 1}, {4, 0, 1},
                                    {2, 3, 1}, {3, 5, 1}, {5, 2, 1}};
  const OverlapResult result{verts, edges, {0, 1, 2, 3, 4, 5}, 6};

  EXPECT_FALSE(CheckRetainedGraphValidity(
      result, edges, result.inputVert2Merged, result.numMergedVerts, 0.01));
}

TEST(Boolean2, CleanupPassMatchesValidAddSinglePass) {
  Polygons polys{RandomTopologicalRing(8, 15)};
  const double eps = InferEps(polys, {});
  const auto [verts, edges] = PolygonsToInput(polys);
  const auto pass1 =
      RemoveOverlaps2D(verts, edges, eps, /*debug=*/false, WindRule::Add);
  EXPECT_TRUE(CheckRetainedGraphValidity(pass1, edges, pass1.inputVert2Merged,
                                         pass1.numMergedVerts, eps));

  const auto pass2 = CleanupPassLikeIterate(pass1, eps);
  const auto pass2Input = EdgesFromOverlapResult(pass1);
  EXPECT_TRUE(CheckRetainedGraphValidity(
      pass2, pass2Input, pass2.inputVert2Merged, pass2.numMergedVerts, eps));
  ExpectSameFingerprint(pass1, pass2, eps);

  const auto pass3 = CleanupPassLikeIterate(pass2, eps);
  const auto pass3Input = EdgesFromOverlapResult(pass2);
  EXPECT_TRUE(CheckRetainedGraphValidity(
      pass3, pass3Input, pass3.inputVert2Merged, pass3.numMergedVerts, eps));
  ExpectSameFingerprint(pass2, pass3, eps);

  const auto pass1Polys = OutEdgesToPolygons(pass1.verts, pass1.edges);
  const auto pass2Polys = OutEdgesToPolygons(pass2.verts, pass2.edges);
  ASSERT_EQ(pass1Polys.size(), 2);
  ASSERT_EQ(pass2Polys.size(), 2);
  EXPECT_EQ(pass1Polys[0].size(), pass2Polys[0].size());
  EXPECT_EQ(pass1Polys[1].size(), pass2Polys[1].size());
  EXPECT_NEAR(TotalSignedArea(pass1Polys), TotalSignedArea(pass2Polys), 1e-12);
}

TEST(Boolean2, OffsetRoundUsesRequestedSegments) {
  SimplePolygon square = {{0, 0}, {20, 0}, {20, 20}, {0, 20}};
  const int segments = 20;
  const double delta = 5.0;

  Polygons rounded = Offset({square}, delta, JoinType::Round, 2.0, segments);

  ASSERT_EQ(rounded.size(), 1);
  EXPECT_EQ(rounded[0].size(), segments + 4);
}

TEST(Boolean2, OffsetRoundClampsHugeSegmentCount) {
  // An absurd requested count is clamped so the vertex count cannot blow up;
  // the clamp is kMaxRoundJoinSegments == 1 << 15 (file-local in offset.cpp).
  SimplePolygon square = {{0, 0}, {20, 0}, {20, 20}, {0, 20}};
  Polygons rounded = Offset({square}, 5.0, JoinType::Round, 2.0, 1 << 20);
  ASSERT_EQ(rounded.size(), 1);
  EXPECT_EQ(rounded[0].size(), (1 << 15) + 4);
}

TEST(Boolean2, OffsetZeroDeltaRejectsNonFiniteInput) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  SimplePolygon bad = {{0.0, 0.0}, {1.0, 0.0}, {nan, 1.0}, {0.0, 1.0}};

  EXPECT_TRUE(Offset({bad}, 0.0, JoinType::Miter).empty());
}

TEST(Boolean2, OffsetExtremeFiniteEdgesStayFinite) {
  // Large-but-finite coordinates must not yield inf/nan. Collinearity uses the
  // canonical CCW predicate, whose area products saturate above ~1e77, so the
  // exercised scale stays under that (still far beyond any real input).
  const double s = 1e70;
  const double delta = 1e60;
  SimplePolygon square = {{0, 0}, {s, 0}, {s, s}, {0, s}};

  Polygons out = Offset({square}, delta, JoinType::Bevel);

  ASSERT_FALSE(out.empty());
  for (const SimplePolygon& ring : out) {
    for (const vec2& v : ring) {
      EXPECT_TRUE(la::all(la::isfinite(v)));
    }
  }
}

TEST(Boolean2, OffsetSquareJoinKeepsCapOnSharpSpike) {
  // A near-degenerate spike (base 100, height 1e-9) survives regularization but
  // drives the square-join half-angle to ~90 degrees, where cosHalf can round
  // below 0. The caps must still be emitted (area ~204, two ~unit caps) rather
  // than silently dropped (which previously shrank the area toward ~200).
  SimplePolygon spike = {{0, 0}, {100, 0}, {50, 1e-9}};

  Polygons out = Offset({spike}, 1.0, JoinType::Square);

  ASSERT_FALSE(out.empty());
  double area = 0;
  for (const SimplePolygon& ring : out) area += SignedArea(ring);
  EXPECT_GT(area, 203.0);
  EXPECT_LT(area, 206.0);
}

TEST(Boolean2, DecomposeContainmentBboxUsesTolerance) {
  const double eps = EpsilonFromScale(0.5);
  const double d = 0.25 * eps;
  SimplePolygon outer = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  SimplePolygon hole = {
      {1 + d, 0.25}, {0.25, 0.25}, {0.25, 0.75}, {1 + d, 0.75}};

  std::vector<Polygons> components = DecomposeByContainment({outer, hole});

  ASSERT_EQ(components.size(), 1);
  ASSERT_EQ(components[0].size(), 2);
}

TEST(Boolean2, DecomposeContainmentDropsDegenerateRings) {
  SimplePolygon outer = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  SimplePolygon line = {{0.25, 0.25}, {0.75, 0.75}};

  std::vector<Polygons> components = DecomposeByContainment({outer, {}, line});

  ASSERT_EQ(components.size(), 1);
  ASSERT_EQ(components[0].size(), 1);
  EXPECT_EQ(components[0][0].size(), outer.size());
}

TEST(Boolean2, OffsetSquareInsetCapStaysOnSolidSide) {
  // L-shape: 2x2 square minus the top-right [1,2]x[1,2] quadrant (CCW). The
  // reflex corner at (1,1) routes through the square join when inset.
  SimplePolygon L = {{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}};

  Polygons out = Offset({L}, -0.25, JoinType::Square);

  // Load-bearing: the square cap belongs on the inset (solid) side, never in
  // the removed notch (1,2)x(1,2). The wrong-side bug puts cap vertices at
  // ~(1.25,1.10)/(1.10,1.25), strictly inside the notch.
  for (const SimplePolygon& ring : out) {
    for (const vec2& v : ring) {
      const bool inNotch = v.x > 1.0 && v.x < 2.0 && v.y > 1.0 && v.y < 2.0;
      EXPECT_FALSE(inNotch) << "cap vertex (" << v.x << ", " << v.y
                            << ") is in the removed notch";
    }
  }
  // Secondary (coarse): the wrong-side cap inflates the area (~1.40 buggy vs
  // ~1.26 correct). Upper bound discriminates the two; not an equality.
  EXPECT_LT(std::fabs(TotalSignedArea(out)), 1.35);

  // Dilate is unchanged by the fix (delta == absDelta); guard it stays sane.
  Polygons grown = Offset({L}, 0.25, JoinType::Square);
  EXPECT_GT(std::fabs(TotalSignedArea(grown)), 3.0);
}

TEST(Boolean2, OffsetLargeMiterLimitStaysBounded) {
  // Thin spike: apex (0,1e6) with near-antiparallel edge normals (1+dotN ~ 1
  // ULP), so the apex miter is near-unbounded. A huge miter_limit must not let
  // MiterPoint escape; the square fallback must engage.
  SimplePolygon spike = {{-0.01291544, 0}, {0.01291544, 0}, {0, 1e6}};

  Polygons out = Offset({spike}, 1.0, JoinType::Miter, 1e10);

  double maxCoord = 0;
  for (const SimplePolygon& ring : out)
    for (const vec2& v : ring)
      maxCoord = std::max(maxCoord, la::maxelem(la::abs(v)));
  // Apex sits at y=1e6; a capped corner stays near that scale. The unclamped
  // miter escapes to ~6e7.
  EXPECT_LT(maxCoord, 1.0e7) << "miter escaped to " << maxCoord;

  bool leftCap = false;
  bool rightCap = false;
  for (const SimplePolygon& ring : out) {
    for (const vec2& v : ring) {
      if (v.y > 9.9e5 && v.y < 1.01e6) {
        leftCap |= v.x < -0.5;
        rightCap |= v.x > 0.5;
      }
    }
  }
  EXPECT_TRUE(leftCap);
  EXPECT_TRUE(rightCap);
}

TEST(Boolean2, OffsetReversalSpikeKeepsFarCap) {
  // A near-zero-width spike whose tip edges are antiparallel triggers the
  // collinear-reversal branch; the offset must still cap the far side rather
  // than collapse the tip to a single point.
  SimplePolygon spike = {
      {-100, -100}, {100, -100}, {100, -1}, {1100, -1}, {100, -1 + 5.63085e-10},
      {100, 100},   {-100, 100}};
  Polygons out = Offset({spike}, 5.0, JoinType::Miter);
  // The tip's far offset endpoint is ~(1100, 4); only the near side (1100,-6)
  // survives if the reversal cap is dropped.
  bool farCap = false;
  for (const SimplePolygon& ring : out)
    for (const vec2& v : ring)
      if (v.x > 1090.0 && v.y > 0.0) farCap = true;
  EXPECT_TRUE(farCap) << "far side of the reversal spike was dropped";
}

TEST(Boolean2, DecomposeContainmentDropsZeroAreaRing) {
  SimplePolygon outer = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
  SimplePolygon collinear = {{1, 1}, {2, 1}, {3, 1}};  // zero area, size 3
  auto comps = DecomposeByContainment({outer, collinear});
  ASSERT_EQ(comps.size(), 1);
  EXPECT_EQ(comps[0].size(), 1);  // sliver dropped, not leaked as a hole
}

TEST(Boolean2, DecomposeContainmentKeepsFiniteThinHole) {
  SimplePolygon outer = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
  SimplePolygon thinHole = {{1, 1}, {1, 1.000001}, {9, 1.000001}, {9, 1}};

  auto comps = DecomposeByContainment({outer, thinHole});

  ASSERT_EQ(comps.size(), 1);
  ASSERT_EQ(comps[0].size(), 2);
  EXPECT_NEAR(TotalSignedArea(comps[0]), 100.0 - 8e-6, 1e-12);
}

TEST(Boolean2, DecomposeContainmentKeepsNestedPositiveRing) {
  SimplePolygon A = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};  // +100
  SimplePolygon B = {{3, 3}, {7, 3}, {7, 7}, {3, 7}};      // +16, inside A
  SimplePolygon H = {{4, 4}, {4, 6}, {6, 6}, {6, 4}};      // -4, inside B
  auto comps = DecomposeByContainment({A, B, H});
  double retained = 0;
  for (const auto& c : comps) retained += TotalSignedArea(c);
  // No silent area loss: the nested positive ring and its hole are kept.
  EXPECT_NEAR(retained, 112.0, 1e-9);
}

TEST(Boolean2, DecomposeContainmentToleranceIsSizeScaledOffOrigin) {
  // Off-origin rings: the containment epsilon must scale with the bbox SIZE,
  // not the coordinate magnitude. A hole sticking out by 1e-10 (above the
  // size-scaled eps, below a position-scaled one) is not contained, so it is
  // dropped. A position-based scale would inflate eps and wrongly attach it.
  const double k = 1000.0;
  const double d = 1e-10;
  SimplePolygon outer = {{k, k}, {k + 1, k}, {k + 1, k + 1}, {k, k + 1}};
  SimplePolygon hole = {{k + 1 + d, k + 0.25},
                        {k + 0.25, k + 0.25},
                        {k + 0.25, k + 0.75},
                        {k + 1 + d, k + 0.75}};
  auto comps = DecomposeByContainment({outer, hole});
  ASSERT_EQ(comps.size(), 1);
  EXPECT_EQ(comps[0].size(), 1);  // hole sticks out past the size-scaled eps
}

// Output topology balance check fails after Add on raw
// multi-contour, near-duplicate-heavy polygons.
TEST(Boolean2, BooleanRobustnessMergeTopologyBalance) {
  const Polygons a = {{{1024., 1024.},
                       {1., 1.},
                       {9.9999999999999995e-07, -9.9999999999999995e-07}}};
  const Polygons b = {{{0., 1.},
                       {9.9999999999999995e-07, 1024.},
                       {-1., -1024.},
                       {-0., 1024.},
                       {1., -1.}},
                      {{1., 1.},
                       {-9.9999999999999995e-07, -0.},
                       {1., 1024.},
                       {9.9999999999999995e-07, 0.}},
                      {{-111.03407576117854, 0.},
                       {560.59976308273758, 2.9313131393310714},
                       {-1024., 1.},
                       {89.678187020143696, 0.},
                       {1024., 1.}},
                      {{0., 0.},
                       {3.2408667776584608, 1024.},
                       {-1000., 9.9999999999999995e-07},
                       {1024., -1.},
                       {-1., -0.}}};
  const auto [verts, edges] = CombinedInput(a, b, /*bMult=*/1);
  ASSERT_FALSE(verts.empty());
  const double eps = InferEps(a, b);
  const auto overlap =
      RemoveOverlaps2D(verts, edges, eps, /*debug=*/false, WindRule::Add);
  EXPECT_TRUE(CheckRetainedGraphValidity(
      overlap, edges, overlap.inputVert2Merged, overlap.numMergedVerts, eps));
}

// Direct-cast disconnected-component winding fallback emitted an
// open spur after Add.
TEST(Boolean2, BooleanRobustnessDirectCastKeepsExpectedArea) {
  const Polygons a = {{{-9.9999999999999995e-07, 9.9999999999999995e-07},
                       {-9.9999999999999995e-07, -9.9999999999999995e-07},
                       {-0., 0.},
                       {1., 1024.}},
                      {{1024., 1024.},
                       {1., 1.},
                       {9.9999999999999995e-07, -9.9999999999999995e-07}}};
  const Polygons b = {{{0., 1.},
                       {9.9999999999999995e-07, 1024.},
                       {-1., -1024.},
                       {-0., 1024.},
                       {1., -1.}},
                      {{1., 1.},
                       {-9.9999999999999995e-07, -0.},
                       {1., 1024.},
                       {9.9999999999999995e-07, 0.}},
                      {{-111.03407576117854, 0.},
                       {560.59976308273758, 2.9313131393310714},
                       {-1024., 1.},
                       {89.678187020143696, 0.},
                       {1024., 1.}}};
  const auto [verts, edges] = CombinedInput(a, b, /*bMult=*/1);
  ASSERT_FALSE(verts.empty());
  const double eps = InferEps(a, b);
  const auto overlap =
      RemoveOverlaps2D(verts, edges, eps, /*debug=*/false, WindRule::Add);
  EXPECT_TRUE(CheckRetainedGraphValidity(
      overlap, edges, overlap.inputVert2Merged, overlap.numMergedVerts, eps));

  const auto polys = OutEdgesToPolygons(overlap.verts, overlap.edges);
  // Exact contour count and full-precision area were brittle on this
  // un-minimized seed. The open spur this guards manifests as a degenerate
  // ring, so assert no ring is degenerate directly (a zero-area spur would
  // slip past an area-only check); the coarser area tolerance still catches
  // real area loss.
  EXPECT_FALSE(polys.empty());
  for (const auto& ring : polys)
    EXPECT_GE(ring.size(), 3u) << "degenerate ring / open spur";
  EXPECT_NEAR(TotalSignedArea(polys), 1678.2538553263785,
              1e-6 * (1.0 + 1678.2538553263785));
}

// Regression: a single closed directed ring must round-trip through
// OutEdgesToPolygons. Earlier closure logic detected closure by trying to
// re-select the start edge as `next`, which is skipped by the visited guard;
// the natural walk terminates with destV == startV instead.
TEST(Boolean2, OutEdgesToPolygonsClosesSimpleRing) {
  const std::vector<vec2> verts = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  const std::vector<OutEdge> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
  const auto polys = OutEdgesToPolygons(verts, edges);
  ASSERT_EQ(polys.size(), 1u);
  EXPECT_EQ(polys[0].size(), 4u);
}

TEST(Boolean2, OutEdgesToPolygonsSplitsExactRepeatedVertex) {
  const std::vector<vec2> verts = {{0, 0}, {1, 0}, {1, -1}, {2, -1}, {1, 1}};
  const std::vector<OutEdge> edges = {{0, 1}, {1, 2}, {2, 3},
                                      {3, 1}, {1, 4}, {4, 0}};
  const auto polys = OutEdgesToPolygons(verts, edges);
  ASSERT_EQ(polys.size(), 2u);
  EXPECT_EQ(polys[0].size(), 3u);
  EXPECT_EQ(polys[1].size(), 3u);
  EXPECT_NEAR(TotalSignedArea(polys), 1.0, 1e-12);
}

TEST(Boolean2, OutEdgesToPolygonsKeepsNearDistinctVertex) {
  constexpr double kDelta = 1e-12;
  const std::vector<vec2> verts = {{0, 0},  {1, 0}, {1, -1},
                                   {2, -1}, {1, 1}, {1 + kDelta, 0}};
  const std::vector<OutEdge> edges = {{0, 1}, {1, 2}, {2, 3},
                                      {3, 5}, {5, 4}, {4, 0}};
  const auto polys = OutEdgesToPolygons(verts, edges);
  ASSERT_EQ(polys.size(), 1u);
  EXPECT_EQ(polys[0].size(), 6u);
  EXPECT_NEAR(TotalSignedArea(polys), 1.0 + kDelta, 1e-12);

  const CrossSection reconsumed(polys);
  EXPECT_FALSE(reconsumed.IsEmpty());
  EXPECT_NEAR(reconsumed.Area(), 1.0, 1e-9);
  const Manifold solid = Manifold::Extrude(reconsumed.ToPolygons(), 1.0);
  EXPECT_EQ(solid.Status(), Manifold::Error::NoError);
  EXPECT_NEAR(solid.Volume(), reconsumed.Area(), 1e-9);
}

TEST(Boolean2, KeepsNearDistinctPresentationVertex) {
  constexpr double kDelta = 1e-12;
  const Polygons input = {
      {{0, 0}, {1, 0}, {1, -1}, {2, -1}, {1 + kDelta, 0}, {1, 1}}};
  const auto polys = Boolean2D(input, {}, OpType::Add, /*eps=*/1e-15);
  ASSERT_EQ(polys.size(), 1u);
  EXPECT_EQ(polys[0].size(), 6u);
  EXPECT_NEAR(TotalSignedArea(polys), 1.0 + kDelta, 1e-12);
}

// The new-to-old snap is gated by perpendicular incidence (point-to-line
// distance), not point-to-point proximity: a generated crossing fuses into a
// nearby old corner only when one of its transversal source edges actually
// passes through that corner within the bounded eps perpendicular band. Mere
// proximity (within the 2*eps broad-phase band) is not enough - distinct
// near-corner crossings whose source edges miss the corner stay separate.
TEST(Boolean2, NewToOldMergeIsPerpIncidenceGated) {
  constexpr double eps = 1e-6;
  const Polygons a = {{{0, 0}, {10, 0}, {10, 1}, {0, 1}}};
  auto countNear = [](const std::vector<vec2>& points, vec2 target, double r) {
    int count = 0;
    for (const vec2& p : points)
      if (dot(p - target, p - target) <= r * r) ++count;
    return count;
  };
  const vec2 corner{0, 0};

  // Not incident: b's vertical left edge crosses a's bottom edge at (xLeft, 0),
  // a perpendicular distance xLeft from corner (0, 0). At xLeft = 1.5*eps the
  // crossing is inside the 2*eps proximity band but its only transversal source
  // edge (b's vertical left edge) is 1.5*eps off the corner, so it must stay a
  // distinct vertex rather than fuse.
  {
    const Polygons b = {
        {{1.5 * eps, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {1.5 * eps, 0.5}}};
    auto [verts, edges] = CombinedInput(a, b, /*bMult=*/1);
    auto result =
        RemoveOverlaps2D(verts, edges, eps, /*debug=*/false, WindRule::Add);
    EXPECT_EQ(countNear(result.verts, corner, 5 * eps), 2)
        << "non-incident near-corner crossing was over-fused";
    EXPECT_TRUE(CheckRetainedGraphValidity(
        result, edges, result.inputVert2Merged, result.numMergedVerts, eps));
  }

  // Incident: b's left edge is the sloped line through corner (0, 0), so the
  // crossing of that edge with a's bottom edge IS the corner; the FP
  // intersection lands within eps of (0, 0). Because the transversal source
  // edge genuinely runs through the corner, the crossing fuses to one vertex.
  // b's left edge runs from (-0.5, 0.5) down through (0, 0) to (0.5, -0.5).
  {
    const Polygons b = {{{0.5, -0.5}, {2.0, -0.5}, {2.0, 0.5}, {-0.5, 0.5}}};
    auto [verts, edges] = CombinedInput(a, b, /*bMult=*/1);
    auto result =
        RemoveOverlaps2D(verts, edges, eps, /*debug=*/false, WindRule::Add);
    EXPECT_EQ(countNear(result.verts, corner, 5 * eps), 1)
        << "genuinely incident crossing failed to fuse onto the corner";
    EXPECT_TRUE(CheckRetainedGraphValidity(
        result, edges, result.inputVert2Merged, result.numMergedVerts, eps));
  }
}

TEST(Boolean2, RemoveOverlapsMergesExactDuplicateCoordinates) {
  const Polygons polys = {{{0, 0}, {1, 0}, {0, 1}}, {{0, 0}, {-1, 0}, {0, -1}}};
  const auto [verts, edges] = PolygonsToInput(polys);
  ASSERT_EQ(verts.size(), 6u);
  const double eps = InferEps(polys, {});
  const auto overlap =
      RemoveOverlaps2D(verts, edges, eps, /*debug=*/false, WindRule::Add);
  ASSERT_EQ(overlap.inputVert2Merged.size(), verts.size());
  ASSERT_GE(overlap.inputVert2Merged[0], 0);
  ASSERT_LT(overlap.inputVert2Merged[0], overlap.numMergedVerts);
  ASSERT_GE(overlap.inputVert2Merged[3], 0);
  ASSERT_LT(overlap.inputVert2Merged[3], overlap.numMergedVerts);
  EXPECT_EQ(overlap.inputVert2Merged[0], overlap.inputVert2Merged[3]);
}

TEST(Boolean2, GraphOrderDetectsProperCrossing) {
  GraphSegment2D a{{0.0, 0.0}, {10.0, 10.0}, 0};
  GraphSegment2D b{{0.0, 10.0}, {10.0, 0.0}, 1};

  const auto order = CompareProjectedOrder(a, b, /*axis=*/0, 0.0, 10.0);
  EXPECT_EQ(order.atMinProjection, GraphOrderKind::ALessOrtho);
  EXPECT_EQ(order.atMaxProjection, GraphOrderKind::AGreaterOrtho);
  EXPECT_FALSE(order.coincidentOverlap);
  EXPECT_TRUE(order.properCrossing);
}

TEST(Boolean2, GraphOrderIsEndpointReversalStable) {
  GraphSegment2D a{{0.0, 0.0}, {10.0, 10.0}, 0};
  GraphSegment2D b{{0.0, 10.0}, {10.0, 0.0}, 1};
  GraphSegment2D aReversed{{10.0, 10.0}, {0.0, 0.0}, 0};
  GraphSegment2D bReversed{{10.0, 0.0}, {0.0, 10.0}, 1};

  const auto order = CompareProjectedOrder(a, b, /*axis=*/0, 0.0, 10.0);
  const auto reversed =
      CompareProjectedOrder(aReversed, bReversed, /*axis=*/0, 0.0, 10.0);
  EXPECT_EQ(order.atMinProjection, reversed.atMinProjection);
  EXPECT_EQ(order.atMaxProjection, reversed.atMaxProjection);
  EXPECT_EQ(order.coincidentOverlap, reversed.coincidentOverlap);
  EXPECT_EQ(order.properCrossing, reversed.properCrossing);
  EXPECT_EQ(order.atMinProjection, GraphOrderKind::ALessOrtho);
  EXPECT_EQ(order.atMaxProjection, GraphOrderKind::AGreaterOrtho);

  const auto aOnlyReversed =
      CompareProjectedOrder(aReversed, b, /*axis=*/0, 0.0, 10.0);
  const auto bOnlyReversed =
      CompareProjectedOrder(a, bReversed, /*axis=*/0, 0.0, 10.0);
  EXPECT_EQ(order.atMinProjection, aOnlyReversed.atMinProjection);
  EXPECT_EQ(order.atMaxProjection, aOnlyReversed.atMaxProjection);
  EXPECT_EQ(order.properCrossing, aOnlyReversed.properCrossing);
  EXPECT_EQ(order.atMinProjection, bOnlyReversed.atMinProjection);
  EXPECT_EQ(order.atMaxProjection, bOnlyReversed.atMaxProjection);
  EXPECT_EQ(order.properCrossing, bOnlyReversed.properCrossing);
}

TEST(Boolean2, GraphOrderSupportsYAxisProjection) {
  GraphSegment2D a{{0.0, 0.0}, {10.0, 10.0}, 0};
  GraphSegment2D b{{10.0, 0.0}, {0.0, 10.0}, 1};

  const auto order = CompareProjectedOrder(a, b, /*axis=*/1, 0.0, 10.0);
  EXPECT_EQ(order.atMinProjection, GraphOrderKind::ALessOrtho);
  EXPECT_EQ(order.atMaxProjection, GraphOrderKind::AGreaterOrtho);
  EXPECT_TRUE(order.properCrossing);
}

TEST(Boolean2, GraphOrderResolvesCoincidentOverlap) {
  GraphSegment2D a{{0.0, 0.0}, {10.0, 0.0}, 0};
  GraphSegment2D b{{0.0, 0.0}, {10.0, 0.0}, 1};

  const auto order = CompareProjectedOrder(a, b, /*axis=*/0, 0.0, 10.0);
  EXPECT_EQ(order.atMinProjection, GraphOrderKind::ALessOrtho);
  EXPECT_EQ(order.atMaxProjection, GraphOrderKind::ALessOrtho);
  EXPECT_TRUE(order.coincidentOverlap);
  EXPECT_FALSE(order.properCrossing);

  const auto swapped = CompareProjectedOrder(b, a, /*axis=*/0, 0.0, 10.0);
  EXPECT_EQ(swapped.atMinProjection, GraphOrderKind::AGreaterOrtho);
  EXPECT_EQ(swapped.atMaxProjection, GraphOrderKind::AGreaterOrtho);
  EXPECT_TRUE(swapped.coincidentOverlap);
  EXPECT_FALSE(swapped.properCrossing);
}

TEST(Boolean2, GraphOrderCanonicalGeometryTieBeforeEdgeId) {
  GraphSegment2D lower{{0.0, 0.0}, {10.0, 0.0}, 100};
  GraphSegment2D upper{{0.0, 0.5}, {10.0, 0.5}, 1};

  const auto order =
      CompareProjectedOrder(lower, upper, /*axis=*/0, 0.0, 10.0, 1.0);
  EXPECT_TRUE(order.coincidentOverlap);
  EXPECT_EQ(order.atMinProjection, GraphOrderKind::ALessOrtho);
  EXPECT_EQ(order.atMaxProjection, GraphOrderKind::ALessOrtho);
  EXPECT_FALSE(order.properCrossing);

  const auto swapped =
      CompareProjectedOrder(upper, lower, /*axis=*/0, 0.0, 10.0, 1.0);
  EXPECT_TRUE(swapped.coincidentOverlap);
  EXPECT_EQ(swapped.atMinProjection, GraphOrderKind::AGreaterOrtho);
  EXPECT_EQ(swapped.atMaxProjection, GraphOrderKind::AGreaterOrtho);
  EXPECT_FALSE(swapped.properCrossing);
}

TEST(Boolean2, GraphOrderKeepsEndpointTouchDegenerate) {
  GraphSegment2D a{{0.0, 0.0}, {10.0, 0.0}, 0};
  GraphSegment2D b{{5.0, 0.0}, {15.0, 1.0}, 1};

  const auto order = CompareProjectedOrder(a, b, /*axis=*/0, 5.0, 10.0);
  EXPECT_EQ(order.atMinProjection, GraphOrderKind::EndpointTouch);
  EXPECT_EQ(order.atMaxProjection, GraphOrderKind::ALessOrtho);
  EXPECT_FALSE(order.coincidentOverlap);
  EXPECT_FALSE(order.properCrossing);
}

// Consolidated Boolean2 IntersectSegments predicate cases, one row per case.
// Each row carries the two segments, eps, expected crossing, and (when it
// crosses) the expected point verbatim. atTol is that case's point tolerance:
// 1e-12 for the fixed-geometry cases, and the scale-derived eps for the
// shallow long-edge case (whose crossing is only resolved to within eps).
// IntersectSegments is a boolean2-internal predicate.
namespace {
struct SegCase {
  const char* name;
  GraphSegment2D a, b;
  double eps;
  bool crosses;
  vec2 at;       // expected intersection point, unused when !crosses
  double atTol;  // point tolerance, unused when !crosses
};

const SegCase kIntersectSegmentsSeeds[] = {
    {"FindsStrictCrossing",
     {{0.0, 0.0}, {10.0, 10.0}, 0},
     {{0.0, 10.0}, {10.0, 0.0}, 1},
     0.0,
     true,
     {5.0, 5.0},
     1e-12},
    {"KeepsOneSidedEpsBandCrossing",
     {{0.0, 0.0}, {10.0, 0.0}, 0},
     {{0.0, -0.5}, {10.0, 2.0}, 1},
     1.0,
     true,
     {2.0, 0.0},
     1e-12},
    {"KeepsTwoSidedEpsBandCrossing",
     {{0.0, 0.0}, {10.0, 0.0}, 0},
     {{0.0, -0.75}, {10.0, 0.75}, 1},
     1.0,
     true,
     {5.0, 0.0},
     1e-12},
    {"KeepsUnderflowingSignChange",
     {{0.0, 0.0}, {10.0, 0.0}, 0},
     {{0.0, -1e-200}, {10.0, 1e-200}, 1},
     1.0,
     true,
     {5.0, 0.0},
     1e-12},
    // A genuine transversal crossing that lands within eps of an endpoint is
    // kept, not dropped: the straddle is sign-confirmed and the near-endpoint
    // resolution is left to insertion-time snapping.
    {"KeepsEpsNearEndpointCrossing",
     {{0.0, 0.0}, {10.0, 0.0}, 0},
     {{0.5, -1.0}, {0.5, 1.0}, 1},
     1.0,
     true,
     {0.5, 0.0},
     1e-12},
    {"KeepsSteepInteriorCrossing",
     {{0.0, 0.0}, {0.0015, 1000.0}, 0},
     {{-1.0, 500.0}, {1.0, 500.0}, 1},
     0.001,
     true,
     {0.00075, 500.0},
     1e-12},
    {"DropsEndpointTouch",
     {{0.0, 0.0}, {10.0, 0.0}, 0},
     {{10.0, 0.0}, {20.0, 10.0}, 1},
     0.0,
     false,
     {},
     0.0},
    {"DropsPositiveOverlapTJunction",
     {{0.0, 0.0}, {10.0, 0.0}, 0},
     {{5.0, 0.0}, {15.0, 1.0}, 1},
     0.0,
     false,
     {},
     0.0},
    {"FindsAxisAlignedStrictCrossing",
     {{0.0, 5.0}, {10.0, 5.0}, 0},
     {{5.0, 0.0}, {5.0, 10.0}, 1},
     0.0,
     true,
     {5.0, 5.0},
     1e-12},
    {"DropsAxisAlignedEndpointTouch",
     {{0.0, 0.0}, {10.0, 0.0}, 0},
     {{10.0, 0.0}, {10.0, 10.0}, 1},
     0.0,
     false,
     {},
     0.0},
    {"DropsCoincidentOverlap",
     {{0.0, 0.0}, {10.0, 0.0}, 0},
     {{2.0, 0.0}, {8.0, 0.0}, 1},
     0.0,
     false,
     {},
     0.0},
    // Formerly standalone, near the merge-verts tests. The shallow long edge
    // crosses the short edge at the origin but only to within the scale eps,
    // so its point tolerance is eps, not 1e-12.
    {"ShallowLongEdgeIntersectionIsNotDropped",
     {{-0.5, 0.0}, {0.5, 0.0}, 0},
     {{-1e9, 0.01}, {1e9, -0.01}, 1},
     EpsilonFromScale(1e9),
     true,
     {0.0, 0.0},
     EpsilonFromScale(1e9)},
    {"NearEndpointIntersectionOutsideSegmentIsDropped",
     {{0.0, 0.0}, {10.0, 0.0}, 0},
     {{8.0, 0.4}, {18.0, 1.4}, 1},
     1.0,
     false,
     {},
     0.0},
    {"EndpointTJunctionIntersectionIsDropped",
     {{0.0, 0.0}, {10.0, 0.0}, 0},
     {{5.0, 0.0}, {5.0, 0.1}, 1},
     1.0,
     false,
     {},
     0.0},
};
}  // namespace

TEST(Boolean2, IntersectSegmentsSeeds) {
  for (const auto& c : kIntersectSegmentsSeeds) {
    SCOPED_TRACE(c.name);
    std::cerr << "[seed] " << c.name << std::endl;
    vec2 out;
    EXPECT_EQ(IntersectSegments(c.a, c.b, c.eps, &out), c.crosses);
    if (c.crosses) {
      EXPECT_NEAR(out.x, c.at.x, c.atTol);
      EXPECT_NEAR(out.y, c.at.y, c.atTol);
    }
  }
}

TEST(Boolean2, NonFiniteInputReturnsEmpty) {
  const double inf = std::numeric_limits<double>::infinity();
  Polygons bad = {{{0.0, 0.0}, {1.0, 0.0}, {inf, 1.0}, {0.0, 1.0}}};
  Polygons finite = {{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}};
  EXPECT_TRUE(Boolean2D(bad, finite, OpType::Add).empty());
}

// Near-degenerate cases at the eps scale.

// A near-origin degenerate Subtract whose regularized graph must stay valid; a
// missed split would leave a retained vertex inside another edge's interior
// band.
TEST(Boolean2, NearOriginSubtractStaysValid) {
  const Polygons a = {{{0, 0},
                       {0, -1.3875065265425851e-15},
                       {0.00013875065265428477, -7.4683028744398387e-16},
                       {0, 1.3875065265425851e-15}}};
  const Polygons b = {{{-2.7755575615628914e-16, -4.1136016361401099e-16},
                       {1.0547118733938987e-15, -8.4012326012156754e-16},
                       {4.0732494168083111e-05, 0.00012536172672923465},
                       {-1.609823385706477e-15, 1.7402932893545559e-17}},
                      {{3.3306690738754696e-16, -9.7814953183777337e-16},
                       {-4.9960036108132044e-16, 1.4436682794117381e-16},
                       {-0.00010663905417918063, -7.7477808007594485e-05},
                       {1.1657341758564144e-15, -2.1006658916167206e-15}}};
  const double eps = InferEps(a, b);
  const auto [verts, edges] = CombinedInput(a, b, /*bMult=*/-1);
  const auto result = RemoveOverlaps2D(verts, edges, eps);
  EXPECT_TRUE(CheckRetainedGraphValidity(result, edges, result.inputVert2Merged,
                                         result.numMergedVerts, eps));
}

// A tall thin rectangle (area d*h = 2.6) with two wider rectangles subtracted
// that overlap its left and right sides; their inner edges cross the short
// bottom edge 0.4 eps apart. The eps-scale merge fuses those near-coincident
// crossings, cancelling the subtraction cleanly back to the rectangle.
TEST(Boolean2, ShortEdgeFusion) {
  const double eps = 1e-6;
  const double d = 2.6 * eps, h = 1.0 / eps, x0 = 1.1 * eps, x1 = 1.5 * eps;
  const SimplePolygon rect = {{0, 0}, {d, 0}, {d, h}, {0, h}};
  const SimplePolygon left = {{-10 * eps, -0.20 * h},
                              {x0, -0.20 * h},
                              {x0, 0.25 * h},
                              {-10 * eps, 0.25 * h}};
  const SimplePolygon right = {{x1, -0.25 * h},
                               {d + 10 * eps, -0.25 * h},
                               {d + 10 * eps, 0.20 * h},
                               {x1, 0.20 * h}};
  const Polygons out =
      Boolean2D({rect, left, right}, {left, right}, OpType::Subtract, eps);
  EXPECT_NEAR(std::fabs(TotalSignedArea(out)), d * h, 1e-3 * d * h)
      << "subtracted rectangles did not cancel back to the rectangle";
}

// The Add/Intersect variant of the short-edge fusion above. Independent
// rectangle arithmetic: left/right are x-disjoint, each area 0.45h*1.11e-5 =
// 4.995; rect area d*h = 2.6; union = 12.59 - 0.275 - 0.22 = 12.095;
// union n (left u right) = left u right = 9.99. Those are the true areas; the
// engine fuses a sub-eps gap and lands within a loose eps*length band of them
// (see the body) - a below-resolution floor, not a regression.
TEST(Boolean2, ShortEdgeFusionAddIntersect) {
  const double eps = 1e-6;
  const double d = 2.6 * eps, h = 1.0 / eps, x0 = 1.1 * eps, x1 = 1.5 * eps;
  const SimplePolygon rect = {{0, 0}, {d, 0}, {d, h}, {0, h}};
  const SimplePolygon left = {{-10 * eps, -0.20 * h},
                              {x0, -0.20 * h},
                              {x0, 0.25 * h},
                              {-10 * eps, 0.25 * h}};
  const SimplePolygon right = {{x1, -0.25 * h},
                               {d + 10 * eps, -0.25 * h},
                               {d + 10 * eps, 0.20 * h},
                               {x1, 0.20 * h}};
  // The two probe edges sit 0.4 eps apart (x0 = 1.1 eps, x1 = 1.5 eps), so the
  // eps-scale merge fuses them - an in-budget sub-eps merge. The fused boundary
  // shifts by <= 0.4 eps, which the h = 1/eps edge levers into an O(0.4) area
  // wobble - observed 0.09 (Add) and 0.18 (Intersect), so the 5e-2 bound leaves
  // ~3-7x headroom (the feature is below the eps resolution). Intersect
  // additionally returns one contour rather than two; that is the same sub-eps
  // fusion and is accepted here, not a separate bug.
  EXPECT_NEAR(std::fabs(TotalSignedArea(Boolean2D(
                  {rect, left, right}, {left, right}, OpType::Add, eps))),
              12.095, 5e-2 * 12.095)
      << "union area outside the eps*length floor";
  EXPECT_NEAR(std::fabs(TotalSignedArea(Boolean2D(
                  {rect, left, right}, {left, right}, OpType::Intersect, eps))),
              9.99, 5e-2 * 9.99)
      << "intersection area outside the eps*length floor";
}
