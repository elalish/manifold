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
//
// Smith section 7.7 / figure 7.16: residual eps-scale edge intersections
// from rounded arithmetic in the first pass are resolved by re-applying
// the algorithm. IterateToFixedPoint runs up to maxIter additional
// passes after the initial one. Fingerprint helpers quantize vert
// positions so cross-iteration vert renumbering doesn't break the
// equality check.

#include <algorithm>
#include <cmath>
#include <cstdint>
#ifdef MANIFOLD_DEBUG
#include <iostream>
#endif
#include <tuple>
#include <utility>
#include <vector>

#include "diagnostics.h"
#include "driver.h"
#include "iterate.h"
#include "predicates.h"

namespace manifold {
namespace boolean2 {

namespace {

// Fingerprints need to ignore sub-eps floating-point noise between cleanup
// iterations while still detecting eps-scale topology changes. eps/100 leaves
// two decimal digits inside the algorithm's geometric tolerance.
constexpr double kFingerprintQuantumEpsFraction = 0.01;

}  // namespace

// =============================================================================
// Iterate-to-fixed-point.
//
// Smith §7.7 / figure 7.16: residual eps-scale edge intersections from
// rounded arithmetic in the first pass are resolved by re-applying the
// algorithm. We iterate up to maxIter additional times beyond the initial
// pass. Termination:
//   - Converged: new fingerprint == previous fingerprint (no change).
//   - Cycle: new fingerprint == any earlier (non-immediate) fingerprint.
//     Pick the iteration with the lex-smallest fingerprint as canonical
//     (deterministic choice among the cycle's equivalent outputs).
//   - MaxedOut: hit maxIter without convergence/cycle; return current.
//
// Fingerprint quantizes vert positions to multiples of eps/100 so that
// cross-iteration vert renumbering doesn't break comparison.
//
// Returns the sorted (vMin.x, vMin.y, vMax.x, vMax.y, mult) tuple
// vector directly: vector equality / lexicographic less-than give us
// the convergence and cycle-detection comparisons we need, without
// stringifying through `<sstream>` / `<string>`.
using FingerprintData =
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int>>;

FingerprintData FingerprintAt(const OverlapResult& r, double quantum) {
  auto q = [quantum](double x) {
    return static_cast<int64_t>(std::round(x / quantum));
  };
  FingerprintData subs;
  subs.reserve(r.edges.size());
  for (const auto& oe : r.edges) {
    vec2 p0 = r.verts[oe.v0];
    vec2 p1 = r.verts[oe.v1];
    auto ka = std::make_pair(q(p0.x), q(p0.y));
    auto kb = std::make_pair(q(p1.x), q(p1.y));
    int mult = oe.mult;
    if (kb < ka) {
      std::swap(ka, kb);
      mult = -mult;
    }
    subs.emplace_back(ka.first, ka.second, kb.first, kb.second, mult);
  }
  manifold::stable_sort(subs.begin(), subs.end());
  return subs;
}

// Fine-grained fingerprint used for idempotence detection. Empirical:
// anything finer than ~eps/100 starts catching pure FP-noise differences.
FingerprintData Fingerprint(const OverlapResult& r, double eps) {
  return FingerprintAt(r, eps * kFingerprintQuantumEpsFraction);
}

// Smith §7.7 / fig 7.16 proves convergence in ≤2 iterations under his
// α-budget framework when intersection positions are tracked symbolically.
// With FP-rounded positions out of `IntersectSegments`, a bounded cleanup pass
// handles residual eps-scale intersections.
//
// `pred` is the first-pass region predicate. Cleanup passes operate on the
// already-filtered output boundary, whose interior is normalized to positive
// winding, so they use the default Add rule.
OverlapResult IterateToFixedPoint(const std::vector<vec2>& vIn,
                                  const std::vector<EdgeM>& eIn, double eps,
                                  int maxIter, int* outIters,
                                  IterStatus* outStatus, WindRule pred) {
  std::vector<OverlapResult> history;
  std::vector<FingerprintData> fps;
  history.push_back(RemoveOverlaps2D(vIn, eIn, eps, /*debug=*/false, pred));
  fps.push_back(Fingerprint(history.back(), eps));
  // composedRemap[orig_input_vert] = current iteration's vert idx. Updated
  // each iteration so callers can validate the final result against the
  // original input. Without this, only first-pass `inputRemap` is meaningful.
  std::vector<int> composedRemap = history.back().inputRemap;
  for (int iter = 1; iter <= maxIter; ++iter) {
    std::vector<EdgeM> nextEdges;
    nextEdges.reserve(history.back().edges.size());
    for (const auto& oe : history.back().edges)
      nextEdges.push_back({oe.v0, oe.v1, oe.mult});
    auto next = RemoveOverlaps2D(history.back().verts, nextEdges, eps);
    // Compose: orig->prev_iter via composedRemap, then prev_iter->next via
    // next.inputRemap.
    for (auto& v : composedRemap) v = next.inputRemap[v];
    next.inputRemap = composedRemap;
    auto nextFp = Fingerprint(next, eps);
    if (nextFp == fps.back()) {
      if (outIters) *outIters = iter;
      if (outStatus) *outStatus = IterStatus::Converged;
      return next;
    }
    // Cycle detection: same fingerprint seen earlier (not just last).
    for (size_t k = 0; k + 1 < fps.size(); ++k) {
      if (fps[k] == nextFp) {
        if (outIters) *outIters = iter;
        if (outStatus) *outStatus = IterStatus::Cycled;
        // Lex-smallest fingerprint wins. Each history entry stores the
        // composed remap for that iteration, so returning an older canonical
        // cycle representative keeps inputRemap consistent with its verts.
        size_t minIdx = 0;
        for (size_t j = 1; j < fps.size(); ++j) {
          if (fps[j] < fps[minIdx]) minIdx = j;
        }
        if (nextFp < fps[minIdx]) {
          return next;
        }
        return std::move(history[minIdx]);
      }
    }
    history.push_back(std::move(next));
    fps.push_back(std::move(nextFp));
  }
  if (outIters) *outIters = maxIter;
  if (outStatus) *outStatus = IterStatus::MaxedOut;
#ifdef MANIFOLD_DEBUG
  if (DebugVerbose()) {
    std::cout << "[IterateToFixedPoint] hit MaxedOut at iter=" << maxIter
              << " (fingerprint did not match prior pass and no cycle "
                 "detected)\n";
  }
#endif
  history.back().inputRemap = std::move(composedRemap);
  return std::move(history.back());
}

}  // namespace boolean2
}  // namespace manifold
