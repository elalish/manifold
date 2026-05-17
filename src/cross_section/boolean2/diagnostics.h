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
// Optional diagnostics for the Boolean2 pipeline.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#ifdef MANIFOLD_DEBUG
#include "manifold/manifold.h"
#endif

namespace manifold {
namespace boolean2 {

struct PhaseAcc {
  std::atomic<int64_t> mergeNs{0};
  std::atomic<int64_t> remapNs{0};
  std::atomic<int64_t> buildListsNs{0};
  std::atomic<int64_t> findIxNs{0};
  std::atomic<int64_t> restructNs{0};
  std::atomic<int64_t> canonNs{0};
  std::atomic<int64_t> filterDcelNs{0};
  std::atomic<int64_t> totalNs{0};
  std::atomic<int64_t> cases{0};
  std::atomic<int64_t> bvhBuildNs{0};
  std::atomic<int64_t> broadPairWorkNs{0};
  std::atomic<int64_t> edgeVertListsNs{0};
  std::atomic<int64_t> intersectionBroadNs{0};
  std::atomic<int64_t> intersectionNarrowNs{0};
  std::atomic<int64_t> intersectionPropagationNs{0};
  std::atomic<int64_t> mergeBvhBuildNs{0};
  std::atomic<int64_t> mergeCollideNs{0};
  std::atomic<int64_t> mergeRestNs{0};
  std::atomic<int64_t> edgeVertCandidates{0};
  std::atomic<int64_t> edgeVertEndpointRejects{0};
  std::atomic<int64_t> edgeVertDegenerateRejects{0};
  std::atomic<int64_t> edgeVertTRangeRejects{0};
  std::atomic<int64_t> edgeVertDistanceRejects{0};
  std::atomic<int64_t> edgeVertApexRejects{0};
  std::atomic<int64_t> edgeVertHits{0};
  std::atomic<int64_t> edgeVertCalls{0};
  std::atomic<int64_t> edgeVertBruteCalls{0};
  std::atomic<int64_t> edgeVertVertexBvhCalls{0};
  std::atomic<int64_t> edgeVertBvhCalls{0};
  std::atomic<int64_t> edgeVertPairDerivedCalls{0};
  std::atomic<int64_t> edgeVertTotalEdges{0};
  std::atomic<int64_t> edgeVertTotalVerts{0};
  std::atomic<int64_t> edgeVertHitsFlat{0};
  std::atomic<int64_t> edgeVertBucketLt64{0};
  std::atomic<int64_t> edgeVertBucketLt256{0};
  std::atomic<int64_t> edgeVertBucketLt1024{0};
  std::atomic<int64_t> edgeVertBucketGe1024{0};
  std::atomic<int64_t> propagationCalls{0};
  std::atomic<int64_t> propagationSkippedNoNearDup{0};

  void Reset() {
    mergeNs = 0;
    remapNs = 0;
    buildListsNs = 0;
    findIxNs = 0;
    restructNs = 0;
    canonNs = 0;
    filterDcelNs = 0;
    totalNs = 0;
    cases = 0;
    bvhBuildNs = 0;
    broadPairWorkNs = 0;
    edgeVertListsNs = 0;
    intersectionBroadNs = 0;
    intersectionNarrowNs = 0;
    intersectionPropagationNs = 0;
    mergeBvhBuildNs = 0;
    mergeCollideNs = 0;
    mergeRestNs = 0;
    edgeVertCandidates = 0;
    edgeVertEndpointRejects = 0;
    edgeVertDegenerateRejects = 0;
    edgeVertTRangeRejects = 0;
    edgeVertDistanceRejects = 0;
    edgeVertApexRejects = 0;
    edgeVertHits = 0;
    edgeVertCalls = 0;
    edgeVertBruteCalls = 0;
    edgeVertVertexBvhCalls = 0;
    edgeVertBvhCalls = 0;
    edgeVertPairDerivedCalls = 0;
    edgeVertTotalEdges = 0;
    edgeVertTotalVerts = 0;
    edgeVertHitsFlat = 0;
    edgeVertBucketLt64 = 0;
    edgeVertBucketLt256 = 0;
    edgeVertBucketLt1024 = 0;
    edgeVertBucketGe1024 = 0;
    propagationCalls = 0;
    propagationSkippedNoNearDup = 0;
  }
};

inline PhaseAcc& GlobalPhases() {
  static PhaseAcc p;
  return p;
}

inline std::atomic<bool>& TimingEnabledFlag() {
  static std::atomic<bool> enabled{false};
  return enabled;
}

inline bool TimingEnabled() {
  return TimingEnabledFlag().load(std::memory_order_relaxed);
}

inline void SetTimingEnabled(bool enabled) {
  TimingEnabledFlag().store(enabled, std::memory_order_relaxed);
}

inline bool DebugVerbose(int level = 2) {
#ifdef MANIFOLD_DEBUG
  return ManifoldParams().verbose >= level;
#else
  (void)level;
  return false;
#endif
}

namespace timing_detail {
using Clock = std::chrono::steady_clock;
inline int64_t Ns(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}
}  // namespace timing_detail

}  // namespace boolean2
}  // namespace manifold
