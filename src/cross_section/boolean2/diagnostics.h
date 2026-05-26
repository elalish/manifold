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
#include <utility>
#include <vector>

#ifdef MANIFOLD_DEBUG
#include <string>

#include "manifold/manifold.h"
#endif

#include "canonicalize.h"

namespace manifold {
namespace boolean2 {

enum class WindRule;
struct Trace;

struct PhaseAcc {
  std::atomic<int64_t> mergeNs{0};
  std::atomic<int64_t> remapNs{0};
  std::atomic<int64_t> findIxNs{0};
  std::atomic<int64_t> nearbyIxMergeNs{0};
  std::atomic<int64_t> canonNs{0};
  std::atomic<int64_t> filterHalfedgeNs{0};
  std::atomic<int64_t> totalNs{0};
  std::atomic<int64_t> cases{0};
  std::atomic<int64_t> bvhBuildNs{0};
  std::atomic<int64_t> broadPairWorkNs{0};
  std::atomic<int64_t> narrowPhaseNs{0};

  void Reset() {
    mergeNs = 0;
    remapNs = 0;
    findIxNs = 0;
    nearbyIxMergeNs = 0;
    canonNs = 0;
    filterHalfedgeNs = 0;
    totalNs = 0;
    cases = 0;
    bvhBuildNs = 0;
    broadPairWorkNs = 0;
    narrowPhaseNs = 0;
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

namespace timing_detail {
using Clock = std::chrono::steady_clock;
inline int64_t Ns(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}
}  // namespace timing_detail

class ScopedTiming {
 public:
  explicit ScopedTiming(std::atomic<int64_t>& target)
      : target_(TimingEnabled() ? &target : nullptr),
        start_(target_ ? timing_detail::Clock::now()
                       : timing_detail::Clock::time_point{}) {}

  ~ScopedTiming() {
    if (target_) {
      target_->fetch_add(timing_detail::Ns(start_, timing_detail::Clock::now()),
                         std::memory_order_relaxed);
    }
  }

  ScopedTiming(const ScopedTiming&) = delete;
  ScopedTiming& operator=(const ScopedTiming&) = delete;

 private:
  std::atomic<int64_t>* target_;
  timing_detail::Clock::time_point start_;
};

inline void CountTimingCase() {
  if (TimingEnabled()) {
    GlobalPhases().cases.fetch_add(1, std::memory_order_relaxed);
  }
}

inline bool DebugVerbose(int level = 2) {
#ifdef MANIFOLD_DEBUG
  return ManifoldParams().verbose >= level;
#else
  (void)level;
  return false;
#endif
}

#ifdef MANIFOLD_DEBUG
struct TracePoint {
  std::string id;
  vec2 p;
  std::string kind;
  std::string source;
  std::string label;
};

struct TraceSegment {
  std::string id;
  vec2 a;
  vec2 b;
  std::string kind;
  std::string source;
  int mult = 0;
  std::string label;
};

struct TracePolygon {
  std::string id;
  std::vector<vec2> verts;
  std::string kind;
  std::string source;
  int winding = 0;
  bool inside = false;
  std::string label;
};

struct TraceAnnotation {
  std::string target;
  std::string key;
  std::string value;
};

struct TracePhase {
  std::string name;
  std::vector<TracePoint> points;
  std::vector<TraceSegment> segments;
  std::vector<TracePolygon> polygons;
  std::vector<TraceAnnotation> annotations;
};

struct Trace {
  double eps = 0.0;
  std::string rule;
  std::vector<TracePhase> phases;

  TracePhase& AddPhase(std::string name) {
    phases.push_back({});
    phases.back().name = std::move(name);
    return phases.back();
  }
};
#endif

class TraceRecorder {
 public:
  TraceRecorder(Trace* trace, double eps, WindRule rule);

  void RecordInput(const std::vector<vec2>& verts,
                   const std::vector<EdgeM>& edges);
  void RecordMergedVertices(const std::vector<vec2>& verts,
                            const std::vector<int>& inputVert2Merged);
  void RecordCollapsedEdges(const std::vector<vec2>& verts,
                            const std::vector<EdgeM>& edges);
  void RecordBroadPhasePairs(const std::vector<vec2>& verts,
                             const std::vector<EdgeM>& edges,
                             const std::vector<std::pair<int, int>>& pairs);
  void RecordEdgeVertLists(const std::vector<vec2>& verts,
                           const std::vector<EdgeM>& edges,
                           const std::vector<std::vector<int>>& lists);
  void RecordInsertedIntersections(const std::vector<vec2>& verts,
                                   const std::vector<EdgeM>& edges,
                                   const std::vector<std::vector<int>>& lists);
  void RecordNearbyIntersectionMerge(
      const std::vector<vec2>& verts, const std::vector<EdgeM>& edges,
      const std::vector<std::vector<int>>& lists);
  void RecordCanonicalSubedges(const std::vector<vec2>& verts,
                               const CanonicalSubEdges& canon);
  void RecordFilteredOutput(const std::vector<vec2>& verts,
                            const std::vector<OutEdge>& edges);

 private:
  Trace* trace_;
};

}  // namespace boolean2
}  // namespace manifold
