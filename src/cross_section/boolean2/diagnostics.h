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
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "manifold/manifold.h"
#endif

namespace manifold {
namespace boolean2 {

struct Trace;

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
  std::atomic<int64_t> mergeMergedComponents{0};
  std::atomic<int64_t> mergeMergedComponentsDriftGtEps{0};
  std::atomic<int64_t> mergeMaxMergedComponentVerts{0};
  std::atomic<int64_t> mergeMaxMergedRepresentativeDriftMilliEps{0};
  std::atomic<int64_t> mergeMaxMergedBboxDiagMilliEps{0};
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
    mergeMergedComponents = 0;
    mergeMergedComponentsDriftGtEps = 0;
    mergeMaxMergedComponentVerts = 0;
    mergeMaxMergedRepresentativeDriftMilliEps = 0;
    mergeMaxMergedBboxDiagMilliEps = 0;
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

namespace trace_detail {
inline void WriteEscaped(std::ostream& os, const std::string& s) {
  os << '"';
  for (char c : s) {
    switch (c) {
      case '"':
        os << "\\\"";
        break;
      case '\\':
        os << "\\\\";
        break;
      case '\n':
        os << "\\n";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\t':
        os << "\\t";
        break;
      default:
        os << c;
        break;
    }
  }
  os << '"';
}

inline void WriteVec2(std::ostream& os, const vec2& p) {
  os << '[' << p.x << ',' << p.y << ']';
}

inline void WriteField(std::ostream& os, const char* name,
                       const std::string& value, bool comma = true) {
  WriteEscaped(os, name);
  os << ':';
  WriteEscaped(os, value);
  if (comma) os << ',';
}

inline void WritePoint(std::ostream& os, const TracePoint& p) {
  os << '{';
  WriteField(os, "id", p.id);
  WriteEscaped(os, "xy");
  os << ':';
  WriteVec2(os, p.p);
  os << ',';
  WriteField(os, "kind", p.kind);
  WriteField(os, "source", p.source);
  WriteField(os, "label", p.label, false);
  os << '}';
}

inline void WriteSegment(std::ostream& os, const TraceSegment& s) {
  os << '{';
  WriteField(os, "id", s.id);
  WriteEscaped(os, "a");
  os << ':';
  WriteVec2(os, s.a);
  os << ',';
  WriteEscaped(os, "b");
  os << ':';
  WriteVec2(os, s.b);
  os << ',';
  WriteField(os, "kind", s.kind);
  WriteField(os, "source", s.source);
  WriteEscaped(os, "mult");
  os << ':' << s.mult << ',';
  WriteField(os, "label", s.label, false);
  os << '}';
}

inline void WritePolygon(std::ostream& os, const TracePolygon& p) {
  os << '{';
  WriteField(os, "id", p.id);
  WriteEscaped(os, "verts");
  os << ":[";
  for (size_t i = 0; i < p.verts.size(); ++i) {
    if (i > 0) os << ',';
    WriteVec2(os, p.verts[i]);
  }
  os << "],";
  WriteField(os, "kind", p.kind);
  WriteField(os, "source", p.source);
  WriteEscaped(os, "winding");
  os << ':' << p.winding << ',';
  WriteEscaped(os, "inside");
  os << ':' << (p.inside ? "true" : "false") << ',';
  WriteField(os, "label", p.label, false);
  os << '}';
}

inline void WriteAnnotation(std::ostream& os, const TraceAnnotation& a) {
  os << '{';
  WriteField(os, "target", a.target);
  WriteField(os, "key", a.key);
  WriteField(os, "value", a.value, false);
  os << '}';
}

template <typename T, typename F>
void WriteArray(std::ostream& os, const std::vector<T>& items, F writeItem) {
  os << '[';
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) os << ',';
    writeItem(os, items[i]);
  }
  os << ']';
}
}  // namespace trace_detail

inline void WriteTraceJson(std::ostream& os, const Trace& trace) {
  os << std::setprecision(17);
  os << "{\n";
  trace_detail::WriteEscaped(os, "eps");
  os << ':' << trace.eps << ",\n";
  trace_detail::WriteField(os, "rule", trace.rule);
  os << "\n";
  trace_detail::WriteEscaped(os, "phases");
  os << ":[\n";
  for (size_t i = 0; i < trace.phases.size(); ++i) {
    const TracePhase& phase = trace.phases[i];
    if (i > 0) os << ",\n";
    os << '{';
    trace_detail::WriteField(os, "name", phase.name);
    trace_detail::WriteEscaped(os, "points");
    os << ':';
    trace_detail::WriteArray(os, phase.points, trace_detail::WritePoint);
    os << ',';
    trace_detail::WriteEscaped(os, "segments");
    os << ':';
    trace_detail::WriteArray(os, phase.segments, trace_detail::WriteSegment);
    os << ',';
    trace_detail::WriteEscaped(os, "polygons");
    os << ':';
    trace_detail::WriteArray(os, phase.polygons, trace_detail::WritePolygon);
    os << ',';
    trace_detail::WriteEscaped(os, "annotations");
    os << ':';
    trace_detail::WriteArray(os, phase.annotations,
                             trace_detail::WriteAnnotation);
    os << '}';
  }
  os << "\n]\n}\n";
}
#endif

namespace timing_detail {
using Clock = std::chrono::steady_clock;
inline int64_t Ns(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}
}  // namespace timing_detail

}  // namespace boolean2
}  // namespace manifold
