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

#include "boolean2_diagnostics.h"

#ifdef MANIFOLD_DEBUG
#include <sstream>
#include <string>
#endif

namespace manifold {

#ifdef MANIFOLD_DEBUG
namespace {

std::string Id(const char* prefix, int i) {
  return std::string(prefix) + std::to_string(i);
}

std::string WindRuleName(WindRule rule) {
  switch (rule) {
    case WindRule::Add:
      return "Add";
    case WindRule::Intersect:
      return "Intersect";
  }
  return "Unknown";
}

void AddPoints(TracePhase* phase, const std::vector<vec2>& verts,
               const std::string& kind) {
  phase->points.reserve(phase->points.size() + verts.size());
  for (int i = 0; i < static_cast<int>(verts.size()); ++i) {
    phase->points.push_back({Id("v", i), verts[i], kind, "", ""});
  }
}

void AddInputEdges(TracePhase* phase, const std::vector<vec2>& verts,
                   const std::vector<EdgeM>& edges, const std::string& kind) {
  phase->segments.reserve(phase->segments.size() + edges.size());
  for (int i = 0; i < static_cast<int>(edges.size()); ++i) {
    const EdgeM& e = edges[i];
    if (e.v0 < 0 || e.v1 < 0 || e.v0 >= static_cast<int>(verts.size()) ||
        e.v1 >= static_cast<int>(verts.size())) {
      continue;
    }
    phase->segments.push_back(
        {Id("e", i), verts[e.v0], verts[e.v1], kind, "", e.mult, ""});
  }
}

void AddOutEdges(TracePhase* phase, const std::vector<vec2>& verts,
                 const std::vector<OutEdge>& edges, const std::string& kind) {
  phase->segments.reserve(phase->segments.size() + edges.size());
  for (int i = 0; i < static_cast<int>(edges.size()); ++i) {
    const OutEdge& e = edges[i];
    if (e.v0 < 0 || e.v1 < 0 || e.v0 >= static_cast<int>(verts.size()) ||
        e.v1 >= static_cast<int>(verts.size())) {
      continue;
    }
    phase->segments.push_back(
        {Id("out", i), verts[e.v0], verts[e.v1], kind, "", e.mult, ""});
  }
}

}  // namespace
#endif

TraceRecorder::TraceRecorder(Trace* trace, double eps, WindRule rule)
    : trace_(trace) {
#ifdef MANIFOLD_DEBUG
  if (trace_) {
    trace_->eps = eps;
    trace_->rule = WindRuleName(rule);
  }
#else
  (void)eps;
  (void)rule;
#endif
}

void TraceRecorder::RecordInput(const std::vector<vec2>& verts,
                                const std::vector<EdgeM>& edges) {
#ifdef MANIFOLD_DEBUG
  if (!trace_) return;
  TracePhase& phase = trace_->AddPhase("input");
  AddPoints(&phase, verts, "input_vertex");
  AddInputEdges(&phase, verts, edges, "input_edge");
#else
  (void)verts;
  (void)edges;
#endif
}

void TraceRecorder::RecordMergedVertices(
    const std::vector<vec2>& verts, const std::vector<int>& inputVert2Merged) {
#ifdef MANIFOLD_DEBUG
  if (!trace_) return;
  TracePhase& phase = trace_->AddPhase("merged_vertices");
  AddPoints(&phase, verts, "merged_vertex");
  phase.annotations.reserve(inputVert2Merged.size());
  for (int i = 0; i < static_cast<int>(inputVert2Merged.size()); ++i) {
    phase.annotations.push_back(
        {Id("in", i), "remap", std::to_string(inputVert2Merged[i])});
  }
#else
  (void)verts;
  (void)inputVert2Merged;
#endif
}

void TraceRecorder::RecordCollapsedEdges(const std::vector<vec2>& verts,
                                         const std::vector<EdgeM>& edges) {
#ifdef MANIFOLD_DEBUG
  if (!trace_) return;
  TracePhase& phase = trace_->AddPhase("collapsed_edges");
  AddPoints(&phase, verts, "merged_vertex");
  AddInputEdges(&phase, verts, edges, "collapsed_edge");
#else
  (void)verts;
  (void)edges;
#endif
}

void TraceRecorder::RecordFilteredOutput(const std::vector<vec2>& verts,
                                         const std::vector<OutEdge>& edges) {
#ifdef MANIFOLD_DEBUG
  if (!trace_) return;
  TracePhase& phase = trace_->AddPhase("filtered_output_edges");
  AddPoints(&phase, verts, "output_vertex");
  AddOutEdges(&phase, verts, edges, "retained_edge");
#else
  (void)verts;
  (void)edges;
#endif
}

}  // namespace manifold
