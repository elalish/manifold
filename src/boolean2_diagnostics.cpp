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

void AddSplitSubsegments(TracePhase* phase, const std::vector<vec2>& verts,
                         const std::vector<EdgeM>& edges,
                         const std::vector<std::vector<int>>& lists,
                         const std::string& kind) {
  for (int eId = 0; eId < static_cast<int>(edges.size()); ++eId) {
    const EdgeM& edge = edges[eId];
    if (edge.v0 < 0 || edge.v1 < 0 ||
        edge.v0 >= static_cast<int>(verts.size()) ||
        edge.v1 >= static_cast<int>(verts.size())) {
      continue;
    }
    if (eId >= static_cast<int>(lists.size()) || lists[eId].empty()) {
      phase->segments.push_back({Id("e", eId), verts[edge.v0], verts[edge.v1],
                                 kind + "_unsplit", "", edge.mult, ""});
      continue;
    }
    const auto& list = lists[eId];
    int v0 = edge.v0;
    for (int i = 0; i <= static_cast<int>(list.size()); ++i) {
      const int v1 = i == static_cast<int>(list.size()) ? edge.v1 : list[i];
      if (v0 < 0 || v1 < 0 || v0 >= static_cast<int>(verts.size()) ||
          v1 >= static_cast<int>(verts.size()) || v0 == v1) {
        v0 = v1;
        continue;
      }
      std::ostringstream label;
      label << "edge " << eId << " split " << i;
      phase->segments.push_back(
          {std::string("s") + std::to_string(eId) + "_" + std::to_string(i),
           verts[v0], verts[v1], kind, "", edge.mult, label.str()});
      v0 = v1;
    }
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

void AddCanonicalEdges(TracePhase* phase, const std::vector<vec2>& verts,
                       const CanonicalSubEdges& canon) {
  phase->segments.reserve(phase->segments.size() + canon.edges.size());
  for (int i = 0; i < static_cast<int>(canon.edges.size()); ++i) {
    const CanonEdge& e = canon.edges[i];
    if (e.vMin < 0 || e.vMax < 0 || e.vMin >= static_cast<int>(verts.size()) ||
        e.vMax >= static_cast<int>(verts.size())) {
      continue;
    }
    phase->segments.push_back({Id("c", i), verts[e.vMin], verts[e.vMax],
                               "canonical_subedge", "", e.mult, ""});
  }
}

void AddCandidatePairs(TracePhase* phase, const std::vector<vec2>& verts,
                       const std::vector<EdgeM>& edges,
                       const std::vector<std::pair<int, int>>& pairs) {
  phase->segments.reserve(phase->segments.size() + 2 * pairs.size());
  for (int i = 0; i < static_cast<int>(pairs.size()); ++i) {
    const auto& pair = pairs[i];
    const int edgeIds[2] = {pair.first, pair.second};
    for (int side = 0; side < 2; ++side) {
      const int edgeId = edgeIds[side];
      if (edgeId < 0 || edgeId >= static_cast<int>(edges.size())) continue;
      const EdgeM& e = edges[edgeId];
      if (e.v0 < 0 || e.v1 < 0 || e.v0 >= static_cast<int>(verts.size()) ||
          e.v1 >= static_cast<int>(verts.size())) {
        continue;
      }
      std::ostringstream label;
      label << "candidate pair " << pair.first << "," << pair.second;
      phase->segments.push_back({Id("pair", i * 2 + side), verts[e.v0],
                                 verts[e.v1], "candidate_edge", "", e.mult,
                                 label.str()});
    }
  }
}

void AddEdgeVertListAnnotations(TracePhase* phase,
                                const std::vector<std::vector<int>>& lists) {
  phase->annotations.reserve(phase->annotations.size() + lists.size());
  for (int i = 0; i < static_cast<int>(lists.size()); ++i) {
    std::ostringstream value;
    for (size_t j = 0; j < lists[i].size(); ++j) {
      if (j > 0) value << ",";
      value << lists[i][j];
    }
    phase->annotations.push_back({Id("e", i), "edgeVertList", value.str()});
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

void TraceRecorder::RecordBroadPhasePairs(
    const std::vector<vec2>& verts, const std::vector<EdgeM>& edges,
    const std::vector<std::pair<int, int>>& pairs) {
#ifdef MANIFOLD_DEBUG
  if (!trace_) return;
  TracePhase& phase = trace_->AddPhase("broad_phase_pairs");
  AddPoints(&phase, verts, "merged_vertex");
  AddInputEdges(&phase, verts, edges, "collapsed_edge");
  AddCandidatePairs(&phase, verts, edges, pairs);
#else
  (void)verts;
  (void)edges;
  (void)pairs;
#endif
}

void TraceRecorder::RecordEdgeVertLists(
    const std::vector<vec2>& verts, const std::vector<EdgeM>& edges,
    const std::vector<std::vector<int>>& lists) {
#ifdef MANIFOLD_DEBUG
  if (!trace_) return;
  TracePhase& phase = trace_->AddPhase("edge_vert_lists");
  AddPoints(&phase, verts, "list_vertex");
  AddSplitSubsegments(&phase, verts, edges, lists, "listed_subsegment");
  AddEdgeVertListAnnotations(&phase, lists);
#else
  (void)verts;
  (void)edges;
  (void)lists;
#endif
}

void TraceRecorder::RecordInsertedIntersections(
    const std::vector<vec2>& verts, const std::vector<EdgeM>& edges,
    const std::vector<std::vector<int>>& lists) {
#ifdef MANIFOLD_DEBUG
  if (!trace_) return;
  TracePhase& phase = trace_->AddPhase("inserted_intersections");
  AddPoints(&phase, verts, "arrangement_vertex");
  AddSplitSubsegments(&phase, verts, edges, lists, "arrangement_subsegment");
  AddEdgeVertListAnnotations(&phase, lists);
#else
  (void)verts;
  (void)edges;
  (void)lists;
#endif
}

void TraceRecorder::RecordNearbyIntersectionMerge(
    const std::vector<vec2>& verts, const std::vector<EdgeM>& edges,
    const std::vector<std::vector<int>>& lists) {
#ifdef MANIFOLD_DEBUG
  if (!trace_) return;
  TracePhase& phase = trace_->AddPhase("nearby_intersection_merge");
  AddPoints(&phase, verts, "nearby_intersection_vertex");
  AddSplitSubsegments(&phase, verts, edges, lists, "nearby_intersection");
  AddEdgeVertListAnnotations(&phase, lists);
#else
  (void)verts;
  (void)edges;
  (void)lists;
#endif
}

void TraceRecorder::RecordCanonicalSubedges(const std::vector<vec2>& verts,
                                            const CanonicalSubEdges& canon) {
#ifdef MANIFOLD_DEBUG
  if (!trace_) return;
  TracePhase& phase = trace_->AddPhase("canonical_subedges");
  AddPoints(&phase, verts, "canonical_vertex");
  AddCanonicalEdges(&phase, verts, canon);
#else
  (void)verts;
  (void)canon;
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
