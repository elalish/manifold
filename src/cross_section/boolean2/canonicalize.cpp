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
// Sub-edge canonicalization. Each input edge is split into segments at
// its on-edge split vertices; segments with matching (vMin, vMax) are merged
// by summing signed multiplicities, and any whose summed mult is zero is
// dropped (Smith Table 7.3, PolySet2 form).
//
// Output `CanonicalSubEdges::edges` is sorted lex-ascending by (vMin,
// vMax), giving the downstream halfedge build and winding filter
// contiguous-memory scans.

#include "canonicalize.h"

namespace manifold {
namespace boolean2 {

void Canonicalize(const std::vector<EdgeM>& edges,
                  const std::vector<std::vector<int>>& lists,
                  CanonicalSubEdges* out) {
  out->edges.clear();
  // Pre-reserve. Each input edge contributes (1 + lists[e].size()) sub-edges.
  size_t total = edges.size();
  for (const auto& l : lists) total += l.size();
  out->edges.reserve(total);
  for (size_t e = 0; e < edges.size(); ++e) {
    int prev = edges[e].v0;
    for (int v : lists[e]) {
      out->Add(prev, v, edges[e].mult);
      prev = v;
    }
    out->Add(prev, edges[e].v1, edges[e].mult);
  }
  out->Finalize();
}

}  // namespace boolean2
}  // namespace manifold
