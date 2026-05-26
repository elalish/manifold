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

#pragma once

#include <algorithm>
#include <vector>

#include "../../parallel.h"
#include "predicates.h"

namespace manifold {
namespace boolean2 {

struct CanonEdge {
  int vMin, vMax;
  int mult;
};

struct CanonicalSubEdges {
  std::vector<CanonEdge> edges;

  inline void Add(int v0, int v1, int mult) {
    if (v0 == v1) return;
    int vMin = std::min(v0, v1);
    int vMax = std::max(v0, v1);
    int signedMult = (v0 < v1) ? mult : -mult;
    edges.push_back({vMin, vMax, signedMult});
  }

  inline void Finalize() {
    manifold::stable_sort(edges.begin(), edges.end(),
                          [](const CanonEdge& a, const CanonEdge& b) {
                            if (a.vMin != b.vMin) return a.vMin < b.vMin;
                            return a.vMax < b.vMax;
                          });
    size_t w = 0;
    for (size_t r = 0; r < edges.size();) {
      size_t k = r;
      int sumMult = 0;
      while (k < edges.size() && edges[k].vMin == edges[r].vMin &&
             edges[k].vMax == edges[r].vMax) {
        sumMult += edges[k].mult;
        ++k;
      }
      if (sumMult != 0) {
        edges[w] = {edges[r].vMin, edges[r].vMax, sumMult};
        ++w;
      }
      r = k;
    }
    edges.resize(w);
  }
};

// Split each directed input edge at the vertices in `lists[e]`, then merge
// matching undirected sub-edges by summing signed multiplicities.
//
// `edges` are the collapsed input edges. `lists[e]` is the sorted list of
// interior vertices that split edge `e`. The returned `edges` vector is sorted
// by `(vMin, vMax)` and omits sub-edges whose summed multiplicity is zero.
CanonicalSubEdges Canonicalize(const std::vector<EdgeM>& edges,
                               const std::vector<std::vector<int>>& lists);

}  // namespace boolean2
}  // namespace manifold
