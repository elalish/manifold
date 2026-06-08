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

#include <utility>
#include <vector>

#include "manifold/common.h"
#include "predicates.h"
#include "winding_filter.h"

namespace manifold {
namespace boolean2 {

double InferEps(const Polygons& a, const Polygons& b);

std::pair<std::vector<vec2>, std::vector<EdgeM>> PolygonsToInput(
    const Polygons& polys);
Polygons OutEdgesToPolygons(const std::vector<vec2>& verts,
                            const std::vector<OutEdge>& edges);

// Regularize one polygon set under the Positive (Add) winding rule at
// machine-scale eps. Fill-rule application, not tolerance decimation.
Polygons ApplyFillRule(const Polygons& polys, double eps);
Polygons Boolean2D(const Polygons& a, const Polygons& b, OpType op,
                   double eps = 0.0, double tolerance = 0.0);

// Polygon offset backing CrossSection::Offset.
enum class OffsetJoinType { Square, Round, Miter, Bevel };
Polygons Offset(const Polygons& in, double delta, OffsetJoinType jt,
                double miterLimit = 2.0, int circularSegments = 0);

// Group regularized simple loops into outer-ring components with their
// directly contained holes, backing CrossSection::Decompose.
std::vector<Polygons> DecomposeByContainment(const Polygons& polys);

}  // namespace boolean2
}  // namespace manifold
