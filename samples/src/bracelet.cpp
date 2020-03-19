// Copyright 2019 Emmett Lalish
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "samples.h"

namespace {

using namespace manifold;

Manifold Base(float radius, float width, float bigRadius, float rr, int n,
              float ri, float a, float ro, int m) {
  Manifold base = Manifold::Cylinder(width, bigRadius + rr / 2);

  Polygons circle(1);
  int k = 20;
  float dPhiDeg = 360.0f / k;
  for (int i = 0; i < k; ++i) {
    circle[0].push_back(
        {glm::vec2(radius * cosd(dPhiDeg * i) + rr, radius * sind(dPhiDeg * i)),
         0, Edge::kNoIdx});
  }
  Manifold decor = std::move(Manifold::Extrude(circle, width, 10, 180)
                                 .Scale({1.0f, 0.5f, 1.0f})
                                 .Translate({0.0f, bigRadius, 0.0f}));

  for (int i = 0; i < n; ++i) {
    base += decor.Rotate(0, 0, 360.0f / n);
  }

  Polygons stretch(1);
  float dPhiRad = 2 * glm::pi<float>() / m;
  glm::vec2 p0(ro, 0.0f);
  glm::vec2 p1(ri, -a);
  glm::vec2 p2(ri, a);
  for (int i = 0; i < m; ++i) {
    stretch[0].push_back({glm::rotate(p0, dPhiRad * i), 0, Edge::kNoIdx});
    stretch[0].push_back({glm::rotate(p1, dPhiRad * i), 0, Edge::kNoIdx});
    stretch[0].push_back({glm::rotate(p2, dPhiRad * i), 0, Edge::kNoIdx});
    stretch[0].push_back({glm::rotate(p0, dPhiRad * i), 0, Edge::kNoIdx});
  }

  return Manifold::Extrude(stretch, width) ^ base;
}
}  // namespace

namespace manifold {

Manifold StretchyBracelet(float radius, float height, float width,
                          float thickness, int n, int m) {
  float rr = glm::pi<float>() * radius / n;
  float r1 = rr * 1.5;
  float ro = radius + (r1 + rr) * 0.5;
  float ri = ro - height;
  float a = 0.5 * (glm::pi<float>() * 2 * ri / m - thickness);
  float rot = 0.5 * thickness * height / a;

  return Base(r1, width, radius, rr, n, ri + thickness, a - rot, ro + rot, m) -
         Base(r1, width, radius - thickness, rr, n, ri, a, ro + 3 * rot, m);
}
}  // namespace manifold
