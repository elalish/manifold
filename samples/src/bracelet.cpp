// Copyright 2021 Emmett Lalish
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

Manifold Base(float width, float radius, float decorRadius, float twistRadius,
              int nDecor, float innerRadius, float outerRadius, float cut,
              int nCut, int nDivision) {
  Manifold base = Manifold::Cylinder(width, radius + twistRadius / 2);

  Polygons circle(1);
  float dPhiDeg = 180.0f / nDivision;
  for (int i = 0; i < 2 * nDivision; ++i) {
    circle[0].push_back(
        {glm::vec2(decorRadius * cosd(dPhiDeg * i) + twistRadius,
                   decorRadius * sind(dPhiDeg * i)),
         0});
  }
  Manifold decor = std::move(Manifold::Extrude(circle, width, nDivision, 180)
                                 .Scale({1.0f, 0.5f, 1.0f})
                                 .Translate({0.0f, radius, 0.0f}));

  for (int i = 0; i < nDecor; ++i) {
    base += decor.Rotate(0, 0, 360.0f / nDecor);
  }

  Polygons stretch(1);
  float dPhiRad = 2 * glm::pi<float>() / nCut;
  glm::vec2 p0(outerRadius, 0.0f);
  glm::vec2 p1(innerRadius, -cut);
  glm::vec2 p2(innerRadius, cut);
  for (int i = 0; i < nCut; ++i) {
    stretch[0].push_back({glm::rotate(p0, dPhiRad * i), 0});
    stretch[0].push_back({glm::rotate(p1, dPhiRad * i), 0});
    stretch[0].push_back({glm::rotate(p2, dPhiRad * i), 0});
    stretch[0].push_back({glm::rotate(p0, dPhiRad * i), 0});
  }

  base = Manifold::Extrude(stretch, width) ^ base;
  base.SetAsOriginal(true);

  return base;
}
}  // namespace

namespace manifold {

Manifold StretchyBracelet(float radius, float height, float width,
                          float thickness, int nDecor, int nCut,
                          int nDivision) {
  float twistRadius = glm::pi<float>() * radius / nDecor;
  float decorRadius = twistRadius * 1.5;
  float outerRadius = radius + (decorRadius + twistRadius) * 0.5;
  float innerRadius = outerRadius - height;
  float cut = 0.5 * (glm::pi<float>() * 2 * innerRadius / nCut - thickness);
  float adjThickness = 0.5 * thickness * height / cut;

  return Base(width, radius, decorRadius, twistRadius, nDecor,
              innerRadius + thickness, outerRadius + adjThickness,
              cut - adjThickness, nCut, nDivision) -
         Base(width, radius - thickness, decorRadius, twistRadius, nDecor,
              innerRadius, outerRadius + 3 * adjThickness, cut, nCut,
              nDivision);
}
}  // namespace manifold
