// Copyright 2021 The Manifold Authors.
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

#include "cross_section.h"
#include "samples.h"

namespace {

using namespace manifold;

Manifold Base(float width, float radius, float decorRadius, float twistRadius,
              int nDecor, float innerRadius, float outerRadius, float cut,
              int nCut, int nDivision) {
  Manifold base = Manifold::Cylinder(width, radius + twistRadius / 2);

  CrossSection circle =
      CrossSection::Circle(decorRadius, nDivision).Translate({twistRadius, 0});
  Manifold decor = Manifold::Extrude(circle, width, nDivision, 180)
                       .Scale({1.0f, 0.5f, 1.0f})
                       .Translate({0.0f, radius, 0.0f});

  for (int i = 0; i < nDecor; ++i) {
    base += decor.Rotate(0, 0, (360.0f / nDecor) * i);
  }

  Polygons stretch(1);
  float dPhiRad = 2 * glm::pi<float>() / nCut;
  glm::vec2 p0(outerRadius, 0.0f);
  glm::vec2 p1(innerRadius, -cut);
  glm::vec2 p2(innerRadius, cut);
  for (int i = 0; i < nCut; ++i) {
    stretch[0].push_back(glm::rotate(p0, dPhiRad * i));
    stretch[0].push_back(glm::rotate(p1, dPhiRad * i));
    stretch[0].push_back(glm::rotate(p2, dPhiRad * i));
    stretch[0].push_back(glm::rotate(p0, dPhiRad * i));
  }

  base = Manifold::Extrude(stretch, width) ^ base;
  // Remove extra edges in coplanar faces
  base = base.AsOriginal();

  return base;
}
}  // namespace

namespace manifold {

/**
 * My Stretchy Bracelet: this is one of my most popular designs, largely because
 * it's quick and easy to 3D print. The defaults are picked to work well; change
 * the radius to fit your wrist. Changing the other values too much may break
 * the design.
 *
 * @param radius The overall size; the radius left for your wrist is roughly
 * radius - height.
 * @param height Thickness of the bracelet around your wrist.
 * @param width The length along your arm (the height of the print).
 * @param thickness The width of the material, which should be equal to your
 * printer's nozzle diameter.
 * @param nDecor The number of twisty shapes around the outside.
 * @param nCut The number of cuts that enable stretching.
 * @param nDivision the number of divisions along the width.
 */
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
