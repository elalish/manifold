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

#include <glm/gtx/rotate_vector.hpp>

#include "samples.h"

namespace {

int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }
}  // namespace

namespace manifold {

Manifold TorusKnot(int p, int q, float majorRadius, float minorRadius,
                   float threadRadius, int circularSegments,
                   int linearSegments) {
  int kLoops = gcd(p, q);
  p /= kLoops;
  q /= kLoops;
  int n = circularSegments > 2 ? circularSegments
                               : Manifold::GetCircularSegments(threadRadius);
  int m =
      linearSegments > 2 ? linearSegments : n * q * majorRadius / threadRadius;

  Polygons circle(1);
  float dPhi = 360.0f / n;
  for (int i = 0; i < n; ++i) {
    circle[0].push_back(
        {glm::vec2(cosd(dPhi * i) + 2.0f, sind(dPhi * i)), 0, Edge::kNoIdx});
  }

  Manifold knot = Manifold::Revolve(circle, m);

  knot.Warp([p, q, majorRadius, minorRadius, threadRadius](glm::vec3& v) {
    float psi = q * atan2(v.x, v.y);
    float theta = psi * p / q;
    glm::vec2 xy = glm::vec2(v);
    float x1 = sqrt(glm::dot(xy, xy));
    float phi = atan2(x1 - 2, v.z);
    v = glm::vec3(cos(phi), 0.0f, sin(phi));
    v *= threadRadius;
    float r = majorRadius + minorRadius * cos(theta);
    v = glm::rotateX(v, -float(atan2(p * minorRadius, q * r)));
    v.x += minorRadius;
    v = glm::rotateY(v, theta);
    v.x += majorRadius;
    v = glm::rotateZ(v, psi);
  });

  if (kLoops > 1) {
    std::vector<Manifold> knots;
    for (float k = 0; k < kLoops; ++k) {
      knots.emplace_back(knot);
      knots.back().Rotate(0, 0, 360.0f * (k / kLoops) * (q / float(p)));
    }
    knot = Manifold::Compose(knots);
  }

  return knot;
}
}  // namespace manifold