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

int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }
}  // namespace

namespace manifold {

/**
 * Creates a classic torus knot, defined as a string wrapping periodically
 * around the surface of an imaginary donut. If p and q have a common factor
 * then you will get multiple separate, interwoven knots. This is an example of
 * using the Manifold.Warp() method, thus avoiding any handling of triangles.
 *
 * @param p The number of times the thread passes through the donut hole.
 * @param q The number of times the thread circles the donut.
 * @param majorRadius Radius of the interior of the imaginary donut.
 * @param minorRadius Radius of the small cross-section of the imaginary donut.
 * @param threadRadius Radius of the small cross-section of the actual object.
 * @param circularSegments Number of linear segments making up the threadRadius
 * circle. Default is Quality.GetCircularSegments().
 * @param linearSegments Number of segments along the length of the knot.
 * Default makes roughly square facets.
 */
Manifold TorusKnot(int p, int q, float majorRadius, float minorRadius,
                   float threadRadius, int circularSegments,
                   int linearSegments) {
  int kLoops = gcd(p, q);
  p /= kLoops;
  q /= kLoops;
  int n = circularSegments > 2 ? circularSegments
                               : Quality::GetCircularSegments(threadRadius);
  int m =
      linearSegments > 2 ? linearSegments : n * q * majorRadius / threadRadius;

  CrossSection circle = CrossSection::Circle(1., n).Translate({2, 0});
  Manifold knot = Manifold::Revolve(circle, m);

  knot =
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
      knots.push_back(
          knot.Rotate(0, 0, 360.0f * (k / kLoops) * (q / float(p))));
    }
    knot = Manifold::Compose(knots);
  }

  return knot;
}
}  // namespace manifold
