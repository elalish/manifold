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

#include "samples.h"

namespace manifold {

/**
 * A cubic frame with cylinders for edges and spheres at the corners.
 * Demonstrates how at 90-degree intersections, the sphere and cylinder facets
 * match up perfectly.
 *
 * @param edgeLength Distance between the corners.
 * @param radius Radius of the frame members.
 * @param circularSegments Number of segments in the cylinders and spheres.
 * Defaults to Quality.GetCircularSegments().
 */
Manifold RoundedFrame(float edgeLength, float radius, int circularSegments) {
  Manifold edge = Manifold::Cylinder(edgeLength, radius, -1, circularSegments);
  Manifold corner = Manifold::Sphere(radius, circularSegments);

  Manifold edge1 = corner + edge;
  edge1 = edge1.Rotate(-90).Translate({-edgeLength / 2, -edgeLength / 2, 0});

  Manifold edge2 = edge1.Rotate(0, 0, 180);
  edge2 += edge1;
  edge2 += edge.Translate({-edgeLength / 2, -edgeLength / 2, 0});

  Manifold edge4 = edge2.Rotate(0, 0, 90);
  edge4 += edge2;

  Manifold frame = edge4.Translate({0, 0, -edgeLength / 2});
  frame += frame.Rotate(180);

  return frame;
}
}  // namespace manifold
