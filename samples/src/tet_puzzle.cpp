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
 * A tetrahedron cut into two identical halves that can screw together as a
 * puzzle. This only outputs one of the halves. This demonstrates how redundant
 * points along a polygon can be used to make twisted extrusions smoother.
 *
 * @param edgeLength Length of each edge of the overall tetrahedron.
 * @param gap Spacing between the two halves to allow sliding.
 * @param nDivisions Number of divisions (both ways) in the screw surface.
 */
Manifold TetPuzzle(float edgeLength, float gap, int nDivisions) {
  const glm::vec3 scale(edgeLength / (2 * sqrt(2)));

  Manifold tet = Manifold::Tetrahedron().Scale(scale);

  Polygons box;
  box.push_back({{2, -2}, {2, 2}});

  for (int i = 0; i <= nDivisions; ++i) {
    box[0].push_back({gap / 2, 2 - i * 4.0f / nDivisions});
  }

  Manifold screw = Manifold::Extrude(box, 2, nDivisions, 270)
                       .Rotate(0, 0, -45)
                       .Translate({0, 0, -1})
                       .Scale(scale);

  return tet ^ screw;
}
}  // namespace manifold