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

namespace {

using namespace manifold;

void Fractal(std::vector<Manifold>& holes, Manifold& hole, float w,
             glm::vec2 position, int depth, int maxDepth) {
  w /= 3;
  holes.push_back(
      hole.Scale({w, w, 1.0f}).Translate(glm::vec3(position, 0.0f)));
  if (depth == maxDepth) return;

  glm::vec2 offsets[8] = {{-w, -w}, {-w, 0.0f}, {-w, w}, {0.0f, w},
                          {w, w},   {w, 0.0f},  {w, -w}, {0.0f, -w}};
  for (int i = 0; i < 8; ++i) {
    Fractal(holes, hole, w, position + offsets[i], depth + 1, maxDepth);
  }
}
}  // namespace

namespace manifold {

/**
 * The classic cubic fractal.
 *
 * @param n Fractal depth. Warning: scales exponentially, n = 4 has almost
 * 400,000 triangles!
 */
Manifold MengerSponge(int n) {
  Manifold result = Manifold::Cube(glm::vec3(1.0f), true);

  std::vector<Manifold> holes;
  Fractal(holes, result, 1.0, {0.0f, 0.0f}, 1, n);

  Manifold hole = Manifold::Compose(holes);

  result -= hole;
  hole = hole.Rotate(90);
  result -= hole;
  hole = hole.Rotate(0, 0, 90);
  result -= hole;

  // Alternative order causes degenerate triangles
  // Manifold tmp1 = hole.Rotate(90) + hole.Rotate(90).Rotate(0, 0, 90);
  // Manifold tmp2 = hole + tmp1;
  // result = result - tmp2;

  return result;
}
}  // namespace manifold
