// Copyright 2022 The Manifold Authors.
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
#include "sdf.h"

namespace {

struct Gyroid {
  __host__ __device__ float operator()(glm::vec3 p) const {
    p -= glm::pi<float>() / 4;
    return cos(p.x) * sin(p.y) + cos(p.y) * sin(p.z) + cos(p.z) * sin(p.x);
  }
};

Manifold RhombicDodecahedron(float size) {
  Manifold box =
      Manifold::Cube(size * glm::sqrt(2.0f) * glm::vec3(1, 1, 2), true);
  Manifold result = box.Rotate(90, 45) ^ box.Rotate(90, 45, 90);
  return result ^ box.Rotate(0, 0, 45);
}

}  // namespace

namespace manifold {

/**
 * Creates a rhombic dodecahedral module of a gyroid manifold, which can be
 * assembled together to tile space continuously. This one is designed to be
 * 3D-printable, as it is oriented with minimal overhangs. This sample
 * demonstrates the use of a Signed Distance Function (SDF) to create smooth,
 * complex manifolds.
 */
Manifold GyroidModule(float size, int n) {
  auto gyroid = [&](float level) {
    const float period = glm::two_pi<float>();
    return Manifold(LevelSet(Gyroid(), {glm::vec3(-period), glm::vec3(period)},
                             period / n, level))
        .Scale(glm::vec3(size / period));
  };

  Manifold result = (RhombicDodecahedron(size) ^ gyroid(-0.4)) - gyroid(0.4);

  return result.Rotate(-45, 0, 90).Translate({0, 0, size / glm::sqrt(2.0f)});
}
}  // namespace manifold