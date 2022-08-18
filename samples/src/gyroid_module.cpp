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
  Manifold result = box.Rotate(45) ^ box.Rotate(0, 45);
  return result ^ box.Rotate(90, 0, 45);
}

}  // namespace

namespace manifold {

/**
 * Creates a
 */
Manifold GyroidModule(float size, int n) {
  Gyroid func;
  const SDF<Gyroid> gyroidSDF(func);

  auto gyroid = [&](float level) {
    const float period = glm::two_pi<float>();
    return Manifold(gyroidSDF.LevelSet({glm::vec3(-period), glm::vec3(period)},
                                       period / n, level))
        .Scale(glm::vec3(size / period));
  };

  Manifold result = (RhombicDodecahedron(size) ^ gyroid(-0.5)) - gyroid(0.5);

  return result.Rotate(-45, 0, 90).Translate({0, 0, size / glm::sqrt(2.0f)});
}
}  // namespace manifold