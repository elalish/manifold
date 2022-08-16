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
}  // namespace

namespace manifold {

/**
 * Creates a
 */
Manifold GyroidModule(int n) {
  Gyroid func;
  const SDF<Gyroid> gyroidSDF(func);

  const float size = glm::two_pi<float>();
  Manifold gyroid(
      gyroidSDF.LevelSet({glm::vec3(-size), glm::vec3(size)}, size / n, 0.5));
  return gyroid;
}
}  // namespace manifold