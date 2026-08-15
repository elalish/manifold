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

#include "manifold/manifold.h"
#include "samples.h"

namespace {
using namespace manifold;

struct Gyroid {
  double operator()(vec3 p) const {
    p -= kPi / 4;
    return cos(p.x) * sin(p.y) + cos(p.y) * sin(p.z) + cos(p.z) * sin(p.x);
  }
};

Manifold RhombicDodecahedron(double size) {
  Manifold box = Manifold::Cube(size * la::sqrt(2.0) * vec3(1, 1, 2), true);
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
 *
 * @param size Creates a module scaled to this dimension between opposite faces.
 * @param n The number of divisions for SDF evaluation across the gyroid's
 * period.
 */
Manifold GyroidModule(double size, int n) {
  auto gyroid = [&](double level) {
    const double period = kTwoPi;
    return Manifold::LevelSet(Gyroid(), {vec3(-period), vec3(period)},
                              period / n, level)
        .Scale(vec3(size / period));
  };

  Manifold result = (RhombicDodecahedron(size) ^ gyroid(-0.4)) - gyroid(0.4);

  return result.Rotate(-45, 0, 90).Translate({0, 0, size / la::sqrt(2.0)});
}
}  // namespace manifold
