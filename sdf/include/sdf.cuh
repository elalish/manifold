// Copyright 2022 Emmett Lalish
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

#pragma once

#include "structs.h"
#include "utils.cuh"
#include "vec_dh.cuh"

namespace {
using namespace manifold;

template <typename Func>
struct Compute {
  const Func func;

  __host__ __device__ void operator()(
      thrust::tuple<float&, glm::vec3> inOut) const {
    float& value = thrust::get<0>(inOut);
    const glm::vec3 vert = thrust::get<1>(inOut);

    value = func(vert);
  }
};
}  // namespace

namespace manifold {
/** @addtogroup Core
 *  @{
 */
template <typename Func>
class SDF {
 public:
  SDF(Func sdf) : sdf_{sdf} {};

  inline __host__ __device__ float operator()(glm::vec3 point) const {
    return sdf_(point);
  }

  inline Mesh LevelSet(Box bounds, float edgeLength, float level = 0) const {
    Mesh out;
    int size = 10;
    VecDH<glm::vec3> verts(size);
    VecDH<float> values(size);
    thrust::fill(verts.beginD(), verts.endD(), glm::vec3(1.0f, 2.0f, 3.0f));
    thrust::for_each_n(zip(values.beginD(), verts.beginD()), size,
                       Compute<Func>({sdf_}));
    verts.Dump();
    values.Dump();
    return out;
  }

 private:
  const Func sdf_;
};

/** @} */
}  // namespace manifold