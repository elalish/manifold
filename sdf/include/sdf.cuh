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

constexpr kOpen = std::numeric_limits<uint64_t>::max();

struct GridVert {
  uint64_t key = kOpen;
  float distance = 0.0f / 0.0f;
  int vertKey = -1;
  int edgeIndex = -1;
  int[7] edgeVerts;
};

class HashTable {
 public:
  HashTable(uint32_t sizeExp = 20, uint32_t step = 127)
      : size_{1 << sizeExp}, step_{step}, table_{1 << sizeExp} {}

  int Entries() const { return used_.H()[0]; }

  float FilledFraction() const {
    return static_cast<float>(used_.H()[0]) / size_;
  }

  __device__ __host__ bool Insert(const GridVert& vert) {
    uint32_t idx = vert.key & (size_ - 1);
    while (1) {
      const uint64_t found =
          AtomicCAS(&table_.ptrD()[idx].key, kOpen, vert.key);
      if (found == kOpen) {
        if (AtomicAdd(&used_.ptrD()[0], 1) * 2 > size_) {
          return true;
        }
        table_.ptrD()[idx] = vert;
        return false;
      }
      if (found == vert.key) return false;
      idx = (idx + step_) & (size_ - 1);
    }
  }

  __device__ __host__ GridVert operator[](uint64_t key) const {
    uint32_t idx = key & (size_ - 1);
    while (1) {
      const GridVert found = table_.ptrD()[idx];
      if (found.key == key) return found;
      if (found.key == kOpen) return GridVert();
      idx = (idx + step_) & (size_ - 1);
    }
  }

 private:
  const uint32_t size_;
  const uint32_t step_;
  VecDH<GridVert> table_;
  VecDH<uint32_t> used_(1, 0);
};

template <typename Func>
struct ComputeVerts {
  glm::vec3* vertPos;
  int* vertIndex;
  HashTable gridVerts;
  const Func func;
  const glm::vec3 origin;
  const float spacing;

  __host__ __device__ void operator()(uint64_t mortonCode) const {
    GridVert gridVert;
    const glm::ivec3 gridIndex = DecodeMorton(mortonCode);
    const glm::vec3 position = origin + spacing * gridIndex;
    value = func(position);
  }
};

struct BuildTris {
  glm::ivec3* triVerts;
  uint64_t* index;
  const HashTable gridVerts;

  __host__ __device__ void operator()(uint64_t mortonCode) const {}
};
}  // namespace

namespace manifold {
/** @addtogroup Core
 *  @{
 */
template <typename Func>
class SDF {
 public:
  SDF(Func sdf) : sdf_{sdf} {}

  inline __host__ __device__ float operator()(glm::vec3 point) const {
    return sdf_(point);
  }

  inline Mesh LevelSet(Box bounds, float edgeLength, float level = 0) const {
    Mesh out;
    HashTable gridVerts(10);
    glm::vec3 dim = bounds.Size();
    // Need to create a new MortonCode function that spreads when needed instead
    // of always by 3, to make non-cubic domains efficient.
    float maxDim = std::max(dim[0], std::max(dim[1], dim[2]));
    int n = maxDim / edgeLength;
    float spacing = maxDim / n;
    int maxMorton = MortonCode(glm::ivec3(n));

    VecDH<glm::vec3> vertPos(gridVerts.Size() * 7);
    VecDH<int> index(1, 0);

    thrust::for_each_n(
        countAt(0), maxMorton,
        ComputeVerts<Func>({vertPos.ptrD(), index.ptrD(), gridVerts, sdf_,
                            bounds.Min(), spacing}));

    vertPos.Resize(index.H()[0]);
    VecDH<uint64_t> triVerts(gridVerts.Entries() * 6);
    index.H()[0] = 0;

    thrust::for_each_n(countAt(0), maxMorton,
                       BuildTris({triVerts.ptrD(), index.ptrD(), gridVerts}));
    triVerts.Resize(index.H()[0]);

    out.vertPos = vertPos.H();
    out.triVerts = triVerts.H();
    return out;
  }

 private:
  const Func sdf_;
};

/** @} */
}  // namespace manifold