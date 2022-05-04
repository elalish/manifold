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

constexpr uint64_t kOpen = std::numeric_limits<uint64_t>::max();

constexpr glm::ivec3 tetTri0[16] = {{-1, -1, -1},  //
                                    {0, 3, 4},     //
                                    {0, 1, 5},     //
                                    {1, 5, 3},     //
                                    {1, 4, 2},     //
                                    {1, 0, 3},     //
                                    {2, 5, 0},     //
                                    {5, 3, 2},     //
                                    {2, 3, 5},     //
                                    {0, 5, 2},     //
                                    {3, 0, 1},     //
                                    {2, 4, 1},     //
                                    {3, 5, 1},     //
                                    {5, 1, 0},     //
                                    {4, 3, 0},     //
                                    {-1, -1, -1}};

constexpr glm::ivec3 tetTri1[16] = {{-1, -1, -1},  //
                                    {-1, -1, -1},  //
                                    {-1, -1, -1},  //
                                    {3, 4, 1},     //
                                    {-1, -1, -1},  //
                                    {3, 2, 1},     //
                                    {0, 4, 2},     //
                                    {-1, -1, -1},  //
                                    {-1, -1, -1},  //
                                    {2, 4, 0},     //
                                    {1, 2, 3},     //
                                    {-1, -1, -1},  //
                                    {1, 4, 3},     //
                                    {-1, -1, -1},  //
                                    {-1, -1, -1},  //
                                    {-1, -1, -1}};

constexpr glm::vec3 edgeVec[7] = {{0.5f, 0.5f, 0.5f},   //
                                  {1.0f, 0.0f, 0.0f},   //
                                  {0.0f, 1.0f, 0.0f},   //
                                  {0.0f, 0.0f, 1.0f},   //
                                  {-0.5f, 0.5f, 0.5f},  //
                                  {0.5f, -0.5f, 0.5f},  //
                                  {0.5f, 0.5f, -0.5f}};

struct GridVert {
  uint64_t key = kOpen;
  float distance = NAN;
  int edgeIndex = -1;
  int edgeVerts[7] = {-1, -1, -1, -1, -1, -1, -1};

  int Inside() const { return edgeIndex == -1 ? (distance >= 0 ? 1 : -1) : 0; }
};

class HashTableD {
 public:
  HashTableD(VecDH<GridVert>& alloc, VecDH<uint32_t>& used, uint32_t step = 127)
      : step_{step}, table_{alloc}, used_{used} {}

  __device__ __host__ int Size() const { return table_.size(); }

  __device__ __host__ bool Insert(const GridVert& vert) {
    uint32_t idx = vert.key & (Size() - 1);
    while (1) {
      const uint64_t found = AtomicCAS(&table_[idx].key, kOpen, vert.key);
      if (found == kOpen) {
        if (AtomicAdd(&used_[0], 1) * 2 > Size()) {
          return true;
        }
        table_[idx] = vert;
        return false;
      }
      if (found == vert.key) return false;
      idx = (idx + step_) & (Size() - 1);
    }
  }

  __device__ __host__ GridVert operator[](uint64_t key) const {
    uint32_t idx = key & (Size() - 1);
    while (1) {
      const GridVert found = table_[idx];
      if (found.key == key) return found;
      if (found.key == kOpen) return GridVert();
      idx = (idx + step_) & (Size() - 1);
    }
  }

  __device__ __host__ GridVert At(int idx) const { return table_[idx]; }

 private:
  const uint32_t step_;
  VecD<GridVert> table_;
  VecD<uint32_t> used_;
};

class HashTable {
 public:
  HashTable(uint32_t sizeExp = 20, uint32_t step = 127)
      : alloc_{1 << sizeExp}, table_{alloc_, used_, step} {}

  HashTableD D() { return table_; }

  int Entries() const { return used_.H()[0]; }

  int Size() const { return table_.Size(); }

  float FilledFraction() const {
    return static_cast<float>(used_.H()[0]) / table_.Size();
  }

 private:
  VecDH<GridVert> alloc_;
  VecDH<uint32_t> used_ = VecDH<uint32_t>(1, 0);
  HashTableD table_;
};

template <typename Func>
struct ComputeVerts {
  glm::vec3* vertPos;
  int* vertIndex;
  HashTable gridVerts;
  const Func sdf;
  const glm::ivec3 gridSize;
  const glm::vec3 origin;
  const glm::vec3 spacing;

  inline __host__ __device__ bool AtBounds(glm::ivec3 gridIndex) const {
    return gridIndex.x == 0 || gridIndex.x == gridSize.x || gridIndex.y == 0 ||
           gridIndex.y == gridSize.y || gridIndex.z == 0 ||
           gridIndex.z == gridSize.z;
  }

  inline __host__ __device__ float Sdf(glm::vec3 base, int i) const {
    sdf(base + (i < 7 ? 1 : -1) * spacing * edgeVec[j])
  }

  // inline __host__ __device__ float BoundedSdf(glm::vec3 base, int i) const {}

  inline __host__ __device__ void operator()(uint64_t mortonCode) {
    GridVert gridVert;
    gridVert.key = mortonCode;
    const glm::ivec3 gridIndex = DecodeMorton(mortonCode);

    // const auto sdfFunc =
    //     AtBounds(gridIndex) ? &ComputeVerts::BoundedSdf : &ComputeVerts::Sdf;

    const glm::vec3 position = origin + spacing * gridIndex;
    gridVert.distance = sdf(position);

    bool keep = false;
    float minDist2 = 0.25 * 0.25;
    for (int i = 0; i < 14; ++i) {
      const int j = i < 7 ? i : i - 7;
      const float val = Sdf(position, i);
      if (val * gridVert.distance > 0 || (val == 0 && gridVert.distance == 0))
        continue;
      keep = true;

      // Record the nearest intersection of all 14 edges, only if it is close
      // enough to allow this gridVert to safely move to it without inverting
      // any tetrahedra.
      const glm::vec3 delta =
          edgeVec[j] * (1 - val / (val - gridVert.distance));
      const float dist2 = glm::dot(delta, delta);
      if (dist2 < minDist2) {
        gridVert.edgeIndex = i;
        minDist2 = dist2;
      }

      // These seven edges are uniquely owned by this gridVert; any of them
      // which intersect the surface create a vert.
      if (i < 7) {
        const int idx = AtomicAdd(vertIndex, 1);
        vertPos[idx] = position + spacing * delta;
        gridVert.edgeVerts[i] = idx;
      }
    }
    if (keep) gridVerts.Insert(gridVert);
  }
};

struct BuildTris {
  glm::ivec3* triVerts;
  int* triIndex;
  const HashTableD gridVerts;

  __host__ __device__ void CreateTris(const glm::ivec4& tet,
                                      const int edges[6]) {
    const int i = (tet[0] > 0 ? 1 : 0) + (tet[1] > 0 ? 2 : 0) +
                  (tet[2] > 0 ? 4 : 0) + (tet[3] > 0 ? 8 : 0);
    glm::ivec3 tri = tetTri0[i];
    if (tri[0] < 0) return;
    int idx = AtomicAdd(triIndex, 1);
    triVerts[idx] = {edges[tri[0]], edges[tri[1]], edges[tri[2]]};

    tri = tetTri1[i];
    if (tri[0] < 0) return;
    idx = AtomicAdd(triIndex, 1);
    triVerts[idx] = {edges[tri[0]], edges[tri[1]], edges[tri[2]]};
  }

  __host__ __device__ void operator()(int idx) {
    const GridVert& base = gridVerts.At(idx);
    if (base.key == kOpen) return;

    const glm::ivec4 baseIndex = DecodeMorton(base.key);

    glm::ivec4 leadIndex = baseIndex;
    if (leadIndex.w == 0)
      leadIndex.w = 1;
    else {
      leadIndex += 1;
      leadIndex.w = 0;
    }
    const GridVert& leadVert = gridVerts[MortonCode(leadIndex)];

    // This GridVert is in charge of the 6 tetrahedra surrounding its edge in
    // the (1,1,1) direction, attached to leadVert.
    glm::ivec4 tet(leadVert.Inside(), base.Inside(), -2, -2);
    glm::ivec4 thisIndex = baseIndex;
    thisIndex[0] += 1;
    GridVert& thisVert = gridVerts[MortonCode(thisIndex)];
    tet[2] = thisVert.Inside();
    int edges[6] = {base.edgeVerts[0], -1, -1, -1, -1, -1};
    for (const int i : {0, 1, 2}) {
      edges[1] = base.edgeVerts[i + 1];
      edges[4] = thisVert.edgeVerts[i + 4];
      edges[5] = base.edgeVerts[prev3[i] + 4];

      thisIndex = leadIndex;
      thisIndex[prev3[i]] -= 1;
      thisVert = gridVerts[MortonCode(thisIndex)];
      tet[3] = thisVert.Inside();
      edges[2] = thisVert.edgeVerts[next3[i] + 4];
      edges[3] = thisVert.edgeVerts[prev3[i] + 1];
      CreateTris(tet, edges);

      edges[1] = edges[5];
      edges[2] = thisVert.edgeVerts[i + 4];
      edges[4] = edges[3];
      edges[5] = base.edgeVerts[next3[i] + 1];

      tet[2] = tet[3];
      glm::ivec4 thisIndex = baseIndex;
      thisIndex[next3[i]] += 1;
      thisVert = gridVerts[MortonCode(thisIndex)];
      tet[3] = thisVert.Inside();
      edges[3] = thisVert.edgeVerts[next3[i] + 4];
      CreateTris(tet, edges);

      tet[2] = tet[3];
    }
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
  SDF(Func sdf) : sdf_{sdf} {}

  inline __host__ __device__ float operator()(glm::vec3 point) const {
    return sdf_(point);
  }

  inline Mesh LevelSet(Box bounds, float edgeLength, float level = 0) const {
    Mesh out;

    glm::vec3 dim = bounds.Size();
    // Need to create a new MortonCode function that spreads when needed instead
    // of always by 3, to make non-cubic domains efficient.
    float maxDim = std::max(dim[0], std::max(dim[1], dim[2]));
    glm::ivec3 gridSize(dim / edgeLength);
    glm::vec3 spacing = dim / glm::vec3(gridSize);
    int maxMorton = MortonCode(gridSize);

    HashTable gridVerts(10);  // maxMorton^(2/3)? Some heuristic with ability to
                              // enlarge if it gets too full.

    VecDH<glm::vec3> vertPos(gridVerts.Size() * 7);
    VecDH<int> index(1, 0);

    thrust::for_each_n(
        countAt(0), maxMorton,
        ComputeVerts<Func>({vertPos.ptrD(), index.ptrD(), gridVerts.D(), sdf_,
                            gridSize, bounds.min, spacing}));
    vertPos.resize(index.H()[0]);

    VecDH<glm::ivec3> triVerts(gridVerts.Entries() * 12);  // worst case

    index.H()[0] = 0;
    thrust::for_each_n(
        countAt(0), gridVerts.Size(),
        BuildTris({triVerts.ptrD(), index.ptrD(), gridVerts.D()}));
    triVerts.resize(index.H()[0]);

    out.vertPos.insert(out.vertPos.end(), vertPos.begin(), vertPos.end());
    out.triVerts.insert(out.triVerts.end(), triVerts.begin(), triVerts.end());
    return out;
  }

 private:
  const Func sdf_;
};

/** @} */
}  // namespace manifold