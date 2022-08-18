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

#pragma once

#include "structs.h"
#include "utils.h"
#include "vec_dh.h"

namespace {
using namespace manifold;

typedef unsigned long long int Uint64;

constexpr Uint64 kOpen = std::numeric_limits<Uint64>::max();

__host__ __device__ Uint64 AtomicCAS(Uint64& target, Uint64 compare,
                                     Uint64 val) {
#ifdef __CUDA_ARCH__
  return atomicCAS(&target, compare, val);
#else
  std::atomic<Uint64>& tar = reinterpret_cast<std::atomic<Uint64>&>(target);
  Uint64 old_val = tar.load();
  tar.compare_exchange_weak(compare, val);
  return old_val;
#endif
}

__host__ __device__ int Next3(int i) {
  constexpr glm::ivec3 next3(1, 2, 0);
  return next3[i];
}

__host__ __device__ int Prev3(int i) {
  constexpr glm::ivec3 prev3(2, 0, 1);
  return prev3[i];
}

__host__ __device__ glm::ivec3 TetTri0(int i) {
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
  return tetTri0[i];
}

__host__ __device__ glm::ivec3 TetTri1(int i) {
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
  return tetTri1[i];
}

__host__ __device__ glm::ivec4 Neighbors(int i) {
  constexpr glm::ivec4 neighbors[14] = {{0, 0, 0, 1},     //
                                        {1, 0, 0, 0},     //
                                        {0, 1, 0, 0},     //
                                        {0, 0, 1, 0},     //
                                        {-1, 0, 0, 1},    //
                                        {0, -1, 0, 1},    //
                                        {0, 0, -1, 1},    //
                                        {-1, -1, -1, 1},  //
                                        {-1, 0, 0, 0},    //
                                        {0, -1, 0, 0},    //
                                        {0, 0, -1, 0},    //
                                        {0, -1, -1, 1},   //
                                        {-1, 0, -1, 1},   //
                                        {-1, -1, 0, 1}};
  return neighbors[i];
}

__host__ __device__ Uint64 SpreadBits3(Uint64 v) {
  v = v & 0x1fffff;
  v = (v | v << 32) & 0x1f00000000ffff;
  v = (v | v << 16) & 0x1f0000ff0000ff;
  v = (v | v << 8) & 0x100f00f00f00f00f;
  v = (v | v << 4) & 0x10c30c30c30c30c3;
  v = (v | v << 2) & 0x1249249249249249;
  return v;
}

__host__ __device__ Uint64 SqeezeBits3(Uint64 v) {
  v = v & 0x1249249249249249;
  v = (v ^ v >> 2) & 0x10c30c30c30c30c3;
  v = (v ^ v >> 4) & 0x100f00f00f00f00f;
  v = (v ^ v >> 8) & 0x1f0000ff0000ff;
  v = (v ^ v >> 16) & 0x1f00000000ffff;
  v = (v ^ v >> 32) & 0x1fffff;
  return v;
}

// This is a modified 3D MortonCode, where the xyz code is shifted by one bit
// and the w bit is added as the least significant. This allows 21 bits per x,
// y, and z channel and 1 for w, filling the 64 bit total.
__host__ __device__ Uint64 MortonCode(const glm::ivec4& index) {
  return static_cast<Uint64>(index.w) | (SpreadBits3(index.x) << 1) |
         (SpreadBits3(index.y) << 2) | (SpreadBits3(index.z) << 3);
}

__host__ __device__ glm::ivec4 DecodeMorton(Uint64 code) {
  glm::ivec4 index;
  index.x = SqeezeBits3(code >> 1);
  index.y = SqeezeBits3(code >> 2);
  index.z = SqeezeBits3(code >> 3);
  index.w = code & 0x1u;
  return index;
}

struct GridVert {
  Uint64 key = kOpen;
  float distance = NAN;
  int edgeIndex = -1;
  int edgeVerts[7] = {-1, -1, -1, -1, -1, -1, -1};

  __host__ __device__ int Inside() const { return distance > 0 ? 1 : -1; }
};

class HashTableD {
 public:
  HashTableD(VecDH<GridVert>& alloc, VecDH<uint32_t>& used, uint32_t step = 127)
      : step_{step}, table_{alloc}, used_{used} {}

  __host__ __device__ int Size() const { return table_.size(); }

  __host__ __device__ bool Insert(const GridVert& vert) {
    uint32_t idx = vert.key & (Size() - 1);
    while (1) {
      Uint64& key = table_[idx].key;
      const Uint64 found = AtomicCAS(key, kOpen, vert.key);
      if (found == kOpen) {
        if (AtomicAdd(used_[0], 0x1u) * 2 > Size()) {
          return true;
        }
        table_[idx] = vert;
        return false;
      }
      if (found == vert.key) return false;
      idx = (idx + step_) & (Size() - 1);
    }
  }

  __host__ __device__ GridVert operator[](Uint64 key) const {
    uint32_t idx = key & (Size() - 1);
    while (1) {
      const GridVert found = table_[idx];
      if (found.key == key || found.key == kOpen) return found;
      idx = (idx + step_) & (Size() - 1);
    }
  }

  __host__ __device__ GridVert At(int idx) const { return table_[idx]; }

 private:
  const uint32_t step_;
  VecD<GridVert> table_;
  VecD<uint32_t> used_;
};

class HashTable {
 public:
  HashTable(uint32_t size, uint32_t step = 127)
      : alloc_{1 << (int)ceil(log2(size)), {}}, table_{alloc_, used_, step} {}

  HashTableD D() { return table_; }

  int Entries() const { return used_[0]; }

  int Size() const { return table_.Size(); }

  float FilledFraction() const {
    return static_cast<float>(used_[0]) / table_.Size();
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
  HashTableD gridVerts;
  const Func sdf;
  const glm::vec3 origin;
  const glm::ivec3 gridSize;
  const glm::vec3 spacing;
  const float level;

  inline __host__ __device__ glm::vec3 Position(glm::ivec4 gridIndex) const {
    return origin + spacing * (-0.5f + glm::vec3(gridIndex) +
                               (gridIndex.w == 1 ? 0.5f : 0.0f) * glm::vec3(1));
  }

  inline __host__ __device__ float BoundedSDF(glm::ivec4 gridIndex) const {
    const float d = sdf(Position(gridIndex)) - level;

    const glm::ivec3 xyz(gridIndex);
    if (glm::any(glm::equal(xyz, glm::ivec3(0))) ||
        glm::any(glm::greaterThanEqual(xyz, gridSize)) ||
        (gridIndex.w == 1 &&
         glm::any(glm::greaterThanEqual(xyz, gridSize - 1))))
      return glm::min(d, 0.0f);

    return d;
  }

  inline __host__ __device__ void operator()(Uint64 mortonCode) {
    const glm::ivec4 gridIndex = DecodeMorton(mortonCode);

    if (glm::any(glm::greaterThan(glm::ivec3(gridIndex), gridSize))) return;

    const glm::vec3 position = Position(gridIndex);

    GridVert gridVert;
    gridVert.key = mortonCode;
    gridVert.distance = BoundedSDF(gridIndex);

    bool keep = false;
    float minDist2 = 0.25 * 0.25;
    for (int i = 0; i < 14; ++i) {
      glm::ivec4 neighborIndex = gridIndex + Neighbors(i);
      if (neighborIndex.w == 2) {
        neighborIndex += 1;
        neighborIndex.w = 0;
      }
      const glm::vec3 iPos = Position(neighborIndex);
      const float val = BoundedSDF(neighborIndex);
      if (val * gridVert.distance > 0 || (val == 0 && gridVert.distance == 0))
        continue;
      keep = true;

      // Record the nearest intersection of all 14 edges, only if it is close
      // enough to allow this gridVert to safely move to it without inverting
      // any tetrahedra.
      const glm::vec3 interp = (val * position - gridVert.distance * iPos) /
                               (val - gridVert.distance);
      const glm::vec3 delta = interp - position;
      const float dist2 = glm::dot(delta, delta);
      if (dist2 < minDist2) {
        // gridVert.edgeIndex = i;
        minDist2 = dist2;
      }

      // These seven edges are uniquely owned by this gridVert; any of them
      // which intersect the surface create a vert.
      if (i < 7) {
        const int idx = AtomicAdd(*vertIndex, 1);
        vertPos[idx] = interp;
        gridVert.edgeVerts[i] = idx;
      }
    }

    if (keep && gridVerts.Insert(gridVert)) printf("out of space!\n");
  }
};

struct BuildTris {
  glm::ivec3* triVerts;
  int* triIndex;
  const HashTableD gridVerts;

  __host__ __device__ void CreateTri(const glm::ivec3& tri,
                                     const int edges[6]) {
    if (tri[0] < 0) return;
    int idx = AtomicAdd(*triIndex, 1);
    triVerts[idx] = {edges[tri[0]], edges[tri[1]], edges[tri[2]]};
  }

  __host__ __device__ void CreateTris(const glm::ivec4& tet,
                                      const int edges[6]) {
    const int i = (tet[0] > 0 ? 1 : 0) + (tet[1] > 0 ? 2 : 0) +
                  (tet[2] > 0 ? 4 : 0) + (tet[3] > 0 ? 8 : 0);
    CreateTri(TetTri0(i), edges);
    CreateTri(TetTri1(i), edges);
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
    if (leadVert.key == kOpen) return;

    // This GridVert is in charge of the 6 tetrahedra surrounding its edge in
    // the (1,1,1) direction, attached to leadVert.
    glm::ivec4 tet(leadVert.Inside(), base.Inside(), -2, -2);
    glm::ivec4 thisIndex = baseIndex;
    thisIndex.x += 1;

    GridVert thisVert = gridVerts[MortonCode(thisIndex)];
    bool skipTet = thisVert.key == kOpen;

    tet[2] = thisVert.Inside();
    int edges[6] = {base.edgeVerts[0], -1, -1, -1, -1, -1};
    for (const int i : {0, 1, 2}) {
      edges[1] = base.edgeVerts[i + 1];
      edges[4] = thisVert.edgeVerts[i + 4];
      edges[5] = base.edgeVerts[Prev3(i) + 4];

      thisIndex = leadIndex;
      thisIndex[Prev3(i)] -= 1;
      thisVert = gridVerts[MortonCode(thisIndex)];

      tet[3] = thisVert.Inside();
      edges[2] = thisVert.edgeVerts[Next3(i) + 4];
      edges[3] = thisVert.edgeVerts[Prev3(i) + 1];

      if (!skipTet && thisVert.key != kOpen) CreateTris(tet, edges);
      skipTet = thisVert.key == kOpen;

      tet[2] = tet[3];
      edges[1] = edges[5];
      edges[2] = thisVert.edgeVerts[i + 4];
      edges[4] = edges[3];
      edges[5] = base.edgeVerts[Next3(i) + 1];

      thisIndex = baseIndex;
      thisIndex[Next3(i)] += 1;
      thisVert = gridVerts[MortonCode(thisIndex)];

      tet[3] = thisVert.Inside();
      edges[3] = thisVert.edgeVerts[Next3(i) + 4];

      if (!skipTet && thisVert.key != kOpen) CreateTris(tet, edges);
      skipTet = thisVert.key == kOpen;

      tet[2] = tet[3];
    }
  }
};
}  // namespace

namespace manifold {

void RemoveUnreferencedVerts(VecDH<glm::vec3>& vertPos,
                             VecDH<glm::ivec3>& triVerts);

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

    const glm::vec3 dim = bounds.Size();
    const float maxDim = std::max(dim[0], std::max(dim[1], dim[2]));
    const glm::ivec3 gridSize(dim / edgeLength);
    const glm::vec3 spacing = dim / (glm::vec3(gridSize));

    const int maxMorton = MortonCode(glm::ivec4(gridSize + 1, 1));

    const int tableSize = glm::min(
        2 * maxMorton, static_cast<int>(100 * glm::pow(maxMorton, 0.667)));
    HashTable gridVerts(tableSize);

    VecDH<glm::vec3> vertPos(gridVerts.Size() * 7);
    VecDH<int> index(1, 0);

    thrust::for_each_n(
        countAt(0), maxMorton + 1,
        ComputeVerts<Func>({vertPos.ptrD(), index.ptrD(), gridVerts.D(), sdf_,
                            bounds.min, gridSize + 1, spacing, level}));
    vertPos.resize(index[0]);

    VecDH<glm::ivec3> triVerts(gridVerts.Entries() * 12);  // worst case

    index[0] = 0;
    thrust::for_each_n(
        countAt(0), gridVerts.Size(),
        BuildTris({triVerts.ptrD(), index.ptrD(), gridVerts.D()}));
    triVerts.resize(index[0]);

    RemoveUnreferencedVerts(vertPos, triVerts);

    out.vertPos.insert(out.vertPos.end(), vertPos.begin(), vertPos.end());
    out.triVerts.insert(out.triVerts.end(), triVerts.begin(), triVerts.end());
    return out;
  }

 private:
  const Func sdf_;
};
/** @} */
}  // namespace manifold