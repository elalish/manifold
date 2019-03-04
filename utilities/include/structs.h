// Copyright 2019 Emmett Lalish
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
#include <glm/glm.hpp>
#include <iostream>
#include <sstream>
#include <vector>

namespace manifold {

using runtimeErr = std::runtime_error;
using logicErr = std::logic_error;

template <typename Ex>
void AlwaysAssert(bool condition, const char* file, int line,
                  const std::string& cond, const std::string& msg) {
  if (!condition) {
    std::ostringstream output;
    output << "Error in file: " << file << " (" << line << "): \'" << cond
           << "\' is false: " << msg;
    throw Ex(output.str());
  }
}

#define ALWAYS_ASSERT(condition, EX, msg) \
  AlwaysAssert<EX>(condition, __FILE__, __LINE__, #condition, msg);

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

struct EdgeIdx {
  uint32_t idx;
  HOST_DEVICE EdgeIdx(int ind, int dir) {
    idx = static_cast<uint32_t>(ind) + (dir > 0 ? 0 : (1U << 31));
  }
  HOST_DEVICE int Dir() const { return idx < (1U << 31) ? 1 : -1; }
  HOST_DEVICE int Idx() const {
    return idx < (1U << 31) ? idx : idx - (1U << 31);
  }
};

struct PolyVert {
  glm::vec2 pos;
  int idx;
};

using EdgeVerts = std::pair<int, int>;
using TriVerts = glm::ivec3;
using TriEdges = glm::tvec3<EdgeIdx>;
using SimplePolygon = std::vector<PolyVert>;
using Polygons = std::vector<SimplePolygon>;

struct Box {
  glm::vec3 min = glm::vec3(0.0f / 0.0f);
  glm::vec3 max = glm::vec3(0.0f / 0.0f);

  HOST_DEVICE Box() {}
  HOST_DEVICE Box(const glm::vec3 p1, const glm::vec3 p2) {
    min = glm::min(p1, p2);
    max = glm::max(p1, p2);
  }

  HOST_DEVICE void Union(const glm::vec3 p) {
    min = glm::min(min, p);
    max = glm::max(max, p);
  }

  HOST_DEVICE Box Union(const Box& box) const {
    Box out;
    out.min = glm::min(min, box.min);
    out.max = glm::max(max, box.max);
    return out;
  }

  HOST_DEVICE Box Transform(const glm::mat4& transform) const {
    Box out;  // TODO: update to support perspective transforms
    out.min = glm::vec3(transform * glm::vec4(min, 1.0f));
    out.max = glm::vec3(transform * glm::vec4(max, 1.0f));
    return out;
  }

  HOST_DEVICE Box operator+(glm::vec3 shift) const {
    Box out;
    out.min = min + shift;
    out.max = max + shift;
    return out;
  }

  HOST_DEVICE Box& operator+=(glm::vec3 shift) {
    min += shift;
    max += shift;
    return *this;
  }

  HOST_DEVICE Box operator*(glm::vec3 scale) const {
    Box out;
    out.min = min * scale;
    out.max = max * scale;
    return out;
  }

  HOST_DEVICE Box& operator*=(glm::vec3 scale) {
    min *= scale;
    max *= scale;
    return *this;
  }

  HOST_DEVICE bool DoesOverlap(const Box& box) const {
    return min.x <= box.max.x && min.y <= box.max.y && min.z <= box.max.z &&
           max.x >= box.min.x && max.y >= box.min.y && max.z >= box.min.z;
  }

  HOST_DEVICE bool DoesOverlap(glm::vec3 p) const {  // projected in z
    return p.x <= max.x && p.x >= min.x && p.y <= max.y && p.y >= min.y;
  }
};

inline std::ostream& operator<<(std::ostream& stream, const Box& box) {
  return stream << "min: " << box.min.x << ", " << box.min.y << ", "
                << box.min.z << ", "
                << "max: " << box.max.x << ", " << box.max.y << ", "
                << box.max.z;
}

inline std::ostream& operator<<(std::ostream& stream, const EdgeIdx& edge) {
  return stream << (edge.Dir() > 0 ? "F-" : "B-") << edge.Idx();
}

template <typename T>
inline std::ostream& operator<<(std::ostream& stream, const glm::tvec3<T>& v) {
  return stream << "x = " << v.x << ", y = " << v.y << ", z = " << v.z;
}
}  // namespace manifold

#undef HOST_DEVICE