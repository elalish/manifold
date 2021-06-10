// Copyright 2021 Emmett Lalish
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
#define GLM_FORCE_EXPLICIT_CTOR
#include <chrono>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <iostream>
#include <sstream>
#include <vector>

namespace manifold {

constexpr float kTolerance = 1e-5;

struct userErr : public virtual std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct topologyErr : public virtual std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct geometryErr : public virtual std::runtime_error {
  using std::runtime_error::runtime_error;
};
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

inline HOST_DEVICE int Signum(float val) { return (val > 0) - (val < 0); }

// Use sind and cosd for inputs in degrees so that multiples of 90 degrees come
// out exact.
inline HOST_DEVICE float sind(float x) {
  if (!std::isfinite(x)) return sin(x);
  if (x < 0.0f) return -sind(-x);
  int quo;
  x = remquo(fabs(x), 90.0f, &quo);
  switch (quo % 4) {
    case 0:
      return sin(glm::radians(x));
    case 1:
      return cos(glm::radians(x));
    case 2:
      return -sin(glm::radians(x));
    case 3:
      return -cos(glm::radians(x));
  }
  return 0.0f;
}

inline HOST_DEVICE float cosd(float x) { return sind(x + 90.0f); }

inline HOST_DEVICE int CCW(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2,
                           float tol) {
  glm::vec2 v1 = p1 - p0;
  glm::vec2 v2 = p2 - p0;
  float area = v1.x * v2.y - v1.y * v2.x;
  float base2 = glm::max(glm::dot(v1, v1), glm::dot(v2, v2));
  if (area * area <= base2 * tol * tol)
    return 0;
  else
    return area > 0 ? 1 : -1;
}

inline HOST_DEVICE glm::mat4x3 RotateUp(glm::vec3 up) {
  up = glm::normalize(up);
  glm::vec3 axis = glm::cross(up, {0, 0, 1});
  float angle = glm::asin(glm::length(axis));
  if (glm::dot(up, {0, 0, 1}) < 0) angle = glm::pi<float>() - angle;
  return glm::mat4x3(glm::rotate(glm::mat4(1), angle, axis));
}

struct ExecutionParams {
  bool intermediateChecks = false;
  bool verbose = false;
  bool suppressErrors = false;
};

struct Halfedge {
  int startVert, endVert;
  int pairedHalfedge;
  int face;
  HOST_DEVICE bool IsForward() const { return startVert < endVert; }
  HOST_DEVICE bool operator<(const Halfedge& other) const {
    return startVert == other.startVert ? endVert < other.endVert
                                        : startVert < other.startVert;
  }
};

struct PolyVert {
  glm::vec2 pos;
  int idx;
};

using SimplePolygon = std::vector<PolyVert>;
using Polygons = std::vector<SimplePolygon>;

struct Mesh {
  std::vector<glm::vec3> vertPos;
  std::vector<glm::vec3> vertNormal;
  std::vector<glm::ivec3> triVerts;
  std::vector<glm::vec4> halfedgeBezier;
};

struct Barycentric {
  int tri;
  glm::vec3 uvw;
};

struct BaryRef {
  int tri;
  glm::ivec3 vertBary;
};

struct MeshRelation {
  std::vector<glm::vec3> barycentric;
  std::vector<BaryRef> triBary;
};

struct Box {
  glm::vec3 min = glm::vec3(1.0f / 0.0f);
  glm::vec3 max = glm::vec3(-1.0f / 0.0f);

  HOST_DEVICE Box() {}
  HOST_DEVICE Box(const glm::vec3 p1, const glm::vec3 p2) {
    min = glm::min(p1, p2);
    max = glm::max(p1, p2);
  }

  HOST_DEVICE glm::vec3 Size() const { return max - min; }

  HOST_DEVICE glm::vec3 Center() const { return 0.5f * (max + min); }

  HOST_DEVICE float Scale() const {
    glm::vec3 absMax = glm::max(glm::abs(min), glm::abs(max));
    return glm::max(absMax.x, glm::max(absMax.y, absMax.z));
  }

  HOST_DEVICE bool Contains(const Box& box) const {
    return glm::all(glm::greaterThanEqual(box.min, min)) &&
           glm::all(glm::greaterThanEqual(max, box.max));
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

  // Ensure the transform passed in is axis-aligned (rotations are all multiples
  // of 90 degrees), or else the resulting bounding box will no longer bound
  // properly.
  HOST_DEVICE Box Transform(const glm::mat4x3& transform) const {
    Box out;
    glm::vec3 minT = transform * glm::vec4(min, 1.0f);
    glm::vec3 maxT = transform * glm::vec4(max, 1.0f);
    out.min = glm::min(minT, maxT);
    out.max = glm::max(minT, maxT);
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

  HOST_DEVICE bool isFinite() const {
    return glm::all(glm::isfinite(min)) && glm::all(glm::isfinite(max));
  }
};

template <typename T>
void Dump(const std::vector<T>& vec) {
  std::cout << "Vec = " << std::endl;
  for (int i = 0; i < vec.size(); ++i) {
    std::cout << i << ", " << vec[i] << ", " << std::endl;
  }
  std::cout << std::endl;
}

inline std::ostream& operator<<(std::ostream& stream, const Box& box) {
  return stream << "min: " << box.min.x << ", " << box.min.y << ", "
                << box.min.z << ", "
                << "max: " << box.max.x << ", " << box.max.y << ", "
                << box.max.z;
}

inline std::ostream& operator<<(std::ostream& stream, const Halfedge& edge) {
  return stream << "startVert = " << edge.startVert
                << ", endVert = " << edge.endVert
                << ", pairedHalfedge = " << edge.pairedHalfedge
                << ", face = " << edge.face;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& stream, const glm::tvec2<T>& v) {
  return stream << "x = " << v.x << ", y = " << v.y;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& stream, const glm::tvec3<T>& v) {
  return stream << "x = " << v.x << ", y = " << v.y << ", z = " << v.z;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& stream, const glm::tvec4<T>& v) {
  return stream << "x = " << v.x << ", y = " << v.y << ", z = " << v.z
                << ", w = " << v.w;
}

inline std::ostream& operator<<(std::ostream& stream, const glm::mat4x3& mat) {
  glm::mat3x4 tam = glm::transpose(mat);
  return stream << tam[0] << std::endl
                << tam[1] << std::endl
                << tam[2] << std::endl;
}

inline std::ostream& operator<<(std::ostream& stream, const BaryRef& ref) {
  return stream << "tri = " << ref.tri << ", uvw idx = " << ref.vertBary;
}

inline std::ostream& operator<<(std::ostream& stream, const Barycentric& bary) {
  return stream << "tri = " << bary.tri << ", uvw idx = " << bary.uvw;
}

}  // namespace manifold

#undef HOST_DEVICE