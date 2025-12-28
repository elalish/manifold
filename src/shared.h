// Copyright 2021 The Manifold Authors.
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

#include <memory>

#include "parallel.h"
#include "utils.h"
#include "vec.h"

namespace manifold {

inline vec3 SafeNormalize(vec3 v) {
  v = la::normalize(v);
  return std::isfinite(v.x) ? v : vec3(0.0);
}

inline double MaxEpsilon(double minEpsilon, const Box& bBox) {
  double epsilon = std::max(minEpsilon, kPrecision * bBox.Scale());
  return std::isfinite(epsilon) ? epsilon : -1;
}

inline int NextHalfedge(int current) {
  ++current;
  if (current % 3 == 0) current -= 3;
  return current;
}

inline mat3 NormalTransform(const mat3x4& transform) {
  return la::inverse(la::transpose(mat3(transform)));
}

/**
 * By using the closest axis-aligned projection to the normal instead of a
 * projection along the normal, we avoid introducing any rounding error.
 */
inline mat2x3 GetAxisAlignedProjection(vec3 normal) {
  vec3 absNormal = la::abs(normal);
  double xyzMax;
  mat3x2 projection;
  if (absNormal.z > absNormal.x && absNormal.z > absNormal.y) {
    projection = mat3x2({1.0, 0.0, 0.0},  //
                        {0.0, 1.0, 0.0});
    xyzMax = normal.z;
  } else if (absNormal.y > absNormal.x) {
    projection = mat3x2({0.0, 0.0, 1.0},  //
                        {1.0, 0.0, 0.0});
    xyzMax = normal.y;
  } else {
    projection = mat3x2({0.0, 1.0, 0.0},  //
                        {0.0, 0.0, 1.0});
    xyzMax = normal.x;
  }
  if (xyzMax < 0) projection[0] *= -1.0;
  return la::transpose(projection);
}

inline vec3 GetBarycentric(const vec3& v, const mat3& triPos,
                           double tolerance) {
  const mat3 edges(triPos[2] - triPos[1], triPos[0] - triPos[2],
                   triPos[1] - triPos[0]);
  const vec3 d2(la::dot(edges[0], edges[0]), la::dot(edges[1], edges[1]),
                la::dot(edges[2], edges[2]));
  const int longSide = d2[0] > d2[1] && d2[0] > d2[2] ? 0
                       : d2[1] > d2[2]                ? 1
                                                      : 2;
  const vec3 crossP = la::cross(edges[0], edges[1]);
  const double area2 = la::dot(crossP, crossP);
  const double tol2 = tolerance * tolerance;

  vec3 uvw(0.0);
  for (const int i : {0, 1, 2}) {
    const vec3 dv = v - triPos[i];
    if (la::dot(dv, dv) < tol2) {
      // Return exactly equal if within tolerance of vert.
      uvw[i] = 1;
      return uvw;
    }
  }

  if (d2[longSide] < tol2) {  // point
    return vec3(1, 0, 0);
  } else if (area2 > d2[longSide] * tol2) {  // triangle
    for (const int i : {0, 1, 2}) {
      const int j = Next3(i);
      const vec3 crossPv = la::cross(edges[i], v - triPos[j]);
      const double area2v = la::dot(crossPv, crossPv);
      // Return exactly equal if within tolerance of edge.
      uvw[i] = area2v < d2[i] * tol2 ? 0 : la::dot(crossPv, crossP);
    }
    uvw /= (uvw[0] + uvw[1] + uvw[2]);
    return uvw;
  } else {  // line
    const int nextV = Next3(longSide);
    const double alpha =
        la::dot(v - triPos[nextV], edges[longSide]) / d2[longSide];
    uvw[longSide] = 0;
    uvw[nextV] = 1 - alpha;
    const int lastV = Next3(nextV);
    uvw[lastV] = alpha;
    return uvw;
  }
}

/**
 * The fundamental component of the halfedge data structure used for storing and
 * operating on the Manifold.
 */
struct Halfedge {
  int startVert, endVert;
  int pairedHalfedge;
  int propVert;
  bool IsForward() const { return startVert < endVert; }
  bool operator<(const Halfedge& other) const {
    return startVert == other.startVert ? endVert < other.endVert
                                        : startVert < other.startVert;
  }
};

struct Barycentric {
  int tri;
  vec4 uvw;
};

struct TriRef {
  /// The unique ID of the mesh instance of this triangle. If .meshID and .tri
  /// match for two triangles, then they are coplanar and came from the same
  /// face.
  int meshID;
  /// The OriginalID of the mesh this triangle came from. This ID is ideal for
  /// reapplying properties like UV coordinates to the output mesh.
  int originalID;
  /// Probably the triangle index of the original triangle this was part of:
  /// Mesh.triVerts[tri], but it's an input, so just pass it along unchanged.
  int faceID;
  /// Triangles with the same coplanar ID are coplanar.
  int coplanarID;

  bool SameFace(const TriRef& other) const {
    return meshID == other.meshID && coplanarID == other.coplanarID &&
           faceID == other.faceID;
  }
};

/**
 * This is a temporary edge structure which only stores edges forward and
 * references the halfedge it was created from.
 */
struct TmpEdge {
  int first, second, halfedgeIdx;

  TmpEdge() {}
  TmpEdge(int start, int end, int idx) {
    first = std::min(start, end);
    second = std::max(start, end);
    halfedgeIdx = idx;
  }

  bool operator<(const TmpEdge& other) const {
    return first == other.first ? second < other.second : first < other.first;
  }
};

Vec<TmpEdge> inline CreateTmpEdges(const VecView<const Halfedge>& halfedge) {
  Vec<TmpEdge> edges(halfedge.size());
  for_each_n(autoPolicy(edges.size()), countAt(0), edges.size(),
             [&edges, &halfedge](const int idx) {
               const Halfedge& half = halfedge[idx];
               edges[idx] = TmpEdge(half.startVert, half.endVert,
                                    half.IsForward() ? idx : -1);
             });

  size_t numEdge =
      remove_if(edges.begin(), edges.end(),
                [](const TmpEdge& edge) { return edge.halfedgeIdx < 0; }) -
      edges.begin();
  DEBUG_ASSERT(numEdge == halfedge.size() / 2, topologyErr, "Not oriented!");
  edges.resize(numEdge);
  return edges;
}

#ifdef MANIFOLD_DEBUG
inline std::ostream& operator<<(std::ostream& stream, const Halfedge& edge) {
  return stream << "startVert = " << edge.startVert
                << ", endVert = " << edge.endVert
                << ", pairedHalfedge = " << edge.pairedHalfedge
                << ", propVert = " << edge.propVert;
}

inline std::ostream& operator<<(std::ostream& stream, const Barycentric& bary) {
  return stream << "tri = " << bary.tri << ", uvw = " << bary.uvw;
}

inline std::ostream& operator<<(std::ostream& stream, const TriRef& ref) {
  return stream << "meshID: " << ref.meshID
                << ", originalID: " << ref.originalID
                << ", faceID: " << ref.faceID
                << ", coplanarID: " << ref.coplanarID;
}
#endif

template <typename T>
class SharedVec {
 public:
  SharedVec() : data_(std::make_shared<Vec<T>>()) {}
  explicit SharedVec(const Vec<T>& vec)
      : data_(std::make_shared<Vec<T>>(vec)) {}
  SharedVec(const SharedVec& other) : data_(other.data_) {}
  SharedVec& operator=(const SharedVec& other) {
    if (this == &other) return *this;
    data_ = other.data_;
    return *this;
  }
  SharedVec& operator=(const Vec<T>& vec) {
    data_ = std::make_shared<Vec<T>>(vec);
    return *this;
  }
  SharedVec& operator=(Vec<T>&& vec) {
    data_ = std::make_shared<Vec<T>>(std::move(vec));
    return *this;
  }
  SharedVec(SharedVec&&) noexcept = default;
  SharedVec& operator=(SharedVec&&) noexcept = default;

  size_t size() const { return data_->size(); }
  bool empty() const { return data_->empty(); }

  T& operator[](size_t idx) { return (*data_)[idx]; }
  const T& operator[](size_t idx) const { return (*data_)[idx]; }

  auto begin() { return data_->begin(); }
  auto end() { return data_->end(); }
  auto begin() const { return data_->begin(); }
  auto end() const { return data_->end(); }
  auto cbegin() const { return data_->cbegin(); }
  auto cend() const { return data_->cend(); }

  void clear(bool shrink = true) { data_->clear(shrink); }
  void resize_nofill(size_t n) { data_->resize_nofill(n); }
  void resize(size_t n, T val = T()) { data_->resize(n, val); }
  void push_back(const T& val) { data_->push_back(val); }
  void extend(size_t n) { data_->extend(n); }
  void reserve(size_t n) { data_->reserve(n); }
  void swap(Vec<T>& other) { data_->swap(other); }
  void swap(SharedVec& other) { data_.swap(other.data_); }
  void shrink_to_fit() { data_->shrink_to_fit(); }

  operator VecView<T>() { return data_->view(); }
  operator VecView<T>() const { return data_->view(); }
  operator VecView<const T>() const { return data_->view(); }

  Vec<T>& vec() { return *data_; }
  const Vec<T>& vec() const { return *data_; }

  std::shared_ptr<Vec<T>> SharedPtr() { return data_; }
  std::shared_ptr<const Vec<T>> SharedPtr() const { return data_; }

  void MakeUnique() {
    if (data_.use_count() > 1) data_ = std::make_shared<Vec<T>>(data_->view());
  }

  Vec<T> AsVec() const { return Vec<T>(data_->view()); }

 private:
  std::shared_ptr<Vec<T>> data_;
};

}  // namespace manifold
