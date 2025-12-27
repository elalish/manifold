// Copyright 2025 The Manifold Authors.
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
#include <optional>

#include "shared.h"

namespace manifold {

class LazyTangents {
 public:
  struct Storage;
  struct Pending {
    std::shared_ptr<Storage> base;
    mat3 transform;
    bool invert;
    std::shared_ptr<const Vec<Halfedge>> halfedge;
  };
  struct Storage {
    Vec<vec4> tangents;
    std::optional<Pending> pending;
    size_t sizeHint = 0;
  };

  LazyTangents();
  explicit LazyTangents(Vec<vec4> tangents);
  LazyTangents(const LazyTangents& other);
  LazyTangents& operator=(const LazyTangents& other);
  LazyTangents(LazyTangents&&) noexcept = default;
  LazyTangents& operator=(LazyTangents&&) noexcept = default;

  static LazyTangents FromTransform(
      const LazyTangents& base, const mat3& transform, bool invert,
      std::shared_ptr<const Vec<Halfedge>> halfedge);

  size_t size() const;
  bool empty() const;

  vec4& operator[](size_t idx);
  const vec4& operator[](size_t idx) const;

  VecView<vec4> view();
  VecView<const vec4> view() const;

  auto begin() {
    EnsureMaterialized();
    return storage_->tangents.begin();
  }
  auto end() {
    EnsureMaterialized();
    return storage_->tangents.end();
  }
  auto begin() const {
    EnsureMaterialized();
    return storage_->tangents.begin();
  }
  auto end() const {
    EnsureMaterialized();
    return storage_->tangents.end();
  }
  auto cbegin() const {
    EnsureMaterialized();
    return storage_->tangents.cbegin();
  }
  auto cend() const {
    EnsureMaterialized();
    return storage_->tangents.cend();
  }

  void clear(bool shrink = true);
  void resize_nofill(size_t n);
  void resize(size_t n, vec4 val = vec4(0.0));
  void swap(Vec<vec4>& other);
  void swap(LazyTangents& other) { storage_.swap(other.storage_); }

  Vec<vec4> AsVec() const;
  operator Vec<vec4>() const { return AsVec(); }
  operator VecView<vec4>() { return view(); }
  operator VecView<const vec4>() const { return view(); }
  std::shared_ptr<const Vec<vec4>> SharedData() const;

 private:
  explicit LazyTangents(std::shared_ptr<Storage> storage);

  void EnsureMaterialized() const;
  static void Materialize(const std::shared_ptr<Storage>& storage);

  std::shared_ptr<Storage> storage_;
};

}  // namespace manifold
