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

#include "lazy_collider.h"

namespace manifold {
LazyCollider::LazyCollider() = default;

LazyCollider::LazyCollider(LeafData&& leafData) : leafData_{leafData} {}

LazyCollider::LazyCollider(std::shared_ptr<const LazyCollider> base,
                           const mat3x4& transform) {
  DEBUG_ASSERT(base != nullptr, logicErr, "invalid base ptr");
  mat3x4 composed = transform;
  while (base) {
    std::lock_guard<std::mutex> lock(base->mutex_);
    if (!base->base_) break;
    composed = composed * Mat4(base->base_->transform);
    base = base->base_->base;
  }

  base_ = {base, composed};
}

LazyCollider::LazyCollider(const LazyCollider& other) {
  std::lock_guard<std::mutex> lock(other.mutex_);
  built_ = other.built_;
  leafData_ = other.leafData_;
  base_ = other.base_;
}

LazyCollider::LazyCollider(LazyCollider&& other) noexcept {
  std::lock_guard<std::mutex> lock(other.mutex_);
  built_ = std::move(other.built_);
  leafData_ = std::move(other.leafData_);
  base_ = std::move(other.base_);
}

LazyCollider& LazyCollider::operator=(const LazyCollider& other) {
  if (this == &other) return *this;
  std::scoped_lock lock(mutex_, other.mutex_);
  built_ = other.built_;
  leafData_ = other.leafData_;
  base_ = other.base_;
  return *this;
}

LazyCollider& LazyCollider::operator=(LazyCollider&& other) noexcept {
  if (this == &other) return *this;
  std::scoped_lock lock(mutex_, other.mutex_);
  built_ = std::move(other.built_);
  leafData_ = std::move(other.leafData_);
  base_ = std::move(other.base_);
  return *this;
}

bool LazyCollider::IsBuilt() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return built_.has_value();
}

bool LazyCollider::IsAxisAligned(const mat3x4& transform) {
  for (int row : {0, 1, 2}) {
    int count = 0;
    for (int col : {0, 1, 2}) {
      if (transform[col][row] == 0.0) ++count;
    }
    if (count != 2) return false;
  }
  return true;
}

const LazyCollider::Built& LazyCollider::EnsureBuilt() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (built_) return built_.value();

  if (base_) {
    const Built& baseBuilt = base_->base->EnsureBuilt();
    mat3x4 composed =
        base_->transform * Mat4(baseBuilt.transform.value_or(la::identity));
    built_ = {baseBuilt.collider, composed};
    base_ = std::nullopt;
  } else {
    DEBUG_ASSERT(leafData_.has_value(), logicErr, "uninitialized collider");
    built_ = {
        std::make_shared<Collider>(leafData_->leafBox, leafData_->leafMorton),
        std::nullopt};
    leafData_ = std::nullopt;
  }
  return built_.value();
}

}  // namespace manifold
