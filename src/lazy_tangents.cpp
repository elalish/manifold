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

#include "lazy_tangents.h"

#include "mesh_fixes.h"

namespace manifold {

LazyTangents::LazyTangents() : storage_(std::make_shared<Storage>()) {}

LazyTangents::LazyTangents(Vec<vec4> tangents)
    : storage_(std::make_shared<Storage>()) {
  storage_->sizeHint = tangents.size();
  storage_->tangents.swap(tangents);
}

LazyTangents::LazyTangents(std::shared_ptr<Storage> storage)
    : storage_(std::move(storage)) {}

LazyTangents::LazyTangents(const LazyTangents& other)
    : storage_(std::make_shared<Storage>(*other.storage_)) {}

LazyTangents& LazyTangents::operator=(const LazyTangents& other) {
  if (this == &other) return *this;
  storage_ = std::make_shared<Storage>(*other.storage_);
  return *this;
}

LazyTangents LazyTangents::FromTransform(
    const LazyTangents& base, const mat3& transform, bool invert,
    std::shared_ptr<const Vec<Halfedge>> halfedge) {
  LazyTangents out;
  if (base.size() == 0) return out;

  out.storage_->sizeHint = base.size();
  out.storage_->pending =
      Pending{base.storage_, transform, invert, std::move(halfedge)};
  out.storage_->tangents.clear(true);
  return out;
}

size_t LazyTangents::size() const {
  return storage_->pending ? storage_->sizeHint : storage_->tangents.size();
}

bool LazyTangents::empty() const { return size() == 0; }

vec4& LazyTangents::operator[](size_t idx) {
  EnsureMaterialized();
  return storage_->tangents[idx];
}

const vec4& LazyTangents::operator[](size_t idx) const {
  EnsureMaterialized();
  return storage_->tangents[idx];
}

VecView<vec4> LazyTangents::view() {
  EnsureMaterialized();
  return storage_->tangents.view();
}

VecView<const vec4> LazyTangents::view() const {
  EnsureMaterialized();
  return storage_->tangents.view();
}

void LazyTangents::clear(bool shrink) {
  storage_->pending.reset();
  storage_->sizeHint = 0;
  storage_->tangents.clear(shrink);
}

void LazyTangents::resize_nofill(size_t n) {
  EnsureMaterialized();
  storage_->tangents.resize_nofill(n);
  storage_->sizeHint = storage_->tangents.size();
}

void LazyTangents::resize(size_t n, vec4 val) {
  EnsureMaterialized();
  storage_->tangents.resize(n, val);
  storage_->sizeHint = storage_->tangents.size();
}

void LazyTangents::swap(Vec<vec4>& other) {
  EnsureMaterialized();
  storage_->tangents.swap(other);
  storage_->sizeHint = storage_->tangents.size();
}

Vec<vec4> LazyTangents::AsVec() const {
  EnsureMaterialized();
  return Vec<vec4>(storage_->tangents.view());
}

std::shared_ptr<const Vec<vec4>> LazyTangents::SharedData() const {
  EnsureMaterialized();
  return std::shared_ptr<const Vec<vec4>>(storage_, &storage_->tangents);
}

void LazyTangents::EnsureMaterialized() const { Materialize(storage_); }

void LazyTangents::Materialize(const std::shared_ptr<Storage>& storage) {
  if (!storage->pending) return;

  Pending pending = *storage->pending;
  Materialize(pending.base);
  const Vec<vec4>& baseTangents = pending.base->tangents;

  storage->tangents.resize_nofill(baseTangents.size());
  if (!baseTangents.empty()) {
    auto policy = autoPolicy(baseTangents.size());
    for_each_n(
        policy, countAt(0), baseTangents.size(),
        TransformTangents({storage->tangents, 0, pending.transform,
                           pending.invert, baseTangents, *pending.halfedge}));
  }
  storage->pending.reset();
  storage->sizeHint = storage->tangents.size();
}

}  // namespace manifold
