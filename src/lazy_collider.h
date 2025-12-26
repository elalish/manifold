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
#include <mutex>

#include "collider.h"

namespace manifold {

class LazyCollider {
 public:
  struct LeafData {
    Vec<Box> leafBox;
    Vec<uint32_t> leafMorton;
  };

  struct Built {
    std::shared_ptr<Collider> collider;
    std::optional<mat3x4> transform;
  };

  struct Base {
    std::shared_ptr<const LazyCollider> base;
    mat3x4 transform;
  };

  static LazyCollider Empty() { return LazyCollider(LeafData{{}, {}}); }
  LazyCollider();
  LazyCollider(LeafData&& leafData);
  LazyCollider(std::shared_ptr<const LazyCollider> base,
               const mat3x4& transform);
  LazyCollider(const LazyCollider& other);
  LazyCollider(LazyCollider&& other) noexcept;
  LazyCollider& operator=(const LazyCollider& other);
  LazyCollider& operator=(LazyCollider&& other) noexcept;
  ~LazyCollider() = default;

  bool IsBuilt() const;

  template <const bool selfCollision, typename Recorder>
  struct Adapter {
    using Local = typename Recorder::Local;
    Recorder* recorder;

    Local& local() { return recorder->local(); }

    void record(int queryIdx, int leafIdx, Local& local) {
      if constexpr (selfCollision) {
        if (leafIdx == queryIdx) return;
      }
      recorder->record(queryIdx, leafIdx, local);
    }
  };

  template <const bool selfCollision = false, typename T, typename Recorder>
  void Collisions(const VecView<const T>& queries, Recorder& recorder,
                  bool parallel = true) const {
    const Built& built = EnsureBuilt();

    Adapter<selfCollision, Recorder> adapter{&recorder};
    built.collider->Collisions<false>(queries, adapter, built.transform,
                                      parallel);
  }

  template <const bool selfCollision = false, typename F, typename Recorder>
  void Collisions(F f, int n, Recorder& recorder, bool parallel = true) const {
    const Built& built = EnsureBuilt();
    Adapter<selfCollision, Recorder> adapter{&recorder};
    built.collider->Collisions<false>(f, n, adapter, built.transform, parallel);
  }

  static bool IsAxisAligned(const mat3x4& transform);

 private:
  const Built& EnsureBuilt() const;

  mutable std::mutex mutex_;

  mutable std::optional<Built> built_;
  mutable std::optional<LeafData> leafData_;
  mutable std::optional<Base> base_;
};

}  // namespace manifold
