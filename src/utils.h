// Copyright 2020 The Manifold Authors, Jared Hoberock and Nathan Bell of
// NVIDIA Research
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
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "manifold/common.h"
#include "vec.h"

#ifndef MANIFOLD_PAR
#error "MANIFOLD_PAR must be defined to either 1 (parallel) or -1 (series)"
#else
#if (MANIFOLD_PAR != 1) && (MANIFOLD_PAR != -1)
#define XSTR(x) STR(x)
#define STR(x) #x
#pragma message "Current value of MANIFOLD_PAR is: " XSTR(MANIFOLD_PAR)
#error "MANIFOLD_PAR must be defined to either 1 (parallel) or -1 (series)"
#endif
#endif

#include "parallel.h"

#if __has_include(<tracy/Tracy.hpp>)
#include <tracy/Tracy.hpp>
#else
#define FrameMarkStart(x)
#define FrameMarkEnd(x)
// putting ZoneScoped in a function will instrument the function execution when
// TRACY_ENABLE is set, which allows the profiler to record more accurate
// timing.
#define ZoneScoped
#define ZoneScopedN(name)
#endif

namespace manifold {

/**
 * Stand-in for C++23's operator""uz (P0330R8)[https://wg21.link/P0330R8].
 */
[[nodiscard]] constexpr std::size_t operator""_uz(
    unsigned long long n) noexcept {
  return n;
}

constexpr double kPrecision = 1e-12;

inline int Next3(int i) {
  constexpr ivec3 next3(1, 2, 0);
  return next3[i];
}

inline int Prev3(int i) {
  constexpr ivec3 prev3(2, 0, 1);
  return prev3[i];
}

template <typename T, typename T1>
void Permute(Vec<T>& inOut, const Vec<T1>& new2Old) {
  Vec<T> tmp(std::move(inOut));
  inOut.resize_nofill(new2Old.size());
  gather(new2Old.begin(), new2Old.end(), tmp.begin(), inOut.begin());
}

template <typename T, typename T1>
void Permute(std::vector<T>& inOut, const Vec<T1>& new2Old) {
  std::vector<T> tmp(std::move(inOut));
  inOut.resize(new2Old.size());
  gather(new2Old.begin(), new2Old.end(), tmp.begin(), inOut.begin());
}

template <typename T>
T AtomicAdd(T& target, T add) {
  std::atomic<T>& tar = reinterpret_cast<std::atomic<T>&>(target);
  T old_val = tar.load();
  while (!tar.compare_exchange_weak(old_val, old_val + add,
                                    std::memory_order_seq_cst)) {
  }
  return old_val;
}

template <>
inline int AtomicAdd(int& target, int add) {
  std::atomic<int>& tar = reinterpret_cast<std::atomic<int>&>(target);
  int old_val = tar.fetch_add(add, std::memory_order_seq_cst);
  return old_val;
}

template <typename T>
class ConcurrentSharedPtr {
 public:
  ConcurrentSharedPtr(T value) : impl(std::make_shared<T>(value)) {}
  ConcurrentSharedPtr(const ConcurrentSharedPtr<T>& other)
      : impl(other.impl), mutex(other.mutex) {}
  ConcurrentSharedPtr& operator=(const ConcurrentSharedPtr<T>& other) {
    if (this == &other) return *this;
    impl = other.impl;
    mutex = other.mutex;
    return *this;
  }
  class SharedPtrGuard {
   public:
    SharedPtrGuard(std::recursive_mutex* mutex, T* content)
        : mutex(mutex), content(content) {
      mutex->lock();
    }
    ~SharedPtrGuard() { mutex->unlock(); }

    T& operator*() { return *content; }
    T* operator->() { return content; }

   private:
    std::recursive_mutex* mutex;
    T* content;
  };
  SharedPtrGuard GetGuard() { return SharedPtrGuard(mutex.get(), impl.get()); };
  unsigned int UseCount() { return impl.use_count(); };

 private:
  std::shared_ptr<T> impl;
  std::shared_ptr<std::recursive_mutex> mutex =
      std::make_shared<std::recursive_mutex>();
};

// from https://github.com/wjakob/dset, changed to add connected component
// computation
class DisjointSets {
 public:
  DisjointSets(uint32_t size) : mData(size) {
    for (uint32_t i = 0; i < size; ++i) mData[i] = (uint32_t)i;
  }

  uint32_t find(uint32_t id) const {
    while (id != parent(id)) {
      uint64_t value = mData[id];
      uint32_t new_parent = parent((uint32_t)value);
      uint64_t new_value = (value & 0xFFFFFFFF00000000ULL) | new_parent;
      /* Try to update parent (may fail, that's ok) */
      if (value != new_value) mData[id].compare_exchange_weak(value, new_value);
      id = new_parent;
    }
    return id;
  }

  uint32_t cfind(uint32_t id) const {
    while (id != parent(id)) {
      uint64_t value = mData[id];
      uint32_t new_parent = parent((uint32_t)value);
      id = new_parent;
    }
    return id;
  }

  bool same(uint32_t id1, uint32_t id2) const {
    for (;;) {
      id1 = find(id1);
      id2 = find(id2);
      if (id1 == id2) return true;
      if (parent(id1) == id1) return false;
    }
  }

  uint32_t unite(uint32_t id1, uint32_t id2) {
    for (;;) {
      id1 = find(id1);
      id2 = find(id2);

      if (id1 == id2) return id1;

      uint32_t r1 = rank(id1), r2 = rank(id2);

      if (r1 > r2 || (r1 == r2 && id1 < id2)) {
        std::swap(r1, r2);
        std::swap(id1, id2);
      }

      uint64_t oldEntry = ((uint64_t)r1 << 32) | id1;
      uint64_t newEntry = ((uint64_t)r1 << 32) | id2;

      if (!mData[id1].compare_exchange_strong(oldEntry, newEntry)) continue;

      if (r1 == r2) {
        oldEntry = ((uint64_t)r2 << 32) | id2;
        newEntry = ((uint64_t)(r2 + 1) << 32) | id2;
        /* Try to update the rank (may fail, retry if rank = 0) */
        if (!mData[id2].compare_exchange_strong(oldEntry, newEntry) && r2 == 0)
          continue;
      }

      break;
    }
    return id2;
  }

  uint32_t size() const { return (uint32_t)mData.size(); }

  uint32_t rank(uint32_t id) const {
    return ((uint32_t)(mData[id] >> 32)) & 0x7FFFFFFFu;
  }

  uint32_t parent(uint32_t id) const { return (uint32_t)mData[id]; }

  int connectedComponents(std::vector<int>& components) {
    components.resize(mData.size());
    int lonelyNodes = 0;
    std::unordered_map<uint32_t, int> toLabel;
    for (size_t i = 0; i < mData.size(); ++i) {
      // we optimize for connected component of size 1
      // no need to put them into the hashmap
      auto iParent = cfind(i);
      if (rank(iParent) == 0) {
        components[i] = static_cast<int>(toLabel.size()) + lonelyNodes++;
        continue;
      }
      auto iter = toLabel.find(iParent);
      if (iter == toLabel.end()) {
        auto s = static_cast<uint32_t>(toLabel.size()) + lonelyNodes;
        toLabel.insert(std::make_pair(iParent, s));
        components[i] = s;
      } else {
        components[i] = iter->second;
      }
    }
    return toLabel.size() + lonelyNodes;
  }

  mutable std::vector<std::atomic<uint64_t>> mData;
};

template <typename T>
struct Identity {
  T operator()(T v) const { return v; }
};

template <typename T>
struct Negate {
  T operator()(T v) const { return -v; }
};

/**
 * Determines if the three points are wound counter-clockwise, clockwise, or
 * colinear within the specified tolerance.
 *
 * @param p0 First point
 * @param p1 Second point
 * @param p2 Third point
 * @param tol Tolerance value for colinearity
 * @return int, like Signum, this returns 1 for CCW, -1 for CW, and 0 if within
 * tol of colinear.
 */
inline int CCW(vec2 p0, vec2 p1, vec2 p2, double tol) {
  vec2 v1 = p1 - p0;
  vec2 v2 = p2 - p0;
  double area = v1.x * v2.y - v1.y * v2.x;
  double base2 = la::max(la::dot(v1, v1), la::dot(v2, v2));
  if (area * area * 4 <= base2 * tol * tol)
    return 0;
  else
    return area > 0 ? 1 : -1;
}

inline mat4 Mat4(mat3x4 a) {
  return mat4({a[0], 0}, {a[1], 0}, {a[2], 0}, {a[3], 1});
}
inline mat3 Mat3(mat2x3 a) { return mat3({a[0], 0}, {a[1], 0}, {a[2], 1}); }

// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
constexpr uint64_t hash64bit(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  x = x ^ (x >> 31);
  return x;
}
}  // namespace manifold
