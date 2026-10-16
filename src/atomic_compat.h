// Copyright 2026 The Manifold Authors.
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
//
// C++17 stand-ins for atomic facilities the library would otherwise get from
// C++20. The two halves have different lifetimes:
//
//   - `AtomicRef<T>` is `std::atomic_ref` (P0019R8) at C++20, and at C++17 a
//     thin wrapper over the reinterpret_cast it replaces. The cast is still
//     undefined behavior, but it is now in one place instead of scattered
//     across the call sites, its preconditions are static_asserts rather than
//     assumptions, and the C++20 lane builds the defined version.
//   - `AtomicLoadShared` / `AtomicStoreShared` wrap the `std::atomic_load` and
//     `std::atomic_store` shared_ptr overloads, which C++20 deprecates and
//     C++26 removes (P2869). These do not go away at C++20: the replacement
//     changes the member's type to `std::atomic<std::shared_ptr<T>>` rather
//     than how it is accessed, and libc++ still leaves
//     `__cpp_lib_atomic_shared_ptr` undefined. Revisit when the baseline moves
//     and libc++ ships it.

#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "manifold/optional_assert.h"

namespace manifold {

#if defined(__cpp_lib_atomic_ref)  // C++20: the real thing

template <typename T>
using AtomicRef = std::atomic_ref<T>;

#else  // C++17: the cast, confined

/**
 * A C++17 stand-in for std::atomic_ref. Reinterprets the referenced object as
 * `std::atomic<T>` and forwards to it, which is what the call sites did before
 * this type existed.
 *
 * That cast is undefined behavior: the object is not an atomic object, and
 * nothing in the standard says the two are layout-compatible. It works on
 * every ABI we ship, and the static_asserts below check the parts a compiler
 * can check, but they cannot make it defined. The point of the type is that
 * there is now exactly one cast to audit, and that at C++20 it is replaced
 * wholesale by the alias above rather than being repaired.
 *
 * The referenced object must outlive the AtomicRef, and while any AtomicRef
 * refers to it every access must go through one.
 *
 * Deliberately a subset of std::atomic_ref: only what the call sites use, so
 * the C++20 alias is a superset and anything written against this compiles
 * either way. Missing on purpose are `operator=`, `operator T`, the other
 * fetch_* operations, the compound assignments, and wait/notify. Adding one
 * here is fine; relying on one that only the alias has builds on the C++20
 * lane and fails everywhere else.
 */
template <typename T>
class AtomicRef {
  // Every use here is a scalar. Narrower than std::atomic_ref's trivially-
  // copyable contract, and it keeps padded types - whose padding bytes would
  // have to be cleared before each compare_exchange - out of the cast.
  static_assert(std::is_scalar<T>::value, "AtomicRef requires a scalar type");
  // What makes the cast work in practice. A lock-based std::atomic<T> carries
  // a lock alongside the value, so these would catch it.
  static_assert(sizeof(std::atomic<T>) == sizeof(T),
                "std::atomic<T> must be the same size as T");
  static_assert(alignof(std::atomic<T>) == alignof(T),
                "std::atomic<T> must be aligned like T");
  static_assert(std::atomic<T>::is_always_lock_free,
                "AtomicRef requires a lock-free std::atomic<T>");

 public:
  explicit AtomicRef(T& obj) : ref_(reinterpret_cast<std::atomic<T>&>(obj)) {
    // Natural alignment is not implied by alignof(T): 32-bit x86 gives
    // alignof(double) == 4. Compiles away without MANIFOLD_DEBUG. The C++20
    // alias has no equivalent check - std::atomic_ref states the requirement
    // as required_alignment and leaves violations undefined - so this catches
    // a misaligned caller only on the C++17 side.
    DEBUG_ASSERT(reinterpret_cast<uintptr_t>(&obj) % sizeof(T) == 0, logicErr,
                 "AtomicRef requires an object aligned to its own size");
  }

  T load(std::memory_order order = std::memory_order_seq_cst) const {
    return ref_.load(order);
  }

  void store(T desired,
             std::memory_order order = std::memory_order_seq_cst) const {
    ref_.store(desired, order);
  }

  T exchange(T desired,
             std::memory_order order = std::memory_order_seq_cst) const {
    return ref_.exchange(desired, order);
  }

  // As in std::atomic_ref, `expected` is updated with the current value when
  // the exchange fails. Only the weak form may fail spuriously, so a caller
  // that treats failure as evidence about the stored value needs the strong
  // one.
  bool compare_exchange_weak(
      T& expected, T desired,
      std::memory_order order = std::memory_order_seq_cst) const {
    return ref_.compare_exchange_weak(expected, desired, order);
  }

  bool compare_exchange_weak(T& expected, T desired, std::memory_order success,
                             std::memory_order failure) const {
    return ref_.compare_exchange_weak(expected, desired, success, failure);
  }

  bool compare_exchange_strong(
      T& expected, T desired,
      std::memory_order order = std::memory_order_seq_cst) const {
    return ref_.compare_exchange_strong(expected, desired, order);
  }

  // Integral only: C++17's std::atomic has no floating-point fetch_add, so
  // this gate is what keeps `double` on AtomicAdd's compare_exchange_weak
  // loop. Keep it.
  template <typename U = T>
  std::enable_if_t<std::is_integral<U>::value, T> fetch_add(
      T arg, std::memory_order order = std::memory_order_seq_cst) const {
    return ref_.fetch_add(arg, order);
  }

 private:
  std::atomic<T>& ref_;
};

#endif  // std::atomic_ref vs the C++17 cast

// Suppresses every deprecation inside its region, so keep the regions to a
// single expression. Neither GCC nor Clang offers per-symbol suppression.
#if defined(_MSC_VER) && !defined(__clang__)
#define MANIFOLD_IGNORE_DEPRECATED_BEGIN \
  __pragma(warning(push)) __pragma(warning(disable : 4996))
#define MANIFOLD_IGNORE_DEPRECATED_END __pragma(warning(pop))
#else
#define MANIFOLD_IGNORE_DEPRECATED_BEGIN \
  _Pragma("GCC diagnostic push")         \
      _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#define MANIFOLD_IGNORE_DEPRECATED_END _Pragma("GCC diagnostic pop")
#endif

// Atomic access to a plain `shared_ptr`, which C++17 can only express through
// these overloads. There is no atomic_ref equivalent: atomic_ref requires a
// trivially copyable type, so the C++20 replacement changes the member's type
// rather than how it is accessed.
template <typename T>
std::shared_ptr<T> AtomicLoadShared(const std::shared_ptr<T>* ptr) {
  MANIFOLD_IGNORE_DEPRECATED_BEGIN
  return std::atomic_load(ptr);
  MANIFOLD_IGNORE_DEPRECATED_END
}

template <typename T>
void AtomicStoreShared(std::shared_ptr<T>* ptr, std::shared_ptr<T> value) {
  MANIFOLD_IGNORE_DEPRECATED_BEGIN
  std::atomic_store(ptr, std::move(value));
  MANIFOLD_IGNORE_DEPRECATED_END
}

}  // namespace manifold
