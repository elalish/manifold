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
// C++20. Everything here is scaffolding for that gap, collected in one file so
// it can be removed in one go once the baseline moves:
//
//   - `AtomicRef<T>` is `std::atomic_ref` (P0019R8). It replaces a
//     reinterpret_cast of a plain `T&` to `std::atomic<T>&`, which was
//     undefined behavior that happened to work on every ABI we ship
//     (elalish/manifold#1153). At C++20 it becomes a plain alias. 32-bit MSVC
//     is the exception: it keeps the cast, since the intrinsics needed to
//     avoid it are not all available there.
//   - `AtomicLoadShared` / `AtomicStoreShared` wrap the `std::atomic_load` and
//     `std::atomic_store` shared_ptr overloads, which C++20 deprecates and
//     C++26 removes (P2869). Their replacement is a member of type
//     `std::atomic<std::shared_ptr<T>>`, which is C++20-only and would delete
//     `Manifold`'s defaulted move operations, so it is a separate change.

#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <type_traits>

// Only the 64-bit MSVC backend below needs these, and <intrin.h> is heavy, so
// keep them out of every other configuration's include graph. They must stay
// at file scope: an #include inside a namespace nests whatever it declares.
#if defined(_MSC_VER) && !defined(__clang__) && !defined(_M_IX86) && \
    !defined(_M_ARM) && !defined(__cpp_lib_atomic_ref)
#include <intrin.h>

#include <cstring>
#endif

#include "manifold/optional_assert.h"

namespace manifold {

#if defined(__cpp_lib_atomic_ref)  // C++20: the real thing

template <typename T>
using AtomicRef = std::atomic_ref<T>;

#else  // C++17: everything below stands in for it

namespace details {

#if defined(_MSC_VER) && !defined(__clang__)  // MSVC backend

// 32-bit MSVC keeps the pre-#1153 behavior. The intrinsics below are not all
// available there (_InterlockedExchange64 and _InterlockedExchangeAdd64 are
// documented for ARM/x64/ARM64 only), and __iso_volatile_load64/store64 are
// not atomic on a 32-bit target - the compiler may split them into two 32-bit
// accesses. Writing the CAS-loop workarounds MSVC STL uses would mean carrying
// code no CI lane can compile, so instead this target falls back to exactly
// what the tree did before: the reinterpret_cast to std::atomic. That is still
// the undefined behavior #1153 is about, but it is the status quo rather than
// a regression, and refusing to build would take away a working configuration.
#if defined(_M_IX86) || defined(_M_ARM)
#pragma message( \
    "manifold: 32-bit MSVC keeps the pre-#1153 reinterpret_cast for atomics; build x64/ARM64 or /std:c++20 for the checked path")

template <typename T>
struct AtomicBackend {
  static std::atomic<T>& Ref(T* p) {
    return reinterpret_cast<std::atomic<T>&>(*p);
  }
  static T Load(T* p, std::memory_order order) { return Ref(p).load(order); }
  static void Store(T* p, T desired, std::memory_order order) {
    Ref(p).store(desired, order);
  }
  static T Exchange(T* p, T desired, std::memory_order order) {
    return Ref(p).exchange(desired, order);
  }
  static bool CompareExchange(T* p, T& expected, T desired,
                              std::memory_order order, bool weak) {
    return weak ? Ref(p).compare_exchange_weak(expected, desired, order)
                : Ref(p).compare_exchange_strong(expected, desired, order);
  }
  // std::atomic<double>::fetch_add does not exist in C++17. This compiles
  // only because AtomicRef::fetch_add is SFINAE-gated on is_integral, so this
  // body is never instantiated for a floating-point T. Keep that gate.
  static T FetchAdd(T* p, T arg, std::memory_order order) {
    return Ref(p).fetch_add(arg, order);
  }
};

#else  // 64-bit MSVC: real intrinsics

// One integer width per object size, reached through memcpy so the same code
// serves `double` as well as the integer types. The read-modify-write
// intrinsics are all full barriers, so a weaker requested order is satisfied
// by being stronger than asked; `Load` alone honors the order exactly.
template <size_t N>
struct MsvcAtomic;

// On ARM this is a fence-based mapping (ldr + dmb) where MSVC's std::atomic
// uses ldar/stlr. The two interoperate because MSVC's seq_cst store ends in a
// full dmb, but nothing here relies on that: no AtomicRef operation needs to
// share the seq_cst total order with a std::atomic one. HashTableD is the only
// structure that mixes the two - its used_ counter sits beside AtomicRef-
// accessed keys - and it reads used_ relaxed on every concurrent path.
// Ordering the two would need checking against that mapping first.
//
// MSVC's own _Compiler_or_memory_barrier: on x86/x64 the hardware already
// orders loads, so acquire and seq_cst need only a compiler barrier, and
// std::atomic_thread_fence would instead emit a locked _InterlockedIncrement.
// ARM is weakly ordered and needs the real dmb.
inline void PostLoadBarrier(std::memory_order order) {
#if defined(_M_ARM) || defined(_M_ARM64) || defined(_M_ARM64EC)
  std::atomic_thread_fence(order);
#else
  std::atomic_signal_fence(order);
#endif
}

// `Load` must stay a plain load: the _Interlocked* family is read-modify-write,
// which would both write through a const reference and turn HashTableD's probe
// loop into locked traffic. This mirrors what MSVC's own <atomic> does - a
// single __iso_volatile_load plus a barrier - except the barrier is
// std::atomic_thread_fence, which costs nothing on x86/x64, emits dmb on ARM64,
// and avoids the deprecated _ReadWriteBarrier that /WX would reject.
#define MANIFOLD_MSVC_ATOMIC(N, Int, IsoInt, Suffix, Bits)                  \
  template <>                                                               \
  struct MsvcAtomic<N> {                                                    \
    using Int_t = Int;                                                      \
    static Int Load(const volatile Int* p, std::memory_order order) {       \
      const Int value = static_cast<Int>(__iso_volatile_load##Bits(         \
          reinterpret_cast<const volatile IsoInt*>(p)));                    \
      if (order != std::memory_order_relaxed) PostLoadBarrier(order);       \
      return value;                                                         \
    }                                                                       \
    static void Store(volatile Int* p, Int v, std::memory_order order) {    \
      if (order != std::memory_order_relaxed &&                             \
          order != std::memory_order_release) {                             \
        _InterlockedExchange##Suffix(p, v);                                 \
        return;                                                             \
      }                                                                     \
      PostLoadBarrier(order);                                               \
      __iso_volatile_store##Bits(reinterpret_cast<volatile IsoInt*>(p), v); \
    }                                                                       \
    static Int Exchange(volatile Int* p, Int v) {                           \
      return _InterlockedExchange##Suffix(p, v);                            \
    }                                                                       \
    static Int Cas(volatile Int* p, Int expected, Int desired) {            \
      return _InterlockedCompareExchange##Suffix(p, desired, expected);     \
    }                                                                       \
    static Int Add(volatile Int* p, Int v) {                                \
      return _InterlockedExchangeAdd##Suffix(p, v);                         \
    }                                                                       \
  }

MANIFOLD_MSVC_ATOMIC(1, char, __int8, 8, 8);
MANIFOLD_MSVC_ATOMIC(2, short, __int16, 16, 16);
MANIFOLD_MSVC_ATOMIC(4, long, __int32, , 32);
MANIFOLD_MSVC_ATOMIC(8, __int64, __int64, 64, 64);
#undef MANIFOLD_MSVC_ATOMIC

// The MSVC backend: every operation goes through the integer of matching
// width, with the value memcpy'd in and out.
template <typename T>
struct AtomicBackend {
  using Int_t = typename MsvcAtomic<sizeof(T)>::Int_t;

  static Int_t ToInt(T value) {
    Int_t bits;
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
  }
  static T FromInt(Int_t bits) {
    T value;
    std::memcpy(&value, &bits, sizeof(T));
    return value;
  }
  // Unavoidable, and narrower than it looks: MSVC's intrinsics are declared
  // per integer width, so this is how you call them at all - even `int` needs
  // it, since _InterlockedExchange takes long*. MSVC's own <atomic> does the
  // same thing (_Atomic_address_as), and MSVC does no type-based alias
  // analysis. Unlike the cast this file removes, it reinterprets an address
  // for a builtin that works on raw memory rather than asserting that T is
  // laid out like something else.
  static volatile Int_t* AsInt(T* p) {
    return reinterpret_cast<volatile Int_t*>(p);
  }

  static T Load(T* p, std::memory_order order) {
    return FromInt(MsvcAtomic<sizeof(T)>::Load(AsInt(p), order));
  }
  static void Store(T* p, T desired, std::memory_order order) {
    MsvcAtomic<sizeof(T)>::Store(AsInt(p), ToInt(desired), order);
  }
  static T Exchange(T* p, T desired, std::memory_order) {
    return FromInt(MsvcAtomic<sizeof(T)>::Exchange(AsInt(p), ToInt(desired)));
  }
  // _InterlockedCompareExchange never fails spuriously, so it serves both the
  // weak and the strong form.
  static bool CompareExchange(T* p, T& expected, T desired, std::memory_order,
                              bool) {
    const T previous = FromInt(
        MsvcAtomic<sizeof(T)>::Cas(AsInt(p), ToInt(expected), ToInt(desired)));
    if (std::memcmp(&previous, &expected, sizeof(T)) == 0) return true;
    expected = previous;
    return false;
  }
  static T FetchAdd(T* p, T arg, std::memory_order) {
    return FromInt(MsvcAtomic<sizeof(T)>::Add(AsInt(p), ToInt(arg)));
  }
};

#endif  // 32-bit MSVC fallback

#else  // GCC/Clang backend

// The GCC/Clang backend: the `__atomic` builtins, which are documented to
// operate on ordinary objects and are what `std::atomic_ref` itself lowers to.
// Orders are switched onto literals rather than passed through as an int:
// libstdc++ can rely on its own enum matching __ATOMIC_*, but this header
// compiles against libstdc++ and libc++ both, so it does not assume that.
template <typename T>
struct AtomicBackend {
  static T Load(T* p, std::memory_order order) {
    T value;
    switch (order) {
      case std::memory_order_relaxed:
        __atomic_load(p, &value, __ATOMIC_RELAXED);
        break;
      case std::memory_order_consume:
      case std::memory_order_acquire:
        __atomic_load(p, &value, __ATOMIC_ACQUIRE);
        break;
      default:
        __atomic_load(p, &value, __ATOMIC_SEQ_CST);
        break;
    }
    return value;
  }
  static void Store(T* p, T desired, std::memory_order order) {
    switch (order) {
      case std::memory_order_relaxed:
        __atomic_store(p, &desired, __ATOMIC_RELAXED);
        break;
      case std::memory_order_release:
        __atomic_store(p, &desired, __ATOMIC_RELEASE);
        break;
      default:
        __atomic_store(p, &desired, __ATOMIC_SEQ_CST);
        break;
    }
  }
  static T Exchange(T* p, T desired, std::memory_order order) {
    T previous;
    switch (order) {
      case std::memory_order_relaxed:
        __atomic_exchange(p, &desired, &previous, __ATOMIC_RELAXED);
        break;
      case std::memory_order_acquire:
        __atomic_exchange(p, &desired, &previous, __ATOMIC_ACQUIRE);
        break;
      case std::memory_order_release:
        __atomic_exchange(p, &desired, &previous, __ATOMIC_RELEASE);
        break;
      case std::memory_order_acq_rel:
        __atomic_exchange(p, &desired, &previous, __ATOMIC_ACQ_REL);
        break;
      default:
        __atomic_exchange(p, &desired, &previous, __ATOMIC_SEQ_CST);
        break;
    }
    return previous;
  }
  // The failure order must not be stronger than success, and must not be a
  // release order, so it is derived rather than passed through.
  static bool CompareExchange(T* p, T& expected, T desired,
                              std::memory_order order, bool weak) {
    switch (order) {
      case std::memory_order_relaxed:
        return __atomic_compare_exchange(p, &expected, &desired, weak,
                                         __ATOMIC_RELAXED, __ATOMIC_RELAXED);
      case std::memory_order_acquire:
        return __atomic_compare_exchange(p, &expected, &desired, weak,
                                         __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE);
      case std::memory_order_release:
        return __atomic_compare_exchange(p, &expected, &desired, weak,
                                         __ATOMIC_RELEASE, __ATOMIC_RELAXED);
      case std::memory_order_acq_rel:
        return __atomic_compare_exchange(p, &expected, &desired, weak,
                                         __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
      default:
        return __atomic_compare_exchange(p, &expected, &desired, weak,
                                         __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
  }
  static T FetchAdd(T* p, T arg, std::memory_order order) {
    switch (order) {
      case std::memory_order_relaxed:
        return __atomic_fetch_add(p, arg, __ATOMIC_RELAXED);
      case std::memory_order_acquire:
        return __atomic_fetch_add(p, arg, __ATOMIC_ACQUIRE);
      case std::memory_order_release:
        return __atomic_fetch_add(p, arg, __ATOMIC_RELEASE);
      case std::memory_order_acq_rel:
        return __atomic_fetch_add(p, arg, __ATOMIC_ACQ_REL);
      default:
        return __atomic_fetch_add(p, arg, __ATOMIC_SEQ_CST);
    }
  }
};

#endif  // MSVC vs GCC/Clang backend

}  // namespace details

/**
 * A C++17 stand-in for std::atomic_ref, over one of the backends above. The
 * referenced object must outlive the AtomicRef, and while any AtomicRef refers
 * to it every access must go through one.
 *
 * Memory orders are honored exactly on GCC/Clang, matching what the C++20
 * alias does. On MSVC the read-modify-write intrinsics are all full barriers,
 * so a weaker request is satisfied by being stronger. `Load` is the exception:
 * a plain load plus a trailing fence, exact on x86/x64, and on ARM64 an
 * acquire load that serves as seq_cst only because the store side goes through
 * the full-barrier _InterlockedExchange. That describes the 64-bit MSVC
 * backend; the 32-bit fallback defers to std::atomic and is exact throughout.
 */
template <typename T>
class AtomicRef {
  // Scalars only, deliberately narrower than std::atomic_ref's trivially-
  // copyable. A padded type would need its padding cleared before every
  // compare_exchange (libstdc++ calls __clear_padding for exactly this), or
  // stale padding bytes make the exchange fail forever. Every use here is a
  // scalar, so the simpler contract is the honest one.
  static_assert(std::is_scalar<T>::value, "AtomicRef requires a scalar type");
  static_assert(sizeof(T) <= 8, "AtomicRef supports up to 8-byte scalars");
  // The same test libc++ and MSVC STL use to decide whether padding needs
  // clearing; scalars that pass it need none.
  static_assert(std::has_unique_object_representations_v<T> ||
                    std::is_floating_point_v<T>,
                "AtomicRef requires a type with no padding bits");

  using Backend = details::AtomicBackend<T>;

 public:
  explicit AtomicRef(T& obj) : ptr_(&obj) {
    // The atomic ops need the object naturally aligned, which alignof(T) does
    // not promise: 32-bit x86 gives alignof(double) == 4. So check the address,
    // as libstdc++ and Boost both do in their constructors. Compiles away
    // entirely without MANIFOLD_DEBUG.
    DEBUG_ASSERT(reinterpret_cast<uintptr_t>(ptr_) % sizeof(T) == 0, logicErr,
                 "AtomicRef requires an object aligned to its own size");
  }

  T load(std::memory_order order = std::memory_order_seq_cst) const {
    return Backend::Load(ptr_, order);
  }

  void store(T desired, std::memory_order order = std::memory_order_seq_cst) {
    Backend::Store(ptr_, desired, order);
  }

  T exchange(T desired, std::memory_order order = std::memory_order_seq_cst) {
    return Backend::Exchange(ptr_, desired, order);
  }

  // As in std::atomic_ref, `expected` is updated with the current value when
  // the exchange fails. Only the weak form may fail spuriously, so a caller
  // that treats failure as evidence about the stored value needs the strong
  // one.
  bool compare_exchange_weak(
      T& expected, T desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return Backend::CompareExchange(ptr_, expected, desired, order,
                                    /*weak=*/true);
  }

  bool compare_exchange_weak(T& expected, T desired, std::memory_order success,
                             std::memory_order) {
    // The failure order is derived from `success` in the backend, matching
    // libstdc++'s __cmpexch_failure_order2, so it is accepted and ignored.
    return Backend::CompareExchange(ptr_, expected, desired, success,
                                    /*weak=*/true);
  }

  bool compare_exchange_strong(
      T& expected, T desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return Backend::CompareExchange(ptr_, expected, desired, order,
                                    /*weak=*/false);
  }

  // Integral only: __atomic_fetch_add and _InterlockedExchangeAdd are both
  // defined for integers and pointers. Floating-point accumulation goes
  // through the compare_exchange_weak loop in AtomicAdd.
  template <typename U = T>
  std::enable_if_t<std::is_integral<U>::value, T> fetch_add(
      T arg, std::memory_order order = std::memory_order_seq_cst) {
    return Backend::FetchAdd(ptr_, arg, order);
  }

 private:
  T* ptr_;
};

#endif  // std::atomic_ref vs the C++17 stand-in

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
