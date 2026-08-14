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

#include "../src/utils.h"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include "../src/atomic_compat.h"
#include "gtest/gtest.h"

using namespace manifold;

namespace {
#ifndef __EMSCRIPTEN__
// Enough contention to catch a lost update, without making the suite slow.
constexpr int kThreads = 8;
constexpr int kPerThread = 20000;
// HashTableD's empty-slot marker.
constexpr uint64_t kOpen = ~uint64_t{0};

// Run `body(t)` on kThreads threads and join them all. Native-only: the WASM
// build is single-threaded, so std::thread construction throws and aborts
// under -fno-exceptions, as in cross_section_test's concurrency tests.
template <typename F>
void RunConcurrently(F body) {
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) threads.emplace_back(body, t);
  for (auto& thread : threads) thread.join();
}
#endif  // __EMSCRIPTEN__
}  // namespace

// Which backend this build selected is not otherwise visible. Report it so a
// CI log says whether the lane covered the std::atomic_ref path or a fallback.
TEST(Utils, AtomicRefBackend) {
#if defined(__cpp_lib_atomic_ref)
  const char* backend = "std::atomic_ref";
#elif defined(_MSC_VER) && !defined(__clang__) && \
    (defined(_M_IX86) || defined(_M_ARM))
  const char* backend = "std::atomic reinterpret_cast (32-bit MSVC)";
#elif defined(_MSC_VER) && !defined(__clang__)
  const char* backend = "_Interlocked* intrinsics";
#else
  const char* backend = "__atomic builtins";
#endif
  RecordProperty("backend", backend);
  std::cout << "AtomicRef backend: " << backend << std::endl;
}

#ifndef __EMSCRIPTEN__
// The whole point of the class: the target is a plain int, not a std::atomic,
// and concurrent increments must still not lose an update.
TEST(Utils, AtomicRefFetchAddLosesNothing) {
  int counter = 0;
  RunConcurrently([&counter](int) {
    for (int i = 0; i < kPerThread; ++i) AtomicRef<int>(counter).fetch_add(1);
  });
  EXPECT_EQ(counter, kThreads * kPerThread);
}

// AtomicAdd's generic overload accumulates through a compare_exchange_weak
// loop, because neither __atomic_fetch_add nor _InterlockedExchangeAdd take
// floating point. Sum 1.0 so the result is exactly representable.
TEST(Utils, AtomicAddAccumulatesDoubles) {
  double total = 0.0;
  RunConcurrently([&total](int) {
    for (int i = 0; i < kPerThread; ++i) AtomicAdd(total, 1.0);
  });
  EXPECT_EQ(total, static_cast<double>(kThreads * kPerThread));
}

// The pattern in AssignNormals and DedupePropVerts: whoever exchanges in the
// 1 first is the single owner, and everyone else must observe it.
TEST(Utils, AtomicRefExchangeHasExactlyOneWinner) {
  uint8_t flag = 0;
  std::atomic<int> winners{0};
  RunConcurrently([&flag, &winners](int) {
    if (AtomicRef<uint8_t>(flag).exchange(static_cast<uint8_t>(1)) == 0) {
      winners.fetch_add(1);
    }
  });
  EXPECT_EQ(winners.load(), 1);
  EXPECT_EQ(flag, 1u);
}

// HashTableD::Insert treats a returned kOpen as proof that it claimed the
// slot, so the CAS must be the strong form. A weak CAS may fail spuriously
// and report the unchanged value, which would let two threads both believe
// they own the same key. Note this cannot actually catch the weak form on
// x86, where lock cmpxchg never fails spuriously - the guarantee is
// architectural, and this pins the intent rather than policing it.
TEST(Utils, AtomicRefCompareExchangeStrongHasExactlyOneWinner) {
  uint64_t slot = kOpen;
  std::atomic<int> claims{0};
  RunConcurrently([&slot, &claims](int t) {
    uint64_t expected = kOpen;
    if (AtomicRef<uint64_t>(slot).compare_exchange_strong(
            expected, static_cast<uint64_t>(t))) {
      claims.fetch_add(1);
    } else {
      // On failure `expected` must carry the value that beat us, never kOpen.
      EXPECT_NE(expected, kOpen);
    }
  });
  EXPECT_EQ(claims.load(), 1);
  EXPECT_NE(slot, kOpen);
}

#endif  // __EMSCRIPTEN__

// A failed compare_exchange reports the current value through `expected`,
// which is what the AtomicAdd retry loop relies on.
TEST(Utils, AtomicRefCompareExchangeReportsCurrentValue) {
  int value = 7;
  int expected = 3;
  EXPECT_FALSE(AtomicRef<int>(value).compare_exchange_strong(expected, 9));
  EXPECT_EQ(expected, 7);
  EXPECT_EQ(value, 7);

  EXPECT_TRUE(AtomicRef<int>(value).compare_exchange_strong(expected, 9));
  EXPECT_EQ(value, 9);
}

// The MSVC backend is specialized per object width with a different intrinsic
// family each time, and those lanes are the only place it compiles, so
// exercise every width plus a non-integer type.
template <typename T>
void ExpectRoundTrip(T a, T b) {
  SCOPED_TRACE(testing::Message() << "sizeof(T) = " << sizeof(T));
  T value = a;
  AtomicRef<T> ref(value);
  EXPECT_EQ(ref.load(), a);

  ref.store(b, std::memory_order_relaxed);
  EXPECT_EQ(ref.load(std::memory_order_relaxed), b);
  ref.store(a, std::memory_order_release);
  EXPECT_EQ(ref.load(std::memory_order_acquire), a);

  EXPECT_EQ(ref.exchange(b), a);
  EXPECT_EQ(value, b);

  T expected = a;  // wrong on purpose: the exchange must fail and report b
  EXPECT_FALSE(ref.compare_exchange_strong(expected, a));
  EXPECT_EQ(expected, b);
  EXPECT_TRUE(ref.compare_exchange_strong(expected, a));
  EXPECT_EQ(value, a);
}

TEST(Utils, AtomicRefRoundTripsEveryWidth) {
  ExpectRoundTrip<uint8_t>(0x12, 0xf0);
  ExpectRoundTrip<uint16_t>(0x1234, 0xf00d);
  ExpectRoundTrip<uint32_t>(0x12345678u, 0xf00dbeefu);
  ExpectRoundTrip<uint64_t>(0x1234567890abcdefull, 0xf00dbeefcafef00dull);
  ExpectRoundTrip<int>(-7, 9);
  ExpectRoundTrip<double>(-2.5, 1e300);
}

// fetch_add across the integer widths, which on MSVC is a fourth intrinsic
// family again.
TEST(Utils, AtomicRefFetchAddEveryWidth) {
  uint8_t u8 = 250;
  EXPECT_EQ(AtomicRef<uint8_t>(u8).fetch_add(3), 250);
  EXPECT_EQ(u8, 253);

  uint16_t u16 = 65000;
  EXPECT_EQ(AtomicRef<uint16_t>(u16).fetch_add(500), 65000);
  EXPECT_EQ(u16, 65500);

  int32_t i32 = -5;
  EXPECT_EQ(AtomicRef<int32_t>(i32).fetch_add(12), -5);
  EXPECT_EQ(i32, 7);

  uint64_t u64 = 1ull << 40;
  EXPECT_EQ(AtomicRef<uint64_t>(u64).fetch_add(1), 1ull << 40);
  EXPECT_EQ(u64, (1ull << 40) + 1);
}

// Every memory order the call sites use has to round-trip, including the
// relaxed store in CalculateNormals/SortVerts and the acquire load in
// HashTableD's probes.
TEST(Utils, AtomicRefLoadStoreRoundTrip) {
  int value = 0;
  AtomicRef<int> ref(value);

  ref.store(1, std::memory_order_relaxed);
  EXPECT_EQ(ref.load(std::memory_order_relaxed), 1);

  ref.store(2, std::memory_order_release);
  EXPECT_EQ(ref.load(std::memory_order_acquire), 2);

  ref.store(3);
  EXPECT_EQ(ref.load(), 3);
  EXPECT_EQ(value, 3);

  // What HashTableD's const probes do: cast away const at the call site.
  const int& constValue = value;
  EXPECT_EQ(AtomicRef<int>(const_cast<int&>(constValue))
                .load(std::memory_order_acquire),
            3);
}
