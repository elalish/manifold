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

#include "../src/atomic_compat.h"
#include "gtest/gtest.h"

using namespace manifold;

// Which backend this build selected is not otherwise visible. Report it so a
// CI log says whether the lane covered the std::atomic_ref path or a fallback.
TEST(Utils, AtomicRefBackend) {
#if defined(__cpp_lib_atomic_ref)
  const char* backend = "std::atomic_ref";
#else
  const char* backend = "C++17 cast";
#endif
  RecordProperty("backend", backend);
}

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

// std::atomic's layout and lock-freedom are asserted per width, and the cast
// relies on both, so exercise every width plus a non-integer type.
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

// fetch_add across the integer widths.
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

// std::atomic_ref's mutators are const - it is a handle, so constness of the
// handle says nothing about the referent. The C++17 stand-in has to match, or
// code written against one standard fails to build on the other.
TEST(Utils, AtomicRefMutatorsAreConst) {
  int value = 0;
  const AtomicRef<int> ref(value);
  ref.store(4);
  EXPECT_EQ(ref.load(), 4);
  EXPECT_EQ(ref.exchange(5), 4);
  int expected = 5;
  EXPECT_TRUE(ref.compare_exchange_strong(expected, 6));
  EXPECT_EQ(ref.fetch_add(1), 6);
  EXPECT_EQ(value, 7);
}
