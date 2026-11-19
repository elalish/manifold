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

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>

#include "../src/atomic_compat.h"
#include "../src/collider.h"
#include "../src/shared.h"
#include "../src/tree2d.h"
#include "gtest/gtest.h"

using namespace manifold;

TEST(Utils, TwoDTreeSkipsDisabledPoints) {
  const std::vector<Rect> queries = {
      Rect(vec2(-10.0), vec2(10.0)),   Rect(vec2(-3, -4), vec2(4, 5)),
      Rect(vec2(0, -10), vec2(0, 10)), Rect(vec2(-10, 0), vec2(10, 0)),
      Rect(vec2(0.0), vec2(0.0)),      Rect(vec2(20.0), vec2(30.0)),
  };
  for (const int count : {0, 1, 8, 9, 17, 64, 257}) {
    SCOPED_TRACE(count);
    Vec<PolyVert> points;
    std::vector<vec2> original;
    for (int i = 0; i < count; ++i) {
      const vec2 position((i * 11) % 13 - 6, (i * 7) % 17 - 8);
      points.push_back({position, i});
      original.push_back(position);
    }
    BuildTwoDTree(points);
    for (const int phase : {0, 1, 2}) {
      SCOPED_TRACE(phase);
      // Disable internal pivots as well as leaves without changing the split
      // coordinates or rebuilding the tree.
      for (auto& point : points) {
        if (phase == 2 || (phase == 1 && point.idx % 3 == 0)) point.idx = -1;
      }
      for (const Rect& query : queries) {
        std::vector<int> expected;
        for (int i = 0; i < count; ++i) {
          const vec2 position = original[i];
          if (phase != 2 && !(phase == 1 && i % 3 == 0) &&
              position.x >= query.min.x && position.x <= query.max.x &&
              position.y >= query.min.y && position.y <= query.max.y)
            expected.push_back(i);
        }
        std::vector<int> actual;
        QueryTwoDTree(points, query,
                      [&](PolyVert point) { actual.push_back(point.idx); });
        std::sort(actual.begin(), actual.end());
        EXPECT_EQ(actual, expected);
      }
    }
  }
}

TEST(Utils, SegmentQueryPrunesOffRayLeaves) {
  // The diagonal crosses 16 cells, but its AABB covers all 256. Count leaf
  // candidates rather than timing the traversal so this stays deterministic.
  const Box gridBounds(vec3(0.0), vec3(16, 16, 1));
  std::vector<std::pair<uint32_t, Box>> cells;
  for (int x = 0; x < 16; ++x) {
    for (int y = 0; y < 16; ++y) {
      const vec3 center(x + 0.5, y + 0.5, 0.5);
      cells.emplace_back(Collider::MortonCode(center, gridBounds),
                         Box(center - 0.25, center + 0.25));
    }
  }
  std::sort(cells.begin(), cells.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  Vec<Box> boxes(cells.size());
  Vec<uint32_t> morton(cells.size());
  for (size_t i = 0; i < cells.size(); ++i) {
    morton[i] = cells[i].first;
    boxes[i] = cells[i].second;
  }
  const Collider collider(boxes, morton);
  const vec3 a(-1, -1, 0.5), b(17, 17, 0.5);
  const Box bounds(a, b);
  std::vector<int> candidates;
  auto record = [&](int, int leaf) { candidates.push_back(leaf); };
  auto recorder = MakeSimpleRecorder(record);
  auto boxQuery = [&](int) { return bounds; };
  collider.Collisions(recorder, boxQuery, 1, false);
  EXPECT_EQ(candidates.size(), 256);

  for (const bool reverse : {false, true}) {
    candidates.clear();
    const collider_internal::SegmentQuery segment{bounds, reverse ? b : a,
                                                  reverse ? a - b : b - a, 0.0};
    auto query = [&](int) { return segment; };
    collider.Collisions(recorder, query, 1, false);
    ASSERT_EQ(candidates.size(), 16);
    for (const int leaf : candidates)
      EXPECT_DOUBLE_EQ(boxes[leaf].min.x, boxes[leaf].min.y);
  }
}

TEST(Utils, SegmentQueryClosedSlabs) {
  // A flat face and a point are valid BVH boxes. Include the endpoint, and
  // handle a direction parallel to a slab without dividing by zero.
  const Box face(vec3(0, 0, 0), vec3(0, 1, 1));
  const Box point(vec3(0.0), vec3(0.0));
  const vec3 a(-1, -1, 0), b(0.0);
  const collider_internal::SegmentQuery endpoint{Box(a, b), a, b - a, 0.0};
  EXPECT_TRUE(endpoint.DoesOverlap(face));
  EXPECT_TRUE(endpoint.DoesOverlap(point));

  const vec3 c(-1, -1, 0), d(1, 1, 0);
  const collider_internal::SegmentQuery crossing{Box(c, d), c, d - c, 0.0};
  EXPECT_TRUE(crossing.DoesOverlap(face));
  EXPECT_TRUE(crossing.DoesOverlap(point));
  const Box offRay(vec3(-0.5, 0.5, 0), vec3(-0.25, 0.75, 0));
  EXPECT_FALSE(crossing.DoesOverlap(offRay));
}

TEST(Utils, SharedVecCopyConstructionAndDetach) {
  const int values[] = {2, 3, 5};
  const SharedVec<int> source(VecView<const int>(values, 3));
  SharedVec<int> copy(source);
  EXPECT_EQ(copy.data(), source.data());
  copy.MakeUnique();
  EXPECT_NE(copy.data(), source.data());
  copy.data()[0] = 7;
  *std::next(copy.begin()) = 11;
  copy.push_back(13);
  EXPECT_EQ(source.size(), 3);
  EXPECT_EQ(source[0], 2);
  EXPECT_EQ(source[1], 3);
  EXPECT_EQ(source[2], 5);
  EXPECT_EQ(copy[0], 7);
  EXPECT_EQ(copy[1], 11);
  EXPECT_EQ(copy[3], 13);

  Vec<int> valueCopy(source);
  EXPECT_NE(valueCopy.data(), source.data());
  valueCopy[2] = 17;
  EXPECT_EQ(source[2], 5);

  SharedVec<int> sharedForMove(source);
  Vec<int> moved(std::move(sharedForMove));
  EXPECT_NE(moved.data(), source.data());
  moved[0] = 19;
  EXPECT_EQ(source[0], 2);

  SharedVec<int> sharedForAssignment(source);
  Vec<int> assigned;
  assigned = std::move(sharedForAssignment);
  EXPECT_NE(assigned.data(), source.data());
  assigned[1] = 29;
  EXPECT_EQ(source[1], 3);

  SharedVec<int> empty;
  SharedVec<int> emptyCopy(empty);
  emptyCopy.MakeUnique();
  emptyCopy.push_back(31);
  EXPECT_TRUE(empty.empty());

  SharedVec<int> sharedMove(std::move(copy));
  SharedVec<int> movedFromCopy(copy);
  EXPECT_TRUE(movedFromCopy.empty());
  movedFromCopy.push_back(37);
  EXPECT_EQ(sharedMove[0], 7);
}

TEST(Utils, SharedVecCopyAssignmentFromMovedFrom) {
  const int values[] = {2, 3, 5};
  SharedVec<int> source(VecView<const int>(values, 3));
  const SharedVec<int> owner(std::move(source));
  const int* const originalData = owner.data();

  const auto assignAndReuse = [&](SharedVec<int> destination) {
    EXPECT_EQ(&(destination = source), &destination);
    EXPECT_TRUE(destination.empty());
    EXPECT_EQ(destination.capacity(), 0);
    EXPECT_EQ(destination.data(), nullptr);
    destination.MakeUnique();
    destination.push_back(7);

    SharedVec<int> copy(destination);
    copy.MakeUnique();
    copy[0] = 11;
    EXPECT_EQ(destination.size(), 1);
    EXPECT_EQ(destination[0], 7);
    EXPECT_EQ(copy[0], 11);
    EXPECT_TRUE(source.empty());
    EXPECT_EQ(owner.data(), originalData);
    ASSERT_EQ(owner.size(), 3);
    for (size_t i = 0; i < owner.size(); ++i) EXPECT_EQ(owner[i], values[i]);
  };

  {
    SCOPED_TRACE("empty destination");
    assignAndReuse(SharedVec<int>());
  }
  {
    SCOPED_TRACE("unique nonempty destination");
    assignAndReuse(SharedVec<int>(2, 17));
  }
  {
    SCOPED_TRACE("shared nonempty destination");
    assignAndReuse(owner);
  }

  source = owner;
  EXPECT_EQ(source.data(), owner.data());
  source.MakeUnique();
  source[0] = 13;
  EXPECT_EQ(owner[0], 2);
}

#ifndef __EMSCRIPTEN__
TEST(Utils, SharedVecConcurrentCopyAndDetach) {
  const SharedVec<int> source(128, 23);
  std::atomic<bool> intact = true;
  std::vector<std::thread> threads;
  for (int thread = 0; thread < 8; ++thread) {
    threads.emplace_back([&source, &intact, thread] {
      for (int iteration = 0; iteration < 100; ++iteration) {
        SharedVec<int> copy(source);
        if (copy.data() != source.data()) intact = false;
        copy.MakeUnique();
        std::fill(copy.begin(), copy.end(), thread);
        for (int value : source) {
          if (value != 23) intact = false;
        }
      }
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_TRUE(intact);
}

#endif  // __EMSCRIPTEN__

TEST(Utils, CosdLargeAngles) {
  for (const double angle : {1e18, -1e18, 1e20, -1e20, 1e100, -1e100,
                             std::numeric_limits<double>::max(),
                             -std::numeric_limits<double>::max()}) {
    SCOPED_TRACE(testing::Message() << "angle = " << angle);
    const double reduced = std::remainder(angle, 360.0);
    EXPECT_NEAR(cosd(angle), std::cos(radians(reduced)), 1e-15);
    EXPECT_NEAR(sind(angle) * sind(angle) + cosd(angle) * cosd(angle), 1.0,
                1e-15);
  }
}

TEST(Utils, CosdQuadrantBoundaries) {
  const double expected[] = {1.0, 0.0, -1.0, 0.0};
  for (int quarter = -8; quarter <= 8; ++quarter) {
    EXPECT_DOUBLE_EQ(cosd(quarter * 90.0), expected[(quarter + 8) % 4]);
  }
  for (const double angle : {std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_TRUE(std::isnan(cosd(angle)));
  }
}

TEST(Utils, BarycentricAllWeightsSnappedToZero) {
  // Both points are within tolerance of every edge, but not any vertex.
  // Scaling both the triangle and tolerance must preserve the equal fallback.
  for (const double scale : {1e-50, 1e-12, 1.0, 1e12, 1e50}) {
    SCOPED_TRACE(testing::Message() << "scale = " << scale);
    const mat3 tri(vec3(0, 0, 0), vec3(2, 0, 0), vec3(1, std::sqrt(3.0), 0));
    for (const bool reversed : {false, true}) {
      SCOPED_TRACE(testing::Message() << "reversed = " << reversed);
      const mat3 scaled =
          scale * (reversed ? mat3(tri[0], tri[2], tri[1]) : tri);
      for (const vec3 inputWeights :
           {vec3(1.0 / 3.0), vec3(0.30, 0.34, 0.36)}) {
        SCOPED_TRACE(testing::Message()
                     << "weights = " << inputWeights.x << ", " << inputWeights.y
                     << ", " << inputWeights.z);
        const vec3 point = inputWeights.x * scaled[0] +
                           inputWeights.y * scaled[1] +
                           inputWeights.z * scaled[2];
        const double tolerance = inputWeights.x == inputWeights.y ? 0.6 : 0.65;
        const vec3 weights = GetBarycentric(point, scaled, tolerance * scale);
        for (int i : {0, 1, 2}) {
          EXPECT_TRUE(std::isfinite(weights[i]));
          EXPECT_DOUBLE_EQ(weights[i], 1.0 / 3.0);
        }
        EXPECT_NEAR(weights.x + weights.y + weights.z, 1.0, 1e-12);
      }
    }
  }
}

TEST(Utils, BarycentricRawWeightsUnderflowToZero) {
  const double scale = 68 * 1e-83;
  const mat3 tri(vec3(0, 0, 0), vec3(2 * scale, 0, 0),
                 vec3(scale, std::sqrt(3.0) * scale, 0));
  const vec3 point = (tri[0] + tri[1] + tri[2]) / 3.0;
  const double tolerance = 0.6 * scale;

  // The triangle passes the area check, but all three raw weights underflow
  // to zero. Normalizing those raw weights would still divide by zero.
  const mat3 edges(tri[2] - tri[1], tri[0] - tri[2], tri[1] - tri[0]);
  const vec3 crossP = la::cross(edges[0], edges[1]);
  const double area2 = la::dot(crossP, crossP);
  double longestSide2 = 0;
  for (const int i : {0, 1, 2}) {
    longestSide2 = std::max(longestSide2, la::dot(edges[i], edges[i]));
    const vec3 crossPv = la::cross(edges[i], point - tri[Next3(i)]);
    ASSERT_DOUBLE_EQ(la::dot(crossPv, crossP), 0.0);
  }
  ASSERT_GT(area2, longestSide2 * (tolerance * tolerance));

  const vec3 weights = GetBarycentric(point, tri, tolerance);
  for (const int i : {0, 1, 2}) {
    EXPECT_TRUE(std::isfinite(weights[i]));
    EXPECT_DOUBLE_EQ(weights[i], 1.0 / 3.0);
  }
  EXPECT_NEAR(weights.x + weights.y + weights.z, 1.0, 1e-12);
}

TEST(Utils, BarycentricRetainsSnapping) {
  const mat3 tri(vec3(0, 0, 0), vec3(2, 0, 0), vec3(1, std::sqrt(3.0), 0));
  const vec3 vertex = GetBarycentric(tri[0], tri, 1e-8);
  EXPECT_DOUBLE_EQ(vertex.x, 1.0);
  EXPECT_DOUBLE_EQ(vertex.y, 0.0);
  EXPECT_DOUBLE_EQ(vertex.z, 0.0);

  const vec3 edge = GetBarycentric(vec3(0.8, 1e-9, 0), tri, 1e-8);
  EXPECT_NEAR(edge.x, 0.6, 1e-9);
  EXPECT_NEAR(edge.y, 0.4, 1e-9);
  EXPECT_DOUBLE_EQ(edge.z, 0.0);

  const vec3 interior =
      GetBarycentric(0.25 * tri[0] + 0.25 * tri[1] + 0.5 * tri[2], tri, 1e-8);
  EXPECT_NEAR(interior.x, 0.25, 1e-12);
  EXPECT_NEAR(interior.y, 0.25, 1e-12);
  EXPECT_NEAR(interior.z, 0.5, 1e-12);
}

TEST(Utils, BarycentricNearDegenerateTriangle) {
  const mat3 tri(vec3(0, 0, 0), vec3(2, 0, 0), vec3(1, 1e-8, 0));
  const vec3 weights = GetBarycentric(vec3(0.5, 0, 0), tri, 1e-7);
  EXPECT_DOUBLE_EQ(weights.x, 0.75);
  EXPECT_DOUBLE_EQ(weights.y, 0.25);
  EXPECT_DOUBLE_EQ(weights.z, 0.0);
}

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
