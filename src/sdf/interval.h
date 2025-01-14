// Copyright 2024 The Manifold Authors.
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

#include <algorithm>
#include <cmath>
#include <limits>

#include "manifold/common.h"

namespace manifold::sdf {

// not really a precise implementation...
template <typename Domain>
struct Interval {
  Domain lower;
  Domain upper;

  static constexpr Domain zero = static_cast<Domain>(0);
  static constexpr Domain one = static_cast<Domain>(1);

  Interval()
      : lower(-std::numeric_limits<Domain>::infinity()),
        upper(std::numeric_limits<Domain>::infinity()) {}
  Interval(Domain v) : lower(v), upper(v) {}
  Interval(Domain lower, Domain upper) : lower(lower), upper(upper) {}
  static Interval constant(Domain v) { return {v, v}; }

  Interval operator+(const Interval &other) const {
    return {lower + other.lower, upper + other.upper};
  }

  Interval operator-() const { return {-upper, -lower}; }

  Interval operator-(const Interval &other) const { return *this + (-other); }

  Interval operator*(Domain d) const {
    if (d > zero) return {lower * d, upper * d};
    return {upper * d, lower * d};
  }

  Interval operator*(const Interval &other) const {
    if (is_const()) return other * lower;
    if (other.is_const()) return *this * other.lower;

    Domain a1b1 = lower * other.lower;
    Domain a2b2 = upper * other.upper;
    // we can write more "fast paths", but at some point it will become slower
    // than just going the general path...
    if (lower >= zero && other.lower >= zero)
      return {a1b1, a2b2};
    else if (upper <= zero && other.upper <= zero)
      return {a2b2, a1b1};

    Domain a1b2 = lower * other.upper;
    Domain a2b1 = upper * other.lower;
    return {std::min(std::min(a1b1, a1b2), std::min(a2b1, a2b2)),
            std::max(std::max(a1b1, a1b2), std::max(a2b1, a2b2))};
  }

  Interval operator/(const Interval &other) const {
    if (other.is_const()) return *this / other.lower;
    constexpr Domain infty = std::numeric_limits<Domain>::infinity();
    Interval reci;
    if (other.lower >= zero || other.upper <= zero) {
      reci.lower = other.upper == zero ? -infty : (one / other.upper);
      reci.upper = other.lower == zero ? infty : (one / other.lower);
    } else {
      reci.lower = -infty;
      reci.upper = infty;
    }
    return *this * reci;
  }

  Interval operator/(Domain d) const {
    if (d > zero) return {lower / d, upper / d};
    return {upper / d, lower / d};
  }

  constexpr bool is_const() const { return lower == upper; }

  Interval operator==(const Interval &other) const {
    if (is_const() && other.is_const() && lower == other.lower)
      return constant(one);  // must be equal
    if (lower > other.upper || upper < other.lower)
      return constant(zero);  // disjoint, cannot possibly be equal
    return {zero, one};
  }

  constexpr bool operator==(Domain d) const { return is_const() && lower == d; }

  Interval operator>(const Interval &other) const {
    if (lower > other.upper) return constant(one);
    if (upper < other.lower) return constant(zero);
    return {zero, one};
  }

  Interval operator<(const Interval &other) const {
    if (upper < other.lower) return constant(one);
    if (lower > other.upper) return constant(zero);
    return {zero, one};
  }

  Interval min(const Interval &other) const {
    return {std::min(lower, other.lower), std::min(upper, other.upper)};
  }

  Interval max(const Interval &other) const {
    return {std::max(lower, other.lower), std::max(upper, other.upper)};
  }

  Interval merge(const Interval &other) const {
    return {std::min(lower, other.lower), std::max(upper, other.upper)};
  }

  template <typename F>
  Interval monotone_map(F f) const {
    if (is_const()) return constant(f(lower));
    return {f(lower), f(upper)};
  }

  template <typename F>
  Interval antimonotone_map(F f) const {
    if (is_const()) return constant(f(lower));
    return {f(upper), f(lower)};
  }

  Interval abs() const {
    if (lower >= zero) return *this;
    if (upper <= zero) return {-upper, -lower};
    return {zero, std::max(-lower, upper)};
  }

  Interval mod(Domain m) const {
    // FIXME: cannot deal with negative m right now...
    Domain diff = std::fmod(lower, m);
    if (diff < zero) diff += m;
    Domain cycle_min = lower - diff;
    // may be disjoint intervals, but we don't deal with that...
    if (upper - cycle_min >= m) return {zero, m};
    return {diff, upper - cycle_min};
  }

  Interval logical_and(const Interval &other) const {
    return {lower == zero || other.lower == zero ? zero : one,
            upper == one && other.upper == one ? one : zero};
  }

  Interval logical_or(const Interval &other) const {
    return {lower == zero && other.lower == zero ? zero : one,
            upper == one || other.upper == one ? one : zero};
  }

  Interval sin() const {
    if (is_const()) return constant(std::sin(lower));
    // largely similar to cos
    int64_t min_pis = static_cast<int64_t>(std::floor(
        (lower - static_cast<Domain>(kHalfPi)) / static_cast<Domain>(kPi)));
    int64_t max_pis = static_cast<int64_t>(std::floor(
        (upper - static_cast<Domain>(kHalfPi)) / static_cast<Domain>(kPi)));

    bool not_cross_pos_1 =
        (min_pis % 2 == 0) ? max_pis - min_pis <= 1 : max_pis == min_pis;
    bool not_cross_neg_1 =
        (min_pis % 2 == 0) ? max_pis == min_pis : max_pis - min_pis <= 1;

    Domain new_min =
        not_cross_neg_1 ? std::min(std::sin(lower), std::sin(upper)) : -one;
    Domain new_max =
        not_cross_pos_1 ? std::max(std::sin(lower), std::sin(upper)) : one;
    return {new_min, new_max};
  }

  Interval cos() const {
    if (is_const()) return constant(std::cos(lower));
    int64_t min_pis =
        static_cast<int64_t>(std::floor(lower / static_cast<Domain>(kPi)));
    int64_t max_pis =
        static_cast<int64_t>(std::floor(upper / static_cast<Domain>(kPi)));

    bool not_cross_pos_1 =
        (min_pis % 2 == 0) ? max_pis - min_pis <= 1 : max_pis == min_pis;
    bool not_cross_neg_1 =
        (min_pis % 2 == 0) ? max_pis == min_pis : max_pis - min_pis <= 1;

    Domain new_min =
        not_cross_neg_1 ? std::min(std::cos(lower), std::cos(upper)) : -one;
    Domain new_max =
        not_cross_pos_1 ? std::max(std::cos(lower), std::cos(upper)) : one;
    return {new_min, new_max};
  }

  Interval tan() const {
    if (is_const()) return constant(std::tan(lower));
    int64_t min_pis = static_cast<int64_t>(std::floor(
        (lower + static_cast<Domain>(kHalfPi)) / static_cast<Domain>(kPi)));
    int64_t max_pis = static_cast<int64_t>(std::floor(
        (upper + static_cast<Domain>(kHalfPi)) / static_cast<Domain>(kPi)));
    if (min_pis != max_pis)
      return {-std::numeric_limits<Domain>::infinity(),
              std::numeric_limits<Domain>::infinity()};
    return monotone_map([](Domain x) { return std::tan(x); });
  }
};

}  // namespace manifold::sdf
