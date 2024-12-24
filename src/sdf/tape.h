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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "interval.h"
#include "manifold/vec_view.h"

namespace manifold::sdf {

enum class OpCode : uint8_t {
  NOP,
  RETURN,
  // CONST,

  // unary operations
  ABS,
  NEG,
  EXP,
  LOG,
  SQRT,
  FLOOR,
  CEIL,
  ROUND,
  SIN,
  COS,
  TAN,
  ASIN,
  ACOS,
  ATAN,

  // binary operations,
  DIV,
  MOD,
  MIN,
  MAX,
  EQ,
  GT,
  AND,
  OR,

  // fast binary operations
  ADD,
  SUB,
  MUL,
  // ternary operations
  FMA,
  CHOICE,
};

template <typename Domain>
struct EvalContext {
  VecView<const uint8_t> tape;
  VecView<Domain> buffer;

  Domain& operand(uint32_t x) { return buffer[x]; }

  static Domain handle_unary(OpCode op, Domain operand);

  static Domain handle_binary(OpCode op, Domain lhs, Domain rhs);

  static Domain handle_choice(Domain cond, Domain lhs, Domain rhs);

  Domain eval() {
    size_t i = 0;
    while (1) {
      OpCode current = static_cast<OpCode>(tape[i]);
      // fast binary/ternary operations
      if (current >= OpCode::ADD) {
        // loop is needed to force the compiler to use a tight code layout
        do {
          size_t result = tape[i + 1];
          Domain lhs = operand(tape[i + 2]);
          Domain rhs = operand(tape[i + 3]);
          i += 4;
          if (current <= OpCode::MUL) {
            if (current == OpCode::ADD)
              operand(result) = lhs + rhs;
            else if (current == OpCode::SUB)
              operand(result) = lhs - rhs;
            else
              operand(result) = lhs * rhs;
          } else {
            Domain z = operand(tape[i++]);
            if (current == OpCode::FMA)
              operand(result) = lhs * rhs + z;
            else
              operand(result) = handle_choice(lhs, rhs, z);
          }
          current = static_cast<OpCode>(tape[i]);
        } while (current >= OpCode::ADD);
      }
      if (current >= OpCode::DIV) {
        Domain lhs = operand(tape[i + 2]);
        Domain rhs = operand(tape[i + 3]);
        operand(tape[i + 1]) = handle_binary(current, lhs, rhs);
        i += 4;
      } else if (current >= OpCode::ABS) {
        Domain x = operand(tape[i + 2]);
        operand(tape[i + 1]) = handle_unary(current, x);
        i += 3;
        // } else if (current == OpCode::CONST) {
        //   double x;
        //   std::memcpy(&x, tape.data() + i + 2, sizeof(x));
        //   operand(tape[i + 1]) = x;
        //   i += 2 + sizeof(x);
      } else if (current == OpCode::RETURN) {
        return operand(tape[i + 1]);
      } else {
        i += 1;
      }
    }
  }
};

template <>
inline double EvalContext<double>::handle_unary(OpCode op, double x) {
  switch (op) {
    case OpCode::ABS:
      return std::abs(x);
    case OpCode::NEG:
      return -x;
    case OpCode::EXP:
      return std::exp(x);
    case OpCode::LOG:
      return std::log(x);
    case OpCode::SQRT:
      return std::sqrt(x);
    case OpCode::FLOOR:
      return std::floor(x);
    case OpCode::CEIL:
      return std::ceil(x);
    case OpCode::ROUND:
      return std::round(x);
    case OpCode::SIN:
      return std::sin(x);
    case OpCode::COS:
      return std::cos(x);
    case OpCode::TAN:
      return std::tan(x);
    case OpCode::ASIN:
      return std::asin(x);
    case OpCode::ACOS:
      return std::acos(x);
    case OpCode::ATAN:
      return std::atan(x);
    default:
      return 0.0;
  }
}

template <>
inline double EvalContext<double>::handle_binary(OpCode op, double lhs,
                                                 double rhs) {
  switch (op) {
    case OpCode::DIV:
      return lhs / rhs;
    case OpCode::MOD:
      // FIXME: negative rhs???
      return std::fmod(std::fmod(lhs, rhs) + rhs, rhs);
    case OpCode::MIN:
      return std::min(lhs, rhs);
    case OpCode::MAX:
      return std::max(lhs, rhs);
    case OpCode::EQ:
      return lhs == rhs ? 1.0 : 0.0;
    case OpCode::GT:
      return lhs > rhs ? 1.0 : 0.0;
    case OpCode::AND:
      return (lhs == 1.0 && rhs == 1.0) ? 1.0 : 0.0;
    case OpCode::OR:
      return (lhs == 1.0 || rhs == 1.0) ? 1.0 : 0.0;
    default:
      return 0;
  }
}

template <>
inline double EvalContext<double>::handle_choice(double cond, double lhs,
                                                 double rhs) {
  if (cond == 1.0) return lhs;
  return rhs;
}

template <>
inline Interval<double> EvalContext<Interval<double>>::handle_unary(
    OpCode op, Interval<double> x) {
  constexpr double infty = std::numeric_limits<double>::infinity();
  switch (op) {
    case OpCode::ABS:
      return x.abs();
    case OpCode::NEG:
      return -x;
    case OpCode::EXP:
      return x.monotone_map([](double v) { return std::exp(v); });
    case OpCode::LOG:
      return x.monotone_map(
          [infty](double v) { return v > 0.0 ? std::log(v) : -infty; });
    case OpCode::SQRT:
      return x.monotone_map(
          [infty](double v) { return v >= 0.0 ? std::sqrt(v) : 0.0; });
    case OpCode::FLOOR:
      return x.monotone_map([](double v) { return std::floor(v); });
    case OpCode::CEIL:
      return x.monotone_map([](double v) { return std::ceil(v); });
    case OpCode::ROUND:
      return x.monotone_map([](double v) { return std::round(v); });
    case OpCode::SIN:
      return x.sin();
    case OpCode::COS:
      return x.cos();
    case OpCode::TAN:
      return x.tan();
    case OpCode::ASIN:
      return x.monotone_map([infty](double v) {
        return v < -1.0 ? -infty : v > 1.0 ? infty : std::asin(v);
      });
    case OpCode::ACOS:
      return x.antimonotone_map([infty](double v) {
        return v < -1.0 ? infty : v > 1.0 ? -infty : std::acos(v);
      });
    case OpCode::ATAN:
      return x.monotone_map([](double v) { return std::atan(v); });
    default:
      return {0.0, 0.0};
  }
}

template <>
inline Interval<double> EvalContext<Interval<double>>::handle_binary(
    OpCode op, Interval<double> lhs, Interval<double> rhs) {
  switch (op) {
    case OpCode::DIV:
      return lhs / rhs;
    case OpCode::MOD:
      return lhs.is_const()
                 ? lhs.mod(rhs.lower)
                 : (rhs.lower < 0
                        ? Interval<double>{rhs.lower, std::max(0.0, rhs.upper)}
                        : Interval<double>{0, rhs.upper});
    case OpCode::MIN:
      return lhs.min(rhs);
    case OpCode::MAX:
      return lhs.max(rhs);
    case OpCode::EQ:
      return lhs == rhs;
    case OpCode::GT:
      return lhs > rhs;
    case OpCode::AND:
      return lhs.logical_and(rhs);
    case OpCode::OR:
      return lhs.logical_or(rhs);
    default:
      return {0.0, 0.0};
  }
}

template <>
inline Interval<double> EvalContext<Interval<double>>::handle_choice(
    Interval<double> cond, Interval<double> lhs, Interval<double> rhs) {
  if (cond.is_const()) {
    if (cond.lower == 1.0) return lhs;
    return rhs;
  }
  return lhs.merge(rhs);
}

}  // namespace manifold::sdf
