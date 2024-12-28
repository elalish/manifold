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

#include "value.h"

#include <chrono>

#include "../utils.h"
#include "context.h"
#include "tape.h"

namespace manifold::sdf {

struct ValueOperation {
  OpCode op;
  std::array<Value, 3> operands;

  ValueOperation(OpCode op, Value a, Value b, Value c)
      : op(op), operands({a, b, c}) {}
};

Value Value::Invalid() { return Value(ValueKind::INVALID, 0.0); }

Value Value::Constant(double d) { return Value(ValueKind::CONSTANT, d); }

Value Value::X() { return Value(ValueKind::X, 0.0); }

Value Value::Y() { return Value(ValueKind::Y, 0.0); }

Value Value::Z() { return Value(ValueKind::Z, 0.0); }

Value Value::operator+(const Value& other) const {
  return Value(ValueKind::OPERATION, std::make_shared<ValueOperation>(
                                         OpCode::ADD, *this, other, Invalid()));
}

Value Value::operator-(const Value& other) const {
  return Value(ValueKind::OPERATION, std::make_shared<ValueOperation>(
                                         OpCode::SUB, *this, other, Invalid()));
}

Value Value::operator*(const Value& other) const {
  return Value(ValueKind::OPERATION, std::make_shared<ValueOperation>(
                                         OpCode::MUL, *this, other, Invalid()));
}

Value Value::operator/(const Value& other) const {
  return Value(ValueKind::OPERATION, std::make_shared<ValueOperation>(
                                         OpCode::DIV, *this, other, Invalid()));
}

Value Value::cond(const Value& then, const Value& otherwise) const {
  return Value(
      ValueKind::OPERATION,
      std::make_shared<ValueOperation>(OpCode::CHOICE, *this, then, otherwise));
}

Value Value::mod(const Value& other) const {
  return Value(ValueKind::OPERATION, std::make_shared<ValueOperation>(
                                         OpCode::MOD, *this, other, Invalid()));
}

Value Value::min(const Value& other) const {
  return Value(ValueKind::OPERATION, std::make_shared<ValueOperation>(
                                         OpCode::MIN, *this, other, Invalid()));
}

Value Value::max(const Value& other) const {
  return Value(ValueKind::OPERATION, std::make_shared<ValueOperation>(
                                         OpCode::MAX, *this, other, Invalid()));
}

Value Value::operator==(const Value& other) const {
  return Value(ValueKind::OPERATION, std::make_shared<ValueOperation>(
                                         OpCode::EQ, *this, other, Invalid()));
}

Value Value::operator>(const Value& other) const {
  return Value(ValueKind::OPERATION, std::make_shared<ValueOperation>(
                                         OpCode::GT, *this, other, Invalid()));
}

Value Value::operator&&(const Value& other) const {
  return Value(ValueKind::OPERATION, std::make_shared<ValueOperation>(
                                         OpCode::MUL, *this, other, Invalid()));
}

Value Value::operator||(const Value& other) const {
  return Value(ValueKind::OPERATION, std::make_shared<ValueOperation>(
                                         OpCode::ADD, *this, other, Invalid()));
}

Value Value::abs() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::ABS, *this, Invalid(),
                                                Invalid()));
}

Value Value::operator-() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::NEG, *this, Invalid(),
                                                Invalid()));
}

Value Value::exp() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::EXP, *this, Invalid(),
                                                Invalid()));
}

Value Value::log() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::LOG, *this, Invalid(),
                                                Invalid()));
}

Value Value::sqrt() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::SQRT, *this, Invalid(),
                                                Invalid()));
}

Value Value::floor() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::FLOOR, *this, Invalid(),
                                                Invalid()));
}

Value Value::ceil() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::CEIL, *this, Invalid(),
                                                Invalid()));
}

Value Value::round() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::ROUND, *this, Invalid(),
                                                Invalid()));
}

Value Value::sin() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::SIN, *this, Invalid(),
                                                Invalid()));
}

Value Value::cos() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::COS, *this, Invalid(),
                                                Invalid()));
}

Value Value::tan() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::TAN, *this, Invalid(),
                                                Invalid()));
}

Value Value::asin() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::ASIN, *this, Invalid(),
                                                Invalid()));
}

Value Value::acos() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::ACOS, *this, Invalid(),
                                                Invalid()));
}

Value Value::atan() const {
  return Value(ValueKind::OPERATION,
               std::make_shared<ValueOperation>(OpCode::ATAN, *this, Invalid(),
                                                Invalid()));
}

Value::~Value() {
  using VO = std::shared_ptr<ValueOperation>;
  std::vector<VO> stack;
  auto push = [&stack](VO&& vo) {
    if (vo.use_count() == 1) stack.emplace_back(vo);
  };
  if (kind == ValueKind::OPERATION) push(std::get<VO>(std::move(v)));
  while (!stack.empty()) {
    auto back = std::move(stack.back());
    stack.pop_back();
    for (auto& value : back->operands) {
      if (value.kind == ValueKind::OPERATION)
        push(std::get<VO>(std::move(value.v)));
    }
  }
}

std::pair<std::vector<uint8_t>, size_t> Value::genTape() const {
  using VO = std::shared_ptr<ValueOperation>;
  Context ctx;
  unordered_map<ValueOperation*, Operand> cache;
  std::vector<ValueOperation*> stack;

  const auto none = Operand::none();
  bool ready = true;
  auto getOperand = [&](const Value& x, bool pushStack) {
    switch (x.kind) {
      case ValueKind::OPERATION: {
        auto iter = cache.find(std::get<VO>(x.v).get());
        if (iter != cache.end()) return iter->second;
        if (pushStack) {
          ready = false;
          stack.push_back(std::get<VO>(x.v).get());
        }
        return none;
      }
      case ValueKind::CONSTANT:
        return ctx.addConstant(std::get<double>(x.v));
      case ValueKind::X:
        return Operand{-1};
      case ValueKind::Y:
        return Operand{-2};
      case ValueKind::Z:
        return Operand{-3};
      default:
        return none;
    }
  };

  auto start = std::chrono::high_resolution_clock::now();
  if (kind == ValueKind::OPERATION) stack.push_back(std::get<VO>(v).get());

  int count = 0;
  while (!stack.empty()) {
    count++;
    ready = true;
    auto current = stack.back();
    Operand a = getOperand(current->operands[0], true);
    Operand b = getOperand(current->operands[1], true);
    Operand c = getOperand(current->operands[2], true);
    if (ready) {
      stack.pop_back();
      // check if inserted... can happen when evaluating with a DAG
      if (cache.find(current) != cache.end()) continue;
      cache.insert({current, ctx.addInstruction({current->op, {a, b, c}})});
    }
  }

  Operand result = getOperand(*this, false);
  ctx.addInstruction({OpCode::RETURN, {result, none, none}});
  auto end = std::chrono::high_resolution_clock::now();
  auto time = static_cast<int>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());
  printf("serialization: %dus with %d nodes\n", time, count);
  start = std::chrono::high_resolution_clock::now();
  ctx.optimize();
  end = std::chrono::high_resolution_clock::now();
  time = static_cast<int>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());
  printf("optimize: %dus\n", time);

  start = std::chrono::high_resolution_clock::now();
  auto tape = ctx.genTape();
  end = std::chrono::high_resolution_clock::now();
  time = static_cast<int>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());
  printf("codegen: %dus with length %ld\n", time, tape.first.size());

  return tape;
}

}  // namespace manifold::sdf
