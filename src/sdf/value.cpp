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

Value Value::cond(const Value& then, const Value& otherwise) const {
  return Value(
      ValueKind::OPERATION,
      std::make_shared<ValueOperation>(OpCode::CHOICE, *this, then, otherwise));
}

#define MAKE_UNARY(NAME, OPCODE)                                          \
  Value Value::NAME() const {                                             \
    return Value(ValueKind::OPERATION,                                    \
                 std::make_shared<ValueOperation>(OpCode::OPCODE, *this,  \
                                                  Invalid(), Invalid())); \
  }
#define MAKE_BINARY(NAME, OPCODE)                                        \
  Value Value::NAME(const Value& other) const {                          \
    return Value(ValueKind::OPERATION,                                   \
                 std::make_shared<ValueOperation>(OpCode::OPCODE, *this, \
                                                  other, Invalid()));    \
  }

MAKE_UNARY(abs, ABS)
MAKE_UNARY(operator-, NEG)
MAKE_UNARY(exp, EXP)
MAKE_UNARY(log, LOG)
MAKE_UNARY(sqrt, SQRT)
MAKE_UNARY(floor, FLOOR)
MAKE_UNARY(ceil, CEIL)
MAKE_UNARY(round, ROUND)
MAKE_UNARY(sin, SIN)
MAKE_UNARY(cos, COS)
MAKE_UNARY(tan, TAN)
MAKE_UNARY(asin, ASIN)
MAKE_UNARY(acos, ACOS)
MAKE_UNARY(atan, ATAN)

MAKE_BINARY(operator+, ADD)
MAKE_BINARY(operator-, SUB)
MAKE_BINARY(operator*, MUL)
MAKE_BINARY(operator/, DIV)
MAKE_BINARY(mod, MOD)
MAKE_BINARY(min, MIN)
MAKE_BINARY(max, MAX)
MAKE_BINARY(operator==, EQ)
MAKE_BINARY(operator>, GT)
MAKE_BINARY(operator&&, MUL)
MAKE_BINARY(operator||, ADD)

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
