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

#include <unordered_map>

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
                                         OpCode::AND, *this, other, Invalid()));
}

Value Value::operator||(const Value& other) const {
  return Value(ValueKind::OPERATION, std::make_shared<ValueOperation>(
                                         OpCode::OR, *this, other, Invalid()));
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

std::pair<std::vector<uint8_t>, std::vector<double>> Value::genTape() const {
  using VO = std::shared_ptr<ValueOperation>;
  Context ctx;
  std::unordered_map<VO, Operand> cache;
  std::vector<VO> stack;
  if (kind == ValueKind::OPERATION) stack.push_back(std::get<VO>(v));

  auto getOperand = [&](Value x, std::function<void(Value)> f) {
    switch (x.kind) {
      case ValueKind::OPERATION: {
        auto iter = cache.find(std::get<VO>(x.v));
        if (iter != cache.end()) return iter->second;
        // stack.push_back(std::get<VO>(x.v));
        // ready = false;
        f(x);
        return Operand::none();
      }
      case ValueKind::CONSTANT:
        return ctx.addConstant(std::get<double>(x.v));
      case ValueKind::X:
        return Operand{-1};
      case ValueKind::Y:
        return Operand{-2};
      case ValueKind::Z:
        return Operand{-3};
      case ValueKind::INVALID:
        return Operand::none();
    }
  };
  while (!stack.empty()) {
    bool ready = true;
    VO current = stack.back();
    auto f = [&](Value x) {
      stack.push_back(std::get<VO>(x.v));
      ready = false;
    };
    Operand a = getOperand(current->operands[0], f);
    Operand b = getOperand(current->operands[1], f);
    Operand c = getOperand(current->operands[2], f);
    if (ready) {
      stack.pop_back();
      // check if inserted... can happen when evaluating with a DAG
      if (cache.find(current) != cache.end()) continue;
      cache.insert({current, ctx.addInstruction(current->op, a, b, c)});
    }
  }

  Operand result = getOperand(*this, [](Value _) {});
  ctx.addInstruction(OpCode::RETURN, result, Operand::none(), Operand::none());

  ctx.dump();
  ctx.optimizeFMA();
  ctx.dump();
  return ctx.genTape();
}

}  // namespace manifold::sdf
