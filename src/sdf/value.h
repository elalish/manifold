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

#include <memory>
#include <variant>
#include <vector>

namespace manifold::sdf {

enum class ValueKind { CONSTANT, X, Y, Z, OPERATION, INVALID };

struct ValueOperation;

class Value {
 public:
  static Value Invalid();
  static Value Constant(double d);
  static Value X();
  static Value Y();
  static Value Z();

  Value operator+(const Value& other) const;
  Value operator-(const Value& other) const;
  Value operator*(const Value& other) const;
  Value operator/(const Value& other) const;
  Value cond(const Value& then, const Value& otherwise) const;
  Value mod(const Value& m) const;
  Value min(const Value& other) const;
  Value max(const Value& other) const;

  // TODO: should we have a boolean value type?
  Value operator==(const Value& other) const;
  Value operator>(const Value& other) const;
  Value operator&&(const Value& other) const;
  Value operator||(const Value& other) const;

  Value abs() const;
  Value operator-() const;
  Value exp() const;
  Value log() const;
  Value sqrt() const;
  Value floor() const;
  Value ceil() const;
  Value round() const;
  Value sin() const;
  Value cos() const;
  Value tan() const;
  Value asin() const;
  Value acos() const;
  Value atan() const;

  // internal use only
  std::pair<std::vector<uint8_t>, std::vector<double>> genTape() const;

 private:
  ValueKind kind = ValueKind::INVALID;
  std::variant<double, std::shared_ptr<ValueOperation>> v;

  Value(ValueKind kind, std::variant<double, std::shared_ptr<ValueOperation>> v)
      : kind(kind), v(v) {}
};
}  // namespace manifold::sdf
