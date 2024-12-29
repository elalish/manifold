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

#include "context.h"

namespace manifold::sdf {
struct AffineValue {
  // value = var * a + b
  Operand var;
  double a;
  double b;

  AffineValue(Operand var, double a, double b) : var(var), a(a), b(b) {}
  AffineValue(double constant) : var(Operand::none()), a(0.0), b(constant) {}
  bool operator==(const AffineValue &other) const {
    return var == other.var && a == other.a && b == other.b;
  }
  AffineValue operator+(double d) { return AffineValue(var, a, b + d); }
  AffineValue operator*(double d) { return AffineValue(var, a * d, b * d); }
};
}  // namespace manifold::sdf

template <>
struct std::hash<AffineValue> {
  size_t operator()(const AffineValue &value) const {
    size_t h = std::hash<int>()(value.var.id);
    hash_combine(h, value.a, value.b);
    return h;
  }
};
