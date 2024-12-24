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

#include <unordered_map>
#include <utility>
#include <vector>

#include "tape.h"

namespace manifold::sdf {

struct Operand {
  int id;

  static Operand none() { return {0}; }
  bool isConst() const { return id <= -4; }
  bool isResult() const { return id > 0; }
  bool isNone() const { return id == 0; }
  size_t toConstIndex() const { return static_cast<size_t>(-(id + 4)); }
  size_t toInstIndex() const { return static_cast<size_t>(id - 1); }
  bool operator==(const Operand& other) const { return id == other.id; }
  bool operator!=(const Operand& other) const { return id != other.id; }
};

class Context {
 public:
  Operand addConstant(double d);
  Operand addInstruction(OpCode op, Operand a = Operand::none(),
                         Operand b = Operand::none(),
                         Operand c = Operand::none());
  void optimizeFMA();
  // TODO: DCE

  std::pair<std::vector<uint8_t>, std::vector<double>> genTape();

  void dump() const;

 private:
  // constants have negative IDs, starting from -4
  // -1, -2 and -3 are reserved for x y z
  std::unordered_map<double, Operand> constantsIds;
  std::vector<double> constants;
  // constant use vector, elements are instruction indices
  // constant with ID -4 is mapped to 0, etc.
  std::vector<std::vector<size_t>> constantUses;
  // instructions, index 0 is mapped to ID 1, etc.
  std::vector<OpCode> operations;
  // instruction value use vector, elements are instruction indices
  std::vector<std::vector<size_t>> opUses;
  // operands, 0 is invalid (uses fewer operands)
  // +ve are instruction results
  // -ve are constants
  std::vector<std::array<Operand, 3>> operands;

  std::vector<uint8_t> tmpTape;
  std::vector<double> tmpBuffer;
};

}  // namespace manifold::sdf
