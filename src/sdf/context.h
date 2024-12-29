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

#include <optional>
#include <utility>
#include <vector>

#include "../utils.h"
#include "small_vector.h"
#include "tape.h"

namespace manifold::sdf {

// operands, 0 is invalid (uses fewer operands)
// +ve are instruction results
// -ve are constants
struct Operand {
  int id;

  static Operand none() { return {0}; }
  static Operand fromInstIndex(size_t i) { return {static_cast<int>(i) + 1}; }
  static Operand fromConstIndex(size_t i) { return {-static_cast<int>(i) - 4}; }
  bool isConst() const { return id <= -4; }
  bool isResult() const { return id > 0; }
  bool isNone() const { return id == 0; }
  size_t toConstIndex() const { return static_cast<size_t>(-(id + 4)); }
  size_t toInstIndex() const { return static_cast<size_t>(id - 1); }
  bool operator==(const Operand& other) const { return id == other.id; }
  bool operator!=(const Operand& other) const { return id != other.id; }
  bool operator<(const Operand& other) const { return id < other.id; }
};

struct Instruction {
  OpCode op;
  std::array<Operand, 3> operands;
  bool operator==(const Instruction& other) const {
    if (op != other.op) return false;
    return operands[0] == other.operands[0] &&
           operands[1] == other.operands[1] && operands[2] == other.operands[2];
  }
};
}  // namespace manifold::sdf

using namespace manifold::sdf;

inline void hash_combine(std::size_t& seed) {}

// note: ankerl hash combine function is too costly
template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, rest...);
}

template <>
struct std::hash<Operand> {
  size_t operator()(const Operand& operand) const { return operand.id; }
};

template <>
struct std::hash<Instruction> {
  size_t operator()(const Instruction& inst) const {
    size_t h = static_cast<size_t>(inst.op);
    hash_combine(h, inst.operands[0], inst.operands[1], inst.operands[2]);
    return h;
  }
};

namespace manifold::sdf {
class Context {
 public:
  using UsesVector = std::vector<size_t>;

  Operand addConstant(double d);
  Operand addInstruction(Instruction);
  void optimize();
  void reschedule();

  std::pair<std::vector<uint8_t>, size_t> genTape();

  void dump() const;

 private:
  // constants have negative IDs, starting from -4
  // -1, -2 and -3 are reserved for x y z
  unordered_map<double, Operand> constantsIds;
  std::vector<double> constants;
  // constant use vector, elements are instruction indices
  // constant with ID -4 is mapped to 0, etc.
  std::vector<UsesVector> constantUses;
  // instructions, index 0 is mapped to ID 1, etc.
  std::vector<Instruction> instructions;
  // instruction value use vector, elements are instruction indices
  std::vector<UsesVector> opUses;
  unordered_map<Instruction, Operand> cache;

  // std::optional<Operand> trySimplify(Instruction);
  Operand addInstructionNoCache(Instruction);
  void combineFMA();
  void optimizeAffine();
  void addUse(Operand operand, size_t inst);
  void removeUse(Operand operand, size_t inst);
  void schedule();

  UsesVector* getUses(Operand operand) {
    if (operand.isResult()) {
      return &opUses[operand.toInstIndex()];
    } else if (operand.isConst()) {
      return &constantUses[operand.toConstIndex()];
    } else {
      return static_cast<UsesVector*>(nullptr);
    }
  };
};

}  // namespace manifold::sdf
