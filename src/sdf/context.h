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

#include "small_vector.h"
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
  bool operator<(const Operand& other) const { return id < other.id; }
};
}  // namespace manifold::sdf

using namespace manifold::sdf;

inline void hash_combine(std::size_t& seed) {}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, rest...);
}

template <>
struct std::hash<Operand> {
  size_t operator()(const Operand& operand) const {
    return std::hash<int>()(operand.id);
  }
};

template <>
struct std::hash<std::pair<OpCode, std::tuple<Operand, Operand, Operand>>> {
  size_t operator()(
      const std::pair<OpCode, std::tuple<Operand, Operand, Operand>>& pair)
      const {
    size_t h = std::hash<uint8_t>()(static_cast<uint8_t>(pair.first));
    hash_combine(h, std::get<0>(pair.second), std::get<1>(pair.second),
                 std::get<2>(pair.second));
    return h;
  }
};

namespace manifold::sdf {
class Context {
 public:
  Operand addConstant(double d);
  Operand addInstruction(OpCode op, Operand a = Operand::none(),
                         Operand b = Operand::none(),
                         Operand c = Operand::none());
  void optimizeFMA();
  void reschedule();

  std::pair<std::vector<uint8_t>, size_t> genTape();

  void dump() const;

 private:
  // constants have negative IDs, starting from -4
  // -1, -2 and -3 are reserved for x y z
  std::unordered_map<double, Operand> constantsIds;
  std::vector<double> constants;
  // constant use vector, elements are instruction indices
  // constant with ID -4 is mapped to 0, etc.
  std::vector<small_vector<size_t, 4>> constantUses;
  // instructions, index 0 is mapped to ID 1, etc.
  std::vector<OpCode> operations;
  // instruction value use vector, elements are instruction indices
  std::vector<small_vector<size_t, 4>> opUses;
  // operands, 0 is invalid (uses fewer operands)
  // +ve are instruction results
  // -ve are constants
  std::vector<std::array<Operand, 3>> operands;

  std::vector<uint8_t> tmpTape;
  std::vector<double> tmpBuffer;
  std::unordered_map<std::pair<OpCode, std::tuple<Operand, Operand, Operand>>,
                     Operand>
      cache;

  Operand addInstructionNoCache(OpCode op, Operand a = Operand::none(),
                                Operand b = Operand::none(),
                                Operand c = Operand::none());

  small_vector<size_t, 4>* getUses(Operand operand) {
    if (operand.isResult()) {
      return &opUses[operand.toInstIndex()];
    } else if (operand.isConst()) {
      return &constantUses[operand.toConstIndex()];
    } else {
      return static_cast<small_vector<size_t, 4>*>(nullptr);
    }
  };
};

}  // namespace manifold::sdf
