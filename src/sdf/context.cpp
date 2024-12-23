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

#include "context.h"

#include <algorithm>
#include <iostream>

#include "manifold/optional_assert.h"

namespace manifold::sdf {

void dumpOpCode(OpCode op) {
  switch (op) {
    case OpCode::NOP:
      std::cout << "NOP";
      break;
    case OpCode::RETURN:
      std::cout << "RETURN";
      break;
    case OpCode::ABS:
      std::cout << "ABS";
      break;
    case OpCode::NEG:
      std::cout << "NEG";
      break;
    case OpCode::EXP:
      std::cout << "EXP";
      break;
    case OpCode::LOG:
      std::cout << "LOG";
      break;
    case OpCode::SQRT:
      std::cout << "SQRT";
      break;
    case OpCode::FLOOR:
      std::cout << "FLOOR";
      break;
    case OpCode::CEIL:
      std::cout << "CEIL";
      break;
    case OpCode::ROUND:
      std::cout << "ROUND";
      break;
    case OpCode::SIN:
      std::cout << "SIN";
      break;
    case OpCode::COS:
      std::cout << "COS";
      break;
    case OpCode::TAN:
      std::cout << "TAN";
      break;
    case OpCode::ASIN:
      std::cout << "ASIN";
      break;
    case OpCode::ACOS:
      std::cout << "ACOS";
      break;
    case OpCode::ATAN:
      std::cout << "ATAN";
      break;
    case OpCode::DIV:
      std::cout << "DIV";
      break;
    case OpCode::MOD:
      std::cout << "MOD";
      break;
    case OpCode::MIN:
      std::cout << "MIN";
      break;
    case OpCode::MAX:
      std::cout << "MAX";
      break;
    case OpCode::EQ:
      std::cout << "EQ";
      break;
    case OpCode::GT:
      std::cout << "GT";
      break;
    case OpCode::AND:
      std::cout << "AND";
      break;
    case OpCode::OR:
      std::cout << "OR";
      break;
    case OpCode::ADD:
      std::cout << "ADD";
      break;
    case OpCode::SUB:
      std::cout << "SUB";
      break;
    case OpCode::MUL:
      std::cout << "MUL";
      break;
    case OpCode::FMA:
      std::cout << "FMA";
      break;
    case OpCode::CHOICE:
      std::cout << "CHOICE";
      break;
  }
}

void Context::dump() const {
  for (size_t i = 0; i < operations.size(); i++) {
    std::cout << i << " ";
    dumpOpCode(operations[i]);
    std::cout << " ";
    for (Operand operand : operands[i]) {
      if (operand.isNone()) break;
      if (operand.isResult())
        std::cout << "r" << operand.toInstIndex();
      else if (operand.isConst())
        std::cout << constants[operand.toConstIndex()];
      else
        std::cout << static_cast<char>('X' - operand.id - 1);
      std::cout << " ";
    }
    std::cout << "{";
    for (size_t use : opUses[i]) std::cout << use << " ";
    std::cout << "}" << std::endl;
  }
  std::cout << "-----------" << std::endl;
}

Operand Context::addConstant(double d) {
  auto result = constantsIds.insert(
      {d, Operand{-4 - static_cast<int>(constants.size())}});
  if (result.second) {
    constants.push_back(d);
    constantUses.emplace_back();
  }
  return result.first->second;
}

// TODO: hashconsing
Operand Context::addInstruction(OpCode op, Operand a, Operand b, Operand c) {
  // constant choice
  if (op == OpCode::CHOICE && a.isConst()) {
    if (constants[a.toConstIndex()] == 1.0) return b;
    return c;
  }
  // constant propagation
  bool all_constants = true;
  for (auto operand : {a, b, c}) {
    if (operand.isNone()) break;
    if (!operand.isConst()) {
      all_constants = false;
      break;
    }
  }
  if (all_constants) {
    tmpTape.clear();
    tmpBuffer.clear();
    tmpTape.push_back(static_cast<uint8_t>(op));
    tmpTape.push_back(0);
    tmpBuffer.push_back(0.0);
    for (Operand x : {a, b, c}) {
      if (!x.isConst()) break;
      tmpTape.push_back(tmpBuffer.size());
      tmpBuffer.push_back(constants[x.toConstIndex()]);
    }
    tmpTape.push_back(static_cast<uint8_t>(OpCode::RETURN));
    tmpTape.push_back(0);
    return addConstant(EvalContext<double>{
        tmpTape, VecView(tmpBuffer.data(), tmpBuffer.size())}
                           .eval());
  }

  size_t i = operations.size();
  operations.push_back(op);
  operands.push_back({a, b, c});
  opUses.emplace_back();
  // update uses
  for (auto operand : {a, b, c}) {
    std::vector<size_t> *target;
    if (operand.isResult()) {
      target = &opUses[operand.toInstIndex()];
    } else if (operand.isConst()) {
      target = &constantUses[operand.toConstIndex()];
    } else {
      continue;
    }
    // avoid duplicates
    if (target->empty() || target->back() != i) target->push_back(i);
  }
  return {static_cast<int>(i) + 1};
}

void Context::optimizeFMA() {
  auto tryApply = [&](int i, Operand lhs, Operand rhs) {
    if (!lhs.isResult()) return false;
    auto lhsInst = lhs.toInstIndex();
    if (operations[lhsInst] != OpCode::MUL || opUses[lhsInst].size() != 1)
      return false;
    operations[i] = OpCode::FMA;
    Operand a = operands[lhsInst][0];
    Operand b = operands[lhsInst][1];
    operands[i][0] = a;
    operands[i][1] = b;
    operands[i][2] = rhs;
    // remove instruction
    operations[lhsInst] = OpCode::NOP;
    operands[lhsInst][0] = Operand::none();
    operands[lhsInst][1] = Operand::none();
    // update uses, note that we need to maintain the order of the indices
    opUses[lhsInst].clear();
    auto updateUses = [&](Operand x) {
      if (!x.isResult() && !x.isConst()) return;
      auto &uses = x.isResult() ? opUses[x.toInstIndex()]
                                : constantUses[x.toConstIndex()];
      auto iter1 = std::lower_bound(uses.begin(), uses.end(), lhsInst);
      DEBUG_ASSERT(*iter1 == lhsInst, logicErr, "expected use");
      uses.erase(iter1);
      auto iter2 = std::lower_bound(uses.begin(), uses.end(), i);
      // make sure there is no duplicate
      if (iter2 == uses.end() || *iter2 != i + 1) uses.insert(iter2, i);
    };
    updateUses(a);
    if (a != b) updateUses(b);
    return true;
  };
  for (size_t i = 0; i < operations.size(); i++) {
    if (operations[i] == OpCode::ADD) {
      // check if lhs/rhs comes from MUL with no other uses
      auto lhs = operands[i][0];
      auto rhs = operands[i][1];
      if (!tryApply(i, lhs, rhs)) tryApply(i, rhs, lhs);
    }
  }
}

std::pair<std::vector<uint8_t>, std::vector<double>> Context::genTape() {
  std::vector<uint8_t> tape;
  std::vector<double> buffer;
  std::vector<uint8_t> constantToReg;
  for (int i : {0, 1, 2})  // x, y, z
    buffer.push_back(0.0);
  // handle constants by putting them inside the buffer/register
  // they are different because they require static lifetime, and cannot be
  // changed in an execution
  // FIXME: this is just temporary, we should optimize by encoding some
  // constants with a minimal number of uses into the read-only code when there
  // is a register pressure (more than 255...)
  for (size_t i = 0; i < constants.size(); i++) {
    constantToReg.push_back(0);
    if (constantUses[i].empty()) continue;
    constantToReg.back() = static_cast<uint8_t>(buffer.size());
    buffer.push_back(constants[i]);
  }

  std::vector<bool> regUsed(buffer.size(), true);
  std::vector<uint8_t> opToReg;
  std::vector<uint8_t> availableReg;

  // FIXME: handle spills
  for (size_t i = 0; i < operations.size(); i++) {
    if (operations[i] == OpCode::NOP) {
      opToReg.push_back(0);
      continue;
    }
    if (operations[i] == OpCode::RETURN) {
      auto operand = operands[i][0];
      tape.push_back(static_cast<uint8_t>(operations[i]));
      if (operand.isResult())
        tape.push_back(opToReg[operand.toInstIndex()]);
      else
        tape.push_back(constantToReg[operand.toConstIndex()]);
      dumpOpCode(operations[i]);
      std::cout << " r" << static_cast<int>(tape.back()) << std::endl;
      break;
    }
    // free up operand registers if possible
    for (auto operand : operands[i]) {
      if (!operand.isResult()) continue;
      auto operandInst = operand.toInstIndex();
      // not the last instruction, cannot free it up
      if (opUses[operandInst].back() != i) continue;
      uint8_t reg = opToReg[operandInst];
      // already freed, probably due to identical arguments
      if (!regUsed[reg]) continue;
      regUsed[reg] = false;
      availableReg.push_back(reg);
    }
    // allocate register
    uint8_t reg;
    if (availableReg.empty()) {
      // GG if we used too many registers, need spilling
      if (buffer.size() == 255) {
        // just return some nonsense that will not crash
        return {{static_cast<uint8_t>(OpCode::RETURN), 0}, {0.0}};
      }
      reg = buffer.size();
      buffer.push_back(0.0);
      regUsed.push_back(true);
    } else {
      reg = availableReg.back();
      availableReg.pop_back();
    }
    opToReg.push_back(reg);
    tape.push_back(static_cast<uint8_t>(operations[i]));
    dumpOpCode(operations[i]);
    std::cout << " r" << static_cast<int>(reg);
    tape.push_back(reg);
    for (auto operand : operands[i]) {
      if (operand.isNone()) break;
      if (operand.isResult())
        tape.push_back(opToReg[operand.toInstIndex()]);
      else
        tape.push_back(constantToReg[operand.toConstIndex()]);
      std::cout << " r" << static_cast<int>(tape.back());
    }
    std::cout << std::endl;
  }
  std::cout << "-----------" << std::endl;
  return std::make_pair(std::move(tape), std::move(buffer));
}

}  // namespace manifold::sdf
