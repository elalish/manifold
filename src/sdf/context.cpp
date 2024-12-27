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

#ifdef MANIFOLD_DEBUG
#include <iostream>
#endif

#include "manifold/optional_assert.h"

namespace manifold::sdf {
void Context::dump() const {
#ifdef MANIFOLD_DEBUG
  for (size_t i = 0; i < instructions.size(); i++) {
    std::cout << i << " ";
    std::cout << " " << dumpOpCode(instructions[i].op) << " ";
    for (Operand operand : instructions[i].operands) {
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
#endif
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

Operand Context::addInstruction(Instruction inst) {
  switch (inst.op) {
    case OpCode::ADD:
    case OpCode::MUL:
    case OpCode::MIN:
    case OpCode::MAX:
    case OpCode::EQ:
    case OpCode::AND:
    case OpCode::OR:
    case OpCode::FMA:
      // first two operands commutative, sort them
      // makes it more likely to find common subexpressions
      if (inst.operands[0].id > inst.operands[1].id)
        std::swap(inst.operands[0], inst.operands[1]);
      break;
    default:
      break;
  }
  // common subexpression elimination
  auto entry = cache.find(inst);
  if (entry != cache.end()) return entry->second;
  auto result = addInstructionNoCache(inst);
  cache.insert({inst, result});
  return result;
}

// bypass the cache because we don't expect to have more common subexpressions
// after optimizations
Operand Context::addInstructionNoCache(Instruction inst) {
  // constant choice
  auto op = inst.op;
  auto &operands = inst.operands;
  if (op == OpCode::CHOICE && operands[0].isConst()) {
    if (constants[operands[0].toConstIndex()] == 1.0) return operands[1];
    return operands[2];
  }
  // constant propagation
  bool all_constants = true;
  for (auto operand : operands) {
    if (!operand.isConst() && !operand.isNone()) all_constants = false;
  }
  // we should not do anything about returning a constant...
  if (all_constants && op != OpCode::RETURN) {
    double result = 0.0;
    switch (op) {
      case OpCode::NOP:
      case OpCode::RETURN:
      case OpCode::CONST:
      case OpCode::STORE:
      case OpCode::LOAD:
        break;
      case OpCode::ABS:
      case OpCode::NEG:
      case OpCode::EXP:
      case OpCode::LOG:
      case OpCode::SQRT:
      case OpCode::FLOOR:
      case OpCode::CEIL:
      case OpCode::ROUND:
      case OpCode::SIN:
      case OpCode::COS:
      case OpCode::TAN:
      case OpCode::ASIN:
      case OpCode::ACOS:
      case OpCode::ATAN:
        result = EvalContext<double>::handle_unary(
            op, constants[operands[0].toConstIndex()]);
        break;
      case OpCode::DIV:
      case OpCode::MOD:
      case OpCode::MIN:
      case OpCode::MAX:
      case OpCode::EQ:
      case OpCode::GT:
      case OpCode::AND:
      case OpCode::OR:
        result = EvalContext<double>::handle_binary(
            op, constants[operands[0].toConstIndex()],
            constants[operands[1].toConstIndex()]);
        break;
      case OpCode::ADD:
        result = constants[operands[0].toConstIndex()] +
                 constants[operands[1].toConstIndex()];
        break;
      case OpCode::SUB:
        result = constants[operands[0].toConstIndex()] -
                 constants[operands[1].toConstIndex()];
        break;
      case OpCode::MUL:
        result = constants[operands[0].toConstIndex()] *
                 constants[operands[1].toConstIndex()];
        break;
      case OpCode::FMA:
        result = constants[operands[0].toConstIndex()] *
                     constants[operands[1].toConstIndex()] +
                 constants[operands[2].toConstIndex()];
        break;
      case OpCode::CHOICE:
        // should be unreachable
        DEBUG_ASSERT(false, logicErr, "unreachable");
        break;
    }
    return addConstant(result);
  }

  size_t i = instructions.size();
  instructions.push_back({op, operands});
  opUses.emplace_back();
  // update uses
  for (auto operand : operands) {
    auto target = getUses(operand);
    if (target == nullptr) continue;
    // avoid duplicates
    if (target->empty() || target->back() != i) target->push_back(i);
  }
  return {static_cast<int>(i) + 1};
}

Context::UsesVector::const_iterator findUse(const Context::UsesVector &uses,
                                            size_t inst) {
  return std::lower_bound(uses.cbegin(), uses.cend(), inst);
}

void Context::peephole() {
  auto tryApply = [&](size_t i, Operand lhs, Operand rhs) {
    if (!lhs.isResult()) return false;
    auto lhsInst = lhs.toInstIndex();
    if (instructions[lhsInst].op != OpCode::MUL || opUses[lhsInst].size() != 1)
      return false;
    Operand a = instructions[lhsInst].operands[0];
    Operand b = instructions[lhsInst].operands[1];
    instructions[i] = {OpCode::FMA, {a, b, rhs}};
    // remove instruction
    auto none = Operand::none();
    instructions[lhsInst] = {OpCode::NOP, {none, none, none}};
    // update uses, note that we need to maintain the order of the indices
    opUses[lhsInst].clear();
    auto updateUses = [&](Operand x) {
      if (!x.isResult() && !x.isConst()) return;
      auto uses = getUses(x);
      auto iter1 = findUse(*uses, lhsInst);
      DEBUG_ASSERT(*iter1 == lhsInst, logicErr, "expected use");
      uses->erase(iter1);
      auto iter2 = findUse(*uses, i);
      // make sure there is no duplicate
      if (iter2 == uses->cend() || *iter2 != i) uses->insert(iter2, i);
    };
    updateUses(a);
    if (a != b) updateUses(b);
    return true;
  };
  for (size_t i = 0; i < instructions.size(); i++) {
    auto &inst = instructions[i];
    if (inst.op == OpCode::ADD) {
      // check if lhs/rhs comes from MUL with no other uses
      auto lhs = inst.operands[0];
      auto rhs = inst.operands[1];
      if (!tryApply(i, lhs, rhs)) tryApply(i, rhs, lhs);
    }
  }
}

struct RegEntry {
  size_t nextUse;
  Operand operand;
  uint8_t reg;

  inline bool operator<(const RegEntry &other) const {
    return nextUse > other.nextUse ||
           (nextUse == other.nextUse && operand.id < other.operand.id);
  }
};

template <typename T>
void addImmediate(std::vector<uint8_t> &tape, T imm) {
  std::array<uint8_t, sizeof(T)> tmpBuffer;
  std::memcpy(tmpBuffer.data(), &imm, sizeof(T));
  for (auto byte : tmpBuffer) tape.push_back(byte);
}

template <typename V>
typename V::value_type pop_back(V &v) {
  auto x = v.back();
  v.pop_back();
  return x;
}

std::pair<std::vector<uint8_t>, size_t> Context::genTape() {
  std::vector<uint8_t> tape;
  size_t bufferSize = 3;
  unordered_map<Operand, uint32_t> spills;
  std::vector<uint32_t> spillSlots;

  std::vector<uint8_t> availableReg;
  // register cache, ordered by next use of the variable
  // when spilling is needed, we will evict the variable where the next use is
  // the furthest
  std::vector<RegEntry> regCache;
  auto insertRegCache = [&](RegEntry entry) {
    regCache.insert(std::lower_bound(regCache.cbegin(), regCache.cend(), entry),
                    entry);
  };

  auto allocateReg = [&]() {
    if (!availableReg.empty()) return pop_back(availableReg);
    // used too many registers, need to spill something
    // note: tested with a limit of 7, spills correctly
    if (bufferSize > 255) {
      auto reg = regCache.front().reg;
      // we can just discard constants, so only spill instruction results
      if (regCache.front().operand.isResult()) {
        uint32_t slot =
            spillSlots.empty() ? bufferSize++ : pop_back(spillSlots);
        spills.insert({regCache.front().operand, slot});
        tape.push_back(static_cast<uint8_t>(OpCode::STORE));
        addImmediate(tape, slot);
        tape.push_back(reg);
      }
      regCache.erase(regCache.begin());
      return reg;
    }
    return static_cast<uint8_t>(bufferSize++);
  };
  auto handleOperands = [&](std::array<Operand, 3> instOperands, size_t inst) {
    auto getOperandReg = [&](Operand operand, size_t inst) {
      // will not be used, so we can return whatever we like
      if (operand.isNone()) return static_cast<uint8_t>(0);
      // special xyz variables with fixed register
      if (!operand.isConst() && !operand.isResult())
        return static_cast<uint8_t>(-(operand.id + 1));
      // the operand, if present, must be at the end of the cache, due to how
      // the cache is ordered
      for (auto it = regCache.rbegin();
           it != regCache.rend() && it->nextUse == inst; ++it)
        if (it->operand == operand) return it->reg;
      // if not found, either a spill or a constant
      auto reg = allocateReg();
      // we will never spill constants
      auto iter = operand.isResult() ? spills.find(operand) : spills.end();
      if (iter == spills.end()) {
        DEBUG_ASSERT(operand.isConst(), logicErr,
                     "can only materialize constants");
        tape.insert(tape.end(), {static_cast<uint8_t>(OpCode::CONST), reg});
        addImmediate(tape, constants[operand.toConstIndex()]);
      } else {
        tape.insert(tape.end(), {static_cast<uint8_t>(OpCode::LOAD), reg});
        addImmediate(tape, iter->second);
        spillSlots.push_back(iter->second);
        spills.erase(iter);
      }
      insertRegCache({inst, operand, reg});
      return reg;
    };
    std::array<uint8_t, 3> regs;
    // note that we have to get the registers first, because we cannot spill the
    // first register and reuse it in the second for example
    for (size_t i : {0, 1, 2}) regs[i] = getOperandReg(instOperands[i], inst);
    // update register cache
    for (size_t i : {0, 1, 2}) {
      if (!instOperands[i].isConst() && !instOperands[i].isResult()) continue;
      bool erased = false;
      for (auto it = regCache.rbegin();
           it != regCache.rend() && it->nextUse == inst; ++it) {
        if (it->operand == instOperands[i]) {
          regCache.erase(std::next(it).base());
          erased = true;
          break;
        }
      }
      // if not found at the end of the cache, this means that it is handled by
      // another operand
      if (!erased) continue;
      auto uses = getUses(instOperands[i]);
      if (uses->back() == inst) {
        // end of lifetime, free register
        availableReg.push_back(regs[i]);
      } else {
        // insert it back with new next use
        // because it is not at the end of its lifetime, the incremented
        // iterator is guaranteed to be valid
        insertRegCache({*(findUse(*uses, inst) + 1), instOperands[i], regs[i]});
      }
    }
    return regs;
  };

  for (size_t i = 0; i < instructions.size(); i++) {
    auto &inst = instructions[i];
    if (inst.op == OpCode::NOP) continue;
    auto instOp = Operand{static_cast<int>(i) + 1};
    auto uses = getUses(instOp);
    // avoid useless ops
    if (inst.op != OpCode::RETURN && uses->empty()) continue;
    auto tmp = handleOperands(inst.operands, i);
    if (inst.op == OpCode::RETURN) {
      tape.insert(tape.end(), {static_cast<uint8_t>(inst.op), tmp[0]});
      break;
    }
    // note that we may spill the operand register, but that is fine
    uint8_t reg = allocateReg();
    insertRegCache({uses->front(), instOp, reg});
    tape.insert(tape.end(), {static_cast<uint8_t>(inst.op), reg});
    for (size_t j : {0, 1, 2}) {
      if (inst.operands[j].isNone()) break;
      tape.push_back(tmp[j]);
    }
  }
  return std::make_pair(std::move(tape), bufferSize);
}

}  // namespace manifold::sdf
