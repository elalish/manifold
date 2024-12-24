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

Operand Context::addInstruction(OpCode op, Operand a, Operand b, Operand c) {
  switch (op) {
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
      if (a.id > b.id) std::swap(a, b);
      break;
    default:
      break;
  }
  // common subexpression elimination
  auto key = std::make_pair(op, std::make_tuple(a, b, c));
  auto entry = cache.find(key);
  if (entry != cache.end()) return entry->second;
  auto result = addInstructionNoCache(op, a, b, c);
  cache.insert({key, result});
  return result;
}

// bypass the cache because we don't expect to have more common subexpressions
// after optimizations
Operand Context::addInstructionNoCache(OpCode op, Operand a, Operand b,
                                       Operand c) {
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
  auto tryApply = [&](size_t i, Operand lhs, Operand rhs) {
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

// this does dead code elimination as well
// assumes the last instruction is return
// and note that this is not optimal, and cannot be optimal without dealing with
// NP-hard stuff...
void Context::reschedule() {
  DEBUG_ASSERT(!operations.empty() && operations.back() == OpCode::RETURN,
               logicErr, "return expected");
  cache.clear();
  auto oldOperations = std::move(operations);
  auto oldOperands = std::move(operands);
  opUses.clear();
  for (auto &uses : constantUses) uses.clear();

  std::unordered_map<size_t, Operand> computedInst;
  std::vector<size_t> stack;
  if (oldOperands.back()[0].isResult())
    stack.push_back(oldOperands.back()[0].toInstIndex());

  std::unordered_map<size_t, uint8_t> bitset;
  std::unordered_map<size_t, size_t> distances;
  std::vector<size_t> tmpStack;

  auto requiresComputation = [&computedInst](Operand operand) {
    return operand.isResult() &&
           computedInst.find(operand.toInstIndex()) == computedInst.end();
  };
  auto toNewOperand = [&computedInst](Operand old) {
    if (old.isResult()) return computedInst[old.toInstIndex()];
    return old;
  };

  while (!stack.empty()) {
    int numResults = 0;
    auto back = stack.back();
    auto &curOperands = oldOperands[back];
    for (auto operand : curOperands)
      if (requiresComputation(operand)) numResults += 1;
    if (numResults > 1) {
      // find common results first
      // does this by recursively marking instructions to be the transitive
      // dependency of operands
      // we use a bitset, so if the bitset & (1 << (numResults + 1)) - 1,
      // it means that the instruction is the common dependency for all operands
      uint8_t mask = (1 << (numResults + 1)) - 1;
      numResults = 0;
      for (auto operand : curOperands) {
        if (!requiresComputation(operand)) continue;
        tmpStack.push_back(operand.toInstIndex());
        while (!tmpStack.empty()) {
          auto current = tmpStack.back();
          tmpStack.pop_back();
          // already computed
          if (computedInst.find(current) != computedInst.end()) continue;
          auto iter = bitset.find(current);
          if (iter == bitset.end()) {
            // new dependency
            bitset.insert({current, 1 << numResults});
          } else {
            iter->second |= 1 << numResults;
          }
          for (auto x : oldOperands[current]) {
            if (!x.isResult()) continue;
            tmpStack.push_back(x.toInstIndex());
          }
        }
        numResults += 1;
      }
      // compute operand costs as distance in the dependency graph
      std::array<size_t, 3> costs = {0, 0, 0};
      std::array<size_t, 3> ids = {0, 1, 2};
      for (size_t i = 0; i < curOperands.size(); i++) {
        auto operand = curOperands[i];
        if (!requiresComputation(operand)) continue;
        tmpStack.push_back(operand.toInstIndex());
        while (!tmpStack.empty()) {
          auto current = tmpStack.back();
          size_t maxDistance = 0;
          for (auto x : oldOperands[current]) {
            if (!x.isResult()) continue;
            auto inst = x.toInstIndex();

            // computed, doesn't affect distance
            if (computedInst.find(inst) != computedInst.end()) continue;

            auto iter1 = bitset.find(inst);
            DEBUG_ASSERT(iter1 != bitset.end(), logicErr, "should be found");
            // shared dependency between operands, also doesn't affect distance
            if ((iter1->second & mask) == mask) continue;

            auto iter2 = distances.find(inst);
            if (iter2 == distances.end()) {
              // not computed
              tmpStack.push_back(x.toInstIndex());
              maxDistance = std::numeric_limits<size_t>::max();
            } else {
              maxDistance = std::max(maxDistance, iter2->second);
            }
          }
          if (maxDistance != std::numeric_limits<size_t>::max()) {
            tmpStack.pop_back();
            distances.insert({current, maxDistance + 1});
          }
        }
        costs[i] = distances[operand.toInstIndex()];
        distances.clear();
      }
      std::sort(ids.begin(), ids.end(),
                [&costs](size_t x, size_t y) { return costs[x] < costs[y]; });
      // expensive operands are placed at the top of the stack, i.e. scheduled
      // earlier
      for (size_t x : ids)
        if (requiresComputation(curOperands[x]))
          stack.push_back(curOperands[x].toInstIndex());

      bitset.clear();
    } else if (numResults == 1) {
      for (auto operand : curOperands)
        if (requiresComputation(operand))
          stack.push_back(operand.toInstIndex());
    } else {
      stack.pop_back();
      Operand result = addInstructionNoCache(
          oldOperations[back], toNewOperand(curOperands[0]),
          toNewOperand(curOperands[1]), toNewOperand(curOperands[2]));
      computedInst.insert({back, result});
    }
  }
  addInstructionNoCache(OpCode::RETURN,
                        computedInst[oldOperands.back()[0].toInstIndex()]);
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

  auto getReg = [&](Operand operand) {
    if (operand.isResult())
      return opToReg[operand.toInstIndex()];
    else if (operand.isConst())
      return constantToReg[operand.toConstIndex()];
    return static_cast<uint8_t>(-(operand.id + 1));
  };

  // FIXME: handle spills
  for (size_t i = 0; i < operations.size(); i++) {
    if (operations[i] == OpCode::NOP) {
      opToReg.push_back(0);
      continue;
    }
    if (operations[i] == OpCode::RETURN) {
      auto operand = operands[i][0];
      tape.push_back(static_cast<uint8_t>(operations[i]));
      tape.push_back(getReg(operand));
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
      regUsed[reg] = true;
    }
    opToReg.push_back(reg);
    tape.push_back(static_cast<uint8_t>(operations[i]));
    tape.push_back(reg);
    for (auto operand : operands[i]) {
      if (operand.isNone()) break;
      tape.push_back(getReg(operand));
    }
  }
  return std::make_pair(std::move(tape), std::move(buffer));
}

}  // namespace manifold::sdf
