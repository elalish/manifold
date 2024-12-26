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
  for (size_t i = 0; i < operations.size(); i++) {
    std::cout << i << " ";
    std::cout << " " << dumpOpCode(operations[i]) << " ";
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

Operand Context::addInstruction(OpCode op, std::array<Operand, 3> operands) {
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
      if (operands[0].id > operands[1].id) std::swap(operands[0], operands[1]);
      break;
    default:
      break;
  }
  // common subexpression elimination
  auto key = std::make_pair(op, operands);
  auto entry = cache.find(key);
  if (entry != cache.end()) return entry->second;
  auto result = addInstructionNoCache(op, operands);
  cache.insert({key, result});
  return result;
}

// bypass the cache because we don't expect to have more common subexpressions
// after optimizations
Operand Context::addInstructionNoCache(OpCode op,
                                       std::array<Operand, 3> operands) {
  // constant choice
  if (op == OpCode::CHOICE && operands[0].isConst()) {
    if (constants[operands[0].toConstIndex()] == 1.0) return operands[1];
    return operands[2];
  }
  // constant propagation
  bool all_constants = true;
  for (auto operand : operands) {
    if (!operand.isConst() && !operand.isNone()) all_constants = false;
  }
  if (all_constants) {
    tmpTape = {static_cast<uint8_t>(op), 0};
    tmpBuffer = {0.0};
    for (Operand x : operands) {
      if (!x.isConst()) break;
      tmpTape.push_back(tmpBuffer.size());
      tmpBuffer.push_back(constants[x.toConstIndex()]);
    }
    tmpTape.insert(tmpTape.end(), {static_cast<uint8_t>(OpCode::RETURN), 0});
    auto bufferView = VecView(tmpBuffer.data(), tmpBuffer.size());
    return addConstant(EvalContext<double>{tmpTape, bufferView}.eval());
  }

  size_t i = operations.size();
  operations.push_back(op);
  this->operands.push_back(operands);
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

void Context::optimizeFMA() {
  auto tryApply = [&](size_t i, Operand lhs, Operand rhs) {
    if (!lhs.isResult()) return false;
    auto lhsInst = lhs.toInstIndex();
    if (operations[lhsInst] != OpCode::MUL || opUses[lhsInst].size() != 1)
      return false;
    operations[i] = OpCode::FMA;
    Operand a = operands[lhsInst][0];
    Operand b = operands[lhsInst][1];
    operands[i] = {a, b, rhs};
    // remove instruction
    operations[lhsInst] = OpCode::NOP;
    operands[lhsInst] = {Operand::none(), Operand::none(), Operand::none()};
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

  std::vector<Operand> computedInst(oldOperands.size(), Operand::none());
  std::vector<size_t> stack;
  stack.reserve(64);
  if (oldOperands.back()[0].isResult())
    stack.push_back(oldOperands.back()[0].toInstIndex());

  std::vector<uint8_t> bitset(oldOperands.size(), 0);
  std::vector<size_t> distances(oldOperands.size(), 0);
  std::vector<size_t> tmpStack;
  tmpStack.reserve(64);

  auto requiresComputation = [&computedInst](Operand operand) {
    return operand.isResult() && computedInst[operand.toInstIndex()].isNone();
  };
  auto toNewOperand = [&computedInst](Operand old) {
    if (old.isResult()) return computedInst[old.toInstIndex()];
    return old;
  };

  while (!stack.empty()) {
    int numResults = 0;
    auto back = stack.back();
    if (!computedInst[back].isNone()) {
      stack.pop_back();
      continue;
    }
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
          if (!computedInst[current].isNone()) continue;
          bitset[current] |= 1 << numResults;
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
            if (!computedInst[inst].isNone()) continue;

            // shared dependency between operands, also doesn't affect distance
            if ((bitset[inst] & mask) == mask) continue;

            auto d = distances[inst];
            if (d == 0) {
              // not computed
              tmpStack.push_back(x.toInstIndex());
              maxDistance = std::numeric_limits<size_t>::max();
            } else {
              maxDistance = std::max(maxDistance, d);
            }
          }
          if (maxDistance != std::numeric_limits<size_t>::max()) {
            tmpStack.pop_back();
            distances[current] = maxDistance + 1;
          }
        }
        costs[i] = distances[operand.toInstIndex()];
        std::fill(distances.begin(), distances.end(), 0);
      }
      std::sort(ids.begin(), ids.end(),
                [&costs](size_t x, size_t y) { return costs[x] < costs[y]; });
      // expensive operands are placed at the top of the stack, i.e. scheduled
      // earlier
      for (size_t x : ids)
        if (requiresComputation(curOperands[x]))
          stack.push_back(curOperands[x].toInstIndex());

      std::fill(bitset.begin(), bitset.end(), 0);
    } else if (numResults == 1) {
      for (auto operand : curOperands)
        if (requiresComputation(operand))
          stack.push_back(operand.toInstIndex());
    } else {
      stack.pop_back();
      std::array<Operand, 3> newOperands;
      for (int i : {0, 1, 2}) newOperands[i] = toNewOperand(curOperands[i]);
      Operand result = addInstructionNoCache(oldOperations[back], newOperands);
      computedInst[back] = result;
    }
  }
  addInstructionNoCache(OpCode::RETURN,
                        {computedInst[oldOperands.back()[0].toInstIndex()],
                         Operand::none(), Operand::none()});
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
  std::unordered_map<Operand, uint32_t> spills;
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
      // the operand, if present, must be at the end of the cache, due to how the
      // cache is ordered
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

  for (size_t i = 0; i < operations.size(); i++) {
    if (operations[i] == OpCode::NOP) continue;
    auto tmp = handleOperands(operands[i], i);
    if (operations[i] == OpCode::RETURN) {
      tape.insert(tape.end(), {static_cast<uint8_t>(operations[i]), tmp[0]});
      break;
    }
    // note that we may spill the operand register, but that is fine
    uint8_t reg = allocateReg();
    auto instOp = Operand{static_cast<int>(i) + 1};
    auto uses = getUses(instOp);
    if (uses->empty()) {
      // immediately available
      availableReg.push_back(reg);
    } else {
      insertRegCache({uses->front(), instOp, reg});
    }
    tape.insert(tape.end(), {static_cast<uint8_t>(operations[i]), reg});
    for (size_t j : {0, 1, 2}) {
      if (operands[i][j].isNone()) break;
      tape.push_back(tmp[j]);
    }
  }
  return std::make_pair(std::move(tape), bufferSize);
}

}  // namespace manifold::sdf
