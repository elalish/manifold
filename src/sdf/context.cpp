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
    if (!operand.isConst() && !operand.isNone()) all_constants = false;
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
    small_vector<size_t, 4> *target;
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
    operands[i] = {a, b, rhs};
    // remove instruction
    operations[lhsInst] = OpCode::NOP;
    operands[lhsInst] = {Operand::none(), Operand::none(), Operand::none()};
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
      if (iter2 == uses.end() || *iter2 != i) uses.insert(iter2, i);
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
      Operand result = addInstructionNoCache(
          oldOperations[back], toNewOperand(curOperands[0]),
          toNewOperand(curOperands[1]), toNewOperand(curOperands[2]));
      computedInst[back] = result;
    }
  }
  addInstructionNoCache(OpCode::RETURN,
                        computedInst[oldOperands.back()[0].toInstIndex()]);
}

struct LruEntry {
  size_t lastUse;
  Operand operand;
  uint8_t reg;

  inline bool operator<(const LruEntry &other) const {
    return lastUse < other.lastUse ||
           (lastUse == other.lastUse && operand.id < other.operand.id);
  }
};

std::pair<std::vector<uint8_t>, size_t> Context::genTape() {
  std::vector<uint8_t> tape;
  size_t bufferSize = 3;
  std::vector<uint8_t> availableReg;
  // we may want to make this wrap around...
  std::vector<LruEntry> lru;
  std::unordered_map<Operand, uint32_t> spills;
  std::vector<uint32_t> spillSlots;

  auto insertLru = [&](LruEntry entry) {
    lru.insert(std::lower_bound(lru.begin(), lru.end(), entry), entry);
  };
  auto allocateReg = [&]() {
    if (!availableReg.empty()) {
      auto reg = availableReg.back();
      availableReg.pop_back();
      return reg;
    }
    // used too many registers, need to spill something
    // note: tested with a limit of 10, spills correctly
    if (bufferSize > 255) {
      uint32_t slot;
      if (spillSlots.empty()) {
        slot = bufferSize++;
      } else {
        slot = spillSlots.back();
        spillSlots.pop_back();
      }
      spills.insert({lru.front().operand, slot});
      tape.push_back(static_cast<uint8_t>(OpCode::STORE));
      std::array<uint8_t, sizeof(uint32_t)> tmpBuffer;
      std::memcpy(tmpBuffer.data(), &slot, sizeof(uint32_t));
      for (auto byte : tmpBuffer) tape.push_back(byte);
      auto reg = lru.front().reg;
      tape.push_back(reg);
      lru.erase(lru.begin());
      return reg;
    }
    auto reg = static_cast<uint8_t>(bufferSize++);
    return reg;
  };
  auto handleOperands = [&](std::array<Operand, 3> instOperands, size_t inst) {
    auto getReg = [&](Operand operand, size_t inst) {
      if (operand.isNone()) return static_cast<uint8_t>(0);
      // special xyz
      if (!operand.isConst() && !operand.isResult())
        return static_cast<uint8_t>(-(operand.id + 1));
      // Assume last use was updated, the operand, if present, must be at the
      // end of the lru cache. Just do a linear scan from the back
      for (auto it = lru.rbegin(); it != lru.rend(); ++it) {
        // no result
        if (it->lastUse != inst) break;
        if (it->operand == operand) {
          return it->reg;
        }
      }
      auto reg = allocateReg();
      auto iter = spills.find(operand);
      if (iter == spills.end()) {
        DEBUG_ASSERT(operand.isConst(), logicErr,
                     "can only materialize constants");
        tape.push_back(static_cast<uint8_t>(OpCode::CONST));
        tape.push_back(reg);
        std::array<uint8_t, sizeof(double)> tmpBuffer;
        std::memcpy(tmpBuffer.data(), &constants[operand.toConstIndex()],
                    sizeof(double));
        for (auto byte : tmpBuffer) tape.push_back(byte);
      } else {
        tape.push_back(static_cast<uint8_t>(OpCode::LOAD));
        tape.push_back(reg);
        std::array<uint8_t, sizeof(uint32_t)> tmpBuffer;
        std::memcpy(tmpBuffer.data(), &iter->second, sizeof(uint32_t));
        for (auto byte : tmpBuffer) tape.push_back(byte);
        spillSlots.push_back(iter->second);
        spills.erase(iter);
      }
      insertLru({inst, operand, reg});
      return reg;
    };
    auto getUses = [&](Operand operand) {
      if (operand.isResult()) {
        return &opUses[operand.toInstIndex()];
      } else if (operand.isConst()) {
        return &constantUses[operand.toConstIndex()];
      } else {
        return static_cast<small_vector<size_t, 4> *>(nullptr);
      }
    };
    auto updateLru = [&](Operand operand, size_t inst) {
      const auto uses = getUses(operand);
      if (uses == nullptr) return;
      auto i = std::distance(
          uses->begin(), std::lower_bound(uses->begin(), uses->end(), inst));
      if (i == 0 && !operand.isResult()) return;
      size_t lastUse = i == 0 ? operand.toInstIndex() : uses->at(i - 1);
      // when finding the entry, register field doesn't matter
      auto iter = std::lower_bound(lru.begin(), lru.end(),
                                   LruEntry{lastUse, operand, 0});
      if (iter != lru.end() && iter->operand == operand) {
        auto entry = *iter;
        entry.lastUse = inst;
        lru.erase(iter);
        insertLru(entry);
      }
    };
    std::array<uint8_t, 3> regs;
    for (size_t i : {0, 1, 2}) updateLru(instOperands[i], inst);
    for (size_t i : {0, 1, 2}) regs[i] = getReg(instOperands[i], inst);
    // after potential rematerialization, see if they are at the end of their
    // lifetime
    for (size_t i : {0, 1, 2}) {
      if (!instOperands[i].isConst() && !instOperands[i].isResult()) continue;
      if (getUses(instOperands[i])->back() != inst) continue;
      // remove from lru, note that it is possible that it can be removed
      // earlier from another operand
      for (auto it = lru.rbegin(); it != lru.rend(); ++it) {
        if (it->lastUse != inst) break;
        if (it->reg == regs[i]) {
          availableReg.push_back(regs[i]);
          lru.erase(std::next(it).base());
        }
      }
    }
    return regs;
  };

  for (size_t i = 0; i < operations.size(); i++) {
    if (operations[i] == OpCode::NOP) continue;
    auto tmp = handleOperands(operands[i], i);
    if (operations[i] == OpCode::RETURN) {
      tape.push_back(static_cast<uint8_t>(operations[i]));
      tape.push_back(tmp[0]);
      break;
    }
    uint8_t reg = allocateReg();
    insertLru({i, Operand{static_cast<int>(i) + 1}, reg});
    tape.push_back(static_cast<uint8_t>(operations[i]));
    tape.push_back(reg);
    for (size_t j : {0, 1, 2}) {
      if (operands[i][j].isNone()) break;
      tape.push_back(tmp[j]);
    }
  }
  return std::make_pair(std::move(tape), bufferSize);
}

}  // namespace manifold::sdf
