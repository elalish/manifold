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
};

template <>
struct std::hash<AffineValue> {
  size_t operator()(const AffineValue &value) const {
    size_t h = std::hash<int>()(value.var.id);
    hash_combine(h, value.a, value.b);
    return h;
  }
};

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
        std::cout << constants[operand.toConstIndex()] << "(" << operand.id
                  << ")";
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
  auto result =
      constantsIds.insert({d, Operand::fromConstIndex(constants.size())});
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
  size_t i = instructions.size();
  instructions.push_back(inst);
  opUses.emplace_back();
  // update uses
  for (auto operand : inst.operands) {
    auto target = getUses(operand);
    if (target == nullptr) continue;
    // avoid duplicates
    if (target->empty() || target->back() != i) target->push_back(i);
  }
  return Operand::fromInstIndex(i);
}

Context::UsesVector::const_iterator findUse(const Context::UsesVector &uses,
                                            size_t inst) {
  return std::lower_bound(uses.cbegin(), uses.cend(), inst);
}

void Context::addUse(Operand operand, size_t inst) {
  if (!operand.isResult() && !operand.isConst()) return;
  auto uses = getUses(operand);
  auto iter = findUse(*uses, inst);
  if (iter == uses->cend() || *iter != inst) uses->insert(iter, inst);
}

void Context::removeUse(Operand operand, size_t inst) {
  if (!operand.isResult() && !operand.isConst()) return;
  auto uses = getUses(operand);
  auto iter = findUse(*uses, inst);
  if (*iter == inst) uses->erase(iter);
}

void Context::combineFMA() {
  const auto none = Operand::none();
  auto tryApply = [&](size_t i, Operand lhs, Operand rhs) {
    if (!lhs.isResult()) return false;
    auto lhsInst = lhs.toInstIndex();
    if (instructions[lhsInst].op != OpCode::MUL || opUses[lhsInst].size() != 1)
      return false;
    Operand a = instructions[lhsInst].operands[0];
    Operand b = instructions[lhsInst].operands[1];
    instructions[i] = {OpCode::FMA, {a, b, rhs}};
    // remove instruction
    instructions[lhsInst] = {OpCode::NOP, {none, none, none}};
    // update uses, note that we need to maintain the order of the indices
    opUses[lhsInst].clear();
    auto updateUses = [&](Operand x) {
      removeUse(x, lhsInst);
      addUse(x, i);
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

void Context::optimizeAffine() {
  const auto none = Operand::none();
  std::vector<AffineValue> affineValues;
  affineValues.reserve(instructions.size());
  unordered_map<AffineValue, int> avcache;

  auto getConstant = [&](Operand operand) -> std::optional<double> {
    if (operand.isConst()) return constants[operand.toConstIndex()];
    if (operand.isResult() && affineValues[operand.toInstIndex()].a == 0.0)
      return affineValues[operand.toInstIndex()].b;
    return {};
  };

  auto replaceInst = [&](int from, int to) {
    auto fromInst = Operand::fromInstIndex(from);
    auto toInst = Operand::fromInstIndex(to);
    for (auto use : opUses[from]) {
      for (auto &operand : instructions[use].operands)
        if (operand == fromInst) operand = toInst;
    }
    opUses[from].clear();
    instructions[from] = {OpCode::NOP, {none, none, none}};
  };

  // abstract interpretation to figure out affine values for each instruction,
  // and replace them as appropriate
  // note that we still need constant propagation because this abstract
  // interpretation can generate constants
  for (size_t i = 0; i < instructions.size(); i++) {
    auto &inst = instructions[i];
    AffineValue result = AffineValue(Operand::fromInstIndex(i), 1, 0);
    switch (inst.op) {
      case OpCode::NOP:
      case OpCode::RETURN:
      case OpCode::CONSTANT:
      case OpCode::LOAD:
      case OpCode::STORE:
        break;
      // notably, neg is special among these unary opcode
      case OpCode::ABS:
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
      case OpCode::ATAN: {
        auto x = getConstant(inst.operands[0]);
        if (x.has_value())
          result = AffineValue(
              EvalContext<double>::handle_unary(inst.op, x.value()));
        break;
      }
      case OpCode::NEG:
        if (inst.operands[0].isConst())
          result = AffineValue(-constants[inst.operands[0].toConstIndex()]);
        else if (inst.operands[0].isResult()) {
          auto av = affineValues[inst.operands[0].toInstIndex()];
          result = AffineValue(av.var, -av.a, -av.b);
        }
        break;
      case OpCode::DIV: {
        // TODO: handle the case where lhs is divisible by rhs despite rhs is
        // not a constant
        auto rhs = getConstant(inst.operands[1]);
        if (rhs.has_value()) {
          if (inst.operands[0].isConst()) {
            result = AffineValue(constants[inst.operands[0].toConstIndex()] /
                                 rhs.value());
          } else if (inst.operands[0].isResult()) {
            auto av = affineValues[inst.operands[0].toInstIndex()];
            result =
                AffineValue(av.var, av.a / rhs.value(), av.b / rhs.value());
          }
        }
        break;
      }
      case OpCode::MOD:
      case OpCode::MIN:
      case OpCode::MAX:
      case OpCode::EQ:
      case OpCode::GT: {
        // TODO: we can do better than just constant propagation...
        auto lhs = getConstant(inst.operands[0]);
        auto rhs = getConstant(inst.operands[1]);
        if (lhs.has_value() && rhs.has_value())
          result = AffineValue(EvalContext<double>::handle_binary(
              inst.op, lhs.value(), rhs.value()));
        break;
      }
      case OpCode::ADD: {
        auto x = inst.operands[0];
        auto y = inst.operands[1];
        auto lhs = getConstant(x);
        auto rhs = getConstant(y);
        if (lhs.has_value() && rhs.has_value()) {
          result = AffineValue(lhs.value() + rhs.value());
        } else if (lhs.has_value() && y.isResult()) {
          result = affineValues[y.toInstIndex()];
          result.b += lhs.value();
        } else if (rhs.has_value() && x.isResult()) {
          result = affineValues[x.toInstIndex()];
          result.b += rhs.value();
        } else if (x.isResult() && y.isResult()) {
          if (affineValues[x.toInstIndex()].var ==
              affineValues[y.toInstIndex()].var) {
            auto other = affineValues[y.toInstIndex()];
            result = affineValues[x.toInstIndex()];
            result.a += other.a;
            result.b += other.b;
          }
        }
        break;
      }
      case OpCode::SUB: {
        auto x = inst.operands[0];
        auto y = inst.operands[1];
        auto lhs = getConstant(x);
        auto rhs = getConstant(y);
        if (lhs.has_value() && rhs.has_value()) {
          result = AffineValue(lhs.value() - rhs.value());
        } else if (lhs.has_value() && y.isResult()) {
          result = affineValues[y.toInstIndex()];
          result.a = -result.a;
          result.b = lhs.value() - result.b;
        } else if (rhs.has_value() && x.isResult()) {
          result = affineValues[x.toInstIndex()];
          result.b -= rhs.value();
        } else if (x.isResult() && y.isResult()) {
          if (affineValues[x.toInstIndex()].var ==
              affineValues[y.toInstIndex()].var) {
            auto other = affineValues[y.toInstIndex()];
            result = affineValues[x.toInstIndex()];
            result.a -= other.a;
            result.b -= other.b;
          }
        }
        break;
      }
      case OpCode::MUL: {
        auto x = inst.operands[0];
        auto y = inst.operands[1];
        auto lhs = getConstant(x);
        auto rhs = getConstant(y);
        if (lhs.has_value() && rhs.has_value()) {
          result = AffineValue(lhs.value() * rhs.value());
        } else if (lhs.has_value() && y.isResult()) {
          result = affineValues[y.toInstIndex()];
          result.a *= lhs.value();
          result.b *= lhs.value();
        } else if (rhs.has_value() && x.isResult()) {
          result = affineValues[x.toInstIndex()];
          result.a *= rhs.value();
          result.b *= rhs.value();
        }
        break;
      }
      case OpCode::FMA: {
        auto x = inst.operands[0];
        auto y = inst.operands[1];
        auto z = inst.operands[2];
        auto a = getConstant(x);
        auto b = getConstant(y);
        auto c = getConstant(z);
        // various cases...
        if (b.has_value() && c.has_value()) {
          result = affineValues[x.toInstIndex()];
          result.a *= b.value();
          result.b = result.b * b.value() + c.value();
        } else if (a.has_value() && c.has_value()) {
          result = affineValues[y.toInstIndex()];
          result.a *= a.value();
          result.b = result.b * a.value() + c.value();
        } else if (a.has_value() && b.has_value()) {
          result = affineValues[z.toInstIndex()];
          result.b += a.value() * b.value();
        }
        break;
      }
      case OpCode::CHOICE: {
        auto c = getConstant(inst.operands[0]);
        auto a = inst.operands[1];
        auto b = inst.operands[2];
        if (c.has_value()) {
          if (c.value() == 0.0)
            result = affineValues[b.toInstIndex()];
          else
            result = affineValues[a.toInstIndex()];
        }
        break;
      }
    }
    affineValues.push_back(result);
    if (result.var != Operand::fromInstIndex(i)) {
      // we did evaluate something
      auto pair = avcache.insert({result, static_cast<int>(i)});
      if (!pair.second) {
        // this result is being optimized away, replace uses with the value
        replaceInst(static_cast<int>(i), pair.first->second);
      } else {
        for (auto operand : inst.operands) removeUse(operand, i);
        if (!result.var.isNone()) addUse(result.var, i);
        // modify instruction
        if (result.a == 1.0 && result.b == 0.0 && result.var.isResult()) {
          // this result is being optimized away, replace uses with the value
          pair.first->second = result.var.toInstIndex();
          replaceInst(static_cast<int>(i),
                      static_cast<int>(result.var.toInstIndex()));
        } else if (result.a == 1.0) {
          auto constant = addConstant(result.b);
          addUse(constant, i);
          instructions[i] = {OpCode::ADD, {constant, result.var, none}};
        } else if (result.a == -1.0 && result.b == 0.0) {
          instructions[i] = {OpCode::NEG, {result.var, none, none}};
        } else if (result.a == -1.0) {
          auto constant = addConstant(result.b);
          addUse(constant, i);
          instructions[i] = {OpCode::SUB, {constant, result.var, none}};
        } else if (result.b == 0.0) {
          auto constant = addConstant(result.a);
          addUse(constant, i);
          instructions[i] = {OpCode::MUL, {constant, result.var, none}};
        } else if (result.a == 0.0) {
          auto a = addConstant(0.0);
          auto b = addConstant(result.b);
          addUse(a, i);
          addUse(b, i);
          instructions[i] = {OpCode::ADD, {b, a, none}};
        } else {
          auto a = addConstant(result.a);
          auto b = addConstant(result.b);
          addUse(a, i);
          addUse(b, i);
          instructions[i] = {OpCode::FMA, {a, result.var, b}};
        }
      }
    }
  }
}

void Context::schedule() {
  cache.clear();
  opUses.clear();
  for (auto &uses : constantUses) uses.clear();
  auto oldInstructions = std::move(this->instructions);
  // compute depth in DG
  std::vector<size_t> levelMap;
  levelMap.reserve(oldInstructions.size());
  for (size_t i = 0; i < oldInstructions.size(); i++) {
    const auto &inst = oldInstructions[i];
    size_t maxLevel = 0;
    for (auto operand : inst.operands) {
      if (!operand.isResult()) continue;
      maxLevel = std::max(maxLevel, levelMap[operand.toInstIndex()]);
    }
    levelMap.push_back(maxLevel + 1);
  }

  std::vector<Operand> computedInst(oldInstructions.size(), Operand::none());
  std::vector<size_t> stack;
  if (oldInstructions.back().operands[0].isResult())
    stack.push_back(oldInstructions.back().operands[0].toInstIndex());

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
    auto &inst = oldInstructions[back];
    std::array<size_t, 3> costs = {0, 0, 0};
    std::array<size_t, 3> ids = {0, 1, 2};
    for (auto i : ids)
      if (requiresComputation(inst.operands[i])) {
        numResults += 1;
        costs[i] = levelMap[inst.operands[i].toInstIndex()];
      }
    if (numResults > 0) {
      std::sort(ids.begin(), ids.end(),
                [&costs](size_t x, size_t y) { return costs[x] < costs[y]; });
      for (size_t x : ids)
        if (requiresComputation(inst.operands[x]))
          stack.push_back(inst.operands[x].toInstIndex());
    } else {
      stack.pop_back();
      std::array<Operand, 3> newOperands;
      for (int i : ids) newOperands[i] = toNewOperand(inst.operands[i]);
      Operand result = addInstructionNoCache({inst.op, newOperands});
      computedInst[back] = result;
    }
  }
  addInstructionNoCache(
      {OpCode::RETURN,
       {computedInst[oldInstructions.back().operands[0].toInstIndex()],
        Operand::none(), Operand::none()}});
}

void Context::optimize() {
  optimizeAffine();
  combineFMA();
  schedule();
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
        tape.insert(tape.end(), {static_cast<uint8_t>(OpCode::CONSTANT), reg});
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
        auto nextUse = *(findUse(*uses, inst) + 1);
        insertRegCache({nextUse, instOperands[i], regs[i]});
      }
    }
    return regs;
  };

  for (size_t i = 0; i < instructions.size(); i++) {
    auto &inst = instructions[i];
    auto instOp = Operand::fromInstIndex(i);
    auto uses = getUses(instOp);
    if (inst.op == OpCode::NOP) continue;
    // if (inst.op != OpCode::RETURN && uses->empty()) continue;
    auto tmp = handleOperands(inst.operands, i);
    if (inst.op == OpCode::RETURN) {
      tape.insert(tape.end(), {static_cast<uint8_t>(inst.op), tmp[0]});
      break;
    }
    // note that we may spill the operand register, but that is fine
    uint8_t reg = allocateReg();
    if (uses->empty()) {
      availableReg.push_back(reg);
    } else {
      insertRegCache({uses->front(), instOp, reg});
    }
    tape.insert(tape.end(), {static_cast<uint8_t>(inst.op), reg});
    for (size_t j : {0, 1, 2}) {
      if (inst.operands[j].isNone()) break;
      tape.push_back(tmp[j]);
    }
  }
  return std::make_pair(std::move(tape), bufferSize);
}

}  // namespace manifold::sdf
