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

#include <vector>

#include "affine_value.h"
#include "interval.h"
#include "manifold/vec_view.h"

namespace manifold::sdf {
class OptimizerContext {
 public:
 private:
  struct TapeMetadata {
    size_t tapeLength;
    size_t instructionCount;
    size_t farDependencyCount;
  };
  struct FarDependency {
    uint32_t instructionIndex;
    // dependency indices for the three operands
    // value std::numeric_limits<uint32_t>::max() means there is no such an
    // operand
    std::array<uint32_t, 3> dependencyIndices;
    bool operator<(const FarDependency &other) const {
      return instructionIndex < other.instructionIndex;
    }
  };

  /* --------------------------------------------------------------------------
   * Multiple tape storage, but each tape is represented with 4 arrays,
   * and we join the individual arrays.
   * Ends are marked with tape metadata.
   *
   * Each tape consists of the underlying opcode, use counts for each
   * instruction result, dependencies for each instruction operand. While
   * dependencies can be reconstructed while executing the code, that adds a
   * considerable overhead *regardless* of whether we can optimize later.
   * Instead, we move the overhead to initialization and in the optimizer (it
   * tracks the use count), with the cost of using more memory.
   *
   * Note that we only track direct dependencies: Instructions can depend on
   * load instructions instead of the actual instruction computing the value of
   * the spill, and the load instruction depends on the corresponding store
   * instruction. This allows us to remove register spills if possible.
   *
   * For dependencies, we use relative index to track them.
   *   ID = current ID - value
   * - If value is 0, this means the operand is not being used, i.e. the
   *   instruction does not have that operand.
   * - If value is 255, this means the dependency is too far away, and we
   *   should look it up in far dependencies.
   * Ideally, we should not have too many far dependencies.
   *
   * Due to the variable length encoding used in the instruction tape, we
   * cannot find the opcode in O(1) time given instruction ID, so things we do
   * during optimization should not involve the opcode until we actually need
   * to generate a new tape.
   */
  std::vector<uint8_t> tapes;
  std::vector<uint8_t> useCounts;
  std::vector<std::array<uint8_t, 3>> dependencies;
  std::vector<FarDependency> farDependencies;
  std::vector<TapeMetadata> tapeMetadata;

  /* --------------------------------------------------------------------------
   * Per evaluation data structures.
   * In principle, these can be constructed per evaluation, but to minimize
   * memory operations, we reuse them.
   *
   * - `buffer` is the regular register buffer for tape evaluation.
   * - `affineValue` is the affine value associated with a register.
   * - `results` contains instruction id + affine value, indicating the
   *   optimized result for some instruction.
   *   Note that this must be sorted according to instruction id.
   * - `uses` is the temporary use count vector that is mutable in each
   *   evaluation. It is reset before each evaluation.
   * - `dead` contains instruction IDs that are dead, for later dead code
   *   elimination.
   */
  VecView<Interval<double>> buffer;
  VecView<AffineValue> affineValues;
  std::vector<std::pair<uint32_t, AffineValue>> results;
  std::vector<uint8_t> uses;
  std::vector<uint32_t> dead;
};
}  // namespace manifold::sdf
