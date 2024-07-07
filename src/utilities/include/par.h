// Copyright 2022 The Manifold Authors.
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
#if MANIFOLD_PAR == 'T'
#if __has_include(<pstl/glue_execution_defs.h>)
#include <execution>
#define HAS_PAR_UNSEQ
#elif __has_include(<oneapi/dpl/execution>)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/memory>
#include <oneapi/dpl/numeric>
#define HAS_PAR_UNSEQ
#endif
#endif
#include <algorithm>
#include <numeric>

#include "iters.h"
#include "public.h"
namespace manifold {

enum class ExecutionPolicy {
  Par,
  Seq,
};

// ExecutionPolicy:
// - Sequential for small workload,
// - Parallel (CPU) for medium workload,
// - GPU for large workload if available.
inline constexpr ExecutionPolicy autoPolicy(size_t size) {
  // some random numbers
  if (size <= (1 << 12)) {
    return ExecutionPolicy::Seq;
  }
  return ExecutionPolicy::Par;
}

#ifdef HAS_PAR_UNSEQ
#define STL_DYNAMIC_BACKEND(NAME, RET)                        \
  template <typename Ret = RET, typename... Args>             \
  Ret NAME(ExecutionPolicy policy, Args... args) {            \
    switch (policy) {                                         \
      case ExecutionPolicy::Par:                              \
        return std::NAME(std::execution::par_unseq, args...); \
      case ExecutionPolicy::Seq:                              \
        break;                                                \
    }                                                         \
    return std::NAME(args...);                                \
  }
#define STL_DYNAMIC_BACKEND_VOID(NAME)                 \
  template <typename... Args>                          \
  void NAME(ExecutionPolicy policy, Args... args) {    \
    switch (policy) {                                  \
      case ExecutionPolicy::Par:                       \
        std::NAME(std::execution::par_unseq, args...); \
        break;                                         \
      case ExecutionPolicy::Seq:                       \
        std::NAME(args...);                            \
        break;                                         \
    }                                                  \
  }
#else
#define STL_DYNAMIC_BACKEND(NAME, RET)             \
  template <typename Ret = RET, typename... Args>  \
  Ret NAME(ExecutionPolicy policy, Args... args) { \
    return std::NAME(args...);                     \
  }
#define STL_DYNAMIC_BACKEND_VOID(NAME)              \
  template <typename... Args>                       \
  void NAME(ExecutionPolicy policy, Args... args) { \
    std::NAME(args...);                             \
  }
#endif

template <typename... Args>
void exclusive_scan(ExecutionPolicy policy, Args... args) {
  // https://github.com/llvm/llvm-project/issues/59810
  std::exclusive_scan(args...);
}

template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator>
void scatter(ExecutionPolicy policy, InputIterator1 first, InputIterator1 last,
             InputIterator2 mapFirst, OutputIterator outputFirst) {
  for_each(policy, countAt(0_z),
           countAt(static_cast<size_t>(std::distance(first, last))),
           [first, mapFirst, outputFirst](size_t i) {
             outputFirst[mapFirst[i]] = first[i];
           });
}

template <typename InputIterator, typename RandomAccessIterator,
          typename OutputIterator>
void gather(ExecutionPolicy policy, InputIterator mapFirst,
            InputIterator mapLast, RandomAccessIterator inputFirst,
            OutputIterator outputFirst) {
  for_each(policy, countAt(0_z),
           countAt(static_cast<size_t>(std::distance(mapFirst, mapLast))),
           [mapFirst, inputFirst, outputFirst](size_t i) {
             outputFirst[i] = inputFirst[mapFirst[i]];
           });
}

template <typename Iterator>
void sequence(ExecutionPolicy policy, Iterator first, Iterator last) {
  for_each(policy, countAt(0_z),
           countAt(static_cast<size_t>(std::distance(first, last))),
           [first](size_t i) { first[i] = i; });
}

// void implies that the user have to specify the return type in the template
// argument, as we are unable to deduce it
STL_DYNAMIC_BACKEND(remove, void)
STL_DYNAMIC_BACKEND(find, void)
STL_DYNAMIC_BACKEND(find_if, void)
STL_DYNAMIC_BACKEND(all_of, bool)
STL_DYNAMIC_BACKEND(is_sorted, bool)
STL_DYNAMIC_BACKEND(reduce, void)
STL_DYNAMIC_BACKEND(count_if, int)
STL_DYNAMIC_BACKEND(remove_if, void)
STL_DYNAMIC_BACKEND(copy_if, void)
STL_DYNAMIC_BACKEND(unique, void)
STL_DYNAMIC_BACKEND(transform_reduce, void)

STL_DYNAMIC_BACKEND_VOID(for_each)
STL_DYNAMIC_BACKEND_VOID(for_each_n)
STL_DYNAMIC_BACKEND_VOID(transform)
STL_DYNAMIC_BACKEND_VOID(uninitialized_fill)
STL_DYNAMIC_BACKEND_VOID(uninitialized_copy)
STL_DYNAMIC_BACKEND_VOID(stable_sort)
STL_DYNAMIC_BACKEND_VOID(fill)
STL_DYNAMIC_BACKEND_VOID(inclusive_scan)
STL_DYNAMIC_BACKEND_VOID(copy)
STL_DYNAMIC_BACKEND_VOID(copy_n)

}  // namespace manifold
