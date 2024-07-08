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
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
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
#include <vector>

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

template <typename Iter, typename F>
void for_each(ExecutionPolicy policy, Iter first, Iter last, F f) {
#if MANIFOLD_PAR == 'T'
  if (policy == ExecutionPolicy::Par) {
    tbb::parallel_for(tbb::blocked_range<Iter>(first, last),
                      [&f](const tbb::blocked_range<Iter> &range) {
                        for (Iter i = range.begin(); i != range.end(); i++)
                          f(*i);
                      });
    return;
  }
#endif
  std::for_each(first, last, f);
}

template <typename Iter, typename F>
void for_each_n(ExecutionPolicy policy, Iter first, size_t n, F f) {
  for_each(policy, first, first + n, f);
}

template <typename InputIter, typename BinaryOp,
          typename T = typename std::iterator_traits<InputIter>::value_type>
T reduce(ExecutionPolicy policy, InputIter first, InputIter last, T init,
         BinaryOp f) {
#if MANIFOLD_PAR == 'T'
  if (policy == ExecutionPolicy::Par) {
    // should we use deterministic reduce here?
    return tbb::parallel_reduce(
        tbb::blocked_range<InputIter>(first, last), init,
        [&f](const tbb::blocked_range<InputIter> &range, T value) {
          for (InputIter i = range.begin(); i != range.end(); i++)
            value = f(value, *i);
          return value;
        },
        f);
  }
#endif
  return std::reduce(first, last, init, f);
}

template <typename InputIter, typename BinaryOp, typename UnaryOp,
          typename T = std::invoke_result_t<
              UnaryOp, typename std::iterator_traits<InputIter>::value_type>>
T transform_reduce(ExecutionPolicy policy, InputIter first, InputIter last,
                   T init, BinaryOp f, UnaryOp g) {
  return reduce(policy, TransformIterator(first, g), TransformIterator(last, g),
                init, f);
}

template <typename InputIter, typename OutputIter,
          typename T = typename std::iterator_traits<InputIter>::value_type,
          typename Dummy = void>
void inclusive_scan(ExecutionPolicy policy, InputIter first, InputIter last,
                    OutputIter d_first) {
#if MANIFOLD_PAR == 'T'
  if (policy == ExecutionPolicy::Par) {
    tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, std::distance(first, last)),
        static_cast<T>(0),
        [&](const tbb::blocked_range<size_t> &range, T sum,
            bool is_final_scan) {
          T temp = sum;
          for (size_t i = range.begin(); i < range.end(); ++i) {
            temp = temp + first[i];
            if (is_final_scan) d_first[i] = temp;
          }
          return temp;
        },
        std::plus<T>());
    return;
  }
#endif
  std::inclusive_scan(first, last, d_first);
}

#if MANIFOLD_PAR == 'T'
template <typename T, typename InputIter, typename OutputIter, typename BinOp>
struct ScanBody {
  T sum;
  T identity;
  BinOp &f;
  InputIter input;
  OutputIter output;

  ScanBody(T sum, T identity, BinOp &f, InputIter input, OutputIter output)
      : sum(sum), identity(identity), f(f), input(input), output(output) {}
  ScanBody(ScanBody &b, tbb::split)
      : sum(b.identity),
        identity(b.identity),
        f(b.f),
        input(b.input),
        output(b.output) {}
  template <typename Tag>
  void operator()(const tbb::blocked_range<size_t> &r, Tag) {
    T temp = sum;
    for (size_t i = r.begin(); i < r.end(); ++i) {
      T inputTmp = input[i];
      if (Tag::is_final_scan()) output[i] = temp;
      temp = f(temp, inputTmp);
    }
    sum = temp;
  }
  T get_sum() const { return sum; }
  void reverse_join(ScanBody &a) { sum = f(a.sum, sum); }
  void assign(ScanBody &b) { sum = b.sum; }
};
#endif

template <typename InputIter, typename OutputIter,
          typename BinOp = decltype(std::plus<typename std::iterator_traits<
                                        InputIter>::value_type>()),
          typename T = typename std::iterator_traits<InputIter>::value_type>
void exclusive_scan(ExecutionPolicy policy, InputIter first, InputIter last,
                    OutputIter d_first, T init = static_cast<T>(0),
                    BinOp f = std::plus<T>(), T identity = static_cast<T>(0)) {
#if MANIFOLD_PAR == 'T'
  if (policy == ExecutionPolicy::Par) {
    ScanBody<T, InputIter, OutputIter, BinOp> body(init, identity, f, first,
                                                   d_first);
    tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, std::distance(first, last)), body);
    return;
  }
#endif
  std::exclusive_scan(first, last, d_first, init, f);
}

// TODO: use STL variant when parallelization is not enabled?

template <typename InputIter, typename OutputIter, typename F>
void transform(ExecutionPolicy policy, InputIter first, InputIter last,
               OutputIter d_first, F f) {
  for_each_n(policy, countAt(0_z), std::distance(first, last),
             [&](size_t i) { d_first[i] = f(first[i]); });
}

template <typename InputIter, typename OutputIter>
void copy(ExecutionPolicy policy, InputIter first, InputIter last,
          OutputIter d_first) {
  for_each_n(policy, countAt(0_z), std::distance(first, last),
             [&](size_t i) { d_first[i] = first[i]; });
}

template <typename InputIter, typename OutputIter>
void copy_n(ExecutionPolicy policy, InputIter first, size_t n,
            OutputIter d_first) {
  for_each_n(policy, countAt(0_z), n, [&](size_t i) { d_first[i] = first[i]; });
}

template <typename OutputIter, typename T>
void fill(ExecutionPolicy policy, OutputIter first, OutputIter last, T value) {
  for_each_n(policy, countAt(0_z), std::distance(first, last),
             [&](size_t i) { first[i] = value; });
}

template <typename InputIter, typename P>
size_t count_if(ExecutionPolicy policy, InputIter first, InputIter last,
                P pred) {
  return reduce(policy, TransformIterator(first, pred),
                TransformIterator(last, pred), 0_z, std::plus<size_t>());
}

template <typename InputIter, typename P>
bool all_of(ExecutionPolicy policy, InputIter first, InputIter last, P pred) {
  // can probably optimize a bit for short-circuiting
  return reduce(policy, TransformIterator(first, pred),
                TransformIterator(last, pred), true,
                [](bool a, bool b) { return a && b; });
}

#if MANIFOLD_PAR == 'T'
template <typename InputIter, typename OutputIter, typename P>
struct CopyIfScanBody {
  size_t sum;
  P &pred;
  InputIter input;
  OutputIter output;

  CopyIfScanBody(P &pred, InputIter input, OutputIter output)
      : sum(0), pred(pred), input(input), output(output) {}
  CopyIfScanBody(CopyIfScanBody &b, tbb::split)
      : sum(0), pred(b.pred), input(b.input), output(b.output) {}
  template <typename Tag>
  void operator()(const tbb::blocked_range<size_t> &r, Tag) {
    size_t temp = sum;
    for (size_t i = r.begin(); i < r.end(); ++i) {
      bool good = pred(input[i]);
      temp += good;
      if (Tag::is_final_scan() && good) output[temp - 1] = input[i];
    }
    sum = temp;
  }
  size_t get_sum() const { return sum; }
  void reverse_join(CopyIfScanBody &a) { sum = a.sum + sum; }
  void assign(CopyIfScanBody &b) { sum = b.sum; }
};
#endif

// note that you should not have alias between input and output...
// in general it is impossible to check if there is any alias, as the input
// iterator can be computed on-the-fly depending on the output position and may
// not have a pointer
template <typename InputIter, typename OutputIter, typename P>
OutputIter copy_if(ExecutionPolicy policy, InputIter first, InputIter last,
                   OutputIter d_first, P pred) {
#if MANIFOLD_PAR == 'T'
  if (policy == ExecutionPolicy::Par) {
    CopyIfScanBody body(pred, first, d_first);
    tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, std::distance(first, last)), body);
    return d_first + body.get_sum();
  }
#endif
  return std::copy_if(first, last, d_first, pred);
}

template <typename Iter, typename P,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter remove_if(ExecutionPolicy policy, Iter first, Iter last, P pred) {
#if MANIFOLD_PAR == 'T'
  if (policy == ExecutionPolicy::Par) {
    std::vector<T> tmp(std::distance(first, last));
    auto back = copy_if(policy, first, last, tmp.begin(),
                        [&](T v) { return !pred(v); });
    copy(policy, tmp.begin(), back, first);
    return first + std::distance(tmp.begin(), back);
  }
#endif
  return std::remove_if(first, last, pred);
}

template <typename Iter,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter remove(ExecutionPolicy policy, Iter first, Iter last, T value) {
#if MANIFOLD_PAR == 'T'
  if (policy == ExecutionPolicy::Par) {
    std::vector<T> tmp(std::distance(first, last));
    auto back = copy_if(policy, first, last, tmp.begin(),
                        [&](T v) { return v != value; });
    copy(policy, tmp.begin(), back, first);
    return first + std::distance(tmp.begin(), back);
  }
#endif
  return std::remove(first, last, value);
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

// void implies that the user have to specify the return type in the template
// argument, as we are unable to deduce it
STL_DYNAMIC_BACKEND(unique, void)
STL_DYNAMIC_BACKEND_VOID(uninitialized_fill)
STL_DYNAMIC_BACKEND_VOID(uninitialized_copy)
STL_DYNAMIC_BACKEND_VOID(stable_sort)

}  // namespace manifold
