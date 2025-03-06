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
//
// Simple implementation of selected functions in PSTL.
// Iterators must be RandomAccessIterator.

#pragma once

#include <thread>

#include "./iters.h"
#if (MANIFOLD_PAR == 1)
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#endif
#include <algorithm>
#include <numeric>

namespace manifold {

enum class ExecutionPolicy {
  Par,
  Seq,
};

constexpr size_t kSeqThreshold = 1e4;
// ExecutionPolicy:
// - Sequential for small workload,
// - Parallel (CPU) for medium workload,
inline constexpr ExecutionPolicy autoPolicy(size_t size,
                                            size_t threshold = kSeqThreshold) {
  if (size <= threshold) {
    return ExecutionPolicy::Seq;
  }
  return ExecutionPolicy::Par;
}

template <typename Iter,
          typename Dummy = std::enable_if_t<!std::is_integral_v<Iter>>>
inline constexpr ExecutionPolicy autoPolicy(Iter first, Iter last,
                                            size_t threshold = kSeqThreshold) {
  if (static_cast<size_t>(std::distance(first, last)) <= threshold) {
    return ExecutionPolicy::Seq;
  }
  return ExecutionPolicy::Par;
}

template <typename InputIter, typename OutputIter>
void copy(ExecutionPolicy policy, InputIter first, InputIter last,
          OutputIter d_first);
template <typename InputIter, typename OutputIter>
void copy(InputIter first, InputIter last, OutputIter d_first);

#if (MANIFOLD_PAR == 1)
namespace details {
using manifold::kSeqThreshold;
// implementation from
// https://duvanenko.tech.blog/2018/01/14/parallel-merge/
// https://github.com/DragonSpit/ParallelAlgorithms
// note that the ranges are now [p, r) to fit our convention.
template <typename SrcIter, typename DestIter, typename Comp>
void mergeRec(SrcIter src, DestIter dest, size_t p1, size_t r1, size_t p2,
              size_t r2, size_t p3, Comp comp) {
  size_t length1 = r1 - p1;
  size_t length2 = r2 - p2;
  if (length1 < length2) {
    std::swap(p1, p2);
    std::swap(r1, r2);
    std::swap(length1, length2);
  }
  if (length1 == 0) return;
  if (length1 + length2 <= kSeqThreshold) {
    std::merge(src + p1, src + r1, src + p2, src + r2, dest + p3, comp);
  } else {
    size_t q1 = p1 + length1 / 2;
    size_t q2 =
        std::distance(src, std::lower_bound(src + p2, src + r2, src[q1], comp));
    size_t q3 = p3 + (q1 - p1) + (q2 - p2);
    dest[q3] = src[q1];
    tbb::parallel_invoke(
        [=] { mergeRec(src, dest, p1, q1, p2, q2, p3, comp); },
        [=] { mergeRec(src, dest, q1 + 1, r1, q2, r2, q3 + 1, comp); });
  }
}

template <typename SrcIter, typename DestIter, typename Comp>
void mergeSortRec(SrcIter src, DestIter dest, size_t begin, size_t end,
                  Comp comp) {
  size_t numElements = end - begin;
  if (numElements <= kSeqThreshold) {
    std::copy(src + begin, src + end, dest + begin);
    std::stable_sort(dest + begin, dest + end, comp);
  } else {
    size_t middle = begin + numElements / 2;
    tbb::parallel_invoke([=] { mergeSortRec(dest, src, begin, middle, comp); },
                         [=] { mergeSortRec(dest, src, middle, end, comp); });
    mergeRec(src, dest, begin, middle, middle, end, begin, comp);
  }
}

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
      if (pred(i)) {
        temp += 1;
        if (Tag::is_final_scan()) output[temp - 1] = input[i];
      }
    }
    sum = temp;
  }
  size_t get_sum() const { return sum; }
  void reverse_join(CopyIfScanBody &a) { sum = a.sum + sum; }
  void assign(CopyIfScanBody &b) { sum = b.sum; }
};

// combine subarray histograms into offsets
// 0 - (k-1): offset for each block (k blocks in total)
// k: full histogram for the array
inline bool sumCounts(std::vector<std::array<size_t, 256>> &hists, size_t n) {
  // prefix sum to compute offsets for each byte pattern
  // we use the final block to store the total, because we need exclusive sum,
  // e.g. (1, 2, 3) => (0, 1, 3, 6)
  auto &total = hists.back();
  std::fill(total.begin(), total.end(), 0);

  for (size_t i = 0; i < hists.size() - 1; ++i) {
    for (size_t j = 0; j < 256; ++j) {
      size_t old = total[j];
      total[j] += hists[i][j];
      hists[i][j] = old;
    }
  }

  // prefix sum on total count
  for (size_t j = 0; j < 256; ++j) {
    if (total[j] == n) return true;
    if (j != 0) total[j] += total[j - 1];
  }

  // add final total to each offset for each byte pattern
  for (size_t i = 0; i < hists.size() - 1; ++i)
    for (size_t j = 1; j < 256; ++j) hists[i][j] += total[j - 1];

  return false;
}

template <typename KeyFn>
struct ByteGetter {
  KeyFn &keyfn;
  ByteGetter(KeyFn &keyfn) : keyfn(keyfn) {}
  template <typename I>
  size_t operator()(I curr, int i) const {
    return static_cast<size_t>((keyfn(*curr) >> (8 * i)) & 0xFF);
  }
};

template <typename U>
struct RadixBlockMetadata {
  U localMin;
  U localMax;
  U prefixMax;
  U suffixMin;
  bool sorted;
  bool canSkip;
};

template <typename I, typename J, typename KeyFn>
void radix_sort_with_key_rec(I start, I end, J dest, int bytes, KeyFn &keyfn,
                             bool writeback) {
  using T = std::remove_reference_t<decltype(*start)>;
  using U = std::invoke_result_t<KeyFn, T>;
  constexpr size_t MIN_BLOCK_SIZE = 10'000;
  size_t n = std::distance(start, end);

  if ((n <= kSeqThreshold && bytes >= 1) || (n < 2'000)) {
    if (writeback) {
      std::stable_sort(start, end, [&keyfn](const T &a, const T &b) {
        return keyfn(a) < keyfn(b);
      });
    } else {
      std::copy(start, end, dest);
      std::stable_sort(dest, dest + n, [&keyfn](const T &a, const T &b) {
        return keyfn(a) < keyfn(b);
      });
    }
    return;
  }
  auto getByte = ByteGetter(keyfn);

  unsigned int hardware_concurrency = std::thread::hardware_concurrency();
  size_t blocks =
      std::max(std::min(hardware_concurrency,
                        static_cast<unsigned int>(n / MIN_BLOCK_SIZE)),
               1u);
#if MANIFOLD_PAR == 1
  if (blocks > 1) {
    size_t blockSize = n / blocks;
    std::vector<std::array<size_t, 256>> hists(blocks + 1);

    // local min, local max, prefix max, suffix min, sorted, can skip
    std::vector<RadixBlockMetadata<U>> metadata(blocks);

    auto taskgroup = tbb::task_group();
    for (size_t i = 0; i < blocks; ++i) {
      taskgroup.run([&, i]() {
        std::fill(hists[i].begin(), hists[i].end(), 0);
        I localStart = start + blockSize * i;
        I localEnd = start + (i == blocks - 1 ? n : blockSize * (i + 1));

        U localMin = keyfn(*localStart);
        U localMax = keyfn(*localStart);
        U prev = keyfn(*localStart);
        bool sorted = true;

        for (I curr = localStart; curr != localEnd; ++curr) {
          U x = keyfn(*curr);
          localMin = std::min(localMin, x);
          localMax = std::max(localMax, x);
          if (prev > x) sorted = false;
          prev = x;
          hists[i][getByte(curr, bytes)]++;
        }
        metadata[i] = {.localMin = localMin,
                       .localMax = localMax,
                       .prefixMax = localMax,
                       .suffixMin = localMin,
                       .sorted = sorted,
                       .canSkip = false};
      });
    }
    taskgroup.wait();

    // compute prefix max
    for (size_t i = 1; i < blocks; ++i)
      metadata[i].prefixMax =
          std::max(metadata[i].prefixMax, metadata[i - 1].prefixMax);
    // suffix min
    for (size_t i = 1; i < blocks; ++i)
      metadata[blocks - i - 1].suffixMin = std::min(
          metadata[blocks - i - 1].suffixMin, metadata[blocks - i].suffixMin);

    int prefixSkips = 0;
    int suffixSkips = 0;
    // check if the prefix can be skipped
    for (size_t i = 0; i < blocks; ++i) {
      auto &m = metadata[i];
      if (m.sorted &&
          (i == blocks - 1 || m.localMax <= metadata[i + 1].suffixMin))
        m.canSkip = true;
      else
        break;
      prefixSkips++;
    }
    // check if the suffix can be skipped
    for (size_t i = 0; i < blocks; ++i) {
      // starting from the last block
      auto &m = metadata[blocks - i - 1];
      if (m.canSkip) {
        // early return, the whole thing is sorted
        if (!writeback) manifold::copy(start, end, dest);
        return;
      }
      // if prefix max <= localMin
      if (m.sorted &&
          (i == blocks - 1 || metadata[blocks - i - 2].prefixMax <= m.localMin))
        m.canSkip = true;
      else
        break;
      suffixSkips++;
    }

    size_t unsortedStart = prefixSkips * blockSize;
    size_t unsortedEnd =
        suffixSkips == 0 ? n : (blocks - suffixSkips) * blockSize;
    // fast path: no need to shuffle, all have the same byte pattern
    if (sumCounts(hists, n)) {
      if (bytes > 0) {
        if (!writeback) {
          manifold::copy(start, start + unsortedStart, dest);
          manifold::copy(start + unsortedEnd, end, dest + unsortedEnd);
        }
        radix_sort_with_key_rec(start + unsortedStart, start + unsortedEnd,
                                dest + unsortedStart, bytes - 1, keyfn,
                                writeback);
      } else if (!writeback) {
        manifold::copy(start, end, dest);
      }
      return;
    }

    for (size_t i = 0; i < blocks; ++i) {
      taskgroup.run([&, i]() {
        I localStart = start + blockSize * i;
        I localEnd = start + (i == blocks - 1 ? n : blockSize * (i + 1));
        if (metadata[i].canSkip) {
          std::copy(localStart, localEnd, dest + blockSize * i);
        } else {
          for (I curr = localStart; curr != localEnd; ++curr)
            dest[hists[i][getByte(curr, bytes)]++] = *curr;
        }
      });
    }
    taskgroup.wait();

    if (bytes > 0) {
      for (size_t i = 0; i < 256; ++i) {
        size_t lower = i == 0 ? 0 : hists.back()[i - 1];
        size_t upper = hists.back()[i];
        if (lower == upper) continue;
        // we can skip sorted regions because they are the same in both
        // start-end and dest
        if (upper <= unsortedStart || lower >= unsortedEnd) continue;
        taskgroup.run([&, lower, upper]() {
          radix_sort_with_key_rec(dest + lower, dest + upper, start + lower,
                                  bytes - 1, keyfn, !writeback);
        });
      }
      taskgroup.wait();
    } else if (writeback) {
      manifold::copy(dest, dest + n, start);
    }
  } else
#endif
  {
    // LSB radix sort, better for single thread
    std::vector<std::array<size_t, 256>> hists(bytes + 1);
    for (size_t i = 0; i < hists.size(); ++i)
      std::fill(hists[i].begin(), hists[i].end(), 0);

    // compute all histograms at once
    for (I curr = start; curr != end; ++curr)
      for (size_t i = 0; i < hists.size(); ++i) hists[i][getByte(curr, i)]++;

    bool inStart = true;
    for (size_t i = 0; i < hists.size(); ++i) {
      size_t count = 0;
      // prefix sum for histogram
      bool canSkip = false;
      for (int j = 0; j < 256; ++j) {
        size_t tmp = hists[i][j];
        hists[i][j] = count;
        count += tmp;
        if (tmp == n) {
          canSkip = true;
          break;
        }
      }
      if (canSkip) continue;
      if (inStart) {
        for (size_t j = 0; j < n; ++j)
          dest[hists[i][getByte(start + j, i)]++] = start[j];
      } else {
        for (size_t j = 0; j < n; ++j)
          start[hists[i][getByte(dest + j, i)]++] = dest[j];
      }
      inStart = !inStart;
    }
    // writeback = should be in start
    if (inStart && !writeback) {
      std::copy(start, end, dest);
    } else if (!inStart && writeback) {
      std::copy(dest, dest + n, start);
    }
  }
}

template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type,
          typename Comp = decltype(std::less<T>())>
void mergeSort(ExecutionPolicy policy, Iterator first, Iterator last,
               Comp comp) {
#if (MANIFOLD_PAR == 1)
  if (policy == ExecutionPolicy::Par) {
    // apparently this prioritizes threads inside here?
    tbb::this_task_arena::isolate([&] {
      size_t length = std::distance(first, last);
      T *tmp = new T[length];
      copy(policy, first, last, tmp);
      details::mergeSortRec(tmp, first, 0, length, comp);
      delete[] tmp;
    });
    return;
  }
#endif
  std::stable_sort(first, last, comp);
}

// stable_sort using merge sort.
//
// For simpler implementation, we do not support types that are not trivially
// destructable.
template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type,
          typename Dummy = void>
struct SortFunctor {
  void operator()(ExecutionPolicy policy, Iterator first, Iterator last) {
    static_assert(
        std::is_convertible_v<
            typename std::iterator_traits<Iterator>::iterator_category,
            std::random_access_iterator_tag>,
        "You can only parallelize RandomAccessIterator.");
    static_assert(std::is_trivially_destructible_v<T>,
                  "Our simple implementation does not support types that are "
                  "not trivially destructable.");
    return mergeSort(policy, first, last, std::less<T>());
  }
};
}  // namespace details

#endif

// Applies the function `f` to each element in the range `[first, last)`
template <typename Iter, typename F>
void for_each(ExecutionPolicy policy, Iter first, Iter last, F f) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<Iter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
#if (MANIFOLD_PAR == 1)
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

// Applies the function `f` to each element in the range `[first, last)`
template <typename Iter, typename F>
void for_each_n(ExecutionPolicy policy, Iter first, size_t n, F f) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<Iter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
  for_each(policy, first, first + n, f);
}

// Reduce the range `[first, last)` using a binary operation `f` with an initial
// value `init`.
//
// The binary operation should be commutative and associative. Otherwise, the
// result is non-deterministic.
template <typename InputIter, typename BinaryOp,
          typename T = typename std::iterator_traits<InputIter>::value_type>
T reduce(ExecutionPolicy policy, InputIter first, InputIter last, T init,
         BinaryOp f) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<InputIter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
#if (MANIFOLD_PAR == 1)
  if (policy == ExecutionPolicy::Par) {
    // should we use deterministic reduce here?
    return tbb::parallel_reduce(
        tbb::blocked_range<InputIter>(first, last, details::kSeqThreshold),
        init,
        [&f](const tbb::blocked_range<InputIter> &range, T value) {
          return std::reduce(range.begin(), range.end(), value, f);
        },
        f);
  }
#endif
  return std::reduce(first, last, init, f);
}

// Reduce the range `[first, last)` using a binary operation `f` with an initial
// value `init`.
//
// The binary operation should be commutative and associative. Otherwise, the
// result is non-deterministic.
template <typename InputIter, typename BinaryOp,
          typename T = typename std::iterator_traits<InputIter>::value_type>
T reduce(InputIter first, InputIter last, T init, BinaryOp f) {
  return reduce(autoPolicy(first, last, 1e5), first, last, init, f);
}

// Transform and reduce the range `[first, last)` by first applying a unary
// function `g`, and then combining the results using a binary operation `f`
// with an initial value `init`.
//
// The binary operation should be commutative and associative. Otherwise, the
// result is non-deterministic.
template <typename InputIter, typename BinaryOp, typename UnaryOp,
          typename T = std::invoke_result_t<
              UnaryOp, typename std::iterator_traits<InputIter>::value_type>>
T transform_reduce(ExecutionPolicy policy, InputIter first, InputIter last,
                   T init, BinaryOp f, UnaryOp g) {
  return reduce(policy, TransformIterator(first, g), TransformIterator(last, g),
                init, f);
}

// Transform and reduce the range `[first, last)` by first applying a unary
// function `g`, and then combining the results using a binary operation `f`
// with an initial value `init`.
//
// The binary operation should be commutative and associative. Otherwise, the
// result is non-deterministic.
template <typename InputIter, typename BinaryOp, typename UnaryOp,
          typename T = std::invoke_result_t<
              UnaryOp, typename std::iterator_traits<InputIter>::value_type>>
T transform_reduce(InputIter first, InputIter last, T init, BinaryOp f,
                   UnaryOp g) {
  return manifold::reduce(TransformIterator(first, g),
                          TransformIterator(last, g), init, f);
}

// Compute the inclusive prefix sum for the range `[first, last)`
// using the summation operator, and store the result in the range
// starting from `d_first`.
//
// The input range `[first, last)` and
// the output range `[d_first, d_first + last - first)`
// must be equal or non-overlapping.
template <typename InputIter, typename OutputIter,
          typename T = typename std::iterator_traits<InputIter>::value_type>
void inclusive_scan(ExecutionPolicy policy, InputIter first, InputIter last,
                    OutputIter d_first) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<InputIter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
  static_assert(
      std::is_convertible_v<
          typename std::iterator_traits<OutputIter>::iterator_category,
          std::random_access_iterator_tag>,
      "You can only parallelize RandomAccessIterator.");
#if (MANIFOLD_PAR == 1)
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

// Compute the inclusive prefix sum for the range `[first, last)` using the
// summation operator, and store the result in the range
// starting from `d_first`.
//
// The input range `[first, last)` and
// the output range `[d_first, d_first + last - first)`
// must be equal or non-overlapping.
template <typename InputIter, typename OutputIter,
          typename T = typename std::iterator_traits<InputIter>::value_type>
void inclusive_scan(InputIter first, InputIter last, OutputIter d_first) {
  return inclusive_scan(autoPolicy(first, last, 1e5), first, last, d_first);
}

// Compute the inclusive prefix sum for the range `[first, last)` using the
// binary operator `f`, with initial value `init` and
// identity element `identity`, and store the result in the range
// starting from `d_first`.
//
// This is different from `exclusive_scan` in the sequential algorithm by
// requiring an identity element. This is needed so that each block can be
// scanned in parallel and combined later.
//
// The input range `[first, last)` and
// the output range `[d_first, d_first + last - first)`
// must be equal or non-overlapping.
template <typename InputIter, typename OutputIter,
          typename BinOp = decltype(std::plus<typename std::iterator_traits<
                                        InputIter>::value_type>()),
          typename T = typename std::iterator_traits<InputIter>::value_type>
void exclusive_scan(ExecutionPolicy policy, InputIter first, InputIter last,
                    OutputIter d_first, T init = static_cast<T>(0),
                    BinOp f = std::plus<T>(), T identity = static_cast<T>(0)) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<InputIter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
  static_assert(
      std::is_convertible_v<
          typename std::iterator_traits<OutputIter>::iterator_category,
          std::random_access_iterator_tag>,
      "You can only parallelize RandomAccessIterator.");
#if (MANIFOLD_PAR == 1)
  if (policy == ExecutionPolicy::Par) {
    details::ScanBody<T, InputIter, OutputIter, BinOp> body(init, identity, f,
                                                            first, d_first);
    tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, std::distance(first, last)), body);
    return;
  }
#endif
  std::exclusive_scan(first, last, d_first, init, f);
}

// Compute the inclusive prefix sum for the range `[first, last)` using the
// binary operator `f`, with initial value `init` and
// identity element `identity`, and store the result in the range
// starting from `d_first`.
//
// This is different from `exclusive_scan` in the sequential algorithm by
// requiring an identity element. This is needed so that each block can be
// scanned in parallel and combined later.
//
// The input range `[first, last)` and
// the output range `[d_first, d_first + last - first)`
// must be equal or non-overlapping.
template <typename InputIter, typename OutputIter,
          typename BinOp = decltype(std::plus<typename std::iterator_traits<
                                        InputIter>::value_type>()),
          typename T = typename std::iterator_traits<InputIter>::value_type>
void exclusive_scan(InputIter first, InputIter last, OutputIter d_first,
                    T init = static_cast<T>(0), BinOp f = std::plus<T>(),
                    T identity = static_cast<T>(0)) {
  exclusive_scan(autoPolicy(first, last, 1e5), first, last, d_first, init, f,
                 identity);
}

// Apply function `f` on the input range `[first, last)` and store the result in
// the range starting from `d_first`.
//
// The input range `[first, last)` and
// the output range `[d_first, d_first + last - first)`
// must be equal or non-overlapping.
template <typename InputIter, typename OutputIter, typename F>
void transform(ExecutionPolicy policy, InputIter first, InputIter last,
               OutputIter d_first, F f) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<InputIter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
  static_assert(
      std::is_convertible_v<
          typename std::iterator_traits<OutputIter>::iterator_category,
          std::random_access_iterator_tag>,
      "You can only parallelize RandomAccessIterator.");
#if (MANIFOLD_PAR == 1)
  if (policy == ExecutionPolicy::Par) {
    tbb::parallel_for(tbb::blocked_range<size_t>(
                          0, static_cast<size_t>(std::distance(first, last))),
                      [&](const tbb::blocked_range<size_t> &range) {
                        std::transform(first + range.begin(),
                                       first + range.end(),
                                       d_first + range.begin(), f);
                      });
    return;
  }
#endif
  std::transform(first, last, d_first, f);
}

// Apply function `f` on the input range `[first, last)` and store the result in
// the range starting from `d_first`.
//
// The input range `[first, last)` and
// the output range `[d_first, d_first + last - first)`
// must be equal or non-overlapping.
template <typename InputIter, typename OutputIter, typename F>
void transform(InputIter first, InputIter last, OutputIter d_first, F f) {
  transform(autoPolicy(first, last, 1e5), first, last, d_first, f);
}

// Copy the input range `[first, last)` to the output range
// starting from `d_first`.
//
// The input range `[first, last)` and
// the output range `[d_first, d_first + last - first)`
// must not overlap.
template <typename InputIter, typename OutputIter>
void copy(ExecutionPolicy policy, InputIter first, InputIter last,
          OutputIter d_first) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<InputIter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
  static_assert(
      std::is_convertible_v<
          typename std::iterator_traits<OutputIter>::iterator_category,
          std::random_access_iterator_tag>,
      "You can only parallelize RandomAccessIterator.");
#if (MANIFOLD_PAR == 1)
  if (policy == ExecutionPolicy::Par) {
    tbb::parallel_for(tbb::blocked_range<size_t>(
                          0, static_cast<size_t>(std::distance(first, last)),
                          details::kSeqThreshold),
                      [&](const tbb::blocked_range<size_t> &range) {
                        std::copy(first + range.begin(), first + range.end(),
                                  d_first + range.begin());
                      });
    return;
  }
#endif
  std::copy(first, last, d_first);
}

// Copy the input range `[first, last)` to the output range
// starting from `d_first`.
//
// The input range `[first, last)` and
// the output range `[d_first, d_first + last - first)`
// must not overlap.
template <typename InputIter, typename OutputIter>
void copy(InputIter first, InputIter last, OutputIter d_first) {
  copy(autoPolicy(first, last, 1e6), first, last, d_first);
}

// Copy the input range `[first, first + n)` to the output range
// starting from `d_first`.
//
// The input range `[first, first + n)` and
// the output range `[d_first, d_first + n)`
// must not overlap.
template <typename InputIter, typename OutputIter>
void copy_n(ExecutionPolicy policy, InputIter first, size_t n,
            OutputIter d_first) {
  copy(policy, first, first + n, d_first);
}

// Copy the input range `[first, first + n)` to the output range
// starting from `d_first`.
//
// The input range `[first, first + n)` and
// the output range `[d_first, d_first + n)`
// must not overlap.
template <typename InputIter, typename OutputIter>
void copy_n(InputIter first, size_t n, OutputIter d_first) {
  copy(autoPolicy(n, 1e6), first, first + n, d_first);
}

// Fill the range `[first, last)` with `value`.
template <typename OutputIter, typename T>
void fill(ExecutionPolicy policy, OutputIter first, OutputIter last, T value) {
  static_assert(
      std::is_convertible_v<
          typename std::iterator_traits<OutputIter>::iterator_category,
          std::random_access_iterator_tag>,
      "You can only parallelize RandomAccessIterator.");
#if (MANIFOLD_PAR == 1)
  if (policy == ExecutionPolicy::Par) {
    tbb::parallel_for(tbb::blocked_range<OutputIter>(first, last),
                      [&](const tbb::blocked_range<OutputIter> &range) {
                        std::fill(range.begin(), range.end(), value);
                      });
    return;
  }
#endif
  std::fill(first, last, value);
}

// Fill the range `[first, last)` with `value`.
template <typename OutputIter, typename T>
void fill(OutputIter first, OutputIter last, T value) {
  fill(autoPolicy(first, last, 5e5), first, last, value);
}

// Count the number of elements in the input range `[first, last)` satisfying
// predicate `pred`, i.e. `pred(x) == true`.
template <typename InputIter, typename P>
size_t count_if(ExecutionPolicy policy, InputIter first, InputIter last,
                P pred) {
#if (MANIFOLD_PAR == 1)
  if (policy == ExecutionPolicy::Par) {
    return reduce(policy, TransformIterator(first, pred),
                  TransformIterator(last, pred), 0, std::plus<size_t>());
  }
#endif
  return std::count_if(first, last, pred);
}

// Count the number of elements in the input range `[first, last)` satisfying
// predicate `pred`, i.e. `pred(x) == true`.
template <typename InputIter, typename P>
size_t count_if(InputIter first, InputIter last, P pred) {
  return count_if(autoPolicy(first, last, 1e4), first, last, pred);
}

// Check if all elements in the input range `[first, last)` satisfy
// predicate `pred`, i.e. `pred(x) == true`.
template <typename InputIter, typename P>
bool all_of(ExecutionPolicy policy, InputIter first, InputIter last, P pred) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<InputIter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
#if (MANIFOLD_PAR == 1)
  if (policy == ExecutionPolicy::Par) {
    // should we use deterministic reduce here?
    return tbb::parallel_reduce(
        tbb::blocked_range<InputIter>(first, last), true,
        [&](const tbb::blocked_range<InputIter> &range, bool value) {
          if (!value) return false;
          for (InputIter i = range.begin(); i != range.end(); i++)
            if (!pred(*i)) return false;
          return true;
        },
        [](bool a, bool b) { return a && b; });
  }
#endif
  return std::all_of(first, last, pred);
}

// Check if all elements in the input range `[first, last)` satisfy
// predicate `pred`, i.e. `pred(x) == true`.
template <typename InputIter, typename P>
bool all_of(InputIter first, InputIter last, P pred) {
  return all_of(autoPolicy(first, last, 1e5), first, last, pred);
}

// Copy values in the input range `[first, last)` to the output range
// starting from `d_first` that satisfies the predicate `pred`,
// i.e. `pred(x) == true`, and returns `d_first + n` where `n` is the number of
// times the predicate is evaluated to true.
//
// This function is stable, meaning that the relative order of elements in the
// output range remains unchanged.
//
// The input range `[first, last)` and
// the output range `[d_first, d_first + last - first)`
// must not overlap.
template <typename InputIter, typename OutputIter, typename P>
OutputIter copy_if(ExecutionPolicy policy, InputIter first, InputIter last,
                   OutputIter d_first, P pred) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<InputIter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
  static_assert(
      std::is_convertible_v<
          typename std::iterator_traits<OutputIter>::iterator_category,
          std::random_access_iterator_tag>,
      "You can only parallelize RandomAccessIterator.");
#if (MANIFOLD_PAR == 1)
  if (policy == ExecutionPolicy::Par) {
    auto pred2 = [&](size_t i) { return pred(first[i]); };
    details::CopyIfScanBody body(pred2, first, d_first);
    tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, std::distance(first, last)), body);
    return d_first + body.get_sum();
  }
#endif
  return std::copy_if(first, last, d_first, pred);
}

// Copy values in the input range `[first, last)` to the output range
// starting from `d_first` that satisfies the predicate `pred`, i.e. `pred(x) ==
// true`, and returns `d_first + n` where `n` is the number of times the
// predicate is evaluated to true.
//
// This function is stable, meaning that the relative order of elements in the
// output range remains unchanged.
//
// The input range `[first, last)` and
// the output range `[d_first, d_first + last - first)`
// must not overlap.
template <typename InputIter, typename OutputIter, typename P>
OutputIter copy_if(InputIter first, InputIter last, OutputIter d_first,
                   P pred) {
  return copy_if(autoPolicy(first, last, 1e5), first, last, d_first, pred);
}

// Remove values in the input range `[first, last)` that satisfies
// the predicate `pred`, i.e. `pred(x) == true`, and returns `first + n`
// where `n` is the number of times the predicate is evaluated to false.
//
// This function is stable, meaning that the relative order of elements that
// remained are unchanged.
//
// Only trivially destructable types are supported.
template <typename Iter, typename P,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter remove_if(ExecutionPolicy policy, Iter first, Iter last, P pred) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<Iter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
  static_assert(std::is_trivially_destructible_v<T>,
                "Our simple implementation does not support types that are "
                "not trivially destructable.");
#if (MANIFOLD_PAR == 1)
  if (policy == ExecutionPolicy::Par) {
    T *tmp = new T[std::distance(first, last)];
    auto back =
        copy_if(policy, first, last, tmp, [&](T v) { return !pred(v); });
    copy(policy, tmp, back, first);
    auto d = std::distance(tmp, back);
    delete[] tmp;
    return first + d;
  }
#endif
  return std::remove_if(first, last, pred);
}

// Remove values in the input range `[first, last)` that satisfies
// the predicate `pred`, i.e. `pred(x) == true`, and
// returns `first + n` where `n` is the number of times the predicate is
// evaluated to false.
//
// This function is stable, meaning that the relative order of elements that
// remained are unchanged.
//
// Only trivially destructable types are supported.
template <typename Iter, typename P,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter remove_if(Iter first, Iter last, P pred) {
  return remove_if(autoPolicy(first, last, 1e4), first, last, pred);
}

// Remove values in the input range `[first, last)` that are equal to `value`.
// Returns `first + n` where `n` is the number of values
// that are not equal to `value`.
//
// This function is stable, meaning that the relative order of elements that
// remained are unchanged.
//
// Only trivially destructable types are supported.
template <typename Iter,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter remove(ExecutionPolicy policy, Iter first, Iter last, T value) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<Iter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
  static_assert(std::is_trivially_destructible_v<T>,
                "Our simple implementation does not support types that are "
                "not trivially destructable.");
#if (MANIFOLD_PAR == 1)
  if (policy == ExecutionPolicy::Par) {
    T *tmp = new T[std::distance(first, last)];
    auto back =
        copy_if(policy, first, last, tmp, [&](T v) { return v != value; });
    copy(policy, tmp, back, first);
    auto d = std::distance(tmp, back);
    delete[] tmp;
    return first + d;
  }
#endif
  return std::remove(first, last, value);
}

// Remove values in the input range `[first, last)` that are equal to `value`.
// Returns `first + n` where `n` is the number of values
// that are not equal to `value`.
//
// This function is stable, meaning that the relative order of elements that
// remained are unchanged.
//
// Only trivially destructable types are supported.
template <typename Iter,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter remove(Iter first, Iter last, T value) {
  return remove(autoPolicy(first, last, 1e4), first, last, value);
}

// For each group of consecutive elements in the range `[first, last)` with the
// same value, unique removes all but the first element of the group. The return
// value is an iterator `new_last` such that no two consecutive elements in the
// range `[first, new_last)` are equal.
//
// This function is stable, meaning that the relative order of elements that
// remained are unchanged.
//
// Only trivially destructable types are supported.
template <typename Iter,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter unique(ExecutionPolicy policy, Iter first, Iter last) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<Iter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
  static_assert(std::is_trivially_destructible_v<T>,
                "Our simple implementation does not support types that are "
                "not trivially destructable.");
#if (MANIFOLD_PAR == 1)
  if (policy == ExecutionPolicy::Par && first != last) {
    Iter newSrcStart = first;
    // cap the maximum buffer size, proved to be beneficial for unique with huge
    // array size
    constexpr size_t MAX_BUFFER_SIZE = 1 << 16;
    T *tmp = new T[std::min(MAX_BUFFER_SIZE,
                            static_cast<size_t>(std::distance(first, last)))];
    auto pred = [&](size_t i) { return tmp[i] != tmp[i + 1]; };
    do {
      size_t length =
          std::min(MAX_BUFFER_SIZE,
                   static_cast<size_t>(std::distance(newSrcStart, last)));
      copy(policy, newSrcStart, newSrcStart + length, tmp);
      *first = *newSrcStart;
      // this is not a typo, the index i is offset by 1, so to compare an
      // element with its predecessor we need to compare i and i + 1.
      details::CopyIfScanBody body(pred, tmp + 1, first + 1);
      tbb::parallel_scan(tbb::blocked_range<size_t>(0, length - 1), body);
      first += body.get_sum() + 1;
      newSrcStart += length;
    } while (newSrcStart != last);
    delete[] tmp;
    return first;
  }
#endif
  return std::unique(first, last);
}

// For each group of consecutive elements in the range `[first, last)` with the
// same value, unique removes all but the first element of the group. The return
// value is an iterator `new_last` such that no two consecutive elements in the
// range `[first, new_last)` are equal.
//
// This function is stable, meaning that the relative order of elements that
// remained are unchanged.
//
// Only trivially destructable types are supported.
template <typename Iter,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter unique(Iter first, Iter last) {
  return unique(autoPolicy(first, last, 1e4), first, last);
}

// Sort the input range `[first, last)` in ascending order.
//
// This function is stable, meaning that the relative order of elements that are
// incomparable remains unchanged.
//
// Only trivially destructable types are supported.
template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type>
void stable_sort(ExecutionPolicy policy, Iterator first, Iterator last) {
#if (MANIFOLD_PAR == 1)
  details::SortFunctor<Iterator, T>()(policy, first, last);
#else
  std::stable_sort(first, last);
#endif
}

// Sort the input range `[first, last)` in ascending order.
//
// This function is stable, meaning that the relative order of elements that are
// incomparable remains unchanged.
//
// Only trivially destructable types are supported.
template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type>
void stable_sort(Iterator first, Iterator last) {
  stable_sort(autoPolicy(first, last, 1e4), first, last);
}

// Sort the input range `[first, last)` in ascending order using the comparison
// function `comp`.
//
// This function is stable, meaning that the relative order of elements that are
// incomparable remains unchanged.
//
// Only trivially destructable types are supported.
template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type,
          typename Comp = decltype(std::less<T>())>
void stable_sort(ExecutionPolicy policy, Iterator first, Iterator last,
                 Comp comp) {
#if (MANIFOLD_PAR == 1)
  details::mergeSort(policy, first, last, comp);
#else
  std::stable_sort(first, last, comp);
#endif
}

// Sort the input range `[first, last)` in ascending order using the comparison
// function `comp`.
//
// This function is stable, meaning that the relative order of elements that are
// incomparable remains unchanged.
//
// Only trivially destructable types are supported.
template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type,
          typename Comp = decltype(std::less<T>())>
void stable_sort(Iterator first, Iterator last, Comp comp) {
  stable_sort(autoPolicy(first, last, 1e4), first, last, comp);
}

// `scatter` copies elements from a source range into an output array according
// to a map. For each iterator `i` in the range `[first, last)`, the value `*i`
// is assigned to `outputFirst[mapFirst[i - first]]`.  If the same index appears
// more than once in the range `[mapFirst, mapFirst + (last - first))`, the
// result is undefined.
//
// The map range, input range and the output range must not overlap.
template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator>
void scatter(ExecutionPolicy policy, InputIterator1 first, InputIterator1 last,
             InputIterator2 mapFirst, OutputIterator outputFirst) {
  for_each(policy, countAt(0),
           countAt(static_cast<size_t>(std::distance(first, last))),
           [first, mapFirst, outputFirst](size_t i) {
             outputFirst[mapFirst[i]] = first[i];
           });
}

// `scatter` copies elements from a source range into an output array according
// to a map. For each iterator `i` in the range `[first, last)`, the value `*i`
// is assigned to `outputFirst[mapFirst[i - first]]`. If the same index appears
// more than once in the range `[mapFirst, mapFirst + (last - first))`,
// the result is undefined.
//
// The map range, input range and the output range must not overlap.
template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator>
void scatter(InputIterator1 first, InputIterator1 last, InputIterator2 mapFirst,
             OutputIterator outputFirst) {
  scatter(autoPolicy(first, last, 1e5), first, last, mapFirst, outputFirst);
}

// `gather` copies elements from a source array into a destination range
// according to a map. For each input iterator `i`
// in the range `[mapFirst, mapLast)`, the value `inputFirst[*i]`
// is assigned to `outputFirst[i - map_first]`.
//
// The map range, input range and the output range must not overlap.
template <typename InputIterator, typename RandomAccessIterator,
          typename OutputIterator>
void gather(ExecutionPolicy policy, InputIterator mapFirst,
            InputIterator mapLast, RandomAccessIterator inputFirst,
            OutputIterator outputFirst) {
  for_each(policy, countAt(0),
           countAt(static_cast<size_t>(std::distance(mapFirst, mapLast))),
           [mapFirst, inputFirst, outputFirst](size_t i) {
             outputFirst[i] = inputFirst[mapFirst[i]];
           });
}

// `gather` copies elements from a source array into a destination range
// according to a map. For each input iterator `i`
// in the range `[mapFirst, mapLast)`, the value `inputFirst[*i]`
// is assigned to `outputFirst[i - map_first]`.
//
// The map range, input range and the output range must not overlap.
template <typename InputIterator, typename RandomAccessIterator,
          typename OutputIterator>
void gather(InputIterator mapFirst, InputIterator mapLast,
            RandomAccessIterator inputFirst, OutputIterator outputFirst) {
  gather(autoPolicy(std::distance(mapFirst, mapLast), 1e5), mapFirst, mapLast,
         inputFirst, outputFirst);
}

// Write `[0, last - first)` to the range `[first, last)`.
template <typename Iterator>
void sequence(ExecutionPolicy policy, Iterator first, Iterator last) {
  for_each(policy, countAt(0),
           countAt(static_cast<size_t>(std::distance(first, last))),
           [first](size_t i) { first[i] = i; });
}

// Write `[0, last - first)` to the range `[first, last)`.
template <typename Iterator>
void sequence(Iterator first, Iterator last) {
  sequence(autoPolicy(first, last, 1e5), first, last);
}

template <typename I, typename KeyFn>
void radix_sort_with_key(I start, I end, KeyFn keyfn) {
  using T = std::remove_reference_t<decltype(*start)>;
  T *tmp = new T[std::distance(start, end)];
  details::radix_sort_with_key_rec(
      start, end, tmp, sizeof(std::invoke_result_t<KeyFn, T>) - 1, keyfn, true);
  delete[] tmp;
}

}  // namespace manifold
