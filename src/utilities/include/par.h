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
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
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

constexpr size_t kSeqThreshold = 10'000;
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

template <typename Iter>
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

#if MANIFOLD_PAR == 'T'
namespace details {
using manifold::kSeqThreshold;
// implementation from
// https://duvanenko.tech.blog/2018/01/14/parallel-merge/
// https://github.com/DragonSpit/ParallelAlgorithms
// note that the ranges are now [p, r) to fit our convention.
template <typename T, typename Comp>
void mergeRec(T *src, T *dest, size_t p1, size_t r1, size_t p2, size_t r2,
              size_t p3, Comp comp) {
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

template <typename T, typename Comp>
void mergeSortRec(T *src, T *dest, size_t begin, size_t end, Comp comp) {
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

template <typename N, const int K>
struct Hist {
  using SizeType = N;
  static constexpr int k = K;
  N hist[k][256] = {{0}};
  void merge(const Hist<N, K> &other) {
    for (int i = 0; i < k; ++i)
      for (int j = 0; j < 256; ++j) hist[i][j] += other.hist[i][j];
  }
  void prefixSum(N total, bool *canSkip) {
    for (int i = 0; i < k; ++i) {
      size_t count = 0;
      for (int j = 0; j < 256; ++j) {
        N tmp = hist[i][j];
        hist[i][j] = count;
        count += tmp;
        if (tmp == total) canSkip[i] = true;
      }
    }
  }
};

template <typename T, typename H>
void histogram(T *ptr, typename H::SizeType n, H &hist) {
  auto worker = [](T *ptr, typename H::SizeType n, H &hist) {
    for (typename H::SizeType i = 0; i < n; ++i)
      for (int k = 0; k < hist.k; ++k)
        ++hist.hist[k][(ptr[i] >> (8 * k)) & 0xFF];
  };
  if (n < kSeqThreshold) {
    worker(ptr, n, hist);
  } else {
    tbb::combinable<H> store;
    tbb::parallel_for(
        tbb::blocked_range<typename H::SizeType>(0, n, kSeqThreshold),
        [&worker, &store, ptr](const auto &r) {
          worker(ptr + r.begin(), r.end() - r.begin(), store.local());
        });
    store.combine_each([&hist](const H &h) { hist.merge(h); });
  }
}

template <typename T, typename H>
void shuffle(T *src, T *target, typename H::SizeType n, H &hist, int k) {
  for (typename H::SizeType i = 0; i < n; ++i)
    target[hist.hist[k][(src[i] >> (8 * k)) & 0xFF]++] = src[i];
}

template <typename T, typename SizeType>
bool LSB_radix_sort(T *input, T *tmp, SizeType n) {
  Hist<SizeType, sizeof(T) / sizeof(char)> hist;
  if (std::is_sorted(input, input + n)) return false;
  histogram(input, n, hist);
  bool canSkip[hist.k] = {0};
  hist.prefixSum(n, canSkip);
  T *a = input, *b = tmp;
  for (int k = 0; k < hist.k; ++k) {
    if (!canSkip[k]) {
      shuffle(a, b, n, hist, k);
      std::swap(a, b);
    }
  }
  return a == tmp;
}

// LSB radix sort with merge
template <typename T, typename SizeType>
struct SortedRange {
  T *input, *tmp;
  SizeType offset = 0, length = 0;
  bool inTmp = false;

  SortedRange(T *input, T *tmp, SizeType offset = 0, SizeType length = 0)
      : input(input), tmp(tmp), offset(offset), length(length) {}
  SortedRange(SortedRange<T, SizeType> &r, tbb::split)
      : input(r.input), tmp(r.tmp) {}
  void operator()(const tbb::blocked_range<SizeType> &range) {
    SortedRange<T, SizeType> rhs(input, tmp, range.begin(),
                                 range.end() - range.begin());
    rhs.inTmp =
        LSB_radix_sort(input + rhs.offset, tmp + rhs.offset, rhs.length);
    if (length == 0)
      *this = rhs;
    else
      join(rhs);
  }
  bool swapBuffer() const {
    T *src = input, *target = tmp;
    if (inTmp) std::swap(src, target);
    copy(src + offset, src + offset + length, target + offset);
    return !inTmp;
  }
  void join(const SortedRange<T, SizeType> &rhs) {
    if (inTmp != rhs.inTmp) {
      if (length < rhs.length)
        inTmp = swapBuffer();
      else
        rhs.swapBuffer();
    }
    T *src = input, *target = tmp;
    if (inTmp) std::swap(src, target);
    if (src[offset + length - 1] > src[rhs.offset]) {
      mergeRec(src, target, offset, offset + length, rhs.offset,
               rhs.offset + rhs.length, offset, std::less<T>());
      inTmp = !inTmp;
    }
    length += rhs.length;
  }
};

template <typename T, typename SizeTy>
void radix_sort(T *input, SizeTy n) {
  T *aux = new T[n];
  SizeTy blockSize = std::max(n / tbb::this_task_arena::max_concurrency() / 4,
                              static_cast<SizeTy>(kSeqThreshold / sizeof(T)));
  SortedRange<T, SizeTy> result(input, aux);
  tbb::parallel_reduce(tbb::blocked_range<SizeTy>(0, n, blockSize), result);
  if (result.inTmp) copy(aux, aux + n, input);
  delete[] aux;
}

template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type,
          typename Comp = decltype(std::less<T>())>
void mergeSort(ExecutionPolicy policy, Iterator first, Iterator last,
               Comp comp) {
#if MANIFOLD_PAR == 'T'
  if (policy == ExecutionPolicy::Par) {
    // apparently this prioritizes threads inside here?
    tbb::this_task_arena::isolate([&] {
      size_t length = std::distance(first, last);
      T *tmp = new T[length];
      copy(policy, first, last, tmp);
      details::mergeSortRec(tmp, &*first, 0, length, comp);
      delete[] tmp;
    });
    return;
  }
#endif
  std::stable_sort(first, last, comp);
}

template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type,
          typename Dummy = void>
struct SortFunctor {
  void operator()(ExecutionPolicy policy, Iterator first, Iterator last) {
    return mergeSort(policy, first, last, std::less<T>());
  }
};

template <typename Iterator, typename T>
struct SortFunctor<Iterator, T, std::enable_if_t<std::is_integral_v<T>>> {
  void operator()(ExecutionPolicy policy, Iterator first, Iterator last) {
#if MANIFOLD_PAR == 'T'
    if (policy == ExecutionPolicy::Par) {
      radix_sort(&*first, static_cast<size_t>(std::distance(first, last)));
      return;
    }
#endif
    stable_sort(policy, first, last, std::less<T>());
  }
};

}  // namespace details

#endif

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

template <typename InputIter, typename BinaryOp,
          typename T = typename std::iterator_traits<InputIter>::value_type>
T reduce(InputIter first, InputIter last, T init, BinaryOp f) {
  return reduce(autoPolicy(first, last, 100'000), first, last, init, f);
}

template <typename InputIter, typename BinaryOp, typename UnaryOp,
          typename T = std::invoke_result_t<
              UnaryOp, typename std::iterator_traits<InputIter>::value_type>>
T transform_reduce(ExecutionPolicy policy, InputIter first, InputIter last,
                   T init, BinaryOp f, UnaryOp g) {
  return reduce(policy, TransformIterator(first, g), TransformIterator(last, g),
                init, f);
}

template <typename InputIter, typename BinaryOp, typename UnaryOp,
          typename T = std::invoke_result_t<
              UnaryOp, typename std::iterator_traits<InputIter>::value_type>>
T transform_reduce(InputIter first, InputIter last, T init, BinaryOp f,
                   UnaryOp g) {
  return manifold::reduce(TransformIterator(first, g),
                          TransformIterator(last, g), init, f);
}

template <typename InputIter, typename OutputIter,
          typename T = typename std::iterator_traits<InputIter>::value_type>
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

template <typename InputIter, typename OutputIter,
          typename T = typename std::iterator_traits<InputIter>::value_type>
void inclusive_scan(InputIter first, InputIter last, OutputIter d_first) {
  return inclusive_scan(autoPolicy(first, last, 100'000), first, last, d_first);
}

template <typename InputIter, typename OutputIter,
          typename BinOp = decltype(std::plus<typename std::iterator_traits<
                                        InputIter>::value_type>()),
          typename T = typename std::iterator_traits<InputIter>::value_type>
void exclusive_scan(ExecutionPolicy policy, InputIter first, InputIter last,
                    OutputIter d_first, T init = static_cast<T>(0),
                    BinOp f = std::plus<T>(), T identity = static_cast<T>(0)) {
#if MANIFOLD_PAR == 'T'
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

template <typename InputIter, typename OutputIter,
          typename BinOp = decltype(std::plus<typename std::iterator_traits<
                                        InputIter>::value_type>()),
          typename T = typename std::iterator_traits<InputIter>::value_type>
void exclusive_scan(InputIter first, InputIter last, OutputIter d_first,
                    T init = static_cast<T>(0), BinOp f = std::plus<T>(),
                    T identity = static_cast<T>(0)) {
  exclusive_scan(autoPolicy(first, last, 100'000), first, last, d_first, init,
                 f, identity);
}

template <typename InputIter, typename OutputIter, typename F>
void transform(ExecutionPolicy policy, InputIter first, InputIter last,
               OutputIter d_first, F f) {
#if MANIFOLD_PAR == 'T'
  if (policy == ExecutionPolicy::Par) {
    tbb::parallel_for(tbb::blocked_range<size_t>(
                          0_z, static_cast<size_t>(std::distance(first, last))),
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

template <typename InputIter, typename OutputIter, typename F>
void transform(InputIter first, InputIter last, OutputIter d_first, F f) {
  transform(autoPolicy(first, last, 100'000), first, last, d_first, f);
}

template <typename InputIter, typename OutputIter>
void copy(ExecutionPolicy policy, InputIter first, InputIter last,
          OutputIter d_first) {
#if MANIFOLD_PAR == 'T'
  if (policy == ExecutionPolicy::Par) {
    tbb::parallel_for(tbb::blocked_range<size_t>(
                          0_z, static_cast<size_t>(std::distance(first, last)),
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

template <typename InputIter, typename OutputIter>
void copy(InputIter first, InputIter last, OutputIter d_first) {
  copy(autoPolicy(first, last, 1'000'000), first, last, d_first);
}

template <typename InputIter, typename OutputIter>
void copy_n(ExecutionPolicy policy, InputIter first, size_t n,
            OutputIter d_first) {
  copy(policy, first, first + n, d_first);
}

template <typename InputIter, typename OutputIter>
void copy_n(InputIter first, size_t n, OutputIter d_first) {
  copy(autoPolicy(n, 1'000'000), first, first + n, d_first);
}

template <typename OutputIter, typename T>
void fill(ExecutionPolicy policy, OutputIter first, OutputIter last, T value) {
#if MANIFOLD_PAR == 'T'
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

template <typename OutputIter, typename T>
void fill(OutputIter first, OutputIter last, T value) {
  fill(autoPolicy(first, last, 500'000), first, last, value);
}

template <typename InputIter, typename P>
size_t count_if(ExecutionPolicy policy, InputIter first, InputIter last,
                P pred) {
#if MANIFOLD_PAR == 'T'
  if (policy == ExecutionPolicy::Par) {
    return reduce(policy, TransformIterator(first, pred),
                  TransformIterator(last, pred), 0_z, std::plus<size_t>());
  }
#endif
  return std::count_if(first, last, pred);
}

template <typename InputIter, typename P>
size_t count_if(InputIter first, InputIter last, P pred) {
  return count_if(autoPolicy(first, last, 10'000), first, last, pred);
}

template <typename InputIter, typename P>
bool all_of(ExecutionPolicy policy, InputIter first, InputIter last, P pred) {
#if MANIFOLD_PAR == 'T'
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

template <typename InputIter, typename P>
bool all_of(InputIter first, InputIter last, P pred) {
  return all_of(autoPolicy(first, last, 100'000), first, last, pred);
}

// note that you should not have alias between input and output...
// in general it is impossible to check if there is any alias, as the input
// iterator can be computed on-the-fly depending on the output position and may
// not have a pointer
template <typename InputIter, typename OutputIter, typename P>
OutputIter copy_if(ExecutionPolicy policy, InputIter first, InputIter last,
                   OutputIter d_first, P pred) {
#if MANIFOLD_PAR == 'T'
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

template <typename InputIter, typename OutputIter, typename P>
OutputIter copy_if(InputIter first, InputIter last, OutputIter d_first,
                   P pred) {
  return copy_if(autoPolicy(first, last, 10'000), first, last, d_first, pred);
}

template <typename Iter, typename P,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter remove_if(ExecutionPolicy policy, Iter first, Iter last, P pred) {
#if MANIFOLD_PAR == 'T'
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

template <typename Iter, typename P,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter remove_if(Iter first, Iter last, P pred) {
  return remove_if(autoPolicy(first, last, 10'000), first, last, pred);
}

template <typename Iter,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter remove(ExecutionPolicy policy, Iter first, Iter last, T value) {
#if MANIFOLD_PAR == 'T'
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

template <typename Iter,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter remove(Iter first, Iter last, T value) {
  return remove(autoPolicy(first, last, 10'000), first, last, value);
}

template <typename Iter,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter unique(ExecutionPolicy policy, Iter first, Iter last) {
#if MANIFOLD_PAR == 'T'
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

template <typename Iter,
          typename T = typename std::iterator_traits<Iter>::value_type>
Iter unique(Iter first, Iter last) {
  return unique(autoPolicy(first, last, 10'000), first, last);
}

template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type>
void stable_sort(ExecutionPolicy policy, Iterator first, Iterator last) {
#if MANIFOLD_PAR == 'T'
  details::SortFunctor<Iterator, T>()(policy, first, last);
#else
  std::stable_sort(first, last);
#endif
}

template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type>
void stable_sort(Iterator first, Iterator last) {
  stable_sort(autoPolicy(first, last, 10'000), first, last);
}

template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type,
          typename Comp = decltype(std::less<T>())>
void stable_sort(ExecutionPolicy policy, Iterator first, Iterator last,
                 Comp comp) {
#if MANIFOLD_PAR == 'T'
  details::mergeSort(policy, first, last, comp);
#else
  std::stable_sort(first, last, comp);
#endif
}

template <typename Iterator,
          typename T = typename std::iterator_traits<Iterator>::value_type,
          typename Comp = decltype(std::less<T>())>
void stable_sort(Iterator first, Iterator last, Comp comp) {
  stable_sort(autoPolicy(first, last, 10'000), first, last, comp);
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

template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator>
void scatter(InputIterator1 first, InputIterator1 last, InputIterator2 mapFirst,
             OutputIterator outputFirst) {
  scatter(autoPolicy(first, last, 100'000), first, last, mapFirst, outputFirst);
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

template <typename InputIterator, typename RandomAccessIterator,
          typename OutputIterator>
void gather(InputIterator mapFirst, InputIterator mapLast,
            RandomAccessIterator inputFirst, OutputIterator outputFirst) {
  gather(autoPolicy(std::distance(mapFirst, mapLast), 100'000), mapFirst,
         mapLast, inputFirst, outputFirst);
}

template <typename Iterator>
void sequence(ExecutionPolicy policy, Iterator first, Iterator last) {
  for_each(policy, countAt(0_z),
           countAt(static_cast<size_t>(std::distance(first, last))),
           [first](size_t i) { first[i] = i; });
}

template <typename Iterator>
void sequence(Iterator first, Iterator last) {
  sequence(autoPolicy(first, last, 100'000), first, last);
}

}  // namespace manifold
