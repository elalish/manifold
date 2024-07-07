// Copyright 2020 The Manifold Authors, Jared Hoberock and Nathan Bell of
// NVIDIA Research
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
#include <atomic>
#include <iterator>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#ifdef MANIFOLD_DEBUG
#include <chrono>
#include <iostream>
#endif

#include "par.h"
#include "vec.h"

#if __has_include(<tracy/Tracy.hpp>)
#include <tracy/Tracy.hpp>
#else
#define FrameMarkStart(x)
#define FrameMarkEnd(x)
// putting ZoneScoped in a function will instrument the function execution when
// TRACY_ENABLE is set, which allows the profiler to record more accurate
// timing.
#define ZoneScoped
#define ZoneScopedN(name)
#endif

namespace manifold {

/** @defgroup Private
 *  @brief Internal classes of the library; not currently part of the public API
 *  @{
 */
#ifdef MANIFOLD_DEBUG
struct Timer {
  std::chrono::high_resolution_clock::time_point start, end;

  void Start() { start = std::chrono::high_resolution_clock::now(); }

  void Stop() { end = std::chrono::high_resolution_clock::now(); }

  float Elapsed() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
  }
  void Print(std::string message) {
    std::cout << "----------- " << std::round(Elapsed()) << " ms for "
              << message << std::endl;
  }
};
#endif

inline int Next3(int i) {
  constexpr glm::ivec3 next3(1, 2, 0);
  return next3[i];
}

inline int Prev3(int i) {
  constexpr glm::ivec3 prev3(2, 0, 1);
  return prev3[i];
}

template <typename T>
void Permute(Vec<T>& inOut, const Vec<int>& new2Old) {
  Vec<T> tmp(std::move(inOut));
  inOut.resize(new2Old.size());
  gather(autoPolicy(new2Old.size()), new2Old.begin(), new2Old.end(),
         tmp.begin(), inOut.begin());
}

template <typename T>
T AtomicAdd(T& target, T add) {
  std::atomic<T>& tar = reinterpret_cast<std::atomic<T>&>(target);
  T old_val = tar.load();
  while (!tar.compare_exchange_weak(old_val, old_val + add,
                                    std::memory_order_seq_cst)) {
  }
  return old_val;
}

template <>
inline int AtomicAdd(int& target, int add) {
  std::atomic<int>& tar = reinterpret_cast<std::atomic<int>&>(target);
  int old_val = tar.fetch_add(add, std::memory_order_seq_cst);
  return old_val;
}

template <typename T>
class ConcurrentSharedPtr {
 public:
  ConcurrentSharedPtr(T value) : impl(std::make_shared<T>(value)) {}
  ConcurrentSharedPtr(const ConcurrentSharedPtr<T>& other)
      : impl(other.impl), mutex(other.mutex) {}
  class SharedPtrGuard {
   public:
    SharedPtrGuard(std::recursive_mutex* mutex, T* content)
        : mutex(mutex), content(content) {
      mutex->lock();
    }
    ~SharedPtrGuard() { mutex->unlock(); }

    T& operator*() { return *content; }
    T* operator->() { return content; }

   private:
    std::recursive_mutex* mutex;
    T* content;
  };
  SharedPtrGuard GetGuard() { return SharedPtrGuard(mutex.get(), impl.get()); };
  unsigned int UseCount() { return impl.use_count(); };

 private:
  std::shared_ptr<T> impl;
  std::shared_ptr<std::recursive_mutex> mutex =
      std::make_shared<std::recursive_mutex>();
};

template <typename I = int, typename R = unsigned char>
struct UnionFind {
  Vec<I> parents;
  // we do union by rank
  // note that we shift rank by 1, rank 0 means it is not connected to anything
  // else
  Vec<R> ranks;

  UnionFind(I numNodes) : parents(numNodes), ranks(numNodes, 0) {
    sequence(autoPolicy(numNodes), parents.begin(), parents.end());
  }

  I find(I x) {
    while (parents[x] != x) {
      parents[x] = parents[parents[x]];
      x = parents[x];
    }
    return x;
  }

  void unionXY(I x, I y) {
    if (x == y) return;
    if (ranks[x] == 0) ranks[x] = 1;
    if (ranks[y] == 0) ranks[y] = 1;
    x = find(x);
    y = find(y);
    if (x == y) return;
    if (ranks[x] < ranks[y]) std::swap(x, y);
    if (ranks[x] == ranks[y]) ranks[x]++;
    parents[y] = x;
  }

  I connectedComponents(std::vector<I>& components) {
    components.resize(parents.size());
    I lonelyNodes = 0;
    std::unordered_map<I, I> toLabel;
    for (size_t i = 0; i < parents.size(); ++i) {
      // we optimize for connected component of size 1
      // no need to put them into the hashmap
      if (ranks[i] == 0) {
        components[i] = static_cast<I>(toLabel.size()) + lonelyNodes++;
        continue;
      }
      parents[i] = find(i);
      auto iter = toLabel.find(parents[i]);
      if (iter == toLabel.end()) {
        I s = static_cast<I>(toLabel.size()) + lonelyNodes;
        toLabel.insert(std::make_pair(parents[i], s));
        components[i] = s;
      } else {
        components[i] = iter->second;
      }
    }
    return toLabel.size() + lonelyNodes;
  }
};

template <typename Iter, typename = void>
struct InnerIter {
  using pointer = typename std::iterator_traits<Iter>::pointer;
  using reference = typename std::iterator_traits<Iter>::reference;
  using difference_type = typename std::iterator_traits<Iter>::difference_type;
  using value_type = typename std::iterator_traits<Iter>::value_type;
  using iterator_category =
      typename std::iterator_traits<Iter>::iterator_category;
};

template <typename Iter>
struct InnerIter<Iter, typename std::enable_if_t<std::is_pointer_v<Iter>>> {
  using pointer = Iter;
  using reference = std::remove_pointer_t<Iter>&;
  using difference_type = std::ptrdiff_t;
  using value_type = std::remove_pointer_t<Iter>;
  using iterator_category = std::random_access_iterator_tag;
};

template <typename F, typename Iter>
struct TransformIterator {
 private:
  Iter iter;
  F f;

 public:
  // users are not suppposed to take pointer/reference of the iterator.
  using pointer = void;
  using reference = void;
  using difference_type = typename InnerIter<Iter>::difference_type;
  using value_type =
      std::invoke_result_t<F, typename InnerIter<Iter>::value_type>;
  using iterator_category = typename InnerIter<Iter>::iterator_category;

  TransformIterator(Iter iter, F f) : iter(iter), f(f) {}

  value_type operator*() const { return f(*iter); }

  value_type operator[](size_t i) const { return f(iter[i]); }

  TransformIterator& operator++() {
    iter++;
    return *this;
  }

  TransformIterator operator+(size_t n) const {
    return TransformIterator(iter + n, f);
  }

  TransformIterator& operator+=(size_t n) {
    iter += n;
    return *this;
  }

  friend bool operator==(TransformIterator a, TransformIterator b) {
    return a.iter == b.iter;
  }

  friend bool operator!=(TransformIterator a, TransformIterator b) {
    return !(a.iter == b.iter);
  }

  friend bool operator<(TransformIterator a, TransformIterator b) {
    return a.iter < b.iter;
  }

  friend difference_type operator-(TransformIterator a, TransformIterator b) {
    return a.iter - b.iter;
  }

  operator TransformIterator<F, const Iter>() const {
    return TransformIterator(f, iter);
  }
};

template <typename T>
struct CountingIterator {
 private:
  T counter;

 public:
  using pointer = void;
  using reference = T;
  using difference_type = std::make_signed_t<T>;
  using value_type = T;
  using iterator_category = std::random_access_iterator_tag;

  CountingIterator(T counter) : counter(counter) {}

  value_type operator*() const { return counter; }
  value_type operator[](T i) const { return counter + i; }
  CountingIterator& operator++() {
    counter++;
    return *this;
  }
  CountingIterator operator+(T n) const {
    return CountingIterator(counter + n);
  }
  CountingIterator& operator+=(T n) {
    counter += n;
    return *this;
  }
  friend bool operator==(CountingIterator a, CountingIterator b) {
    return a.counter == b.counter;
  }
  friend bool operator!=(CountingIterator a, CountingIterator b) {
    return a.counter != b.counter;
  }
  friend bool operator<(CountingIterator a, CountingIterator b) {
    return a.counter < b.counter;
  }
  friend difference_type operator-(CountingIterator a, CountingIterator b) {
    return a.counter - b.counter;
  }
  operator CountingIterator<const T>() const {
    return CountingIterator(counter);
  }
};

template <typename T>
CountingIterator<T> countAt(T i) {
  return CountingIterator(i);
}

template <typename Iter>
struct StridedRange {
 private:
  struct StridedRangeIter {
   private:
    Iter iter;
    size_t stride;

   public:
    using pointer = void;
    using reference = void;
    using difference_type = typename InnerIter<Iter>::difference_type;
    using value_type = typename InnerIter<Iter>::value_type;
    using iterator_category = typename InnerIter<Iter>::iterator_category;

    StridedRangeIter(Iter iter, int stride) : iter(iter), stride(stride) {}

    value_type& operator*() { return *iter; }

    const value_type& operator*() const { return *iter; }

    value_type& operator[](size_t i) { return iter[i * stride]; }

    const value_type& operator[](size_t i) const { return iter[i * stride]; }

    StridedRangeIter& operator++() {
      iter += stride;
      return *this;
    }

    StridedRangeIter operator+(size_t n) const {
      return StridedRangeIter(iter + n * stride, stride);
    }

    StridedRangeIter& operator+=(size_t n) {
      iter += n * stride;
      return *this;
    }

    friend bool operator==(StridedRangeIter a, StridedRangeIter b) {
      return a.iter == b.iter;
    }

    friend bool operator!=(StridedRangeIter a, StridedRangeIter b) {
      return !(a.iter == b.iter);
    }

    friend bool operator<(StridedRangeIter a, StridedRangeIter b) {
      return a.iter < b.iter;
    }

    friend difference_type operator-(StridedRangeIter a, StridedRangeIter b) {
      // note that this is not well-defined if a.stride != b.stride...
      return (a.iter - b.iter) / a.stride;
    }
  };
  Iter _start, _end;
  const size_t stride;

 public:
  StridedRange(Iter start, Iter end, size_t stride)
      : _start(start), _end(end), stride(stride) {}

  StridedRangeIter begin() const { return StridedRangeIter(_start, stride); }

  StridedRangeIter end() const {
    return StridedRangeIter(_start, stride) +
           ((std::distance(_start, _end) + (stride - 1)) / stride);
  }
};

/** @} */
}  // namespace manifold
