// Copyright 2021 Emmett Lalish
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
#include <thrust/execution_policy.h>
#include <thrust/universal_vector.h>

namespace manifold {

/*
 * Host and device vector implementation. This uses `thrust::universal_vector`
 * for storage, so data can be moved by the hardware on demand, allows using
 * more memory than the available GPU memory, reduce memory overhead and provide
 * speedup due to less synchronization.
 *
 * Due to https://github.com/NVIDIA/thrust/issues/1690 , `push_back` operations
 * on universal vectors are *VERY* slow, so a `std::vector` is used as a cache.
 * The cache will be created when we perform `push_back` or `reserve` operations
 * on the `VecDH`, and destroyed when we try to access device iterator/pointer.
 * For better performance, please avoid interspersing `push_back` between device
 * memory accesses, as that will cause repeated synchronization and hurts
 * performance.
 * Note that it is *NOT SAFE* to first obtain a host(device) pointer, perform
 * some device(host) modification, and then read the host(device) pointer again
 * (on the same vector). The memory will be inconsistent in that case.
 */
template <typename T>
class VecDH {
 public:
  VecDH() { impl_ = thrust::universal_vector<T>(); }

  VecDH(int size, T val = T()) {
    impl_.resize(size, val);
    implModified = true;
    cacheModified = false;
  }

  VecDH(const std::vector<T>& vec) {
    cache = vec;
    cacheModified = true;
    implModified = false;
  }

  VecDH(const VecDH<T>& other) {
    if (!other.cacheModified) {
      if (other.impl_.size() > 0)
        impl_ = other.impl_;
      else
        impl_.clear();
      implModified = true;
      cacheModified = false;
    } else {
      cache = other.cache;
      implModified = false;
      cacheModified = true;
    }
  }

  VecDH(VecDH<T>&& other) {
    if (!other.cacheModified) {
      if (other.impl_.size() > 0)
        impl_ = std::move(other.impl_);
      else
        impl_.clear();
      implModified = true;
      cacheModified = false;
    } else {
      cache = std::move(other.cache);
      cacheModified = true;
      implModified = false;
    }
  }

  VecDH<T>& operator=(const VecDH<T>& other) {
    if (!other.cacheModified) {
      if (other.impl_.size() > 0)
        impl_ = other.impl_;
      else
        impl_.clear();
      implModified = true;
      cacheModified = false;
    } else {
      cache = other.cache;
      cacheModified = true;
      implModified = false;
    }
    return *this;
  }

  VecDH<T>& operator=(VecDH<T>&& other) {
    if (!other.cacheModified) {
      if (other.impl_.size() > 0)
        impl_ = std::move(other.impl_);
      else
        impl_.clear();
      implModified = true;
      cacheModified = false;
    } else {
      cache = std::move(other.cache);
      cacheModified = true;
      implModified = false;
    }
    return *this;
  }

  int size() const {
    if (!cacheModified) return impl_.size();
    return cache.size();
  }

  void resize(int newSize, T val = T()) {
    bool shrink = size() > 2 * newSize;
    if (cacheModified) {
      cache.resize(newSize, val);
      if (shrink) cache.shrink_to_fit();
    } else {
      impl_.resize(newSize, val);
      if (shrink) impl_.shrink_to_fit();
      implModified = true;
    }
  }

  void swap(VecDH<T>& other) {
    if (!cacheModified && !other.cacheModified) {
      implModified = true;
      other.implModified = true;
      impl_.swap(other.impl_);
    } else {
      syncCache();
      other.syncCache();
      cacheModified = true;
      implModified = false;
      other.cacheModified = true;
      other.implModified = false;
      cache.swap(other.cache);
    }
  }

  using Iter = typename thrust::universal_vector<T>::iterator;
  using IterC = typename thrust::universal_vector<T>::const_iterator;

  Iter begin() {
    syncImpl();
    implModified = true;
    return impl_.begin();
  }

  Iter end() {
    syncImpl();
    implModified = true;
    return impl_.end();
  }

  IterC cbegin() const {
    syncImpl();
    return impl_.cbegin();
  }

  IterC cend() const {
    syncImpl();
    return impl_.cend();
  }

  IterC begin() const { return cbegin(); }
  IterC end() const { return cend(); }

  T* ptrD() {
    if (size() == 0) return nullptr;
    syncImpl();
    implModified = true;
    return impl_.data().get();
  }

  const T* cptrD() const {
    if (size() == 0) return nullptr;
    syncImpl();
    return impl_.data().get();
  }

  const T* ptrD() const { return cptrD(); }

  T* ptrH() {
    if (size() == 0) return nullptr;
    if (cacheModified) {
      return cache.data();
    } else {
      implModified = true;
      return impl_.data().get();
    }
  }

  const T* cptrH() const {
    if (size() == 0) return nullptr;
    if (cacheModified) {
      return cache.data();
    } else {
      return impl_.data().get();
    }
  }

  const T* ptrH() const { return cptrH(); }

  T& operator[](int i) {
    if (!cacheModified) {
      implModified = true;
      return impl_[i];
    } else {
      cacheModified = true;
      return cache[i];
    }
  }

  const T& operator[](int i) const {
    if (!cacheModified) {
      return impl_[i];
    } else {
      return cache[i];
    }
  }

  T& back() {
    if (!cacheModified) {
      implModified = true;
      return impl_.back();
    } else {
      return cache.back();
    }
  }

  const T& back() const {
    if (!cacheModified) {
      return impl_.back();
    } else {
      return cache.back();
    }
  }

  void push_back(const T& val) {
    syncCache();
    cacheModified = true;
    cache.push_back(val);
  }

  void reserve(int n) {
    syncCache();
    cacheModified = true;
    cache.reserve(n);
  }

  void Dump() const {
    syncCache();
    manifold::Dump(cache);
  }

 private:
  mutable thrust::universal_vector<T> impl_;

  mutable bool implModified = false;
  mutable bool cacheModified = false;
  mutable std::vector<T> cache;

  void syncImpl() const {
    if (cacheModified) {
      impl_ = cache;
      cache.clear();
    }
    cacheModified = false;
  }

  void syncCache() const {
    if (implModified || cache.empty()) {
      cache = std::vector<T>(impl_.begin(), impl_.end());
    }
    implModified = false;
  }
};

template <typename T>
class VecD {
 public:
  VecD(const VecDH<T>& vec) : ptr_(vec.ptrD()), size_(vec.size()) {}

  __host__ __device__ const T& operator[](int i) const { return ptr_[i]; }
  __host__ __device__ int size() const { return size_; }

 private:
  T const* const ptr_;
  const int size_;
};
/** @} */
}  // namespace manifold
