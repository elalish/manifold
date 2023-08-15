// Copyright 2021 The Manifold Authors.
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
#include <exception>
#if TRACY_ENABLE && TRACY_MEMORY_USAGE
#include <tracy/Tracy.hpp>
#else
#define TracyAllocS(ptr, size, n) (void)0
#define TracyFreeS(ptr, n) (void)0
#endif

// #include "optional_assert.h"
#include "par.h"
#include "public.h"

namespace manifold {

/** @addtogroup Private
 *  @{
 */

/*
 * Specialized vector implementation with multithreaded fill and uninitialized
 * memory optimizations.
 * Note that the constructor and resize function will not perform initialization
 * if the parameter val is not set. Also, this implementation is a toy
 * implementation that did not consider things like non-trivial
 * constructor/destructor, please keep T trivial.
 */
template <typename T>
class VecDH {
 public:
  using Iter = T *;
  using IterC = const T *;

  VecDH() {}

  // Note that the vector constructed with this constructor will contain
  // uninitialized memory. Please specify `val` if you need to make sure that
  // the data is initialized.
  VecDH(int size) {
    reserve(size);
    size_ = size;
  }

  VecDH(int size, T val) { resize(size, val); }

  VecDH(const std::vector<T> &vec) {
    size_ = vec.size();
    capacity_ = size_;
    auto policy = autoPolicy(size_);
    if (size_ != 0) {
      ptr_ = reinterpret_cast<T *>(malloc(size_ * sizeof(T)));
      if (ptr_ == nullptr) throw std::bad_alloc();
      TracyAllocS(ptr_, size_ * sizeof(T), 3);
      uninitialized_copy(policy, vec.begin(), vec.end(), ptr_);
    }
  }

  VecDH(const VecDH<T> &vec) {
    size_ = vec.size_;
    capacity_ = size_;
    auto policy = autoPolicy(size_);
    if (size_ != 0) {
      ptr_ = reinterpret_cast<T *>(malloc(size_ * sizeof(T)));
      if (ptr_ == nullptr) throw std::bad_alloc();
      TracyAllocS(ptr_, size_ * sizeof(T), 3);
      uninitialized_copy(policy, vec.begin(), vec.end(), ptr_);
    }
  }

  VecDH(VecDH<T> &&vec) {
    ptr_ = vec.ptr_;
    size_ = vec.size_;
    capacity_ = vec.capacity_;
    vec.ptr_ = nullptr;
    vec.size_ = 0;
    vec.capacity_ = 0;
  }

  ~VecDH() {
    if (ptr_ != nullptr) {
      TracyFreeS(ptr_, 3);
      free(ptr_);
    }
    ptr_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  VecDH<T> &operator=(const VecDH<T> &other) {
    if (&other == this) return *this;
    if (ptr_ != nullptr) {
      TracyFreeS(ptr_, 3);
      free(ptr_);
    }
    size_ = other.size_;
    capacity_ = other.size_;
    auto policy = autoPolicy(size_);
    if (size_ != 0) {
      ptr_ = reinterpret_cast<T *>(malloc(size_ * sizeof(T)));
      if (ptr_ == nullptr) throw std::bad_alloc();
      TracyAllocS(ptr_, size_ * sizeof(T), 3);
      uninitialized_copy(policy, other.begin(), other.end(), ptr_);
    }
    return *this;
  }

  VecDH<T> &operator=(VecDH<T> &&other) {
    if (&other == this) return *this;
    if (ptr_ != nullptr) {
      TracyFreeS(ptr_, 3);
      free(ptr_);
    }
    size_ = other.size_;
    capacity_ = other.capacity_;
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    return *this;
  }

  int size() const { return size_; }

  void swap(VecDH<T> &other) {
    std::swap(ptr_, other.ptr_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
  }

  Iter begin() { return ptr_; }

  Iter end() { return ptr_ + size_; }

  IterC cbegin() const { return ptr_; }

  IterC cend() const { return ptr_ + size_; }

  IterC begin() const { return cbegin(); }
  IterC end() const { return cend(); }

  T *ptrD() { return ptr_; }

  const T *cptrD() const { return ptr_; }

  const T *ptrD() const { return cptrD(); }

  T *ptrH() { return ptr_; }

  const T *cptrH() const { return ptr_; }

  const T *ptrH() const { return cptrH(); }

  inline T &operator[](int i) {
    if (i < 0 || i >= size_) throw std::out_of_range("VecDH out of range");
    return ptr_[i];
  }

  inline const T &operator[](int i) const {
    if (i < 0 || i >= size_) throw std::out_of_range("VecDH out of range");
    return ptr_[i];
  }

  T &back() {
    if (size_ == 0) throw std::out_of_range("VecDH out of range");
    return ptr_[size_ - 1];
  }

  const T &back() const {
    if (size_ == 0) throw std::out_of_range("VecDH out of range");
    return ptr_[size_ - 1];
  }

  inline void push_back(const T &val) {
    if (size_ >= capacity_) {
      // avoid dangling pointer in case val is a reference of our array
      T val_copy = val;
      reserve(capacity_ == 0 ? 128 : capacity_ * 2);
      ptr_[size_++] = val_copy;
      return;
    }
    ptr_[size_++] = val;
  }

  void reserve(int n) {
    if (n > capacity_) {
      T *newBuffer = reinterpret_cast<T *>(malloc(n * sizeof(T)));
      if (newBuffer == nullptr) throw std::bad_alloc();
      TracyAllocS(newBuffer, n * sizeof(T), 3);
      if (size_ > 0)
        uninitialized_copy(autoPolicy(size_), ptr_, ptr_ + size_, newBuffer);
      if (ptr_ != nullptr) {
        TracyFreeS(ptr_, 3);
        free(ptr_);
      }
      ptr_ = newBuffer;
      capacity_ = n;
    }
  }

  void resize(int newSize, T val = T()) {
    bool shrink = size_ > 2 * newSize;
    reserve(newSize);
    if (size_ < newSize) {
      uninitialized_fill(autoPolicy(newSize - size_), ptr_ + size_,
                         ptr_ + newSize, val);
    }
    size_ = newSize;
    if (shrink) shrink_to_fit();
  }

  void shrink_to_fit() {
    T *newBuffer = nullptr;
    if (size_ > 0) {
      newBuffer = reinterpret_cast<T *>(malloc(size_ * sizeof(T)));
      if (newBuffer == nullptr) throw std::bad_alloc();
      TracyAllocS(newBuffer, size_ * sizeof(T), 3);
      uninitialized_copy(autoPolicy(size_), ptr_, ptr_ + size_, newBuffer);
      if (ptr_ != nullptr) {
        TracyFreeS(ptr_, 3);
        free(ptr_);
      }
    }
    ptr_ = newBuffer;
    capacity_ = size_;
  }

#ifdef MANIFOLD_DEBUG
  void Dump() const {
    std::cout << "VecDH = " << std::endl;
    for (int i = 0; i < size_; ++i) {
      std::cout << i << ", " << ptr_[i] << ", " << std::endl;
    }
    std::cout << std::endl;
  }
#endif

 private:
  int size_ = 0;
  int capacity_ = 0;
  T *ptr_ = nullptr;
};

template <typename T>
class VecDc {
 public:
  VecDc(const VecDH<T> &vec) : ptr_(vec.ptrD()), size_(vec.size()) {}

  __host__ __device__ const T &operator[](int i) const { return ptr_[i]; }
  __host__ __device__ int size() const { return size_; }

 private:
  T const *const ptr_;
  const int size_;
};

template <typename T>
class VecD {
 public:
  VecD(VecDH<T> &vec) : ptr_(vec.ptrD()), size_(vec.size()) {}

  __host__ __device__ T &operator[](int i) const { return ptr_[i]; }
  __host__ __device__ int size() const { return size_; }

 private:
  T *ptr_;
  int size_;
};
/** @} */
}  // namespace manifold
