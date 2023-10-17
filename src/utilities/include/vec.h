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
#include "tracy/Tracy.hpp"
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
template <typename T>
class Vec;

/**
 * View for Vec, can perform offset operation.
 * This will be invalidated when the original vector is dropped or changes
 * length.
 */
template <typename T>
class VecView {
 public:
  using Iter = T *;
  using IterC = const T *;

  VecView(T *ptr_, int size_) : ptr_(ptr_), size_(size_) {}

  VecView(const VecView &other) {
    ptr_ = other.ptr_;
    size_ = other.size_;
  }

  VecView &operator=(const VecView &other) {
    ptr_ = other.ptr_;
    size_ = other.size_;
    return *this;
  }

  // allows conversion to a const VecView
  operator VecView<const T>() const { return {ptr_, size_}; }

  inline const T &operator[](int i) const {
    if (i < 0 || i >= size_) throw std::out_of_range("Vec out of range");
    return ptr_[i];
  }

  inline T &operator[](int i) {
    if (i < 0 || i >= size_) throw std::out_of_range("Vec out of range");
    return ptr_[i];
  }

  IterC cbegin() const { return ptr_; }
  IterC cend() const { return ptr_ + size_; }

  IterC begin() const { return cbegin(); }
  IterC end() const { return cend(); }

  Iter begin() { return ptr_; }
  Iter end() { return ptr_ + size_; }

  const T &front() const {
    if (size_ == 0)
      throw std::out_of_range("attempt to take the front of an empty vector");
    return ptr_[0];
  }

  const T &back() const {
    if (size_ == 0)
      throw std::out_of_range("attempt to take the back of an empty vector");
    return ptr_[size_ - 1];
  }

  T &front() {
    if (size_ == 0)
      throw std::out_of_range("attempt to take the front of an empty vector");
    return ptr_[0];
  }

  T &back() {
    if (size_ == 0)
      throw std::out_of_range("attempt to take the back of an empty vector");
    return ptr_[size_ - 1];
  }

  int size() const { return size_; }

  bool empty() const { return size_ == 0; }

#ifdef MANIFOLD_DEBUG
  void Dump() {
    std::cout << "Vec = " << std::endl;
    for (int i = 0; i < size(); ++i) {
      std::cout << i << ", " << ptr_[i] << ", " << std::endl;
    }
    std::cout << std::endl;
  }
#endif

 protected:
  T *ptr_ = nullptr;
  int size_ = 0;

  VecView() = default;
  friend class Vec<T>;
  friend class Vec<typename std::remove_const<T>::type>;
  friend class VecView<typename std::remove_const<T>::type>;
};

/*
 * Specialized vector implementation with multithreaded fill and uninitialized
 * memory optimizations.
 * Note that the constructor and resize function will not perform initialization
 * if the parameter val is not set. Also, this implementation is a toy
 * implementation that did not consider things like non-trivial
 * constructor/destructor, please keep T trivial.
 */
template <typename T>
class Vec : public VecView<T> {
 public:
  Vec() {}

  // Note that the vector constructed with this constructor will contain
  // uninitialized memory. Please specify `val` if you need to make sure that
  // the data is initialized.
  Vec(int size) {
    reserve(size);
    this->size_ = size;
  }

  Vec(int size, T val) { resize(size, val); }

  Vec(const Vec<T> &vec) {
    this->size_ = vec.size();
    this->capacity_ = this->size_;
    auto policy = autoPolicy(this->size_);
    if (this->size_ != 0) {
      this->ptr_ = reinterpret_cast<T *>(malloc(this->size_ * sizeof(T)));
      if (this->ptr_ == nullptr) throw std::bad_alloc();
      TracyAllocS(ptr_, size_ * sizeof(T), 3);
      uninitialized_copy(policy, vec.begin(), vec.end(), this->ptr_);
    }
  }

  Vec(const std::vector<T> &vec) {
    this->size_ = vec.size();
    this->capacity_ = this->size_;
    auto policy = autoPolicy(this->size_);
    if (this->size_ != 0) {
      this->ptr_ = reinterpret_cast<T *>(malloc(this->size_ * sizeof(T)));
      if (this->ptr_ == nullptr) throw std::bad_alloc();
      TracyAllocS(ptr_, size_ * sizeof(T), 3);
      uninitialized_copy(policy, vec.begin(), vec.end(), this->ptr_);
    }
  }

  Vec(Vec<T> &&vec) {
    this->ptr_ = vec.ptr_;
    this->size_ = vec.size_;
    capacity_ = vec.capacity_;
    vec.ptr_ = nullptr;
    vec.size_ = 0;
    vec.capacity_ = 0;
  }

  operator VecView<T>() { return {this->ptr_, this->size_}; }
  operator VecView<const T>() const { return {this->ptr_, this->size_}; }

  ~Vec() {
    if (this->ptr_ != nullptr) {
      TracyFreeS(ptr_, 3);
      free(this->ptr_);
    }
    this->ptr_ = nullptr;
    this->size_ = 0;
    capacity_ = 0;
  }

  Vec<T> &operator=(const Vec<T> &other) {
    if (&other == this) return *this;
    if (this->ptr_ != nullptr) {
      TracyFreeS(this->ptr_, 3);
      free(this->ptr_);
    }
    this->size_ = other.size_;
    capacity_ = other.size_;
    auto policy = autoPolicy(this->size_);
    if (this->size_ != 0) {
      this->ptr_ = reinterpret_cast<T *>(malloc(this->size_ * sizeof(T)));
      if (this->ptr_ == nullptr) throw std::bad_alloc();
      TracyAllocS(ptr_, size_ * sizeof(T), 3);
      uninitialized_copy(policy, other.begin(), other.end(), this->ptr_);
    }
    return *this;
  }

  Vec<T> &operator=(Vec<T> &&other) {
    if (&other == this) return *this;
    if (this->ptr_ != nullptr) {
      TracyFreeS(ptr_, 3);
      free(this->ptr_);
    }
    this->size_ = other.size_;
    capacity_ = other.capacity_;
    this->ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    return *this;
  }

  operator VecView<T>() const { return {this->ptr_, this->size_}; }

  void swap(Vec<T> &other) {
    std::swap(this->ptr_, other.ptr_);
    std::swap(this->size_, other.size_);
    std::swap(capacity_, other.capacity_);
  }

  inline void push_back(const T &val) {
    if (this->size_ >= capacity_) {
      // avoid dangling pointer in case val is a reference of our array
      T val_copy = val;
      reserve(capacity_ == 0 ? 128 : capacity_ * 2);
      this->ptr_[this->size_++] = val_copy;
      return;
    }
    this->ptr_[this->size_++] = val;
  }

  void reserve(int n) {
    if (n > capacity_) {
      T *newBuffer = reinterpret_cast<T *>(malloc(n * sizeof(T)));
      if (newBuffer == nullptr) throw std::bad_alloc();
      TracyAllocS(newBuffer, n * sizeof(T), 3);
      if (this->size_ > 0)
        uninitialized_copy(autoPolicy(this->size_), this->ptr_,
                           this->ptr_ + this->size_, newBuffer);
      if (this->ptr_ != nullptr) {
        TracyFreeS(ptr_, 3);
        free(this->ptr_);
      }
      this->ptr_ = newBuffer;
      capacity_ = n;
    }
  }

  void resize(int newSize, T val = T()) {
    bool shrink = this->size_ > 2 * newSize;
    reserve(newSize);
    if (this->size_ < newSize) {
      uninitialized_fill(autoPolicy(newSize - this->size_),
                         this->ptr_ + this->size_, this->ptr_ + newSize, val);
    }
    this->size_ = newSize;
    if (shrink) shrink_to_fit();
  }

  void shrink_to_fit() {
    T *newBuffer = nullptr;
    if (this->size_ > 0) {
      newBuffer = reinterpret_cast<T *>(malloc(this->size_ * sizeof(T)));
      if (newBuffer == nullptr) throw std::bad_alloc();
      TracyAllocS(newBuffer, size_ * sizeof(T), 3);
      uninitialized_copy(autoPolicy(this->size_), this->ptr_,
                         this->ptr_ + this->size_, newBuffer);
    }
    if (this->ptr_ != nullptr) {
      TracyFreeS(ptr_, 3);
      free(this->ptr_);
    }
    this->ptr_ = newBuffer;
    capacity_ = this->size_;
  }

  VecView<T> view(int offset = 0, int length = -1) {
    if (length == -1) {
      length = this->size_ - offset;
      if (length < 0) throw std::out_of_range("Vec::view out of range");
    } else if (offset + length > this->size_ || offset < 0) {
      throw std::out_of_range("Vec::view out of range");
    } else if (length < 0) {
      throw std::out_of_range("Vec::view negative length is not allowed");
    }
    return VecView<T>(this->ptr_ + offset, length);
  }

  VecView<const T> cview(int offset = 0, int length = -1) const {
    if (length == -1) {
      length = this->size_ - offset;
      if (length < 0) throw std::out_of_range("Vec::cview out of range");
    } else if (offset + length > this->size_ || offset < 0) {
      throw std::out_of_range("Vec::cview out of range");
    } else if (length < 0) {
      throw std::out_of_range("Vec::cview negative length is not allowed");
    }
    return VecView<const T>(this->ptr_ + offset, length);
  }

  VecView<const T> view(int offset = 0, int length = -1) const {
    return cview(offset, length);
  }

  T *data() { return this->ptr_; }
  const T *data() const { return this->ptr_; }

 private:
  int capacity_ = 0;
};
/** @} */
}  // namespace manifold
