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
//
// rewrite of https://github.com/p-ranav/small_vector
#pragma once
#include <array>
#include <vector>

namespace manifold {

// note that this will not work with non-trivial data (custom
// constructor/destructor)
template <class T, std::size_t N>
class small_vector {
  std::array<T, N> stack_;
  std::vector<T> heap_;
  std::size_t size_{0};

 public:
  using value_type = T;
  using size_type = std::size_t;
  using reference = value_type &;
  using const_reference = const value_type &;
  using pointer = T *;
  using const_pointer = const T *;

  small_vector() = default;

  explicit small_vector(size_type count, const T &value = T()) {
    if (count <= N) {
      std::fill(stack_.begin(), stack_.begin() + count, value);
    } else {
      // use heap
      heap_.resize(count, value);
    }
    size_ = count;
  }

  small_vector(const small_vector &other)
      : stack_(other.stack_), heap_(other.heap_), size_(other.size_) {}

  small_vector(small_vector &&other)
      : stack_(std::move(other.stack_)),
        heap_(std::move(other.heap_)),
        size_(other.size_) {}

  small_vector(std::initializer_list<T> initlist) {
    const auto input_size = initlist.size();
    if (input_size <= N) {
      std::copy(initlist.begin(), initlist.end(), stack_.begin());
    } else {
      std::copy(initlist.begin(), initlist.end(), std::back_inserter(heap_));
    }
    size_ = input_size;
  }

  small_vector &operator=(const small_vector &rhs) {
    stack_ = rhs.stack_;
    heap_ = rhs.heap_;
    size_ = rhs.size_;
    return *this;
  }

  small_vector &operator=(small_vector &&rhs) {
    stack_ = std::move(rhs.stack_);
    heap_ = std::move(rhs.heap_);
    size_ = rhs.size_;
    rhs.size_ = 0;
    return *this;
  }

  small_vector &operator=(std::initializer_list<value_type> rhs) {
    if (rhs.size() <= N) {
      stack_ = rhs;
    } else {
      heap_ = rhs;
    }
    size_ = rhs.size();
  }

  reference at(size_type pos) {
    if (size_ <= N) {
      return stack_.at(pos);
    } else {
      return heap_.at(pos);
    }
  }

  const_reference at(size_type pos) const {
    if (size_ <= N) {
      return stack_.at(pos);
    } else {
      return heap_.at(pos);
    }
  }

  reference operator[](size_type pos) {
    if (size_ <= N) {
      return stack_[pos];
    } else {
      return heap_[pos];
    }
  }

  const_reference operator[](size_type pos) const {
    if (size_ <= N) {
      return stack_[pos];
    } else {
      return heap_[pos];
    }
  }

  reference front() {
    if (size_ <= N) {
      return stack_.front();
    } else {
      return heap_.front();
    }
  }

  const_reference front() const {
    if (size_ <= N) {
      return stack_.front();
    } else {
      return heap_.front();
    }
  }

  reference back() {
    if (size_ <= N) {
      return stack_[size_ - 1];
    } else {
      return heap_[size_ - 1];
    }
  }

  const_reference back() const {
    if (size_ <= N) {
      return stack_[size_ - 1];
    } else {
      return heap_[size_ - 1];
    }
  }

  pointer data() noexcept {
    if (size_ <= N) {
      return stack_.data();
    } else {
      return heap_.data();
    }
  }

  const_pointer data() const noexcept {
    if (size_ <= N) {
      return stack_.data();
    } else {
      return heap_.data();
    }
  }

  bool empty() const { return size_ == 0; }

  size_type size() const { return size_; }

  void shrink_to_fit() {
    if (size_ > N) {
      heap_.shrink_to_fit();
    }
  }

  void push_back(const T &value) {
    if (size_ < N) {
      stack_[size_] = value;
    } else {
      if (size_ == N) {
        std::move(stack_.begin(), stack_.end(), std::back_inserter(heap_));
      }
      heap_.emplace_back(value);
    }
    size_ += 1;
  }

  void pop_back() {
    if (size_ == 0) return;
    if (size_ <= N) {
      size_ -= 1;
    } else {
      // currently using heap
      heap_.pop_back();
      size_ -= 1;
      // now check if all data can fit on stack
      // if so, move back to stack
      if (size_ <= N) {
        std::move(heap_.begin(), heap_.end(), stack_.begin());
        heap_.clear();
      }
    }
  }

  // Resizes the container to contain count elements.
  void resize(size_type count, T value = T()) {
    if (count <= N) {
      // new `count` of elements completely fit on stack
      if (size_ >= N) {
        // currently, all data on heap
        // move back to stack
        std::move(heap_.begin(), heap_.end(), stack_.begin());
      } else {
        // all data already on stack
        // just update size
      }
    } else {
      // new `count` of data is going to be on the heap
      // check if data is currently on the stack
      if (size_ <= N) {
        // move to heap
        std::move(stack_.begin(), stack_.end(), std::back_inserter(heap_));
      }
      heap_.resize(count, value);
    }
    size_ = count;
  }

  void clear() {
    if (size_ > N) {
      heap_.clear();
    }
    size_ = 0;
  }

  void swap(small_vector &other) noexcept {
    std::swap(stack_, other.stack_);
    std::swap(heap_, other.heap_);
    std::swap(size_, other.size_);
  };

  class iterator {
   public:
    using self_type = iterator;
    using value_type = T;
    using reference = T &;
    using pointer = T *;
    using difference_type = int;
    using iterator_category = std::bidirectional_iterator_tag;
    iterator(pointer ptr) : ptr_(ptr) {}
    self_type operator++() {
      ptr_++;
      return *this;
    }
    self_type operator++(int) {
      self_type i = *this;
      ptr_++;
      return i;
    }
    self_type operator--() {
      ptr_--;
      return *this;
    }
    self_type operator--(int) {
      self_type i = *this;
      ptr_--;
      return i;
    }
    self_type operator+(size_type i) const { return self_type(ptr_ + i); }
    self_type operator-(size_type i) const { return self_type(ptr_ - i); }
    reference operator*() { return *ptr_; }
    const value_type &operator*() const { return *ptr_; }
    pointer operator->() { return ptr_; }
    const pointer operator->() const { return ptr_; }
    bool operator==(const self_type &rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const self_type &rhs) const { return ptr_ != rhs.ptr_; }

   private:
    pointer ptr_;
  };

  class const_iterator {
   public:
    using self_type = const_iterator;
    using value_type = T;
    using reference = const T &;
    using pointer = const T *;
    using difference_type = int;
    using iterator_category = std::bidirectional_iterator_tag;
    const_iterator(pointer ptr) : ptr_(ptr) {}
    self_type operator++() {
      ptr_++;
      return *this;
    }
    self_type operator++(int) {
      self_type i = *this;
      ptr_++;
      return i;
    }
    self_type operator--() {
      ptr_--;
      return *this;
    }
    self_type operator--(int) {
      self_type i = *this;
      ptr_--;
      return i;
    }
    self_type operator+(size_type i) const { return self_type(ptr_ + i); }
    self_type operator-(size_type i) const { return self_type(ptr_ - i); }
    reference operator*() const { return *ptr_; }
    pointer operator->() const { return ptr_; }
    bool operator==(const self_type &rhs) const { return ptr_ == rhs.ptr_; }
    bool operator!=(const self_type &rhs) const { return ptr_ != rhs.ptr_; }

   private:
    pointer ptr_;
  };

  iterator begin() {
    if (size_ <= N) {
      return iterator(stack_.data());
    } else {
      return iterator(heap_.data());
    }
  }

  iterator end() {
    if (size_ <= N) {
      return iterator(stack_.data() + size_);
    } else {
      return iterator(heap_.data() + size_);
    }
  }

  const_iterator cbegin() const {
    if (size_ <= N) {
      return const_iterator(stack_.data());
    } else {
      return const_iterator(heap_.data());
    }
  }

  const_iterator cend() const {
    if (size_ <= N) {
      return const_iterator(stack_.data() + size_);
    } else {
      return const_iterator(heap_.data() + size_);
    }
  }

  const_iterator begin() const { return cbegin(); }

  const_iterator end() const { return cend(); }

  void erase(iterator iter) {
    size_type i = std::distance(begin(), iter);
    if (size_ <= N) {
      std::move_backward(stack_.begin() + i + 1, stack_.begin() + size_,
                         stack_.begin() + i);
    } else {
      heap_.erase(heap_.begin() + i);
    }
    size_ -= 1;
  }

  void erase(const_iterator iter) {
    erase(begin() + std::distance(cbegin(), iter));
  }

  void insert(iterator iter, const T &value) {
    size_type i = std::distance(begin(), iter);
    if (size_ < N) {
      if (i + 1 < size_)
        std::move_backward(stack_.begin() + i, stack_.begin() + size_,
                           stack_.begin() + i + 1);
      stack_[i] = value;
    } else {
      if (size_ == N)
        std::move(stack_.begin(), stack_.end(), std::back_inserter(heap_));
      heap_.insert(heap_.begin() + i, value);
    }
    size_ += 1;
  }

  void insert(const_iterator iter, const T &value) {
    insert(begin() + std::distance(cbegin(), iter), value);
  }
};

}  // namespace manifold
