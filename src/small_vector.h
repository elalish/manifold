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
  using iterator = T *;
  using const_iterator = const T *;

  small_vector() = default;

  explicit small_vector(size_type count, const T &value = T()) {
    if (count <= N) {
      std::uninitialized_fill_n(stack_.begin(), count, value);
    } else {
      heap_ = std::vector<T>(count, value);
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
      std::uninitialized_copy(initlist.begin(), initlist.end(), stack_.begin());
    } else {
      std::copy(initlist.begin(), initlist.end(), std::back_inserter(heap_));
    }
    size_ = input_size;
  }

  small_vector &operator=(const small_vector &rhs) {
    if (this == &rhs) return *this;
    if (size_ <= N) clear();  // clear initialized data
    if (rhs.size_ <= N) {
      std::uninitialized_copy(rhs.begin(), rhs.end(), stack_.begin());
    } else {
      heap_ = rhs.heap_;
    }
    size_ = rhs.size_;
    return *this;
  }

  small_vector &operator=(small_vector &&rhs) {
    if (this == &rhs) return *this;
    if (size_ <= N) clear();  // clear initialized data
    if (rhs.size_ <= N) {
      std::uninitialized_move(rhs.begin(), rhs.end(), stack_.begin());
    } else {
      heap_ = std::move(rhs.heap_);
    }
    size_ = rhs.size_;
    rhs.size_ = 0;
    return *this;
  }

  small_vector &operator=(std::initializer_list<value_type> rhs) {
    if (size_ <= N) clear();  // clear initialized data
    if (rhs.size() <= N) {
      std::uninitialized_copy(rhs.begin(), rhs.end(), stack_.begin());
    } else {
      heap_ = rhs;
    }
    size_ = rhs.size();
  }

  reference at(size_type pos) {
    return size_ <= N ? stack_.at(pos) : heap_.at(pos);
  }

  const_reference at(size_type pos) const {
    return size_ <= N ? stack_.at(pos) : heap_.at(pos);
  }

  reference operator[](size_type pos) {
    return size_ <= N ? stack_[pos] : heap_[pos];
  }

  const_reference operator[](size_type pos) const {
    return size_ <= N ? stack_[pos] : heap_[pos];
  }

  reference front() { return size_ <= N ? stack_.front() : heap_.front(); }

  const_reference front() const {
    return size_ <= N ? stack_.front() : heap_.front();
  }

  reference back() { return size_ <= N ? stack_[size_ - 1] : heap_.back(); }

  const_reference back() const {
    return size_ <= N ? stack_[size_ - 1] : heap_.back();
  }

  pointer data() noexcept { return size_ <= N ? stack_.data() : heap_.data(); }

  const_pointer data() const noexcept {
    return size_ <= N ? stack_.data() : heap_.data();
  }

  bool empty() const { return size_ == 0; }

  size_type size() const { return size_; }

  void shrink_to_fit() {
    if (size_ > N) heap_.shrink_to_fit();
  }

  void push_back(const T &value) {
    if (size_ < N) {
      stack_[size_] = value;
    } else {
      if (size_ == N)
        std::move(stack_.begin(), stack_.end(), std::back_inserter(heap_));
      heap_.emplace_back(value);
    }
    size_ += 1;
  }

  void pop_back() {
    if (size_ == 0) return;
    if (size_ <= N) {
      size_ -= 1;
      std::destroy_at(&stack_[size_]);
    } else {
      // currently using heap
      heap_.pop_back();
      size_ -= 1;
      // now check if all data can fit on stack
      // if so, move back to stack
      if (size_ <= N) {
        std::uninitialized_move(heap_.begin(), heap_.end(), stack_.begin());
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
        std::uninitialized_move(heap_.begin(), heap_.begin() + count,
                                stack_.begin());
        heap_.clear();
      } else {
        if (size_ < count)
          std::uninitialized_fill(stack_.begin() + size_,
                                  stack_.begin() + count, value);
        else if (count < size_)
          std::destroy(stack_.begin() + count, stack_.begin() + size_);
      }
    } else {
      // new `count` of data is going to be on the heap
      // check if data is currently on the stack
      if (size_ <= N)
        std::move(stack_.begin(), stack_.begin() + size_,
                  std::back_inserter(heap_));
      heap_.resize(count, value);
    }
    size_ = count;
  }

  void clear() {
    if (size_ > N) {
      heap_.clear();
    } else {
      std::destroy(begin(), end());
    }
    size_ = 0;
  }

  void swap(small_vector &other) noexcept {
    std::swap(stack_, other.stack_);
    std::swap(heap_, other.heap_);
    std::swap(size_, other.size_);
  };

  iterator begin() { return size_ <= N ? stack_.data() : heap_.data(); }

  iterator end() { return (size_ <= N ? stack_.data() : heap_.data()) + size_; }

  const_iterator cbegin() const {
    return size_ <= N ? stack_.data() : heap_.data();
  }

  const_iterator cend() const {
    return (size_ <= N ? stack_.data() : heap_.data()) + size_;
  }

  const_iterator begin() const { return cbegin(); }

  const_iterator end() const { return cend(); }

  void erase(iterator iter) {
    if (size_ == 0) return;
    size_type i = std::distance(begin(), iter);
    if (size_ <= N) {
      if (i < size_ - 1)
        // probably need a custom loop if we want to work with non-trivial data
        // type
        std::move(stack_.begin() + i + 1, stack_.begin() + size_,
                  stack_.begin() + i);
    } else {
      heap_.erase(heap_.begin() + i);
      if (size_ == N + 1) {
        std::uninitialized_move(heap_.begin(), heap_.end(), stack_.begin());
        heap_.clear();
      }
    }
    size_ -= 1;
  }

  void erase(const_iterator iter) {
    erase(begin() + std::distance(cbegin(), iter));
  }

  void insert(iterator iter, const T &value) {
    size_type i = std::distance(begin(), iter);
    if (size_ < N) {
      if (i < size_)
        // probably need a custom loop if we want to work with non-trivial data
        // type
        std::move_backward(stack_.begin() + i, stack_.begin() + size_,
                           stack_.begin() + size_ + 1);
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
