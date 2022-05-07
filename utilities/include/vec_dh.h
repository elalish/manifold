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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace manifold {

/** @addtogroup Private
 *  @{
 */
template <typename T>
using VecH = thrust::host_vector<T>;

template <typename T>
void Dump(const VecH<T>& vec) {
  std::cout << "VecDH = " << std::endl;
  for (int i = 0; i < vec.size(); ++i) {
    std::cout << i << ", " << vec[i] << ", " << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
class VecDH {
 public:
  VecDH() {}

  VecDH(int size, T val = T()) {
    device_.resize(size, val);
    host_valid_ = false;
  }

  VecDH(const std::vector<T>& vec) {
    host_ = vec;
    device_valid_ = false;
  }

  int size() const { return device_valid_ ? device_.size() : host_.size(); }

  void resize(int newSize, T val = T()) {
    bool shrink = size() > 2 * newSize;
    if (device_valid_) {
      device_.resize(newSize, val);
      if (shrink) device_.shrink_to_fit();
    }
    if (host_valid_) {
      host_.resize(newSize, val);
      if (shrink) host_.shrink_to_fit();
    }
  }

  void swap(VecDH<T>& other) {
    host_.swap(other.host_);
    device_.swap(other.device_);
    thrust::swap(host_valid_, other.host_valid_);
    thrust::swap(device_valid_, other.device_valid_);
  }

  using IterD = typename thrust::device_vector<T>::iterator;
  using IterH = typename thrust::host_vector<T>::iterator;
  using IterDc = typename thrust::device_vector<T>::const_iterator;
  using IterHc = typename thrust::host_vector<T>::const_iterator;

  IterH begin() {
    RefreshHost();
    device_valid_ = false;
    return host_.begin();
  }

  IterH end() {
    RefreshHost();
    device_valid_ = false;
    return host_.end();
  }

  IterHc cbegin() const {
    RefreshHost();
    return host_.cbegin();
  }

  IterHc cend() const {
    RefreshHost();
    return host_.cend();
  }

  IterHc begin() const { return cbegin(); }
  IterHc end() const { return cend(); }

  IterD beginD() {
    RefreshDevice();
    host_valid_ = false;
    return device_.begin();
  }

  IterD endD() {
    RefreshDevice();
    host_valid_ = false;
    return device_.end();
  }

  IterDc cbeginD() const {
    RefreshDevice();
    return device_.cbegin();
  }

  IterDc cendD() const {
    RefreshDevice();
    return device_.cend();
  }

  IterDc beginD() const { return cbeginD(); }
  IterDc endD() const { return cendD(); }

  T* ptrD() {
    if (size() == 0) return nullptr;
    RefreshDevice();
    host_valid_ = false;
    return device_.data().get();
  }

  const T* cptrD() const {
    if (size() == 0) return nullptr;
    RefreshDevice();
    return device_.data().get();
  }

  const T* ptrD() const { return cptrD(); }

  T* ptrH() {
    if (size() == 0) return nullptr;
    RefreshHost();
    device_valid_ = false;
    return host_.data();
  }

  const T* cptrH() const {
    if (size() == 0) return nullptr;
    RefreshHost();
    return host_.data();
  }

  const T* ptrH() const { return cptrH(); }

  const VecH<T>& H() const {
    RefreshHost();
    return host_;
  }

  VecH<T>& H() {
    RefreshHost();
    device_valid_ = false;
    return host_;
  }

  void Dump() const { manifold::Dump(H()); }

 private:
  mutable bool host_valid_ = true;
  mutable bool device_valid_ = true;
  mutable thrust::host_vector<T> host_;
  mutable thrust::device_vector<T> device_;

  void RefreshHost() const {
    if (!host_valid_) {
      host_ = device_;
      host_valid_ = true;
    }
  }

  void RefreshDevice() const {
    if (!device_valid_) {
      device_ = host_;
      device_valid_ = true;
    }
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