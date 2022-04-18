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
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "structs.h"
#include "utils.cuh"
#include "vec_dh.cuh"

#include <math.h>

namespace manifold {

/** @ingroup Private */
class SparseIndices {
  // COO-style sparse matrix storage. Values corresponding to these indicies are
  // stored in vectors separate from this class, but having the same length.
 public:
  SparseIndices(int size = 0) : p(size), q(size) {}
  typedef typename VecDH<int>::IterD Iter;
  typedef typename thrust::zip_iterator<thrust::tuple<Iter, Iter>> Zip;
  Zip beginDpq() { return zip(p.beginD(), q.beginD()); }
  Zip endDpq() { return zip(p.endD(), q.endD()); }
  Iter beginD(bool use_q) { return use_q ? q.beginD() : p.beginD(); }
  Iter endD(bool use_q) { return use_q ? q.endD() : p.endD(); }
  int* ptrD(bool use_q) { return use_q ? q.ptrD() : p.ptrD(); }
  thrust::pair<int*, int*> ptrDpq(int idx = 0) {
    return thrust::make_pair(p.ptrD() + idx, q.ptrD() + idx);
  }
  const thrust::pair<const int*, const int*> ptrDpq(int idx = 0) const {
    return thrust::make_pair(p.ptrD() + idx, q.ptrD() + idx);
  }
  const VecDH<int>& Get(bool use_q) const { return use_q ? q : p; }
  VecDH<int> Copy(bool use_q) const {
    VecDH<int> out = use_q ? q : p;
    return out;
  }

  typedef typename VecDH<int>::IterDc IterC;
  typedef typename thrust::zip_iterator<thrust::tuple<IterC, IterC>> ZipC;
  ZipC beginDpq() const { return zip(p.beginD(), q.beginD()); }
  ZipC endDpq() const { return zip(p.endD(), q.endD()); }
  IterC beginD(bool use_q) const { return use_q ? q.beginD() : p.beginD(); }
  IterC endD(bool use_q) const { return use_q ? q.endD() : p.endD(); }
  const int* ptrD(bool use_q) const { return use_q ? q.ptrD() : p.ptrD(); }

  typedef typename VecDH<int>::IterHc IterHC;
  typedef typename thrust::zip_iterator<thrust::tuple<IterHC, IterHC>> ZipHC;
  ZipHC beginHpq() const { return zip(p.begin(), q.begin()); }
  ZipHC endHpq() const { return zip(p.end(), q.end()); }

  int size() const { return p.size(); }
  void SwapPQ() { p.swap(q); }

  void Sort() { thrust::sort(beginDpq(), endDpq()); }

  void Resize(int size) {
    p.resize(size, -1);
    q.resize(size, -1);
  }

  void Unique() {
    Sort();
    int newSize = thrust::unique(beginDpq(), endDpq()) - beginDpq();
    Resize(newSize);
  }

  struct firstZero {
    __host__ __device__ bool operator()(thrust::tuple<int, int, int> x) const {
      return thrust::get<0>(x) == 0;
    }
  };

  size_t RemoveZeros(VecDH<int>& S) {
    ALWAYS_ASSERT(S.size() == p.size(), userErr,
                  "Different number of values than indicies!");
    auto zBegin = zip(S.beginD(), beginD(false), beginD(true));
    auto zEnd = zip(S.endD(), endD(false), endD(true));
    size_t size = thrust::remove_if(zBegin, zEnd, firstZero()) - zBegin;
    S.resize(size, -1);
    p.resize(size, -1);
    q.resize(size, -1);
    return size;
  }

  template <typename T>
  struct firstNonFinite {
    __host__ __device__ bool NotFinite(float v) const { return !isfinite(v); }
    __host__ __device__ bool NotFinite(glm::vec2 v) const {
      return !isfinite(v[0]);
    }
    __host__ __device__ bool NotFinite(glm::vec3 v) const {
      return !isfinite(v[0]);
    }
    __host__ __device__ bool NotFinite(glm::vec4 v) const {
      return !isfinite(v[0]);
    }

    __host__ __device__ bool operator()(
        thrust::tuple<T, int, int, int> x) const {
      bool result = NotFinite(thrust::get<0>(x));
      return result;
    }
  };

  template <typename T>
  size_t KeepFinite(VecDH<T>& v, VecDH<int>& x) {
    ALWAYS_ASSERT(x.size() == p.size(), userErr,
                  "Different number of values than indicies!");
    auto zBegin = zip(v.beginD(), x.beginD(), beginD(false), beginD(true));
    auto zEnd = zip(v.endD(), x.endD(), endD(false), endD(true));
    size_t size = thrust::remove_if(zBegin, zEnd, firstNonFinite<T>()) - zBegin;
    v.resize(size);
    x.resize(size, -1);
    p.resize(size, -1);
    q.resize(size, -1);
    return size;
  }

  template <typename Iter, typename T>
  VecDH<T> Gather(const VecDH<T>& val, const Iter pqBegin, const Iter pqEnd,
                  T missingVal) const {
    ALWAYS_ASSERT(val.size() == p.size(), userErr,
                  "Different number of values than indicies!");
    size_t size = pqEnd - pqBegin;
    VecDH<T> result(size);
    VecDH<bool> found(size);
    VecDH<int> temp(size);
    thrust::fill(result.beginD(), result.endD(), missingVal);
    thrust::binary_search(beginDpq(), endDpq(), pqBegin, pqEnd, found.beginD());
    thrust::lower_bound(beginDpq(), endDpq(), pqBegin, pqEnd, temp.beginD());
    thrust::gather_if(temp.beginD(), temp.endD(), found.beginD(), val.beginD(),
                      result.beginD());
    return result;
  }

  void Dump() const {
    const auto& p = Get(0).H();
    const auto& q = Get(1).H();
    std::cout << "SparseIndices = " << std::endl;
    for (int i = 0; i < size(); ++i) {
      std::cout << i << ", p = " << p[i] << ", q = " << q[i] << std::endl;
    }
    std::cout << std::endl;
  }

 private:
  VecDH<int> p, q;
};
}  // namespace manifold