#pragma once
#include "structs.h"
#include "utils.cuh"
#include "vec_dh.cuh"

#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace manifold {

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
  thrust::pair<int*, int*> ptrDpq() {
    return thrust::make_pair(p.ptrD(), q.ptrD());
  }
  const thrust::pair<const int*, const int*> ptrDpq() const {
    return thrust::make_pair(p.ptrD(), q.ptrD());
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
  void Swap() { p.swap(q); }

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
    ALWAYS_ASSERT(S.size() == p.size(), runtimeErr,
                  "Different number of values than indicies!");
    auto zBegin = zip(S.beginD(), beginD(false), beginD(true));
    auto zEnd = zip(S.endD(), endD(false), endD(true));
    size_t size = thrust::remove_if(zBegin, zEnd, firstZero()) - zBegin;
    S.resize(size, -1);
    p.resize(size, -1);
    q.resize(size, -1);
    return size;
  }

  template <typename Iter, typename T>
  VecDH<T> Gather(const VecDH<T>& val, const Iter pqBegin, const Iter pqEnd,
                  T missingVal) const {
    ALWAYS_ASSERT(val.size() == p.size(), runtimeErr,
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

 private:
  VecDH<int> p, q;
};
}  // namespace manifold