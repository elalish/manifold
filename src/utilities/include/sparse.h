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
#include <math.h>

#include "optional_assert.h"
#include "par.h"
#include "public.h"
#include "utils.h"
#include "vec_dh.h"

namespace manifold {

/** @ingroup Private */
class SparseIndices {
  // sparse indices where {p1: q1, p2: q2, ...} are laid out as
  // p1 q1 p2 q2 or q1 p1 q2 p2, depending on endianness
  // such that the indices are sorted by (p << 32) | q
 public:
#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN ||                 \
    defined(__BIG_ENDIAN__) || defined(__ARMEB__) || defined(__THUMBEB__) || \
    defined(__AARCH64EB__) || defined(_MIBSEB) || defined(__MIBSEB) ||       \
    defined(__MIBSEB__)
  static constexpr size_t pOffset = 0;
#elif defined(__BYTE_ORDER) && __BYTE_ORDER == __LITTLE_ENDIAN ||          \
    defined(__LITTLE_ENDIAN__) || defined(__ARMEL__) ||                    \
    defined(__THUMBEL__) || defined(__AARCH64EL__) || defined(_MIPSEL) ||  \
    defined(__MIPSEL) || defined(__MIPSEL__) || defined(__EMSCRIPTEN__) || \
    defined(_MSC_VER)
  static constexpr size_t pOffset = 1;
#else
#error "unknown architecture"
#endif
  static constexpr int64_t EncodePQ(int p, int q) {
    return (int64_t(p) << 32) | q;
  }

  SparseIndices() = default;
  SparseIndices(size_t size) : data(size) {}

  int size() const { return data.size(); }

  VecDH<int> Copy(bool use_q) const {
    VecDH<int> out(data.size());
    int offset = pOffset;
    if (use_q) offset = 1 - offset;
    const int* p = ptr();
    for_each(autoPolicy(data.size()), countAt(0), countAt((int)data.size()),
             [&](int i) { out[i] = p[i * 2 + offset]; });
    return out;
  }

  void Sort(ExecutionPolicy policy) { sort(policy, data.begin(), data.end()); }

  void Resize(int size) { data.resize(size, -1); }

  inline int& Get(int i, bool use_q) {
    if (use_q)
      return ptr()[2 * i + 1 - pOffset];
    else
      return ptr()[2 * i + pOffset];
  }

  inline int Get(int i, bool use_q) const {
    if (use_q)
      return ptr()[2 * i + 1 - pOffset];
    else
      return ptr()[2 * i + pOffset];
  }

  inline int64_t GetPQ(int i) const { return data[i]; }

  inline void Set(int i, int p, int q) { data[i] = EncodePQ(p, q); }

  inline void SetPQ(int i, int64_t pq) { data[i] = pq; }

  const VecDH<int64_t>& AsVec64() const { return data; }

  inline void Add(int p, int q) { data.push_back(EncodePQ(p, q)); }

  void Unique(ExecutionPolicy policy) {
    Sort(policy);
    int newSize =
        unique<decltype(data.begin())>(policy, data.begin(), data.end()) -
        data.begin();
    Resize(newSize);
  }

  size_t RemoveZeros(VecDH<int>& S) {
    ASSERT(S.size() == data.size(), userErr,
           "Different number of values than indicies!");
    auto zBegin = zip(S.begin(), data.begin());
    auto zEnd = zip(S.end(), data.end());
    size_t size =
        remove_if<decltype(zBegin)>(autoPolicy(S.size()), zBegin, zEnd,
                                    [](thrust::tuple<int, int64_t> x) {
                                      return thrust::get<0>(x) == 0;
                                    }) -
        zBegin;
    S.resize(size, -1);
    Resize(size);
    return size;
  }

  template <typename T>
  struct firstNonFinite {
    bool NotFinite(float v) const { return !isfinite(v); }
    bool NotFinite(glm::vec2 v) const { return !isfinite(v[0]); }
    bool NotFinite(glm::vec3 v) const { return !isfinite(v[0]); }
    bool NotFinite(glm::vec4 v) const { return !isfinite(v[0]); }

    bool operator()(thrust::tuple<T, int, int64_t> x) const {
      bool result = NotFinite(thrust::get<0>(x));
      return result;
    }
  };

  template <typename T>
  size_t KeepFinite(VecDH<T>& v, VecDH<int>& x) {
    ASSERT(x.size() == data.size(), userErr,
           "Different number of values than indicies!");
    auto zBegin = zip(v.begin(), x.begin(), data.begin());
    auto zEnd = zip(v.end(), x.end(), data.end());
    size_t size = remove_if<decltype(zBegin)>(autoPolicy(v.size()), zBegin,
                                              zEnd, firstNonFinite<T>()) -
                  zBegin;
    v.resize(size);
    x.resize(size);
    Resize(size);
    return size;
  }

#ifdef MANIFOLD_DEBUG
  void Dump() const {
    std::cout << "SparseIndices = " << std::endl;
    const int* p = ptr();
    for (int i = 0; i < size(); ++i) {
      std::cout << i << ", p = " << Get(i, false) << ", q = " << Get(i, true)
                << std::endl;
    }
    std::cout << std::endl;
  }
#endif

 private:
  VecDH<int64_t> data;
  inline int* ptr() { return reinterpret_cast<int32_t*>(data.data()); }
  inline const int* ptr() const {
    return reinterpret_cast<const int32_t*>(data.data());
  }
};

}  // namespace manifold
