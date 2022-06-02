#pragma once
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

#include <thrust/system/cpp/execution_policy.h>

#if MANIFOLD_PAR == 'O'
#include <thrust/system/omp/execution_policy.h>
#define MANIFOLD_PAR_NS omp
#elif MANIFOLD_PAR == 'T'
#include <thrust/system/tbb/execution_policy.h>
#define MANIFOLD_PAR_NS tbb
#else
#define MANIFOLD_PAR_NS cpp
#endif

#ifdef MANIFOLD_USE_CUDA
#include <thrust/system/cuda/execution_policy.h>
#endif

namespace manifold {

void check_cuda_available();
#ifdef MANIFOLD_USE_CUDA
extern int CUDA_ENABLED;
#else
constexpr int CUDA_ENABLED = 0;
#endif

enum class ExecutionPolicy {
  ParUnseq,
  Par,
  Seq,
};

constexpr ExecutionPolicy ParUnseq = ExecutionPolicy::ParUnseq;
constexpr ExecutionPolicy Par = ExecutionPolicy::Par;
constexpr ExecutionPolicy Seq = ExecutionPolicy::Seq;

// ExecutionPolicy:
// - Sequential for small workload,
// - Parallel (CPU) for medium workload,
// - GPU for large workload if available.
inline ExecutionPolicy autoPolicy(int size) {
  // some random numbers
  if (size <= (1 << 12)) {
    return Seq;
  }
  if (size <= (1 << 16) || CUDA_ENABLED != 1) {
    return Par;
  }
  return ParUnseq;
}

#ifdef MANIFOLD_USE_CUDA
#define THRUST_DYNAMIC_BACKEND_VOID(NAME)                                      \
  template <typename... Args>                                                  \
  void NAME(ExecutionPolicy policy, Args... args) {                            \
    switch (policy) {                                                          \
    case ExecutionPolicy::ParUnseq:                                            \
      thrust::NAME(thrust::cuda::par, args...);                                \
      break;                                                                   \
    case ExecutionPolicy::Par:                                                 \
      thrust::NAME(thrust::MANIFOLD_PAR_NS::par, args...);                     \
      break;                                                                   \
    case ExecutionPolicy::Seq:                                                 \
      thrust::NAME(thrust::cpp::par, args...);                                 \
      break;                                                                   \
    }                                                                          \
  }
#define THRUST_DYNAMIC_BACKEND(NAME, RET)                                      \
  template <typename Ret = RET, typename... Args>                              \
  Ret NAME(ExecutionPolicy policy, Args... args) {                             \
    switch (policy) {                                                          \
    case ExecutionPolicy::ParUnseq:                                            \
      return thrust::NAME(thrust::cuda::par, args...);                         \
    case ExecutionPolicy::Par:                                                 \
      return thrust::NAME(thrust::MANIFOLD_PAR_NS::par, args...);              \
    case ExecutionPolicy::Seq:                                                 \
      break;                                                                   \
    }                                                                          \
    return thrust::NAME(thrust::cpp::par, args...);                            \
  }
#else
#define THRUST_DYNAMIC_BACKEND_VOID(NAME)                                      \
  template <typename... Args>                                                  \
  void NAME(ExecutionPolicy policy, Args... args) {                            \
    switch (policy) {                                                          \
    case ExecutionPolicy::ParUnseq:                                            \
    case ExecutionPolicy::Par:                                                 \
      thrust::NAME(thrust::MANIFOLD_PAR_NS::par, args...);                     \
      break;                                                                   \
    case ExecutionPolicy::Seq:                                                 \
      thrust::NAME(thrust::cpp::par, args...);                                 \
      break;                                                                   \
    }                                                                          \
  }

#define THRUST_DYNAMIC_BACKEND(NAME, RET)                                      \
  template <typename Ret = RET, typename... Args>                              \
  Ret NAME(ExecutionPolicy policy, Args... args) {                             \
    switch (policy) {                                                          \
    case ExecutionPolicy::ParUnseq:                                            \
    case ExecutionPolicy::Par:                                                 \
      return thrust::NAME(thrust::MANIFOLD_PAR_NS::par, args...);              \
    case ExecutionPolicy::Seq:                                                 \
      break;                                                                   \
    }                                                                          \
    return thrust::NAME(thrust::cpp::par, args...);                            \
  }
#endif

THRUST_DYNAMIC_BACKEND_VOID(gather)
THRUST_DYNAMIC_BACKEND_VOID(scatter)
THRUST_DYNAMIC_BACKEND_VOID(for_each)
THRUST_DYNAMIC_BACKEND_VOID(for_each_n)
THRUST_DYNAMIC_BACKEND_VOID(sort)
THRUST_DYNAMIC_BACKEND_VOID(stable_sort)
THRUST_DYNAMIC_BACKEND_VOID(fill)
THRUST_DYNAMIC_BACKEND_VOID(sequence)
THRUST_DYNAMIC_BACKEND_VOID(sort_by_key)
THRUST_DYNAMIC_BACKEND_VOID(copy)
THRUST_DYNAMIC_BACKEND_VOID(transform)
THRUST_DYNAMIC_BACKEND_VOID(inclusive_scan)
THRUST_DYNAMIC_BACKEND_VOID(exclusive_scan)
THRUST_DYNAMIC_BACKEND_VOID(uninitialized_fill)
THRUST_DYNAMIC_BACKEND_VOID(uninitialized_copy)

THRUST_DYNAMIC_BACKEND(all_of, bool)
THRUST_DYNAMIC_BACKEND(is_sorted, bool)
THRUST_DYNAMIC_BACKEND(reduce, void)
THRUST_DYNAMIC_BACKEND(count_if, int)
THRUST_DYNAMIC_BACKEND(binary_search, bool)
// void implies that the user have to specify the return type in the template
// argument, as we are unable to deduce it
THRUST_DYNAMIC_BACKEND(remove, void)
THRUST_DYNAMIC_BACKEND(copy_if, void)
THRUST_DYNAMIC_BACKEND(remove_if, void)
THRUST_DYNAMIC_BACKEND(unique, void)
THRUST_DYNAMIC_BACKEND(find, void)
THRUST_DYNAMIC_BACKEND(reduce_by_key, void)
THRUST_DYNAMIC_BACKEND(transform_reduce, void)
THRUST_DYNAMIC_BACKEND(lower_bound, void)
THRUST_DYNAMIC_BACKEND(gather_if, void)

} // namespace manifold
