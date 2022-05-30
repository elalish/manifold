#include "thrust/execution_policy.h"
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <cuda_runtime.h>

namespace manifold {
int CUDA_ENABLED = -1;
void check_cuda_available() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  CUDA_ENABLED = device_count != 0;
}
}
#else
namespace manifold {
void check_cuda_available() {}
}
#endif
