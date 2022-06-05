#ifdef MANIFOLD_USE_CUDA
#include <cuda_runtime.h>

namespace manifold {
int CUDA_ENABLED = -1;
void check_cuda_available() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  CUDA_ENABLED = device_count != 0;
}
}  // namespace manifold
#else
namespace manifold {
void check_cuda_available() {}
}  // namespace manifold
#endif
