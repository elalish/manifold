// Copyright 2022 The Manifold Authors.
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

#ifdef MANIFOLD_USE_CUDA
#include <cuda_runtime.h>

namespace {
int CUDA_DEVICES = -1;
}
namespace manifold {

bool CudaEnabled() {
  if (CUDA_DEVICES >= 0) return CUDA_DEVICES > 0;

  cudaError_t error = cudaGetDeviceCount(&CUDA_DEVICES);
  if (error != cudaSuccess) CUDA_DEVICES = 0;

  return CUDA_DEVICES > 0;
}
}  // namespace manifold
#else
namespace manifold {
bool CudaEnabled() { return false; }
}  // namespace manifold
#endif
