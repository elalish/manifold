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
// Metal GPU compute backend for parallel operations on macOS/Apple Silicon.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace manifold {
namespace metal {

// Minimum number of elements to justify GPU dispatch overhead.
// On Apple Silicon with unified memory, the threshold can be lower since
// there's no CPU↔GPU data transfer cost. The main overhead is command
// buffer encoding + kernel launch (~5-15μs).
constexpr size_t kGPUThreshold = 10000;

// Initialize Metal compute backend. Returns true if Metal GPU is available.
// Safe to call multiple times — subsequent calls are no-ops.
bool Initialize();

// Check if Metal backend is available and initialized.
bool IsAvailable();

// Shutdown and release Metal resources.
void Shutdown();

// ---------------------------------------------------------------------------
// GPU-accelerated parallel primitives for typed numeric data
// ---------------------------------------------------------------------------

// Radix sort uint32_t array in-place. Returns true on success.
bool RadixSortUint32(uint32_t* data, size_t count);

// Radix sort key-value pairs where keys are uint32_t and values are int32_t.
// Sorts by key, permuting values accordingly. Returns true on success.
bool RadixSortKeyValue(uint32_t* keys, int32_t* values, size_t count);

// Exclusive prefix scan (sum) on int32_t array.
// output[i] = init + input[0] + input[1] + ... + input[i-1]
// Input and output may alias. Returns true on success.
bool ExclusiveScanInt(const int32_t* input, int32_t* output, size_t count,
                      int32_t init = 0);

// Inclusive prefix scan (sum) on int32_t array.
// output[i] = input[0] + input[1] + ... + input[i]
// Input and output may alias. Returns true on success.
bool InclusiveScanInt(const int32_t* input, int32_t* output, size_t count);

// Inclusive prefix scan (sum) on double array.
bool InclusiveScanDouble(const double* input, double* output, size_t count);

// Parallel reduce (sum) on double array.
double ReduceSumDouble(const double* data, size_t count);

// Parallel reduce (sum) on int32_t array.
int32_t ReduceSumInt(const int32_t* data, size_t count);

// Parallel reduce to find min/max of float array (returns {min, max}).
std::pair<double, double> ReduceMinMaxDouble(const double* data, size_t count);

// Fill int32_t array with sequential values starting from `start`.
bool SequenceFill(int32_t* data, size_t count, int32_t start = 0);

// Scatter: output[indices[i]] = input[i] for i in [0, count)
bool ScatterInt(const int32_t* input, const int32_t* indices, int32_t* output,
                size_t count);

// Gather: output[i] = input[indices[i]] for i in [0, count)
bool GatherInt(const int32_t* input, const int32_t* indices, int32_t* output,
               size_t count);

// Query GPU memory and compute capabilities
struct DeviceInfo {
  const char* name;
  size_t maxBufferLength;       // Maximum single buffer size in bytes
  size_t recommendedMaxMemory;  // Recommended working set size
  uint32_t maxThreadsPerGroup;  // Max threads per threadgroup
  bool hasUnifiedMemory;        // True on Apple Silicon
};
DeviceInfo GetDeviceInfo();

}  // namespace metal
}  // namespace manifold
