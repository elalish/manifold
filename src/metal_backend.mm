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
// Metal GPU compute backend implementation for macOS/Apple Silicon.
// Uses unified memory architecture for zero-copy GPU dispatch.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_backend.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

namespace manifold {
namespace metal {

// ---------------------------------------------------------------------------
// Metal backend singleton state
// ---------------------------------------------------------------------------

namespace {

struct MetalState {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> commandQueue = nil;
  id<MTLLibrary> library = nil;

  // Pre-compiled pipeline states for each kernel
  id<MTLComputePipelineState> radixHistogramPSO = nil;
  id<MTLComputePipelineState> radixPrefixSumPSO = nil;
  id<MTLComputePipelineState> radixScatterKVPSO = nil;
  id<MTLComputePipelineState> exclusiveScanIntBlockPSO = nil;
  id<MTLComputePipelineState> scanAddBlockOffsetsPSO = nil;
  id<MTLComputePipelineState> inclusiveScanIntBlockPSO = nil;
  id<MTLComputePipelineState> inclusiveScanDoubleBlockPSO = nil;
  id<MTLComputePipelineState> scanAddBlockOffsetsDoublePSO = nil;
  id<MTLComputePipelineState> reduceSumIntPSO = nil;
  id<MTLComputePipelineState> reduceSumFloatPSO = nil;
  id<MTLComputePipelineState> reduceMinMaxFloatPSO = nil;
  id<MTLComputePipelineState> sequenceFillPSO = nil;
  id<MTLComputePipelineState> scatterIntPSO = nil;
  id<MTLComputePipelineState> gatherIntPSO = nil;
  id<MTLComputePipelineState> computeMortonCodesPSO = nil;
  id<MTLComputePipelineState> computeFaceBoxMortonPSO = nil;
  id<MTLComputePipelineState> reindexFacesPSO = nil;
  id<MTLComputePipelineState> reindexVertsInHalfedgesPSO = nil;
  id<MTLComputePipelineState> markUsedPropsPSO = nil;
  id<MTLComputePipelineState> compactPropertiesPSO = nil;
  id<MTLComputePipelineState> updatePropVertsPSO = nil;

  bool initialized = false;
  uint32_t maxThreadsPerGroup = 256;
  char deviceName[256] = {};
};

MetalState g_metal;
std::once_flag g_initFlag;

id<MTLComputePipelineState> CreatePipeline(const char* functionName) {
  NSString* name = [NSString stringWithUTF8String:functionName];
  id<MTLFunction> function = [g_metal.library newFunctionWithName:name];
  if (!function) {
    NSLog(@"[Manifold Metal] Failed to find kernel function: %s", functionName);
    return nil;
  }

  NSError* error = nil;
  id<MTLComputePipelineState> pso = [g_metal.device newComputePipelineStateWithFunction:function
                                                                                  error:&error];
  if (error) {
    NSLog(@"[Manifold Metal] Failed to create PSO for %s: %@", functionName, error);
    return nil;
  }
  return pso;
}

void DoInitialize() {
  @autoreleasepool {
    g_metal.device = MTLCreateSystemDefaultDevice();
    if (!g_metal.device) {
      NSLog(@"[Manifold Metal] No Metal device available");
      return;
    }

    // Store device name
    const char* name = [g_metal.device.name UTF8String];
    strncpy(g_metal.deviceName, name, sizeof(g_metal.deviceName) - 1);

    g_metal.commandQueue = [g_metal.device newCommandQueue];
    if (!g_metal.commandQueue) {
      NSLog(@"[Manifold Metal] Failed to create command queue");
      g_metal.device = nil;
      return;
    }

    // Load the Metal shader library
    // First try to find the compiled .metallib next to the executable
    NSBundle* bundle = [NSBundle mainBundle];
    NSString* libPath = [bundle pathForResource:@"metal_kernels" ofType:@"metallib"];

    if (libPath) {
      NSError* error = nil;
      NSURL* libURL = [NSURL fileURLWithPath:libPath];
      g_metal.library = [g_metal.device newLibraryWithURL:libURL error:&error];
      if (error) {
        NSLog(@"[Manifold Metal] Failed to load metallib: %@", error);
      }
    }

    // Fallback: compile from source at runtime
    if (!g_metal.library) {
      // Try to find the .metal source file
      NSString* srcPath = [bundle pathForResource:@"metal_kernels" ofType:@"metal"];
      if (!srcPath) {
        // Try relative to executable for development builds
        NSString* execPath = [[NSProcessInfo processInfo].arguments firstObject];
        NSString* execDir = [execPath stringByDeletingLastPathComponent];
        NSArray* searchPaths = @[
          [execDir stringByAppendingPathComponent:@"metal_kernels.metal"],
          [execDir stringByAppendingPathComponent:@"../Resources/metal_kernels.metal"],
          [execDir stringByAppendingPathComponent:@"../share/openscad/metal_kernels.metal"],
        ];
        for (NSString* path in searchPaths) {
          if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
            srcPath = path;
            break;
          }
        }
      }

      if (srcPath) {
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:srcPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (source && !error) {
          MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
          options.fastMathEnabled = YES;
          g_metal.library = [g_metal.device newLibraryWithSource:source
                                                         options:options
                                                           error:&error];
          if (error) {
            NSLog(@"[Manifold Metal] Shader compile error: %@", error);
          }
        }
      }
    }

    if (!g_metal.library) {
      NSLog(@"[Manifold Metal] Failed to load Metal shader library");
      g_metal.commandQueue = nil;
      g_metal.device = nil;
      return;
    }

    // Create pipeline states for all kernels
    g_metal.radixHistogramPSO = CreatePipeline("radix_histogram");
    g_metal.radixPrefixSumPSO = CreatePipeline("radix_prefix_sum");
    g_metal.radixScatterKVPSO = CreatePipeline("radix_scatter_kv");
    g_metal.exclusiveScanIntBlockPSO = CreatePipeline("exclusive_scan_int_block");
    g_metal.scanAddBlockOffsetsPSO = CreatePipeline("scan_add_block_offsets");
    g_metal.inclusiveScanIntBlockPSO = CreatePipeline("inclusive_scan_int_block");
    g_metal.inclusiveScanDoubleBlockPSO = CreatePipeline("inclusive_scan_double_block");
    g_metal.scanAddBlockOffsetsDoublePSO = CreatePipeline("scan_add_block_offsets_double");
    g_metal.reduceSumIntPSO = CreatePipeline("reduce_sum_int");
    g_metal.reduceSumFloatPSO = CreatePipeline("reduce_sum_float");
    g_metal.reduceMinMaxFloatPSO = CreatePipeline("reduce_minmax_float");
    g_metal.sequenceFillPSO = CreatePipeline("sequence_fill");
    g_metal.scatterIntPSO = CreatePipeline("scatter_int");
    g_metal.gatherIntPSO = CreatePipeline("gather_int");
    g_metal.computeMortonCodesPSO = CreatePipeline("compute_morton_codes");
    g_metal.computeFaceBoxMortonPSO = CreatePipeline("compute_face_box_morton");
    g_metal.reindexFacesPSO = CreatePipeline("reindex_faces");
    g_metal.reindexVertsInHalfedgesPSO = CreatePipeline("reindex_verts_in_halfedges");
    g_metal.markUsedPropsPSO = CreatePipeline("mark_used_props");
    g_metal.compactPropertiesPSO = CreatePipeline("compact_properties");
    g_metal.updatePropVertsPSO = CreatePipeline("update_prop_verts");

    g_metal.maxThreadsPerGroup = (uint32_t)g_metal.radixHistogramPSO.maxTotalThreadsPerThreadgroup;
    if (g_metal.maxThreadsPerGroup > 1024) g_metal.maxThreadsPerGroup = 1024;

    g_metal.initialized = true;
    NSLog(@"[Manifold Metal] GPU compute initialized: %s (unified memory: %@)", g_metal.deviceName,
          g_metal.device.hasUnifiedMemory ? @"YES" : @"NO");
  }
}

// Helper: create a buffer wrapping existing memory (zero-copy on unified memory)
id<MTLBuffer> WrapBuffer(const void* ptr, size_t size) {
  if (g_metal.device.hasUnifiedMemory) {
    // On Apple Silicon: shared memory, no copy needed
    return [g_metal.device newBufferWithBytesNoCopy:const_cast<void*>(ptr)
                                             length:size
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
  } else {
    // On discrete GPU: must copy
    return [g_metal.device newBufferWithBytes:ptr length:size options:MTLResourceStorageModeShared];
  }
}

// Helper: create a mutable buffer wrapping existing memory
id<MTLBuffer> WrapMutableBuffer(void* ptr, size_t size) {
  if (g_metal.device.hasUnifiedMemory) {
    return [g_metal.device newBufferWithBytesNoCopy:ptr
                                             length:size
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
  } else {
    return [g_metal.device newBufferWithBytes:ptr length:size options:MTLResourceStorageModeShared];
  }
}

// Helper: create a new GPU buffer
id<MTLBuffer> NewBuffer(size_t size) {
  return [g_metal.device newBufferWithLength:size options:MTLResourceStorageModeShared];
}

// Helper: dispatch a compute kernel and wait
void DispatchAndWait(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pso,
                     uint32_t count) {
  uint32_t threadGroupSize =
      std::min(g_metal.maxThreadsPerGroup, (uint32_t)pso.maxTotalThreadsPerThreadgroup);
  // Ensure power of 2 for reduction kernels
  uint32_t numGroups = (count + threadGroupSize - 1) / threadGroupSize;

  [encoder dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API implementation
// ---------------------------------------------------------------------------

bool Initialize() {
  std::call_once(g_initFlag, DoInitialize);
  return g_metal.initialized;
}

bool IsAvailable() { return g_metal.initialized; }

void Shutdown() {
  @autoreleasepool {
    // Release all pipeline states
    g_metal.radixHistogramPSO = nil;
    g_metal.radixPrefixSumPSO = nil;
    g_metal.radixScatterKVPSO = nil;
    g_metal.exclusiveScanIntBlockPSO = nil;
    g_metal.scanAddBlockOffsetsPSO = nil;
    g_metal.inclusiveScanIntBlockPSO = nil;
    g_metal.inclusiveScanDoubleBlockPSO = nil;
    g_metal.scanAddBlockOffsetsDoublePSO = nil;
    g_metal.reduceSumIntPSO = nil;
    g_metal.reduceSumFloatPSO = nil;
    g_metal.reduceMinMaxFloatPSO = nil;
    g_metal.sequenceFillPSO = nil;
    g_metal.scatterIntPSO = nil;
    g_metal.gatherIntPSO = nil;
    g_metal.computeMortonCodesPSO = nil;
    g_metal.computeFaceBoxMortonPSO = nil;
    g_metal.reindexFacesPSO = nil;
    g_metal.reindexVertsInHalfedgesPSO = nil;
    g_metal.markUsedPropsPSO = nil;
    g_metal.compactPropertiesPSO = nil;
    g_metal.updatePropVertsPSO = nil;

    g_metal.library = nil;
    g_metal.commandQueue = nil;
    g_metal.device = nil;
    g_metal.initialized = false;
  }
}

DeviceInfo GetDeviceInfo() {
  DeviceInfo info = {};
  if (!g_metal.initialized) return info;

  info.name = g_metal.deviceName;
  info.maxBufferLength = [g_metal.device maxBufferLength];
  info.recommendedMaxMemory = [g_metal.device recommendedMaxWorkingSetSize];
  info.maxThreadsPerGroup = g_metal.maxThreadsPerGroup;
  info.hasUnifiedMemory = [g_metal.device hasUnifiedMemory];
  return info;
}

// ---------------------------------------------------------------------------
// Radix Sort (key-value)
// ---------------------------------------------------------------------------

bool RadixSortKeyValue(uint32_t* keys, int32_t* values, size_t count) {
  if (!g_metal.initialized || count < kGPUThreshold) return false;

  @autoreleasepool {
    size_t keyBytes = count * sizeof(uint32_t);
    size_t valBytes = count * sizeof(int32_t);

    // Allocate GPU buffers. Use newBufferWithBytes for input (copies once),
    // then all 4 passes run entirely on GPU without CPU round-trips.
    id<MTLBuffer> keysA = [g_metal.device newBufferWithBytes:keys
                                                      length:keyBytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> valsA = [g_metal.device newBufferWithBytes:values
                                                      length:valBytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> keysB = NewBuffer(keyBytes);
    id<MTLBuffer> valsB = NewBuffer(valBytes);
    id<MTLBuffer> histogram = NewBuffer(256 * sizeof(uint32_t));

    if (!keysA || !valsA || !keysB || !valsB || !histogram) return false;

    // Single command buffer for all 4 radix sort passes — avoids
    // 4x command buffer commit+wait overhead (~60μs per round-trip).
    id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
    if (!cmdBuf) return false;

    bool swapped = false;
    uint32_t cnt32 = (uint32_t)count;

    for (uint32_t pass = 0; pass < 4; pass++) {
      uint32_t shift = pass * 8;

      // Clear histogram
      id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
      [blit fillBuffer:histogram range:NSMakeRange(0, 256 * sizeof(uint32_t)) value:0];
      [blit endEncoding];

      // Phase 1: Compute histogram
      {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:g_metal.radixHistogramPSO];
        [enc setBuffer:(swapped ? keysB : keysA) offset:0 atIndex:0];
        [enc setBuffer:histogram offset:0 atIndex:1];
        [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&shift length:sizeof(uint32_t) atIndex:3];

        uint32_t groupSize = std::min(g_metal.maxThreadsPerGroup, 256u);
        uint32_t numGroups = std::min((uint32_t)((count + groupSize - 1) / groupSize), 64u);
        [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
        [enc endEncoding];
      }

      // Phase 2: Prefix sum on histogram
      {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:g_metal.radixPrefixSumPSO];
        [enc setBuffer:histogram offset:0 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
      }

      // Phase 3: Scatter key-value pairs
      {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:g_metal.radixScatterKVPSO];
        [enc setBuffer:(swapped ? keysB : keysA) offset:0 atIndex:0];
        [enc setBuffer:(swapped ? valsB : valsA) offset:0 atIndex:1];
        [enc setBuffer:(swapped ? keysA : keysB) offset:0 atIndex:2];
        [enc setBuffer:(swapped ? valsA : valsB) offset:0 atIndex:3];
        [enc setBuffer:histogram offset:0 atIndex:4];
        [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&shift length:sizeof(uint32_t) atIndex:6];

        uint32_t groupSize = g_metal.maxThreadsPerGroup;
        uint32_t numGroups = (uint32_t)((count + groupSize - 1) / groupSize);
        [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
        [enc endEncoding];
      }

      swapped = !swapped;
    }

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (cmdBuf.status == MTLCommandBufferStatusError) {
      NSLog(@"[Manifold Metal] Radix sort failed: %@", cmdBuf.error);
      return false;
    }

    // Copy result back from whichever buffer has the final data
    if (swapped) {
      memcpy(keys, keysB.contents, keyBytes);
      memcpy(values, valsB.contents, valBytes);
    } else {
      memcpy(keys, keysA.contents, keyBytes);
      memcpy(values, valsA.contents, valBytes);
    }

    return true;
  }
}

bool RadixSortUint32(uint32_t* data, size_t count) {
  if (!g_metal.initialized || count < kGPUThreshold) return false;

  // Create identity index array and sort as key-value
  std::vector<int32_t> indices(count);
  for (size_t i = 0; i < count; i++) indices[i] = (int32_t)i;

  // For keys-only sort, we can use a simplified version
  // but reuse the KV sort for correctness
  std::vector<uint32_t> keyCopy(data, data + count);
  if (!RadixSortKeyValue(keyCopy.data(), indices.data(), count)) return false;

  memcpy(data, keyCopy.data(), count * sizeof(uint32_t));
  return true;
}

// ---------------------------------------------------------------------------
// Prefix Scan
// ---------------------------------------------------------------------------

bool InclusiveScanInt(const int32_t* input, int32_t* output, size_t count) {
  if (!g_metal.initialized || count < kGPUThreshold) return false;
  if (!g_metal.inclusiveScanIntBlockPSO || !g_metal.scanAddBlockOffsetsPSO) return false;

  @autoreleasepool {
    uint32_t groupSize =
        std::min(g_metal.maxThreadsPerGroup,
                 (uint32_t)g_metal.inclusiveScanIntBlockPSO.maxTotalThreadsPerThreadgroup);
    // Ensure power of 2
    groupSize = 1u << (31 - __builtin_clz(groupSize));
    if (groupSize > 1024) groupSize = 1024;

    uint32_t numGroups = (uint32_t)((count + groupSize - 1) / groupSize);

    id<MTLBuffer> inputBuf = WrapBuffer(input, count * sizeof(int32_t));
    id<MTLBuffer> outputBuf = WrapMutableBuffer(output, count * sizeof(int32_t));
    id<MTLBuffer> blockSums = NewBuffer(numGroups * sizeof(int32_t));

    if (!inputBuf || !outputBuf || !blockSums) return false;

    // Phase 1: Block-level inclusive scan
    {
      id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:g_metal.inclusiveScanIntBlockPSO];
      [enc setBuffer:inputBuf offset:0 atIndex:0];
      [enc setBuffer:outputBuf offset:0 atIndex:1];
      [enc setBuffer:blockSums offset:0 atIndex:2];
      uint32_t cnt32 = (uint32_t)count;
      [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    // Phase 2: Scan block sums on CPU (small array)
    if (numGroups > 1) {
      int32_t* sums = (int32_t*)blockSums.contents;
      for (uint32_t i = 1; i < numGroups; i++) {
        sums[i] += sums[i - 1];
      }

      // Phase 3: Add block offsets
      id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:g_metal.scanAddBlockOffsetsPSO];
      [enc setBuffer:outputBuf offset:0 atIndex:0];
      [enc setBuffer:blockSums offset:0 atIndex:1];
      uint32_t cnt32 = (uint32_t)count;
      [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    // Copy back if not using unified memory wrapping
    if (!g_metal.device.hasUnifiedMemory) {
      memcpy(output, outputBuf.contents, count * sizeof(int32_t));
    }

    return true;
  }
}

bool ExclusiveScanInt(const int32_t* input, int32_t* output, size_t count, int32_t init) {
  if (!g_metal.initialized || count < kGPUThreshold) return false;
  if (!g_metal.exclusiveScanIntBlockPSO || !g_metal.scanAddBlockOffsetsPSO) return false;

  @autoreleasepool {
    uint32_t groupSize =
        std::min(g_metal.maxThreadsPerGroup,
                 (uint32_t)g_metal.exclusiveScanIntBlockPSO.maxTotalThreadsPerThreadgroup);
    groupSize = 1u << (31 - __builtin_clz(groupSize));
    if (groupSize > 1024) groupSize = 1024;

    uint32_t numGroups = (uint32_t)((count + groupSize - 1) / groupSize);

    id<MTLBuffer> inputBuf = WrapBuffer(input, count * sizeof(int32_t));
    id<MTLBuffer> outputBuf = WrapMutableBuffer(output, count * sizeof(int32_t));
    id<MTLBuffer> blockSums = NewBuffer(numGroups * sizeof(int32_t));

    if (!inputBuf || !outputBuf || !blockSums) return false;

    // Phase 1: Block-level exclusive scan
    {
      id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:g_metal.exclusiveScanIntBlockPSO];
      [enc setBuffer:inputBuf offset:0 atIndex:0];
      [enc setBuffer:outputBuf offset:0 atIndex:1];
      [enc setBuffer:blockSums offset:0 atIndex:2];
      uint32_t cnt32 = (uint32_t)count;
      [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    // Phase 2: Scan block sums on CPU
    if (numGroups > 1) {
      int32_t* sums = (int32_t*)blockSums.contents;
      for (uint32_t i = 1; i < numGroups; i++) {
        sums[i] += sums[i - 1];
      }

      // Phase 3: Add block offsets
      id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:g_metal.scanAddBlockOffsetsPSO];
      [enc setBuffer:outputBuf offset:0 atIndex:0];
      [enc setBuffer:blockSums offset:0 atIndex:1];
      uint32_t cnt32 = (uint32_t)count;
      [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    // Add init value if non-zero
    if (init != 0) {
      for (size_t i = 0; i < count; i++) output[i] += init;
    }

    if (!g_metal.device.hasUnifiedMemory) {
      memcpy(output, outputBuf.contents, count * sizeof(int32_t));
    }

    return true;
  }
}

bool InclusiveScanDouble(const double* input, double* output, size_t count) {
  if (!g_metal.initialized || count < kGPUThreshold) return false;
  if (!g_metal.inclusiveScanDoubleBlockPSO) return false;

  // Metal doesn't support double natively on all devices.
  // Convert to float, scan, convert back. For scan this is acceptable
  // since the cumulative error is bounded.
  @autoreleasepool {
    std::vector<float> floatInput(count);
    for (size_t i = 0; i < count; i++) floatInput[i] = (float)input[i];

    uint32_t groupSize =
        std::min(g_metal.maxThreadsPerGroup,
                 (uint32_t)g_metal.inclusiveScanDoubleBlockPSO.maxTotalThreadsPerThreadgroup);
    groupSize = 1u << (31 - __builtin_clz(groupSize));
    if (groupSize > 1024) groupSize = 1024;

    uint32_t numGroups = (uint32_t)((count + groupSize - 1) / groupSize);

    id<MTLBuffer> inputBuf = WrapBuffer(floatInput.data(), count * sizeof(float));
    id<MTLBuffer> outputBuf = NewBuffer(count * sizeof(float));
    id<MTLBuffer> blockSums = NewBuffer(numGroups * sizeof(float));

    if (!inputBuf || !outputBuf || !blockSums) return false;

    {
      id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:g_metal.inclusiveScanDoubleBlockPSO];
      [enc setBuffer:inputBuf offset:0 atIndex:0];
      [enc setBuffer:outputBuf offset:0 atIndex:1];
      [enc setBuffer:blockSums offset:0 atIndex:2];
      uint32_t cnt32 = (uint32_t)count;
      [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    if (numGroups > 1) {
      float* sums = (float*)blockSums.contents;
      for (uint32_t i = 1; i < numGroups; i++) sums[i] += sums[i - 1];

      id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:g_metal.scanAddBlockOffsetsDoublePSO];
      [enc setBuffer:outputBuf offset:0 atIndex:0];
      [enc setBuffer:blockSums offset:0 atIndex:1];
      uint32_t cnt32 = (uint32_t)count;
      [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    float* result = (float*)outputBuf.contents;
    for (size_t i = 0; i < count; i++) output[i] = (double)result[i];

    return true;
  }
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

double ReduceSumDouble(const double* data, size_t count) {
  if (!g_metal.initialized || count < kGPUThreshold || !g_metal.reduceSumFloatPSO)
    return std::numeric_limits<double>::quiet_NaN();

  @autoreleasepool {
    // Convert to float for GPU
    std::vector<float> floatData(count);
    for (size_t i = 0; i < count; i++) floatData[i] = (float)data[i];

    uint32_t groupSize = g_metal.maxThreadsPerGroup;
    groupSize = 1u << (31 - __builtin_clz(groupSize));
    if (groupSize > 1024) groupSize = 1024;
    uint32_t numGroups = (uint32_t)((count + groupSize - 1) / groupSize);

    id<MTLBuffer> inputBuf = WrapBuffer(floatData.data(), count * sizeof(float));
    id<MTLBuffer> outputBuf = NewBuffer(numGroups * sizeof(float));

    {
      id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:g_metal.reduceSumFloatPSO];
      [enc setBuffer:inputBuf offset:0 atIndex:0];
      [enc setBuffer:outputBuf offset:0 atIndex:1];
      uint32_t cnt32 = (uint32_t)count;
      [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    // Final reduction on CPU (small array)
    float* partials = (float*)outputBuf.contents;
    double sum = 0.0;
    for (uint32_t i = 0; i < numGroups; i++) sum += (double)partials[i];
    return sum;
  }
}

int32_t ReduceSumInt(const int32_t* data, size_t count) {
  if (!g_metal.initialized || count < kGPUThreshold || !g_metal.reduceSumIntPSO) return 0;

  @autoreleasepool {
    uint32_t groupSize = g_metal.maxThreadsPerGroup;
    groupSize = 1u << (31 - __builtin_clz(groupSize));
    if (groupSize > 1024) groupSize = 1024;
    uint32_t numGroups = (uint32_t)((count + groupSize - 1) / groupSize);

    id<MTLBuffer> inputBuf = WrapBuffer(data, count * sizeof(int32_t));
    id<MTLBuffer> outputBuf = NewBuffer(numGroups * sizeof(int32_t));

    {
      id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:g_metal.reduceSumIntPSO];
      [enc setBuffer:inputBuf offset:0 atIndex:0];
      [enc setBuffer:outputBuf offset:0 atIndex:1];
      uint32_t cnt32 = (uint32_t)count;
      [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    int32_t* partials = (int32_t*)outputBuf.contents;
    int32_t sum = 0;
    for (uint32_t i = 0; i < numGroups; i++) sum += partials[i];
    return sum;
  }
}

std::pair<double, double> ReduceMinMaxDouble(const double* data, size_t count) {
  if (!g_metal.initialized || count < kGPUThreshold || !g_metal.reduceMinMaxFloatPSO)
    return {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};

  @autoreleasepool {
    std::vector<float> floatData(count);
    for (size_t i = 0; i < count; i++) floatData[i] = (float)data[i];

    uint32_t groupSize = g_metal.maxThreadsPerGroup;
    groupSize = 1u << (31 - __builtin_clz(groupSize));
    if (groupSize > 1024) groupSize = 1024;
    uint32_t numGroups = (uint32_t)((count + groupSize - 1) / groupSize);

    id<MTLBuffer> inputBuf = WrapBuffer(floatData.data(), count * sizeof(float));
    id<MTLBuffer> minBuf = NewBuffer(numGroups * sizeof(float));
    id<MTLBuffer> maxBuf = NewBuffer(numGroups * sizeof(float));

    {
      id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:g_metal.reduceMinMaxFloatPSO];
      [enc setBuffer:inputBuf offset:0 atIndex:0];
      [enc setBuffer:minBuf offset:0 atIndex:1];
      [enc setBuffer:maxBuf offset:0 atIndex:2];
      uint32_t cnt32 = (uint32_t)count;
      [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    float* mins = (float*)minBuf.contents;
    float* maxs = (float*)maxBuf.contents;
    double globalMin = std::numeric_limits<double>::infinity();
    double globalMax = -std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < numGroups; i++) {
      globalMin = std::min(globalMin, (double)mins[i]);
      globalMax = std::max(globalMax, (double)maxs[i]);
    }
    return {globalMin, globalMax};
  }
}

// ---------------------------------------------------------------------------
// Utility operations
// ---------------------------------------------------------------------------

bool SequenceFill(int32_t* data, size_t count, int32_t start) {
  if (!g_metal.initialized || count < kGPUThreshold || !g_metal.sequenceFillPSO) return false;

  @autoreleasepool {
    id<MTLBuffer> outputBuf = WrapMutableBuffer(data, count * sizeof(int32_t));
    if (!outputBuf) return false;

    id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:g_metal.sequenceFillPSO];
    [enc setBuffer:outputBuf offset:0 atIndex:0];
    [enc setBytes:&start length:sizeof(int32_t) atIndex:1];
    uint32_t cnt32 = (uint32_t)count;
    [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:2];
    DispatchAndWait(enc, g_metal.sequenceFillPSO, (uint32_t)count);
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (!g_metal.device.hasUnifiedMemory) {
      memcpy(data, outputBuf.contents, count * sizeof(int32_t));
    }
    return true;
  }
}

bool ScatterInt(const int32_t* input, const int32_t* indices, int32_t* output, size_t count) {
  if (!g_metal.initialized || count < kGPUThreshold || !g_metal.scatterIntPSO) return false;

  @autoreleasepool {
    id<MTLBuffer> inputBuf = WrapBuffer(input, count * sizeof(int32_t));
    id<MTLBuffer> indicesBuf = WrapBuffer(indices, count * sizeof(int32_t));
    // Output size unknown (could be larger), use input size as estimate
    id<MTLBuffer> outputBuf = WrapMutableBuffer(output, count * sizeof(int32_t));

    if (!inputBuf || !indicesBuf || !outputBuf) return false;

    id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:g_metal.scatterIntPSO];
    [enc setBuffer:inputBuf offset:0 atIndex:0];
    [enc setBuffer:indicesBuf offset:0 atIndex:1];
    [enc setBuffer:outputBuf offset:0 atIndex:2];
    uint32_t cnt32 = (uint32_t)count;
    [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:3];
    DispatchAndWait(enc, g_metal.scatterIntPSO, (uint32_t)count);
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (!g_metal.device.hasUnifiedMemory) {
      memcpy(output, outputBuf.contents, count * sizeof(int32_t));
    }
    return true;
  }
}

bool GatherInt(const int32_t* input, const int32_t* indices, int32_t* output, size_t count) {
  if (!g_metal.initialized || count < kGPUThreshold || !g_metal.gatherIntPSO) return false;

  @autoreleasepool {
    id<MTLBuffer> inputBuf = WrapBuffer(input, count * sizeof(int32_t));
    id<MTLBuffer> indicesBuf = WrapBuffer(indices, count * sizeof(int32_t));
    id<MTLBuffer> outputBuf = WrapMutableBuffer(output, count * sizeof(int32_t));

    if (!inputBuf || !indicesBuf || !outputBuf) return false;

    id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:g_metal.gatherIntPSO];
    [enc setBuffer:inputBuf offset:0 atIndex:0];
    [enc setBuffer:indicesBuf offset:0 atIndex:1];
    [enc setBuffer:outputBuf offset:0 atIndex:2];
    uint32_t cnt32 = (uint32_t)count;
    [enc setBytes:&cnt32 length:sizeof(uint32_t) atIndex:3];
    DispatchAndWait(enc, g_metal.gatherIntPSO, (uint32_t)count);
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (!g_metal.device.hasUnifiedMemory) {
      memcpy(output, outputBuf.contents, count * sizeof(int32_t));
    }
    return true;
  }
}

}  // namespace metal
}  // namespace manifold
