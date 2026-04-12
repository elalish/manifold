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
// Metal compute shader kernels for GPU-accelerated parallel primitives.
// These kernels are used by the Manifold boolean/CSG pipeline for
// sorting, scanning, reducing, and transforming mesh data on the GPU.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constant uint THREADS_PER_GROUP [[function_constant(0)]];

// ---------------------------------------------------------------------------
// Radix Sort Kernels (LSB radix sort, 8-bit passes)
// ---------------------------------------------------------------------------

// Per-threadgroup histogram for one radix digit pass
kernel void radix_histogram(
    device const uint* input [[buffer(0)]],
    device atomic_uint* histograms [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& shift [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint groupSize [[threads_per_threadgroup]],
    uint numGroups [[threadgroups_per_grid]])
{
    // Local histogram for this threadgroup
    threadgroup uint localHist[256];
    if (tid < 256) {
        localHist[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread processes multiple elements
    uint elementsPerGroup = (count + numGroups - 1) / numGroups;
    uint start = gid * elementsPerGroup;
    uint end = min(start + elementsPerGroup, count);

    for (uint i = start + tid; i < end; i += groupSize) {
        uint digit = (input[i] >> shift) & 0xFF;
        atomic_fetch_add_explicit(
            reinterpret_cast<threadgroup atomic_uint*>(&localHist[digit]),
            1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write local histogram to global memory
    if (tid < 256) {
        atomic_fetch_add_explicit(&histograms[tid], localHist[tid],
                                  memory_order_relaxed);
    }
}

// Prefix sum on the 256-element histogram (single threadgroup)
kernel void radix_prefix_sum(
    device uint* histograms [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint temp[256];
    if (tid < 256) {
        temp[tid] = histograms[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan
    for (uint offset = 1; offset < 256; offset <<= 1) {
        uint val = 0;
        if (tid >= offset && tid < 256) {
            val = temp[tid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 256) {
            temp[tid] += val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Convert to exclusive scan
    if (tid < 256) {
        histograms[tid] = (tid > 0) ? temp[tid - 1] : 0;
    }
}

// Scatter elements according to radix digit offsets
kernel void radix_scatter(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& shift [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    uint val = input[gid];
    uint digit = (val >> shift) & 0xFF;

    // Use atomic to get unique position for this digit
    uint pos = atomic_fetch_add_explicit(
        reinterpret_cast<device atomic_uint*>(&output[0]) + digit,
        1, memory_order_relaxed);
    // Note: this simple scatter is used with pre-computed offsets
    // The actual implementation uses a two-pass approach
}

// Scatter key-value pairs for radix sort
kernel void radix_scatter_kv(
    device const uint* inputKeys [[buffer(0)]],
    device const int* inputVals [[buffer(1)]],
    device uint* outputKeys [[buffer(2)]],
    device int* outputVals [[buffer(3)]],
    device atomic_uint* digitOffsets [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& shift [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    uint key = inputKeys[gid];
    uint digit = (key >> shift) & 0xFF;
    uint pos = atomic_fetch_add_explicit(&digitOffsets[digit], 1,
                                         memory_order_relaxed);
    outputKeys[pos] = key;
    outputVals[pos] = inputVals[gid];
}

// ---------------------------------------------------------------------------
// Prefix Scan Kernels (Blelloch-style work-efficient scan)
// ---------------------------------------------------------------------------

// Block-level exclusive scan for int32
kernel void exclusive_scan_int_block(
    device const int* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    device int* blockSums [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint groupId [[threadgroup_position_in_grid]],
    uint groupSize [[threads_per_threadgroup]])
{
    threadgroup int shared[1024];

    uint idx = gid;
    shared[tid] = (idx < count) ? input[idx] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce)
    for (uint stride = 1; stride < groupSize; stride <<= 1) {
        uint index = (tid + 1) * (stride << 1) - 1;
        if (index < groupSize) {
            shared[index] += shared[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Save block sum and clear last element
    if (tid == groupSize - 1) {
        if (blockSums) {
            blockSums[groupId] = shared[tid];
        }
        shared[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint stride = groupSize >> 1; stride > 0; stride >>= 1) {
        uint index = (tid + 1) * (stride << 1) - 1;
        if (index < groupSize) {
            int temp = shared[index - stride];
            shared[index - stride] = shared[index];
            shared[index] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (idx < count) {
        output[idx] = shared[tid];
    }
}

// Add block offsets to complete multi-block scan
kernel void scan_add_block_offsets(
    device int* data [[buffer(0)]],
    device const int* blockOffsets [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint groupId [[threadgroup_position_in_grid]])
{
    if (gid < count && groupId > 0) {
        data[gid] += blockOffsets[groupId];
    }
}

// Inclusive scan for int32 (block-level)
kernel void inclusive_scan_int_block(
    device const int* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    device int* blockSums [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint groupId [[threadgroup_position_in_grid]],
    uint groupSize [[threads_per_threadgroup]])
{
    threadgroup int shared[1024];

    uint idx = gid;
    shared[tid] = (idx < count) ? input[idx] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan
    for (uint offset = 1; offset < groupSize; offset <<= 1) {
        int val = 0;
        if (tid >= offset) {
            val = shared[tid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] += val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (idx < count) {
        output[idx] = shared[tid];
    }

    // Save block sum
    if (tid == groupSize - 1 && blockSums) {
        blockSums[groupId] = shared[tid];
    }
}

// Inclusive scan for double (block-level)
kernel void inclusive_scan_double_block(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* blockSums [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint groupId [[threadgroup_position_in_grid]],
    uint groupSize [[threads_per_threadgroup]])
{
    threadgroup float shared[1024];

    uint idx = gid;
    shared[tid] = (idx < count) ? input[idx] : 0.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan
    for (uint offset = 1; offset < groupSize; offset <<= 1) {
        float val = 0.0;
        if (tid >= offset) {
            val = shared[tid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] += val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (idx < count) {
        output[idx] = shared[tid];
    }

    if (tid == groupSize - 1 && blockSums) {
        blockSums[groupId] = shared[tid];
    }
}

kernel void scan_add_block_offsets_double(
    device float* data [[buffer(0)]],
    device const float* blockOffsets [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint groupId [[threadgroup_position_in_grid]])
{
    if (gid < count && groupId > 0) {
        data[gid] += blockOffsets[groupId];
    }
}

// ---------------------------------------------------------------------------
// Reduction Kernels
// ---------------------------------------------------------------------------

// Sum reduction for int32
kernel void reduce_sum_int(
    device const int* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint groupId [[threadgroup_position_in_grid]],
    uint groupSize [[threads_per_threadgroup]])
{
    threadgroup int shared[1024];

    uint idx = gid;
    shared[tid] = (idx < count) ? input[idx] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = groupSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[groupId] = shared[0];
    }
}

// Sum reduction for float (used for double via two-pass)
kernel void reduce_sum_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint groupId [[threadgroup_position_in_grid]],
    uint groupSize [[threads_per_threadgroup]])
{
    threadgroup float shared[1024];

    uint idx = gid;
    shared[tid] = (idx < count) ? input[idx] : 0.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = groupSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[groupId] = shared[0];
    }
}

// Min/Max reduction for float
kernel void reduce_minmax_float(
    device const float* input [[buffer(0)]],
    device float* outputMin [[buffer(1)]],
    device float* outputMax [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint groupId [[threadgroup_position_in_grid]],
    uint groupSize [[threads_per_threadgroup]])
{
    threadgroup float sharedMin[1024];
    threadgroup float sharedMax[1024];

    uint idx = gid;
    float val = (idx < count) ? input[idx] : INFINITY;
    sharedMin[tid] = val;
    sharedMax[tid] = (idx < count) ? val : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = groupSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMin[tid] = min(sharedMin[tid], sharedMin[tid + stride]);
            sharedMax[tid] = max(sharedMax[tid], sharedMax[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        outputMin[groupId] = sharedMin[0];
        outputMax[groupId] = sharedMax[0];
    }
}

// ---------------------------------------------------------------------------
// Utility Kernels
// ---------------------------------------------------------------------------

// Fill array with sequential integers
kernel void sequence_fill(
    device int* output [[buffer(0)]],
    constant int& start [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        output[gid] = start + int(gid);
    }
}

// Scatter: output[indices[i]] = input[i]
kernel void scatter_int(
    device const int* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device int* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        output[indices[gid]] = input[gid];
    }
}

// Gather: output[i] = input[indices[i]]
kernel void gather_int(
    device const int* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device int* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        output[gid] = input[indices[gid]];
    }
}

// ---------------------------------------------------------------------------
// CSG/Boolean-specific Kernels
// ---------------------------------------------------------------------------

// Morton code computation for 3D positions
// Expands a 10-bit integer into 30 bits by inserting 2 zeros between each bit
inline uint expandBits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Compute Morton code for a 3D point within a bounding box
kernel void compute_morton_codes(
    device const float* positions [[buffer(0)]],  // packed xyz
    device uint* mortonCodes [[buffer(1)]],
    constant float3& bboxMin [[buffer(2)]],
    constant float3& bboxMax [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    float3 pos = float3(positions[gid * 3], positions[gid * 3 + 1],
                        positions[gid * 3 + 2]);

    // Check for NaN (marks removed vertices)
    if (isnan(pos.x)) {
        mortonCodes[gid] = 0xFFFFFFFF;
        return;
    }

    float3 range = bboxMax - bboxMin;
    float3 normalized = (pos - bboxMin) / range;
    normalized = clamp(normalized, 0.0f, 1.0f);

    uint x = uint(normalized.x * 1023.0f);
    uint y = uint(normalized.y * 1023.0f);
    uint z = uint(normalized.z * 1023.0f);

    mortonCodes[gid] = expandBits(x) | (expandBits(y) << 1) |
                       (expandBits(z) << 2);
}

// Compute face bounding boxes and Morton codes
kernel void compute_face_box_morton(
    device const float* vertPos [[buffer(0)]],       // xyz per vertex
    device const int* halfedges [[buffer(1)]],        // packed halfedge data
    device float* faceBoxMin [[buffer(2)]],           // xyz per face
    device float* faceBoxMax [[buffer(3)]],           // xyz per face
    device uint* faceMorton [[buffer(4)]],
    constant float3& bboxMin [[buffer(5)]],
    constant float3& bboxMax [[buffer(6)]],
    constant uint& numTri [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numTri) return;

    // Each face has 3 halfedges; halfedge stores startVert at offset 0
    // Halfedge struct: { int startVert, endVert, pairedHalfedge, propVert }
    int he0 = halfedges[gid * 3 * 4];  // startVert of first halfedge
    int he1 = halfedges[(gid * 3 + 1) * 4];
    int he2 = halfedges[(gid * 3 + 2) * 4];

    // Check for removed face
    int paired0 = halfedges[gid * 3 * 4 + 2];
    if (paired0 < 0) {
        faceMorton[gid] = 0xFFFFFFFF;
        return;
    }

    float3 v0 = float3(vertPos[he0 * 3], vertPos[he0 * 3 + 1],
                        vertPos[he0 * 3 + 2]);
    float3 v1 = float3(vertPos[he1 * 3], vertPos[he1 * 3 + 1],
                        vertPos[he1 * 3 + 2]);
    float3 v2 = float3(vertPos[he2 * 3], vertPos[he2 * 3 + 1],
                        vertPos[he2 * 3 + 2]);

    float3 boxMin = min(v0, min(v1, v2));
    float3 boxMax = max(v0, max(v1, v2));
    float3 center = (v0 + v1 + v2) / 3.0f;

    faceBoxMin[gid * 3] = boxMin.x;
    faceBoxMin[gid * 3 + 1] = boxMin.y;
    faceBoxMin[gid * 3 + 2] = boxMin.z;
    faceBoxMax[gid * 3] = boxMax.x;
    faceBoxMax[gid * 3 + 1] = boxMax.y;
    faceBoxMax[gid * 3 + 2] = boxMax.z;

    // Morton code for center
    float3 range = bboxMax - bboxMin;
    float3 normalized = (center - bboxMin) / range;
    normalized = clamp(normalized, 0.0f, 1.0f);

    uint x = uint(normalized.x * 1023.0f);
    uint y = uint(normalized.y * 1023.0f);
    uint z = uint(normalized.z * 1023.0f);

    faceMorton[gid] = expandBits(x) | (expandBits(y) << 1) |
                      (expandBits(z) << 2);
}

// Reindex halfedges after face sorting (parallel face gather)
kernel void reindex_faces(
    device const int* oldHalfedges [[buffer(0)]],     // source halfedge data
    device int* newHalfedges [[buffer(1)]],            // destination
    device const int* faceNew2Old [[buffer(2)]],
    device const int* faceOld2New [[buffer(3)]],
    constant uint& numTri [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numTri) return;

    int oldFace = faceNew2Old[gid];
    for (int i = 0; i < 3; i++) {
        int oldEdge = 3 * oldFace + i;
        int newEdge = 3 * int(gid) + i;

        // Copy halfedge (4 ints per halfedge)
        int startVert = oldHalfedges[oldEdge * 4];
        int endVert = oldHalfedges[oldEdge * 4 + 1];
        int paired = oldHalfedges[oldEdge * 4 + 2];
        int propVert = oldHalfedges[oldEdge * 4 + 3];

        // Remap pairedHalfedge to new face ordering
        int pairedFace = paired / 3;
        int offset = paired - 3 * pairedFace;
        paired = 3 * faceOld2New[pairedFace] + offset;

        newHalfedges[newEdge * 4] = startVert;
        newHalfedges[newEdge * 4 + 1] = endVert;
        newHalfedges[newEdge * 4 + 2] = paired;
        newHalfedges[newEdge * 4 + 3] = propVert;
    }
}

// Vertex reindexing: update halfedge vertex references after vertex sort
kernel void reindex_verts_in_halfedges(
    device int* halfedges [[buffer(0)]],
    device const int* vertOld2New [[buffer(1)]],
    constant uint& numHalfedges [[buffer(2)]],
    constant int& hasProp [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numHalfedges) return;

    int startVert = halfedges[gid * 4];
    if (startVert < 0) return;

    halfedges[gid * 4] = vertOld2New[startVert];
    halfedges[gid * 4 + 1] = vertOld2New[halfedges[gid * 4 + 1]];
    if (hasProp == 0) {
        halfedges[gid * 4 + 3] = halfedges[gid * 4];  // propVert = startVert
    }
}

// Property compaction: mark used property vertices
kernel void mark_used_props(
    device const int* halfedges [[buffer(0)]],
    device atomic_int* keep [[buffer(1)]],
    constant uint& numHalfedges [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numHalfedges) return;
    int propVert = halfedges[gid * 4 + 3];
    atomic_store_explicit(&keep[propVert], 1, memory_order_relaxed);
}

// Compact properties using scan results
kernel void compact_properties(
    device const float* oldProps [[buffer(0)]],
    device float* newProps [[buffer(1)]],
    device const int* propOld2New [[buffer(2)]],
    device const int* keep [[buffer(3)]],
    constant uint& numVerts [[buffer(4)]],
    constant uint& numProp [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numVerts) return;
    if (keep[gid] == 0) return;

    int newIdx = propOld2New[gid + 1];  // +1 because exclusive scan
    for (uint p = 0; p < numProp; p++) {
        newProps[newIdx * numProp + p] = oldProps[gid * numProp + p];
    }
}

// Update propVert in halfedges after property compaction
kernel void update_prop_verts(
    device int* halfedges [[buffer(0)]],
    device const int* propOld2New [[buffer(1)]],
    constant uint& numHalfedges [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numHalfedges) return;
    int propVert = halfedges[gid * 4 + 3];
    halfedges[gid * 4 + 3] = propOld2New[propVert];
}
