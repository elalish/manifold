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

#include <numeric>
#include <set>

#include "collider.h"
#include "graph.h"
#include "public.h"

using namespace manifold;

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

namespace {

struct Edge {
  int first, second;
  bool forward;

  Edge(int start, int end) {
    forward = start < end;
    if (forward) {
      first = start;
      second = end;
    } else {
      first = end;
      second = start;
    }
  }

  bool operator<(const Edge& other) const {
    return first == other.first ? second < other.second : first < other.first;
  }
};

struct VertMortonBox {
  const float* vertProperties;
  const int numProp;
  const float tol;
  const Box bBox;

  __host__ __device__ void operator()(
      thrust::tuple<uint32_t&, Box&, int> inout) {
    uint32_t& mortonCode = thrust::get<0>(inout);
    Box& vertBox = thrust::get<1>(inout);
    int vert = thrust::get<2>(inout);

    const glm::vec3 center(vertProperties[numProp * vert],
                           vertProperties[numProp * vert + 1],
                           vertProperties[numProp * vert + 2]);

    vertBox.min = center - tol / 2;
    vertBox.max = center + tol / 2;

    mortonCode = MortonCode(center, bBox);
  }
};
}  // namespace

MeshGL::MeshGL(const Mesh& mesh) {
  numProp = 3;
  vertProperties.resize(numProp * mesh.vertPos.size());
  for (int i = 0; i < mesh.vertPos.size(); ++i) {
    for (int j : {0, 1, 2}) vertProperties[3 * i + j] = mesh.vertPos[i][j];
  }
  triVerts.resize(3 * mesh.triVerts.size());
  for (int i = 0; i < mesh.triVerts.size(); ++i) {
    for (int j : {0, 1, 2}) triVerts[3 * i + j] = mesh.triVerts[i][j];
  }
  halfedgeTangent.resize(4 * mesh.halfedgeTangent.size());
  for (int i = 0; i < mesh.halfedgeTangent.size(); ++i) {
    for (int j : {0, 1, 2, 3})
      halfedgeTangent[4 * i + j] = mesh.halfedgeTangent[i][j];
  }
}

bool MeshGL::Merge() {
  std::multiset<Edge> openEdges;

  std::vector<int> merge(NumVert());
  std::iota(merge.begin(), merge.end(), 0);
  for (int i = 0; i < mergeFromVert.size(); ++i) {
    merge[mergeFromVert[i]] = mergeToVert[i];
  }

  const int numTri = NumTri();
  const int next[3] = {1, 2, 0};
  for (int tri = 0; tri < numTri; ++tri) {
    for (int i : {0, 1, 2}) {
      Edge edge(merge[triVerts[3 * tri + i]],
                merge[triVerts[3 * tri + next[i]]]);
      auto range = openEdges.equal_range(edge);
      bool found = false;
      for (auto it = range.first; it != range.second; ++it) {
        if (it->forward != edge.forward) {
          openEdges.erase(it);
          found = true;
          break;
        }
      }
      if (!found) openEdges.insert(edge);
    }
  }

  if (openEdges.empty()) {
    return false;
  }

  // TODO: calculate bounding box and fill in openVerts list
  Box bBox;
  VecDH<int> openVerts(numOpenVert);

  VecDH<float> vertPropD(vertProperties);
  VecDH<Box> vertBox(numOpenVert);
  VecDH<uint32_t> vertMorton(numOpenVert);
  for_each_n(autoPolicy(NumTri()),
             zip(vertMorton.begin(), vertBox.begin(), openVerts.cbegin()),
             numOpenVert,
             VertMortonBox({vertPropD.cptrD(), numProp, kTolerance, bBox}));

  Collider collider(vertBox, vertMorton);
  // TODO: ignore self-collisions
  SparseIndices toMerge = collider.Collisions(vertBox);

  Graph graph;
  for (int i = 0; i < numVert; ++i) {
    graph.add_nodes(i);
  }
  // TODO: incorporate input merge vec here
  for (int i = 0; i < toMerge.size(); ++i) {
    graph.add_edge(*(toMerge.begin(0) + i), *(toMerge.begin(1) + i));
  }

  std::vector<int> vertLabels;
  const int numComponents = ConnectedComponents(vertLabels, graph);
  // TODO: rebuild merge vec from vertLabels

  return true;
}