// Copyright 2019 Emmett Lalish
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

// #include <nvgraph.h>
#include <thrust/iterator/constant_iterator.h>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include "connected_components.cuh"

// 2^31-1 is nvgraph's unreachable flag
// constexpr int kNvgraphInvalid = std::numeric_limits<int>::max();

namespace {
using namespace manifold;

// void CheckStatus(nvgraphStatus_t status) {
//   ALWAYS_ASSERT((int)status == 0, runtimeErr, "nvGraph error: " + status);
// }

// struct DuplicateEdges {
//   int* source;
//   int* sink;

//   __host__ __device__ void operator()(thrust::tuple<int, Halfedge> in) {
//     int idx = thrust::get<0>(in);
//     Halfedge halfedges = thrust::get<1>(in);
//     source[idx] = halfedges.startVert;
//     sink[idx] = halfedges.endVert;
//   }
// };

// struct DuplicateKeep {
//   int* edgeMask;

//   __host__ __device__ void operator()(thrust::tuple<int, bool, Halfedge> in)
//   {
//     int idx = thrust::get<0>(in);
//     bool keep = thrust::get<1>(in);
//     Halfedge halfedge = thrust::get<2>(in);
//     edgeMask[idx] = keep;
//     edgeMask[halfedge.pairedHalfedge] = keep;
//   }
// };

// void Edges2CSR(VecDH<int>& rowOffsets, VecDH<int>& sink, VecDH<int>&
// edgeMask,
//                const VecDH<Halfedge>& halfedges, const VecDH<bool>& keep,
//                int numVert) {
//   // Duplicate undirected graph edges
//   int numHalfedge = halfedges.size();
//   VecDH<int> source(numHalfedge);
//   sink.resize(numHalfedge);
//   thrust::for_each_n(
//       zip(thrust::make_counting_iterator(0), halfedges.cbeginD()),
//       numHalfedge, DuplicateEdges({source.ptrD(), sink.ptrD()}));
//   // Build symmetric CSR adjacency matrix
//   VecDH<int> degree(numVert, 0);
//   VecDH<int> vid(numVert);
//   VecDH<int> temp(numVert);
//   if (keep.size() > 0) {
//     edgeMask.resize(numHalfedge);
//     thrust::for_each_n(zip(thrust::make_counting_iterator(0), keep.cbeginD(),
//                            halfedges.cbeginD()),
//                        numHalfedge, DuplicateKeep({edgeMask.ptrD()}));
//     thrust::sort_by_key(zip(source.beginD(), sink.beginD()),
//                         zip(source.endD(), sink.endD()), edgeMask.beginD());
//   } else {
//     thrust::sort(zip(source.beginD(), sink.beginD()),
//                  zip(source.endD(), sink.endD()));
//   }
//   auto endPair = thrust::reduce_by_key(source.beginD(), source.endD(),
//                                        thrust::constant_iterator<int>(1),
//                                        vid.beginD(), temp.beginD());
//   thrust::scatter(temp.beginD(), endPair.second, vid.beginD(),
//   degree.beginD()); rowOffsets.resize(numVert + 1, 0);
//   thrust::inclusive_scan(degree.beginD(), degree.endD(),
//                          rowOffsets.beginD() + 1);
// }

// struct Reachable {
//   __host__ __device__ bool operator()(int x) { return x != kNvgraphInvalid; }
// };

struct NextStart : public thrust::binary_function<int, int, bool> {
  __host__ __device__ bool operator()(int value, int component) {
    // mismatch finds the point where this is false, so this is inverted.
    return !(component >= 0 && value != kInvalidInt);
  }
};

struct NextLabel {
  __host__ __device__ bool operator()(int component) { return component >= 0; }
};

struct FloodComponent {
  int value;
  int label;

  __host__ __device__ void operator()(thrust::tuple<int&, int&> inOut) {
    int& valueOut = thrust::get<0>(inOut);
    int& labelOut = thrust::get<1>(inOut);

    if (labelOut == label) {
      labelOut = -1;
      valueOut = value;
    }
  }
};

}  // namespace

namespace manifold {

int ConnectedComponents(VecDH<int>& components, int numVert,
                        const VecDH<Halfedge>& halfedges,
                        const VecDH<bool>& keep) {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  // Using the CPU version of connected components even when GPU is available,
  // because it is dramatically faster.
  return ConnectedComponentsCPU(components, numVert, halfedges, keep);
#else
  return ConnectedComponentsCPU(components, numVert, halfedges, keep);
#endif
}

// int ConnectedComponentsGPU(VecDH<int>& components, int numVert,
//                            const VecDH<Halfedge>& halfedges,
//                            const VecDH<bool>& keep) {
//   VecDH<int> rowOffsets, sink, edgeMask;
//   Edges2CSR(rowOffsets, sink, edgeMask, halfedges, keep, numVert);
//   // Set up graph
//   nvgraphHandle_t handle;
//   nvgraphGraphDescr_t graph;
//   constexpr size_t vertDataSize = 2;
//   cudaDataType_t vertDataTypes[vertDataSize];
//   constexpr size_t distIdx = 0;
//   constexpr size_t predIdx = 1;
//   vertDataTypes[distIdx] = CUDA_R_32I;
//   vertDataTypes[predIdx] = CUDA_R_32I;
//   CheckStatus(nvgraphCreate(&handle));
//   CheckStatus(nvgraphCreateGraphDescr(handle, &graph));
//   // Set up traversal
//   nvgraphTraversalParameter_t traversal;
//   nvgraphTraversalParameterInit(&traversal);
//   nvgraphTraversalSetDistancesIndex(&traversal, distIdx);
//   nvgraphTraversalSetPredecessorsIndex(&traversal, predIdx);
//   nvgraphTraversalSetUndirectedFlag(&traversal, false);
//   // Fill graph
//   nvgraphCSRTopology32I_st adjacencyCSR;
//   adjacencyCSR.nvertices = numVert;
//   adjacencyCSR.nedges = sink.size();
//   adjacencyCSR.source_offsets = rowOffsets.ptrD();
//   adjacencyCSR.destination_indices = sink.ptrD();
//   CheckStatus(nvgraphSetGraphStructure(
//       handle, graph, static_cast<void*>(&adjacencyCSR), NVGRAPH_CSR_32));
//   CheckStatus(
//       nvgraphAllocateVertexData(handle, graph, vertDataSize, vertDataTypes));
//   // Apply mask if given
//   if (keep.size() > 0) {
//     constexpr size_t edgeDataSize = 1, maskIdx = 0;
//     cudaDataType_t edgeDataTypes[edgeDataSize];
//     edgeDataTypes[maskIdx] = CUDA_R_32I;
//     CheckStatus(
//         nvgraphAllocateEdgeData(handle, graph, edgeDataSize, edgeDataTypes));
//     CheckStatus(nvgraphSetEdgeData(
//         handle, graph, static_cast<void*>(edgeMask.ptrD()), maskIdx));
//     CheckStatus(nvgraphTraversalSetEdgeMaskIndex(&traversal, maskIdx));
//   }
//   components.resize(numVert);
//   thrust::fill(components.beginD(), components.endD(), -1);
//   VecDH<int> distBFS(numVert);
//   int numComponent = 0;
//   while (1) {
//     // Find the first vertex that hasn't been visited
//     int sourceVert = thrust::find(components.beginD(), components.endD(), -1)
//     -
//                      components.beginD();
//     if (sourceVert >= components.size()) break;
//     // Find the sourceVert connected component using breadth-first search
//     CheckStatus(nvgraphTraversal(handle, graph, NVGRAPH_TRAVERSAL_BFS,
//                                  &sourceVert, traversal));
//     CheckStatus(nvgraphGetVertexData(
//         handle, graph, static_cast<void*>(distBFS.ptrD()), distIdx));
//     // Use numComponent as the component label
//     thrust::replace_if(components.beginD(), components.endD(),
//     distBFS.beginD(),
//                        Reachable(), numComponent++);
//   }

//   CheckStatus(nvgraphDestroyGraphDescr(handle, graph));
//   CheckStatus(nvgraphDestroy(handle));

//   return numComponent;
// }

int ConnectedComponentsCPU(VecDH<int>& components, int numVert,
                           const VecDH<Halfedge>& halfedges,
                           const VecDH<bool>& keep) {
  boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph(
      numVert);
  for (int i = 0; i < halfedges.size(); ++i) {
    const Halfedge halfedge = halfedges.H()[i];
    if (halfedge.IsForward() && (keep.size() == 0 || keep.H()[i])) {
      boost::add_edge(halfedge.startVert, halfedge.endVert, graph);
    }
  }
  components.resize(numVert);
  int numComponent = boost::connected_components(graph, components.H().data());
  return numComponent;
}

void FloodComponents(VecDH<int>& valuesInOut, VecDH<int>& componentLabels,
                     int numComponent) {
  // componentLabels will be replaced entirely with -1
  ALWAYS_ASSERT(valuesInOut.size() == componentLabels.size(), logicErr,
                "These vectors must both be NumVert long.");
  for (int comp = 0; comp < numComponent; ++comp) {
    // find first vertex in component that is also has a value
    int sourceVert = thrust::mismatch(valuesInOut.begin(), valuesInOut.end(),
                                      componentLabels.begin(), NextStart())
                         .first -
                     valuesInOut.begin();
    int label, value;
    if (sourceVert < valuesInOut.size()) {
      label = componentLabels.H()[sourceVert];
      value = valuesInOut.H()[sourceVert];
    } else {
      // If no vertices in a component have a value, then their value must be
      // zero, because zeros are removed from the sparse representation.
      sourceVert = thrust::find_if(componentLabels.begin(),
                                   componentLabels.end(), NextLabel()) -
                   componentLabels.begin();
      label = componentLabels.H()[sourceVert];
      value = 0;
      ALWAYS_ASSERT(sourceVert < componentLabels.size(), logicErr,
                    "Failed to find component!");
    }
    thrust::for_each_n(zip(valuesInOut.beginD(), componentLabels.beginD()),
                       valuesInOut.size(), FloodComponent({value, label}));
  }
}

}  // namespace manifold