// Copyright 2021 Emmett Lalish
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

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include "connected_components.cuh"

namespace {
using namespace manifold;

struct InitTriLabels {
  const Halfedge* halfedge;

  __host__ __device__ void operator()(thrust::tuple<int, int&> inOut) {
    const int edge = 3 * thrust::get<0>(inOut);
    int& label = thrust::get<1>(inOut);

    label = glm::min(
        glm::min(halfedge[edge].startVert, halfedge[edge + 1].startVert),
        halfedge[edge + 2].startVert);
  }
};

struct UpdateTriLabels {
  int* triLabel;
  bool* isFinished;
  const Halfedge* halfedge;
  const bool* keep;

  __host__ __device__ void operator()(int tri) {
    const int edge = 3 * tri;
    if (keep[edge] && keep[edge + 1] && keep[edge + 2]) {
      int label = triLabel[tri];
      label = glm::min(
          glm::min(glm::min(label, triLabel[halfedge[edge].pairedHalfedge / 3]),
                   triLabel[halfedge[edge + 1].pairedHalfedge / 3]),
          triLabel[halfedge[edge + 2].pairedHalfedge / 3]);

      if (label != triLabel[tri]) {
        triLabel[tri] = label;
        *isFinished = false;
      }
    }
  }
};

struct TriLabelsToVert {
  int* vertLabel;
  const Halfedge* halfedge;
  const bool* keep;

  __host__ __device__ void operator()(thrust::tuple<int, int> in) {
    const int edge = 3 * thrust::get<0>(in);
    const int label = thrust::get<1>(in);

    if (keep[edge] && keep[edge + 1] && keep[edge + 2]) {
      for (const int i : {0, 1, 2}) {
        vertLabel[halfedge[edge + i].startVert] = label;
      }
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

int ConnectedComponentsGPU(VecDH<int>& components, int numVert,
                           const VecDH<Halfedge>& halfedges,
                           const VecDH<bool>& keep) {
  const int numTri = halfedges.size() / 3;
  VecDH<int> triLabel(numTri);
  thrust::for_each_n(zip(countAt(0), triLabel.beginD()), numTri,
                     InitTriLabels({halfedges.cptrD()}));

  VecDH<bool> isFinished(1, false);
  while (!isFinished.H()[0]) {
    isFinished.H()[0] = true;
    thrust::for_each_n(countAt(0), numTri,
                       UpdateTriLabels({triLabel.ptrD(), isFinished.ptrD(),
                                        halfedges.cptrD(), keep.cptrD()}));
  }

  components.resize(numVert);
  thrust::for_each_n(
      zip(countAt(0), triLabel.beginD()), numTri,
      TriLabelsToVert({components.ptrD(), halfedges.cptrD(), keep.cptrD()}));

  VecDH<int> minVerts = components;
  thrust::sort(minVerts.beginD(), minVerts.endD());
  int numComponent =
      thrust::unique(minVerts.beginD(), minVerts.endD()) - minVerts.beginD();
  return numComponent;
}

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
}  // namespace manifold