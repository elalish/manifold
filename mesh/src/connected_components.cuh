#pragma once
#include "mesh.cuh"

namespace manifold {

int ConnectedComponents(VecDH<int>& components, int numVert,
                        const VecDH<EdgeVertsD>& edgeVerts,
                        const VecDH<bool>& keep = VecDH<bool>());

int ConnectedComponentsGPU(VecDH<int>& components, int numVert,
                           const VecDH<EdgeVertsD>& edgeVerts,
                           const VecDH<bool>& keep = VecDH<bool>());

int ConnectedComponentsCPU(VecDH<int>& components, int numVert,
                           const VecDH<EdgeVertsD>& edgeVerts,
                           const VecDH<bool>& keep = VecDH<bool>());

void FloodComponents(VecDH<int>& valuesInOut, VecDH<int>& componentsToDie,
                     int numComponent);
}