#include "impl.h"

namespace {
using namespace manifold;

__host__ __device__ inline int FlipHalfedge(int halfedge) {
  const int tri = halfedge / 3;
  const int vert = 2 - (halfedge - 3 * tri);
  return 3 * tri + vert;
}

struct TransformNormals {
  const glm::mat3 transform;

  __host__ __device__ glm::vec3 operator()(glm::vec3 normal) {
    normal = glm::normalize(transform * normal);
    if (isnan(normal.x)) normal = glm::vec3(0.0f);
    return normal;
  }
};

struct TransformTangents {
  const glm::mat3 transform;
  const bool invert;
  const glm::vec4* oldTangents;
  const Halfedge* halfedge;

  __host__ __device__ void operator()(thrust::tuple<glm::vec4&, int> inOut) {
    glm::vec4& tangent = thrust::get<0>(inOut);
    int edge = thrust::get<1>(inOut);
    if (invert) {
      edge = halfedge[FlipHalfedge(edge)].pairedHalfedge;
    }

    tangent = glm::vec4(transform * glm::vec3(oldTangents[edge]),
                        oldTangents[edge].w);
  }
};

struct FlipTris {
  Halfedge* halfedge;

  __host__ __device__ void operator()(thrust::tuple<TriRef&, int> inOut) {
    TriRef& bary = thrust::get<0>(inOut);
    const int tri = thrust::get<1>(inOut);

    thrust::swap(halfedge[3 * tri], halfedge[3 * tri + 2]);

    for (const int i : {0, 1, 2}) {
      thrust::swap(halfedge[3 * tri + i].startVert,
                   halfedge[3 * tri + i].endVert);
      halfedge[3 * tri + i].pairedHalfedge =
          FlipHalfedge(halfedge[3 * tri + i].pairedHalfedge);
    }
  }
};
}  // namespace
