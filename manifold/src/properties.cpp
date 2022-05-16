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

#include <thrust/count.h>
#include <thrust/logical.h>
#include <thrust/transform_reduce.h>
#include <limits>

#include "impl.h"

namespace {
using namespace manifold;

struct FaceAreaVolume {
  const Halfedge* halfedges;
  const glm::vec3* vertPos;
  const float precision;

  __host__ __device__ thrust::pair<float, float> operator()(int face) {
    float perimeter = 0;
    glm::vec3 edge[3];
    for (int i : {0, 1, 2}) {
      const int j = (i + 1) % 3;
      edge[i] = vertPos[halfedges[3 * face + j].startVert] -
                vertPos[halfedges[3 * face + i].startVert];
      perimeter += glm::length(edge[i]);
    }
    glm::vec3 crossP = glm::cross(edge[0], edge[1]);

    float area = glm::length(crossP);
    float volume = glm::dot(crossP, vertPos[halfedges[3 * face].startVert]);

    return area > perimeter * precision
               ? thrust::make_pair(area / 2.0f, volume / 6.0f)
               : thrust::make_pair(0.0f, 0.0f);
  }
};

struct PosMin
    : public thrust::binary_function<glm::vec3, glm::vec3, glm::vec3> {
  __host__ __device__ glm::vec3 operator()(glm::vec3 a, glm::vec3 b) {
    if (isnan(a.x)) return b;
    if (isnan(b.x)) return a;
    return glm::min(a, b);
  }
};

struct PosMax
    : public thrust::binary_function<glm::vec3, glm::vec3, glm::vec3> {
  __host__ __device__ glm::vec3 operator()(glm::vec3 a, glm::vec3 b) {
    if (isnan(a.x)) return b;
    if (isnan(b.x)) return a;
    return glm::max(a, b);
  }
};

struct SumPair : public thrust::binary_function<thrust::pair<float, float>,
                                                thrust::pair<float, float>,
                                                thrust::pair<float, float>> {
  __host__ __device__ thrust::pair<float, float> operator()(
      thrust::pair<float, float> a, thrust::pair<float, float> b) {
    a.first += b.first;
    a.second += b.second;
    return a;
  }
};

struct CurvatureAngles {
  float* meanCurvature;
  float* gaussianCurvature;
  float* area;
  float* degree;
  const Halfedge* halfedge;
  const glm::vec3* vertPos;
  const glm::vec3* triNormal;

  __host__ __device__ void operator()(int tri) {
    glm::vec3 edge[3];
    glm::vec3 edgeLength(0.0);
    for (int i : {0, 1, 2}) {
      const int startVert = halfedge[3 * tri + i].startVert;
      const int endVert = halfedge[3 * tri + i].endVert;
      edge[i] = vertPos[endVert] - vertPos[startVert];
      edgeLength[i] = glm::length(edge[i]);
      edge[i] /= edgeLength[i];
      const int neighborTri = halfedge[3 * tri + i].pairedHalfedge / 3;
      const float dihedral =
          0.25 * edgeLength[i] *
          glm::asin(glm::dot(glm::cross(triNormal[tri], triNormal[neighborTri]),
                             edge[i]));
      AtomicAdd(meanCurvature[startVert], dihedral);
      AtomicAdd(meanCurvature[endVert], dihedral);
      AtomicAdd(degree[startVert], 1.0f);
    }

    glm::vec3 phi;
    phi[0] = glm::acos(-glm::dot(edge[2], edge[0]));
    phi[1] = glm::acos(-glm::dot(edge[0], edge[1]));
    phi[2] = glm::pi<float>() - phi[0] - phi[1];
    const float area3 = edgeLength[0] * edgeLength[1] *
                        glm::length(glm::cross(edge[0], edge[1])) / 6;

    for (int i : {0, 1, 2}) {
      const int vert = halfedge[3 * tri + i].startVert;
      AtomicAdd(gaussianCurvature[vert], -phi[i]);
      AtomicAdd(area[vert], area3);
    }
  }
};

struct NormalizeCurvature {
  __host__ __device__ void operator()(
      thrust::tuple<float&, float&, float, float> inOut) {
    float& meanCurvature = thrust::get<0>(inOut);
    float& gaussianCurvature = thrust::get<1>(inOut);
    float area = thrust::get<2>(inOut);
    float degree = thrust::get<3>(inOut);
    float factor = degree / (6 * area);
    meanCurvature *= factor;
    gaussianCurvature *= factor;
  }
};

struct CheckManifold {
  const Halfedge* halfedges;

  __host__ __device__ bool operator()(int edge) {
    const Halfedge halfedge = halfedges[edge];
    if (halfedge.startVert == -1 && halfedge.endVert == -1 &&
        halfedge.pairedHalfedge == -1)
      return true;

    const Halfedge paired = halfedges[halfedge.pairedHalfedge];
    bool good = true;
    good &= paired.pairedHalfedge == edge;
    good &= halfedge.startVert != halfedge.endVert;
    good &= halfedge.startVert == paired.endVert;
    good &= halfedge.endVert == paired.startVert;
    return good;
  }
};

struct NoDuplicates {
  const Halfedge* halfedges;

  __host__ __device__ bool operator()(int edge) {
    const Halfedge halfedge = halfedges[edge];
    if (halfedge.startVert == -1 && halfedge.endVert == -1 &&
        halfedge.pairedHalfedge == -1)
      return true;
    return halfedge.startVert != halfedges[edge + 1].startVert ||
           halfedge.endVert != halfedges[edge + 1].endVert;
  }
};

struct CheckCCW {
  const Halfedge* halfedges;
  const glm::vec3* vertPos;
  const glm::vec3* triNormal;
  const float tol;

  __host__ __device__ bool operator()(int face) {
    if (halfedges[3 * face].pairedHalfedge < 0) return true;

    const glm::mat3x2 projection = GetAxisAlignedProjection(triNormal[face]);
    glm::vec2 v[3];
    for (int i : {0, 1, 2})
      v[i] = projection * vertPos[halfedges[3 * face + i].startVert];

    int ccw = CCW(v[0], v[1], v[2], glm::abs(tol));
    bool check = tol > 0 ? ccw >= 0 : ccw == 0;

    if (tol > 0 && !check) {
      glm::vec2 v1 = v[1] - v[0];
      glm::vec2 v2 = v[2] - v[0];
      float area = v1.x * v2.y - v1.y * v2.x;
      float base2 = glm::max(glm::dot(v1, v1), glm::dot(v2, v2));
      float base = glm::sqrt(base2);
      glm::vec3 V0 = vertPos[halfedges[3 * face].startVert];
      glm::vec3 V1 = vertPos[halfedges[3 * face + 1].startVert];
      glm::vec3 V2 = vertPos[halfedges[3 * face + 2].startVert];
      glm::vec3 norm = glm::cross(V1 - V0, V2 - V0);
      printf(
          "Tri %d does not match normal, approx height = %g, base = %g\n"
          "tol = %g, area2 = %g, base2*tol2 = %g\n"
          "normal = %g, %g, %g\n"
          "norm = %g, %g, %g\nverts: %d, %d, %d\n",
          face, area / base, base, tol, area * area, base2 * tol * tol,
          triNormal[face].x, triNormal[face].y, triNormal[face].z, norm.x,
          norm.y, norm.z, halfedges[3 * face].startVert,
          halfedges[3 * face + 1].startVert, halfedges[3 * face + 2].startVert);
    }
    return check;
  }
};
}  // namespace

namespace manifold {

/**
 * Returns true if this manifold is in fact an oriented 2-manifold and all of
 * the data structures are consistent.
 */
bool Manifold::Impl::IsManifold() const {
  if (halfedge_.size() == 0) return true;
  bool isManifold = thrust::all_of(thrust::device, countAt(0), countAt(halfedge_.size()),
                                   CheckManifold({halfedge_.cptrD()}));

  VecDH<Halfedge> halfedge(halfedge_);
  thrust::sort(thrust::device, halfedge.begin(), halfedge.end());
  isManifold &= thrust::all_of(thrust::device, countAt(0), countAt(2 * NumEdge() - 1),
                               NoDuplicates({halfedge.cptrD()}));
  return isManifold;
}

/**
 * Returns true if all triangles are CCW relative to their triNormals_.
 */
bool Manifold::Impl::MatchesTriNormals() const {
  if (halfedge_.size() == 0 || faceNormal_.size() != NumTri()) return true;
  return thrust::all_of(thrust::device, countAt(0), countAt(NumTri()),
                        CheckCCW({halfedge_.cptrD(), vertPos_.cptrD(),
                                  faceNormal_.cptrD(), 2 * precision_}));
}

/**
 * Returns the number of triangles that are colinear within precision_.
 */
int Manifold::Impl::NumDegenerateTris() const {
  if (halfedge_.size() == 0 || faceNormal_.size() != NumTri()) return true;
  return thrust::count_if(thrust::device, countAt(0), countAt(NumTri()),
                          CheckCCW({halfedge_.cptrD(), vertPos_.cptrD(),
                                    faceNormal_.cptrD(), -1 * precision_ / 2}));
}

Properties Manifold::Impl::GetProperties() const {
  if (IsEmpty()) return {0, 0};
  ApplyTransform();
  thrust::pair<float, float> areaVolume = thrust::transform_reduce(
      thrust::device, countAt(0), countAt(NumTri()),
      FaceAreaVolume({halfedge_.cptrD(), vertPos_.cptrD(), precision_}),
      thrust::make_pair(0.0f, 0.0f), SumPair());
  return {areaVolume.first, areaVolume.second};
}

Curvature Manifold::Impl::GetCurvature() const {
  Curvature result;
  if (IsEmpty()) return result;
  ApplyTransform();
  VecDH<float> vertMeanCurvature(NumVert(), 0);
  VecDH<float> vertGaussianCurvature(NumVert(), glm::two_pi<float>());
  VecDH<float> vertArea(NumVert(), 0);
  VecDH<float> degree(NumVert(), 0);
  thrust::for_each(
      thrust::device, countAt(0), countAt(NumTri()),
      CurvatureAngles({vertMeanCurvature.ptrD(), vertGaussianCurvature.ptrD(),
                       vertArea.ptrD(), degree.ptrD(), halfedge_.cptrD(),
                       vertPos_.cptrD(), faceNormal_.cptrD()}));
  thrust::for_each_n(
      thrust::device, zip(vertMeanCurvature.begin(), vertGaussianCurvature.begin(),
          vertArea.begin(), degree.begin()),
      NumVert(), NormalizeCurvature());
  result.minMeanCurvature =
      thrust::reduce(thrust::device, vertMeanCurvature.begin(), vertMeanCurvature.end(),
                     std::numeric_limits<float>::infinity(), thrust::minimum<float>());
  result.maxMeanCurvature =
      thrust::reduce(thrust::device, vertMeanCurvature.begin(), vertMeanCurvature.end(),
                     -std::numeric_limits<float>::infinity(), thrust::maximum<float>());
  result.minGaussianCurvature = thrust::reduce(
      thrust::device, vertGaussianCurvature.begin(), vertGaussianCurvature.end(), std::numeric_limits<float>::infinity(),
      thrust::minimum<float>());
  result.maxGaussianCurvature = thrust::reduce(
      thrust::device, vertGaussianCurvature.begin(), vertGaussianCurvature.end(),
      -std::numeric_limits<float>::infinity(), thrust::maximum<float>());
  result.vertMeanCurvature.insert(result.vertMeanCurvature.end(),
                                  vertMeanCurvature.begin(),
                                  vertMeanCurvature.end());
  result.vertGaussianCurvature.insert(result.vertGaussianCurvature.end(),
                                      vertGaussianCurvature.begin(),
                                      vertGaussianCurvature.end());
  return result;
}

/**
 * Calculates the bounding box of the entire manifold, which is stored
 * internally to short-cut Boolean operations and to serve as the precision
 * range for Morton code calculation.
 */
void Manifold::Impl::CalculateBBox() {
  bBox_.min = thrust::reduce(thrust::device, vertPos_.begin(), vertPos_.end(),
                             glm::vec3(std::numeric_limits<float>::infinity()), PosMin());
  bBox_.max = thrust::reduce(thrust::device, vertPos_.begin(), vertPos_.end(),
                             glm::vec3(-std::numeric_limits<float>::infinity()), PosMax());
}
}  // namespace manifold
