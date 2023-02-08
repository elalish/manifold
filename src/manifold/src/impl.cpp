// Copyright 2021 The Manifold Authors.
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

#include "impl.h"

#include <thrust/logical.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <numeric>

#include "graph.h"
#include "hashtable.h"
#include "par.h"

namespace {
using namespace manifold;

__host__ __device__ void AtomicAddVec3(glm::vec3& target,
                                       const glm::vec3& add) {
  for (int i : {0, 1, 2}) {
#ifdef __CUDA_ARCH__
    atomicAdd(&target[i], add[i]);
#else
    std::atomic<float>& tar = reinterpret_cast<std::atomic<float>&>(target[i]);
    float old_val = tar.load(std::memory_order_relaxed);
    while (!tar.compare_exchange_weak(old_val, old_val + add[i],
                                      std::memory_order_relaxed))
      ;
#endif
  }
}

__host__ __device__ int FlipHalfedge(int halfedge) {
  const int tri = halfedge / 3;
  const int vert = 2 - (halfedge - 3 * tri);
  return 3 * tri + vert;
}

struct Normalize {
  __host__ __device__ void operator()(glm::vec3& v) { v = SafeNormalize(v); }
};

struct Transform4x3 {
  const glm::mat4x3 transform;

  __host__ __device__ glm::vec3 operator()(glm::vec3 position) {
    return transform * glm::vec4(position, 1.0f);
  }
};

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

struct AssignNormals {
  glm::vec3* vertNormal;
  const glm::vec3* vertPos;
  const Halfedge* halfedges;
  const float precision;
  const bool calculateTriNormal;

  __host__ __device__ void operator()(thrust::tuple<glm::vec3&, int> in) {
    glm::vec3& triNormal = thrust::get<0>(in);
    const int face = thrust::get<1>(in);

    glm::ivec3 triVerts;
    for (int i : {0, 1, 2}) triVerts[i] = halfedges[3 * face + i].startVert;

    glm::vec3 edge[3];
    for (int i : {0, 1, 2}) {
      const int j = (i + 1) % 3;
      edge[i] = glm::normalize(vertPos[triVerts[j]] - vertPos[triVerts[i]]);
    }

    if (calculateTriNormal) {
      triNormal = glm::normalize(glm::cross(edge[0], edge[1]));
      if (isnan(triNormal.x)) triNormal = glm::vec3(0, 0, 1);
    }

    // corner angles
    glm::vec3 phi;
    float dot = -glm::dot(edge[2], edge[0]);
    phi[0] = dot >= 1 ? 0 : (dot <= -1 ? glm::pi<float>() : glm::acos(dot));
    dot = -glm::dot(edge[0], edge[1]);
    phi[1] = dot >= 1 ? 0 : (dot <= -1 ? glm::pi<float>() : glm::acos(dot));
    phi[2] = glm::pi<float>() - phi[0] - phi[1];

    // assign weighted sum
    for (int i : {0, 1, 2}) {
      AtomicAddVec3(vertNormal[triVerts[i]], phi[i] * triNormal);
    }
  }
};

struct Tri2Halfedges {
  Halfedge* halfedges;
  glm::uint64_t* edges;

  __host__ __device__ void operator()(
      thrust::tuple<int, const glm::ivec3&> in) {
    const int tri = thrust::get<0>(in);
    const glm::ivec3& triVerts = thrust::get<1>(in);
    for (const int i : {0, 1, 2}) {
      const int j = (i + 1) % 3;
      const int edge = 3 * tri + i;
      halfedges[edge] = {triVerts[i], triVerts[j], -1, tri};
      // Sort the forward halfedges in front of the backward ones by setting the
      // highest-order bit.
      edges[edge] = glm::uint64_t(triVerts[i] < triVerts[j] ? 1 : 0) << 63 |
                    ((glm::uint64_t)glm::min(triVerts[i], triVerts[j])) << 32 |
                    glm::max(triVerts[i], triVerts[j]);
    }
  }
};

struct LinkHalfedges {
  Halfedge* halfedges;
  const int* ids;
  const int numEdge;

  __host__ __device__ void operator()(int i) {
    const int pair0 = ids[i];
    const int pair1 = ids[i + numEdge];
    halfedges[pair0].pairedHalfedge = pair1;
    halfedges[pair1].pairedHalfedge = pair0;
  }
};

struct MarkVerts {
  int* vert;

  __host__ __device__ void operator()(glm::ivec3 triVerts) {
    for (int i : {0, 1, 2}) {
      vert[triVerts[i]] = 1;
    }
  }
};

struct ReindexTriVerts {
  const int* old2new;

  __host__ __device__ void operator()(glm::ivec3& triVerts) {
    for (int i : {0, 1, 2}) {
      triVerts[i] = old2new[triVerts[i]];
    }
  }
};

struct InitializeTriRef {
  const int meshID;
  const Halfedge* halfedge;

  __host__ __device__ void operator()(thrust::tuple<TriRef&, int> inOut) {
    TriRef& baryRef = thrust::get<0>(inOut);
    int tri = thrust::get<1>(inOut);

    baryRef.meshID = meshID;
    baryRef.originalID = meshID;
    baryRef.tri = tri;
  }
};

struct MarkMeshID {
  HashTableD<uint32_t> table;

  __host__ __device__ void operator()(TriRef& ref) {
    if (table.Full()) return;
    table.Insert(ref.meshID, 1);
  }
};

struct UpdateMeshID {
  const HashTableD<uint32_t> meshIDold2new;
  const int meshIDoffset;

  __host__ __device__ void operator()(TriRef& ref) {
    ref.meshID = meshIDold2new[ref.meshID] + meshIDoffset;
  }
};

struct CoplanarEdge {
  float* triArea;
  const Halfedge* halfedge;
  const glm::vec3* vertPos;
  const TriRef* triRef;
  const glm::ivec3* triProp;
  const float* prop;
  const float* propTol;
  const int numProp;
  const float precision;

  __host__ __device__ void operator()(
      thrust::tuple<thrust::pair<int, int>&, thrust::pair<int, int>&, int>
          inOut) {
    thrust::pair<int, int>& face2face = thrust::get<0>(inOut);
    thrust::pair<int, int>& vert2vert = thrust::get<1>(inOut);
    const int edgeIdx = thrust::get<2>(inOut);

    const Halfedge edge = halfedge[edgeIdx];
    if (!edge.IsForward()) return;
    const Halfedge pair = halfedge[edge.pairedHalfedge];
    if (triRef[edge.face].meshID != triRef[pair.face].meshID) return;

    const glm::vec3 base = vertPos[edge.startVert];

    const int baseNum = edgeIdx - 3 * edge.face;
    const int jointNum = edge.pairedHalfedge - 3 * pair.face;
    const int edgeNum = baseNum == 0 ? 2 : baseNum - 1;
    const int pairNum = jointNum == 0 ? 2 : jointNum - 1;

    if (numProp > 0) {
      const int prop0 = triProp[edge.face][baseNum];
      const int prop1 = triProp[edge.face][edgeNum];
      bool propEqual = true;
      for (int p = 0; p < numProp; ++p) {
        if (glm::abs(prop[numProp * prop0 + p] - prop[numProp * prop1 + p]) >
            propTol[p]) {
          propEqual = false;
          break;
        }
      }
      if (propEqual) {
        vert2vert.first = prop0;
        vert2vert.second = prop1;
      }
    }

    const glm::vec3 jointVec = vertPos[pair.startVert] - base;
    const glm::vec3 edgeVec =
        vertPos[halfedge[3 * edge.face + edgeNum].startVert] - base;
    const glm::vec3 pairVec =
        vertPos[halfedge[3 * pair.face + pairNum].startVert] - base;

    const float length = glm::max(glm::length(jointVec), glm::length(edgeVec));
    const float lengthPair =
        glm::max(glm::length(jointVec), glm::length(pairVec));
    glm::vec3 normal = glm::cross(jointVec, edgeVec);
    const float area = glm::length(normal);
    const float areaPair = glm::length(glm::cross(pairVec, jointVec));
    triArea[edge.face] = area;
    triArea[pair.face] = areaPair;
    // Don't link degenerate triangles
    if (area < length * precision || areaPair < lengthPair * precision) return;

    const float volume = glm::abs(glm::dot(normal, pairVec));
    // Only operate on coplanar triangles
    if (volume > glm::max(area, areaPair) * precision) return;

    // Check property linearity
    if (area > 0) {
      normal /= area;
      for (int i = 0; i < numProp; ++i) {
        const float scale = precision / propTol[i];

        const float baseProp = prop[numProp * triProp[edge.face][baseNum] + i];
        const float jointProp =
            prop[numProp * triProp[pair.face][jointNum] + i];
        const float edgeProp = prop[numProp * triProp[edge.face][edgeNum] + i];
        const float pairProp = prop[numProp * triProp[pair.face][pairNum] + i];

        const glm::vec3 iJointVec =
            jointVec + normal * scale * (jointProp - baseProp);
        const glm::vec3 iEdgeVec =
            edgeVec + normal * scale * (edgeProp - baseProp);
        const glm::vec3 iPairVec =
            pairVec + normal * scale * (pairProp - baseProp);

        glm::vec3 cross = glm::cross(iJointVec, iEdgeVec);
        const float areaP = glm::max(
            glm::length(cross), glm::length(glm::cross(iPairVec, iJointVec)));
        const float volumeP = glm::abs(glm::dot(cross, iPairVec));
        // Only operate on consistent triangles
        if (volumeP > areaP * precision) return;
      }
    }

    face2face.first = edge.face;
    face2face.second = pair.face;
  }
};

struct EdgeBox {
  const glm::vec3* vertPos;

  __host__ __device__ void operator()(
      thrust::tuple<Box&, const TmpEdge&> inout) {
    const TmpEdge& edge = thrust::get<1>(inout);
    thrust::get<0>(inout) = Box(vertPos[edge.first], vertPos[edge.second]);
  }
};

int GetLabels(std::vector<int>& components,
              const VecDH<thrust::pair<int, int>>& edges, int numNodes) {
  Graph graph;
  for (int i = 0; i < numNodes; ++i) {
    graph.add_nodes(i);
  }
  for (int i = 0; i < edges.size(); ++i) {
    const thrust::pair<int, int> edge = edges[i];
    if (edge.first < 0) continue;
    graph.add_edge(edge.first, edge.second);
  }

  return ConnectedComponents(components, graph);
}

void DedupePropVerts(manifold::VecDH<glm::ivec3>& triProp,
                     const VecDH<thrust::pair<int, int>>& vert2vert) {
  std::vector<int> vertLabels;
  const int numLabels = GetLabels(vertLabels, vert2vert, vert2vert.size());

  std::vector<int> label2vert(numLabels);
  for (int v = 0; v < vert2vert.size(); ++v) {
    label2vert[vertLabels[v]] = v;
  }
  for (int tri = 0; tri < triProp.size(); ++tri) {
    for (int i : {0, 1, 2})
      triProp[tri][i] = label2vert[vertLabels[triProp[tri][i]]];
  }
}
}  // namespace

namespace manifold {

std::atomic<uint32_t> Manifold::Impl::meshIDCounter_(1);

uint32_t Manifold::Impl::ReserveIDs(uint32_t n) {
  return Manifold::Impl::meshIDCounter_.fetch_add(n, std::memory_order_relaxed);
}

Manifold::Impl::Impl(const MeshGL& meshGL,
                     std::vector<float> propertyTolerance) {
  Mesh mesh;
  const int numVert = meshGL.NumVert();
  const int numTri = meshGL.NumTri();

  if (meshGL.numProp < 3) {
    MarkFailure(Error::MISSING_POSITION_PROPERTIES);
    return;
  }

  mesh.triVerts.resize(numTri);
  if (meshGL.mergeFromVert.size() != meshGL.mergeToVert.size()) {
    MarkFailure(Error::MERGE_VECTORS_DIFFERENT_LENGTHS);
    return;
  }

  if (!meshGL.transform.empty() &&
      12 * meshGL.originalID.size() != meshGL.transform.size()) {
    MarkFailure(Error::TRANSFORM_WRONG_LENGTH);
    return;
  }

  if (!meshGL.originalID.empty() && !meshGL.runIndex.empty() &&
      meshGL.originalID.size() + 1 != meshGL.runIndex.size()) {
    MarkFailure(Error::RUN_INDEX_WRONG_LENGTH);
    return;
  }

  if (!meshGL.faceID.empty() && meshGL.faceID.size() != meshGL.NumTri()) {
    MarkFailure(Error::FACE_ID_WRONG_LENGTH);
    return;
  }

  std::vector<int> prop2vert(numVert);
  std::iota(prop2vert.begin(), prop2vert.end(), 0);
  for (int i = 0; i < meshGL.mergeFromVert.size(); ++i) {
    const int from = meshGL.mergeFromVert[i];
    const int to = meshGL.mergeToVert[i];
    if (from >= numVert || to >= numVert) {
      MarkFailure(Error::MERGE_INDEX_OUT_OF_BOUNDS);
      return;
    }
    prop2vert[from] = to;
  }
  for (int i = 0; i < numTri; ++i) {
    for (const int j : {0, 1, 2}) {
      const int vert = meshGL.triVerts[3 * i + j];
      if (vert < 0 || vert >= numVert) {
        MarkFailure(Error::VERTEX_INDEX_OUT_OF_BOUNDS);
        return;
      }
      mesh.triVerts[i][j] = prop2vert[vert];
    }
  }

  MeshRelationD relation;

  if (meshGL.numProp > 3) {
    relation.triProperties.resize(numTri);
    for (int i = 0; i < numTri; ++i) {
      for (const int j : {0, 1, 2}) {
        relation.triProperties[i][j] = meshGL.triVerts[3 * i + j];
      }
    }
  }

  const int numProp = meshGL.numProp - 3;
  relation.numProp = numProp;
  relation.properties.resize(meshGL.NumVert() * numProp);
  // This will have unreferenced duplicate positions that will be removed by
  // Impl::RemoveUnreferencedVerts().
  mesh.vertPos.resize(meshGL.NumVert());

  for (int i = 0; i < meshGL.NumVert(); ++i) {
    for (const int j : {0, 1, 2})
      mesh.vertPos[i][j] = meshGL.vertProperties[meshGL.numProp * i + j];
    for (int j = 0; j < numProp; ++j)
      relation.properties[i * numProp + j] =
          meshGL.vertProperties[meshGL.numProp * i + 3 + j];
  }

  mesh.halfedgeTangent.resize(meshGL.halfedgeTangent.size() / 4);
  for (int i = 0; i < mesh.halfedgeTangent.size(); ++i) {
    for (const int j : {0, 1, 2, 3})
      mesh.halfedgeTangent[i][j] = meshGL.halfedgeTangent[4 * i + j];
  }

  if (meshGL.originalID.empty()) {
    relation.originalID = Impl::ReserveIDs(1);
  } else {
    std::vector<uint32_t> runIndex = meshGL.runIndex;
    if (runIndex.empty()) {
      runIndex = {0, 3 * meshGL.NumTri()};
    }
    relation.triRef.resize(meshGL.NumTri());
    const int startID = Impl::ReserveIDs(meshGL.originalID.size());
    for (int i = 0; i < meshGL.originalID.size(); ++i) {
      const int meshID = startID + i;
      const int originalID = meshGL.originalID[i];
      for (int tri = runIndex[i] / 3; tri < runIndex[i + 1] / 3; ++tri) {
        TriRef& ref = relation.triRef[tri];
        ref.meshID = meshID;
        ref.originalID = originalID;
        ref.tri = meshGL.faceID.empty() ? tri : meshGL.faceID[tri];
      }

      if (meshGL.transform.empty()) {
        relation.meshIDtransform[meshID] = {};
      } else {
        const float* m = meshGL.transform.data() + 12 * i;
        relation.meshIDtransform[meshID] = {{m[0], m[1], m[2], m[3], m[4], m[5],
                                             m[6], m[7], m[8], m[9], m[10],
                                             m[11]}};
      }
    }
  }

  *this = Impl(mesh, relation, propertyTolerance, !meshGL.faceID.empty());

  // A Manifold created from an input mesh is never an original - the input is
  // the original.
  meshRelation_.originalID = -1;
}

/**
 * Create a manifold from an input triangle Mesh. Will return an empty Manifold
 * and set an Error Status if the Mesh is not manifold or otherwise invalid.
 * TODO: update halfedgeTangent during SimplifyTopology.
 */
Manifold::Impl::Impl(const Mesh& mesh, const MeshRelationD& relation,
                     const std::vector<float>& propertyTolerance,
                     bool hasFaceIDs)
    : vertPos_(mesh.vertPos),
      halfedgeTangent_(mesh.halfedgeTangent),
      meshRelation_(relation) {
  VecDH<glm::ivec3> triVerts = mesh.triVerts;
  if (!IsIndexInBounds(triVerts)) {
    MarkFailure(Error::VERTEX_INDEX_OUT_OF_BOUNDS);
    return;
  }
  RemoveUnreferencedVerts(triVerts);

  CalculateBBox();
  if (!IsFinite()) {
    MarkFailure(Error::NON_FINITE_VERTEX);
    return;
  }
  SetPrecision();

  CreateHalfedges(triVerts);
  if (!IsManifold()) {
    MarkFailure(Error::NOT_MANIFOLD);
    return;
  }
  CalculateNormals();

  InitializeOriginal();
  if (!hasFaceIDs) {
    CreateFaces(propertyTolerance);
  }

  SimplifyTopology();
  Finish();
}

/**
 * Create either a unit tetrahedron, cube or octahedron. The cube is in the
 * first octant, while the others are symmetric about the origin.
 */
Manifold::Impl::Impl(Shape shape) {
  std::vector<glm::vec3> vertPos;
  std::vector<glm::ivec3> triVerts;
  switch (shape) {
    case Shape::TETRAHEDRON:
      vertPos = {{-1.0f, -1.0f, 1.0f},
                 {-1.0f, 1.0f, -1.0f},
                 {1.0f, -1.0f, -1.0f},
                 {1.0f, 1.0f, 1.0f}};
      triVerts = {{2, 0, 1}, {0, 3, 1}, {2, 3, 0}, {3, 2, 1}};
      break;
    case Shape::CUBE:
      vertPos = {{0.0f, 0.0f, 0.0f},  //
                 {1.0f, 0.0f, 0.0f},  //
                 {1.0f, 1.0f, 0.0f},  //
                 {0.0f, 1.0f, 0.0f},  //
                 {0.0f, 0.0f, 1.0f},  //
                 {1.0f, 0.0f, 1.0f},  //
                 {1.0f, 1.0f, 1.0f},  //
                 {0.0f, 1.0f, 1.0f}};
      triVerts = {{0, 2, 1}, {0, 3, 2},  //
                  {4, 5, 6}, {4, 6, 7},  //
                  {0, 1, 5}, {0, 5, 4},  //
                  {1, 2, 6}, {1, 6, 5},  //
                  {2, 3, 7}, {2, 7, 6},  //
                  {3, 0, 4}, {3, 4, 7}};
      break;
    case Shape::OCTAHEDRON:
      vertPos = {{1.0f, 0.0f, 0.0f},   //
                 {-1.0f, 0.0f, 0.0f},  //
                 {0.0f, 1.0f, 0.0f},   //
                 {0.0f, -1.0f, 0.0f},  //
                 {0.0f, 0.0f, 1.0f},   //
                 {0.0f, 0.0f, -1.0f}};
      triVerts = {{0, 2, 4}, {1, 5, 3},  //
                  {2, 1, 4}, {3, 5, 0},  //
                  {1, 3, 4}, {0, 5, 2},  //
                  {3, 0, 4}, {2, 5, 1}};
      break;
  }
  vertPos_ = vertPos;
  CreateHalfedges(triVerts);
  Finish();
  meshRelation_.originalID = ReserveIDs(1);
  InitializeOriginal();
  CreateFaces();
}

void Manifold::Impl::RemoveUnreferencedVerts(VecDH<glm::ivec3>& triVerts) {
  VecDH<int> vertOld2New(NumVert() + 1, 0);
  auto policy = autoPolicy(NumVert());
  for_each(policy, triVerts.cbegin(), triVerts.cend(),
           MarkVerts({vertOld2New.ptrD() + 1}));

  const VecDH<glm::vec3> oldVertPos = vertPos_;
  vertPos_.resize(copy_if<decltype(vertPos_.begin())>(
                      policy, oldVertPos.cbegin(), oldVertPos.cend(),
                      vertOld2New.cbegin() + 1, vertPos_.begin(),
                      thrust::identity<int>()) -
                  vertPos_.begin());

  inclusive_scan(policy, vertOld2New.begin() + 1, vertOld2New.end(),
                 vertOld2New.begin() + 1);

  for_each(policy, triVerts.begin(), triVerts.end(),
           ReindexTriVerts({vertOld2New.cptrD()}));
}

void Manifold::Impl::InitializeOriginal() {
  const int meshID = meshRelation_.originalID;
  // Don't initialize if it's not an original
  if (meshID < 0) return;
  meshRelation_.triRef.resize(NumTri());
  for_each_n(autoPolicy(NumTri()),
             zip(meshRelation_.triRef.begin(), countAt(0)), NumTri(),
             InitializeTriRef({meshID, halfedge_.cptrD()}));
  meshRelation_.meshIDtransform.clear();
  meshRelation_.meshIDtransform[meshID] = {};
}

void Manifold::Impl::CreateFaces(const std::vector<float>& propertyTolerance) {
  VecDH<float> propertyToleranceD =
      propertyTolerance.empty()
          ? VecDH<float>(meshRelation_.numProp, kTolerance)
          : propertyTolerance;

  VecDH<thrust::pair<int, int>> face2face(halfedge_.size(), {-1, -1});
  VecDH<thrust::pair<int, int>> vert2vert(halfedge_.size(), {-1, -1});
  VecDH<float> triArea(NumTri());
  for_each_n(
      autoPolicy(halfedge_.size()),
      zip(face2face.begin(), vert2vert.begin(), countAt(0)), halfedge_.size(),
      CoplanarEdge(
          {triArea.ptrD(), halfedge_.cptrD(), vertPos_.cptrD(),
           meshRelation_.triRef.cptrD(), meshRelation_.triProperties.cptrD(),
           meshRelation_.properties.cptrD(), propertyToleranceD.cptrD(),
           meshRelation_.numProp, precision_}));

  if (meshRelation_.triProperties.size() > 0) {
    DedupePropVerts(meshRelation_.triProperties, vert2vert);
  }

  std::vector<int> components;
  const int numComponent = GetLabels(components, face2face, NumTri());

  std::vector<int> comp2tri(numComponent, -1);
  for (int tri = 0; tri < NumTri(); ++tri) {
    const int comp = components[tri];
    const int current = comp2tri[comp];
    if (current < 0 || triArea[tri] > triArea[current]) {
      comp2tri[comp] = tri;
      triArea[comp] = triArea[tri];
    }
  }

  VecDH<TriRef>& triRef = meshRelation_.triRef;
  for (int tri = 0; tri < NumTri(); ++tri)
    triRef[tri].tri = comp2tri[components[tri]];
}

/**
 * Create the halfedge_ data structure from an input triVerts array like Mesh.
 */
void Manifold::Impl::CreateHalfedges(const VecDH<glm::ivec3>& triVerts) {
  const int numTri = triVerts.size();
  const int numEdge = 3 * numTri / 2;
  // drop the old value first to avoid copy
  halfedge_.resize(0);
  halfedge_.resize(2 * numEdge);
  VecDH<uint64_t> edge(2 * numEdge);
  VecDH<int> ids(2 * numEdge);
  auto policy = autoPolicy(numTri);
  sequence(policy, ids.begin(), ids.end());
  for_each_n(policy, zip(countAt(0), triVerts.begin()), numTri,
             Tri2Halfedges({halfedge_.ptrD(), edge.ptrD()}));
  // Stable sort is required here so that halfedges from the same face are
  // paired together (the triangles were created in face order). In some
  // degenerate situations the triangulator can add the same internal edge in
  // two different faces, causing this edge to not be 2-manifold. These are
  // fixed by duplicating verts in SimplifyTopology.
  stable_sort_by_key(policy, edge.begin(), edge.end(), ids.begin());
  // Once sorted, the first half of the range is the forward halfedges, which
  // correspond to their backward pair at the same offset in the second half
  // of the range.
  for_each_n(policy, countAt(0), numEdge,
             LinkHalfedges({halfedge_.ptrD(), ids.ptrD(), numEdge}));
}

/**
 * Does a full recalculation of the face bounding boxes, including updating
 * the collider, but does not resort the faces.
 */
void Manifold::Impl::Update() {
  CalculateBBox();
  VecDH<Box> faceBox;
  VecDH<uint32_t> faceMorton;
  GetFaceBoxMorton(faceBox, faceMorton);
  collider_.UpdateBoxes(faceBox);
}

void Manifold::Impl::MarkFailure(Error status) {
  bBox_ = Box();
  vertPos_.resize(0);
  halfedge_.resize(0);
  vertNormal_.resize(0);
  faceNormal_.resize(0);
  halfedgeTangent_.resize(0);
  meshRelation_ = MeshRelationD();
  status_ = status;
}

Manifold::Impl Manifold::Impl::Transform(const glm::mat4x3& transform_) const {
  if (transform_ == glm::mat4x3(1.0f)) return *this;
  auto policy = autoPolicy(NumVert());
  Impl result;
  result.collider_ = collider_;
  result.meshRelation_ = meshRelation_;
  result.precision_ = precision_;
  result.bBox_ = bBox_;
  result.halfedge_ = halfedge_;
  result.halfedgeTangent_.resize(halfedgeTangent_.size());

  for (auto& m : result.meshRelation_.meshIDtransform) {
    m.second.transform = transform_ * glm::mat4(m.second.transform);
  }

  result.vertPos_.resize(NumVert());
  result.faceNormal_.resize(faceNormal_.size());
  result.vertNormal_.resize(vertNormal_.size());
  transform(policy, vertPos_.begin(), vertPos_.end(), result.vertPos_.begin(),
            Transform4x3({transform_}));

  glm::mat3 normalTransform = NormalTransform(transform_);
  transform(policy, faceNormal_.begin(), faceNormal_.end(),
            result.faceNormal_.begin(), TransformNormals({normalTransform}));
  transform(policy, vertNormal_.begin(), vertNormal_.end(),
            result.vertNormal_.begin(), TransformNormals({normalTransform}));

  const bool invert = glm::determinant(glm::mat3(transform_)) < 0;

  if (halfedgeTangent_.size() > 0) {
    for_each_n(
        policy, zip(result.halfedgeTangent_.begin(), countAt(0)),
        halfedgeTangent_.size(),
        TransformTangents({glm::mat3(transform_), invert,
                           halfedgeTangent_.cptrD(), halfedge_.cptrD()}));
  }

  if (invert) {
    for_each_n(policy, zip(result.meshRelation_.triRef.begin(), countAt(0)),
               result.NumTri(), FlipTris({result.halfedge_.ptrD()}));
  }

  // This optimization does a cheap collider update if the transform is
  // axis-aligned.
  if (!result.collider_.Transform(transform_)) result.Update();

  result.CalculateBBox();
  float scale = 0;
  for (int i : {0, 1, 2})
    scale =
        glm::max(scale, transform_[0][i] + transform_[1][i] + transform_[2][i]);
  result.precision_ *= scale;
  // Maximum of inherited precision loss and translational precision loss.
  result.SetPrecision(result.precision_);
  return result;
}

/**
 * Sets the precision based on the bounding box, and limits its minimum value
 * by the optional input.
 */
void Manifold::Impl::SetPrecision(float minPrecision) {
  precision_ = glm::max(minPrecision, kTolerance * bBox_.Scale());
  if (!glm::isfinite(precision_)) precision_ = -1;
}

/**
 * If face normals are already present, this function uses them to compute
 * vertex normals (angle-weighted pseudo-normals); otherwise it also computes
 * the face normals. Face normals are only calculated when needed because
 * nearly degenerate faces will accrue rounding error, while the Boolean can
 * retain their original normal, which is more accurate and can help with
 * merging coplanar faces.
 *
 * If the face normals have been invalidated by an operation like Warp(),
 * ensure you do faceNormal_.resize(0) before calling this function to force
 * recalculation.
 */
void Manifold::Impl::CalculateNormals() {
  vertNormal_.resize(NumVert());
  auto policy = autoPolicy(NumTri());
  fill(policy, vertNormal_.begin(), vertNormal_.end(), glm::vec3(0));
  bool calculateTriNormal = false;
  if (faceNormal_.size() != NumTri()) {
    faceNormal_.resize(NumTri());
    calculateTriNormal = true;
  }
  for_each_n(
      policy, zip(faceNormal_.begin(), countAt(0)), NumTri(),
      AssignNormals({vertNormal_.ptrD(), vertPos_.cptrD(), halfedge_.cptrD(),
                     precision_, calculateTriNormal}));
  for_each(policy, vertNormal_.begin(), vertNormal_.end(), Normalize());
}

/**
 * Remaps all the contained meshIDs to new unique values to represent new
 * instances of these meshes.
 */
void Manifold::Impl::IncrementMeshIDs() {
  const int numTri = NumTri();
  const auto policy = autoPolicy(numTri);
  HashTable<uint32_t> meshIDold2new(meshRelation_.meshIDtransform.size() * 2);

  while (1) {
    for_each_n(policy, meshRelation_.triRef.begin(), numTri,
               MarkMeshID({meshIDold2new.D()}));
    if (!meshIDold2new.Full()) break;
    meshIDold2new = HashTable<uint32_t>(meshIDold2new.Size() * 2);
  }
  inclusive_scan(autoPolicy(meshIDold2new.Size()),
                 meshIDold2new.GetValueStore().begin(),
                 meshIDold2new.GetValueStore().end(),
                 meshIDold2new.GetValueStore().begin());
  const int numMeshIDs = meshIDold2new.GetValueStore().back();
  const int meshIDstart = ReserveIDs(numMeshIDs);
  // We do start - 1 because the inclusive scan makes our first index 1
  // instead of 0.
  for_each_n(policy, meshRelation_.triRef.begin(), numTri,
             UpdateMeshID({meshIDold2new.D(), meshIDstart}));
  // Update keys of the transform map
  std::map<int, Relation> oldTransforms;
  std::swap(meshRelation_.meshIDtransform, oldTransforms);
  const int tableSize = meshIDold2new.Size();
  for (int i = 0; i < tableSize; ++i) {
    const auto oldID = meshIDold2new.D().KeyAt(i);
    if (oldID != HashTable<uint32_t>::Open()) {
      meshRelation_.meshIDtransform[meshIDold2new.D().At(i) + meshIDstart] =
          oldTransforms[oldID];
    }
  }
}

/**
 * Returns a sparse array of the bounding box overlaps between the edges of
 * the input manifold, Q and the faces of this manifold. Returned indices only
 * point to forward halfedges.
 */
SparseIndices Manifold::Impl::EdgeCollisions(const Impl& Q) const {
  VecDH<TmpEdge> edges = CreateTmpEdges(Q.halfedge_);
  const int numEdge = edges.size();
  VecDH<Box> QedgeBB(numEdge);
  auto policy = autoPolicy(numEdge);
  for_each_n(policy, zip(QedgeBB.begin(), edges.cbegin()), numEdge,
             EdgeBox({Q.vertPos_.cptrD()}));

  SparseIndices q1p2 = collider_.Collisions(QedgeBB);

  for_each(policy, q1p2.begin(0), q1p2.end(0), ReindexEdge({edges.cptrD()}));
  return q1p2;
}

/**
 * Returns a sparse array of the input vertices that project inside the XY
 * bounding boxes of the faces of this manifold.
 */
SparseIndices Manifold::Impl::VertexCollisionsZ(
    const VecDH<glm::vec3>& vertsIn) const {
  return collider_.Collisions(vertsIn);
}
}  // namespace manifold
