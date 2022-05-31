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

#include <thrust/logical.h>

#include <algorithm>
#include <map>
#include <atomic>

#include "graph.h"
#include "impl.h"

namespace {
using namespace manifold;

__host__ __device__ void AtomicAddVec3(glm::vec3& target,
                                       const glm::vec3& add) {
  for (int i : {0, 1, 2}) {
#ifdef __CUDA_ARCH__
    atomicAdd(&target[i], add[i]);
#else
    std::atomic<float> &tar = reinterpret_cast<std::atomic<float>&>(target[i]);
    float old_val = tar.load();
    while (!tar.compare_exchange_weak(old_val, old_val + add[i]));
#endif
  }
}

struct Normalize {
  __host__ __device__ void operator()(glm::vec3& v) { v = SafeNormalize(v); }
};

struct Transform4x3 {
  const glm::mat4x3 transform;

  __host__ __device__ void operator()(glm::vec3& position) {
    position = transform * glm::vec4(position, 1.0f);
  }
};

struct TransformNormals {
  const glm::mat3 transform;

  __host__ __device__ void operator()(glm::vec3& normal) {
    normal = glm::normalize(transform * normal);
    if (isnan(normal.x)) normal = glm::vec3(0.0f);
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
  TmpEdge* edges;

  __host__ __device__ void operator()(
      thrust::tuple<int, const glm::ivec3&> in) {
    const int tri = thrust::get<0>(in);
    const glm::ivec3& triVerts = thrust::get<1>(in);
    for (const int i : {0, 1, 2}) {
      const int j = (i + 1) % 3;
      const int edge = 3 * tri + i;
      halfedges[edge] = {triVerts[i], triVerts[j], -1, tri};
      edges[edge] = TmpEdge(triVerts[i], triVerts[j], edge);
    }
  }
};

struct LinkHalfedges {
  Halfedge* halfedges;
  const TmpEdge* edges;

  __host__ __device__ void operator()(int k) {
    const int i = 2 * k;
    const int j = i + 1;
    const int pair0 = edges[i].halfedgeIdx;
    const int pair1 = edges[j].halfedgeIdx;
    halfedges[pair0].pairedHalfedge = pair1;
    halfedges[pair1].pairedHalfedge = pair0;
  }
};

struct SwapHalfedges {
  Halfedge* halfedges;
  const TmpEdge* edges;

  __host__ void operator()(int k) {
    const int i = 2 * k;
    const int j = i - 2;
    const TmpEdge thisEdge = edges[i];
    const TmpEdge lastEdge = edges[j];
    if (thisEdge.first == lastEdge.first &&
        thisEdge.second == lastEdge.second) {
      const int swap0idx = thisEdge.halfedgeIdx;
      Halfedge& swap0 = halfedges[swap0idx];
      const int swap1idx = swap0.pairedHalfedge;
      Halfedge& swap1 = halfedges[swap1idx];

      const int next0idx = swap0idx + ((swap0idx + 1) % 3 == 0 ? -2 : 1);
      const int next1idx = swap1idx + ((swap1idx + 1) % 3 == 0 ? -2 : 1);
      Halfedge& next0 = halfedges[next0idx];
      Halfedge& next1 = halfedges[next1idx];

      next0.startVert = swap0.endVert = next1.endVert;
      swap0.pairedHalfedge = next1.pairedHalfedge;
      halfedges[swap0.pairedHalfedge].pairedHalfedge = swap0idx;

      next1.startVert = swap1.endVert = next0.endVert;
      swap1.pairedHalfedge = next0.pairedHalfedge;
      halfedges[swap1.pairedHalfedge].pairedHalfedge = swap1idx;

      next0.pairedHalfedge = next1idx;
      next1.pairedHalfedge = next0idx;
    }
  }
};

struct InitializeBaryRef {
  const int meshID;
  const Halfedge* halfedge;

  __host__ __device__ void operator()(thrust::tuple<BaryRef&, int> inOut) {
    BaryRef& baryRef = thrust::get<0>(inOut);
    int tri = thrust::get<1>(inOut);

    baryRef.meshID = meshID;
    baryRef.tri = tri;
    baryRef.vertBary = {-3, -2, -1};
  }
};

struct CheckProperties {
  const int numSets;

  __host__ __device__ bool operator()(glm::ivec3 triProp) {
    bool good = true;
    for (int i : {0, 1, 2}) good &= (triProp[i] >= 0 && triProp[i] < numSets);
    return good;
  }
};

struct CoplanarEdge {
  float* triArea;
  const Halfedge* halfedge;
  const glm::vec3* vertPos;
  const glm::ivec3* triProp;
  const float* prop;
  const float* propTol;
  const int numProp;
  const float precision;

  __host__ __device__ void operator()(
      thrust::tuple<thrust::pair<int, int>&, int> inOut) {
    thrust::pair<int, int>& face2face = thrust::get<0>(inOut);
    const int edgeIdx = thrust::get<1>(inOut);

    const Halfedge edge = halfedge[edgeIdx];
    if (!edge.IsForward()) return;
    const Halfedge pair = halfedge[edge.pairedHalfedge];
    const glm::vec3 base = vertPos[edge.startVert];

    const int baseNum = edgeIdx - 3 * edge.face;
    const int jointNum = edge.pairedHalfedge - 3 * pair.face;
    const int edgeNum = baseNum == 0 ? 2 : baseNum - 1;
    const int pairNum = jointNum == 0 ? 2 : jointNum - 1;

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
        const float area = glm::max(
            glm::length(cross), glm::length(glm::cross(iPairVec, iJointVec)));
        const float volume = glm::abs(glm::dot(cross, iPairVec));
        // Only operate on consistent triangles
        if (volume > area * precision) return;
      }
    }

    triArea[edge.face] = area;
    triArea[pair.face] = areaPair;
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
}  // namespace

namespace manifold {

std::atomic<int> Manifold::Impl::meshIDCounter_(1);

/**
 * Create a manifold from an input triangle Mesh. Will throw if the Mesh is not
 * manifold. TODO: update halfedgeTangent during SimplifyTopology.
 */
Manifold::Impl::Impl(const Mesh& mesh,
                     const std::vector<glm::ivec3>& triProperties,
                     const std::vector<float>& properties,
                     const std::vector<float>& propertyTolerance)
    : vertPos_(mesh.vertPos), halfedgeTangent_(mesh.halfedgeTangent) {
  CheckDevice();
  CalculateBBox();
  SetPrecision();
  CreateAndFixHalfedges(mesh.triVerts);
  ALWAYS_ASSERT(IsManifold(), topologyErr, "Input mesh is not manifold!");
  CalculateNormals();
  InitializeNewReference(triProperties, properties, propertyTolerance);
  SimplifyTopology();
  Finish();
}

/**
 * Create eiter a unit tetrahedron, cube or octahedron. The cube is in the first
 * octant, while the others are symmetric about the origin.
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
    default:
      throw userErr("Unrecognized shape!");
  }
  vertPos_ = vertPos;
  CreateHalfedges(triVerts);
  Finish();
  InitializeNewReference();
}

void Manifold::Impl::ReinitializeReference(int meshID) {
  // instead of storing the meshID, we store 0 and set the mapping to
  // 0 -> meshID, because the meshID after boolean operation also starts from 0.
  thrust::for_each_n(thrust::device, zip(meshRelation_.triBary.begin(), countAt(0)), NumTri(),
                     InitializeBaryRef({0, halfedge_.cptrD()}));
  meshRelation_.originalID.clear();
  meshRelation_.originalID[0] = meshID;
}

int Manifold::Impl::InitializeNewReference(
    const std::vector<glm::ivec3>& triProperties,
    const std::vector<float>& properties,
    const std::vector<float>& propertyTolerance) {
  meshRelation_.triBary.resize(NumTri());
  const int nextMeshID = meshIDCounter_.fetch_add(1);
  ReinitializeReference(nextMeshID);

  const int numProps = propertyTolerance.size();

  VecDH<glm::ivec3> triPropertiesD(triProperties);
  VecDH<float> propertiesD(properties);
  VecDH<float> propertyToleranceD(propertyTolerance);

  if (numProps > 0) {
    ALWAYS_ASSERT(
        triProperties.size() == NumTri() || triProperties.size() == 0, userErr,
        "If specified, triProperties vector length must match NumTri().");
    ALWAYS_ASSERT(properties.size() % numProps == 0, userErr,
                  "properties vector must be a multiple of the size of "
                  "propertyTolerance.");

    const int numSets = properties.size() / numProps;
    ALWAYS_ASSERT(thrust::all_of(thrust::device, triPropertiesD.begin(), triPropertiesD.end(),
                                 CheckProperties({numSets})),
                  userErr,
                  "triProperties value is outside the properties range.");
  }

  VecDH<thrust::pair<int, int>> face2face(halfedge_.size(), {-1, -1});
  VecDH<float> triArea(NumTri());
  thrust::for_each_n(
      thrust::device, zip(face2face.begin(), countAt(0)), halfedge_.size(),
      CoplanarEdge({triArea.ptrD(), halfedge_.cptrD(), vertPos_.cptrD(),
                    triPropertiesD.cptrD(), propertiesD.cptrD(),
                    propertyToleranceD.cptrD(), numProps, precision_}));

  Graph graph;
  for (int i = 0; i < NumTri(); ++i) {
    graph.add_nodes(i);
  }
  for (int i = 0; i < face2face.size(); ++i) {
    const thrust::pair<int, int> edge = face2face[i];
    if (edge.first < 0) continue;
    graph.add_edge(edge.first, edge.second);
  }

  std::vector<int> components;
  const int numComponent = ConnectedComponents(components, graph);

  std::vector<int> comp2tri(numComponent, -1);
  for (int tri = 0; tri < NumTri(); ++tri) {
    const int comp = components[tri];
    const int current = comp2tri[comp];
    if (current < 0 || triArea[tri] > triArea[current]) {
      comp2tri[comp] = tri;
      triArea[comp] = triArea[tri];
    }
  }

  VecDH<BaryRef>& triBary = meshRelation_.triBary;
  std::map<std::pair<int, int>, int> triVert2bary;

  for (int tri = 0; tri < NumTri(); ++tri) {
    const int refTri = comp2tri[components[tri]];
    if (refTri == tri) continue;

    glm::mat3 triPos;
    for (int i : {0, 1, 2}) {
      const int vert = halfedge_[3 * refTri + i].startVert;
      triPos[i] = vertPos_[vert];
      triVert2bary[{refTri, vert}] = i - 3;
    }

    glm::ivec3 vertBary;
    bool coplanar = true;
    for (int i : {0, 1, 2}) {
      const int vert = halfedge_[3 * tri + i].startVert;
      if (triVert2bary.find({refTri, vert}) == triVert2bary.end()) {
        const glm::vec3 uvw =
            GetBarycentric(vertPos_[vert], triPos, precision_);
        if (isnan(uvw[0])) {
          coplanar = false;
          triVert2bary[{refTri, vert}] = -4;
          break;
        }
        triVert2bary[{refTri, vert}] = meshRelation_.barycentric.size();
        meshRelation_.barycentric.push_back(uvw);
      }
      const int bary = triVert2bary[{refTri, vert}];
      if (bary < -3) {
        coplanar = false;
        break;
      }
      vertBary[i] = bary;
    }

    if (coplanar) {
      BaryRef& ref = triBary[tri];
      ref.tri = refTri;
      ref.vertBary = vertBary;
    }
  }

  return nextMeshID;
}

/**
 * Create the halfedge_ data structure from an input triVerts array like Mesh.
 */
void Manifold::Impl::CreateHalfedges(const VecDH<glm::ivec3>& triVerts) {
  const int numTri = triVerts.size();
  halfedge_.resize(3 * numTri);
  VecDH<TmpEdge> edge(3 * numTri);
  thrust::for_each_n(thrust::device, zip(countAt(0), triVerts.begin()), numTri,
                     Tri2Halfedges({halfedge_.ptrD(), edge.ptrD()}));
  thrust::sort(thrust::device, edge.begin(), edge.end());
  thrust::for_each_n(thrust::device, countAt(0), halfedge_.size() / 2,
                     LinkHalfedges({halfedge_.ptrD(), edge.cptrD()}));
}

/**
 * Create the halfedge_ data structure from an input triVerts array like Mesh.
 * Check that the input is an even-manifold, and if it is not 2-manifold,
 * perform edge swaps until it is. This is a host function.
 */
void Manifold::Impl::CreateAndFixHalfedges(const VecDH<glm::ivec3>& triVerts) {
  const int numTri = triVerts.size();
  // drop the old value first to avoid copy
  halfedge_.resize(0);
  halfedge_.resize(3 * numTri);
  VecDH<TmpEdge> edge(3 * numTri);
  thrust::for_each_n(thrust::device, zip(countAt(0), triVerts.begin()), numTri,
                     Tri2Halfedges({halfedge_.ptrD(), edge.ptrD()}));
  // Stable sort is required here so that halfedges from the same face are
  // paired together (the triangles were created in face order). In some
  // degenerate situations the triangulator can add the same internal edge in
  // two different faces, causing this edge to not be 2-manifold. We detect this
  // and fix it by swapping one of the identical edges, so it is important that
  // we have the edges paired according to their face.
  thrust::stable_sort(thrust::device, edge.begin(), edge.end());
  thrust::for_each_n(thrust::host, countAt(0), halfedge_.size() / 2,
                     LinkHalfedges({halfedge_.ptrH(), edge.cptrH()}));
  thrust::for_each(thrust::host, countAt(1), countAt(halfedge_.size() / 2),
                   SwapHalfedges({halfedge_.ptrH(), edge.cptrH()}));
}

/**
 * Does a full recalculation of the face bounding boxes, including updating the
 * collider, but does not resort the faces.
 */
void Manifold::Impl::Update() {
  CalculateBBox();
  VecDH<Box> faceBox;
  VecDH<uint32_t> faceMorton;
  GetFaceBoxMorton(faceBox, faceMorton);
  collider_.UpdateBoxes(faceBox);
}

void Manifold::Impl::ApplyTransform() const {
  // This const_cast is here because these operations cancel out, leaving the
  // state conceptually unchanged. This enables lazy transformation evaluation.
  const_cast<Impl*>(this)->ApplyTransform();
}

/**
 * Bake the manifold's transform into its vertices. This function allows lazy
 * evaluation, which is important because often several transforms are applied
 * between operations.
 */
void Manifold::Impl::ApplyTransform() {
  if (transform_ == glm::mat4x3(1.0f)) return;
  thrust::for_each(thrust::device, vertPos_.begin(), vertPos_.end(),
                   Transform4x3({transform_}));

  glm::mat3 normalTransform =
      glm::inverse(glm::transpose(glm::mat3(transform_)));
  thrust::for_each(thrust::device, faceNormal_.begin(), faceNormal_.end(),
                   TransformNormals({normalTransform}));
  thrust::for_each(thrust::device, vertNormal_.begin(), vertNormal_.end(),
                   TransformNormals({normalTransform}));
  // This optimization does a cheap collider update if the transform is
  // axis-aligned.
  if (!collider_.Transform(transform_)) Update();

  const float oldScale = bBox_.Scale();
  transform_ = glm::mat4x3(1.0f);
  CalculateBBox();

  const float newScale = bBox_.Scale();
  precision_ *= glm::max(1.0f, newScale / oldScale) *
                glm::max(glm::length(transform_[0]),
                         glm::max(glm::length(transform_[1]),
                                  glm::length(transform_[2])));

  // Maximum of inherited precision loss and translational precision loss.
  SetPrecision(precision_);
}

/**
 * Sets the precision based on the bounding box, and limits its minimum value by
 * the optional input.
 */
void Manifold::Impl::SetPrecision(float minPrecision) {
  precision_ = glm::max(minPrecision, kTolerance * bBox_.Scale());
  if (!glm::isfinite(precision_)) precision_ = -1;
}

/**
 * If face normals are already present, this function uses them to compute
 * vertex normals (angle-weighted pseudo-normals); otherwise it also computes
 * the face normals. Face normals are only calculated when needed because nearly
 * degenerate faces will accrue rounding error, while the Boolean can retain
 * their original normal, which is more accurate and can help with merging
 * coplanar faces.
 *
 * If the face normals have been invalidated by an operation like Warp(), ensure
 * you do faceNormal_.resize(0) before calling this function to force
 * recalculation.
 */
void Manifold::Impl::CalculateNormals() {
  vertNormal_.resize(NumVert());
  thrust::fill(thrust::device, vertNormal_.begin(), vertNormal_.end(), glm::vec3(0));
  bool calculateTriNormal = false;
  if (faceNormal_.size() != NumTri()) {
    faceNormal_.resize(NumTri());
    calculateTriNormal = true;
  }
  thrust::for_each_n(
      thrust::device, zip(faceNormal_.begin(), countAt(0)), NumTri(),
      AssignNormals({vertNormal_.ptrD(), vertPos_.cptrD(), halfedge_.cptrD(),
                     precision_, calculateTriNormal}));
  thrust::for_each(thrust::device, vertNormal_.begin(), vertNormal_.end(), Normalize());
}

/**
 * Update meshID and originalID in meshRelation_.triBary[startTri..startTri+n]
 * according to meshIDs -> originalIDs mapping. The updated meshID will start
 * from startID.
 * Will raise an exception if meshRelation_.triBary[startTri..startTri+n]
 * contains a meshID not in meshIDs.
 *
 * We remap them into indices starting from startID. The exact value value is not
 * important as long as
 * 1. They are distinct
 * 2. `originalID[meshID]` is the original mesh ID of the triangle
 *
 * Use this when the mesh is a combination of several meshes or a subset of a
 * larger mesh, e.g. after performing boolean operations, compose or decompose.
 */
void Manifold::Impl::UpdateMeshIDs(VecDH<int> &meshIDs, VecDH<int> &originalIDs,
                                   int startTri, int n, int startID) {
  if (n == -1)
    n = meshRelation_.triBary.size();
  thrust::sort_by_key(thrust::host, meshIDs.begin(), meshIDs.end(),
                      originalIDs.begin());
  constexpr int kOccurred = 1 << 30;
  VecDH<int> error(1, -1);
  const int numMesh = meshIDs.size();
  const int *meshIDsPtr = meshIDs.cptrD();
  int *originalPtr = originalIDs.ptrD();
  int *errorPtr = error.ptrD();
  thrust::for_each(thrust::device,
                   meshRelation_.triBary.begin() + startTri,
                   meshRelation_.triBary.begin() + startTri + n,
                   [=] __host__ __device__(BaryRef & b) {
                     int index =
                         thrust::lower_bound(meshIDsPtr, meshIDsPtr + numMesh,
                                             b.meshID) -
                         meshIDsPtr;
                     if (index >= numMesh || meshIDsPtr[index] != b.meshID) {
                       *errorPtr = b.meshID;
                     }
                     b.meshID = index + startID;
                     originalPtr[index] |= kOccurred;
                   });

  if (error[0] != -1) {
    std::stringstream ss;
    ss << "Manifold::UpdateMeshIDs: meshID " << error[0]
       << " not found in meshIDs.";
    throw std::runtime_error(ss.str());
  }
  for (int i = 0; i < numMesh; ++i) {
    if (originalIDs[i] & kOccurred) {
      originalIDs[i] &= ~kOccurred;
      meshRelation_.originalID[i + startID] = originalIDs[i];
    }
  }
}

/**
 * Returns a sparse array of the bounding box overlaps between the edges of the
 * input manifold, Q and the faces of this manifold. Returned indices only
 * point to forward halfedges.
 */
SparseIndices Manifold::Impl::EdgeCollisions(const Impl& Q) const {
  VecDH<TmpEdge> edges = CreateTmpEdges(Q.halfedge_);
  const int numEdge = edges.size();
  VecDH<Box> QedgeBB(numEdge);
  thrust::for_each_n(thrust::device, zip(QedgeBB.begin(), edges.cbegin()), numEdge,
                     EdgeBox({Q.vertPos_.cptrD()}));

  SparseIndices q1p2 = collider_.Collisions(QedgeBB);

  thrust::for_each(thrust::device, q1p2.begin(0), q1p2.end(0), ReindexEdge({edges.cptrD()}));
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
