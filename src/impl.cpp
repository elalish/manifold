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

#include "./impl.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <optional>

#include "./hashtable.h"
#include "./mesh_fixes.h"
#include "./parallel.h"
#include "./svd.h"

#ifdef MANIFOLD_EXPORT
#include <string.h>

#include <iostream>
#endif

namespace {
using namespace manifold;

constexpr uint64_t kRemove = std::numeric_limits<uint64_t>::max();

void AtomicAddVec3(vec3& target, const vec3& add) {
  for (int i : {0, 1, 2}) {
    std::atomic<double>& tar =
        reinterpret_cast<std::atomic<double>&>(target[i]);
    double old_val = tar.load(std::memory_order_relaxed);
    while (!tar.compare_exchange_weak(old_val, old_val + add[i],
                                      std::memory_order_relaxed)) {
    }
  }
}

struct Transform4x3 {
  const mat3x4 transform;

  vec3 operator()(vec3 position) { return transform * vec4(position, 1.0); }
};

struct UpdateMeshID {
  const HashTableD<uint32_t> meshIDold2new;

  void operator()(TriRef& ref) { ref.meshID = meshIDold2new[ref.meshID]; }
};

int GetLabels(std::vector<int>& components,
              const Vec<std::pair<int, int>>& edges, int numNodes) {
  UnionFind<> uf(numNodes);
  for (auto edge : edges) {
    if (edge.first == -1 || edge.second == -1) continue;
    uf.unionXY(edge.first, edge.second);
  }

  return uf.connectedComponents(components);
}
}  // namespace

namespace manifold {

std::atomic<uint32_t> Manifold::Impl::meshIDCounter_(1);

uint32_t Manifold::Impl::ReserveIDs(uint32_t n) {
  return Manifold::Impl::meshIDCounter_.fetch_add(n, std::memory_order_relaxed);
}

/**
 * Create either a unit tetrahedron, cube or octahedron. The cube is in the
 * first octant, while the others are symmetric about the origin.
 */
Manifold::Impl::Impl(Shape shape, const mat3x4 m) {
  std::vector<vec3> vertPos;
  std::vector<ivec3> triVerts;
  switch (shape) {
    case Shape::Tetrahedron:
      vertPos = {{-1.0, -1.0, 1.0},
                 {-1.0, 1.0, -1.0},
                 {1.0, -1.0, -1.0},
                 {1.0, 1.0, 1.0}};
      triVerts = {{2, 0, 1}, {0, 3, 1}, {2, 3, 0}, {3, 2, 1}};
      break;
    case Shape::Cube:
      vertPos = {{0.0, 0.0, 0.0},  //
                 {0.0, 0.0, 1.0},  //
                 {0.0, 1.0, 0.0},  //
                 {0.0, 1.0, 1.0},  //
                 {1.0, 0.0, 0.0},  //
                 {1.0, 0.0, 1.0},  //
                 {1.0, 1.0, 0.0},  //
                 {1.0, 1.0, 1.0}};
      triVerts = {{1, 0, 4}, {2, 4, 0},  //
                  {1, 3, 0}, {3, 1, 5},  //
                  {3, 2, 0}, {3, 7, 2},  //
                  {5, 4, 6}, {5, 1, 4},  //
                  {6, 4, 2}, {7, 6, 2},  //
                  {7, 3, 5}, {7, 5, 6}};
      break;
    case Shape::Octahedron:
      vertPos = {{1.0, 0.0, 0.0},   //
                 {-1.0, 0.0, 0.0},  //
                 {0.0, 1.0, 0.0},   //
                 {0.0, -1.0, 0.0},  //
                 {0.0, 0.0, 1.0},   //
                 {0.0, 0.0, -1.0}};
      triVerts = {{0, 2, 4}, {1, 5, 3},  //
                  {2, 1, 4}, {3, 5, 0},  //
                  {1, 3, 4}, {0, 5, 2},  //
                  {3, 0, 4}, {2, 5, 1}};
      break;
  }
  vertPos_ = vertPos;
  for (auto& v : vertPos_) v = m * vec4(v, 1.0);
  CreateHalfedges(triVerts);
  Finish();
  InitializeOriginal();
  CreateFaces();
}

void Manifold::Impl::RemoveUnreferencedVerts() {
  ZoneScoped;
  const int numVert = NumVert();
  Vec<int> keep(numVert, 0);
  auto policy = autoPolicy(numVert, 1e5);
  for_each(policy, halfedge_.cbegin(), halfedge_.cend(), [&keep](Halfedge h) {
    if (h.startVert >= 0) {
      reinterpret_cast<std::atomic<int>*>(&keep[h.startVert])
          ->store(1, std::memory_order_relaxed);
    }
  });

  for_each_n(policy, countAt(0), numVert, [&keep, this](int v) {
    if (keep[v] == 0) {
      vertPos_[v] = vec3(NAN);
    }
  });
}

void Manifold::Impl::InitializeOriginal(bool keepFaceID) {
  const int meshID = ReserveIDs(1);
  meshRelation_.originalID = meshID;
  auto& triRef = meshRelation_.triRef;
  triRef.resize_nofill(NumTri());
  for_each_n(autoPolicy(NumTri(), 1e5), countAt(0), NumTri(),
             [meshID, keepFaceID, &triRef](const int tri) {
               triRef[tri] = {meshID, meshID, tri,
                              keepFaceID ? triRef[tri].faceID : tri};
             });
  meshRelation_.meshIDtransform.clear();
  meshRelation_.meshIDtransform[meshID] = {meshID};
}

void Manifold::Impl::CreateFaces() {
  ZoneScoped;
  const int numTri = NumTri();
  struct TriPriority {
    double area2;
    int tri;
  };
  Vec<TriPriority> triPriority(numTri);
  for_each_n(autoPolicy(numTri), countAt(0), numTri,
             [&triPriority, this](int tri) {
               meshRelation_.triRef[tri].faceID = -1;
               if (halfedge_[3 * tri].startVert < 0) {
                 triPriority[tri] = {0, tri};
                 return;
               }
               const vec3 v = vertPos_[halfedge_[3 * tri].startVert];
               triPriority[tri] = {
                   length2(cross(vertPos_[halfedge_[3 * tri].endVert] - v,
                                 vertPos_[halfedge_[3 * tri + 1].endVert] - v)),
                   tri};
             });

  stable_sort(triPriority.begin(), triPriority.end(),
              [](auto a, auto b) { return a.area2 > b.area2; });

  Vec<int> interiorHalfedges;
  for (const auto tp : triPriority) {
    if (meshRelation_.triRef[tp.tri].faceID >= 0) continue;

    meshRelation_.triRef[tp.tri].faceID = tp.tri;
    if (halfedge_[3 * tp.tri].startVert < 0) continue;
    const vec3 base = vertPos_[halfedge_[3 * tp.tri].startVert];
    const vec3 normal = faceNormal_[tp.tri];
    interiorHalfedges.resize(3);
    interiorHalfedges[0] = 3 * tp.tri;
    interiorHalfedges[1] = 3 * tp.tri + 1;
    interiorHalfedges[2] = 3 * tp.tri + 2;
    while (!interiorHalfedges.empty()) {
      const int h =
          NextHalfedge(halfedge_[interiorHalfedges.back()].pairedHalfedge);
      interiorHalfedges.pop_back();
      if (meshRelation_.triRef[h / 3].faceID >= 0) continue;

      const vec3 v = vertPos_[halfedge_[h].endVert];
      if (std::abs(dot(v - base, normal)) < tolerance_) {
        meshRelation_.triRef[h / 3].faceID = tp.tri;

        if (interiorHalfedges.empty() ||
            h != halfedge_[interiorHalfedges.back()].pairedHalfedge) {
          interiorHalfedges.push_back(h);
        } else {
          interiorHalfedges.pop_back();
        }
        const int hNext = NextHalfedge(h);
        interiorHalfedges.push_back(hNext);
      }
    }
  }
}

void Manifold::Impl::DedupePropVerts() {
  ZoneScoped;
  const size_t numProp = NumProp();
  if (numProp == 0) return;

  Vec<std::pair<int, int>> vert2vert(halfedge_.size(), {-1, -1});
  for_each_n(
      autoPolicy(halfedge_.size(), 1e4), countAt(0), halfedge_.size(),
      [&vert2vert, numProp, this](const int edgeIdx) {
        const Halfedge edge = halfedge_[edgeIdx];
        const Halfedge pair = halfedge_[edge.pairedHalfedge];
        const int edgeFace = edgeIdx / 3;
        const int pairFace = edge.pairedHalfedge / 3;

        if (meshRelation_.triRef[edgeFace].meshID !=
            meshRelation_.triRef[pairFace].meshID)
          return;

        const int baseNum = edgeIdx - 3 * edgeFace;
        const int jointNum = edge.pairedHalfedge - 3 * pairFace;

        const int prop0 = meshRelation_.triProperties[edgeFace][baseNum];
        const int prop1 =
            meshRelation_
                .triProperties[pairFace][jointNum == 2 ? 0 : jointNum + 1];
        bool propEqual = true;
        for (size_t p = 0; p < numProp; ++p) {
          if (meshRelation_.properties[numProp * prop0 + p] !=
              meshRelation_.properties[numProp * prop1 + p]) {
            propEqual = false;
            break;
          }
        }
        if (propEqual) {
          vert2vert[edgeIdx] = std::make_pair(prop0, prop1);
        }
      });

  std::vector<int> vertLabels;
  const size_t numPropVert = NumPropVert();
  const int numLabels = GetLabels(vertLabels, vert2vert, numPropVert);

  std::vector<int> label2vert(numLabels);
  for (size_t v = 0; v < numPropVert; ++v) label2vert[vertLabels[v]] = v;
  for (auto& prop : meshRelation_.triProperties)
    for (int i : {0, 1, 2}) prop[i] = label2vert[vertLabels[prop[i]]];
}

/**
 * Create the halfedge_ data structure from an input triVerts array like Mesh.
 */
void Manifold::Impl::CreateHalfedges(const Vec<ivec3>& triVerts) {
  ZoneScoped;
  const size_t numTri = triVerts.size();
  const int numHalfedge = 3 * numTri;
  // drop the old value first to avoid copy
  halfedge_.clear(true);
  halfedge_.resize_nofill(numHalfedge);
  Vec<uint64_t> edge(numHalfedge);
  Vec<int> ids(numHalfedge);
  auto policy = autoPolicy(numTri, 1e5);
  sequence(ids.begin(), ids.end());
  for_each_n(policy, countAt(0), numTri,
             [this, &edge, &triVerts](const int tri) {
               const ivec3& verts = triVerts[tri];
               for (const int i : {0, 1, 2}) {
                 const int j = (i + 1) % 3;
                 const int e = 3 * tri + i;
                 halfedge_[e] = {verts[i], verts[j], -1};
                 // Sort the forward halfedges in front of the backward ones
                 // by setting the highest-order bit.
                 edge[e] = uint64_t(verts[i] < verts[j] ? 1 : 0) << 63 |
                           ((uint64_t)std::min(verts[i], verts[j])) << 32 |
                           std::max(verts[i], verts[j]);
               }
             });
  // Stable sort is required here so that halfedges from the same face are
  // paired together (the triangles were created in face order). In some
  // degenerate situations the triangulator can add the same internal edge in
  // two different faces, causing this edge to not be 2-manifold. These are
  // fixed by duplicating verts in SimplifyTopology.
  stable_sort(ids.begin(), ids.end(), [&edge](const int& a, const int& b) {
    return edge[a] < edge[b];
  });

  // Mark opposed triangles for removal - this may strand unreferenced verts
  // which are removed later by RemoveUnreferencedVerts() and Finish().
  const int numEdge = numHalfedge / 2;

  constexpr int removedHalfedge = -2;
  const auto body = [&, removedHalfedge](int i, int consecutiveStart, int segmentEnd) {
    const int pair0 = ids[i];
    Halfedge& h0 = halfedge_[pair0];
    int k = consecutiveStart + numEdge;
    while (1) {
      const int pair1 = ids[k];
      Halfedge& h1 = halfedge_[pair1];
      if (h0.startVert != h1.endVert || h0.endVert != h1.startVert) break;
      if (halfedge_[NextHalfedge(pair0)].endVert ==
          halfedge_[NextHalfedge(pair1)].endVert) {
        h0.pairedHalfedge = h1.pairedHalfedge = removedHalfedge;
        // Reorder so that remaining edges pair up
        if (k != i + numEdge) std::swap(ids[i + numEdge], ids[k]);
        break;
      }
      ++k;
      if (k >= segmentEnd + numEdge) break;
    }
    if (i + 1 == segmentEnd) return consecutiveStart;
    Halfedge& h1 = halfedge_[ids[i + 1]];
    if (h0.startVert == h1.startVert && h0.endVert == h1.endVert)
      return consecutiveStart;
    return i + 1;
  };

#if MANIFOLD_PAR == 1
  Vec<std::pair<int, int>> ranges;
  const int increment = std::min(
      std::max(numEdge / tbb::this_task_arena::max_concurrency() / 2, 1024),
      numEdge);
  const auto duplicated = [&](int a, int b) {
    const Halfedge& h0 = halfedge_[ids[a]];
    const Halfedge& h1 = halfedge_[ids[b]];
    return h0.startVert == h1.startVert && h0.endVert == h1.endVert;
  };
  int end = 0;
  while (end < numEdge) {
    const int start = end;
    end = std::min(end + increment, numEdge);
    // make sure duplicated halfedges are in the same partition
    while (end < numEdge && duplicated(end - 1, end)) end++;
    ranges.push_back(std::make_pair(start, end));
  }
  for_each(ExecutionPolicy::Par, ranges.begin(), ranges.end(),
           [&](const std::pair<int, int>& range) {
             const auto [start, end] = range;
             int consecutiveStart = start;
             for (int i = start; i < end; ++i)
               consecutiveStart = body(i, consecutiveStart, end);
           });
#else
  int consecutiveStart = 0;
  for (int i = 0; i < numEdge; ++i)
    consecutiveStart = body(i, consecutiveStart, numEdge);
#endif

  // Once sorted, the first half of the range is the forward halfedges, which
  // correspond to their backward pair at the same offset in the second half
  // of the range.
  for_each_n(policy, countAt(0), numEdge, [this, &ids, numEdge, removedHalfedge](int i) {
    const int pair0 = ids[i];
    const int pair1 = ids[i + numEdge];
    if (halfedge_[pair0].pairedHalfedge != removedHalfedge) {
      halfedge_[pair0].pairedHalfedge = pair1;
      halfedge_[pair1].pairedHalfedge = pair0;
    } else {
      halfedge_[pair0] = halfedge_[pair1] = {-1, -1, -1};
    }
  });
}

/**
 * Does a full recalculation of the face bounding boxes, including updating
 * the collider, but does not resort the faces.
 */
void Manifold::Impl::Update() {
  CalculateBBox();
  Vec<Box> faceBox;
  Vec<uint32_t> faceMorton;
  GetFaceBoxMorton(faceBox, faceMorton);
  collider_.UpdateBoxes(faceBox);
}

void Manifold::Impl::MakeEmpty(Error status) {
  bBox_ = Box();
  vertPos_.clear();
  halfedge_.clear();
  vertNormal_.clear();
  faceNormal_.clear();
  halfedgeTangent_.clear();
  meshRelation_ = MeshRelationD();
  status_ = status;
}

void Manifold::Impl::Warp(std::function<void(vec3&)> warpFunc) {
  WarpBatch([&warpFunc](VecView<vec3> vecs) {
    for_each(ExecutionPolicy::Seq, vecs.begin(), vecs.end(), warpFunc);
  });
}

void Manifold::Impl::WarpBatch(std::function<void(VecView<vec3>)> warpFunc) {
  warpFunc(vertPos_.view());
  CalculateBBox();
  if (!IsFinite()) {
    MakeEmpty(Error::NonFiniteVertex);
    return;
  }
  Update();
  faceNormal_.clear();  // force recalculation of triNormal
  SetEpsilon();
  Finish();
  CreateFaces();
  meshRelation_.originalID = -1;
}

Manifold::Impl Manifold::Impl::Transform(const mat3x4& transform_) const {
  ZoneScoped;
  if (transform_ == mat3x4(la::identity)) return *this;
  auto policy = autoPolicy(NumVert());
  Impl result;
  if (status_ != Manifold::Error::NoError) {
    result.status_ = status_;
    return result;
  }
  if (!all(la::isfinite(transform_))) {
    result.MakeEmpty(Error::NonFiniteVertex);
    return result;
  }
  result.collider_ = collider_;
  result.meshRelation_ = meshRelation_;
  result.epsilon_ = epsilon_;
  result.tolerance_ = tolerance_;
  result.bBox_ = bBox_;
  result.halfedge_ = halfedge_;
  result.halfedgeTangent_.resize(halfedgeTangent_.size());

  result.meshRelation_.originalID = -1;
  for (auto& m : result.meshRelation_.meshIDtransform) {
    m.second.transform = transform_ * Mat4(m.second.transform);
  }

  result.vertPos_.resize(NumVert());
  result.faceNormal_.resize(faceNormal_.size());
  result.vertNormal_.resize(vertNormal_.size());
  transform(vertPos_.begin(), vertPos_.end(), result.vertPos_.begin(),
            Transform4x3({transform_}));

  mat3 normalTransform = NormalTransform(transform_);
  transform(faceNormal_.begin(), faceNormal_.end(), result.faceNormal_.begin(),
            TransformNormals({normalTransform}));
  transform(vertNormal_.begin(), vertNormal_.end(), result.vertNormal_.begin(),
            TransformNormals({normalTransform}));

  const bool invert = la::determinant(mat3(transform_)) < 0;

  if (halfedgeTangent_.size() > 0) {
    for_each_n(policy, countAt(0), halfedgeTangent_.size(),
               TransformTangents({result.halfedgeTangent_, 0, mat3(transform_),
                                  invert, halfedgeTangent_, halfedge_}));
  }

  if (invert) {
    for_each_n(policy, countAt(0), result.NumTri(),
               FlipTris({result.halfedge_}));
  }

  // This optimization does a cheap collider update if the transform is
  // axis-aligned.
  if (!result.collider_.Transform(transform_)) result.Update();

  result.CalculateBBox();
  // Scale epsilon by the norm of the 3x3 portion of the transform.
  result.epsilon_ *= SpectralNorm(mat3(transform_));
  // Maximum of inherited epsilon loss and translational epsilon loss.
  result.SetEpsilon(result.epsilon_);
  return result;
}

/**
 * Sets epsilon based on the bounding box, and limits its minimum value
 * by the optional input.
 */
void Manifold::Impl::SetEpsilon(double minEpsilon, bool useSingle) {
  epsilon_ = MaxEpsilon(minEpsilon, bBox_);
  double minTol = epsilon_;
  if (useSingle)
    minTol =
        std::max(minTol, std::numeric_limits<float>::epsilon() * bBox_.Scale());
  tolerance_ = std::max(tolerance_, minTol);
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
  ZoneScoped;
  vertNormal_.resize(NumVert());
  auto policy = autoPolicy(NumTri());
  bool calculateTriNormal = false;

  std::vector<std::atomic<int>> vertHalfedgeMap(NumVert());
  for_each_n(policy, countAt(0), NumVert(), [&](const size_t vert) {
    vertHalfedgeMap[vert] = std::numeric_limits<int>::max();
  });

  auto atomicMin = [&vertHalfedgeMap](int value, int vert) {
    if (vert < 0) return;
    int old = std::numeric_limits<int>::max();
    while (!vertHalfedgeMap[vert].compare_exchange_strong(old, value))
      if (old < value) break;
  };
  if (faceNormal_.size() != NumTri()) {
    faceNormal_.resize(NumTri());
    calculateTriNormal = true;
    for_each_n(policy, countAt(0), NumTri(), [&](const int face) {
      vec3& triNormal = faceNormal_[face];
      if (halfedge_[3 * face].startVert < 0) {
        triNormal = vec3(0, 0, 1);
        return;
      }

      ivec3 triVerts;
      for (int i : {0, 1, 2}) {
        int v = halfedge_[3 * face + i].startVert;
        triVerts[i] = v;
        atomicMin(3 * face + i, v);
      }

      vec3 edge[3];
      for (int i : {0, 1, 2}) {
        const int j = (i + 1) % 3;
        edge[i] = la::normalize(vertPos_[triVerts[j]] - vertPos_[triVerts[i]]);
      }
      triNormal = la::normalize(la::cross(edge[0], edge[1]));
      if (std::isnan(triNormal.x)) triNormal = vec3(0, 0, 1);
    });
  } else {
    for_each_n(policy, countAt(0), halfedge_.size(),
               [&](const int i) { atomicMin(i, halfedge_[i].startVert); });
  }

  for_each_n(policy, countAt(0), NumVert(), [&](const size_t vert) {
    int firstEdge = vertHalfedgeMap[vert].load();
    // not referenced
    if (firstEdge == std::numeric_limits<int>::max()) {
      vertNormal_[vert] = vec3(0.0);
      return;
    }
    vec3 normal = vec3(0.0);
    ForVert(firstEdge, [&](int edge) {
      ivec3 triVerts = {halfedge_[edge].startVert, halfedge_[edge].endVert,
                        halfedge_[NextHalfedge(edge)].endVert};
      vec3 currEdge =
          la::normalize(vertPos_[triVerts[1]] - vertPos_[triVerts[0]]);
      vec3 prevEdge =
          la::normalize(vertPos_[triVerts[0]] - vertPos_[triVerts[2]]);

      // if it is not finite, this means that the triangle is degenerate, and we
      // should just exclude it from the normal calculation...
      if (!la::isfinite(currEdge[0]) || !la::isfinite(prevEdge[0])) return;
      double dot = -la::dot(prevEdge, currEdge);
      double phi = dot >= 1 ? 0 : (dot <= -1 ? kPi : std::acos(dot));
      normal += phi * faceNormal_[edge / 3];
    });
    vertNormal_[vert] = SafeNormalize(normal);
  });
}

/**
 * Remaps all the contained meshIDs to new unique values to represent new
 * instances of these meshes.
 */
void Manifold::Impl::IncrementMeshIDs() {
  HashTable<uint32_t> meshIDold2new(meshRelation_.meshIDtransform.size() * 2);
  // Update keys of the transform map
  std::map<int, Relation> oldTransforms;
  std::swap(meshRelation_.meshIDtransform, oldTransforms);
  const int numMeshIDs = oldTransforms.size();
  int nextMeshID = ReserveIDs(numMeshIDs);
  for (const auto& pair : oldTransforms) {
    meshIDold2new.D().Insert(pair.first, nextMeshID);
    meshRelation_.meshIDtransform[nextMeshID++] = pair.second;
  }

  const size_t numTri = NumTri();
  for_each_n(autoPolicy(numTri, 1e5), meshRelation_.triRef.begin(), numTri,
             UpdateMeshID({meshIDold2new.D()}));
}

#ifdef MANIFOLD_DEBUG
std::ostream& operator<<(std::ostream& stream, const Manifold::Impl& impl) {
  stream << std::setprecision(17);  // for double precision
  stream << "# ======= begin mesh ======" << std::endl;
  stream << "# tolerance = " << impl.tolerance_ << std::endl;
  stream << "# epsilon = " << impl.epsilon_ << std::endl;
  // TODO: Mesh relation, vertex normal and face normal
  for (const vec3& v : impl.vertPos_)
    stream << "v " << v.x << " " << v.y << " " << v.z << std::endl;
  std::vector<ivec3> triangles;
  triangles.reserve(impl.halfedge_.size() / 3);
  for (size_t i = 0; i < impl.halfedge_.size(); i += 3)
    triangles.emplace_back(impl.halfedge_[i].startVert + 1,
                           impl.halfedge_[i + 1].startVert + 1,
                           impl.halfedge_[i + 2].startVert + 1);
  sort(triangles.begin(), triangles.end());
  for (const auto& tri : triangles)
    stream << "f " << tri.x << " " << tri.y << " " << tri.z << std::endl;
  stream << "# ======== end mesh =======" << std::endl;
  return stream;
}
#endif

#ifdef MANIFOLD_EXPORT
Manifold Manifold::ImportMeshGL64(std::istream& stream) {
  MeshGL64 mesh;
  std::optional<double> epsilon;
  stream.precision(17);
  while (true) {
    char c = stream.get();
    if (stream.eof()) break;
    switch (c) {
      case '#': {
        char c = stream.get();
        if (c == ' ') {
          constexpr int SIZE = 10;
          std::array<char, SIZE> tmp;
          stream.get(tmp.data(), SIZE, '\n');
          if (strncmp(tmp.data(), "tolerance", SIZE) == 0) {
            // skip 3 letters
            for (int i : {0, 1, 2}) stream.get();
            stream >> mesh.tolerance;
          } else if (strncmp(tmp.data(), "epsilon =", SIZE) == 0) {
            double tmp;
            stream >> tmp;
            epsilon = {tmp};
          } else {
            // add it back because it is not what we want
            int end = 0;
            while (end < SIZE && tmp[end] != 0) end++;
            while (--end > -1) stream.putback(tmp[end]);
          }
          c = stream.get();
        }
        // just skip the remaining comment
        while (c != '\n' && !stream.eof()) {
          c = stream.get();
        }
        break;
      }
      case 'v':
        for (int i : {0, 1, 2}) {
          double x;
          stream >> x;
          mesh.vertProperties.push_back(x);
        }
        break;
      case 'f':
        for (int i : {0, 1, 2}) {
          uint64_t x;
          stream >> x;
          mesh.triVerts.push_back(x - 1);
        }
        break;
      case '\n':
        break;
      default:
        DEBUG_ASSERT(false, userErr, "unexpected character in MeshGL64 import");
    }
  }
  auto m = std::make_shared<Manifold::Impl>(mesh);
  if (epsilon) m->SetEpsilon(*epsilon);
  return Manifold(m);
}
#endif

}  // namespace manifold
