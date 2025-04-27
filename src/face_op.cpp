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

#include <unordered_set>

#include "impl.h"
#include "manifold/common.h"
#include "manifold/polygon.h"
#include "parallel.h"
#include "shared.h"

#if (MANIFOLD_PAR == 1) && __has_include(<tbb/concurrent_map.h>)
#include <tbb/tbb.h>
#define TBB_PREVIEW_CONCURRENT_ORDERED_CONTAINERS 1
#include <tbb/concurrent_map.h>
#endif

namespace {
using namespace manifold;

/**
 * Returns an assembled set of vertex index loops of the input list of
 * Halfedges, where each vert must be referenced the same number of times as a
 * startVert and endVert. If startHalfedgeIdx is given, instead of putting
 * vertex indices into the returned polygons structure, it will use the halfedge
 * indices instead.
 */
std::vector<std::vector<int>> AssembleHalfedges(VecView<Halfedge>::IterC start,
                                                VecView<Halfedge>::IterC end,
                                                const int startHalfedgeIdx) {
  std::multimap<int, int> vert_edge;
  for (auto edge = start; edge != end; ++edge) {
    vert_edge.emplace(
        std::make_pair(edge->startVert, static_cast<int>(edge - start)));
  }

  std::vector<std::vector<int>> polys;
  int startEdge = 0;
  int thisEdge = startEdge;
  while (1) {
    if (thisEdge == startEdge) {
      if (vert_edge.empty()) break;
      startEdge = vert_edge.begin()->second;
      thisEdge = startEdge;
      polys.push_back({});
    }
    polys.back().push_back(startHalfedgeIdx + thisEdge);
    const auto result = vert_edge.find((start + thisEdge)->endVert);
    DEBUG_ASSERT(result != vert_edge.end(), topologyErr, "non-manifold edge");
    thisEdge = result->second;
    vert_edge.erase(result);
  }
  return polys;
}

/**
 * Add the vertex position projection to the indexed polygons.
 */
PolygonsIdx ProjectPolygons(const std::vector<std::vector<int>>& polys,
                            const Vec<Halfedge>& halfedge,
                            const Vec<vec3>& vertPos, mat2x3 projection) {
  PolygonsIdx polygons;
  for (const auto& poly : polys) {
    polygons.push_back({});
    for (const auto& edge : poly) {
      polygons.back().push_back(
          {projection * vertPos[halfedge[edge].startVert], edge});
    }  // for vert
  }  // for poly
  return polygons;
}

uint64_t PackEdgeFace(int edge, int face) {
  return static_cast<uint64_t>(edge) << 32 | static_cast<uint64_t>(face);
}

int UnpackEdge(uint64_t data) { return static_cast<int>(data >> 32); }
int UnpackFace(uint64_t data) { return static_cast<int>(data & 0xFFFFFFFF); }
}  // namespace

namespace manifold {

using GeneralTriangulation = std::function<std::vector<ivec3>(int)>;
using AddTriangle = std::function<void(int, ivec3, vec3, TriRef)>;

/**
 * Triangulates the faces. In this case, the halfedge_ vector is not yet a set
 * of triangles as required by this data structure, but is instead a set of
 * general faces with the input faceEdge vector having length of the number of
 * faces + 1. The values are indicies into the halfedge_ vector for the first
 * edge of each face, with the final value being the length of the halfedge_
 * vector itself. Upon return, halfedge_ has been lengthened and properly
 * represents the mesh as a set of triangles as usual. In this process the
 * faceNormal_ values are retained, repeated as necessary.
 */
void Manifold::Impl::Face2Tri(const Vec<int>& faceEdge,
                              const Vec<TriRef>& halfedgeRef,
                              bool allowConvex) {
  ZoneScoped;
  Vec<ivec3> triVerts;
  Vec<vec3> triNormal;
  Vec<ivec3> triProp;
  Vec<TriRef>& triRef = meshRelation_.triRef;
  triRef.clear();
  auto processFace = [&](GeneralTriangulation general, AddTriangle addTri,
                         int face) {
    const int firstEdge = faceEdge[face];
    const int lastEdge = faceEdge[face + 1];
    const int numEdge = lastEdge - firstEdge;
    if (numEdge == 0) return;
    DEBUG_ASSERT(numEdge >= 3, topologyErr, "face has less than three edges.");
    const vec3 normal = faceNormal_[face];

    if (numEdge == 3) {  // Single triangle
      ivec3 triEdge(firstEdge, firstEdge + 1, firstEdge + 2);
      ivec3 tri(halfedge_[firstEdge].startVert,
                halfedge_[firstEdge + 1].startVert,
                halfedge_[firstEdge + 2].startVert);
      ivec3 ends(halfedge_[firstEdge].endVert, halfedge_[firstEdge + 1].endVert,
                 halfedge_[firstEdge + 2].endVert);
      if (ends[0] == tri[2]) {
        std::swap(triEdge[1], triEdge[2]);
        std::swap(tri[1], tri[2]);
        std::swap(ends[1], ends[2]);
      }
      DEBUG_ASSERT(ends[0] == tri[1] && ends[1] == tri[2] && ends[2] == tri[0],
                   topologyErr, "These 3 edges do not form a triangle!");

      addTri(face, triEdge, normal, halfedgeRef[firstEdge]);
      // } else if (numEdge == 4) {  // Pair of triangles
      //   const mat2x3 projection = GetAxisAlignedProjection(normal);
      //   auto triCCW = [&projection, this](const ivec3 tri) {
      //     return CCW(projection * this->vertPos_[tri[0]],
      //                projection * this->vertPos_[tri[1]],
      //                projection * this->vertPos_[tri[2]], epsilon_) >= 0;
      //   };

      //   ivec3 tri0(halfedge_[firstEdge].startVert,
      //   halfedge_[firstEdge].endVert,
      //              -1);
      //   ivec3 tri1(-1, -1, tri0[0]);
      //   for (const int i : {1, 2, 3}) {
      //     if (halfedge_[firstEdge + i].startVert == tri0[1]) {
      //       tri0[2] = halfedge_[firstEdge + i].endVert;
      //       tri1[0] = tri0[2];
      //     }
      //     if (halfedge_[firstEdge + i].endVert == tri0[0]) {
      //       tri1[1] = halfedge_[firstEdge + i].startVert;
      //     }
      //   }
      //   DEBUG_ASSERT(la::all(la::gequal(tri0, ivec3(0))) &&
      //                    la::all(la::gequal(tri1, ivec3(0))),
      //                topologyErr, "non-manifold quad!");
      //   bool firstValid = triCCW(tri0) && triCCW(tri1);
      //   tri0[2] = tri1[1];
      //   tri1[2] = tri0[1];
      //   bool secondValid = triCCW(tri0) && triCCW(tri1);

      //   if (!secondValid) {
      //     tri0[2] = tri1[0];
      //     tri1[2] = tri0[0];
      //   } else if (firstValid) {
      //     vec3 firstCross = vertPos_[tri0[0]] - vertPos_[tri1[0]];
      //     vec3 secondCross = vertPos_[tri0[1]] - vertPos_[tri1[1]];
      //     if (la::dot(firstCross, firstCross) <
      //         la::dot(secondCross, secondCross)) {
      //       tri0[2] = tri1[0];
      //       tri1[2] = tri0[0];
      //     }
      //   }

      //   for (const auto& tri : {tri0, tri1}) {
      //     addTri(face, tri, normal, halfedgeRef[firstEdge]);
      //   }
    } else {  // General triangulation
      for (const auto& tri : general(face)) {
        addTri(face, tri, normal, halfedgeRef[firstEdge]);
      }
    }
  };
  auto generalTriangulation = [&](int face) {
    const vec3 normal = faceNormal_[face];
    const mat2x3 projection = GetAxisAlignedProjection(normal);
    const PolygonsIdx polys = ProjectPolygons(
        AssembleHalfedges(halfedge_.cbegin() + faceEdge[face],
                          halfedge_.cbegin() + faceEdge[face + 1],
                          faceEdge[face]),
        halfedge_, vertPos_, projection);
    return TriangulateIdx(polys, epsilon_, allowConvex);
  };
#if (MANIFOLD_PAR == 1) && __has_include(<tbb/tbb.h>)
  tbb::task_group group;
  // map from face to triangle
  tbb::concurrent_unordered_map<int, std::vector<ivec3>> results;
  Vec<size_t> triCount(faceEdge.size());
  triCount.back() = 0;
  // precompute number of triangles per face, and launch async tasks to
  // triangulate complex faces
  for_each(autoPolicy(faceEdge.size(), 1e5), countAt(0_uz),
           countAt(faceEdge.size() - 1), [&](size_t face) {
             triCount[face] = faceEdge[face + 1] - faceEdge[face] - 2;
             DEBUG_ASSERT(triCount[face] >= 1, topologyErr,
                          "face has less than three edges.");
             if (triCount[face] > 1)
               group.run([&, face] {
                 std::vector<ivec3> newTris = generalTriangulation(face);
                 triCount[face] = newTris.size();
                 results[face] = std::move(newTris);
               });
           });
  group.wait();
  // prefix sum computation (assign unique index to each face) and preallocation
  exclusive_scan(triCount.begin(), triCount.end(), triCount.begin(), 0_uz);
  triVerts.resize(triCount.back());
  triProp.resize(triCount.back());
  triNormal.resize(triCount.back());
  triRef.resize(triCount.back());

  auto processFace2 = std::bind(
      processFace, [&](size_t face) { return std::move(results[face]); },
      [&](size_t face, ivec3 tri, vec3 normal, TriRef r) {
        for (const int i : {0, 1, 2}) {
          triVerts[triCount[face]][i] = halfedge_[tri[i]].startVert;
          triProp[triCount[face]][i] = halfedge_[tri[i]].propVert;
        }
        triNormal[triCount[face]] = normal;
        triRef[triCount[face]] = r;
        triCount[face]++;
      },
      std::placeholders::_1);
  // set triangles in parallel
  for_each(autoPolicy(faceEdge.size(), 1e4), countAt(0_uz),
           countAt(faceEdge.size() - 1), processFace2);
#else
  triVerts.reserve(faceEdge.size());
  triNormal.reserve(faceEdge.size());
  triRef.reserve(faceEdge.size());
  auto processFace2 = std::bind(
      processFace, generalTriangulation,
      [&](size_t, ivec3 tri, vec3 normal, TriRef r) {
        ivec3 verts;
        ivec3 props;
        for (const int i : {0, 1, 2}) {
          verts[i] = halfedge_[tri[i]].startVert;
          props[i] = halfedge_[tri[i]].propVert;
        }
        triVerts.push_back(verts);
        triProp.push_back(props);
        triNormal.push_back(normal);
        triRef.push_back(r);
      },
      std::placeholders::_1);
  for (size_t face = 0; face < faceEdge.size() - 1; ++face) {
    processFace2(face);
  }
#endif

  faceNormal_ = std::move(triNormal);
  CreateHalfedges(triProp, triVerts);
}

constexpr uint64_t kRemove = std::numeric_limits<uint64_t>::max();

void Manifold::Impl::FlattenFaces() {
  Vec<uint64_t> edgeFace(halfedge_.size());
  const size_t numTri = NumTri();
  const auto policy = autoPolicy(numTri);

  std::vector<std::atomic<int>> vertDegree(NumVert());
  for_each(policy, vertDegree.begin(), vertDegree.end(),
           [](auto& v) { v.store(0); });

  for_each_n(policy, countAt(0_uz), numTri,
             [&edgeFace, &vertDegree, this](size_t tri) {
               for (const int i : {0, 1, 2}) {
                 const int edge = 3 * tri + i;
                 const int pair = halfedge_[edge].pairedHalfedge;
                 if (pair < 0) {
                   edgeFace[edge] = kRemove;
                   return;
                 }
                 const auto& ref = meshRelation_.triRef[tri];
                 if (ref.SameFace(meshRelation_.triRef[pair / 3])) {
                   edgeFace[edge] = kRemove;
                 } else {
                   edgeFace[edge] = (static_cast<uint64_t>(ref.meshID) << 32) +
                                    static_cast<uint64_t>(ref.coplanarID);
                   ++vertDegree[halfedge_[edge].startVert];
                 }
               }
             });

  Vec<size_t> newHalf2Old(halfedge_.size());
  sequence(newHalf2Old.begin(), newHalf2Old.end());
  stable_sort(
      newHalf2Old.begin(), newHalf2Old.end(),
      [&edgeFace](size_t a, size_t b) { return edgeFace[a] < edgeFace[b]; });
  newHalf2Old.resize(std::find_if(countAt(0_uz), countAt(halfedge_.size()),
                                  [&](const size_t i) {
                                    return edgeFace[newHalf2Old[i]] == kRemove;
                                  }) -
                     countAt(0_uz));

  Vec<Halfedge> newHalfedge(newHalf2Old.size());
  for_each_n(policy, countAt(0_uz), newHalf2Old.size(),
             [&](size_t i) { newHalfedge[i] = halfedge_[newHalf2Old[i]]; });

  Vec<int> faceEdge(1, 0);
  for (size_t i = 1; i < newHalf2Old.size(); ++i) {
    if (edgeFace[newHalf2Old[i]] != edgeFace[newHalf2Old[i - 1]]) {
      faceEdge.push_back(i);
    }
  }
  const size_t numFace = faceEdge.size();
  faceEdge.push_back(newHalf2Old.size());

  if (numFace < 4) {
    MakeEmpty(Error::NoError);
    return;  // empty
  } else if (numFace == numTri) {
    return;  // already flat
  }

  Vec<TriRef> halfedgeRef(halfedge_.size());
  Vec<vec3> oldFaceNormal = std::move(faceNormal_);
  faceNormal_.resize(numFace);

  std::atomic<uint64_t> faceData(0);
  Vec<int> faceEdge2(faceEdge.size());
  faceEdge2[0] = 0;
  Vec<Halfedge> newHalfedge2(newHalfedge.size());
  for_each_n(policy, countAt(0_uz), numFace, [&](size_t inFace) {
    std::vector<std::vector<int>> polys = AssembleHalfedges(
        newHalfedge.cbegin() + faceEdge[inFace],
        newHalfedge.cbegin() + faceEdge[inFace + 1], faceEdge[inFace]);
    Vec<Halfedge> polys2;
    for (const auto& poly : polys) {
      int start = -1;
      int last = -1;
      for (const int edge : poly) {
        // only keep vert touching 3 or more faces
        if (vertDegree[newHalfedge[edge].startVert] < 3) continue;
        if (start == -1) {
          start = edge;
        } else if (last != -1) {
          polys2.push_back({newHalfedge[last].startVert,
                            newHalfedge[edge].startVert, -1,
                            newHalfedge[last].propVert});
        }
        last = edge;
      }
      polys2.push_back({newHalfedge[last].startVert,
                        newHalfedge[start].startVert, -1,
                        newHalfedge[last].propVert});
    }

    const int numEdge = polys2.size();
    if (numEdge < 3) return;
    const uint64_t inc = PackEdgeFace(numEdge, 1);
    const uint64_t data = faceData.fetch_add(inc);
    const int start = UnpackEdge(data);
    const int outFace = UnpackFace(data);
    if (numEdge > 0) {
      std::copy(polys2.begin(), polys2.end(), newHalfedge2.begin() + start);
    }
    faceEdge2[outFace + 1] = start + numEdge;

    const int oldFace = newHalf2Old[faceEdge[inFace]] / 3;
    // only fill values that will get read.
    halfedgeRef[start] = meshRelation_.triRef[oldFace];
    faceNormal_[outFace] = oldFaceNormal[oldFace];
  });

  const int startFace = UnpackEdge(faceData);
  const int outFace = UnpackFace(faceData);

  if (startFace == 0) {
    MakeEmpty(Error::NoError);
    return;  // empty
  }

  newHalfedge2.resize(startFace);
  faceEdge2.resize(outFace + 1);
  halfedge_ = std::move(newHalfedge2);

  // mark unreferenced verts for removal
  for_each_n(policy, countAt(0_uz), NumVert(), [&](size_t vert) {
    if (vertDegree[vert] < 3) vertPos_[vert] = vec3(NAN);
  });

  Face2Tri(faceEdge2, halfedgeRef, false);
  Finish();
}

Polygons Manifold::Impl::Slice(double height) const {
  Box plane = bBox_;
  plane.min.z = plane.max.z = height;
  Vec<Box> query;
  query.push_back(plane);

  std::unordered_set<int> tris;
  auto recordCollision = [&](int, int tri) {
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    for (const int j : {0, 1, 2}) {
      const double z = vertPos_[halfedge_[3 * tri + j].startVert].z;
      min = std::min(min, z);
      max = std::max(max, z);
    }

    if (min <= height && max > height) {
      tris.insert(tri);
    }
  };

  auto recorder = MakeSimpleRecorder(recordCollision);
  collider_.Collisions<false>(query.cview(), recorder, false);

  Polygons polys;
  while (!tris.empty()) {
    const int startTri = *tris.begin();
    SimplePolygon poly;

    int k = 0;
    for (const int j : {0, 1, 2}) {
      if (vertPos_[halfedge_[3 * startTri + j].startVert].z > height &&
          vertPos_[halfedge_[3 * startTri + Next3(j)].startVert].z <= height) {
        k = Next3(j);
        break;
      }
    }

    int tri = startTri;
    do {
      tris.erase(tris.find(tri));
      if (vertPos_[halfedge_[3 * tri + k].endVert].z <= height) {
        k = Next3(k);
      }

      Halfedge up = halfedge_[3 * tri + k];
      const vec3 below = vertPos_[up.startVert];
      const vec3 above = vertPos_[up.endVert];
      const double a = (height - below.z) / (above.z - below.z);
      poly.push_back(vec2(la::lerp(below, above, a)));

      const int pair = up.pairedHalfedge;
      tri = pair / 3;
      k = Next3(pair % 3);
    } while (tri != startTri);

    polys.push_back(poly);
  }

  return polys;
}

Polygons Manifold::Impl::Project() const {
  const mat2x3 projection = GetAxisAlignedProjection({0, 0, 1});
  Vec<Halfedge> cusps(NumEdge());
  cusps.resize(
      copy_if(
          halfedge_.cbegin(), halfedge_.cend(), cusps.begin(),
          [&](Halfedge edge) {
            return faceNormal_[halfedge_[edge.pairedHalfedge].pairedHalfedge /
                               3]
                           .z >= 0 &&
                   faceNormal_[edge.pairedHalfedge / 3].z < 0;
          }) -
      cusps.begin());

  PolygonsIdx polysIndexed =
      ProjectPolygons(AssembleHalfedges(cusps.cbegin(), cusps.cend(), 0), cusps,
                      vertPos_, projection);

  Polygons polys;
  for (const auto& poly : polysIndexed) {
    SimplePolygon simple;
    for (const PolyVert& polyVert : poly) {
      simple.push_back(polyVert.pos);
    }
    polys.push_back(simple);
  }

  return polys;
}
}  // namespace manifold
