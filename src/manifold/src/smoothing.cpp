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

#include <map>

#include "impl.h"
#include "par.h"

template <>
struct std::hash<glm::ivec4> {
  size_t operator()(const glm::ivec4& p) const {
    return std::hash<int>()(p.x) ^ std::hash<int>()(p.y) ^
           std::hash<int>()(p.z) ^ std::hash<int>()(p.w);
  }
};

namespace {
using namespace manifold;

glm::vec3 OrthogonalTo(glm::vec3 in, glm::vec3 ref) {
  in -= glm::dot(in, ref) * ref;
  return in;
}

// Calculate a tangent vector in the form of a weighted cubic Bezier taking as
// input the desired tangent direction (length doesn't matter) and the edge
// vector to the neighboring vertex. In a symmetric situation where the tangents
// at each end are mirror images of each other, this will result in a circular
// arc.
glm::vec4 CircularTangent(const glm::vec3& tangent, const glm::vec3& edgeVec) {
  const glm::vec3 dir = SafeNormalize(tangent);

  float weight = glm::abs(glm::dot(dir, SafeNormalize(edgeVec)));
  if (weight == 0) {
    weight = 1;
  }
  // Quadratic weighted bezier for circular interpolation
  const glm::vec4 bz2 =
      weight * glm::vec4(dir * glm::length(edgeVec) / (2 * weight), 1);
  // Equivalent cubic weighted bezier
  const glm::vec4 bz3 = glm::mix(glm::vec4(0, 0, 0, 1), bz2, 2 / 3.0f);
  // Convert from homogeneous form to geometric form
  return glm::vec4(glm::vec3(bz3) / bz3.w, bz3.w);
}

struct ReindexHalfedge {
  VecView<int> half2Edge;
  const VecView<Halfedge> halfedges;

  void operator()(thrust::tuple<int, TmpEdge> in) {
    const int edge = thrust::get<0>(in);
    const int halfedge = thrust::get<1>(in).halfedgeIdx;

    half2Edge[halfedge] = edge;
    half2Edge[halfedges[halfedge].pairedHalfedge] = edge;
  }
};

struct SmoothBezier {
  const Manifold::Impl* impl;
  VecView<const glm::vec3> vertNormal;

  void operator()(thrust::tuple<glm::vec4&, Halfedge, int> inOut) {
    glm::vec4& tangent = thrust::get<0>(inOut);
    const Halfedge edge = thrust::get<1>(inOut);
    const int edgeIdx = thrust::get<2>(inOut);

    if (impl->IsInsideQuad(edgeIdx)) {
      tangent = glm::vec4(0, 0, 0, -1);
      return;
    }

    const glm::vec3 edgeVec =
        impl->vertPos_[edge.endVert] - impl->vertPos_[edge.startVert];
    const glm::vec3 edgeNormal =
        (impl->faceNormal_[edge.face] +
         impl->faceNormal_[impl->halfedge_[edge.pairedHalfedge].face]) /
        2.0f;
    glm::vec3 dir =
        glm::cross(glm::cross(edgeNormal, edgeVec), vertNormal[edge.startVert]);
    tangent = CircularTangent(dir, edgeVec);
  }
};

struct InterpTri {
  const Manifold::Impl* impl;

  glm::vec4 Homogeneous(glm::vec4 v) const {
    v.x *= v.w;
    v.y *= v.w;
    v.z *= v.w;
    return v;
  }

  glm::vec4 Homogeneous(glm::vec3 v) const { return glm::vec4(v, 1.0f); }

  glm::vec3 HNormalize(glm::vec4 v) const { return glm::vec3(v) / v.w; }

  glm::vec4 Bezier(glm::vec3 point, glm::vec4 tangent) const {
    return Homogeneous(glm::vec4(point, 0) + tangent);
  }

  glm::mat2x4 CubicBezier2Linear(glm::vec4 p0, glm::vec4 p1, glm::vec4 p2,
                                 glm::vec4 p3, float x) const {
    glm::mat2x4 out;
    glm::vec4 p12 = glm::mix(p1, p2, x);
    out[0] = glm::mix(glm::mix(p0, p1, x), p12, x);
    out[1] = glm::mix(p12, glm::mix(p2, p3, x), x);
    return out;
  }

  glm::vec3 BezierPoint(glm::mat2x4 points, float x) const {
    return HNormalize(glm::mix(points[0], points[1], x));
  }

  glm::vec3 BezierTangent(glm::mat2x4 points) const {
    return glm::normalize(HNormalize(points[1]) - HNormalize(points[0]));
  }

  glm::mat2x4 Bezier2Bezier(const glm::mat2x3& corners,
                            const glm::mat2x4& tangentsX,
                            const glm::mat2x4& tangentsY, float x) const {
    const glm::mat2x4 bez = CubicBezier2Linear(
        Homogeneous(corners[0]), Bezier(corners[0], tangentsX[0]),
        Bezier(corners[1], tangentsX[1]), Homogeneous(corners[1]), x);
    const glm::vec3 end = BezierPoint(bez, x);
    const glm::vec3 tangent = BezierTangent(bez);

    const glm::mat2x3 biTangents = {
        SafeNormalize(OrthogonalTo(glm::vec3(tangentsY[0]),
                                   SafeNormalize(glm::vec3(tangentsX[0])))),
        SafeNormalize(OrthogonalTo(glm::vec3(tangentsY[1]),
                                   -SafeNormalize(glm::vec3(tangentsX[1]))))};
    const glm::vec3 normal = SafeNormalize(
        glm::cross(glm::mix(biTangents[0], biTangents[1], x), tangent));
    const glm::vec3 delta = OrthogonalTo(
        glm::mix(glm::vec3(tangentsY[0]), glm::vec3(tangentsY[1]), x), normal);
    const float deltaW = glm::mix(tangentsY[0].w, tangentsY[1].w, x);

    return {Homogeneous(end), Homogeneous(glm::vec4(end + delta, deltaW))};
  }

  glm::vec3 Bezier2D(const glm::mat4x3& corners, const glm::mat4& tangentsX,
                     const glm::mat4& tangentsY, float x, float y) const {
    glm::mat2x4 bez0 =
        Bezier2Bezier({corners[0], corners[1]}, {tangentsX[0], tangentsX[1]},
                      {tangentsY[0], tangentsY[1]}, x);
    glm::mat2x4 bez1 =
        Bezier2Bezier({corners[2], corners[3]}, {tangentsX[2], tangentsX[3]},
                      {tangentsY[2], tangentsY[3]}, 1 - x);

    const glm::mat2x4 bez =
        CubicBezier2Linear(bez0[0], bez0[1], bez1[1], bez1[0], y);
    return BezierPoint(bez, y);
  }

  void operator()(thrust::tuple<glm::vec3&, Barycentric> inOut) {
    glm::vec3& pos = thrust::get<0>(inOut);
    const int tri = thrust::get<1>(inOut).tri;
    const glm::vec4 uvw = thrust::get<1>(inOut).uvw;

    const glm::ivec4 halfedges = impl->GetHalfedges(tri);
    const glm::mat4x3 corners = {
        impl->vertPos_[impl->halfedge_[halfedges[0]].startVert],
        impl->vertPos_[impl->halfedge_[halfedges[1]].startVert],
        impl->vertPos_[impl->halfedge_[halfedges[2]].startVert],
        halfedges[3] < 0
            ? glm::vec3(0)
            : impl->vertPos_[impl->halfedge_[halfedges[3]].startVert]};

    for (const int i : {0, 1, 2, 3}) {
      if (uvw[i] == 1) {
        pos = corners[i];
        return;
      }
    }

    glm::vec4 posH(0);

    if (halfedges[3] < 0) {  // tri
      const glm::mat3x4 tangentR = {impl->halfedgeTangent_[halfedges[0]],
                                    impl->halfedgeTangent_[halfedges[1]],
                                    impl->halfedgeTangent_[halfedges[2]]};
      const glm::mat3x4 tangentL = {
          impl->halfedgeTangent_[impl->halfedge_[halfedges[2]].pairedHalfedge],
          impl->halfedgeTangent_[impl->halfedge_[halfedges[0]].pairedHalfedge],
          impl->halfedgeTangent_[impl->halfedge_[halfedges[1]].pairedHalfedge]};

      for (const int i : {0, 1, 2}) {
        const int j = (i + 1) % 3;
        const int k = (i + 2) % 3;
        const float x = uvw[k] / (1 - uvw[i]);

        const glm::mat2x4 bez =
            Bezier2Bezier({corners[j], corners[k]}, {tangentR[j], tangentL[k]},
                          {tangentL[j], tangentR[k]}, x);

        const glm::mat2x4 bez1 = CubicBezier2Linear(
            bez[0], bez[1],
            // Homogeneous(end), Homogeneous(glm::vec4(end + delta, deltaW)),
            Bezier(corners[i], glm::mix(tangentR[i], tangentL[i], x)),
            Homogeneous(corners[i]), uvw[i]);
        const glm::vec3 p = BezierPoint(bez1, uvw[i]);
        float w = uvw[j] * uvw[j] * uvw[k] * uvw[k];
        posH += Homogeneous(glm::vec4(p, w));
      }
    } else {  // quad
      const glm::mat4 tangentsX = {
          impl->halfedgeTangent_[halfedges[0]],
          impl->halfedgeTangent_[impl->halfedge_[halfedges[0]].pairedHalfedge],
          impl->halfedgeTangent_[halfedges[2]],
          impl->halfedgeTangent_[impl->halfedge_[halfedges[2]].pairedHalfedge]};
      const glm::mat4 tangentsY = {
          impl->halfedgeTangent_[impl->halfedge_[halfedges[3]].pairedHalfedge],
          impl->halfedgeTangent_[halfedges[1]],
          impl->halfedgeTangent_[impl->halfedge_[halfedges[1]].pairedHalfedge],
          impl->halfedgeTangent_[halfedges[3]]};
      const float x = uvw[1] + uvw[2];
      const float y = uvw[2] + uvw[3];
      const glm::vec3 pX = Bezier2D(corners, tangentsX, tangentsY, x, y);
      const glm::vec3 pY = Bezier2D(
          {corners[1], corners[2], corners[3], corners[0]},
          {tangentsY[1], tangentsY[2], tangentsY[3], tangentsY[0]},
          {tangentsX[1], tangentsX[2], tangentsX[3], tangentsX[0]}, y, 1 - x);
      posH += Homogeneous(glm::vec4(pX, x * (1 - x)));
      posH += Homogeneous(glm::vec4(pY, y * (1 - y)));
    }
    pos = HNormalize(posH);
  }
};

class Partition {
 public:
  // The cached partitions don't have idx - it's added to the copy returned
  // from GetPartition that contains the mapping of the input divisions into the
  // sorted divisions that are uniquely cached.
  glm::ivec4 idx;
  glm::ivec4 sortedDivisions;
  Vec<glm::vec4> vertBary;
  Vec<glm::ivec3> triVert;

  int InteriorOffset() const {
    return sortedDivisions[0] + sortedDivisions[1] + sortedDivisions[2] +
           sortedDivisions[3];
  }

  int NumInterior() const { return vertBary.size() - InteriorOffset(); }

  static Partition GetPartition(glm::ivec4 divisions) {
    if (divisions[0] == 0) return Partition();  // skip wrong side of quad

    glm::ivec4 sortedDiv = divisions;
    glm::ivec4 triIdx = {0, 1, 2, 3};
    if (divisions[3] == 0) {  // triangle
      if (sortedDiv[2] > sortedDiv[1]) {
        std::swap(sortedDiv[2], sortedDiv[1]);
        std::swap(triIdx[2], triIdx[1]);
      }
      if (sortedDiv[1] > sortedDiv[0]) {
        std::swap(sortedDiv[1], sortedDiv[0]);
        std::swap(triIdx[1], triIdx[0]);
        if (sortedDiv[2] > sortedDiv[1]) {
          std::swap(sortedDiv[2], sortedDiv[1]);
          std::swap(triIdx[2], triIdx[1]);
        }
      }
    } else {  // quad
      int minIdx = 0;
      int min = divisions[minIdx];
      int next = divisions[1];
      for (const int i : {1, 2, 3}) {
        const int n = divisions[(i + 1) % 4];
        if (divisions[i] < min || (divisions[i] == min && n < next)) {
          minIdx = i;
          min = divisions[i];
          next = n;
        }
      }
      // Backwards (mirrored) quads get a separate cache key for now for
      // simplicity, so there is no reversal necessary for quads when
      // re-indexing.
      glm::ivec4 tmp = sortedDiv;
      for (const int i : {0, 1, 2, 3}) {
        triIdx[i] = (i + minIdx) % 4;
        sortedDiv[i] = tmp[triIdx[i]];
      }
    }

    Partition partition = GetCachedPartition(sortedDiv);
    partition.idx = triIdx;

    return partition;
  }

  Vec<glm::ivec3> Reindex(glm::ivec4 tri, glm::ivec4 edgeOffsets,
                          glm::bvec4 edgeFwd, int interiorOffset) const {
    Vec<int> newVerts;
    newVerts.reserve(vertBary.size());
    glm::ivec4 triIdx = idx;
    glm::ivec4 outTri = {0, 1, 2, 3};
    if (tri[3] < 0 && idx[1] != Next3(idx[0])) {
      triIdx = {idx[2], idx[0], idx[1], idx[3]};
      edgeFwd = glm::not_(edgeFwd);
      std::swap(outTri[0], outTri[1]);
    }
    for (const int i : {0, 1, 2, 3}) {
      if (tri[triIdx[i]] >= 0) newVerts.push_back(tri[triIdx[i]]);
    }
    for (const int i : {0, 1, 2, 3}) {
      const int n = sortedDivisions[i] - 1;
      int offset = edgeOffsets[idx[i]] + (edgeFwd[idx[i]] ? 0 : n - 1);
      for (int j = 0; j < n; ++j) {
        newVerts.push_back(offset);
        offset += edgeFwd[idx[i]] ? 1 : -1;
      }
    }
    const int offset = interiorOffset - newVerts.size();
    for (int i = newVerts.size(); i < vertBary.size(); ++i) {
      newVerts.push_back(i + offset);
    }

    const int numTri = triVert.size();
    Vec<glm::ivec3> newTriVert(numTri);
    for_each_n(
        autoPolicy(numTri), zip(newTriVert.begin(), triVert.begin()), numTri,
        [&outTri, &newVerts](thrust::tuple<glm::ivec3&, glm::ivec3> inOut) {
          for (const int j : {0, 1, 2}) {
            thrust::get<0>(inOut)[outTri[j]] =
                newVerts[thrust::get<1>(inOut)[j]];
          }
        });
    return newTriVert;
  }

 private:
  static inline auto cacheLock = std::mutex();
  static inline auto cache =
      std::unordered_map<glm::ivec4, std::unique_ptr<Partition>>();

  // This triangulation is purely topological - it depends only on the number of
  // divisions of the three sides of the triangle. This allows them to be cached
  // and reused for similar triangles. The shape of the final surface is defined
  // by the tangents and the barycentric coordinates of the new verts. For
  // triangles, the input must be sorted: n[0] >= n[1] >= n[2] > 0.
  static Partition GetCachedPartition(glm::ivec4 n) {
    {
      auto lockGuard = std::lock_guard<std::mutex>(cacheLock);
      auto cached = cache.find(n);
      if (cached != cache.end()) {
        return *cached->second;
      }
    }
    Partition partition;
    partition.sortedDivisions = n;
    if (n[3] > 0) {  // quad
      partition.vertBary.push_back({1, 0, 0, 0});
      partition.vertBary.push_back({0, 1, 0, 0});
      partition.vertBary.push_back({0, 0, 1, 0});
      partition.vertBary.push_back({0, 0, 0, 1});
      glm::ivec4 edgeOffsets;
      edgeOffsets[0] = 4;
      for (const int i : {0, 1, 2, 3}) {
        if (i > 0) {
          edgeOffsets[i] = edgeOffsets[i - 1] + n[i - 1] - 1;
        }
        const glm::vec4 nextBary = partition.vertBary[(i + 1) % 4];
        for (int j = 1; j < n[i]; ++j) {
          partition.vertBary.push_back(
              glm::mix(partition.vertBary[i], nextBary, (float)j / n[i]));
        }
      }
      PartitionQuad(partition.triVert, partition.vertBary, {0, 1, 2, 3},
                    edgeOffsets, n - 1, {true, true, true, true});
    } else {  // tri
      partition.vertBary.push_back({1, 0, 0, 0});
      partition.vertBary.push_back({0, 1, 0, 0});
      partition.vertBary.push_back({0, 0, 1, 0});
      for (const int i : {0, 1, 2}) {
        const glm::vec4 nextBary = partition.vertBary[(i + 1) % 3];
        for (int j = 1; j < n[i]; ++j) {
          partition.vertBary.push_back(
              glm::mix(partition.vertBary[i], nextBary, (float)j / n[i]));
        }
      }
      const glm::ivec3 edgeOffsets = {3, 3 + n[0] - 1, 3 + n[0] - 1 + n[1] - 1};

      const float f = n[2] * n[2] + n[0] * n[0];
      if (n[1] == 1) {
        if (n[0] == 1) {
          partition.triVert.push_back({0, 1, 2});
        } else {
          PartitionFan(partition.triVert, {0, 1, 2}, n[0] - 1, edgeOffsets[0]);
        }
      } else if (n[1] * n[1] >
                 f - glm::sqrt(2.0f) * n[0] * n[2]) {  // acute-ish
        partition.triVert.push_back({edgeOffsets[1] - 1, 1, edgeOffsets[1]});
        PartitionQuad(partition.triVert, partition.vertBary,
                      {edgeOffsets[1] - 1, edgeOffsets[1], 2, 0},
                      {-1, edgeOffsets[1] + 1, edgeOffsets[2], edgeOffsets[0]},
                      {0, n[1] - 2, n[2] - 1, n[0] - 2},
                      {true, true, true, true});
      } else {  // obtuse -> spit into two acute
        // portion of n[0] under n[2]
        const int ns =
            glm::min(n[0] - 2, (int)glm::round((f - n[1] * n[1]) / (2 * n[0])));
        // height from n[0]: nh <= n[2]
        const int nh =
            glm::max(1., glm::round(glm::sqrt(n[2] * n[2] - ns * ns)));

        const int hOffset = partition.vertBary.size();
        const glm::vec4 middleBary =
            partition.vertBary[edgeOffsets[0] + ns - 1];
        for (int j = 1; j < nh; ++j) {
          partition.vertBary.push_back(
              glm::mix(partition.vertBary[2], middleBary, (float)j / nh));
        }

        partition.triVert.push_back({edgeOffsets[1] - 1, 1, edgeOffsets[1]});
        PartitionQuad(
            partition.triVert, partition.vertBary,
            {edgeOffsets[1] - 1, edgeOffsets[1], 2, edgeOffsets[0] + ns - 1},
            {-1, edgeOffsets[1] + 1, hOffset, edgeOffsets[0] + ns},
            {0, n[1] - 2, nh - 1, n[0] - ns - 2}, {true, true, true, true});

        if (n[2] == 1) {
          PartitionFan(partition.triVert, {0, edgeOffsets[0] + ns - 1, 2},
                       ns - 1, edgeOffsets[0]);
        } else {
          if (ns == 1) {
            partition.triVert.push_back({hOffset, 2, edgeOffsets[2]});
            PartitionQuad(partition.triVert, partition.vertBary,
                          {hOffset, edgeOffsets[2], 0, edgeOffsets[0]},
                          {-1, edgeOffsets[2] + 1, -1, hOffset + nh - 2},
                          {0, n[2] - 2, ns - 1, nh - 2},
                          {true, true, true, false});
          } else {
            partition.triVert.push_back({hOffset - 1, 0, edgeOffsets[0]});
            PartitionQuad(
                partition.triVert, partition.vertBary,
                {hOffset - 1, edgeOffsets[0], edgeOffsets[0] + ns - 1, 2},
                {-1, edgeOffsets[0] + 1, hOffset + nh - 2, edgeOffsets[2]},
                {0, ns - 2, nh - 1, n[2] - 2}, {true, true, false, true});
          }
        }
      }
    }

    auto lockGuard = std::lock_guard<std::mutex>(cacheLock);
    cache.insert({n, std::make_unique<Partition>(partition)});
    return partition;
  }

  // Side 0 has added edges while sides 1 and 2 do not. Fan spreads from vert 2.
  static void PartitionFan(Vec<glm::ivec3>& triVert, glm::ivec3 cornerVerts,
                           int added, int edgeOffset) {
    int last = cornerVerts[0];
    for (int i = 0; i < added; ++i) {
      const int next = edgeOffset + i;
      triVert.push_back({last, next, cornerVerts[2]});
      last = next;
    }
    triVert.push_back({last, cornerVerts[1], cornerVerts[2]});
  }

  // Partitions are parallel to the first edge unless two consecutive edgeAdded
  // are zero, in which case a terminal triangulation is performed.
  static void PartitionQuad(Vec<glm::ivec3>& triVert, Vec<glm::vec4>& vertBary,
                            glm::ivec4 cornerVerts, glm::ivec4 edgeOffsets,
                            glm::ivec4 edgeAdded, glm::bvec4 edgeFwd) {
    auto GetEdgeVert = [&](int edge, int idx) {
      return edgeOffsets[edge] + (edgeFwd[edge] ? 1 : -1) * idx;
    };

    ASSERT(glm::all(glm::greaterThanEqual(edgeAdded, glm::ivec4(0))), logicErr,
           "negative divisions!");

    int corner = -1;
    int last = 3;
    int maxEdge = -1;
    for (const int i : {0, 1, 2, 3}) {
      if (corner == -1 && edgeAdded[i] == 0 && edgeAdded[last] == 0) {
        corner = i;
      }
      if (edgeAdded[i] > 0) {
        maxEdge = maxEdge == -1 ? i : -2;
      }
      last = i;
    }
    if (corner >= 0) {  // terminate
      if (maxEdge >= 0) {
        glm::ivec4 edge = (glm::ivec4(0, 1, 2, 3) + maxEdge) % 4;
        const int middle = edgeAdded[maxEdge] / 2;
        triVert.push_back({cornerVerts[edge[2]], cornerVerts[edge[3]],
                           GetEdgeVert(maxEdge, middle)});
        int last = cornerVerts[edge[0]];
        for (int i = 0; i <= middle; ++i) {
          const int next = GetEdgeVert(maxEdge, i);
          triVert.push_back({cornerVerts[edge[3]], last, next});
          last = next;
        }
        last = cornerVerts[edge[1]];
        for (int i = edgeAdded[maxEdge] - 1; i >= middle; --i) {
          const int next = GetEdgeVert(maxEdge, i);
          triVert.push_back({cornerVerts[edge[2]], next, last});
          last = next;
        }
      } else {
        int sideVert = cornerVerts[0];  // initial value is unused
        for (const int j : {1, 2}) {
          const int side = (corner + j) % 4;
          if (j == 2 && edgeAdded[side] > 0) {
            triVert.push_back(
                {cornerVerts[side], GetEdgeVert(side, 0), sideVert});
          } else {
            sideVert = cornerVerts[side];
          }
          for (int i = 0; i < edgeAdded[side]; ++i) {
            const int nextVert = GetEdgeVert(side, i);
            triVert.push_back({cornerVerts[corner], sideVert, nextVert});
            sideVert = nextVert;
          }
          if (j == 2 || edgeAdded[side] == 0) {
            triVert.push_back({cornerVerts[corner], sideVert,
                               cornerVerts[(corner + j + 1) % 4]});
          }
        }
      }
      return;
    }
    // recursively partition
    const int partitions = 1 + glm::min(edgeAdded[1], edgeAdded[3]);
    glm::ivec4 newCornerVerts = {cornerVerts[1], -1, -1, cornerVerts[0]};
    glm::ivec4 newEdgeOffsets = {
        edgeOffsets[1], -1, GetEdgeVert(3, edgeAdded[3] + 1), edgeOffsets[0]};
    glm::ivec4 newEdgeAdded = {0, -1, 0, edgeAdded[0]};
    glm::bvec4 newEdgeFwd = {edgeFwd[1], true, edgeFwd[3], edgeFwd[0]};

    for (int i = 1; i < partitions; ++i) {
      const int cornerOffset1 = (edgeAdded[1] * i) / partitions;
      const int cornerOffset3 =
          edgeAdded[3] - 1 - (edgeAdded[3] * i) / partitions;
      const int nextOffset1 = GetEdgeVert(1, cornerOffset1 + 1);
      const int nextOffset3 = GetEdgeVert(3, cornerOffset3 + 1);
      const int added = glm::round(glm::mix(
          (float)edgeAdded[0], (float)edgeAdded[2], (float)i / partitions));

      newCornerVerts[1] = GetEdgeVert(1, cornerOffset1);
      newCornerVerts[2] = GetEdgeVert(3, cornerOffset3);
      newEdgeAdded[0] = std::abs(nextOffset1 - newEdgeOffsets[0]) - 1;
      newEdgeAdded[1] = added;
      newEdgeAdded[2] = std::abs(nextOffset3 - newEdgeOffsets[2]) - 1;
      newEdgeOffsets[1] = vertBary.size();
      newEdgeOffsets[2] = nextOffset3;

      for (int j = 0; j < added; ++j) {
        vertBary.push_back(glm::mix(vertBary[newCornerVerts[1]],
                                    vertBary[newCornerVerts[2]],
                                    (j + 1.0f) / (added + 1.0f)));
      }

      PartitionQuad(triVert, vertBary, newCornerVerts, newEdgeOffsets,
                    newEdgeAdded, newEdgeFwd);

      newCornerVerts[0] = newCornerVerts[1];
      newCornerVerts[3] = newCornerVerts[2];
      newEdgeAdded[3] = newEdgeAdded[1];
      newEdgeOffsets[0] = nextOffset1;
      newEdgeOffsets[3] = newEdgeOffsets[1] + newEdgeAdded[1] - 1;
      newEdgeFwd[3] = false;
    }

    newCornerVerts[1] = cornerVerts[2];
    newCornerVerts[2] = cornerVerts[3];
    newEdgeOffsets[1] = edgeOffsets[2];
    newEdgeAdded[0] =
        edgeAdded[1] - std::abs(newEdgeOffsets[0] - edgeOffsets[1]);
    newEdgeAdded[1] = edgeAdded[2];
    newEdgeAdded[2] = std::abs(newEdgeOffsets[2] - edgeOffsets[3]) - 1;
    newEdgeOffsets[2] = edgeOffsets[3];
    newEdgeFwd[1] = edgeFwd[2];

    PartitionQuad(triVert, vertBary, newCornerVerts, newEdgeOffsets,
                  newEdgeAdded, newEdgeFwd);
  }
};
}  // namespace

namespace manifold {

/**
 * Get the property normal associated with the startVert of this halfedge, where
 * normalIdx shows the beginning of where normals are stored in the properties.
 */
glm::vec3 Manifold::Impl::GetNormal(int halfedge, int normalIdx) const {
  const int tri = halfedge / 3;
  const int j = halfedge % 3;
  const int prop = meshRelation_.triProperties[tri][j];
  glm::vec3 normal;
  for (const int i : {0, 1, 2}) {
    normal[i] =
        meshRelation_.properties[prop * meshRelation_.numProp + normalIdx + i];
  }
  return normal;
}

/**
 * Returns true if this halfedge should be marked as the interior of a quad, as
 * defined by its two triangles referring to the same face, and those triangles
 * having no further face neighbors beyond.
 */
bool Manifold::Impl::IsInsideQuad(int halfedge) const {
  if (halfedgeTangent_.size() > 0) {
    return halfedgeTangent_[halfedge].w < 0;
  }
  const int tri = halfedge_[halfedge].face;
  const TriRef ref = meshRelation_.triRef[tri];
  const int pair = halfedge_[halfedge].pairedHalfedge;
  const int pairTri = halfedge_[pair].face;
  const TriRef pairRef = meshRelation_.triRef[pairTri];
  if (!ref.SameFace(pairRef)) return false;

  auto SameFace = [this](int halfedge, const TriRef& ref) {
    return ref.SameFace(
        meshRelation_.triRef[halfedge_[halfedge].pairedHalfedge / 3]);
  };

  int neighbor = NextHalfedge(halfedge);
  if (SameFace(neighbor, ref)) return false;
  neighbor = NextHalfedge(neighbor);
  if (SameFace(neighbor, ref)) return false;
  neighbor = NextHalfedge(pair);
  if (SameFace(neighbor, pairRef)) return false;
  neighbor = NextHalfedge(neighbor);
  if (SameFace(neighbor, pairRef)) return false;
  return true;
}

/**
 * Returns true if this halfedge is an interior of a quad, as defined by its
 * halfedge tangent having negative weight.
 */
bool Manifold::Impl::IsMarkedInsideQuad(int halfedge) const {
  return halfedgeTangent_.size() > 0 && halfedgeTangent_[halfedge].w < 0;
}

/**
 * Returns the tri index of the other side of this quad if this tri is part of a
 * quad, or -1 otherwise.
 */
int Manifold::Impl::GetNeighbor(int tri) const {
  int neighbor = -1;
  for (const int i : {0, 1, 2}) {
    if (IsMarkedInsideQuad(3 * tri + i)) {
      neighbor = neighbor == -1 ? i : -2;
    }
  }
  return neighbor;
}

/**
 * For the given triangle index, returns either the three halfedge indices of
 * that triangle and halfedges[3] = -1, or if the triangle is part of a quad, it
 * returns those four indices. If the triangle is part of a quad and is not the
 * lower of the two triangle indices, it returns all -1s.
 */
glm::ivec4 Manifold::Impl::GetHalfedges(int tri) const {
  glm::ivec4 halfedges(-1);
  for (const int i : {0, 1, 2}) {
    halfedges[i] = 3 * tri + i;
  }
  const int neighbor = GetNeighbor(tri);
  if (neighbor >= 0) {  // quad
    const int pair = halfedge_[3 * tri + neighbor].pairedHalfedge;
    if (pair / 3 < tri) {
      return glm::ivec4(-1);  // only process lower tri index
    }
    glm::ivec2 otherHalf;
    otherHalf[0] = NextHalfedge(pair);
    otherHalf[1] = NextHalfedge(otherHalf[0]);
    halfedges[neighbor] = otherHalf[0];
    if (neighbor == 2) {
      halfedges[3] = otherHalf[1];
    } else if (neighbor == 1) {
      halfedges[3] = halfedges[2];
      halfedges[2] = otherHalf[1];
    } else {
      halfedges[3] = halfedges[2];
      halfedges[2] = halfedges[1];
      halfedges[1] = otherHalf[1];
    }
  }
  return halfedges;
}

/**
 * Returns the BaryIndices, which gives the tri and indices (0-3), such that
 * GetHalfedges(val.tri)[val.start4] points back to this halfedge, and val.end4
 * will point to the next one. This function handles this for both triangles and
 * quads.
 */
Manifold::Impl::BaryIndices Manifold::Impl::GetIndices(int halfedge) const {
  int tri = halfedge / 3;
  int idx = halfedge % 3;
  const int neighbor = GetNeighbor(tri);
  if (idx == neighbor) {
    return {-1, -1, -1};
  }

  if (neighbor < 0) {  // tri
    return {tri, idx, Next3(idx)};
  } else {  // quad
    const int pair = halfedge_[3 * tri + neighbor].pairedHalfedge;
    if (pair / 3 < tri) {
      tri = pair / 3;
      const int j = pair % 3;
      idx = Next3(neighbor) == idx ? j : (j + 1) % 4;
    }
    return {tri, idx, (idx + 1) % 4};
  }
}

/**
 * Retained verts are part of several triangles, and it doesn't matter which one
 * the vertBary refers to. Here, whichever is last will win and it's done on the
 * CPU for simplicity for now. Using AtomicCAS on .tri should work for a GPU
 * version if desired.
 */
void Manifold::Impl::FillRetainedVerts(Vec<Barycentric>& vertBary) const {
  const int numTri = halfedge_.size() / 3;
  for (int tri = 0; tri < numTri; ++tri) {
    for (const int i : {0, 1, 2}) {
      const BaryIndices indices = GetIndices(3 * tri + i);
      if (indices.start4 < 0) continue;
      glm::vec4 uvw(0);
      uvw[indices.start4] = 1;
      vertBary[halfedge_[3 * tri + i].startVert] = {indices.tri, uvw};
    }
  }
}

// sharpenedEdges are referenced to the input Mesh, but the triangles have
// been sorted in creating the Manifold, so the indices are converted using
// meshRelation_.
std::vector<Smoothness> Manifold::Impl::UpdateSharpenedEdges(
    const std::vector<Smoothness>& sharpenedEdges) const {
  std::unordered_map<int, int> oldHalfedge2New;
  for (int tri = 0; tri < NumTri(); ++tri) {
    int oldTri = meshRelation_.triRef[tri].tri;
    for (int i : {0, 1, 2}) oldHalfedge2New[3 * oldTri + i] = 3 * tri + i;
  }
  std::vector<Smoothness> newSharp = sharpenedEdges;
  for (Smoothness& edge : newSharp) {
    edge.halfedge = oldHalfedge2New[edge.halfedge];
  }
  return newSharp;
}

// Find faces containing at least 3 triangles - these will not have
// interpolated normals - all their vert normals must match their face normal.
Vec<bool> Manifold::Impl::FlatFaces() const {
  const int numTri = NumTri();
  Vec<bool> triIsFlatFace(numTri, false);
  for_each_n(autoPolicy(numTri), countAt(0), numTri,
             [this, &triIsFlatFace](const int tri) {
               const TriRef& ref = meshRelation_.triRef[tri];
               int faceNeighbors = 0;
               glm::ivec3 faceTris = {-1, -1, -1};
               for (const int j : {0, 1, 2}) {
                 const int neighborTri =
                     halfedge_[halfedge_[3 * tri + j].pairedHalfedge].face;
                 const TriRef& jRef = meshRelation_.triRef[neighborTri];
                 if (jRef.SameFace(ref)) {
                   ++faceNeighbors;
                   faceTris[j] = neighborTri;
                 }
               }
               if (faceNeighbors > 1) {
                 triIsFlatFace[tri] = true;
                 for (const int j : {0, 1, 2}) {
                   if (faceTris[j] >= 0) {
                     triIsFlatFace[faceTris[j]] = true;
                   }
                 }
               }
             });
  return triIsFlatFace;
}

// Returns a vector of length numVert that has a tri that is part of a
// neighboring flat face if there is only one flat face. If there are none it
// gets -1, and if there are more than one it gets -2.
Vec<int> Manifold::Impl::VertFlatFace(const Vec<bool>& flatFaces) const {
  Vec<int> vertFlatFace(NumVert(), -1);
  Vec<TriRef> vertRef(NumVert(), {-1, -1, -1});
  for (int tri = 0; tri < NumTri(); ++tri) {
    if (flatFaces[tri]) {
      for (const int j : {0, 1, 2}) {
        const int vert = halfedge_[3 * tri + j].startVert;
        if (vertRef[vert].SameFace(meshRelation_.triRef[tri])) continue;
        vertRef[vert] = meshRelation_.triRef[tri];
        vertFlatFace[vert] = vertFlatFace[vert] == -1 ? tri : -2;
      }
    }
  }
  return vertFlatFace;
}

std::vector<Smoothness> Manifold::Impl::SharpenEdges(
    float minSharpAngle, float minSmoothness) const {
  std::vector<Smoothness> sharpenedEdges;
  const float minRadians = glm::radians(minSharpAngle);
  for (int e = 0; e < halfedge_.size(); ++e) {
    if (!halfedge_[e].IsForward()) continue;
    const int pair = halfedge_[e].pairedHalfedge;
    const float dihedral =
        glm::acos(glm::dot(faceNormal_[e / 3], faceNormal_[pair / 3]));
    if (dihedral > minRadians) {
      sharpenedEdges.push_back({e, minSmoothness});
      sharpenedEdges.push_back({pair, minSmoothness});
    }
  }
  return sharpenedEdges;
}

/**
 * Instead of calculating the internal shared normals like CalculateNormals
 * does, this method fills in vertex properties, unshared across edges that
 * are bent more than minSharpAngle.
 */
void Manifold::Impl::SetNormals(int normalIdx, float minSharpAngle) {
  if (IsEmpty()) return;
  if (normalIdx < 0) return;

  const int oldNumProp = NumProp();
  const int numTri = NumTri();

  Vec<bool> triIsFlatFace = FlatFaces();
  Vec<int> vertFlatFace = VertFlatFace(triIsFlatFace);
  Vec<int> vertNumSharp(NumVert(), 0);
  for (int e = 0; e < halfedge_.size(); ++e) {
    if (!halfedge_[e].IsForward()) continue;
    const int pair = halfedge_[e].pairedHalfedge;
    const int tri1 = e / 3;
    const int tri2 = pair / 3;
    const float dihedral =
        glm::degrees(glm::acos(glm::dot(faceNormal_[tri1], faceNormal_[tri2])));
    if (dihedral > minSharpAngle) {
      ++vertNumSharp[halfedge_[e].startVert];
      ++vertNumSharp[halfedge_[e].endVert];
    } else {
      const bool faceSplit =
          triIsFlatFace[tri1] != triIsFlatFace[tri2] ||
          (triIsFlatFace[tri1] && triIsFlatFace[tri2] &&
           !meshRelation_.triRef[tri1].SameFace(meshRelation_.triRef[tri2]));
      if (vertFlatFace[halfedge_[e].startVert] == -2 && faceSplit) {
        ++vertNumSharp[halfedge_[e].startVert];
      }
      if (vertFlatFace[halfedge_[e].endVert] == -2 && faceSplit) {
        ++vertNumSharp[halfedge_[e].endVert];
      }
    }
  }

  const int numProp = glm::max(oldNumProp, normalIdx + 3);
  Vec<float> oldProperties(numProp * NumPropVert(), 0);
  meshRelation_.properties.swap(oldProperties);
  meshRelation_.numProp = numProp;
  if (meshRelation_.triProperties.size() == 0) {
    meshRelation_.triProperties.resize(numTri);
    for_each_n(autoPolicy(numTri), countAt(0), numTri, [this](int tri) {
      for (const int j : {0, 1, 2})
        meshRelation_.triProperties[tri][j] = halfedge_[3 * tri + j].startVert;
    });
  }
  Vec<glm::ivec3> oldTriProp(numTri, {-1, -1, -1});
  meshRelation_.triProperties.swap(oldTriProp);

  for (int tri = 0; tri < numTri; ++tri) {
    for (const int i : {0, 1, 2}) {
      if (meshRelation_.triProperties[tri][i] >= 0) continue;
      int startEdge = 3 * tri + i;
      const int vert = halfedge_[startEdge].startVert;

      if (vertNumSharp[vert] < 2) {
        const glm::vec3 normal = vertFlatFace[vert] >= 0
                                     ? faceNormal_[vertFlatFace[vert]]
                                     : vertNormal_[vert];
        int lastProp = -1;
        ForVert(startEdge, [&](int current) {
          const int thisTri = current / 3;
          const int j = current - 3 * thisTri;
          const int prop = oldTriProp[thisTri][j];
          meshRelation_.triProperties[thisTri][j] = prop;
          if (prop == lastProp) return;
          lastProp = prop;
          auto start = oldProperties.begin() + prop * oldNumProp;
          std::copy(start, start + oldNumProp,
                    meshRelation_.properties.begin() + prop * numProp);
          for (const int i : {0, 1, 2})
            meshRelation_.properties[prop * numProp + normalIdx + i] =
                normal[i];
        });
      } else {
        const glm::vec3 centerPos = vertPos_[vert];
        // Length degree
        std::vector<int> group;
        // Length number of normals
        std::vector<glm::vec3> normals;
        int current = startEdge;
        int prevFace = halfedge_[current].face;

        do {
          int next = NextHalfedge(halfedge_[current].pairedHalfedge);
          const int face = halfedge_[next].face;

          const float dihedral = glm::degrees(
              glm::acos(glm::dot(faceNormal_[face], faceNormal_[prevFace])));
          if (dihedral > minSharpAngle ||
              triIsFlatFace[face] != triIsFlatFace[prevFace] ||
              (triIsFlatFace[face] && triIsFlatFace[prevFace] &&
               !meshRelation_.triRef[face].SameFace(
                   meshRelation_.triRef[prevFace]))) {
            break;
          }
          current = next;
          prevFace = face;
        } while (current != startEdge);

        const int endEdge = current;

        struct FaceEdge {
          int face;
          glm::vec3 edgeVec;
        };

        ForVert<FaceEdge>(
            endEdge,
            [this, centerPos](int current) {
              return FaceEdge(
                  {halfedge_[current].face,
                   glm::normalize(vertPos_[halfedge_[current].endVert] -
                                  centerPos)});
            },
            [this, &triIsFlatFace, &normals, &group, minSharpAngle](
                int current, const FaceEdge& here, const FaceEdge& next) {
              const float dihedral = glm::degrees(glm::acos(
                  glm::dot(faceNormal_[here.face], faceNormal_[next.face])));
              if (dihedral > minSharpAngle ||
                  triIsFlatFace[here.face] != triIsFlatFace[next.face] ||
                  (triIsFlatFace[here.face] && triIsFlatFace[next.face] &&
                   !meshRelation_.triRef[here.face].SameFace(
                       meshRelation_.triRef[next.face]))) {
                normals.push_back(glm::vec3(0));
              }
              group.push_back(normals.size() - 1);
              float dot = glm::dot(here.edgeVec, next.edgeVec);
              const float phi =
                  dot >= 1 ? kTolerance
                           : (dot <= -1 ? glm::pi<float>() : glm::acos(dot));
              normals.back() += faceNormal_[next.face] * phi;
            });

        for (auto& normal : normals) {
          normal = glm::normalize(normal);
        }

        int lastGroup = 0;
        int lastProp = -1;
        int newProp = -1;
        int idx = 0;
        ForVert(endEdge, [&](int current1) {
          const int thisTri = current1 / 3;
          const int j = current1 - 3 * thisTri;
          const int prop = oldTriProp[thisTri][j];
          auto start = oldProperties.begin() + prop * oldNumProp;

          if (group[idx] != lastGroup && group[idx] != 0 && prop == lastProp) {
            lastGroup = group[idx];
            newProp = NumPropVert();
            meshRelation_.properties.resize(meshRelation_.properties.size() +
                                            numProp);
            std::copy(start, start + oldNumProp,
                      meshRelation_.properties.begin() + newProp * numProp);
            for (const int i : {0, 1, 2}) {
              meshRelation_.properties[newProp * numProp + normalIdx + i] =
                  normals[group[idx]][i];
            }
          } else if (prop != lastProp) {
            lastProp = prop;
            newProp = prop;
            std::copy(start, start + oldNumProp,
                      meshRelation_.properties.begin() + prop * numProp);
            for (const int i : {0, 1, 2})
              meshRelation_.properties[prop * numProp + normalIdx + i] =
                  normals[group[idx]][i];
          }

          meshRelation_.triProperties[thisTri][j] = newProp;
          ++idx;
        });
      }
    }
  }
}

/**
 * Calculates halfedgeTangent_, allowing the manifold to be refined and
 * smoothed. The tangents form weighted cubic Beziers along each edge. This
 * function creates circular arcs where possible (minimizing maximum curvature),
 * constrained to the indicated property normals. Across edges that form
 * discontinuities in the normals, the tangent vectors are zero-length, allowing
 * the shape to form a sharp corner with minimal oscillation.
 */
void Manifold::Impl::CreateTangents(int normalIdx) {
  ZoneScoped;
  const int numVert = NumVert();
  const int numHalfedge = halfedge_.size();
  halfedgeTangent_.resize(0);
  Vec<glm::vec4> tangent(numHalfedge);

  Vec<glm::vec3> vertNormal(numVert);
  Vec<glm::ivec2> vertSharpHalfedge(numVert, glm::ivec2(-1));
  for (int e = 0; e < numHalfedge; ++e) {
    const int vert = halfedge_[e].startVert;
    auto& sharpHalfedge = vertSharpHalfedge[vert];
    if (sharpHalfedge[0] >= 0 && sharpHalfedge[1] >= 0) continue;

    int idx = 0;
    // Only used when there is only one.
    glm::vec3& lastNormal = vertNormal[vert];

    ForVert<glm::vec3>(
        e,
        [normalIdx, this](int halfedge) {
          return GetNormal(halfedge, normalIdx);
        },
        [&sharpHalfedge, &idx, &lastNormal](int halfedge,
                                            const glm::vec3& normal,
                                            const glm::vec3& nextNormal) {
          const glm::vec3 diff = nextNormal - normal;
          if (glm::dot(diff, diff) > kTolerance * kTolerance) {
            if (idx > 1) {
              sharpHalfedge[0] = -1;
            } else {
              sharpHalfedge[idx++] = halfedge;
            }
          }
          lastNormal = normal;
        });
  }

  for_each_n(autoPolicy(numHalfedge),
             zip(tangent.begin(), halfedge_.cbegin(), countAt(0)), numHalfedge,
             SmoothBezier({this, vertNormal}));

  halfedgeTangent_.swap(tangent);

  for (int vert = 0; vert < numVert; ++vert) {
    const int first = vertSharpHalfedge[vert][0];
    const int second = vertSharpHalfedge[vert][1];
    if (second == -1) continue;
    if (first != -1) {  // Make continuous edge
      const glm::vec3 newTangent = glm::normalize(glm::cross(
          GetNormal(first, normalIdx), GetNormal(second, normalIdx)));
      if (!isfinite(newTangent[0])) continue;

      halfedgeTangent_[first] = CircularTangent(
          newTangent, vertPos_[halfedge_[first].endVert] - vertPos_[vert]);
      halfedgeTangent_[second] = CircularTangent(
          -newTangent, vertPos_[halfedge_[second].endVert] - vertPos_[vert]);

      ForVert(first, [this, first, second](int current) {
        if (current != first && current != second &&
            !IsMarkedInsideQuad(current)) {
          halfedgeTangent_[current] = glm::vec4(0);
        }
      });
    } else {  // Sharpen vertex uniformly
      ForVert(first, [this](int current) {
        if (!IsMarkedInsideQuad(current)) {
          halfedgeTangent_[current] = glm::vec4(0);
        }
      });
    }
  }
}

/**
 * Calculates halfedgeTangent_, allowing the manifold to be refined and
 * smoothed. The tangents form weighted cubic Beziers along each edge. This
 * function creates circular arcs where possible (minimizing maximum curvature),
 * constrained to the vertex normals. Where sharpenedEdges are specified, the
 * tangents are shortened that intersect the sharpened edge, concentrating the
 * curvature there, while the tangents of the sharp edges themselves are aligned
 * for continuity.
 */
void Manifold::Impl::CreateTangents(std::vector<Smoothness> sharpenedEdges) {
  ZoneScoped;
  const int numHalfedge = halfedge_.size();
  halfedgeTangent_.resize(0);
  Vec<glm::vec4> tangent(numHalfedge);

  Vec<bool> triIsFlatFace = FlatFaces();
  Vec<int> vertFlatFace = VertFlatFace(triIsFlatFace);
  Vec<glm::vec3> vertNormal = vertNormal_;
  for (int v = 0; v < NumVert(); ++v) {
    if (vertFlatFace[v] >= 0) {
      vertNormal[v] = faceNormal_[vertFlatFace[v]];
    }
  }

  for_each_n(autoPolicy(numHalfedge),
             zip(tangent.begin(), halfedge_.cbegin(), countAt(0)), numHalfedge,
             SmoothBezier({this, vertNormal}));

  halfedgeTangent_.swap(tangent);

  // Add sharpened edges around faces, just on the face side.
  for (int tri = 0; tri < NumTri(); ++tri) {
    if (!triIsFlatFace[tri]) continue;
    for (const int j : {0, 1, 2}) {
      const int tri2 = halfedge_[3 * tri + j].pairedHalfedge / 3;
      if (!triIsFlatFace[tri2] ||
          !meshRelation_.triRef[tri].SameFace(meshRelation_.triRef[tri2])) {
        sharpenedEdges.push_back({3 * tri + j, 0});
      }
    }
  }

  if (sharpenedEdges.empty()) return;

  using Pair = std::pair<Smoothness, Smoothness>;
  // Fill in missing pairs with default smoothness = 1.
  std::map<int, Pair> edges;
  for (Smoothness edge : sharpenedEdges) {
    if (edge.smoothness >= 1) continue;
    const bool forward = halfedge_[edge.halfedge].IsForward();
    const int pair = halfedge_[edge.halfedge].pairedHalfedge;
    const int idx = forward ? edge.halfedge : pair;
    if (edges.find(idx) == edges.end()) {
      edges[idx] = {edge, {pair, 1}};
      if (!forward) std::swap(edges[idx].first, edges[idx].second);
    } else {
      Smoothness& e = forward ? edges[idx].first : edges[idx].second;
      e.smoothness = glm::min(edge.smoothness, e.smoothness);
    }
  }

  std::map<int, std::vector<Pair>> vertTangents;
  for (const auto& value : edges) {
    const Pair edge = value.second;
    vertTangents[halfedge_[edge.first.halfedge].startVert].push_back(edge);
    vertTangents[halfedge_[edge.second.halfedge].startVert].push_back(
        {edge.second, edge.first});
  }

  for (const auto& value : vertTangents) {
    const std::vector<Pair>& vert = value.second;
    // Sharp edges that end are smooth at their terminal vert.
    if (vert.size() == 1) continue;
    if (vert.size() == 2) {  // Make continuous edge
      const int first = vert[0].first.halfedge;
      const int second = vert[1].first.halfedge;
      const glm::vec3 newTangent =
          glm::normalize(glm::vec3(halfedgeTangent_[first]) -
                         glm::vec3(halfedgeTangent_[second]));

      const glm::vec3 pos = vertPos_[halfedge_[first].startVert];
      halfedgeTangent_[first] =
          CircularTangent(newTangent, vertPos_[halfedge_[first].endVert] - pos);
      halfedgeTangent_[second] = CircularTangent(
          -newTangent, vertPos_[halfedge_[second].endVert] - pos);

      float smoothness =
          (vert[0].second.smoothness + vert[1].first.smoothness) / 2;
      ForVert(first, [this, &smoothness, &vert, first, second](int current) {
        if (current == second) {
          smoothness =
              (vert[1].second.smoothness + vert[0].first.smoothness) / 2;
        } else if (current != first && !IsMarkedInsideQuad(current)) {
          halfedgeTangent_[current] = smoothness * halfedgeTangent_[current];
        }
      });
    } else {  // Sharpen vertex uniformly
      float smoothness = 0;
      for (const Pair& pair : vert) {
        smoothness += pair.first.smoothness;
        smoothness += pair.second.smoothness;
      }
      smoothness /= 2 * vert.size();

      ForVert(vert[0].first.halfedge, [this, smoothness](int current) {
        if (!IsMarkedInsideQuad(current)) {
          halfedgeTangent_[current] = smoothness * halfedgeTangent_[current];
        }
      });
    }
  }
}

/**
 * Split each edge into n pieces as defined by calling the edgeDivisions
 * function, and sub-triangulate each triangle accordingly. This function
 * doesn't run Finish(), as that is expensive and it'll need to be run after
 * the new vertices have moved, which is a likely scenario after refinement
 * (smoothing).
 */
Vec<Barycentric> Manifold::Impl::Subdivide(
    std::function<int(glm::vec3)> edgeDivisions) {
  Vec<TmpEdge> edges = CreateTmpEdges(halfedge_);
  const int numVert = NumVert();
  const int numEdge = edges.size();
  const int numTri = NumTri();
  Vec<int> half2Edge(2 * numEdge);
  auto policy = autoPolicy(numEdge);
  for_each_n(policy, zip(countAt(0), edges.begin()), numEdge,
             ReindexHalfedge({half2Edge, halfedge_}));

  Vec<glm::ivec4> faceHalfedges(numTri);
  for_each_n(policy, zip(faceHalfedges.begin(), countAt(0)), numTri,
             [this](thrust::tuple<glm::ivec4&, int> inOut) {
               glm::ivec4& halfedges = thrust::get<0>(inOut);
               const int tri = thrust::get<1>(inOut);
               halfedges = GetHalfedges(tri);
             });

  Vec<int> edgeAdded(numEdge);
  for_each_n(policy, zip(edgeAdded.begin(), edges.cbegin()), numEdge,
             [edgeDivisions, this](thrust::tuple<int&, TmpEdge> inOut) {
               int& divisions = thrust::get<0>(inOut);
               const TmpEdge edge = thrust::get<1>(inOut);
               if (IsMarkedInsideQuad(edge.halfedgeIdx)) {
                 divisions = 0;
                 return;
               }
               const glm::vec3 vec =
                   vertPos_[edge.first] - vertPos_[edge.second];
               divisions = edgeDivisions(vec);
             });

  Vec<int> edgeOffset(numEdge);
  exclusive_scan(policy, edgeAdded.begin(), edgeAdded.end(), edgeOffset.begin(),
                 numVert);

  Vec<Barycentric> vertBary(edgeOffset.back() + edgeAdded.back());
  const int totalEdgeAdded = vertBary.size() - numVert;
  FillRetainedVerts(vertBary);
  for_each_n(policy, zip(edges.begin(), edgeAdded.begin(), edgeOffset.begin()),
             numEdge, [this, &vertBary](thrust::tuple<TmpEdge, int, int> in) {
               const TmpEdge edge = thrust::get<0>(in);
               const int n = thrust::get<1>(in);
               const int offset = thrust::get<2>(in);

               const BaryIndices indices = GetIndices(edge.halfedgeIdx);
               if (indices.tri < 0) {
                 return;  // inside quad
               }
               const float frac = 1.0f / (n + 1);

               for (int i = 0; i < n; ++i) {
                 glm::vec4 uvw(0);
                 uvw[indices.end4] = (i + 1) * frac;
                 uvw[indices.start4] = 1 - uvw[indices.end4];
                 vertBary[offset + i].uvw = uvw;
                 vertBary[offset + i].tri = indices.tri;
               }
             });

  std::vector<Partition> subTris(numTri);
  for_each_n(policy, countAt(0), numTri,
             [this, &subTris, &half2Edge, &edgeAdded, &faceHalfedges](int tri) {
               const glm::ivec4 halfedges = faceHalfedges[tri];
               glm::ivec4 divisions(0);
               for (const int i : {0, 1, 2, 3}) {
                 if (halfedges[i] >= 0) {
                   divisions[i] = edgeAdded[half2Edge[halfedges[i]]] + 1;
                 }
               }
               subTris[tri] = Partition::GetPartition(divisions);
             });

  Vec<int> triOffset(numTri);
  auto numSubTris = thrust::make_transform_iterator(
      subTris.begin(),
      [](const Partition& part) { return part.triVert.size(); });
  exclusive_scan(policy, numSubTris, numSubTris + numTri, triOffset.begin(), 0);

  Vec<int> interiorOffset(numTri);
  auto numInterior = thrust::make_transform_iterator(
      subTris.begin(),
      [](const Partition& part) { return part.NumInterior(); });
  exclusive_scan(policy, numInterior, numInterior + numTri,
                 interiorOffset.begin(), vertBary.size());

  Vec<glm::ivec3> triVerts(triOffset.back() + subTris.back().triVert.size());
  vertBary.resize(interiorOffset.back() + subTris.back().NumInterior());
  Vec<TriRef> triRef(triVerts.size());
  for_each_n(
      policy, countAt(0), numTri,
      [this, &triVerts, &triRef, &vertBary, &subTris, &edgeOffset, &half2Edge,
       &triOffset, &interiorOffset, &faceHalfedges](int tri) {
        const glm::ivec4 halfedges = faceHalfedges[tri];
        if (halfedges[0] < 0) return;
        glm::ivec4 tri3;
        glm::ivec4 edgeOffsets;
        glm::bvec4 edgeFwd;
        for (const int i : {0, 1, 2, 3}) {
          if (halfedges[i] < 0) {
            tri3[i] = -1;
            continue;
          }
          const Halfedge& halfedge = halfedge_[halfedges[i]];
          tri3[i] = halfedge.startVert;
          edgeOffsets[i] = edgeOffset[half2Edge[halfedges[i]]];
          edgeFwd[i] = halfedge.IsForward();
        }

        Vec<glm::ivec3> newTris = subTris[tri].Reindex(
            tri3, edgeOffsets, edgeFwd, interiorOffset[tri]);
        copy(ExecutionPolicy::Seq, newTris.begin(), newTris.end(),
             triVerts.begin() + triOffset[tri]);
        auto start = triRef.begin() + triOffset[tri];
        fill(ExecutionPolicy::Seq, start, start + newTris.size(),
             meshRelation_.triRef[tri]);

        const glm::ivec4 idx = subTris[tri].idx;
        const glm::ivec4 vIdx =
            halfedges[3] >= 0 || idx[1] == Next3(idx[0])
                ? idx
                : glm::ivec4(idx[2], idx[0], idx[1], idx[3]);
        glm::ivec4 rIdx;
        for (const int i : {0, 1, 2, 3}) {
          rIdx[vIdx[i]] = i;
        }

        const auto& subBary = subTris[tri].vertBary;
        transform(ExecutionPolicy::Seq,
                  subBary.begin() + subTris[tri].InteriorOffset(),
                  subBary.end(), vertBary.begin() + interiorOffset[tri],
                  [tri, rIdx](glm::vec4 bary) {
                    return Barycentric({tri,
                                        {bary[rIdx[0]], bary[rIdx[1]],
                                         bary[rIdx[2]], bary[rIdx[3]]}});
                  });
      });
  meshRelation_.triRef = triRef;

  Vec<glm::vec3> newVertPos(vertBary.size());
  for_each_n(
      policy, zip(newVertPos.begin(), vertBary.begin()), vertBary.size(),
      [this, &faceHalfedges](thrust::tuple<glm::vec3&, Barycentric> inOut) {
        glm::vec3& pos = thrust::get<0>(inOut);
        const Barycentric bary = thrust::get<1>(inOut);

        const glm::ivec4 halfedges = faceHalfedges[bary.tri];
        if (halfedges[3] < 0) {
          glm::mat3 triPos;
          for (const int i : {0, 1, 2}) {
            triPos[i] = vertPos_[halfedge_[halfedges[i]].startVert];
          }
          pos = triPos * glm::vec3(bary.uvw);
        } else {
          glm::mat4x3 quadPos;
          for (const int i : {0, 1, 2, 3}) {
            quadPos[i] = vertPos_[halfedge_[halfedges[i]].startVert];
          }
          pos = quadPos * bary.uvw;
        }
      });
  vertPos_ = newVertPos;

  faceNormal_.resize(0);

  if (meshRelation_.numProp > 0) {
    const int numPropVert = NumPropVert();
    const int addedVerts = NumVert() - numVert;
    const int propOffset = numPropVert - numVert;
    Vec<float> prop(meshRelation_.numProp *
                    (numPropVert + addedVerts + totalEdgeAdded));

    // copy retained prop verts
    copy(policy, meshRelation_.properties.begin(),
         meshRelation_.properties.end(), prop.begin());

    // copy interior prop verts and forward edge prop verts
    for_each_n(
        policy, zip(countAt(numPropVert), vertBary.begin() + numVert),
        addedVerts,
        [this, &prop, &faceHalfedges](thrust::tuple<int, Barycentric> in) {
          const int vert = thrust::get<0>(in);
          const Barycentric bary = thrust::get<1>(in);
          const glm::ivec4 halfedges = faceHalfedges[bary.tri];
          auto& rel = meshRelation_;

          for (int p = 0; p < rel.numProp; ++p) {
            if (halfedges[3] < 0) {
              glm::vec3 triProp;
              for (const int i : {0, 1, 2}) {
                triProp[i] = rel.properties[rel.triProperties[bary.tri][i] *
                                                rel.numProp +
                                            p];
              }
              prop[vert * rel.numProp + p] =
                  glm::dot(triProp, glm::vec3(bary.uvw));
            } else {
              glm::vec4 quadProp;
              for (const int i : {0, 1, 2, 3}) {
                const int tri = halfedges[i] / 3;
                const int j = halfedges[i] % 3;
                quadProp[i] =
                    rel.properties[rel.triProperties[tri][j] * rel.numProp + p];
              }
              prop[vert * rel.numProp + p] = glm::dot(quadProp, bary.uvw);
            }
          }
        });

    // copy backward edge prop verts
    for_each_n(
        policy, zip(edges.begin(), edgeAdded.begin(), edgeOffset.begin()),
        numEdge,
        [this, &prop, propOffset,
         addedVerts](thrust::tuple<TmpEdge, int, int> in) {
          const TmpEdge edge = thrust::get<0>(in);
          const int n = thrust::get<1>(in);
          const int offset = thrust::get<2>(in) + propOffset + addedVerts;
          auto& rel = meshRelation_;

          const float frac = 1.0f / (n + 1);
          const int halfedgeIdx = halfedge_[edge.halfedgeIdx].pairedHalfedge;
          const int v0 = halfedgeIdx % 3;
          const int tri = halfedgeIdx / 3;
          const int prop0 = rel.triProperties[tri][v0];
          const int prop1 = rel.triProperties[tri][Next3(v0)];
          for (int i = 0; i < n; ++i) {
            for (int p = 0; p < rel.numProp; ++p) {
              prop[(offset + i) * rel.numProp + p] = glm::mix(
                  rel.properties[prop0 * rel.numProp + p],
                  rel.properties[prop1 * rel.numProp + p], (i + 1) * frac);
            }
          }
        });

    Vec<glm::ivec3> triProp(triVerts.size());
    for_each_n(policy, countAt(0), numTri,
               [this, &triProp, &subTris, &edgeOffset, &half2Edge, &triOffset,
                &interiorOffset, &faceHalfedges, propOffset,
                addedVerts](const int tri) {
                 const glm::ivec4 halfedges = faceHalfedges[tri];
                 if (halfedges[0] < 0) return;

                 auto& rel = meshRelation_;
                 glm::ivec4 tri3;
                 glm::ivec4 edgeOffsets;
                 glm::bvec4 edgeFwd(true);
                 for (const int i : {0, 1, 2, 3}) {
                   if (halfedges[i] < 0) {
                     tri3[i] = -1;
                     continue;
                   }
                   const int thisTri = halfedges[i] / 3;
                   const int j = halfedges[i] % 3;
                   const Halfedge& halfedge = halfedge_[halfedges[i]];
                   tri3[i] = rel.triProperties[thisTri][j];
                   edgeOffsets[i] = edgeOffset[half2Edge[halfedges[i]]];
                   if (!halfedge.IsForward()) {
                     const int pairTri = halfedge.pairedHalfedge / 3;
                     const int k = halfedge.pairedHalfedge % 3;
                     if (rel.triProperties[pairTri][k] !=
                             rel.triProperties[thisTri][Next3(j)] ||
                         rel.triProperties[pairTri][Next3(k)] !=
                             rel.triProperties[thisTri][j]) {
                       edgeOffsets[i] += addedVerts;
                     } else {
                       edgeFwd[i] = false;
                     }
                   }
                 }

                 Vec<glm::ivec3> newTris = subTris[tri].Reindex(
                     tri3, edgeOffsets + propOffset, edgeFwd,
                     interiorOffset[tri] + propOffset);
                 copy(ExecutionPolicy::Seq, newTris.begin(), newTris.end(),
                      triProp.begin() + triOffset[tri]);
               });

    meshRelation_.properties = prop;
    meshRelation_.triProperties = triProp;
  }

  CreateHalfedges(triVerts);

  return vertBary;
}

void Manifold::Impl::Refine(std::function<int(glm::vec3)> edgeDivisions) {
  if (IsEmpty()) return;
  Manifold::Impl old = *this;
  Vec<Barycentric> vertBary = Subdivide(edgeDivisions);
  if (vertBary.size() == 0) return;

  if (old.halfedgeTangent_.size() == old.halfedge_.size()) {
    for_each_n(autoPolicy(NumTri()), zip(vertPos_.begin(), vertBary.begin()),
               NumVert(), InterpTri({&old}));
    // Make original since the subdivided faces have been warped into
    // being non-coplanar, and hence not being related to the original faces.
    meshRelation_.originalID = ReserveIDs(1);
    InitializeOriginal();
  }

  halfedgeTangent_.resize(0);
  Finish();
}

}  // namespace manifold
