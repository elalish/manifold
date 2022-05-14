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

#include <map>

#include "impl.h"
#include "polygon.h"

namespace manifold {

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
void Manifold::Impl::Face2Tri(const VecDH<int>& faceEdge,
                              const VecDH<BaryRef>& faceRef,
                              const VecDH<int>& halfedgeBary) {
  VecDH<glm::ivec3> triVerts;
  VecDH<glm::vec3> triNormal;
  VecDH<BaryRef> &triBary = meshRelation_.triBary;
  triBary.resize(0);
  triVerts.reserve(faceEdge.size());
  triNormal.reserve(faceEdge.size());
  triBary.reserve(faceEdge.size()*3);

  const VecDH<glm::vec3>& vertPos = vertPos_;
  const VecDH<int>& faceEdgeH = faceEdge;
  const VecDH<Halfedge>& halfedge = halfedge_;
  const VecDH<glm::vec3>& faceNormal = faceNormal_;
  // meshRelation_.triBary.resize(0);
  // std::vector<glm::ivec3> triVerts;
  // std::vector<glm::vec3> triNormal;
  // std::vector<BaryRef> triBary;

  for (int face = 0; face < faceEdgeH.size() - 1; ++face) {
    const int firstEdge = faceEdgeH[face];
    const int lastEdge = faceEdgeH[face + 1];
    const int numEdge = lastEdge - firstEdge;
    ALWAYS_ASSERT(numEdge >= 3, topologyErr, "face has less than three edges.");
    const glm::vec3 normal = faceNormal[face];

    auto linearSearch = [](const int* mapping, int value) {
      int i = 0;
      while (mapping[i] != value)
        ++i;
      return i;
    };

    if (numEdge == 3) {  // Single triangle
      int mapping[3] = {halfedge[firstEdge].startVert,
                        halfedge[firstEdge + 1].startVert,
                        halfedge[firstEdge + 2].startVert};
      glm::ivec3 tri(halfedge[firstEdge].startVert,
                     halfedge[firstEdge + 1].startVert,
                     halfedge[firstEdge + 2].startVert);
      glm::ivec3 ends(halfedge[firstEdge].endVert,
                      halfedge[firstEdge + 1].endVert,
                      halfedge[firstEdge + 2].endVert);
      if (ends[0] == tri[2]) {
        std::swap(tri[1], tri[2]);
        std::swap(ends[1], ends[2]);
      }
      ALWAYS_ASSERT(ends[0] == tri[1] && ends[1] == tri[2] && ends[2] == tri[0],
                    topologyErr, "These 3 edges do not form a triangle!");

      triVerts.push_back(tri);
      triNormal.push_back(normal);
      triBary.push_back(faceRef[face]);
      for (int k : {0, 1, 2}) {
        int index = linearSearch(mapping, tri[k]);
        triBary.back().vertBary[k] = halfedgeBary[firstEdge + index];
      }
    } else if (numEdge == 4) {  // Pair of triangles
      int mapping[4] = {halfedge[firstEdge].startVert,
                        halfedge[firstEdge + 1].startVert,
                        halfedge[firstEdge + 2].startVert,
                        halfedge[firstEdge + 3].startVert};
      const glm::mat3x2 projection = GetAxisAlignedProjection(normal);
      auto triCCW = [&projection, &vertPos, this](const glm::ivec3 tri) {
        return CCW(projection * vertPos[tri[0]], projection * vertPos[tri[1]],
                   projection * vertPos[tri[2]], precision_) >= 0;
      };

      glm::ivec3 tri0(halfedge[firstEdge].startVert,
                      halfedge[firstEdge].endVert, -1);
      glm::ivec3 tri1(-1, -1, tri0[0]);
      for (const int i : {1, 2, 3}) {
        if (halfedge[firstEdge + i].startVert == tri0[1]) {
          tri0[2] = halfedge[firstEdge + i].endVert;
          tri1[0] = tri0[2];
        }
        if (halfedge[firstEdge + i].endVert == tri0[0]) {
          tri1[1] = halfedge[firstEdge + i].startVert;
        }
      }
      ALWAYS_ASSERT(glm::all(glm::greaterThanEqual(tri0, glm::ivec3(0))) &&
                        glm::all(glm::greaterThanEqual(tri1, glm::ivec3(0))),
                    topologyErr, "non-manifold quad!");
      bool firstValid = triCCW(tri0) && triCCW(tri1);
      tri0[2] = tri1[1];
      tri1[2] = tri0[1];
      bool secondValid = triCCW(tri0) && triCCW(tri1);

      if (!secondValid) {
        tri0[2] = tri1[0];
        tri1[2] = tri0[0];
      } else if (firstValid) {
        glm::vec3 firstCross = vertPos[tri0[0]] - vertPos[tri1[0]];
        glm::vec3 secondCross = vertPos[tri0[1]] - vertPos[tri1[1]];
        if (glm::dot(firstCross, firstCross) <
            glm::dot(secondCross, secondCross)) {
          tri0[2] = tri1[0];
          tri1[2] = tri0[0];
        }
      }

      for (auto tri : { tri0, tri1 }) {
        triVerts.push_back(tri);
        triNormal.push_back(normal);
        triBary.push_back(faceRef[face]);
        for (int k : {0, 1, 2}) {
          int index = linearSearch(mapping, tri[k]);
          triBary.back().vertBary[k] = halfedgeBary[firstEdge + index];
        }
      }
    } else {  // General triangulation
      const glm::mat3x2 projection = GetAxisAlignedProjection(normal);

      std::map<int, int> vertBary;
      for (int j = firstEdge; j < lastEdge; ++j)
        vertBary[halfedge[j].startVert] = halfedgeBary[j];

      Polygons polys;
      try {
        polys = Face2Polygons(face, projection, faceEdgeH);
      } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        for (int edge = faceEdgeH[face]; edge < faceEdgeH[face + 1]; ++edge)
          std::cout << "halfedge: " << edge << ", " << halfedge[edge]
                    << std::endl;
        throw;
      }

      std::vector<glm::ivec3> newTris = Triangulate(polys, precision_);

      for (auto tri : newTris) {
        triVerts.push_back(tri);
        triNormal.push_back(normal);
        triBary.push_back(faceRef[face]);
        for (int k : {0, 1, 2}) {
          triBary.back().vertBary[k] =
              vertBary[tri[k]];
        }
      }
    }
  }
  faceNormal_ = std::move(triNormal);
  CreateAndFixHalfedges(triVerts);
}

/**
 * For the input face index, return a set of 2D polygons formed by the input
 * projection of the vertices.
 */
Polygons Manifold::Impl::Face2Polygons(int face, glm::mat3x2 projection,
                                       const VecDH<int>& faceEdge) const {
  const VecDH<glm::vec3>& vertPos = vertPos_;
  const VecDH<Halfedge>& halfedge = halfedge_;
  const int firstEdge = faceEdge[face];
  const int lastEdge = faceEdge[face + 1];

  std::map<int, int> vert_edge;
  for (int edge = firstEdge; edge < lastEdge; ++edge) {
    ALWAYS_ASSERT(
        vert_edge.emplace(std::make_pair(halfedge[edge].startVert, edge))
            .second,
        topologyErr, "face has duplicate vertices.");
  }

  Polygons polys;
  int startEdge = 0;
  int thisEdge = startEdge;
  while (1) {
    if (thisEdge == startEdge) {
      if (vert_edge.empty()) break;
      startEdge = vert_edge.begin()->second;
      thisEdge = startEdge;
      polys.push_back({});
    }
    int vert = halfedge[thisEdge].startVert;
    polys.back().push_back({projection * vertPos[vert], vert});
    const auto result = vert_edge.find(halfedge[thisEdge].endVert);
    ALWAYS_ASSERT(result != vert_edge.end(), topologyErr, "nonmanifold edge");
    thisEdge = result->second;
    vert_edge.erase(result);
  }
  return polys;
}
}  // namespace manifold
