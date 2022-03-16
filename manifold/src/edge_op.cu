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

#include "impl.cuh"

namespace {
using namespace manifold;

__host__ __device__ glm::ivec3 TriOf(int edge) {
  glm::ivec3 triEdge;
  triEdge[0] = edge;
  triEdge[1] = NextHalfedge(triEdge[0]);
  triEdge[2] = NextHalfedge(triEdge[1]);
  return triEdge;
}

__host__ __device__ bool Is01Longest(glm::vec2 v0, glm::vec2 v1, glm::vec2 v2) {
  const glm::vec2 e[3] = {v1 - v0, v2 - v1, v0 - v2};
  float l[3];
  for (int i : {0, 1, 2}) l[i] = glm::dot(e[i], e[i]);
  return l[0] > l[1] && l[0] > l[2];
}

struct ShortEdge {
  const Halfedge* halfedge;
  const glm::vec3* vertPos;
  const float precision;

  __host__ __device__ bool operator()(int edge) {
    if (halfedge[edge].pairedHalfedge < 0) return false;
    // Flag short edges
    const glm::vec3 delta =
        vertPos[halfedge[edge].endVert] - vertPos[halfedge[edge].startVert];
    return glm::dot(delta, delta) < precision * precision;
  }
};

struct FlagEdge {
  const Halfedge* halfedge;
  const BaryRef* triBary;

  __host__ __device__ bool operator()(int edge) {
    if (halfedge[edge].pairedHalfedge < 0) return false;
    // Flag redundant edges - those where the startVert is surrounded by only
    // two original triangles.
    const BaryRef ref0 = triBary[edge / 3];
    int current = NextHalfedge(halfedge[edge].pairedHalfedge);
    const BaryRef ref1 = triBary[current / 3];
    while (current != edge) {
      current = NextHalfedge(halfedge[current].pairedHalfedge);
      int tri = current / 3;
      const BaryRef ref = triBary[tri];
      if ((ref.meshID != ref0.meshID || ref.tri != ref0.tri) &&
          (ref.meshID != ref1.meshID || ref.tri != ref1.tri))
        return false;
    }
    return true;
  }
};

struct SwappableEdge {
  const Halfedge* halfedge;
  const glm::vec3* vertPos;
  const glm::vec3* triNormal;
  const float precision;

  __host__ __device__ bool operator()(int edge) {
    if (halfedge[edge].pairedHalfedge < 0) return false;

    int tri = halfedge[edge].face;
    glm::ivec3 triedge = TriOf(edge);
    glm::mat3x2 projection = GetAxisAlignedProjection(triNormal[tri]);
    glm::vec2 v[3];
    for (int i : {0, 1, 2})
      v[i] = projection * vertPos[halfedge[triedge[i]].startVert];
    // if (CCW(v[0], v[1], v[2], precision) < 0) printf("tri %d is CW!\n", tri);
    if (CCW(v[0], v[1], v[2], precision) > 0 || !Is01Longest(v[0], v[1], v[2]))
      return false;

    // Switch to neighbor's projection.
    edge = halfedge[edge].pairedHalfedge;
    tri = halfedge[edge].face;
    triedge = TriOf(edge);
    projection = GetAxisAlignedProjection(triNormal[tri]);
    for (int i : {0, 1, 2})
      v[i] = projection * vertPos[halfedge[triedge[i]].startVert];
    return CCW(v[0], v[1], v[2], precision) > 0 ||
           Is01Longest(v[0], v[1], v[2]);
  }
};
}  // namespace

namespace manifold {

/**
 * Collapses degenerate triangles by removing edges shorter than precision_ and
 * any edge that is preceeded by an edge that joins the same two face relations.
 * It also performs edge swaps on the long edges of degenerate triangles, though
 * there are some configurations of degenerates that cannot be removed this way.
 *
 * Note when an edge collapse would result in something non-manifold, the
 * vertices are duplicated in such a way as to remove handles or separate
 * meshes, thus decreasing the Genus(). It only increases when meshes that have
 * collapsed to just a pair of triangles are removed entirely.
 *
 * Rather than actually removing the edges, this step merely marks them for
 * removal, by setting vertPos to NaN and halfedge to {-1, -1, -1, -1}.
 */
void Manifold::Impl::SimplifyTopology() {
  VecDH<int> flaggedEdges(halfedge_.size());
  int numFlagged =
      thrust::copy_if(
          countAt(0), countAt(halfedge_.size()), flaggedEdges.beginD(),
          ShortEdge({halfedge_.cptrD(), vertPos_.cptrD(), precision_})) -
      flaggedEdges.beginD();
  flaggedEdges.resize(numFlagged);

  for (const int edge : flaggedEdges.H()) CollapseEdge(edge);

  flaggedEdges.resize(halfedge_.size());
  numFlagged =
      thrust::copy_if(
          countAt(0), countAt(halfedge_.size()), flaggedEdges.beginD(),
          FlagEdge({halfedge_.cptrD(), meshRelation_.triBary.cptrD()})) -
      flaggedEdges.beginD();
  flaggedEdges.resize(numFlagged);

  for (const int edge : flaggedEdges.H()) CollapseEdge(edge);

  flaggedEdges.resize(halfedge_.size());
  numFlagged = thrust::copy_if(
                   countAt(0), countAt(halfedge_.size()), flaggedEdges.beginD(),
                   SwappableEdge({halfedge_.cptrD(), vertPos_.cptrD(),
                                  faceNormal_.cptrD(), precision_})) -
               flaggedEdges.beginD();
  flaggedEdges.resize(numFlagged);

  for (const int edge : flaggedEdges.H()) {
    RecursiveEdgeSwap(edge);
  }
}

void Manifold::Impl::PairUp(int edge0, int edge1) {
  VecH<Halfedge>& halfedge = halfedge_.H();
  halfedge[edge0].pairedHalfedge = edge1;
  halfedge[edge1].pairedHalfedge = edge0;
}

// Traverses CW around startEdge.endVert from startEdge to endEdge
// (edgeEdge.endVert must == startEdge.endVert), updating each edge to point
// to vert instead.
void Manifold::Impl::UpdateVert(int vert, int startEdge, int endEdge) {
  VecH<Halfedge>& halfedge = halfedge_.H();
  while (startEdge != endEdge) {
    halfedge[startEdge].endVert = vert;
    startEdge = NextHalfedge(startEdge);
    halfedge[startEdge].startVert = vert;
    startEdge = halfedge[startEdge].pairedHalfedge;
  }
}

// In the event that the edge collapse would create a non-manifold edge,
// instead we duplicate the two verts and attach the manifolds the other way
// across this edge.
void Manifold::Impl::FormLoop(int current, int end) {
  VecH<Halfedge>& halfedge = halfedge_.H();
  VecH<glm::vec3>& vertPos = vertPos_.H();

  int startVert = vertPos.size();
  vertPos.push_back(vertPos[halfedge[current].startVert]);
  int endVert = vertPos.size();
  vertPos.push_back(vertPos[halfedge[current].endVert]);

  int oldMatch = halfedge[current].pairedHalfedge;
  int newMatch = halfedge[end].pairedHalfedge;

  UpdateVert(startVert, oldMatch, newMatch);
  UpdateVert(endVert, end, current);

  halfedge[current].pairedHalfedge = newMatch;
  halfedge[newMatch].pairedHalfedge = current;
  halfedge[end].pairedHalfedge = oldMatch;
  halfedge[oldMatch].pairedHalfedge = end;

  RemoveIfFolded(end);
}

void Manifold::Impl::CollapseTri(const glm::ivec3& triEdge) {
  VecH<Halfedge>& halfedge = halfedge_.H();
  int pair1 = halfedge[triEdge[1]].pairedHalfedge;
  int pair2 = halfedge[triEdge[2]].pairedHalfedge;
  halfedge[pair1].pairedHalfedge = pair2;
  halfedge[pair2].pairedHalfedge = pair1;
  for (int i : {0, 1, 2}) {
    halfedge[triEdge[i]] = {-1, -1, -1, -1};
  }
}

void Manifold::Impl::RemoveIfFolded(int edge) {
  VecH<Halfedge>& halfedge = halfedge_.H();
  VecH<glm::vec3>& vertPos = vertPos_.H();
  const glm::ivec3 tri0edge = TriOf(edge);
  const glm::ivec3 tri1edge = TriOf(halfedge[edge].pairedHalfedge);
  if (halfedge[tri0edge[1]].endVert == halfedge[tri1edge[1]].endVert) {
    for (int i : {0, 1, 2}) {
      vertPos[halfedge[tri0edge[i]].startVert] = glm::vec3(NAN);
      halfedge[tri0edge[i]] = {-1, -1, -1, -1};
      halfedge[tri1edge[i]] = {-1, -1, -1, -1};
    }
  }
}

void Manifold::Impl::CollapseEdge(const int edge) {
  VecH<Halfedge>& halfedge = halfedge_.H();
  VecH<glm::vec3>& vertPos = vertPos_.H();
  VecH<glm::vec3>& triNormal = faceNormal_.H();
  VecH<BaryRef>& triBary = meshRelation_.triBary.H();

  const Halfedge toRemove = halfedge[edge];
  if (toRemove.pairedHalfedge < 0) return;

  const int endVert = toRemove.endVert;
  const glm::ivec3 tri0edge = TriOf(edge);
  const glm::ivec3 tri1edge = TriOf(toRemove.pairedHalfedge);

  const glm::vec3 pNew = vertPos[endVert];
  const glm::vec3 pOld = vertPos[toRemove.startVert];
  const glm::vec3 delta = pNew - pOld;
  const bool shortEdge = glm::dot(delta, delta) < precision_ * precision_;

  std::vector<int> edges;
  // Orbit endVert
  int current = halfedge[tri0edge[1]].pairedHalfedge;
  while (current != tri1edge[2]) {
    current = NextHalfedge(current);
    edges.push_back(current);
    current = halfedge[current].pairedHalfedge;
  }

  // Orbit startVert
  int start = halfedge[tri1edge[1]].pairedHalfedge;
  const BaryRef ref0 = triBary[edge / 3];
  const BaryRef ref1 = triBary[toRemove.pairedHalfedge / 3];
  if (!shortEdge) {
    current = start;
    glm::vec3 pLast = vertPos[halfedge[tri1edge[1]].endVert];
    while (current != tri0edge[2]) {
      current = NextHalfedge(current);
      glm::vec3 pNext = vertPos[halfedge[current].endVert];
      const int tri = current / 3;
      const BaryRef ref = triBary[tri];
      // Don't collapse if the edge is not redundant (this may have changed due
      // to the collapse of neighbors).
      if ((ref.meshID != ref0.meshID || ref.tri != ref0.tri) &&
          (ref.meshID != ref1.meshID || ref.tri != ref1.tri))
        return;

      // Don't collapse edge if it would cause a triangle to invert.
      const glm::mat3x2 projection = GetAxisAlignedProjection(triNormal[tri]);
      if (CCW(projection * pNext, projection * pLast, projection * pNew,
              precision_) < 0)
        return;

      pLast = pNext;
      current = halfedge[current].pairedHalfedge;
    }
  }

  // Remove toRemove.startVert and replace with endVert.
  vertPos[toRemove.startVert] = glm::vec3(NAN);
  CollapseTri(tri1edge);

  // Orbit startVert
  current = start;
  while (current != tri0edge[2]) {
    current = NextHalfedge(current);

    if (!shortEdge) {
      // Update the shifted triangles to the vertBary of endVert
      const int tri = current / 3;
      const int vIdx = current - 3 * tri;
      triBary[tri].vertBary[vIdx] =
          (ref0.meshID == triBary[tri].meshID && ref0.tri == triBary[tri].tri)
              ? ref0.vertBary[(edge + 1) % 3]
              : ref1.vertBary[toRemove.pairedHalfedge % 3];
    }

    const int vert = halfedge[current].endVert;
    const int next = halfedge[current].pairedHalfedge;
    for (int i = 0; i < edges.size(); ++i) {
      if (vert == halfedge[edges[i]].endVert) {
        FormLoop(edges[i], current);
        start = next;
        edges.resize(i);
        break;
      }
    }
    current = next;
  }

  UpdateVert(endVert, start, tri0edge[2]);
  CollapseTri(tri0edge);
  RemoveIfFolded(start);
}

void Manifold::Impl::RecursiveEdgeSwap(const int edge) {
  const VecH<glm::vec3>& vertPos = vertPos_.H();
  VecH<Halfedge>& halfedge = halfedge_.H();
  VecH<glm::vec3>& triNormal = faceNormal_.H();
  VecH<BaryRef>& triBary = meshRelation_.triBary.H();

  if (halfedge[edge].pairedHalfedge < 0) return;

  const int pair = halfedge[edge].pairedHalfedge;
  const glm::ivec3 tri0edge = TriOf(edge);
  const glm::ivec3 tri1edge = TriOf(pair);
  const glm::ivec3 perm0 = TriOf(edge % 3);
  const glm::ivec3 perm1 = TriOf(pair % 3);

  glm::mat3x2 projection = GetAxisAlignedProjection(triNormal[edge / 3]);
  glm::vec2 v[4];
  for (int i : {0, 1, 2})
    v[i] = projection * vertPos[halfedge[tri0edge[i]].startVert];
  // Only operate on the long edge of a degenerate triangle.
  if (CCW(v[0], v[1], v[2], precision_) > 0 || !Is01Longest(v[0], v[1], v[2]))
    return;

  // Switch to neighbor's projection.
  projection = GetAxisAlignedProjection(triNormal[halfedge[pair].face]);
  for (int i : {0, 1, 2})
    v[i] = projection * vertPos[halfedge[tri0edge[i]].startVert];
  v[3] = projection * vertPos[halfedge[tri1edge[2]].startVert];

  auto SwapEdge = [&]() {
    // The 0-verts are swapped to the opposite 2-verts.
    const int v0 = halfedge[tri0edge[2]].startVert;
    const int v1 = halfedge[tri1edge[2]].startVert;
    halfedge[tri0edge[0]].startVert = v1;
    halfedge[tri0edge[2]].endVert = v1;
    halfedge[tri1edge[0]].startVert = v0;
    halfedge[tri1edge[2]].endVert = v0;
    PairUp(tri0edge[0], halfedge[tri1edge[2]].pairedHalfedge);
    PairUp(tri1edge[0], halfedge[tri0edge[2]].pairedHalfedge);
    PairUp(tri0edge[2], tri1edge[2]);
    // Both triangles are now subsets of the neighboring triangle.
    const int tri0 = halfedge[tri0edge[0]].face;
    const int tri1 = halfedge[tri1edge[0]].face;
    triNormal[tri0] = triNormal[tri1];
    triBary[tri0] = triBary[tri1];
    triBary[tri0].vertBary[perm0[1]] = triBary[tri1].vertBary[perm1[0]];
    triBary[tri0].vertBary[perm0[0]] = triBary[tri1].vertBary[perm1[2]];
    // Calculate a new barycentric coordinate for the split triangle.
    const glm::vec3 uvw0 = UVW(triBary[tri1].vertBary[perm1[0]],
                               meshRelation_.barycentric.cptrH());
    const glm::vec3 uvw1 = UVW(triBary[tri1].vertBary[perm1[1]],
                               meshRelation_.barycentric.cptrH());
    const float l01 = glm::length(v[1] - v[0]);
    const float l02 = glm::length(v[2] - v[0]);
    const float a = glm::max(0.0f, glm::min(1.0f, l02 / l01));
    const glm::vec3 uvw2 = a * uvw0 + (1 - a) * uvw1;
    // And assign it.
    const int newBary = meshRelation_.barycentric.size();
    meshRelation_.barycentric.H().push_back(uvw2);
    triBary[tri1].vertBary[perm1[0]] = newBary;
    triBary[tri0].vertBary[perm0[2]] = newBary;

    // if the new edge already exists, duplicate the verts and split the mesh.
    int current = halfedge[tri1edge[0]].pairedHalfedge;
    const int endVert = halfedge[tri1edge[1]].endVert;
    while (current != tri0edge[1]) {
      current = NextHalfedge(current);
      if (halfedge[current].endVert == endVert) {
        FormLoop(tri0edge[2], current);
        RemoveIfFolded(tri0edge[2]);
        return;
      }
      current = halfedge[current].pairedHalfedge;
    }
  };

  // Only operate if the other triangles are not degenerate.
  if (CCW(v[1], v[0], v[3], precision_) <= 0) {
    if (!Is01Longest(v[1], v[0], v[3])) return;
    // Two facing, long-edge degenerates can swap.
    SwapEdge();
    const glm::vec2 e23 = v[3] - v[2];
    if (glm::dot(e23, e23) < precision_ * precision_) {
      CollapseEdge(tri0edge[2]);
    } else {
      RecursiveEdgeSwap(tri0edge[0]);
      RecursiveEdgeSwap(tri0edge[1]);
      RecursiveEdgeSwap(tri1edge[0]);
      RecursiveEdgeSwap(tri1edge[1]);
    }
    return;
  } else if (CCW(v[0], v[3], v[2], precision_) <= 0 ||
             CCW(v[1], v[2], v[3], precision_) <= 0) {
    return;
  }
  // Normal path
  SwapEdge();
  RecursiveEdgeSwap(halfedge[tri0edge[1]].pairedHalfedge);
  RecursiveEdgeSwap(halfedge[tri1edge[0]].pairedHalfedge);
}
}  // namespace manifold