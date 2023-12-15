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

namespace {
using namespace manifold;

glm::dvec3 OrthogonalTo(glm::dvec3 in, glm::dvec3 ref) {
  in -= glm::dot(in, ref) * ref;
  return in;
}

/**
 * The total number of verts if a triangle is subdivided naturally such that
 * each edge has edgeVerts verts along it (edgeVerts >= -1).
 */
int VertsPerTri(int edgeVerts) {
  return (edgeVerts * edgeVerts + edgeVerts) / 2;
}

/**
 * Retained verts are part of several triangles, and it doesn't matter which one
 * the vertBary refers to. Here, whichever is last will win and it's done on the
 * CPU for simplicity for now. Using AtomicCAS on .tri should work for a GPU
 * version if desired.
 */
void FillRetainedVerts(Vec<Barycentric>& vertBary,
                       const Vec<Halfedge>& halfedge_) {
  const int numTri = halfedge_.size() / 3;
  for (int tri = 0; tri < numTri; ++tri) {
    for (const int i : {0, 1, 2}) {
      glm::dvec3 uvw(0);
      uvw[i] = 1;
      vertBary[halfedge_[3 * tri + i].startVert] = {tri, uvw};
    }
  }
}

struct ReindexHalfedge {
  VecView<int> half2Edge;

  void operator()(thrust::tuple<int, TmpEdge> in) {
    const int edge = thrust::get<0>(in);
    const int halfedge = thrust::get<1>(in).halfedgeIdx;

    half2Edge[halfedge] = edge;
  }
};

struct EdgeVerts {
  VecView<glm::dvec3> vertPos;
  VecView<Barycentric> vertBary;
  const int startIdx;
  const int n;

  void operator()(thrust::tuple<int, TmpEdge> in) {
    int edge = thrust::get<0>(in);
    TmpEdge edgeVerts = thrust::get<1>(in);

    double invTotal = 1.0 / n;
    for (int i = 1; i < n; ++i) {
      const int vert = startIdx + (n - 1) * edge + i - 1;
      const double v = i * invTotal;
      const double u = 1 - v;
      vertPos[vert] =
          u * vertPos[edgeVerts.first] + v * vertPos[edgeVerts.second];

      const int tri = edgeVerts.halfedgeIdx / 3;
      const int idx = edgeVerts.halfedgeIdx - 3 * tri;
      glm::dvec3 uvw(0);
      uvw[idx] = u;
      uvw[Next3(idx)] = v;
      vertBary[vert] = {tri, uvw};
    }
  }
};

struct InteriorVerts {
  VecView<glm::dvec3> vertPos;
  VecView<Barycentric> vertBary;
  const int startIdx;
  const int n;
  VecView<const Halfedge> halfedge;

  void operator()(int tri) {
    const double invTotal = 1.0 / n;
    int pos = startIdx + tri * VertsPerTri(n - 2);
    for (int i = 0; i <= n; ++i) {
      for (int j = 0; j <= n - i; ++j) {
        const int k = n - i - j;
        const double u = invTotal * j;
        const double v = invTotal * k;
        const double w = invTotal * i;
        if (i == 0 || j == 0 || k == 0 || j == n - i) continue;

        vertPos[pos] = u * vertPos[halfedge[3 * tri].startVert] +      //
                       v * vertPos[halfedge[3 * tri + 1].startVert] +  //
                       w * vertPos[halfedge[3 * tri + 2].startVert];

        vertBary[pos++] = {tri, {u, v, w}};
      }
    }
  }
};

struct SplitTris {
  VecView<glm::ivec3> triVerts;
  VecView<Halfedge> halfedge;
  VecView<const int> half2Edge;
  const int edgeIdx;
  const int triIdx;
  const int n;

  int EdgeVert(int i, int inHalfedge) const {
    bool forward = halfedge[inHalfedge].IsForward();
    int edge = forward ? half2Edge[inHalfedge]
                       : half2Edge[halfedge[inHalfedge].pairedHalfedge];
    return edgeIdx + (n - 1) * edge + (forward ? i - 1 : n - 1 - i);
  }

  int TriVert(int i, int j, int tri) const {
    --i;
    --j;
    int m = n - 2;
    int vertsPerTri = (m * m + m) / 2;
    int vertOffset = (i * (2 * m - i + 1)) / 2 + j;
    return triIdx + vertsPerTri * tri + vertOffset;
  }

  int Vert(int i, int j, int tri) const {
    bool edge0 = i == 0;
    bool edge1 = j == 0;
    bool edge2 = j == n - i;
    if (edge0) {
      if (edge1)
        return halfedge[3 * tri + 1].startVert;
      else if (edge2)
        return halfedge[3 * tri].startVert;
      else
        return EdgeVert(n - j, 3 * tri);
    } else if (edge1) {
      if (edge2)
        return halfedge[3 * tri + 2].startVert;
      else
        return EdgeVert(i, 3 * tri + 1);
    } else if (edge2)
      return EdgeVert(j, 3 * tri + 2);
    else
      return TriVert(i, j, tri);
  }

  void operator()(int tri) {
    int pos = n * n * tri;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n - i; ++j) {
        int a = Vert(i, j, tri);
        int b = Vert(i + 1, j, tri);
        int c = Vert(i, j + 1, tri);
        triVerts[pos++] = glm::ivec3(c, a, b);
        if (j < n - 1 - i) {
          int d = Vert(i + 1, j + 1, tri);
          triVerts[pos++] = glm::ivec3(b, d, c);
        }
      }
    }
  }
};

struct SmoothBezier {
  VecView<const glm::dvec3> vertPos;
  VecView<const glm::dvec3> triNormal;
  VecView<const glm::dvec3> vertNormal;
  VecView<const Halfedge> halfedge;

  void operator()(thrust::tuple<glm::dvec4&, Halfedge> inOut) {
    glm::dvec4& tangent = thrust::get<0>(inOut);
    const Halfedge edge = thrust::get<1>(inOut);

    const glm::dvec3 startV = vertPos[edge.startVert];
    const glm::dvec3 edgeVec = vertPos[edge.endVert] - startV;
    const glm::dvec3 edgeNormal =
        (triNormal[edge.face] + triNormal[halfedge[edge.pairedHalfedge].face]) /
        2.0;
    glm::dvec3 dir = SafeNormalize(glm::cross(glm::cross(edgeNormal, edgeVec),
                                              vertNormal[edge.startVert]));

    double weight = glm::abs(glm::dot(dir, SafeNormalize(edgeVec)));
    if (weight == 0) {
      weight = 1;
    }
    // Quadratic weighted bezier for circular interpolation
    const glm::dvec4 bz2 =
        weight *
        glm::dvec4(startV + dir * glm::length(edgeVec) / (2 * weight), 1.0);
    // Equivalent cubic weighted bezier
    const glm::dvec4 bz3 = glm::mix(glm::dvec4(startV, 1.0), bz2, 2 / 3.0);
    // Convert from homogeneous form to geometric form
    tangent = glm::dvec4(glm::dvec3(bz3) / bz3.w - startV, bz3.w);
  }
};

struct InterpTri {
  VecView<const Halfedge> halfedge;
  VecView<const glm::dvec4> halfedgeTangent;
  VecView<const glm::dvec3> vertPos;

  glm::dvec4 Homogeneous(glm::dvec4 v) const {
    v.x *= v.w;
    v.y *= v.w;
    v.z *= v.w;
    return v;
  }

  glm::dvec4 Homogeneous(glm::dvec3 v) const { return glm::dvec4(v, 1.0); }

  glm::dvec3 HNormalize(glm::dvec4 v) const { return glm::dvec3(v) / v.w; }

  glm::dvec4 Bezier(glm::dvec3 point, glm::dvec4 tangent) const {
    return Homogeneous(glm::dvec4(point, 0) + tangent);
  }

  glm::dmat2x4 CubicBezier2Linear(glm::dvec4 p0, glm::dvec4 p1, glm::dvec4 p2,
                                  glm::dvec4 p3, double x) const {
    glm::dmat2x4 out;
    glm::dvec4 p12 = glm::mix(p1, p2, x);
    out[0] = glm::mix(glm::mix(p0, p1, x), p12, x);
    out[1] = glm::mix(p12, glm::mix(p2, p3, x), x);
    return out;
  }

  glm::dvec3 BezierPoint(glm::dmat2x4 points, double x) const {
    return HNormalize(glm::mix(points[0], points[1], x));
  }

  glm::dvec3 BezierTangent(glm::dmat2x4 points) const {
    return glm::normalize(HNormalize(points[1]) - HNormalize(points[0]));
  }

  void operator()(thrust::tuple<glm::dvec3&, Barycentric> inOut) {
    glm::dvec3& pos = thrust::get<0>(inOut);
    const int tri = thrust::get<1>(inOut).tri;
    const glm::dvec3 uvw = thrust::get<1>(inOut).uvw;

    glm::dvec4 posH(0);
    const glm::dmat3 corners = {vertPos[halfedge[3 * tri].startVert],
                                vertPos[halfedge[3 * tri + 1].startVert],
                                vertPos[halfedge[3 * tri + 2].startVert]};

    for (const int i : {0, 1, 2}) {
      if (uvw[i] == 1) {
        pos = glm::dvec3(corners[i]);
        return;
      }
    }

    const glm::dmat3x4 tangentR = {halfedgeTangent[3 * tri],
                                   halfedgeTangent[3 * tri + 1],
                                   halfedgeTangent[3 * tri + 2]};
    const glm::dmat3x4 tangentL = {
        halfedgeTangent[halfedge[3 * tri + 2].pairedHalfedge],
        halfedgeTangent[halfedge[3 * tri].pairedHalfedge],
        halfedgeTangent[halfedge[3 * tri + 1].pairedHalfedge]};

    for (const int i : {0, 1, 2}) {
      const int j = (i + 1) % 3;
      const int k = (i + 2) % 3;
      const double x = uvw[k] / (1 - uvw[i]);

      const glm::dmat2x4 bez = CubicBezier2Linear(
          Homogeneous(corners[j]), Bezier(corners[j], tangentR[j]),
          Bezier(corners[k], tangentL[k]), Homogeneous(corners[k]), x);
      const glm::dvec3 end = BezierPoint(bez, x);
      const glm::dvec3 tangent = BezierTangent(bez);

      const glm::dvec3 jBitangent = SafeNormalize(OrthogonalTo(
          glm::dvec3(tangentL[j]), SafeNormalize(glm::dvec3(tangentR[j]))));
      const glm::dvec3 kBitangent = SafeNormalize(OrthogonalTo(
          glm::dvec3(tangentR[k]), -SafeNormalize(glm::dvec3(tangentL[k]))));
      const glm::dvec3 normal = SafeNormalize(
          glm::cross(glm::mix(jBitangent, kBitangent, x), tangent));
      const glm::dvec3 delta = OrthogonalTo(
          glm::mix(glm::dvec3(tangentL[j]), glm::dvec3(tangentR[k]), x),
          normal);
      const double deltaW = glm::mix(tangentL[j].w, tangentR[k].w, x);

      const glm::dmat2x4 bez1 = CubicBezier2Linear(
          Homogeneous(end), Homogeneous(glm::dvec4(end + delta, deltaW)),
          Bezier(corners[i], glm::mix(tangentR[i], tangentL[i], x)),
          Homogeneous(corners[i]), uvw[i]);
      const glm::dvec3 p = BezierPoint(bez1, uvw[i]);
      double w = uvw[j] * uvw[j] * uvw[k] * uvw[k];
      posH += Homogeneous(glm::dvec4(p, w));
    }
    pos = HNormalize(posH);
  }
};
}  // namespace

namespace manifold {

/**
 * Calculates halfedgeTangent_, allowing the manifold to be refined and
 * smoothed. The tangents form weighted cubic Beziers along each edge. This
 * function creates circular arcs where possible (minimizing maximum curvature),
 * constrained to the vertex normals. Where sharpenedEdges are specified, the
 * tangents are shortened that intersect the sharpened edge, concentrating the
 * curvature there, while the tangents of the sharp edges themselves are aligned
 * for continuity.
 */
void Manifold::Impl::CreateTangents(
    const std::vector<Smoothness>& sharpenedEdges) {
  ZoneScoped;
  const int numHalfedge = halfedge_.size();
  halfedgeTangent_.resize(numHalfedge);

  for_each_n(autoPolicy(numHalfedge),
             zip(halfedgeTangent_.begin(), halfedge_.cbegin()), numHalfedge,
             SmoothBezier({vertPos_, faceNormal_, vertNormal_, halfedge_}));

  if (!sharpenedEdges.empty()) {
    const Vec<TriRef>& triRef = meshRelation_.triRef;

    // sharpenedEdges are referenced to the input Mesh, but the triangles have
    // been sorted in creating the Manifold, so the indices are converted using
    // meshRelation_.
    std::vector<int> oldHalfedge2New(halfedge_.size());
    for (int tri = 0; tri < NumTri(); ++tri) {
      int oldTri = triRef[tri].tri;
      for (int i : {0, 1, 2}) oldHalfedge2New[3 * oldTri + i] = 3 * tri + i;
    }

    using Pair = std::pair<Smoothness, Smoothness>;
    // Fill in missing pairs with default smoothness = 1.
    std::map<int, Pair> edges;
    for (Smoothness edge : sharpenedEdges) {
      if (edge.smoothness == 1) continue;
      edge.halfedge = oldHalfedge2New[edge.halfedge];
      int pair = halfedge_[edge.halfedge].pairedHalfedge;
      if (edges.find(pair) == edges.end()) {
        edges[edge.halfedge] = {edge, {pair, 1}};
      } else {
        edges[pair].second = edge;
      }
    }

    std::map<int, std::vector<Pair>> vertTangents;
    for (const auto& value : edges) {
      const Pair edge = value.second;
      vertTangents[halfedge_[edge.first.halfedge].startVert].push_back(edge);
      vertTangents[halfedge_[edge.second.halfedge].startVert].push_back(
          {edge.second, edge.first});
    }

    Vec<glm::dvec4>& tangent = halfedgeTangent_;
    for (const auto& value : vertTangents) {
      const std::vector<Pair>& vert = value.second;
      // Sharp edges that end are smooth at their terminal vert.
      if (vert.size() == 1) continue;
      if (vert.size() == 2) {  // Make continuous edge
        const int first = vert[0].first.halfedge;
        const int second = vert[1].first.halfedge;
        const glm::dvec3 newTangent = glm::normalize(
            glm::dvec3(tangent[first]) - glm::dvec3(tangent[second]));
        tangent[first] =
            glm::dvec4(glm::length(glm::dvec3(tangent[first])) * newTangent,
                       tangent[first].w);
        tangent[second] =
            glm::dvec4(-glm::length(glm::dvec3(tangent[second])) * newTangent,
                       tangent[second].w);

        auto SmoothHalf = [&](int first, int last, double smoothness) {
          int current = NextHalfedge(halfedge_[first].pairedHalfedge);
          while (current != last) {
            const double cosBeta = glm::dot(
                newTangent, glm::normalize(glm::dvec3(tangent[current])));
            const double factor =
                (1 - smoothness) * cosBeta * cosBeta + smoothness;
            tangent[current] = glm::dvec4(factor * glm::dvec3(tangent[current]),
                                          tangent[current].w);
            current = NextHalfedge(halfedge_[current].pairedHalfedge);
          }
        };

        SmoothHalf(first, second,
                   (vert[0].second.smoothness + vert[1].first.smoothness) / 2);
        SmoothHalf(second, first,
                   (vert[1].second.smoothness + vert[0].first.smoothness) / 2);

      } else {  // Sharpen vertex uniformly
        double smoothness = 0;
        for (const Pair& pair : vert) {
          smoothness += pair.first.smoothness;
          smoothness += pair.second.smoothness;
        }
        smoothness /= 2 * vert.size();

        const int start = vert[0].first.halfedge;
        int current = start;
        do {
          tangent[current] = glm::dvec4(
              smoothness * glm::dvec3(tangent[current]), tangent[current].w);
          current = NextHalfedge(halfedge_[current].pairedHalfedge);
        } while (current != start);
      }
    }
  }
}

/**
 * Split each edge into n pieces and sub-triangulate each triangle accordingly.
 * This function doesn't run Finish(), as that is expensive and it'll need to be
 * run after the new vertices have moved, which is a likely scenario after
 * refinement (smoothing).
 */
Vec<Barycentric> Manifold::Impl::Subdivide(int n) {
  ZoneScoped;
  if (n < 2) return Vec<Barycentric>();
  faceNormal_.resize(0);
  vertNormal_.resize(0);
  const int numVert = NumVert();
  const int numEdge = NumEdge();
  const int numTri = NumTri();
  // Append new verts
  const int vertsPerEdge = n - 1;
  const int triVertStart = numVert + numEdge * vertsPerEdge;
  vertPos_.resize(triVertStart + numTri * VertsPerTri(n - 2));
  Vec<Barycentric> vertBary(vertPos_.size());
  FillRetainedVerts(vertBary, halfedge_);

  MeshRelationD oldMeshRelation = std::move(meshRelation_);
  meshRelation_.triRef.resize(n * n * numTri);
  meshRelation_.originalID = oldMeshRelation.originalID;

  Vec<TmpEdge> edges = CreateTmpEdges(halfedge_);
  Vec<int> half2Edge(2 * numEdge);
  auto policy = autoPolicy(numEdge);
  for_each_n(policy, zip(countAt(0), edges.begin()), numEdge,
             ReindexHalfedge({half2Edge}));
  for_each_n(policy, zip(countAt(0), edges.begin()), numEdge,
             EdgeVerts({vertPos_, vertBary, numVert, n}));
  for_each_n(policy, countAt(0), numTri,
             InteriorVerts({vertPos_, vertBary, triVertStart, n, halfedge_}));
  // Create sub-triangles
  Vec<glm::ivec3> triVerts(n * n * numTri);
  for_each_n(
      policy, countAt(0), numTri,
      SplitTris({triVerts, halfedge_, half2Edge, numVert, triVertStart, n}));
  CreateHalfedges(triVerts);
  // Make original since the subdivided faces are intended to be warped into
  // being non-coplanar, and hence not being related to the original faces.
  meshRelation_.originalID = ReserveIDs(1);
  InitializeOriginal();

  if (meshRelation_.numProp > 0) {
    meshRelation_.properties.resize(meshRelation_.numProp * numVert);
    meshRelation_.triProperties.resize(meshRelation_.triRef.size());
    // Fill properties according to barycentric.
    // Set triProp to share properties on continuous edges.
    // Duplicate properties will be removed during sorting.
  }

  return vertBary;
}

void Manifold::Impl::Refine(int n) {
  Manifold::Impl old = *this;
  Vec<Barycentric> vertBary = Subdivide(n);
  if (vertBary.size() == 0) return;

  if (old.halfedgeTangent_.size() == old.halfedge_.size()) {
    for_each_n(autoPolicy(NumTri()), zip(vertPos_.begin(), vertBary.begin()),
               NumVert(),
               InterpTri({old.halfedge_, old.halfedgeTangent_, old.vertPos_}));
  }

  halfedgeTangent_.resize(0);
  Finish();
}
}  // namespace manifold
