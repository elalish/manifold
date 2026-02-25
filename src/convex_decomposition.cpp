// Copyright 2025 The Manifold Authors.
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "csg_tree.h"
#include "impl.h"
#include "iters.h"
#include "parallel.h"

namespace manifold {

namespace {

// ---------------------------------------------------------------------------
// Delaunay tetrahedralization helpers
// ---------------------------------------------------------------------------

// Face indices for each of the 4 faces of a tetrahedron.
const int kTetFaces[4][3] = {{2, 1, 0}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};

// Deterministic perturbation for DT robustness on cospheric points. Used only
// for computing DT connectivity — all output uses original unperturbed
// positions.
inline double DeterministicEps(size_t index, int component) {
  constexpr double eps = 1e-10;
  uint64_t h = index * 2654435761ULL + component * 40503ULL;
  h = (h ^ (h >> 16)) * 0x45d9f3bULL;
  return -eps + 2.0 * ((h & 0xFFFFFF) / double(0xFFFFFF)) * eps;
}

vec3 GetCircumCenter(const vec3& p0, const vec3& p1, const vec3& p2,
                     const vec3& p3) {
  vec3 b = p1 - p0, c = p2 - p0, d = p3 - p0;
  double det =
      2.0 * (b.x * (c.y * d.z - c.z * d.y) - b.y * (c.x * d.z - c.z * d.x) +
             b.z * (c.x * d.y - c.y * d.x));
  if (det == 0.0) return p0;
  vec3 v = la::cross(c, d) * la::dot(b, b) + la::cross(d, b) * la::dot(c, c) +
           la::cross(b, c) * la::dot(d, d);
  return p0 + v / det;
}

// Scale-invariant tet quality: returns 1.0 for a regular tet, 0.0 for
// degenerate. Negative means inverted orientation.
double TetQuality(const vec3& p0, const vec3& p1, const vec3& p2,
                  const vec3& p3) {
  vec3 d0 = p1 - p0, d1 = p2 - p0, d2 = p3 - p0;
  vec3 d3 = p2 - p1, d4 = p3 - p2, d5 = p1 - p3;
  double s0 = la::length(d0), s1 = la::length(d1), s2 = la::length(d2);
  double s3 = la::length(d3), s4 = la::length(d4), s5 = la::length(d5);
  double ms = (s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4 + s5 * s5) / 6.0;
  double rms = std::sqrt(ms);
  if (rms == 0.0) return 0.0;
  double vol = la::dot(d0, la::cross(d1, d2)) / 6.0;
  return 12.0 / std::sqrt(2.0) * vol / (rms * rms * rms);
}

struct Edge {
  int id0, id1, tetNr, faceNr;
  bool operator<(const Edge& o) const {
    return id0 < o.id0 || (id0 == o.id0 && id1 < o.id1);
  }
  bool operator==(const Edge& o) const { return id0 == o.id0 && id1 == o.id1; }
};

// Incremental Delaunay tetrahedralization via Bowyer-Watson.
std::vector<uint32_t> CreateTetIds(std::vector<vec3>& verts,
                                   double minQuality) {
  std::vector<int> tetIds;
  std::vector<int> neighbors;
  std::vector<int> tetMarks;
  int tetMark = 0;
  int firstFreeTet = -1;
  std::vector<vec3> planesN;
  std::vector<double> planesD;
  int firstBig = (int)verts.size() - 4;
  const size_t maxTotalTets = (size_t)firstBig * 20;

  tetIds.push_back(firstBig);
  tetIds.push_back(firstBig + 1);
  tetIds.push_back(firstBig + 2);
  tetIds.push_back(firstBig + 3);
  tetMarks.push_back(0);

  for (int i = 0; i < 4; i++) {
    neighbors.push_back(-1);
    vec3 p0 = verts[firstBig + kTetFaces[i][0]];
    vec3 p1 = verts[firstBig + kTetFaces[i][1]];
    vec3 p2 = verts[firstBig + kTetFaces[i][2]];
    vec3 n = la::normalize(la::cross(p1 - p0, p2 - p0));
    planesN.push_back(n);
    planesD.push_back(la::dot(p0, n));
  }

  int skippedPoints = 0;

  for (int i = 0; i < firstBig; i++) {
    vec3 p = verts[i];
    if ((size_t)tetIds.size() / 4 > maxTotalTets) {
#ifdef MANIFOLD_DEBUG
      printf("ConvexDecomposition DT: tet limit reached, skipping %d points\n",
             firstBig - i);
#endif
      break;
    }

    // Walk to the containing tet.
    int tetNr = 0;
    while (tetIds[4 * tetNr] < 0) tetNr++;

    tetMark++;
    bool found = false;
    int walkSteps = 0;
    const int maxWalkSteps = (int)tetIds.size() / 4 + 100;
    while (!found) {
      if (tetNr < 0 || tetMarks[tetNr] == tetMark) break;
      if (++walkSteps > maxWalkSteps) break;
      tetMarks[tetNr] = tetMark;
      vec3 center =
          (verts[tetIds[4 * tetNr]] + verts[tetIds[4 * tetNr + 1]] +
           verts[tetIds[4 * tetNr + 2]] + verts[tetIds[4 * tetNr + 3]]) *
          0.25;
      double minT = std::numeric_limits<double>::infinity();
      int minFaceNr = -1;
      for (int j = 0; j < 4; j++) {
        vec3 n = planesN[4 * tetNr + j];
        double d = planesD[4 * tetNr + j];
        double hp = la::dot(n, p) - d;
        double hc = la::dot(n, center) - d;
        double t = hp - hc;
        if (t == 0) continue;
        t = -hc / t;
        if (t >= 0.0 && t < minT) {
          minT = t;
          minFaceNr = j;
        }
      }
      if (minT >= 1.0)
        found = true;
      else
        tetNr = neighbors[4 * tetNr + minFaceNr];
    }
    if (!found) {
      skippedPoints++;
      continue;
    }

    // Find all tets whose circumsphere contains the new point.
    tetMark++;
    std::vector<int> violatingTets;
    std::vector<int> stack = {tetNr};
    while (!stack.empty()) {
      tetNr = stack.back();
      stack.pop_back();
      if (tetMarks[tetNr] == tetMark) continue;
      tetMarks[tetNr] = tetMark;
      violatingTets.push_back(tetNr);
      for (int j = 0; j < 4; j++) {
        int n = neighbors[4 * tetNr + j];
        if (n < 0 || tetMarks[n] == tetMark) continue;
        vec3 c =
            GetCircumCenter(verts[tetIds[4 * n]], verts[tetIds[4 * n + 1]],
                            verts[tetIds[4 * n + 2]], verts[tetIds[4 * n + 3]]);
        double r = la::length(verts[tetIds[4 * n]] - c);
        if (la::length(p - c) < r) stack.push_back(n);
      }
    }

    // Replace violating tets with new tets connecting boundary faces to point.
    std::vector<Edge> edges;
    for (int j = 0; j < (int)violatingTets.size(); j++) {
      tetNr = violatingTets[j];
      int ids[4], ns[4];
      for (int k = 0; k < 4; k++) {
        ids[k] = tetIds[4 * tetNr + k];
        ns[k] = neighbors[4 * tetNr + k];
      }
      tetIds[4 * tetNr] = -1;
      tetIds[4 * tetNr + 1] = firstFreeTet;
      firstFreeTet = tetNr;

      for (int k = 0; k < 4; k++) {
        int n = ns[k];
        if (n >= 0 && tetMarks[n] == tetMark) continue;
        int newTetNr = firstFreeTet;
        if (newTetNr >= 0) {
          firstFreeTet = tetIds[4 * firstFreeTet + 1];
        } else {
          newTetNr = (int)tetIds.size() / 4;
          tetMarks.push_back(0);
          for (int l = 0; l < 4; l++) {
            tetIds.push_back(-1);
            neighbors.push_back(-1);
            planesN.push_back(vec3());
            planesD.push_back(0.0);
          }
        }
        int nid0 = ids[kTetFaces[k][2]];
        int nid1 = ids[kTetFaces[k][1]];
        int nid2 = ids[kTetFaces[k][0]];
        tetIds[4 * newTetNr] = nid0;
        tetIds[4 * newTetNr + 1] = nid1;
        tetIds[4 * newTetNr + 2] = nid2;
        tetIds[4 * newTetNr + 3] = i;
        neighbors[4 * newTetNr] = n;
        if (n >= 0) {
          for (int l = 0; l < 4; l++) {
            if (neighbors[4 * n + l] == tetNr) neighbors[4 * n + l] = newTetNr;
          }
        }
        neighbors[4 * newTetNr + 1] = -1;
        neighbors[4 * newTetNr + 2] = -1;
        neighbors[4 * newTetNr + 3] = -1;
        for (int l = 0; l < 4; l++) {
          vec3 fp0 = verts[tetIds[4 * newTetNr + kTetFaces[l][0]]];
          vec3 fp1 = verts[tetIds[4 * newTetNr + kTetFaces[l][1]]];
          vec3 fp2 = verts[tetIds[4 * newTetNr + kTetFaces[l][2]]];
          vec3 newN = la::normalize(la::cross(fp1 - fp0, fp2 - fp0));
          planesN[4 * newTetNr + l] = newN;
          planesD[4 * newTetNr + l] = la::dot(newN, fp0);
        }

        Edge e;
        e.tetNr = newTetNr;
        e.id0 = std::min(nid0, nid1);
        e.id1 = std::max(nid0, nid1);
        e.faceNr = 1;
        edges.push_back(e);
        e.id0 = std::min(nid1, nid2);
        e.id1 = std::max(nid1, nid2);
        e.faceNr = 2;
        edges.push_back(e);
        e.id0 = std::min(nid2, nid0);
        e.id1 = std::max(nid2, nid0);
        e.faceNr = 3;
        edges.push_back(e);
      }
    }

    // Connect new tets to each other via shared edges.
    std::sort(edges.begin(), edges.end());
    size_t nr = 0;
    while (nr < edges.size()) {
      Edge e0 = edges[nr++];
      if (nr < edges.size() && edges[nr] == e0) {
        Edge e1 = edges[nr++];
        neighbors[4 * e0.tetNr + e0.faceNr] = e1.tetNr;
        neighbors[4 * e1.tetNr + e1.faceNr] = e0.tetNr;
      }
    }
  }

#ifdef MANIFOLD_DEBUG
  if (skippedPoints > 0)
    printf("ConvexDecomposition DT: %d/%d points skipped (degenerate)\n",
           skippedPoints, firstBig);
#endif

  // Filter out super-tet vertices and degenerate tets.
  int numTets = (int)tetIds.size() / 4;
  std::vector<uint32_t> result;
  for (int i = 0; i < numTets; i++) {
    int id0 = tetIds[4 * i], id1 = tetIds[4 * i + 1];
    int id2 = tetIds[4 * i + 2], id3 = tetIds[4 * i + 3];
    if (id0 < 0 || id0 >= firstBig || id1 >= firstBig || id2 >= firstBig ||
        id3 >= firstBig)
      continue;
    if (TetQuality(verts[id0], verts[id1], verts[id2], verts[id3]) < minQuality)
      continue;
    result.push_back((uint32_t)id0);
    result.push_back((uint32_t)id1);
    result.push_back((uint32_t)id2);
    result.push_back((uint32_t)id3);
  }
  return result;
}

// Build tet-to-tet adjacency via shared triangular faces.
std::vector<int> BuildTetAdjacency(const std::vector<uint32_t>& faces) {
  int numTets = (int)faces.size() / 4;
  std::vector<int> adj(numTets * 4, -1);
  std::unordered_map<uint64_t, std::pair<int, int>> faceMap;
  faceMap.reserve(numTets * 4);
  auto sortedFaceKey = [](uint32_t a, uint32_t b, uint32_t c) -> uint64_t {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return (uint64_t)a | ((uint64_t)b << 21) | ((uint64_t)c << 42);
  };
  for (int i = 0; i < numTets; i++) {
    uint32_t v[4] = {faces[4 * i], faces[4 * i + 1], faces[4 * i + 2],
                     faces[4 * i + 3]};
    for (int j = 0; j < 4; j++) {
      uint64_t key = sortedFaceKey(v[kTetFaces[j][0]], v[kTetFaces[j][1]],
                                   v[kTetFaces[j][2]]);
      auto it = faceMap.find(key);
      if (it != faceMap.end()) {
        adj[4 * i + j] = it->second.first;
        adj[4 * it->second.first + it->second.second] = i;
        faceMap.erase(it);
      } else {
        faceMap[key] = {i, j};
      }
    }
  }
  return adj;
}

std::vector<vec3> ExtractVertices(const Manifold& m) {
  MeshGL64 mesh = m.GetMeshGL64();
  size_t nv = mesh.vertProperties.size() / mesh.numProp;
  std::vector<vec3> verts;
  verts.reserve(nv);
  for (size_t i = 0; i < nv; i++)
    verts.push_back(vec3(mesh.vertProperties[i * mesh.numProp],
                         mesh.vertProperties[i * mesh.numProp + 1],
                         mesh.vertProperties[i * mesh.numProp + 2]));
  return verts;
}

// ---------------------------------------------------------------------------
// Coplanar reflex plane detection (pre-pass)
// ---------------------------------------------------------------------------

// Scan all reflex edges for a coplanar chain (>= 6 connected edges on one
// plane). When found, return the two halves from splitting. This resolves
// sphere intersection seams perfectly: e.g. TwoSpheres goes from 271 to 2
// pieces.
bool TrySplitByCoplanarReflexPlane(
    const Manifold& shape,
    const std::shared_ptr<const Manifold::Impl>& shapeImpl, Manifold& outA,
    Manifold& outB) {
  const auto& he = shapeImpl->halfedge_;
  const auto& fn = shapeImpl->faceNormal_;
  const auto& vp = shapeImpl->vertPos_;
  const size_t nbEdges = he.size();

  // Collect reflex edges.
  std::vector<std::pair<int, int>> reflexEdges;
  for (size_t idx = 0; idx < nbEdges; idx++) {
    auto edge = he[idx];
    if (!edge.IsForward()) continue;
    vec3 n0 = fn[idx / 3];
    vec3 n1 = fn[edge.pairedHalfedge / 3];
    vec3 edgeVec = vp[edge.endVert] - vp[edge.startVert];
    if (la::dot(edgeVec, la::cross(n0, n1)) < 0)
      reflexEdges.push_back({edge.startVert, edge.endVert});
  }

  if (reflexEdges.size() < 2) return false;

  // Build vertex adjacency among reflex edges and compute bounding spread.
  std::unordered_map<int, std::unordered_set<int>> adj;
  std::unordered_set<int> reflexVertSet;
  for (auto& [sv, ev] : reflexEdges) {
    adj[sv].insert(ev);
    adj[ev].insert(sv);
    reflexVertSet.insert(sv);
    reflexVertSet.insert(ev);
  }
  std::vector<int> allReflexVerts(reflexVertSet.begin(), reflexVertSet.end());

  vec3 centroid(0, 0, 0);
  for (int v : allReflexVerts) centroid += vp[v];
  centroid /= (double)allReflexVerts.size();
  double spread = 0;
  for (int v : allReflexVerts)
    spread = std::max(spread, la::length(vp[v] - centroid));
  if (spread < 1e-10) return false;

  // For each pair of adjacent reflex edges, compute a candidate plane and
  // count how many reflex vertices lie on it (within 2% of spread).
  constexpr double kCoplanarTol = 0.02;
  double tol = kCoplanarTol * spread;

  struct Candidate {
    vec3 normal;
    double offset;
    int edgesOnPlane;
  };
  std::vector<Candidate> candidates;

  // To avoid duplicates, track seen plane normals (quantized).
  auto planeKey = [](vec3 n, double d) -> uint64_t {
    auto qi = [](double v) -> int { return (int)std::round(v * 1000); };
    // Canonicalize sign: first nonzero component positive.
    if (n.x < -1e-6 || (std::abs(n.x) < 1e-6 && n.y < -1e-6) ||
        (std::abs(n.x) < 1e-6 && std::abs(n.y) < 1e-6 && n.z < -1e-6)) {
      n = -n;
      d = -d;
    }
    uint64_t h = (uint64_t)(qi(n.x) + 2000) * 4001 * 4001 * 10001 +
                 (uint64_t)(qi(n.y) + 2000) * 4001 * 10001 +
                 (uint64_t)(qi(n.z) + 2000) * 10001 + (uint64_t)(qi(d) + 5000);
    return h;
  };
  std::unordered_set<uint64_t> seenPlanes;

  for (auto& [sv, ev] : reflexEdges) {
    vec3 p0 = vp[sv], p1 = vp[ev];
    vec3 v1 = p1 - p0;
    for (int c : adj[sv]) {
      if (c == ev) continue;
      vec3 v2 = vp[c] - p0;
      vec3 n = la::cross(v1, v2);
      double nl = la::length(n);
      if (nl < 1e-10) continue;
      n /= nl;
      double d = la::dot(n, p0);

      uint64_t key = planeKey(n, d);
      if (!seenPlanes.insert(key).second) continue;

      int edges = 0;
      for (auto& [a, b] : reflexEdges) {
        if (std::abs(la::dot(n, vp[a]) - d) < tol &&
            std::abs(la::dot(n, vp[b]) - d) < tol)
          edges++;
      }
      // Accept planes where coplanar reflex edges form a significant
      // fraction of all reflex edges (>= 25%) and there are at least 6
      // edges (a proper ring, not scattered noise). This catches
      // individual cylinder seams in multi-hole shapes like CubeSphereHole
      // where each hole accounts for ~33% of reflex edges.
      if (edges >= 6 && edges * 4 >= (int)reflexEdges.size())
        candidates.push_back({n, d, edges});
    }
  }

  if (candidates.empty()) return false;

  // Sort by most edges on plane (best candidates first).
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.edgesOnPlane > b.edgesOnPlane;
            });

  // Try top candidates. Accept the first that produces a valid split
  // where the smaller half has > 1% of the original volume.
  double origVol = shape.Volume();
  int maxTry = std::min((int)candidates.size(), 10);
  for (int i = 0; i < maxTry; i++) {
    auto& cand = candidates[i];
    auto [a, b] = shape.SplitByPlane(cand.normal, cand.offset);
    double aVol = a.Volume(), bVol = b.Volume();
    double minRatio = std::min(aVol, bVol) / origVol;
    if (a.IsEmpty() || b.IsEmpty() || minRatio <= 0.01 ||
        std::abs(aVol + bVol - origVol) >= origVol * 0.01)
      continue;
    outA = a;
    outB = b;
    return true;
  }
  return false;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Decompose into approximately convex pieces.
 *
 * Algorithm overview:
 *   1. Pre-pass: detect coplanar reflex edge chains and split by plane
 *   2. Delaunay tetrahedralization of mesh vertices
 *   3. Clip each DT tet against the mesh surface
 *   4. Cospheric merge: group tets sharing a circumcenter
 *   5. Priority-queue merge: biggest-first pairwise convex merges
 *   6. Reflex edge splitting of remaining non-convex pieces
 */
std::vector<Manifold> Manifold::Impl::ConvexDecomposition(int maxDepth) const {
  std::vector<Manifold> outputs;

  auto thisImpl = std::make_shared<Impl>(*this);
  Manifold curShape(std::make_shared<CsgLeafNode>(thisImpl));

  if (curShape.IsEmpty()) return outputs;
  if (IsConvex()) {
    outputs.push_back(curShape.Hull());
    return outputs;
  }

  // Handle multiple disconnected components independently.
  std::vector<Manifold> shapes = curShape.Decompose();
  if (shapes.empty()) {
    outputs.push_back(curShape);
    return outputs;
  }

  for (auto& shape : shapes) {
    if (shape.IsEmpty()) continue;

    auto shapeImpl = shape.GetCsgLeafNode().GetImpl();
    if (shapeImpl->IsConvex()) {
      outputs.push_back(shape.Hull());
      continue;
    }

    // Step 1: Try splitting by a coplanar reflex edge plane. This resolves
    // sphere intersection seams without DT seam vertex poisoning.
    {
      Manifold halfA, halfB;
      if (TrySplitByCoplanarReflexPlane(shape, shapeImpl, halfA, halfB)) {
        auto aSub =
            halfA.GetCsgLeafNode().GetImpl()->ConvexDecomposition(maxDepth);
        outputs.insert(outputs.end(), aSub.begin(), aSub.end());
        auto bSub =
            halfB.GetCsgLeafNode().GetImpl()->ConvexDecomposition(maxDepth);
        outputs.insert(outputs.end(), bSub.begin(), bSub.end());
        continue;
      }
    }

    constexpr double kMergeTol = 1e-8;

    // Extract vertex positions for the DT.
    MeshGL64 meshGL = shape.GetMeshGL64();
    size_t numVerts = meshGL.vertProperties.size() / meshGL.numProp;
    if (numVerts < 4) {
      outputs.push_back(shape);
      continue;
    }

    std::vector<vec3> origPos(numVerts);
    for (size_t i = 0; i < numVerts; i++)
      origPos[i] = vec3(meshGL.vertProperties[i * meshGL.numProp + 0],
                        meshGL.vertProperties[i * meshGL.numProp + 1],
                        meshGL.vertProperties[i * meshGL.numProp + 2]);

    // Step 2: Unconstrained Delaunay tetrahedralization with deterministic
    // perturbation for cospheric robustness.
    std::vector<vec3> tetVerts(numVerts);
    for (size_t i = 0; i < numVerts; i++)
      tetVerts[i] =
          origPos[i] + vec3(DeterministicEps(i, 0), DeterministicEps(i, 1),
                            DeterministicEps(i, 2));

    // Add 8 expanded bounding box corners as DT vertices. This creates
    // tets near the mesh surface that capture thin features (e.g. notches
    // in FlatSlab) which would otherwise fall between degenerate tets
    // formed only by the distant super-tet corners.
    auto bbox = shape.BoundingBox();
    vec3 bMin(bbox.min.x, bbox.min.y, bbox.min.z);
    vec3 bMax(bbox.max.x, bbox.max.y, bbox.max.z);
    vec3 bCenter = (bMin + bMax) * 0.5;
    vec3 bHalf = (bMax - bMin) * 0.75;  // 50% expansion
    bMin = bCenter - bHalf;
    bMax = bCenter + bHalf;
    for (int i = 0; i < 8; i++) {
      vec3 corner(i & 1 ? bMax.x : bMin.x, i & 2 ? bMax.y : bMin.y,
                  i & 4 ? bMax.z : bMin.z);
      origPos.push_back(corner);
      tetVerts.push_back(corner + vec3(DeterministicEps(tetVerts.size(), 0),
                                       DeterministicEps(tetVerts.size(), 1),
                                       DeterministicEps(tetVerts.size(), 2)));
    }

    // Super-tet corners must encompass everything including bbox corners.
    vec3 center(0, 0, 0);
    for (const auto& p : tetVerts) center += p;
    center /= (double)tetVerts.size();
    double radius = 0.0;
    for (const auto& p : tetVerts)
      radius = std::max(radius, la::length(p - center));
    double s = 5.0 * radius;
    tetVerts.push_back(vec3(-s, 0.0, -s));
    tetVerts.push_back(vec3(s, 0.0, -s));
    tetVerts.push_back(vec3(0.0, s, s));
    tetVerts.push_back(vec3(0.0, -s, s));

    std::vector<uint32_t> flatTets = CreateTetIds(tetVerts, 0.0);

    // Step 2b: Reflex edge recovery. Check which reflex edges are missing
    // from the DT and insert Steiner midpoints to bias the DT toward
    // including them. This aligns tet boundaries with concavity boundaries.
    {
      const auto& he = shapeImpl->halfedge_;
      const auto& fn = shapeImpl->faceNormal_;

      // Build set of DT edges for fast lookup.
      std::unordered_set<uint64_t> dtEdges;
      int nt = (int)flatTets.size() / 4;
      for (int i = 0; i < nt; i++) {
        uint32_t v[4] = {flatTets[4 * i], flatTets[4 * i + 1],
                         flatTets[4 * i + 2], flatTets[4 * i + 3]};
        for (int a = 0; a < 4; a++)
          for (int b = a + 1; b < 4; b++) {
            uint32_t lo = std::min(v[a], v[b]);
            uint32_t hi = std::max(v[a], v[b]);
            if (lo < numVerts && hi < numVerts)
              dtEdges.insert((uint64_t)lo | ((uint64_t)hi << 32));
          }
      }

      // Find missing reflex edges and collect Steiner midpoints.
      std::vector<vec3> steinerPoints;
      int missingCount = 0, totalReflex = 0;
      for (size_t idx = 0; idx < he.size(); idx++) {
        auto edge = he[idx];
        if (!edge.IsForward()) continue;
        vec3 n0 = fn[idx / 3];
        vec3 n1 = fn[edge.pairedHalfedge / 3];
        vec3 edgeVec = origPos[edge.endVert] - origPos[edge.startVert];
        if (la::dot(edgeVec, la::cross(n0, n1)) >= 0) continue;
        totalReflex++;

        uint32_t lo =
            std::min((uint32_t)edge.startVert, (uint32_t)edge.endVert);
        uint32_t hi =
            std::max((uint32_t)edge.startVert, (uint32_t)edge.endVert);
        if (dtEdges.count((uint64_t)lo | ((uint64_t)hi << 32)) == 0) {
          missingCount++;
          steinerPoints.push_back(
              (origPos[edge.startVert] + origPos[edge.endVert]) * 0.5);
        }
      }

      // If any reflex edges are missing, add Steiner points and rebuild DT.
      if (!steinerPoints.empty()) {
#ifdef MANIFOLD_DEBUG
        printf(
            "ConvexDecomposition: %d/%d reflex edges missing from DT, "
            "adding %d Steiner points\n",
            missingCount, totalReflex, (int)steinerPoints.size());
#endif
        size_t oldSize = tetVerts.size();
        // Remove the 4 super-tet corners before appending.
        tetVerts.resize(oldSize - 4);
        origPos.resize(origPos.size());  // origPos doesn't have super-tet

        for (size_t i = 0; i < steinerPoints.size(); i++) {
          origPos.push_back(steinerPoints[i]);
          tetVerts.push_back(steinerPoints[i] +
                             vec3(DeterministicEps(tetVerts.size(), 0),
                                  DeterministicEps(tetVerts.size(), 1),
                                  DeterministicEps(tetVerts.size(), 2)));
        }

        // Re-add super-tet corners.
        tetVerts.push_back(vec3(-s, 0.0, -s));
        tetVerts.push_back(vec3(s, 0.0, -s));
        tetVerts.push_back(vec3(0.0, s, s));
        tetVerts.push_back(vec3(0.0, -s, s));

        flatTets = CreateTetIds(tetVerts, 0.0);
      }
    }

    int numTets = (int)flatTets.size() / 4;
    if (numTets == 0) {
      outputs.push_back(shape);
      continue;
    }

    // Step 3: Clip each DT tet against the mesh surface. Interior tets
    // (where clipped volume ≈ hull volume) keep the original hull to avoid
    // introducing boolean vertices at tet boundaries.
    auto adj = BuildTetAdjacency(flatTets);
    double totalVol = shape.Volume();
    auto shapeBbox = shape.BoundingBox();

    std::vector<Manifold> pieces(numTets);
    std::vector<bool> valid(numTets, false);

    for_each_n(autoPolicy(numTets, 16), countAt(0), numTets, [&](int i) {
      uint32_t i0 = flatTets[4 * i], i1 = flatTets[4 * i + 1],
               i2 = flatTets[4 * i + 2], i3 = flatTets[4 * i + 3];
      vec3 p0 = origPos[i0], p1 = origPos[i1], p2 = origPos[i2],
           p3 = origPos[i3];

      if (std::abs(TetQuality(p0, p1, p2, p3)) < 1e-6) return;

      // Bounding box overlap test.
      vec3 tMin = la::min(la::min(p0, p1), la::min(p2, p3));
      vec3 tMax = la::max(la::max(p0, p1), la::max(p2, p3));
      if (tMax.x < shapeBbox.min.x || tMin.x > shapeBbox.max.x ||
          tMax.y < shapeBbox.min.y || tMin.y > shapeBbox.max.y ||
          tMax.z < shapeBbox.min.z || tMin.z > shapeBbox.max.z)
        return;

      Manifold tetHull = Manifold::Hull({p0, p1, p2, p3});
      if (tetHull.IsEmpty()) return;
      Manifold clipped = shape ^ tetHull;
      if (clipped.IsEmpty()) return;

      double hullVol = tetHull.Volume();
      if (clipped.Volume() >= hullVol * (1.0 - 1e-9))
        pieces[i] = tetHull;  // fully interior — keep original hull
      else
        pieces[i] = clipped;
      valid[i] = true;
    });

    // Step 3b: Recover any regions not covered by clipped tets. The bbox
    // corners ensure most coverage, but thin features or degenerate tets
    // can leave gaps. Recover these as additional pieces.
    double step3Vol = 0;
    for (int i = 0; i < numTets; i++)
      if (valid[i]) step3Vol += pieces[i].Volume();

    if (step3Vol < totalVol * 0.999) {
      Manifold covered;
      for (int i = 0; i < numTets; i++) {
        if (!valid[i]) continue;
        covered = covered.IsEmpty() ? pieces[i] : (covered + pieces[i]);
      }
      Manifold uncovered = shape - covered;
      if (!uncovered.IsEmpty()) {
        for (auto& part : uncovered.Decompose()) {
          if (part.IsEmpty() || part.Volume() < 1e-10) continue;
          pieces.push_back(part);
          valid.push_back(true);
          numTets++;
        }
      }
    }

    // Step 4: Priority-queue merge — biggest-first pairwise convex merges.
    // Uses a max-heap keyed by merged volume. Union-find detects stale
    // entries. Only neighbors of newly merged pieces are re-evaluated,
    // giving O(P * D * log P) instead of O(P^2 * H).
    std::vector<double> volumes(numTets, 0.0);
    std::vector<std::vector<vec3>> pieceVerts(numTets);
    for (int i = 0; i < numTets; i++) {
      if (!valid[i]) continue;
      volumes[i] = pieces[i].Volume();
      pieceVerts[i] = ExtractVertices(pieces[i]);
    }

    std::vector<int> parent(numTets);
    for (int i = 0; i < numTets; i++) parent[i] = i;
    auto find = [&](int x) -> int {
      while (parent[x] != x) x = parent[x] = parent[parent[x]];
      return x;
    };

    std::vector<std::unordered_set<int>> neighbors(numTets);
    int origTets = (int)adj.size() / 4;
    for (int i = 0; i < origTets; i++) {
      if (!valid[i]) continue;
      for (int f = 0; f < 4; f++) {
        int j = adj[4 * i + f];
        if (j >= 0 && j < numTets && valid[j] && j != i) {
          neighbors[i].insert(j);
          neighbors[j].insert(i);
        }
      }
    }

    // Step 4a: Cospheric merge — group adjacent tets sharing a circumcenter
    // (computed from unperturbed positions). This undoes the arbitrary tet
    // assignments that DT perturbation introduces for cospheric point sets
    // (e.g. cube vertices).
    {
      std::vector<vec3> circumcenters(origTets);
      for (int i = 0; i < origTets; i++) {
        if (!valid[i]) continue;
        uint32_t ci0 = flatTets[4 * i], ci1 = flatTets[4 * i + 1],
                 ci2 = flatTets[4 * i + 2], ci3 = flatTets[4 * i + 3];
        circumcenters[i] = GetCircumCenter(origPos[ci0], origPos[ci1],
                                           origPos[ci2], origPos[ci3]);
      }

      constexpr double ccEps = 1e-10;
      std::vector<int> ccParent(origTets);
      for (int i = 0; i < origTets; i++) ccParent[i] = i;
      auto ccFind = [&](int x) -> int {
        while (ccParent[x] != x) x = ccParent[x] = ccParent[ccParent[x]];
        return x;
      };

      for (int i = 0; i < origTets; i++) {
        if (!valid[i]) continue;
        for (int n : neighbors[i]) {
          if (n >= origTets || !valid[n]) continue;
          if (la::length(circumcenters[i] - circumcenters[n]) < ccEps) {
            int ri = ccFind(i), rn = ccFind(n);
            if (ri != rn) ccParent[rn] = ri;
          }
        }
      }

      std::unordered_map<int, std::vector<int>> groups;
      for (int i = 0; i < origTets; i++) {
        if (!valid[i]) continue;
        groups[ccFind(i)].push_back(i);
      }

      for (auto& [root, members] : groups) {
        if (members.size() < 2) continue;
        std::vector<vec3> combined;
        double sumVol = 0;
        for (int m : members) {
          combined.insert(combined.end(), pieceVerts[m].begin(),
                          pieceVerts[m].end());
          sumVol += volumes[m];
        }
        Manifold hull = Manifold::Hull(combined);
        if (hull.IsEmpty() || sumVol <= 0.0) continue;
        double hullVol = hull.Volume();
        if (hullVol > sumVol * (1.0 + kMergeTol)) continue;

        pieces[root] = hull;
        pieceVerts[root] = ExtractVertices(hull);
        volumes[root] = hullVol;
        for (int m : members) {
          if (m == root) continue;
          valid[m] = false;
          parent[m] = root;
          for (int nb : neighbors[m]) {
            if (nb != root && valid[nb]) {
              neighbors[root].insert(nb);
              neighbors[nb].erase(m);
              neighbors[nb].insert(root);
            }
          }
          neighbors[m].clear();
        }
      }
    }

    // Step 4b: Priority-queue merge — biggest-first pairwise convex merges.
    struct MergeCandidate {
      double mergedVolume;
      int pieceA, pieceB;
      bool operator<(const MergeCandidate& o) const {
        return mergedVolume < o.mergedVolume;  // max-heap
      }
    };
    std::priority_queue<MergeCandidate> pq;

    // Try merging a pair and push onto PQ if convex.
    auto tryPush = [&](int a, int b) {
      if (a == b || !valid[a] || !valid[b]) return;
      double sumVol = volumes[a] + volumes[b];
      if (sumVol <= 0.0) return;
      std::vector<vec3> combined;
      combined.reserve(pieceVerts[a].size() + pieceVerts[b].size());
      combined.insert(combined.end(), pieceVerts[a].begin(),
                      pieceVerts[a].end());
      combined.insert(combined.end(), pieceVerts[b].begin(),
                      pieceVerts[b].end());
      Manifold hull = Manifold::Hull(combined);
      if (hull.IsEmpty()) return;
      double hullVol = hull.Volume();
      if (hullVol <= sumVol * (1.0 + kMergeTol))
        pq.push({hullVol, a, b});
    };

    // Seed PQ with all adjacent valid pairs.
    for (int i = 0; i < numTets; i++) {
      if (!valid[i]) continue;
      for (int n : neighbors[i]) {
        if (n > i && valid[n]) tryPush(i, n);
      }
    }

    // Drain PQ — biggest merges first.
    while (!pq.empty()) {
      auto [vol, a, b] = pq.top();
      pq.pop();
      int ra = find(a), rb = find(b);
      if (ra == rb || !valid[ra] || !valid[rb]) continue;
      if (ra != a || rb != b) continue;  // stale entry

      // Re-verify hull (vertices may have changed from prior merges).
      double sumVol = volumes[ra] + volumes[rb];
      std::vector<vec3> combined;
      combined.reserve(pieceVerts[ra].size() + pieceVerts[rb].size());
      combined.insert(combined.end(), pieceVerts[ra].begin(),
                      pieceVerts[ra].end());
      combined.insert(combined.end(), pieceVerts[rb].begin(),
                      pieceVerts[rb].end());
      Manifold hull = Manifold::Hull(combined);
      if (hull.IsEmpty()) continue;
      double hullVol = hull.Volume();
      if (sumVol <= 0.0 || hullVol > sumVol * (1.0 + kMergeTol)) continue;

      // Commit merge: ra absorbs rb.
      pieces[ra] = hull;
      pieceVerts[ra] = ExtractVertices(hull);
      volumes[ra] = hullVol;
      valid[rb] = false;
      parent[rb] = ra;

      // Inherit rb's neighbors.
      for (int nb : neighbors[rb]) {
        if (nb != ra && valid[nb]) {
          neighbors[ra].insert(nb);
          neighbors[nb].erase(rb);
          neighbors[nb].insert(ra);
        }
      }
      neighbors[rb].clear();

      // Push new merge candidates for all neighbors of the merged piece.
      for (int nb : neighbors[ra]) {
        if (valid[nb]) tryPush(ra, nb);
      }
    }

    // Step 6: Split remaining non-convex pieces by cutting through reflex
    // edges with dihedral bisector planes. Keep iterating until all pieces
    // are convex or no further progress is made.
    std::vector<Manifold> pending;
    for (int i = 0; i < numTets; i++) {
      if (!valid[i] || pieces[i].IsEmpty()) continue;
      if (pieces[i].Volume() < 1e-10) continue;  // skip degenerate slivers
      auto impl = pieces[i].GetCsgLeafNode().GetImpl();
      if (impl->IsConvex())
        outputs.push_back(pieces[i]);
      else
        pending.push_back(pieces[i]);
    }

    while (!pending.empty()) {
      std::vector<Manifold> nextPending;
      bool anyProgress = false;
      for (auto& piece : pending) {
        auto pImpl = piece.GetCsgLeafNode().GetImpl();
        if (pImpl->IsConvex()) {
          outputs.push_back(piece);
          continue;
        }
        bool didSplit = false;
        const size_t nbEdges = pImpl->halfedge_.size();
        for (size_t idx = 0; idx < nbEdges; idx++) {
          auto edge = pImpl->halfedge_[idx];
          if (!edge.IsForward()) continue;
          vec3 n0 = pImpl->faceNormal_[idx / 3];
          vec3 n1 = pImpl->faceNormal_[edge.pairedHalfedge / 3];
          vec3 edgeVec =
              pImpl->vertPos_[edge.endVert] - pImpl->vertPos_[edge.startVert];
          if (la::dot(edgeVec, la::cross(n0, n1)) >= 0) continue;

          vec3 bisector = la::normalize(n0 + n1);
          vec3 planeNormal = la::cross(la::normalize(edgeVec), bisector);
          double pnLen = la::length(planeNormal);
          if (pnLen < 1e-10) continue;
          planeNormal /= pnLen;
          vec3 edgeMid = (pImpl->vertPos_[edge.startVert] +
                          pImpl->vertPos_[edge.endVert]) *
                         0.5;
          double originOffset = la::dot(planeNormal, edgeMid);

          auto [a, b] = piece.SplitByPlane(planeNormal, originOffset);
          if (!a.IsEmpty() && !b.IsEmpty() && a.Volume() > 1e-10 &&
              b.Volume() > 1e-10) {
            auto aImpl = a.GetCsgLeafNode().GetImpl();
            auto bImpl = b.GetCsgLeafNode().GetImpl();
            if (aImpl->IsConvex())
              outputs.push_back(a);
            else
              nextPending.push_back(a);
            if (bImpl->IsConvex())
              outputs.push_back(b);
            else
              nextPending.push_back(b);
            didSplit = true;
            anyProgress = true;
            break;
          }
        }
        if (!didSplit) outputs.push_back(piece);
      }
      pending = std::move(nextPending);
      if (!anyProgress) {
        for (auto& p : pending) outputs.push_back(p);
        break;
      }
    }
  }

  // Post-process: fix remaining non-convex pieces.
  // 1. Simplify to remove numerical noise vertices (fixes barely-NC pieces)
  // 2. Decompose self-intersecting pieces (genus < 0) into convex components
  // 3. Filter degenerate slivers
  std::vector<Manifold> postProcessed;
  for (auto& p : outputs) {
    if (p.IsEmpty() || p.Volume() <= 0) continue;
    auto impl = p.GetCsgLeafNode().GetImpl();
    if (impl->IsConvex()) {
      postProcessed.push_back(p);
      continue;
    }

    // Try simplify first — removes noise vertices from boolean operations.
    double tol = p.GetTolerance();
    Manifold simplified = p.AsOriginal().Simplify(tol);
    auto simpImpl = simplified.GetCsgLeafNode().GetImpl();
    if (simpImpl->IsConvex()) {
      postProcessed.push_back(simplified);
      continue;
    }

    // Decompose self-intersecting pieces (genus < 0) into components.
    // Recovery booleans can create overlapping convex regions that
    // Decompose() cleanly separates.
    auto parts = simplified.Decompose();
    if ((int)parts.size() > 1) {
      bool allConvex = true;
      for (auto& part : parts) {
        if (part.IsEmpty() || part.Volume() <= 0) continue;
        auto partImpl = part.GetCsgLeafNode().GetImpl();
        if (!partImpl->IsConvex()) allConvex = false;
      }
      if (allConvex) {
        for (auto& part : parts)
          if (!part.IsEmpty() && part.Volume() > 0)
            postProcessed.push_back(part);
        continue;
      }
    }

    // Still non-convex — keep as-is.
    postProcessed.push_back(simplified);
  }

  // Filter degenerate slivers. Any piece with volume < 1e-6 of the total
  // is numerical noise from tet clipping or boolean operations.
  double totalOutputVol = 0;
  for (auto& p : postProcessed) totalOutputVol += p.Volume();
  double minPieceVol = std::max(1e-10, totalOutputVol * 1e-6);
  std::vector<Manifold> filtered;
  for (auto& p : postProcessed)
    if (p.Volume() > minPieceVol) filtered.push_back(p);
  return filtered;
}

}  // namespace manifold
