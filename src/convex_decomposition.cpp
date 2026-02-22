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
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "csg_tree.h"
#include "impl.h"
#include "iters.h"
#include "parallel.h"

namespace manifold {

namespace {

// Face indices for each of the 4 faces of a tetrahedron.
const int kTetFaces[4][3] = {{2, 1, 0}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};

// Deterministic perturbation for DT robustness. Used only for computing
// DT connectivity — all output uses original unperturbed positions.
inline double DeterministicEps(size_t index, int component) {
  constexpr double eps = 1e-10;
  uint64_t h = index * 2654435761ULL + component * 40503ULL;
  h = (h ^ (h >> 16)) * 0x45d9f3bULL;
  return -eps + 2.0 * ((h & 0xFFFFFF) / double(0xFFFFFF)) * eps;
}

vec3 GetCircumCenter(const vec3& p0, const vec3& p1, const vec3& p2,
                     const vec3& p3) {
  vec3 b = p1 - p0;
  vec3 c = p2 - p0;
  vec3 d = p3 - p0;
  double det =
      2.0 * (b.x * (c.y * d.z - c.z * d.y) - b.y * (c.x * d.z - c.z * d.x) +
             b.z * (c.x * d.y - c.y * d.x));
  if (det == 0.0) return p0;
  vec3 v = la::cross(c, d) * la::dot(b, b) + la::cross(d, b) * la::dot(c, c) +
           la::cross(b, c) * la::dot(d, d);
  return p0 + v / det;
}

double TetQuality(const vec3& p0, const vec3& p1, const vec3& p2,
                  const vec3& p3) {
  vec3 d0 = p1 - p0, d1 = p2 - p0, d2 = p3 - p0;
  vec3 d3 = p2 - p1, d4 = p3 - p2, d5 = p1 - p3;
  double s0 = la::length(d0), s1 = la::length(d1), s2 = la::length(d2);
  double s3 = la::length(d3), s4 = la::length(d4), s5 = la::length(d5);
  double ms = (s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4 + s5 * s5) / 6.0;
  double rms = std::sqrt(ms);
  if (rms == 0.0) return 0.0;
  double s = 12.0 / std::sqrt(2.0);
  double vol = la::dot(d0, la::cross(d1, d2)) / 6.0;
  return s * vol / (rms * rms * rms);
}

struct Edge {
  int id0, id1, tetNr, faceNr;
  bool operator<(const Edge& o) const {
    return id0 < o.id0 || (id0 == o.id0 && id1 < o.id1);
  }
  bool operator==(const Edge& o) const { return id0 == o.id0 && id1 == o.id1; }
};

// Incremental Delaunay tetrahedralization.
// Vertices should be perturbed before calling to avoid degeneracies.
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

  // Safeguard: cap total tet storage to prevent OOM
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

    // Safeguard: abort if tet count is excessive
    if ((size_t)tetIds.size() / 4 > maxTotalTets) {
#ifdef MANIFOLD_DEBUG
      printf(
          "ConvexDecomposition DT: tet limit reached (%zu), "
          "skipping remaining %d points\n",
          tetIds.size() / 4, firstBig - i);
#endif
      break;
    }

    // Find non-deleted tet
    int tetNr = 0;
    while (tetIds[4 * tetNr] < 0) tetNr++;

    // Walk to containing tet
    tetMark++;
    bool found = false;
    int walkSteps = 0;
    const int maxWalkSteps = (int)tetIds.size() / 4 + 100;
    while (!found) {
      if (tetNr < 0 || tetMarks[tetNr] == tetMark) break;
      if (++walkSteps > maxWalkSteps) break;  // prevent infinite walk
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

    // Find violating tets — strict < handles cospheric points by NOT
    // flipping them, allowing multiple tets to share a circumcenter.
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

}  // anonymous namespace

/**
 * Decompose into approximately convex pieces.
 *
 * Uses unconstrained Delaunay tetrahedralization of mesh vertices, clips each
 * tet against the mesh, then greedily merges adjacent convex pieces by hull.
 * Follows the same Impl method pattern as Minkowski.
 */
std::vector<Manifold> Manifold::Impl::ConvexDecomposition(int maxClusterSize,
                                                          int maxDepth) const {
  std::vector<Manifold> outputs;

  // Wrap this Impl as a Manifold for boolean operations
  auto thisImpl = std::make_shared<Impl>(*this);
  Manifold curShape(std::make_shared<CsgLeafNode>(thisImpl));

  if (curShape.IsEmpty()) return outputs;

  // Fast convexity check using halfedge dihedral angles (same as Minkowski)
  if (IsConvex()) {
    outputs.push_back(curShape.Hull());
    return outputs;
  }

  // Decompose into connected components
  std::vector<Manifold> shapes = curShape.Decompose();
  if (shapes.empty()) {
    outputs.push_back(curShape);
    return outputs;
  }

  for (auto& shape : shapes) {
    if (shape.IsEmpty()) continue;

    // Per-component convexity check via halfedge dihedral angles
    auto shapeImpl = shape.GetCsgLeafNode().GetImpl();
    if (shapeImpl->IsConvex()) {
      outputs.push_back(shape.Hull());
      continue;
    }

    MeshGL64 meshGL = shape.GetMeshGL64();
    size_t numVerts = meshGL.vertProperties.size() / meshGL.numProp;

    std::vector<double> verts;
    verts.reserve(numVerts * 3);
    for (size_t i = 0; i < numVerts; i++) {
      verts.push_back(meshGL.vertProperties[i * meshGL.numProp + 0]);
      verts.push_back(meshGL.vertProperties[i * meshGL.numProp + 1]);
      verts.push_back(meshGL.vertProperties[i * meshGL.numProp + 2]);
    }

    // Step 1: Unconstrained DT
    if (numVerts < 4) {
      outputs.push_back(shape);
      continue;
    }

    // Perturb vertices for DT robustness — output uses original positions
    std::vector<vec3> tetVerts;
    tetVerts.reserve(numVerts + 4);
    for (size_t i = 0; i < numVerts; i++)
      tetVerts.push_back(vec3(verts[i * 3 + 0] + DeterministicEps(i, 0),
                              verts[i * 3 + 1] + DeterministicEps(i, 1),
                              verts[i * 3 + 2] + DeterministicEps(i, 2)));

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

    // Use quality threshold > 0 to exclude degenerate tets that cause
    // assertion failures in Hull() with MANIFOLD_DEBUG enabled.
    std::vector<uint32_t> flatTets = CreateTetIds(tetVerts, 0.001);
    int numTets = (int)flatTets.size() / 4;
    if (numTets == 0) {
      outputs.push_back(shape);
      continue;
    }

    // Step 2: Clip tets against mesh
    auto adj = BuildTetAdjacency(flatTets);
    double totalVol = shape.Volume();
    double minPieceVol = totalVol * 0.00001;
    auto shapeBbox = shape.BoundingBox();

    std::vector<Manifold> pieces(numTets);
    std::vector<bool> valid(numTets, false);

    for_each_n(autoPolicy(numTets, 16), countAt(0), numTets, [&](int i) {
      uint32_t i0 = flatTets[4 * i], i1 = flatTets[4 * i + 1],
               i2 = flatTets[4 * i + 2], i3 = flatTets[4 * i + 3];
      vec3 p0(verts[i0 * 3], verts[i0 * 3 + 1], verts[i0 * 3 + 2]);
      vec3 p1(verts[i1 * 3], verts[i1 * 3 + 1], verts[i1 * 3 + 2]);
      vec3 p2(verts[i2 * 3], verts[i2 * 3 + 1], verts[i2 * 3 + 2]);
      vec3 p3(verts[i3 * 3], verts[i3 * 3 + 1], verts[i3 * 3 + 2]);

      vec3 tMin = la::min(la::min(p0, p1), la::min(p2, p3));
      vec3 tMax = la::max(la::max(p0, p1), la::max(p2, p3));
      if (tMax.x < shapeBbox.min.x || tMin.x > shapeBbox.max.x ||
          tMax.y < shapeBbox.min.y || tMin.y > shapeBbox.max.y ||
          tMax.z < shapeBbox.min.z || tMin.z > shapeBbox.max.z)
        return;

      Manifold tetHull = Manifold::Hull({p0, p1, p2, p3});
      if (tetHull.IsEmpty()) return;
      Manifold clipped = shape ^ tetHull;
      if (clipped.IsEmpty() || clipped.Volume() < minPieceVol) return;
      pieces[i] = clipped;
      valid[i] = true;
    });

    // Recover uncovered regions
    double step2Vol = 0;
    for (int i = 0; i < numTets; i++)
      if (valid[i]) step2Vol += pieces[i].Volume();

    if (std::abs(totalVol - step2Vol) > totalVol * 0.001) {
      Manifold covered;
      for (int i = 0; i < numTets; i++) {
        if (!valid[i]) continue;
        covered = covered.IsEmpty() ? pieces[i] : (covered + pieces[i]);
      }
      Manifold uncovered = shape - covered;
      if (!uncovered.IsEmpty() && uncovered.Volume() > minPieceVol) {
        for (auto& part : uncovered.Decompose()) {
          if (part.IsEmpty() || part.Volume() < minPieceVol) continue;
          pieces.push_back(part);
          valid.push_back(true);
          numTets++;
        }
      }
    }

    // Step 3: Greedy merge
    const double mergeTol = 0.001;
    std::vector<double> volumes(numTets, 0.0);
    std::vector<std::vector<vec3>> pieceVerts(numTets);
    for (int i = 0; i < numTets; i++) {
      if (!valid[i]) continue;
      volumes[i] = pieces[i].Volume();
      pieceVerts[i] = ExtractVertices(pieces[i]);
    }

    std::vector<int> parent(numTets);
    for (int i = 0; i < numTets; i++) parent[i] = i;
    std::function<int(int)> find = [&](int x) -> int {
      return parent[x] == x ? x : (parent[x] = find(parent[x]));
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

    // Phase 0: Cospheric merge — using ORIGINAL (unperturbed) positions,
    // compute circumcenters and merge groups of adjacent tets that share one.
    // This resolves ambiguity from cospheric vertices without perturbation
    // artifacts in the output. Groups all connected cospheric tets and merges
    // each group at once (handles 3+ tets sharing a circumcenter).
    {
      std::vector<vec3> circumcenters(origTets);
      for (int i = 0; i < origTets; i++) {
        if (!valid[i]) continue;
        uint32_t i0 = flatTets[4 * i], i1 = flatTets[4 * i + 1],
                 i2 = flatTets[4 * i + 2], i3 = flatTets[4 * i + 3];
        vec3 p0(verts[i0 * 3], verts[i0 * 3 + 1], verts[i0 * 3 + 2]);
        vec3 p1(verts[i1 * 3], verts[i1 * 3 + 1], verts[i1 * 3 + 2]);
        vec3 p2(verts[i2 * 3], verts[i2 * 3 + 1], verts[i2 * 3 + 2]);
        vec3 p3(verts[i3 * 3], verts[i3 * 3 + 1], verts[i3 * 3 + 2]);
        circumcenters[i] = GetCircumCenter(p0, p1, p2, p3);
      }

      // Union-find to group adjacent tets with matching circumcenters
      constexpr double ccEps = 1e-10;
      std::vector<int> ccParent(origTets);
      for (int i = 0; i < origTets; i++) ccParent[i] = i;
      std::function<int(int)> ccFind = [&](int x) -> int {
        return ccParent[x] == x ? x : (ccParent[x] = ccFind(ccParent[x]));
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

      // Collect groups and merge each
      std::unordered_map<int, std::vector<int>> groups;
      for (int i = 0; i < origTets; i++) {
        if (!valid[i]) continue;
        groups[ccFind(i)].push_back(i);
      }

      for (auto& [root, members] : groups) {
        if (members.size() < 2) continue;

        // Merge all members into root
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
        if (std::abs(hullVol - sumVol) > sumVol * 0.01) continue;

        // Apply merge: keep root, invalidate others
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

    auto tryMerge = [&](int root, int other) -> bool {
      if (root == other || !valid[root] || !valid[other]) return false;
      double sumVol = volumes[root] + volumes[other];
      std::vector<vec3> combined;
      combined.reserve(pieceVerts[root].size() + pieceVerts[other].size());
      combined.insert(combined.end(), pieceVerts[root].begin(),
                      pieceVerts[root].end());
      combined.insert(combined.end(), pieceVerts[other].begin(),
                      pieceVerts[other].end());
      Manifold hull = Manifold::Hull(combined);
      if (hull.IsEmpty()) return false;
      double hullVol = hull.Volume();
      if (sumVol > 0.0 && std::abs(hullVol - sumVol) < sumVol * mergeTol) {
        pieces[root] = hull;
        pieceVerts[root] = ExtractVertices(hull);
        volumes[root] = hullVol;
        valid[other] = false;
        parent[other] = root;
        for (int n : neighbors[other]) {
          if (n != root && valid[n]) {
            neighbors[root].insert(n);
            neighbors[n].erase(other);
            neighbors[n].insert(root);
          }
        }
        neighbors[other].clear();
        return true;
      }
      return false;
    };

    auto applyMerge = [&](int root, const std::vector<int>& others,
                          const Manifold& hull) {
      pieces[root] = hull;
      pieceVerts[root] = ExtractVertices(hull);
      volumes[root] = hull.Volume();
      for (int o : others) {
        valid[o] = false;
        parent[o] = root;
        for (int nb : neighbors[o]) {
          if (nb != root && valid[nb]) {
            neighbors[root].insert(nb);
            neighbors[nb].erase(o);
            neighbors[nb].insert(root);
          }
        }
        neighbors[o].clear();
      }
    };

    // Phase A: greedy pair merges via DT adjacency
    bool merged = true;
    while (merged) {
      merged = false;
      for (int i = 0; i < numTets; i++) {
        if (!valid[i]) continue;
        std::vector<int> nbrs(neighbors[i].begin(), neighbors[i].end());
        for (int n : nbrs) {
          int rn = find(n);
          if (rn == i || !valid[rn]) continue;
          if (tryMerge(i, rn)) merged = true;
        }
      }
    }

    // Cluster merges (tightest-first)
    struct AABB {
      vec3 mn, mx;
    };
    std::vector<AABB> bboxes(numTets);
    std::vector<int> validIds;

    for (int clusterSize = 2; clusterSize <= maxClusterSize; clusterSize++) {
      validIds.clear();
      for (int i = 0; i < numTets; i++)
        if (valid[i]) validIds.push_back(i);
      constexpr double bboxEps = 1e-10;
      for (int i : validIds) {
        auto bb = pieces[i].BoundingBox();
        bboxes[i] = {
            vec3(bb.min.x - bboxEps, bb.min.y - bboxEps, bb.min.z - bboxEps),
            vec3(bb.max.x + bboxEps, bb.max.y + bboxEps, bb.max.z + bboxEps)};
      }

      merged = true;
      while (merged && (int)validIds.size() >= clusterSize) {
        merged = false;
        int bestRoot = -1;
        std::vector<int> bestOthers;
        double bestRatio = 1e18;
        Manifold bestHull;

        for (int i : validIds) {
          if (!valid[i]) continue;
          std::vector<int> nbrs;
          if (clusterSize == 2) {
            for (int j : validIds) {
              if (j == i || !valid[j]) continue;
              if (bboxes[i].mx.x < bboxes[j].mn.x ||
                  bboxes[j].mx.x < bboxes[i].mn.x ||
                  bboxes[i].mx.y < bboxes[j].mn.y ||
                  bboxes[j].mx.y < bboxes[i].mn.y ||
                  bboxes[i].mx.z < bboxes[j].mn.z ||
                  bboxes[j].mx.z < bboxes[i].mn.z)
                continue;
              nbrs.push_back(j);
            }
          } else {
            for (int n : neighbors[i])
              if (valid[n]) nbrs.push_back(n);
          }
          if ((int)nbrs.size() < clusterSize - 1) continue;

          int pick = clusterSize - 1;
          std::vector<int> idx(pick);
          for (int p = 0; p < pick; p++) idx[p] = p;
          while (true) {
            std::vector<vec3> combined = pieceVerts[i];
            double sumVol = volumes[i];
            std::vector<int> others;
            for (int p = 0; p < pick; p++) {
              int j = nbrs[idx[p]];
              combined.insert(combined.end(), pieceVerts[j].begin(),
                              pieceVerts[j].end());
              sumVol += volumes[j];
              others.push_back(j);
            }
            Manifold hull = Manifold::Hull(combined);
            if (!hull.IsEmpty() && sumVol > 0.0) {
              double ratio = hull.Volume() / sumVol;
              if (ratio < 1.0 + mergeTol && ratio < bestRatio) {
                bestRatio = ratio;
                bestRoot = i;
                bestOthers = others;
                bestHull = hull;
              }
            }
            int p = pick - 1;
            while (p >= 0 && idx[p] == (int)nbrs.size() - pick + p) p--;
            if (p < 0) break;
            idx[p]++;
            for (int q = p + 1; q < pick; q++) idx[q] = idx[q - 1] + 1;
          }
        }
        if (bestRoot >= 0) {
          applyMerge(bestRoot, bestOthers, bestHull);
          auto bb = bestHull.BoundingBox();
          bboxes[bestRoot] = {vec3(bb.min.x, bb.min.y, bb.min.z),
                              vec3(bb.max.x, bb.max.y, bb.max.z)};
          merged = true;
          validIds.clear();
          for (int i = 0; i < numTets; i++)
            if (valid[i]) validIds.push_back(i);
        }
      }
    }

    // Step 4: Collect results — recursively decompose non-convex pieces
    for (int i = 0; i < numTets; i++) {
      if (!valid[i] || pieces[i].IsEmpty()) continue;
      auto pieceImpl = pieces[i].GetCsgLeafNode().GetImpl();
      if (pieceImpl->IsConvex()) {
        outputs.push_back(pieces[i]);
        continue;
      }
      // Non-convex piece: try recursive decomposition if depth allows
      if (maxDepth > 0 && pieces[i].NumVert() >= 4) {
        auto subPieces =
            pieceImpl->ConvexDecomposition(maxClusterSize, maxDepth - 1);
        outputs.insert(outputs.end(), subPieces.begin(), subPieces.end());
      } else {
        // Max depth reached: keep as-is
        outputs.push_back(pieces[i]);
      }
    }
  }

  return outputs;
}

}  // namespace manifold
