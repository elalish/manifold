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
  vec3 b = p1 - p0, c = p2 - p0, d = p3 - p0;
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
 * Algorithm: DT of mesh vertices → clip tets against mesh → recover
 * uncovered regions → cospheric merge → greedy hull merge → reflex
 * edge splitting of remaining non-convex pieces.
 */
std::vector<Manifold> Manifold::Impl::ConvexDecomposition(int maxClusterSize,
                                                          int maxDepth) const {
  std::vector<Manifold> outputs;

  auto thisImpl = std::make_shared<Impl>(*this);
  Manifold curShape(std::make_shared<CsgLeafNode>(thisImpl));

  if (curShape.IsEmpty()) return outputs;

  if (IsConvex()) {
    outputs.push_back(curShape.Hull());
    return outputs;
  }

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

    constexpr double kMergeTol = 1e-8;

    // Step 0: Greedy surface hull carving. Only for shapes with many vertices
    // (curved surfaces). Simple shapes are handled optimally by DT+merge.
    if (shapeImpl->NumVert() < 100)
      goto skip_carving;  // NOLINT Starting from each unclaimed
    // vertex, incrementally grow a patch by adding neighbors one at a time,
    // checking that the hull of the patch stays inside the shape after each
    // addition. When the patch can't grow further, emit it as a convex piece
    // and subtract from the shape. This pre-shaves large convex chunks.
    {
      auto sImpl = shape.GetCsgLeafNode().GetImpl();
      const auto& he = sImpl->halfedge_;
      const auto& vp = sImpl->vertPos_;
      size_t nv = vp.size();

      // Build full vertex adjacency from halfedges
      std::vector<std::vector<int>> adj(nv);
      for (size_t idx = 0; idx < he.size(); idx++) {
        auto edge = he[idx];
        if (!edge.IsForward()) continue;
        adj[edge.startVert].push_back(edge.endVert);
        adj[edge.endVert].push_back(edge.startVert);
      }

      std::vector<bool> claimed(nv, false);
      bool anyCarved = true;
      while (anyCarved) {
        anyCarved = false;
        for (size_t seed = 0; seed < nv; seed++) {
          if (claimed[seed]) continue;

          // Start patch with seed + its first 3 unclaimed neighbors
          // (need at least 4 points for a valid hull)
          std::vector<int> patch = {(int)seed};
          std::vector<bool> inPatch(nv, false);
          inPatch[seed] = true;

          // Collect candidate neighbors (BFS frontier)
          std::vector<int> frontier;
          for (int nb : adj[seed]) {
            if (!claimed[nb] && !inPatch[nb]) frontier.push_back(nb);
          }

          // Grow patch greedily until hull would exit the shape
          Manifold lastGoodHull;
          std::vector<int> lastGoodPatch;

          while (!frontier.empty()) {
            // Try adding the first frontier vertex
            int candidate = frontier.back();
            frontier.pop_back();
            if (inPatch[candidate] || claimed[candidate]) continue;

            // Tentatively add to patch
            patch.push_back(candidate);
            inPatch[candidate] = true;

            if (patch.size() >= 4) {
              // Check hull containment
              std::vector<vec3> pts;
              for (int v : patch) pts.push_back(vp[v]);
              Manifold hull = Manifold::Hull(pts);
              if (!hull.IsEmpty()) {
                double hullVol = hull.Volume();
                Manifold clipped = shape ^ hull;
                if (!clipped.IsEmpty() &&
                    clipped.Volume() >= hullVol * (1.0 - kMergeTol)) {
                  // Hull fits! Save as last good state
                  lastGoodHull = hull;
                  lastGoodPatch = patch;
                  // Add this vertex's neighbors to frontier
                  for (int nb : adj[candidate]) {
                    if (!claimed[nb] && !inPatch[nb]) frontier.push_back(nb);
                  }
                  continue;  // keep growing
                }
              }
            } else {
              // Not enough for hull yet, add neighbors and continue
              for (int nb : adj[candidate]) {
                if (!claimed[nb] && !inPatch[nb]) frontier.push_back(nb);
              }
              continue;
            }

            // Hull doesn't fit — remove candidate and stop trying this one
            patch.pop_back();
            inPatch[candidate] = false;
          }

          // Emit the largest valid patch (only if large enough to be worth
          // carving — small patches are better handled by the DT+merge step)
          if (!lastGoodHull.IsEmpty() && lastGoodPatch.size() >= 16) {
            outputs.push_back(lastGoodHull);
            for (int v : lastGoodPatch) claimed[v] = true;
            shape = shape - lastGoodHull;
            if (shape.IsEmpty()) break;
            sImpl = shape.GetCsgLeafNode().GetImpl();
            anyCarved = true;
            break;  // restart scan with updated shape
          }
        }
        if (shape.IsEmpty()) break;
      }

      if (shape.IsEmpty()) continue;
      sImpl = shape.GetCsgLeafNode().GetImpl();
      if (sImpl->IsConvex()) {
        outputs.push_back(shape.Hull());
        continue;
      }
    }

  skip_carving:
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

    // Inject bounding-sphere centers of connected components as Steiner
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

    std::vector<uint32_t> flatTets = CreateTetIds(tetVerts, 0.0);
    int numTets = (int)flatTets.size() / 4;
    if (numTets == 0) {
      outputs.push_back(shape);
      continue;
    }

    // Step 2: Clip tets against mesh (parallel)
    // Build a set of mesh surface triangles for fast lookup. If a tet has
    // a face matching a mesh triangle and the 4th vertex is on the interior
    // side, the tet is entirely inside — skip the expensive boolean.
    auto adj = BuildTetAdjacency(flatTets);
    double totalVol = shape.Volume();
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

      if (std::abs(TetQuality(p0, p1, p2, p3)) < 0.001) return;

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
      // If clipped volume ≈ hull volume, the tet is fully inside the mesh.
      // Use the hull directly to preserve original DT vertex positions
      // (the boolean intersection introduces new vertices at boundaries).
      double hullVol = tetHull.Volume();
      if (clipped.Volume() >= hullVol * (1.0 - 1e-9)) {
        pieces[i] = tetHull;
      } else {
        pieces[i] = clipped;
      }
      valid[i] = true;
    });

    // Step 2b: Recover uncovered regions
    double step2Vol = 0;
    for (int i = 0; i < numTets; i++)
      if (valid[i]) step2Vol += pieces[i].Volume();

    if (step2Vol < totalVol * 0.999) {
      Manifold covered;
      for (int i = 0; i < numTets; i++) {
        if (!valid[i]) continue;
        covered = covered.IsEmpty() ? pieces[i] : (covered + pieces[i]);
      }
      Manifold uncovered = shape - covered;
      if (!uncovered.IsEmpty()) {
        for (auto& part : uncovered.Decompose()) {
          if (part.IsEmpty()) continue;
          pieces.push_back(part);
          valid.push_back(true);
          numTets++;
        }
      }
    }

    // Step 3: Greedy merge
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

    // Phase 0: Cospheric merge — group adjacent interior tets sharing a
    // circumcenter (from original unperturbed positions) and merge by hull.
    {
      std::vector<vec3> circumcenters(origTets);
      for (int i = 0; i < origTets; i++) {
        if (!valid[i]) continue;
        uint32_t ci0 = flatTets[4 * i], ci1 = flatTets[4 * i + 1],
                 ci2 = flatTets[4 * i + 2], ci3 = flatTets[4 * i + 3];
        circumcenters[i] = GetCircumCenter(
            vec3(verts[ci0 * 3], verts[ci0 * 3 + 1], verts[ci0 * 3 + 2]),
            vec3(verts[ci1 * 3], verts[ci1 * 3 + 1], verts[ci1 * 3 + 2]),
            vec3(verts[ci2 * 3], verts[ci2 * 3 + 1], verts[ci2 * 3 + 2]),
            vec3(verts[ci3 * 3], verts[ci3 * 3 + 1], verts[ci3 * 3 + 2]));
      }

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

    // Phase A: greedy pair merges via adjacency
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
      if (sumVol > 0.0 && hullVol <= sumVol * (1.0 + kMergeTol)) {
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
              if (ratio <= 1.0 + kMergeTol && ratio < bestRatio) {
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

    // Step 4: Collect convex pieces. Split non-convex pieces by cutting
    // through reflex edges with dihedral bisector planes.
    std::vector<Manifold> pending;
    for (int i = 0; i < numTets; i++) {
      if (!valid[i] || pieces[i].IsEmpty()) continue;
      auto impl = pieces[i].GetCsgLeafNode().GetImpl();
      if (impl->IsConvex())
        outputs.push_back(pieces[i]);
      else
        pending.push_back(pieces[i]);
    }

    for (int iter = 0; iter < maxDepth * 4 && !pending.empty(); iter++) {
      std::vector<Manifold> nextPending;
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
          if (!a.IsEmpty() && !b.IsEmpty()) {
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
            break;
          }
        }
        if (!didSplit) outputs.push_back(piece);
      }
      pending = std::move(nextPending);
    }
    for (auto& p : pending) outputs.push_back(p);
  }

  return outputs;
}

/**
 * Pure carve-only convex decomposition — no DT. Iteratively grows the
 * largest convex surface patch, emits its hull, subtracts, and repeats.
 */
std::vector<Manifold> Manifold::Impl::ConvexDecompositionCarveOnly() const {
  constexpr double kTol = 1e-8;
  std::vector<Manifold> outputs;

  auto thisImpl = std::make_shared<Impl>(*this);
  Manifold shape(std::make_shared<CsgLeafNode>(thisImpl));

  if (shape.IsEmpty() || shape.Volume() < 1e-12) return outputs;
  if (IsConvex()) {
    outputs.push_back(shape.Hull());
    return outputs;
  }

  for (int iteration = 0; iteration < 1000 && !shape.IsEmpty(); iteration++) {
    auto sImpl = shape.GetCsgLeafNode().GetImpl();
    if (sImpl->IsConvex()) {
      outputs.push_back(shape.Hull());
      break;
    }

    const auto& he = sImpl->halfedge_;
    const auto& vp = sImpl->vertPos_;
    size_t nv = vp.size();
    if (nv < 4) {
      outputs.push_back(shape);
      break;
    }

    // Build vertex adjacency
    std::vector<std::vector<int>> adj(nv);
    for (size_t idx = 0; idx < he.size(); idx++) {
      auto edge = he[idx];
      if (!edge.IsForward()) continue;
      adj[edge.startVert].push_back(edge.endVert);
      adj[edge.endVert].push_back(edge.startVert);
    }

    // Find a valid convex hull — use first seed that produces a patch
    Manifold bestHull;
    for (size_t seed = 0; seed < nv; seed++) {
      std::vector<int> patch = {(int)seed};
      std::vector<bool> inPatch(nv, false);
      inPatch[seed] = true;

      std::vector<int> frontier;
      for (int nb : adj[seed])
        if (!inPatch[nb]) frontier.push_back(nb);

      Manifold lastGoodHull;

      while (!frontier.empty()) {
        int c = frontier.back();
        frontier.pop_back();
        if (inPatch[c]) continue;
        patch.push_back(c);
        inPatch[c] = true;

        if (patch.size() >= 4) {
          std::vector<vec3> pts;
          for (int v : patch) pts.push_back(vp[v]);
          Manifold hull = Manifold::Hull(pts);
          if (!hull.IsEmpty()) {
            double hv = hull.Volume();
            Manifold clipped = shape ^ hull;
            if (!clipped.IsEmpty() && clipped.Volume() >= hv * (1.0 - kTol)) {
              lastGoodHull = hull;
              for (int nb : adj[c])
                if (!inPatch[nb]) frontier.push_back(nb);
              continue;
            }
          }
        } else {
          for (int nb : adj[c])
            if (!inPatch[nb]) frontier.push_back(nb);
          continue;
        }
        patch.pop_back();
        inPatch[c] = false;
      }

      if (!lastGoodHull.IsEmpty()) {
        bestHull = lastGoodHull;
        break;  // use first valid patch, don't search all seeds
      }
    }

    if (bestHull.IsEmpty()) {
      // Can't carve — iteratively split all reflex edges, then recurse
      // on the resulting pieces (handles multiple reflex edges per piece).
      std::vector<Manifold> work = {shape};
      for (int splitIter = 0; splitIter < 100 && !work.empty(); splitIter++) {
        std::vector<Manifold> next;
        bool anySplit = false;
        for (auto& piece : work) {
          if (piece.Volume() < 1e-10) continue;
          auto pImpl = piece.GetCsgLeafNode().GetImpl();
          if (pImpl->IsConvex()) {
            outputs.push_back(piece);
            continue;
          }
          bool split = false;
          const auto& heS = pImpl->halfedge_;
          const auto& fnS = pImpl->faceNormal_;
          const auto& vpS = pImpl->vertPos_;
          for (size_t idx = 0; idx < heS.size(); idx++) {
            auto edge = heS[idx];
            if (!edge.IsForward()) continue;
            vec3 n0 = fnS[idx / 3];
            vec3 n1 = fnS[edge.pairedHalfedge / 3];
            vec3 edgeVec = vpS[edge.endVert] - vpS[edge.startVert];
            if (la::dot(edgeVec, la::cross(n0, n1)) >= 0) continue;
            vec3 bisector = la::normalize(n0 + n1);
            vec3 planeNormal = la::cross(la::normalize(edgeVec), bisector);
            double pnLen = la::length(planeNormal);
            if (pnLen < 1e-10) continue;
            planeNormal /= pnLen;
            vec3 edgeMid = (vpS[edge.startVert] + vpS[edge.endVert]) * 0.5;
            double originOffset = la::dot(planeNormal, edgeMid);
            auto [a, b] = piece.SplitByPlane(planeNormal, originOffset);
            if (!a.IsEmpty() && !b.IsEmpty()) {
              next.push_back(a);
              next.push_back(b);
              split = true;
              anySplit = true;
              break;
            }
          }
          if (!split) next.push_back(piece);
        }
        work = std::move(next);
        if (!anySplit) break;
      }
      // Recurse carve-only on split results, DT fallback for insoluble
      for (auto& piece : work) {
        if (piece.Volume() < 1e-10) continue;
        auto pImpl = piece.GetCsgLeafNode().GetImpl();
        if (pImpl->IsConvex()) {
          outputs.push_back(piece);
        } else {
          auto fallback = pImpl->ConvexDecomposition(2, 1);
          outputs.insert(outputs.end(), fallback.begin(), fallback.end());
        }
      }
      break;
    }

    outputs.push_back(bestHull);
    shape = shape - bestHull;
    // Skip degenerate remainders from boolean subtraction
    if (shape.Volume() < bestHull.Volume() * 1e-10) break;

    // Decompose remainder in case subtraction split it
    if (!shape.IsEmpty()) {
      auto parts = shape.Decompose();
      if (parts.size() > 1) {
        for (size_t i = 1; i < parts.size(); i++) {
          if (parts[i].Volume() < 1e-15) continue;
          auto pImpl = parts[i].GetCsgLeafNode().GetImpl();
          auto sub = pImpl->ConvexDecompositionCarveOnly();
          outputs.insert(outputs.end(), sub.begin(), sub.end());
        }
        shape = parts[0];
      }
    }
  }

  // Filter degenerate zero-volume pieces from boolean artifacts
  std::vector<Manifold> filtered;
  for (auto& p : outputs)
    if (p.Volume() > 1e-10) filtered.push_back(p);
  return filtered;
}

/**
 * Onion-peel convex decomposition. Finds all reflex edges, hulls their
 * vertices, intersects the hull with the shape to peel off a convex
 * layer, subtracts it, and recurses on the remainder. If there's only
 * one reflex edge, does a plane split instead.
 */
std::vector<Manifold> Manifold::Impl::ConvexDecompositionOnionPeel() const {
  std::vector<Manifold> outputs;

  auto thisImpl = std::make_shared<Impl>(*this);
  Manifold shape(std::make_shared<CsgLeafNode>(thisImpl));

  if (shape.IsEmpty() || shape.Volume() < 1e-12) return outputs;
  if (IsConvex()) {
    outputs.push_back(shape.Hull());
    return outputs;
  }

  for (int iteration = 0; iteration < 1000 && !shape.IsEmpty(); iteration++) {
    if (shape.Volume() < 1e-12) break;
    auto sImpl = shape.GetCsgLeafNode().GetImpl();
    if (sImpl->IsConvex()) {
      outputs.push_back(shape.Hull());
      break;
    }

    const auto& he = sImpl->halfedge_;
    const auto& fn = sImpl->faceNormal_;
    const auto& vp = sImpl->vertPos_;
    size_t nEdges = he.size();

    // Collect all reflex edge vertices
    std::vector<vec3> reflexVerts;
    int reflexCount = 0;
    vec3 lastN0, lastN1, lastEdgeVec, lastEdgeMid;
    for (size_t idx = 0; idx < nEdges; idx++) {
      auto edge = he[idx];
      if (!edge.IsForward()) continue;
      vec3 n0 = fn[idx / 3];
      vec3 n1 = fn[edge.pairedHalfedge / 3];
      vec3 edgeVec = vp[edge.endVert] - vp[edge.startVert];
      if (la::dot(edgeVec, la::cross(n0, n1)) >= 0) continue;
      reflexVerts.push_back(vp[edge.startVert]);
      reflexVerts.push_back(vp[edge.endVert]);
      lastN0 = n0;
      lastN1 = n1;
      lastEdgeVec = edgeVec;
      lastEdgeMid = (vp[edge.startVert] + vp[edge.endVert]) * 0.5;
      reflexCount++;
    }

    if (reflexCount == 0) {
      outputs.push_back(shape.Hull());
      break;
    }

    // Single reflex edge — plane split
    if (reflexCount == 1) {
      vec3 bisector = la::normalize(lastN0 + lastN1);
      vec3 planeNormal = la::cross(la::normalize(lastEdgeVec), bisector);
      double pnLen = la::length(planeNormal);
      if (pnLen < 1e-10) {
        outputs.push_back(shape);
        break;
      }
      planeNormal /= pnLen;
      double originOffset = la::dot(planeNormal, lastEdgeMid);
      auto [a, b] = shape.SplitByPlane(planeNormal, originOffset);
      if (!a.IsEmpty() && a.Volume() > 1e-12) {
        auto aSub =
            a.GetCsgLeafNode().GetImpl()->ConvexDecompositionOnionPeel();
        outputs.insert(outputs.end(), aSub.begin(), aSub.end());
      }
      if (!b.IsEmpty() && b.Volume() > 1e-12) {
        auto bSub =
            b.GetCsgLeafNode().GetImpl()->ConvexDecompositionOnionPeel();
        outputs.insert(outputs.end(), bSub.begin(), bSub.end());
      }
      break;
    }

    // Multiple reflex edges — hull their vertices, intersect with shape
    // to peel off the concave region, then subtract it
    Manifold reflexHull = Manifold::Hull(reflexVerts);

    // If reflex hull is degenerate (coplanar reflex vertices, e.g.
    // intersection circle of two spheres), fit a plane and split instead
    if (reflexHull.IsEmpty() || std::abs(reflexHull.Volume()) < 1e-12) {
      // Fit plane through reflex vertices via centroid + normal from
      // cross product of two edges
      vec3 centroid(0, 0, 0);
      for (auto& v : reflexVerts) centroid += v;
      centroid /= (double)reflexVerts.size();
      // Fit plane using Newell's method (robust for coplanar polygons)
      vec3 planeNormal(0, 0, 0);
      for (size_t i = 0; i < reflexVerts.size(); i++) {
        vec3 cur = reflexVerts[i];
        vec3 nxt = reflexVerts[(i + 1) % reflexVerts.size()];
        planeNormal.x += (cur.y - nxt.y) * (cur.z + nxt.z);
        planeNormal.y += (cur.z - nxt.z) * (cur.x + nxt.x);
        planeNormal.z += (cur.x - nxt.x) * (cur.y + nxt.y);
      }
      double pnLen = la::length(planeNormal);
      if (pnLen > 1e-10) {
        planeNormal /= pnLen;
        double originOffset = la::dot(planeNormal, centroid);
        auto [a, b] = shape.SplitByPlane(planeNormal, originOffset);
        if (!a.IsEmpty() && a.Volume() > 1e-12) {
          auto aSub =
              a.GetCsgLeafNode().GetImpl()->ConvexDecompositionOnionPeel();
          outputs.insert(outputs.end(), aSub.begin(), aSub.end());
        }
        if (!b.IsEmpty() && b.Volume() > 1e-12) {
          auto bSub =
              b.GetCsgLeafNode().GetImpl()->ConvexDecompositionOnionPeel();
          outputs.insert(outputs.end(), bSub.begin(), bSub.end());
        }
        break;
      }
      outputs.push_back(shape);
      break;
    }

    Manifold peel = shape ^ reflexHull;
    if (peel.IsEmpty() || peel.Volume() < 1e-12) {
      outputs.push_back(shape);
      break;
    }

    // The "peel" is the concave region. The convex outer layer is
    // shape - peel. Decompose both and recurse.
    Manifold outer = shape - peel;

    // Outer layer: decompose (may be multiple disconnected convex pieces)
    if (!outer.IsEmpty() && outer.Volume() > 1e-12) {
      for (auto& part : outer.Decompose()) {
        if (part.Volume() < 1e-12) continue;
        auto pImpl = part.GetCsgLeafNode().GetImpl();
        if (pImpl->IsConvex()) {
          outputs.push_back(part.Hull());
        } else {
          auto sub = pImpl->ConvexDecompositionOnionPeel();
          outputs.insert(outputs.end(), sub.begin(), sub.end());
        }
      }
    }

    // Recurse on the peel (inner concave remainder)
    shape = peel;
  }

  // Filter degenerates
  std::vector<Manifold> filtered;
  for (auto& p : outputs)
    if (p.Volume() > 1e-10) filtered.push_back(p);
  return filtered;
}

}  // namespace manifold
