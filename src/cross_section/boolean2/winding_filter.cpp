// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Winding-rule filter over the canonical sub-edges. Builds the halfedge
// arrangement, walks each face once via BFS to propagate winding numbers
// (seeded by one ray-cast per connected component on the outer face),
// then keeps only the sub-edges that separate "inside" from "outside"
// under the chosen rule (Add / Intersect / EvenOdd / NonZero / Negative).
//
// FastEdge + CastWindingRay are the ray-cast helpers for the outer-
// face seed; FilterByWindingHalfedges is the entry point. iostream usage
// gated by MANIFOLD_DEBUG.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#ifdef MANIFOLD_DEBUG
#include <iostream>
#include <map>
#include <string>
#endif
#include <utility>
#include <vector>

#include "../../parallel.h"
#include "canonicalize.h"
#include "diagnostics.h"
#include "predicates.h"
#include "winding_filter.h"

namespace manifold {
namespace boolean2 {

namespace {

constexpr double kSeedStepCoordScaleUlps = 16.0;
constexpr double kSeedStepEpsFraction = 0.25;
constexpr double kSeedStepEdgeLengthFraction = 1e-3;

bool IsInside(WindRule rule, int w) {
  switch (rule) {
    case WindRule::Add:
      return w > 0;
    case WindRule::Intersect:
      return w > 1;
    case WindRule::EvenOdd:
      return (w & 1) != 0;
    case WindRule::NonZero:
      return w != 0;
    case WindRule::Negative:
      return w < 0;
  }
  return false;
}

// Pre-flattened canonical sub-edges for seed ray-casts. Hoisting to a flat
// array of (yMin, yMax, x0, dx_dy, signedMult) turns the crossing loop into a
// tight cache-friendly scan and removes per-call indirection through canon
// edges / verts[].
//
// signedMult encodes both direction and magnitude: positive when the
// canonical (vMin -> vMax) form is upward (p0.y < p1.y), negative when
// downward. The ray-cast then just sums signedMult over crossed edges.
struct FastEdge {
  double yMin, yMax;  // canonical orientation: yMin < yMax
  double x0;          // x at y = yMin
  double xSlope;      // segment dx/dy, used to interpolate x at ray y
  int signedMult;     // mult if upward in canonical form, else -mult
};

std::vector<FastEdge> BuildFastEdges(const CanonicalSubEdges& canon,
                                     const std::vector<vec2>& verts) {
  std::vector<FastEdge> out;
  out.reserve(canon.edges.size());
  for (const auto& edge : canon.edges) {
    const int mult = edge.mult;
    vec2 p0 = verts[edge.vMin];
    vec2 p1 = verts[edge.vMax];
    const bool upward = p0.y < p1.y;
    if (!upward) std::swap(p0, p1);
    if (p0.y == p1.y) continue;  // horizontal edges contribute nothing
    FastEdge e;
    e.yMin = p0.y;
    e.yMax = p1.y;
    e.x0 = p0.x;
    e.xSlope = (p1.x - p0.x) / (p1.y - p0.y);
    e.signedMult = upward ? mult : -mult;
    out.push_back(e);
  }
  return out;
}

int CastWindingRay(vec2 origin, const std::vector<FastEdge>& edges) {
  int winding = 0;
  for (const auto& e : edges) {
    if (origin.y < e.yMin || origin.y >= e.yMax) continue;
    const double xCross = e.x0 + e.xSlope * (origin.y - e.yMin);
    if (xCross <= origin.x) continue;
    winding += e.signedMult;
  }
  return winding;
}

// =============================================================================
// Planar halfedge face traversal. The actual winding-rule filter.
//
// Builds the same halfedge structure manifold's 3D mesh
// `Manifold::Impl::halfedge_` uses from the canonical sub-edges, walks face
// cycles to identify each planar face, seeds one face per connected component
// by ray-cast, and propagates winding numbers across halfedge twins. An edge
// is kept iff its left and right faces disagree under the selected rule.
// =============================================================================

struct Halfedge {
  int twin;    // index of the twin halfedge in halfedges[]
  int next;    // next halfedge along the same face's CCW boundary
  int origin;  // vertex this halfedge starts at
  int face;    // face id (-1 until assigned)
  int mult;    // signed multiplicity in this direction
};

#ifdef MANIFOLD_DEBUG
void RecordHalfedgeFaces(Trace* trace, WindRule rule,
                         const std::vector<vec2>& verts,
                         const std::vector<Halfedge>& halfedges,
                         const std::vector<int>& faceStartHE,
                         const std::vector<int>& faceWind,
                         const std::vector<double>& faceArea, int outerFace) {
  if (!trace) return;
  TracePhase& phase = trace->AddPhase("halfedge_faces");
  const int nFaces = static_cast<int>(faceStartHE.size());
  for (int f = 0; f < nFaces; ++f) {
    SimplePolygon loop;
    const int h0 = faceStartHE[f];
    if (h0 >= 0) {
      int h = h0;
      int safety = 0;
      do {
        loop.push_back(verts[halfedges[h].origin]);
        h = halfedges[h].next;
        if (h < 0 || ++safety > static_cast<int>(halfedges.size())) break;
      } while (h != h0);
    }
    const bool isOuter = f == outerFace;
    phase.polygons.push_back(
        {std::string("face") + std::to_string(f), std::move(loop),
         isOuter ? "outer_face_unbounded" : "face", "", faceWind[f],
         IsInside(rule, faceWind[f]), isOuter ? "unbounded outer face" : ""});
    phase.annotations.push_back({std::string("face") + std::to_string(f),
                                 "area", std::to_string(faceArea[f])});
  }
}
#endif

// The halfedge filter body, parameterized by the winding rule.
std::vector<OutEdge> FilterByWindingHalfedgesImpl(
    const CanonicalSubEdges& canon, const std::vector<vec2>& verts, bool debug,
    WindRule rule, Trace* trace) {
#ifdef MANIFOLD_DEBUG
  if (debug && DebugVerbose()) {
    std::cout << "[FilterByWindingHalfedges] canon.edges.size()="
              << canon.edges.size() << " verts.size()=" << verts.size() << "\n";
  }
#endif
  // 1. Build halfedges. Each canonical (vMin, vMax) with mult m becomes:
  //    - hA: vMin -> vMax, mult = m
  //    - hB: vMax -> vMin, mult = -m
  //    Twins are paired (hA.twin = hB, hB.twin = hA).
  //
  thread_local static std::vector<Halfedge> halfedgesBuf;
  auto& halfedges = halfedgesBuf;
  halfedges.resize(2 * canon.edges.size());
  for (size_t i = 0; i < canon.edges.size(); ++i) {
    const auto& edge = canon.edges[i];
    const int hA = static_cast<int>(2 * i);
    halfedges[hA] = {hA + 1, -1, edge.vMin, -1, edge.mult};
    halfedges[hA + 1] = {hA, -1, edge.vMax, -1, -edge.mult};
  }
  if (halfedges.empty()) return {};

  // 2. Group halfedges by origin vertex; sort each group by direction angle
  //    (CCW). This is the "rotational order" needed to compute next pointers.
  //    Flat CSR layout (outOff[v]..outOff[v+1] indexes into outFlat) avoids
  //    the per-vert vector<int> allocations a vector-of-vector would pay.
  const int nVerts = static_cast<int>(verts.size());
  thread_local static std::vector<int> outOffBuf;
  thread_local static std::vector<int> outFlatBuf;
  thread_local static std::vector<int> outCurBuf;
  auto& outOff = outOffBuf;
  auto& outFlat = outFlatBuf;
  auto& outCur = outCurBuf;
  outOff.assign(nVerts + 1, 0);
  for (int i = 0; i < (int)halfedges.size(); ++i) {
    ++outOff[halfedges[i].origin + 1];
  }
  for (int v = 1; v <= nVerts; ++v) outOff[v] += outOff[v - 1];
  outFlat.resize(halfedges.size());
  outCur.assign(outOff.begin(), outOff.end());
  for (int i = 0; i < (int)halfedges.size(); ++i) {
    outFlat[outCur[halfedges[i].origin]++] = i;
  }
  // atan2-free angular comparator: split the plane into two half-planes, then
  // compare by cross product. Same CCW order from +x, without libm in the
  // per-comparison hot path.
  auto bucketOf = [](const vec2& d) {
    return (d.y > 0 || (d.y == 0 && d.x > 0)) ? 0 : 1;
  };
  manifold::for_each(
      manifold::autoPolicy(nVerts), manifold::countAt(0),
      manifold::countAt(nVerts), [&](int v) {
        const int beg = outOff[v];
        const int end = outOff[v + 1];
        const int n = end - beg;
        if (n < 2) return;
        const vec2 vp = verts[v];
        if (n == 2) {
          // Fast path for typical polygon corners (degree 2). Skip the
          // std::sort overhead - direct compare-and-swap.
          const vec2 dA =
              verts[halfedges[halfedges[outFlat[beg]].twin].origin] - vp;
          const vec2 dB =
              verts[halfedges[halfedges[outFlat[beg + 1]].twin].origin] - vp;
          const int bA = bucketOf(dA), bB = bucketOf(dB);
          bool aFirst;
          if (bA != bB)
            aFirst = bA < bB;
          else
            aFirst = (dA.x * dB.y - dA.y * dB.x > 0);
          if (!aFirst) std::swap(outFlat[beg], outFlat[beg + 1]);
          return;
        }
        manifold::stable_sort(
            outFlat.begin() + beg, outFlat.begin() + end, [&](int a, int b) {
              const vec2 dA = verts[halfedges[halfedges[a].twin].origin] - vp;
              const vec2 dB = verts[halfedges[halfedges[b].twin].origin] - vp;
              const int bA = bucketOf(dA), bB = bucketOf(dB);
              if (bA != bB) return bA < bB;
              return dA.x * dB.y - dA.y * dB.x > 0;
            });
      });

  // 3. Compute next pointers: each halfedge takes the smallest left turn at
  //    its destination, which is the entry just before its twin in the CCW
  //    outgoing list. Degree-2 vertices hide this choice; intersections do not.
  // Each .next write is independent; reads are from the sorted outgoing CSR.
  manifold::for_each(
      manifold::autoPolicy(halfedges.size()), manifold::countAt(0),
      manifold::countAt(static_cast<int>(halfedges.size())), [&](int i) {
        const int twinIdx = halfedges[i].twin;
        const int destV = halfedges[twinIdx].origin;
        const int beg = outOff[destV];
        const int end = outOff[destV + 1];
        auto it =
            std::find(outFlat.begin() + beg, outFlat.begin() + end, twinIdx);
        if (it == outFlat.begin() + end) return;
        auto prevIt = (it == outFlat.begin() + beg)
                          ? (outFlat.begin() + end - 1)
                          : (it - 1);
        halfedges[i].next = *prevIt;
      });

  // 4. Walk face cycles, assign face IDs. Each unmarked halfedge starts a
  //    new face; follow `next` chain back to the start.
  int nFaces = 0;
  for (int i = 0; i < (int)halfedges.size(); ++i) {
    if (halfedges[i].face != -1) continue;
    int h = i;
    int safety = 0;
    bool malformed = false;
    do {
      if (halfedges[h].next == -1 || safety++ > (int)halfedges.size()) {
        // Malformed cycle; bail rather than infinite-loop. Indicates an
        // upstream bug (mismatched twin/next pointers from the angular
        // sort, or a non-2-manifold canonical edge set). Surface under
        // MANIFOLD_DEBUG; the silent drop downstream produces wrong
        // topology that's hard to debug later.
        malformed = true;
        break;
      }
      halfedges[h].face = nFaces;
      h = halfedges[h].next;
    } while (h != i);
#ifdef MANIFOLD_DEBUG
    if (malformed && DebugVerbose()) {
      std::cout
          << "[FilterByWindingHalfedges] malformed face cycle starting at "
             "halfedge "
          << i << " (face " << nFaces << " safety=" << safety
          << " halfedges=" << halfedges.size() << ")\n";
    }
#endif
    if (malformed) {
      assert(false && "malformed halfedge cycle");
      return {};
    }
    ++nFaces;
  }

  // 5. Compute signed area per face. Centering the shoelace sum avoids
  // cancellation from large translations or skinny-but-valid faces.
  // First halfedge encountered per face. Used both as the centering
  // reference for the shoelace area (below) and as the starting halfedge
  // for the per-face winding ray-cast.
  thread_local static std::vector<int> faceStartHE;
  thread_local static std::vector<double> faceArea;
  faceStartHE.assign(nFaces, -1);
  for (int i = 0; i < (int)halfedges.size(); ++i) {
    if (halfedges[i].face >= 0 && faceStartHE[halfedges[i].face] == -1)
      faceStartHE[halfedges[i].face] = i;
  }
  faceArea.assign(nFaces, 0.0);
  for (int i = 0; i < (int)halfedges.size(); ++i) {
    if (halfedges[i].face < 0) continue;
    const int faceRefHE = faceStartHE[halfedges[i].face];
    if (faceRefHE < 0) continue;
    const vec2 ref = verts[halfedges[faceRefHE].origin];
    const vec2 a = verts[halfedges[i].origin] - ref;
    const vec2 b = verts[halfedges[halfedges[i].twin].origin] - ref;
    faceArea[halfedges[i].face] += (a.x * b.y - b.x * a.y) * 0.5;
  }
  int outerFace = 0;
  for (int f = 1; f < nFaces; ++f) {
    if (faceArea[f] < faceArea[outerFace]) outerFace = f;
  }
#ifdef MANIFOLD_DEBUG
  if (debug && DebugVerbose()) {
    std::cout << "Halfedges: " << halfedges.size() << " halfedges, " << nFaces
              << " faces\n";
    int negAreaCount = 0;
    for (int f = 0; f < nFaces; ++f) {
      std::cout << "  face " << f << " area=" << faceArea[f]
                << (f == outerFace ? "  <-- outer" : "") << "\n";
      if (faceArea[f] < 0 && f != outerFace) ++negAreaCount;
    }
    if (negAreaCount > 0) {
      std::cout << "  WARNING: " << negAreaCount
                << " bounded face(s) have negative signed area; cycle "
                   "convention may be inverted\n";
    }
    // Group halfedges by face, count mults.
    std::map<int, std::map<int, int>> faceMults;
    for (int i = 0; i < (int)halfedges.size(); ++i) {
      faceMults[halfedges[i].face][halfedges[i].mult]++;
    }
    for (auto& [f, m] : faceMults) {
      std::cout << "  face " << f << " mults:";
      for (auto& [mu, c] : m) std::cout << " " << mu << "x" << c;
      std::cout << "\n";
    }
  }
#endif

  // 6. Seed one face per connected component, then propagate winding across
  // twin pointers: crossing h from left to right subtracts h.mult.
  std::vector<FastEdge> fastEdges;
  bool fastEdgesBuilt = false;
  auto ensureFastEdges = [&]() {
    if (!fastEdgesBuilt) {
      fastEdges = BuildFastEdges(canon, verts);
      fastEdgesBuilt = true;
    }
  };
  thread_local static std::vector<int> faceWind;
  thread_local static std::vector<uint8_t> wAssigned;
  faceWind.assign(nFaces, 0);
  wAssigned.assign(nFaces, 0);
  double coordScale = 0.0;
  vec2 bboxMin(std::numeric_limits<double>::infinity());
  vec2 bboxMax(-std::numeric_limits<double>::infinity());
  for (const vec2& v : verts) {
    coordScale = std::max(coordScale, std::max(std::fabs(v.x), std::fabs(v.y)));
    bboxMin.x = std::min(bboxMin.x, v.x);
    bboxMin.y = std::min(bboxMin.y, v.y);
    bboxMax.x = std::max(bboxMax.x, v.x);
    bboxMax.y = std::max(bboxMax.y, v.y);
  }
  const double bboxHalfExtent =
      0.5 * std::max(bboxMax.x - bboxMin.x, bboxMax.y - bboxMin.y);
  // Keep seed samples close to boundary features, but at least several ULPs
  // from displaced input edges and never more than a small edge-length
  // fraction.
  const double seedStepCap =
      std::max(kSeedStepCoordScaleUlps * kU * coordScale,
               kSeedStepEpsFraction * EpsilonFromScale(bboxHalfExtent));
  auto castFaceHalfedge = [&](int h) {
    if (h < 0) return 0;
    const vec2 a = verts[halfedges[h].origin];
    const vec2 b = verts[halfedges[halfedges[h].twin].origin];
    const vec2 mid = (a + b) * 0.5;
    const vec2 d = b - a;
    const double len = length(d);
    if (len == 0) return 0;
    const vec2 perp(-d.y / len, d.x / len);
    const double step =
        std::min(len * kSeedStepEdgeLengthFraction, seedStepCap);
    const vec2 pInF = mid + perp * step;
    ensureFastEdges();
    return CastWindingRay(pInF, fastEdges);
  };
  auto seedRayCast = [&](int f) {
    const int h0 = faceStartHE[f];
    faceWind[f] = h0 < 0 ? 0 : castFaceHalfedge(h0);
    wAssigned[f] = 1;
  };
  // The unbounded outer face is conventionally outside, so winding = 0.
  faceWind[outerFace] = 0;
  wAssigned[outerFace] = 1;
  thread_local static std::vector<int> bfsQ;
  bfsQ.clear();
  bfsQ.reserve(nFaces);
  auto propagateFrom = [&](int seed) {
    bfsQ.clear();
    bfsQ.push_back(seed);
    size_t head = 0;
    while (head < bfsQ.size()) {
      const int f = bfsQ[head++];
      const int h0 = faceStartHE[f];
      if (h0 < 0) continue;
      int hh = h0;
      int safety = 0;
      do {
        const int twinH = halfedges[hh].twin;
        const int adj = halfedges[twinH].face;
        if (adj >= 0 && !wAssigned[adj]) {
          // Stepping LEFT of hh (= f) -> RIGHT of hh (= adj):
          // winding loses the +mult contribution that the LEFT side saw.
          faceWind[adj] = faceWind[f] - halfedges[hh].mult;
          wAssigned[adj] = 1;
          bfsQ.push_back(adj);
        }
        hh = halfedges[hh].next;
        if (hh < 0 || ++safety > (int)halfedges.size()) break;
      } while (hh != h0);
    }
  };
  propagateFrom(outerFace);
  // Pick up any disconnected components with their own seed ray-cast.
  thread_local static std::vector<int> componentQ;
  thread_local static std::vector<int> componentFaces;
  thread_local static std::vector<uint8_t> componentSeen;
  thread_local static std::vector<uint8_t> componentLocalOuter;
  componentQ.clear();
  componentFaces.clear();
  componentSeen.assign(nFaces, 0);
  componentLocalOuter.assign(nFaces, 0);
  componentLocalOuter[outerFace] = 1;
  auto collectComponent = [&](int seed) {
    componentQ.clear();
    componentFaces.clear();
    componentQ.push_back(seed);
    componentSeen[seed] = 1;
    size_t head = 0;
    while (head < componentQ.size()) {
      const int f = componentQ[head++];
      componentFaces.push_back(f);
      const int h0 = faceStartHE[f];
      if (h0 < 0) continue;
      int hh = h0;
      int safety = 0;
      do {
        const int adj = halfedges[halfedges[hh].twin].face;
        if (adj >= 0 && !wAssigned[adj] && !componentSeen[adj]) {
          componentSeen[adj] = 1;
          componentQ.push_back(adj);
        }
        hh = halfedges[hh].next;
        if (hh < 0 || ++safety > (int)halfedges.size()) break;
      } while (hh != h0);
    }
  };
  for (int f = 0; f < nFaces; ++f) {
    if (!wAssigned[f]) {
      collectComponent(f);
      int localOuter = f;
      for (int cf : componentFaces) {
        if (faceArea[cf] < faceArea[localOuter]) localOuter = cf;
      }
      componentLocalOuter[localOuter] = 1;
      seedRayCast(localOuter);
      propagateFrom(localOuter);
    }
  }

#ifdef MANIFOLD_DEBUG
  if (debug && DebugVerbose()) {
    std::cout << "  face windings:";
    for (int f = 0; f < nFaces; ++f) {
      std::cout << " f" << f << "=" << faceWind[f];
    }
    std::cout << "\n";
  }
  RecordHalfedgeFaces(trace, rule, verts, halfedges, faceStartHE, faceWind,
                      faceArea, outerFace);
#endif

  // 7. Filter canonical sub-edges by left/right face windings. The first
  //    halfedge of each pair (the (vMin -> vMax) direction) is at index
  //    2*i; its twin (vMax -> vMin) is at 2*i + 1.
  std::vector<OutEdge> out;
  out.reserve(canon.edges.size());
  int hi = 0;
  for (const auto& edge : canon.edges) {
    const int hA = hi;
    const int hB = hi + 1;
    hi += 2;
    const int leftFace = halfedges[hA].face;
    const int rightFace = halfedges[hB].face;
    if (leftFace < 0 || rightFace < 0) continue;
    // A component's local outer cycle can be split by other components;
    // classify the side adjacent to this edge instead of reusing one propagated
    // winding.
    const auto faceWindingAtHalfedge = [&](int face, int halfedge) {
      if (rule == WindRule::NonZero && componentLocalOuter[face]) {
        return castFaceHalfedge(halfedge);
      }
      return faceWind[face];
    };
    const int wL = faceWindingAtHalfedge(leftFace, hA);
    const int wR = faceWindingAtHalfedge(rightFace, hB);
    const bool leftIn = IsInside(rule, wL);
    const bool rightIn = IsInside(rule, wR);
    if (leftIn == rightIn) continue;
    if (leftIn) {
      out.push_back({edge.vMin, edge.vMax, 1});
    } else {
      out.push_back({edge.vMax, edge.vMin, 1});
    }
  }
  return out;
}
}  // namespace

std::vector<OutEdge> FilterByWindingHalfedges(const CanonicalSubEdges& canon,
                                              const std::vector<vec2>& verts,
                                              bool debug, WindRule rule,
                                              Trace* trace) {
#ifndef MANIFOLD_DEBUG
  (void)trace;
#endif
  return FilterByWindingHalfedgesImpl(canon, verts, debug, rule, trace);
}

}  // namespace boolean2
}  // namespace manifold
