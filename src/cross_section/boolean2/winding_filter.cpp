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
// FastEdge + CastWindingRayFast are the ray-cast helpers for the outer-
// face seed; FilterByWindingHalfedges is the entry point. iostream usage
// gated by MANIFOLD_DEBUG.

#include <algorithm>
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
#include "bvh.h"
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

}  // namespace

// =============================================================================
// Ray-cast winding-number primitives (legacy seed; the BFS face traversal
// below is the primary path).
// =============================================================================

// Cast a horizontal ray to +x from `origin`. Count signed crossings of
// directed sub-edges. For each sub-edge (vMin, vMax) with signed
// multiplicity m, the contribution is +m if the edge goes upward (in
// canonical (vMin, vMax) form, "upward" = vMin's y < vMax's y when
// crossed from below), -m if downward.
//
// Used per-face by `FilterByWindingHalfedges`: the origin is offset
// perpendicularly LEFT of a boundary halfedge, which puts it strictly
// inside the face and away from any vertex, so the ray never hits a
// vertex exactly under non-adversarial inputs.
int CastWindingRay(vec2 origin, const CanonicalSubEdges& canon,
                   const std::vector<vec2>& verts) {
  int winding = 0;
  for (const auto& edge : canon.edges) {
    const int mult = edge.mult;
    vec2 p0 = verts[edge.vMin];
    vec2 p1 = verts[edge.vMax];
    // Order so p0.y <= p1.y for crossing test.
    bool upward = p0.y < p1.y;
    if (!upward) std::swap(p0, p1);
    // Strictly half-open in y to avoid double-counting at vertices.
    if (origin.y < p0.y || origin.y >= p1.y) continue;
    // Compute x of the segment at origin.y.
    double t = (origin.y - p0.y) / (p1.y - p0.y);
    double xCross = p0.x + t * (p1.x - p0.x);
    if (xCross < origin.x) continue;
    // Crossing direction: original direction was upward iff key.first <
    // key.second and positions matched -- already encoded in `mult`'s sign. We
    // need the signed contribution to the winding number when crossing
    // left-to-right. For a positive-multiplicity edge oriented (vMin -> vMax)
    // in canonical form, an upward crossing (with origin to the left of the
    // edge) increments the winding number on the right side by mult. Since we
    // cast +x, we are computing winding on the LEFT of the ray, which is the
    // side we're at the origin. This is +mult per upward crossing, -mult per
    // downward crossing.
    winding += upward ? mult : -mult;
  }
  return winding;
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
  double dxdy;        // (x1 - x0) / (yMax - yMin)
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
    e.dxdy = (p1.x - p0.x) / (p1.y - p0.y);
    e.signedMult = upward ? mult : -mult;
    out.push_back(e);
  }
  return out;
}

int CastWindingRayFast(vec2 origin, const std::vector<FastEdge>& edges) {
  int winding = 0;
  for (const auto& e : edges) {
    if (origin.y < e.yMin || origin.y >= e.yMax) continue;
    const double xCross = e.x0 + e.dxdy * (origin.y - e.yMin);
    if (xCross < origin.x) continue;
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

namespace halfedge_internal {
struct Halfedge {
  int twin;    // index of the twin halfedge in halfedges[]
  int next;    // next halfedge along the same face's CCW boundary
  int origin;  // vertex this halfedge starts at
  int face;    // face id (-1 until assigned)
  int mult;    // signed multiplicity in this direction
};
}  // namespace halfedge_internal

// Internal predicate over a face's winding number, deciding whether
// the face is "inside" the result region. Five rules cover production
// and diagnostic-driver corpus needs:
//   - Add:       w > 0     (default; Smith's wind > 0 union)
//   - Intersect: w > 1     (both normalized operands cover)
//   - EvenOdd:   w & 1     (used internally by Xor; SVG/Clipper2 EVENODD)
//   - NonZero:   w != 0    (Clipper2 NONZERO, mfogel union of pre-filled)
//   - Negative:  w < 0     (Clipper2 NEGATIVE; CW-oriented input regions)
// An edge is retained iff its left and right faces disagree on the
// rule.
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

namespace detail {
#ifdef MANIFOLD_DEBUG
void RecordHalfedgeFaces(
    Trace* trace, WindRule rule, const std::vector<vec2>& verts,
    const std::vector<halfedge_internal::Halfedge>& halfedges,
    const std::vector<int>& faceStartHE, const std::vector<int>& faceWind,
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
  using halfedge_internal::Halfedge;
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
  // Per-vertex angular sort: each vertex's outgoing list is independent
  // (read-only on halfedges/verts; writes only its own slot). atan2 is
  // expensive and the inner sort is O(d log d) per vertex of degree d;
  // for big arrangements the total is the second-largest cost inside
  // this filter. Output is deterministic because the sort is pure.
  // atan2-free angular comparator: split the plane into two half-planes
  // (bucket 0 = upper + +x axis, bucket 1 = lower + -x axis); within a
  // bucket, compare by sign of the cross product. Sorts CCW from +x.
  // Same monotone order as atan2 but no transcendental in the per-
  // comparison hot path.
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

  // 3. Compute next pointers. For halfedge h arriving at vertex v
  //    (= h.twin.origin), h.next must be the outgoing edge that makes
  //    the SMALLEST LEFT TURN from h's incoming direction. h.incoming
  //    direction is opposite of h.twin's outgoing direction; the smallest
  //    CCW rotation from h.incoming visits halfedges starting from
  //    "h.incoming + small CCW" and finds the first entry. In the sorted
  //    CCW outgoing list, this corresponds to **one step CW** from
  //    h.twin (with wraparound).
  //
  //    Equivalent: starting at angle (h.twin + π) = h.incoming, sweep CCW
  //    by ε, look up the first sorted entry. That entry is at sorted-list
  //    position one before h.twin (= it - 1, with wraparound).
  //
  //    Using "it+1" instead of "it-1" picks the halfedge ALMOST A FULL
  //    REVOLUTION CCW from h.twin = a RIGHT turn at v. For degree-2
  //    vertices (chains, simple polygon corners), N=2 symmetry makes
  //    "it+1" and "it-1" equivalent and both work. For degree-≥3
  //    vertices (intersection points after FindAndInsertIntersections),
  //    "it+1" and "it-1" differ and
  //    only "it-1" produces correctly-oriented face cycles.
  // Each halfedge's .next is determined independently by reading the
  // (now-sorted) outgoing list at its destination vertex. Writes are to
  // independent slots; reads are read-only. Tried caching position-of-
  // self in an int-per-halfedge array - for the typical degree-2
  // polygon-corner case the std::find on a 2-element vector is faster
  // than building the cache, so kept the find.
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
    ++nFaces;
  }

  // 5. Compute signed area per face. Outer face has the most negative area.
  //    CRITICAL: at displaced coords (e.g. 1.5e9), the raw shoelace
  //    `a.x * b.y - b.x * a.y` has each product on order 2.25e18 with
  //    ULP ~2000, so summation precision swamps a typical face area of
  //    O(1). The sign becomes random and outer-face detection breaks.
  //    Fix: center each face's coordinates relative to its first vertex
  //    before summing. With centered coords O(edge length), products
  //    are O((edge length)²) and the sum is precise.
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

  // 6. Compute winding per face by propagation through twin pointers.
  //
  //    Earlier this did F independent ray-casts (one per face), which
  //    is O(F·E) work. Replace with: ray-cast a single seed face per
  //    connected component of the halfedge graph, then BFS-propagate to all
  //    other faces in the component. Stepping from face_A to face_B
  //    across a halfedge h (h.face=A, h.twin.face=B) crosses h's
  //    canonical-edge contribution, so faceWind[B] = faceWind[A] -
  //    h.mult (we're stepping from LEFT of h, where +mult contributes,
  //    to RIGHT, where 0 does). BFS reaches every face in the
  //    component exactly once, so total work is O(E + F) instead of
  //    O(F·E).
  //
  //    Multi-component arrangements (real for self-intersecting input
  //    whose result has multiple disjoint regions): when BFS finishes,
  //    any unvisited face is in another component; ray-cast its
  //    interior to seed and BFS again. Most cases are single-component
  //    so this fallback rarely fires.
  //
  //    The FastEdge hoist is only needed for disconnected components after
  //    the outer-face propagation. Most arrangements are one component, so
  //    build it lazily when a second seed is actually needed.
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
  // Cap the normal offset by local geometry scale so seed samples stay in
  // narrow faces; keep a coordinate-scale floor so displaced inputs still move
  // by several ULPs. The eps fraction keeps the sample close to the boundary
  // features created by the boolean2 epsilon budget, while the ULP floor avoids
  // sampling the original edge again at large translated coordinates. The
  // per-edge step below is also capped at 0.1% of edge length so short boundary
  // edges seed inside their adjacent face; larger fractions can jump across
  // thin faces in disconnected-component winding seeds.
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
    return CastWindingRayFast(pInF, fastEdges);
  };
  auto seedRayCastLeastBiased = [&](int f) {
    int h0 = faceStartHE[f];
    if (h0 < 0) {
      faceWind[f] = 0;
      wAssigned[f] = 1;
      return;
    }
    // A disconnected component's local outer face can pass through regions
    // separated by other components. Sample each boundary edge and use the
    // least-biased absolute winding so cleanup passes keep disjoint islands
    // instead of inheriting a neighboring component's winding.
    int best = castFaceHalfedge(h0);
    int hh = halfedges[h0].next;
    int safety = 0;
    while (hh >= 0 && hh != h0 && ++safety <= (int)halfedges.size()) {
      const int w = castFaceHalfedge(hh);
      if (std::abs(w) < std::abs(best) ||
          (std::abs(w) == std::abs(best) && w < best)) {
        best = w;
      }
      hh = halfedges[hh].next;
    }
    faceWind[f] = best;
    wAssigned[f] = 1;
  };
  // Outer face is the convention-fixed seed: by topology it sits
  // outside the arrangement, so faceWind = 0. The shoelace area test
  // identified it above.
  //
  // By topology the outer face is unbounded, so winding=0 is the convention.
  // Using that directly avoids a numerically sensitive seed ray-cast.
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
      seedRayCastLeastBiased(localOuter);
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
    // A component's local outer cycle is not necessarily a single global face:
    // other disconnected components can split the space it surrounds. For
    // local outers, classify the side adjacent to this particular edge instead
    // of reusing one propagated winding for the whole cycle.
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
}  // namespace detail

// Primary entry: dispatch on WindRule. Production callers (Boolean2D /
// Xor / Simplify) go through here.
std::vector<OutEdge> FilterByWindingHalfedges(const CanonicalSubEdges& canon,
                                              const std::vector<vec2>& verts,
                                              bool debug, WindRule rule,
                                              Trace* trace) {
#ifndef MANIFOLD_DEBUG
  (void)trace;
#endif
  return detail::FilterByWindingHalfedgesImpl(canon, verts, debug, rule, trace);
}

}  // namespace boolean2
}  // namespace manifold
