// Copyright 2019 Emmett Lalish
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

#include "polygon.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>

namespace {
using namespace manifold;

ExecutionParams params;

constexpr float kTolerance = 1e-5;
constexpr float kTolerance2 = kTolerance * kTolerance;

struct VertAdj;
typedef std::list<VertAdj>::iterator VertItr;

struct EdgePair;
typedef std::list<EdgePair>::iterator PairItr;

/**
 * This is the data structure of the polygons themselves. They are stored as a
 * list in sweep-line order. The left and right pointers form the polygons,
 * while the mesh_idx describes the input indices that will be tranfered to the
 * output triangulation. The edgeRight value represents an extra contraint from
 * the mesh Boolean algorithm.
 */
struct VertAdj {
  glm::vec2 pos;
  int mesh_idx;   // This is a global index into the manifold.
  int edgeRight;  // Cannot join identical edges with a triangle.
  int index;
  VertItr left, right;
  PairItr pair;

  bool Processed() const { return index < 0; }
  void SetProcessed(bool processed) { index = processed ? -1 : 0; }
  bool IsStart() const { return left->pos.y >= pos.y && right->pos.y > pos.y; }
  bool IsPast(const VertAdj &other) const {
    return pos.y > other.pos.y + kTolerance;
  }
  bool operator<(const VertAdj &other) const { return pos.y < other.pos.y; }
};

/**
 * The EdgePairs form the two active edges of a monotone polygon as they are
 * being constructed. The sweep-line is horizontal and moves from -y to +y, or
 * South to North. The West edge is a backwards edge while the East edge is
 * forwards, a topological constraint. If the polygon is geometrically valid,
 * then the West edge will also be to the -x side of the East edge, hence the
 * name.
 *
 * The purpose of the certainty booleans is to represent if we're sure the pairs
 * (or monotones) are in the right order. This is uncertain if they are
 * degenerate, for instance if several active edges are colinear (within
 * tolerance). If the order is uncertain, then as each vert is processed, if it
 * yields new information, it can cause the order to be updated until certain.
 */

enum MergeType { WEST, EAST, NONE };

struct EdgePair {
  VertItr vWest, vEast;
  MergeType merge;
  PairItr lastNeighbor;
  bool westCertain, eastCertain;

  bool &getCertainty(bool westSide) {
    return westSide ? westCertain : eastCertain;
  }
  int westOf(VertItr vert) const {
    return CCW(vEast->right->pos, vEast->pos, vert->pos);
  }
  int eastOf(VertItr vert) const {
    return CCW(vWest->pos, vWest->left->pos, vert->pos);
  }
};

bool SharedEdge(glm::ivec2 edges0, glm::ivec2 edges1) {
  return (edges0[0] != Edge::kNoIdx &&
          (edges0[0] == edges1[0] || edges0[0] == edges1[1])) ||
         (edges0[1] != Edge::kNoIdx &&
          (edges0[1] == edges1[0] || edges0[1] == edges1[1]));
}

/**
 * This class takes sequential verts of a monotone polygon and outputs a
 * geometrically valid triangulation, step by step.
 */
class Triangulator {
 public:
  Triangulator(VertItr vert) {
    reflex_chain_.push(vert);
    other_side_ = vert;
  }
  int NumTriangles() { return triangles_output_; }

  /**
   * The vert, vi, must attach to the free end (specified by onRight) of the
   * polygon that has been input so far. The verts must also be processed in
   * sweep-line order to get a geometrically valid result. If not, then the
   * polygon is not monotone, as the result should be topologically valid, but
   * not geometrically. The parameter, last, must be set true only for the final
   * point, as this ensures the last triangle is output.
   */
  void ProcessVert(const VertItr vi, bool onRight, bool last,
                   std::vector<glm::ivec3> &triangles) {
    VertItr v_top = reflex_chain_.top();
    if (reflex_chain_.size() < 2) {
      reflex_chain_.push(vi);
      onRight_ = onRight;
      return;
    }
    reflex_chain_.pop();
    VertItr vj = reflex_chain_.top();
    if (onRight_ == onRight && !last) {
      // This only creates enough triangles to ensure the reflex chain is still
      // reflex.
      if (params.verbose) std::cout << "same chain" << std::endl;
      while (CCW(vi->pos, vj->pos, v_top->pos) == (onRight_ ? 1 : -1) ||
             (CCW(vi->pos, vj->pos, v_top->pos) == 0 && !SharesEdge(vi, vj))) {
        AddTriangle(triangles, vi->mesh_idx, vj->mesh_idx, v_top->mesh_idx);
        v_top = vj;
        reflex_chain_.pop();
        if (reflex_chain_.empty()) break;
        vj = reflex_chain_.top();
      }
      reflex_chain_.push(v_top);
      reflex_chain_.push(vi);
    } else {
      // This branch empties the reflex chain and switches sides. It must be
      // used for the last point, as it will output all the triangles regardless
      // of geometry.
      if (params.verbose) std::cout << "different chain" << std::endl;
      onRight_ = !onRight_;
      VertItr v_last = v_top;
      while (!reflex_chain_.empty()) {
        vj = reflex_chain_.top();
        AddTriangle(triangles, vi->mesh_idx, v_last->mesh_idx, vj->mesh_idx);
        v_last = vj;
        reflex_chain_.pop();
      }
      reflex_chain_.push(v_top);
      reflex_chain_.push(vi);
      other_side_ = v_top;
    }
  }

 private:
  std::stack<VertItr> reflex_chain_;
  VertItr other_side_;  // The end vertex across from the reflex chain
  bool onRight_;        // The side the reflex chain is on
  int triangles_output_ = 0;

  void AddTriangle(std::vector<glm::ivec3> &triangles, int v0, int v1, int v2) {
    // if (v0 == v1 || v1 == v2 || v2 == v0) return;
    if (onRight_)
      triangles.emplace_back(v0, v1, v2);
    else
      triangles.emplace_back(v0, v2, v1);
    ++triangles_output_;
    if (params.verbose) std::cout << triangles.back() << std::endl;
  }

  // This checks the extra edge constraint from the mesh Boolean.
  bool SharesEdge(const VertItr v0, const VertItr v1) {
    glm::ivec2 e0(v0->edgeRight, v0->left->edgeRight);
    glm::ivec2 e1(v1->edgeRight, v1->left->edgeRight);
    return SharedEdge(e0, e1);
  }
};

/**
 * The class first turns input polygons into monotone polygons, then
 * triangulates them using the above class.
 */
class Monotones {
 public:
  Monotones(const Polygons &polys) {
    VertItr start, last, current;
    for (const SimplePolygon &poly : polys) {
      for (int i = 0; i < poly.size(); ++i) {
        monotones_.push_back({poly[i].pos,       //
                              poly[i].idx,       //
                              poly[i].nextEdge,  //
                              0});
        current = std::prev(monotones_.end());
        if (i == 0)
          start = current;
        else
          Link(last, current);
        last = current;
      }
      Link(current, start);
    }

    if (SweepForward()) return;
    Check();

    if (SweepBack()) return;
    Check();
  }

  void Triangulate(std::vector<glm::ivec3> &triangles) {
    // Save the sweep-line order in the vert to check further down.
    int i = 1;
    for (auto &vert : monotones_) {
      vert.index = i++;
    }
    int triangles_left = monotones_.size();
    VertItr start = monotones_.begin();
    while (start != monotones_.end()) {
      if (params.verbose) std::cout << start->mesh_idx << std::endl;
      Triangulator triangulator(start);
      start->SetProcessed(true);
      VertItr vR = start->right;
      VertItr vL = start->left;
      while (vR != vL) {
        // Process the neighbor vert that is next in the sweep-line.
        if (vR->index < vL->index) {
          if (params.verbose) std::cout << vR->mesh_idx << std::endl;
          triangulator.ProcessVert(vR, true, false, triangles);
          vR->SetProcessed(true);
          vR = vR->right;
        } else {
          if (params.verbose) std::cout << vL->mesh_idx << std::endl;
          triangulator.ProcessVert(vL, false, false, triangles);
          vL->SetProcessed(true);
          vL = vL->left;
        }
      }
      if (params.verbose) std::cout << vR->mesh_idx << std::endl;
      triangulator.ProcessVert(vR, true, true, triangles);
      vR->SetProcessed(true);
      // validation
      ALWAYS_ASSERT(triangulator.NumTriangles() > 0, logicErr,
                    "Monotone produced no triangles.");
      triangles_left -= 2 + triangulator.NumTriangles();
      // Find next monotone
      start = std::find_if(monotones_.begin(), monotones_.end(),
                           [](const VertAdj &v) { return !v.Processed(); });
    }
    ALWAYS_ASSERT(triangles_left == 0, logicErr,
                  "Triangulation produced wrong number of triangles.");
  }

  // A variety of sanity checks on the data structure. Expensive checks are only
  // performed if params.intermediateChecks = true.
  void Check() {
    if (!params.intermediateChecks) return;
    std::vector<Halfedge> edges;
    for (VertItr vert = monotones_.begin(); vert != monotones_.end(); vert++) {
      vert->SetProcessed(false);
      edges.push_back({vert->mesh_idx, vert->right->mesh_idx, Edge::kNoIdx});
      ALWAYS_ASSERT(vert->right->right != vert, logicErr, "two-edge monotone!");
      ALWAYS_ASSERT(vert->left->right == vert, logicErr,
                    "monotone vert neighbors don't agree!");
    }
    if (params.verbose) {
      VertItr start = monotones_.begin();
      while (start != monotones_.end()) {
        start->SetProcessed(true);
        std::cout << "monotone start: " << start->mesh_idx << ", "
                  << start->pos.y << std::endl;
        VertItr v = start->right;
        while (v != start) {
          std::cout << v->mesh_idx << ", " << v->pos.y << std::endl;
          v->SetProcessed(true);
          v = v->right;
        }
        std::cout << std::endl;
        start = std::find_if(monotones_.begin(), monotones_.end(),
                             [](const VertAdj &v) { return !v.Processed(); });
      }
    }
  }

 private:
  enum VertType { START, LEFTWARDS, RIGHTWARDS, MERGE, END, SKIP };
  std::list<VertAdj> monotones_;     // sweep-line list of verts
  std::list<EdgePair> activePairs_;  // west to east list of monotone edge pairs
  std::list<EdgePair> inactivePairs_;  // completed monotones

  void Link(VertItr left, VertItr right) {
    left->right = right;
    right->left = left;
  }

  int VertWestOfPair(VertItr vert, PairItr pair) {
    const PairItr pairWest =
        pair->merge == MergeType::WEST ? std::prev(pair) : pair;
    return pairWest->eastOf(vert);
  }

  int VertEastOfPair(VertItr vert, PairItr pair) {
    const PairItr pairEast =
        pair->merge == MergeType::EAST ? std::next(pair) : pair;
    return pairEast->westOf(vert);
  }

  /**
   * Remove this pair, but save it and mark the pair it was next to. When the
   * reverse sweep happens, it will be placed next to its last neighbor instead
   * of using geometry. Note that when sweeping back, the order of the pairs is
   * reversed so we mark the previous neighbor even though this pair will be
   * inserted previous to that mark in the reverse sweep.
   */
  void RemovePair(PairItr pair) {
    pair->lastNeighbor =
        pair == activePairs_.begin() ? activePairs_.end() : std::prev(pair);
    inactivePairs_.splice(inactivePairs_.end(), activePairs_, pair);
  }

  /**
   * This is the only function that actually changes monotones_; all the rest is
   * bookkeeping. This divides or attaches polygons by connecting two verts. It
   * duplicates these verts to break the polygons, then attaches them across to
   * each other with two new edges.
   */
  VertItr SplitVerts(VertItr north) {
    const PairItr pair = north->pair;
    if (pair->merge == MergeType::NONE) return monotones_.end();
    const VertItr south =
        pair->merge == MergeType::WEST ? pair->vWest : pair->vEast;
    const PairItr pairWest =
        pair->merge == MergeType::WEST ? std::prev(pair) : pair;
    const PairItr pairEast =
        pair->merge == MergeType::WEST ? pair : std::next(pair);

    // at split events, add duplicate vertices to end of list and reconnect
    if (params.verbose)
      std::cout << "split from " << north->mesh_idx << " to " << south->mesh_idx
                << std::endl;

    VertItr insertAt = north == pairWest->vWest ? std::next(north) : north;
    VertItr northEast = monotones_.insert(insertAt, *north);
    Link(north->left, northEast);
    northEast->SetProcessed(true);
    north->pair = pairWest;
    northEast->pair = pairEast;

    VertItr southEast = monotones_.insert(south, *south);
    Link(southEast, south->right);
    southEast->SetProcessed(true);
    south->pair = pairWest;
    southEast->pair = pairEast;

    Link(south, north);
    Link(northEast, southEast);

    if (north == pairWest->vWest || north == pairWest->vEast) {
      if (params.verbose) std::cout << "removing pair West" << std::endl;
      RemovePair(pairWest);
    }
    if (north == pairEast->vEast || north == pairEast->vWest) {
      if (params.verbose) std::cout << "removing pair East" << std::endl;
      RemovePair(pairEast);
    }

    pairWest->vEast = north;
    pairEast->vWest = northEast;
    pairWest->merge = MergeType::NONE;
    pairEast->merge = MergeType::NONE;
    return northEast;
  }

  /**
   * This is the key function for handling degeneracies, and is the purpose of
   * running the sweep-line forwards and backwards. Splits are only performed
   * after a merge vert has been found, which means we have as much information
   * as we can about where it is geometrically. This is the function that uses
   * that new information to reorder the uncertain monotone edge pairs.
   *
   * This function is designed to search both ways, with direction chosen by the
   * input boolean, west.
   */
  void Reorder(const VertItr vert, const bool west) {
    const PairItr inputPair = vert->pair;
    PairItr potentialPair = inputPair;

    if (potentialPair->getCertainty(west)) return;

    PairItr end = west ? activePairs_.begin() : std::prev(activePairs_.end());
    while (potentialPair != end) {
      potentialPair =
          west ? std::prev(potentialPair) : std::next(potentialPair);
      int eastOf =
          west ? potentialPair->westOf(vert) : potentialPair->eastOf(vert);
      if (eastOf >= 0) {   // in the right place
        if (eastOf > 0) {  // certain
          inputPair->getCertainty(west) = true;
        }
        PairItr loc = west ? std::next(potentialPair) : potentialPair;
        activePairs_.splice(loc, activePairs_, inputPair);
        break;
      }
      eastOf = -1 * (west ? potentialPair->eastOf(vert)
                          : potentialPair->westOf(vert));
      if (eastOf >= 0) {  // in the right place
        PairItr loc = west ? potentialPair : std::next(potentialPair);
        activePairs_.splice(loc, activePairs_, inputPair);
        inputPair->getCertainty(!west) = false;
        if (eastOf > 0) {  // certainly a hole
          std::swap(potentialPair->vWest, inputPair->vWest);
          potentialPair->vWest->pair = potentialPair;
          inputPair->vWest->pair = inputPair;
          inputPair->eastCertain = true;
          inputPair->westCertain = true;
        }
        break;
      }
    }
  }

  bool SweepForward() {
    // Reversed so that minimum element is at queue.top() / vector.back().
    auto cmp = [](VertItr a, VertItr b) { return *b < *a; };
    std::priority_queue<VertItr, std::vector<VertItr>, decltype(cmp)>
        nextAttached(cmp);

    std::vector<VertItr> starts;
    for (VertItr v = monotones_.begin(); v != monotones_.end(); v++) {
      if (v->IsStart()) {
        starts.push_back(v);
      }
    }
    std::sort(starts.begin(), starts.end(), cmp);

    std::vector<VertItr> skipped;
    VertItr insertAt = monotones_.begin();

    while (insertAt != monotones_.end()) {
      VertItr vert = insertAt;
      if (!nextAttached.empty() &&
          (starts.empty() || !nextAttached.top()->IsPast(*starts.back()))) {
        vert = nextAttached.top();
        nextAttached.pop();
      } else if (!starts.empty()) {
        vert = starts.back();
        starts.pop_back();
      }

      if (params.verbose)
        std::cout << "mesh_idx = " << vert->mesh_idx << std::endl;

      if (vert->Processed()) continue;

      if (!skipped.empty() && vert->IsPast(*skipped.back())) {
        if (params.verbose)
          std::cout
              << "Not Geometrically Valid! None of the skipped verts is valid."
              << std::endl;
        return true;
      }

      VertType type = CategorizeVert(vert);

      PairItr loc = activePairs_.begin();
      int isStart = 0;
      if (type == START) {
        if (activePairs_.empty()) {
          isStart = 1;
        } else {
          for (; loc != activePairs_.end(); ++loc) {
            if (VertEastOfPair(vert, loc) < 0) break;
          }
          isStart = loc == activePairs_.end()
                        ? VertEastOfPair(vert, std::prev(activePairs_.end()))
                        : VertWestOfPair(vert, loc);
        }
        int isStart2 = CCW(vert->left->pos, vert->pos, vert->right->pos);
        // Disagreement is not geometrically valid, so skip to find a better
        // order.
        if (isStart * isStart2 < 0) {
          type = SKIP;
        }
        // Confidence takes precendence.
        isStart += isStart2;
      }

      if (type == SKIP) {
        if (vert == insertAt) {
          if (params.verbose)
            std::cout << "Not Geometrically Valid! Tried to skip final vert."
                      << std::endl;
          return true;
        }
        skipped.push_back(vert);
        if (params.verbose) std::cout << "Skipping vert" << std::endl;
        continue;
      }

      if (vert == insertAt)
        ++insertAt;
      else
        monotones_.splice(insertAt, monotones_, vert);

      switch (type) {
        case LEFTWARDS:
          nextAttached.push(vert->left);
          Leftwards(vert);
          break;
        case RIGHTWARDS:
          nextAttached.push(vert->right);
          Rightwards(vert);
          break;
        case START:
          nextAttached.push(vert->left);
          nextAttached.push(vert->right);
          if (isStart >= 0) {
            Start(vert, loc, isStart);
          } else {
            Hole(vert, loc);
          }
          break;
        case MERGE:
          Merge(vert);
          break;
        case END:
          End(vert);
          break;
      }

      vert->SetProcessed(true);
      // Push skipped verts back into unprocessed queue.
      while (!skipped.empty()) {
        starts.push_back(skipped.back());
        skipped.pop_back();
      }

      // Debug
      if (params.verbose) ListPairs();
    }
    return false;
  }

  bool SweepBack() {
    for (auto &vert : monotones_) vert.SetProcessed(false);
    monotones_.reverse();
    for (VertItr vert = monotones_.begin(); vert != monotones_.end(); ++vert) {
      if (params.verbose)
        std::cout << "mesh_idx = " << vert->mesh_idx << std::endl;

      VertType type = CategorizeVert(vert);

      PairItr newPair = vert->pair;
      switch (type) {
        case LEFTWARDS:
          Leftwards(vert);
          break;
        case RIGHTWARDS:
          Rightwards(vert);
          break;
        case START:
          if (params.verbose) std::cout << "START" << std::endl;
          activePairs_.splice(newPair->lastNeighbor, inactivePairs_, newPair);
          vert->pair->vWest = vert;
          vert->pair->vEast = vert;
          break;
        case MERGE:
          Merge(vert);
          break;
        case END:
          End(vert);
          break;
        case SKIP:
          std::cout << "SKIP should not happen on reverse sweep!" << std::endl;
          return true;
      }

      vert->SetProcessed(true);

      // Debug
      if (params.verbose) ListPairs();
    }
    return false;
  }

  VertType CategorizeVert(VertItr vert) {
    if (vert->right->Processed()) {
      if (vert->left->Processed()) {
        Reorder(vert, true);  // TODO: this might need some thought...
        Reorder(vert, false);
        PairItr pairLeft = vert->left->pair;
        PairItr pairRight = vert->right->pair;

        vert->pair = pairLeft;
        if (pairLeft == pairRight || (pairRight != activePairs_.end() &&
                                      std::next(pairRight) == pairLeft)) {
          // facing in
          return END;
        } else if (pairLeft != activePairs_.end() &&
                   std::next(pairLeft) == pairRight) {
          // facing out
          return MERGE;
        } else {  // not neighbors
          return SKIP;
        }
      } else {
        Reorder(vert, true);
        Reorder(vert, false);
        return LEFTWARDS;
      }
    } else {
      if (vert->left->Processed()) {
        Reorder(vert, true);
        Reorder(vert, false);
        return RIGHTWARDS;
      } else {
        return START;
      }
    }
  }

  void Leftwards(VertItr vert) {
    if (params.verbose) std::cout << "LEFTWARDS" << std::endl;
    vert->pair->vWest = vert;
    vert->left->pair = vert->pair;
    SplitVerts(vert);
  }

  void Rightwards(VertItr vert) {
    if (params.verbose) std::cout << "RIGHTWARDS" << std::endl;
    vert->pair->vEast = vert;
    vert->right->pair = vert->pair;
    SplitVerts(vert);
  }

  void Start(VertItr vert, PairItr loc, int isStart) {
    if (params.verbose) std::cout << "START" << std::endl;
    bool westCertain =
        loc == activePairs_.begin() || VertEastOfPair(vert, loc) > 0;
    bool eastCertain = isStart > 0;
    vert->pair = activePairs_.insert(
        loc, {vert, vert, MergeType::NONE, activePairs_.end(), westCertain,
              eastCertain});
    vert->left->pair = vert->pair;
    vert->right->pair = vert->pair;
  }

  void Hole(VertItr vert, PairItr loc) {
    if (params.verbose) std::cout << "HOLE" << std::endl;
    vert->pair = loc;
    VertItr vertEast = SplitVerts(vert);
    // If a split occurred then no pairs have to change.
    if (vertEast != monotones_.end()) {
      vert->right->pair = vert->pair;
      vertEast->left->pair = vertEast->pair;
      return;
    }

    if (loc == activePairs_.end()) --loc;
    PairItr pairWest = activePairs_.insert(
        loc,
        {loc->vWest, vert, MergeType::NONE, activePairs_.end(), true, true});
    vert->pair = pairWest;
    loc->vWest->pair = pairWest;
    loc->vWest->left->pair = pairWest;
    loc->merge = MergeType::NONE;

    vert->left->pair = loc;
    vert->right->pair = pairWest;
    loc->vWest = vert;
  }

  void Merge(VertItr vert) {
    if (params.verbose) std::cout << "MERGE" << std::endl;
    PairItr pairWest = vert->left->pair;
    PairItr pairEast = vert->right->pair;
    pairWest->vEast = vert;
    pairEast->vWest = vert;

    vert->pair = pairEast;
    const VertItr vertEast = SplitVerts(vert);
    if (vertEast != monotones_.end()) {
      vert = vertEast;
      pairWest->vEast = vert;
    }
    pairEast = vert->right->pair;

    vert->pair = pairWest;
    SplitVerts(vert);

    pairWest = vert->left->pair;
    pairWest->merge = MergeType::EAST;
    pairEast->merge = MergeType::WEST;
    pairEast->vWest = vert;
  }

  void End(VertItr vert) {
    if (params.verbose) std::cout << "END" << std::endl;
    PairItr pairWest = vert->right->pair;
    PairItr pairEast = vert->left->pair;

    pairWest->vWest = vert;
    pairEast->vEast = vert;

    const VertItr vertEast = SplitVerts(vert);
    // If a split occurred then both pairs have already been removed.
    if (vertEast != monotones_.end()) return;

    RemovePair(vert->pair);
  }

  void ListPairs() {
    std::cout << "active edges:" << std::endl;
    for (EdgePair &pair : activePairs_) {
      if (pair.merge == MergeType::WEST) std::cout << "merge West" << std::endl;
      // std::cout << (pair.westCertain ? "certain " : "uncertain ");
      std::cout << "edge West: S = " << pair.vWest->mesh_idx
                << ", N = " << pair.vWest->left->mesh_idx << std::endl;

      // std::cout << (pair.eastCertain ? "certain " : "uncertain ");
      std::cout << "edge East: S = " << pair.vEast->mesh_idx
                << ", N = " << pair.vEast->right->mesh_idx << std::endl;
      if (pair.merge == MergeType::EAST) std::cout << "merge East" << std::endl;
    }
  }
};

void PrintFailure(const std::exception &e, const Polygons &polys,
                  std::vector<glm::ivec3> &triangles) {
  std::cout << "-----------------------------------" << std::endl;
  std::cout << "Triangulation failed!" << std::endl;
  std::cout << e.what() << std::endl;
  Dump(polys);
  std::cout << "produced this triangulation:" << std::endl;
  for (int j = 0; j < triangles.size(); ++j) {
    std::cout << triangles[j][0] << ", " << triangles[j][1] << ", "
              << triangles[j][2] << std::endl;
  }
}
}  // namespace

namespace manifold {
// This is nearly the only function to do a floating point comparison in this
// whole triangulator (the other is the check for sweep-line degeneracies).
// This is done to maintain maximum consistency.
int CCW(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2) {
  glm::vec2 v1 = p1 - p0;
  glm::vec2 v2 = p2 - p0;
  float area = v1.x * v2.y - v1.y * v2.x;
  float base2 = glm::max(glm::dot(v1, v1), glm::dot(v2, v2));
  if (area * area <= base2 * kTolerance2)
    return 0;
  else
    return area > 0 ? 1 : -1;
}

std::vector<glm::ivec3> Triangulate(const Polygons &polys) {
  std::vector<glm::ivec3> triangles;
  try {
    Monotones monotones(polys);
    monotones.Triangulate(triangles);
    CheckTopology(triangles, polys);
    if (params.checkGeometry) CheckGeometry(triangles, polys);
  } catch (const runtimeErr &e) {
    if (params.checkGeometry && !params.suppressErrors) {
      PrintFailure(e, polys, triangles);
    }
    throw;
  } catch (const std::exception &e) {
    PrintFailure(e, polys, triangles);
    throw;
  }
  return triangles;
}

std::vector<Halfedge> Polygons2Edges(const Polygons &polys) {
  std::vector<Halfedge> halfedges;
  for (const auto &poly : polys) {
    for (int i = 1; i < poly.size(); ++i) {
      halfedges.push_back(
          {poly[i - 1].idx, poly[i].idx, -1, poly[i - 1].nextEdge});
    }
    halfedges.push_back(
        {poly.back().idx, poly[0].idx, -1, poly.back().nextEdge});
  }
  return halfedges;
}

std::vector<Halfedge> Triangles2Edges(
    const std::vector<glm::ivec3> &triangles) {
  std::vector<Halfedge> halfedges;
  for (const glm::ivec3 &tri : triangles) {
    // Differentiate edges of triangles by setting index to Edge::kInterior.
    halfedges.push_back({tri[0], tri[1], -1, Edge::kInterior});
    halfedges.push_back({tri[1], tri[2], -1, Edge::kInterior});
    halfedges.push_back({tri[2], tri[0], -1, Edge::kInterior});
  }
  return halfedges;
}

void CheckTopology(const std::vector<Halfedge> &halfedges) {
  ALWAYS_ASSERT(halfedges.size() % 2 == 0, runtimeErr,
                "Odd number of halfedges.");
  size_t n_edges = halfedges.size() / 2;
  std::vector<Halfedge> forward(halfedges.size()), backward(halfedges.size());

  auto end = std::copy_if(halfedges.begin(), halfedges.end(), forward.begin(),
                          [](Halfedge e) { return e.endVert > e.startVert; });
  ALWAYS_ASSERT(std::distance(forward.begin(), end) == n_edges, runtimeErr,
                "Half of halfedges should be forward.");
  forward.resize(n_edges);

  end = std::copy_if(halfedges.begin(), halfedges.end(), backward.begin(),
                     [](Halfedge e) { return e.endVert < e.startVert; });
  ALWAYS_ASSERT(std::distance(backward.begin(), end) == n_edges, runtimeErr,
                "Half of halfedges should be backward.");
  backward.resize(n_edges);

  std::for_each(backward.begin(), backward.end(),
                [](Halfedge &e) { std::swap(e.startVert, e.endVert); });
  auto cmp = [](const Halfedge &a, const Halfedge &b) {
    return a.startVert < b.startVert ||
           (a.startVert == b.startVert && a.endVert < b.endVert);
  };
  std::sort(forward.begin(), forward.end(), cmp);
  std::sort(backward.begin(), backward.end(), cmp);
  for (int i = 0; i < n_edges; ++i) {
    ALWAYS_ASSERT(forward[i].startVert == backward[i].startVert &&
                      forward[i].endVert == backward[i].endVert,
                  runtimeErr, "Forward and backward edge do not match.");
    if (i > 0) {
      ALWAYS_ASSERT(forward[i - 1].startVert != forward[i].startVert ||
                        forward[i - 1].endVert != forward[i].endVert,
                    runtimeErr, "Not a 2-manifold.");
      ALWAYS_ASSERT(backward[i - 1].startVert != backward[i].startVert ||
                        backward[i - 1].endVert != backward[i].endVert,
                    runtimeErr, "Not a 2-manifold.");
    }
  }
  // Check that no interior edges link vertices that share the same edge data.
  std::map<int, glm::ivec2> vert2edges;
  for (Halfedge halfedge : halfedges) {
    if (halfedge.face == Edge::kInterior)
      continue;  // only interested in polygon edges
    auto vert = vert2edges.emplace(halfedge.startVert,
                                   glm::ivec2(halfedge.face, Edge::kInvalid));
    if (!vert.second) (vert.first->second)[1] = halfedge.face;

    vert = vert2edges.emplace(halfedge.endVert,
                              glm::ivec2(halfedge.face, Edge::kInvalid));
    if (!vert.second) (vert.first->second)[1] = halfedge.face;
  }
  for (int i = 0; i < n_edges; ++i) {
    if (forward[i].face == Edge::kInterior &&
        backward[i].face == Edge::kInterior) {
      glm::ivec2 TwoEdges0 = vert2edges.find(forward[i].startVert)->second;
      glm::ivec2 TwoEdges1 = vert2edges.find(forward[i].endVert)->second;
      ALWAYS_ASSERT(!SharedEdge(TwoEdges0, TwoEdges1), runtimeErr,
                    "Added an interface edge!");
    }
  }
}

void CheckTopology(const std::vector<glm::ivec3> &triangles,
                   const Polygons &polys) {
  std::vector<Halfedge> halfedges = Triangles2Edges(triangles);
  std::vector<Halfedge> openEdges = Polygons2Edges(polys);
  for (Halfedge e : openEdges) {
    halfedges.push_back({e.endVert, e.startVert, -1, e.face});
  }
  CheckTopology(halfedges);
}

void CheckGeometry(const std::vector<glm::ivec3> &triangles,
                   const Polygons &polys) {
  std::map<int, glm::vec2> vertPos;
  for (const auto &poly : polys) {
    for (int i = 0; i < poly.size(); ++i) {
      vertPos[poly[i].idx] = poly[i].pos;
    }
  }
  ALWAYS_ASSERT(std::all_of(triangles.begin(), triangles.end(),
                            [&vertPos](const glm::ivec3 &tri) {
                              return CCW(vertPos[tri[0]], vertPos[tri[1]],
                                         vertPos[tri[2]]) >= 0;
                            }),
                runtimeErr, "triangulation is not entirely CCW!");
}

void Dump(const Polygons &polys) {
  for (auto poly : polys) {
    std::cout << "polys.push_back({" << std::setprecision(9) << std::endl;
    for (auto v : poly) {
      std::cout << "    {glm::vec2(" << v.pos.x << ", " << v.pos.y << "), "
                << v.idx << ", " << v.nextEdge << "},  //" << std::endl;
    }
    std::cout << "});" << std::endl;
  }
  for (auto poly : polys) {
    std::cout << "array([" << std::endl;
    for (auto v : poly) {
      std::cout << "  [" << v.pos.x << ", " << v.pos.y << "]," << std::endl;
    }
    std::cout << "])" << std::endl;
  }
}

ExecutionParams &PolygonParams() { return params; }

}  // namespace manifold