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

/**
 * The class first turns input polygons into monotone polygons, then
 * triangulates them using the above class.
 */
class Monotones {
 public:
  Monotones(const Polygons &polys, float precision) : precision_(precision) {
    VertItr start, last, current;
    float bound = 0;
    for (const SimplePolygon &poly : polys) {
      for (int i = 0; i < poly.size(); ++i) {
        monotones_.push_back({poly[i].pos,  //
                              poly[i].idx,  //
                              0, monotones_.end(), monotones_.end(),
                              activePairs_.end(), activePairs_.end()});
        bound = glm::max(
            bound, glm::max(glm::abs(poly[i].pos.x), glm::abs(poly[i].pos.y)));

        current = std::prev(monotones_.end());
        if (i == 0)
          start = current;
        else
          Link(last, current);
        last = current;
      }
      Link(current, start);
    }

    if (precision_ < 0) precision_ = bound * kTolerance;

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
      Triangulator triangulator(start, precision_);
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
      ALWAYS_ASSERT(triangulator.NumTriangles() > 0, topologyErr,
                    "Monotone produced no triangles.");
      triangles_left -= 2 + triangulator.NumTriangles();
      // Find next monotone
      start = std::find_if(monotones_.begin(), monotones_.end(),
                           [](const VertAdj &v) { return !v.Processed(); });
    }
    ALWAYS_ASSERT(triangles_left == 0, topologyErr,
                  "Triangulation produced wrong number of triangles.");
  }

  // A variety of sanity checks on the data structure. Expensive checks are only
  // performed if params.intermediateChecks = true.
  void Check() {
    if (!params.intermediateChecks) return;
    std::vector<Halfedge> edges;
    for (VertItr vert = monotones_.begin(); vert != monotones_.end(); vert++) {
      vert->SetProcessed(false);
      edges.push_back({vert->mesh_idx, vert->right->mesh_idx});
      ALWAYS_ASSERT(vert->right->right != vert, topologyErr,
                    "two-edge monotone!");
      ALWAYS_ASSERT(vert->left->right == vert, topologyErr,
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
  struct VertAdj;
  typedef std::list<VertAdj>::iterator VertItr;
  struct EdgePair;
  typedef std::list<EdgePair>::iterator PairItr;
  enum VertType { START, WESTSIDE, EASTSIDE, MERGE, END, SKIP };

  std::list<VertAdj> monotones_;     // sweep-line list of verts
  std::list<EdgePair> activePairs_;  // west to east list of monotone edge pairs
  std::list<EdgePair> inactivePairs_;  // completed monotones
  float precision_;  // a triangle of this height or less is degenerate

  /**
   * This is the data structure of the polygons themselves. They are stored as a
   * list in sweep-line order. The left and right pointers form the polygons,
   * while the mesh_idx describes the input indices that will be tranfered to
   * the output triangulation. The edgeRight value represents an extra contraint
   * from the mesh Boolean algorithm.
   */
  struct VertAdj {
    glm::vec2 pos;
    int mesh_idx;  // This is a global index into the manifold.
    int index;
    VertItr left, right;
    PairItr eastPair, westPair;

    bool Processed() const { return index < 0; }
    void SetProcessed(bool processed) { index = processed ? -1 : 0; }
    bool IsStart() const {
      return (left->pos.y >= pos.y && right->pos.y > pos.y) ||
             (left->pos.y == pos.y && right->pos.y == pos.y &&
              left->pos.x <= pos.x && right->pos.x < pos.x);
    }
    bool IsPast(const VertItr other, float precision) const {
      return pos.y > other->pos.y + precision;
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
   * The purpose of the certainty booleans is to represent if we're sure the
   * pairs (or monotones) are in the right order. This is uncertain if they are
   * degenerate, for instance if several active edges are colinear (within
   * tolerance). If the order is uncertain, then as each vert is processed, if
   * it yields new information, it can cause the order to be updated until
   * certain.
   */

  struct EdgePair {
    VertItr vWest, vEast, vMerge;
    PairItr nextPair;
    bool westCertain, eastCertain, startCertain;

    int WestOf(VertItr vert, float precision) const {
      int westOf = CCW(vEast->right->pos, vEast->pos, vert->pos, precision);
      if (westOf == 0 && !vert->right->Processed())
        westOf =
            CCW(vEast->right->pos, vEast->pos, vert->right->pos, precision);
      if (westOf == 0 && !vert->left->Processed())
        westOf = CCW(vEast->right->pos, vEast->pos, vert->left->pos, precision);
      return westOf;
    }

    int EastOf(VertItr vert, float precision) const {
      int eastOf = CCW(vWest->pos, vWest->left->pos, vert->pos, precision);
      if (eastOf == 0 && !vert->right->Processed())
        eastOf = CCW(vWest->pos, vWest->left->pos, vert->right->pos, precision);
      if (eastOf == 0 && !vert->left->Processed())
        eastOf = CCW(vWest->pos, vWest->left->pos, vert->left->pos, precision);
      return eastOf;
    }
  };

  /**
   * This class takes sequential verts of a monotone polygon and outputs a
   * geometrically valid triangulation, step by step.
   */
  class Triangulator {
   public:
    Triangulator(VertItr vert, float precision) : precision_(precision) {
      reflex_chain_.push(vert);
      other_side_ = vert;
    }
    int NumTriangles() const { return triangles_output_; }

    /**
     * The vert, vi, must attach to the free end (specified by onRight) of the
     * polygon that has been input so far. The verts must also be processed in
     * sweep-line order to get a geometrically valid result. If not, then the
     * polygon is not monotone, as the result should be topologically valid, but
     * not geometrically. The parameter, last, must be set true only for the
     * final point, as this ensures the last triangle is output.
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
        // This only creates enough triangles to ensure the reflex chain is
        // still reflex.
        if (params.verbose) std::cout << "same chain" << std::endl;
        int ccw = CCW(vi->pos, vj->pos, v_top->pos, precision_);
        while (ccw == (onRight_ ? 1 : -1) || ccw == 0) {
          AddTriangle(triangles, vi, vj, v_top);
          v_top = vj;
          reflex_chain_.pop();
          if (reflex_chain_.empty()) break;
          vj = reflex_chain_.top();
          ccw = CCW(vi->pos, vj->pos, v_top->pos, precision_);
        }
        reflex_chain_.push(v_top);
        reflex_chain_.push(vi);
      } else {
        // This branch empties the reflex chain and switches sides. It must be
        // used for the last point, as it will output all the triangles
        // regardless of geometry.
        if (params.verbose) std::cout << "different chain" << std::endl;
        onRight_ = !onRight_;
        VertItr v_last = v_top;
        while (!reflex_chain_.empty()) {
          vj = reflex_chain_.top();
          AddTriangle(triangles, vi, v_last, vj);
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
    const float precision_;

    void AddTriangle(std::vector<glm::ivec3> &triangles, VertItr v0, VertItr v1,
                     VertItr v2) {
      if (!onRight_) std::swap(v1, v2);
      triangles.emplace_back(v0->mesh_idx, v1->mesh_idx, v2->mesh_idx);
      ++triangles_output_;
      if (params.verbose) std::cout << triangles.back() << std::endl;
    }
  };

  void Link(VertItr left, VertItr right) {
    left->right = right;
    right->left = left;
  }

  void SetVWest(PairItr pair, VertItr vert) {
    pair->vWest = vert;
    vert->eastPair = pair;
  }

  void SetVEast(PairItr pair, VertItr vert) {
    pair->vEast = vert;
    vert->westPair = pair;
  }

  void SetEastCertainty(PairItr westPair, bool certain) {
    westPair->eastCertain = certain;
    std::next(westPair)->westCertain = certain;
  }

  PairItr GetPair(VertItr vert, VertType type) const {
    // MERGE returns westPair, as this is the one that will be removed.
    return type == WESTSIDE ? vert->eastPair : vert->westPair;
  }

  bool Coincident(glm::vec2 p0, glm::vec2 p1) const {
    glm::vec2 sep = p0 - p1;
    return glm::dot(sep, sep) < precision_ * precision_;
  }

  void CloseEnd(VertItr vert) {
    PairItr eastPair = vert->right->eastPair;
    PairItr westPair = vert->left->westPair;
    SetVWest(eastPair, vert);
    SetVEast(westPair, vert);
    westPair->westCertain = true;
    westPair->eastCertain = true;
  }

  /**
   * This function is shared between the forward and backward sweeps and
   * determines the topology of the vertex relative to the sweep line.
   */
  VertType ProcessVert(VertItr vert) {
    PairItr eastPair = vert->right->eastPair;
    PairItr westPair = vert->left->westPair;
    if (vert->right->Processed()) {
      if (vert->left->Processed()) {
        if (westPair == eastPair) {
          // facing in
          if (params.verbose) std::cout << "END" << std::endl;
          CloseEnd(vert);
          return END;
        } else if (westPair != activePairs_.end() &&
                   std::next(westPair) == eastPair) {
          // facing out
          if (params.verbose) std::cout << "MERGE" << std::endl;
          CloseEnd(vert);
          // westPair will be removed and eastPair takes over.
          SetVWest(eastPair, westPair->vWest);
          return MERGE;
        } else {  // not neighbors
          if (params.verbose) std::cout << "SKIP" << std::endl;
          return SKIP;
        }
      } else {
        SetVWest(eastPair, vert);
        if (params.verbose) std::cout << "WESTSIDE" << std::endl;
        return WESTSIDE;
      }
    } else {
      if (vert->left->Processed()) {
        SetVEast(westPair, vert);
        if (params.verbose) std::cout << "EASTSIDE" << std::endl;
        return EASTSIDE;
      } else {
        if (params.verbose) std::cout << "START" << std::endl;
        return START;
      }
    }
  }

  /**
   * Remove this pair, but save it and mark the pair it was next to. When the
   * reverse sweep happens, it will be placed next to its last neighbor instead
   * of using geometry.
   */
  void RemovePair(PairItr pair) {
    ALWAYS_ASSERT(pair != activePairs_.end(), logicErr, "No pair to remove!");
    pair->nextPair = std::next(pair);
    inactivePairs_.splice(inactivePairs_.end(), activePairs_, pair);
  }

  /**
   * When vert is a START, this determines if it is backwards (forming a void or
   * hole). Usually the first return is adequate, but if it is degenerate, the
   * function will continue to search up the neighbors until the degeneracy is
   * broken and a certain answer is returned.
   */
  bool IsHole(VertItr vert) const {
    VertItr left = vert->left;
    VertItr right = vert->right;
    VertItr center = vert;
    // TODO: if left or right is Processed(), determine from east/west
    while (left != right) {
      if (Coincident(left->pos, center->pos)) {
        left = left->left;
        continue;
      }
      if (Coincident(right->pos, center->pos)) {
        right = right->right;
        continue;
      }
      if (Coincident(left->pos, right->pos)) {
        vert = center;
        center = left;
        left = left->left;
        if (left == right) break;
        right = right->right;
        continue;
      }
      int isHole = CCW(right->pos, center->pos, left->pos, precision_);
      if (center != vert) {
        isHole += CCW(left->pos, center->pos, vert->pos, precision_) +
                  CCW(vert->pos, center->pos, right->pos, precision_);
      }
      if (isHole != 0) return isHole > 0;

      glm::vec2 edgeLeft = left->pos - center->pos;
      glm::vec2 edgeRight = right->pos - center->pos;
      if (glm::dot(edgeLeft, edgeRight) > 0) {
        if (glm::dot(edgeLeft, edgeLeft) < glm::dot(edgeRight, edgeRight)) {
          center = left;
          left = left->left;
        } else {
          center = right;
          right = right->right;
        }
      } else {
        if (left->pos.y < right->pos.y) {
          left = left->left;
        } else {
          right = right->right;
        }
      }
    }
    return false;
  }

  /**
   * A backwards pair (hole) must be interior to a forwards pair for geometric
   * validity. In this situation, this function is used to swap their east edges
   * such that they become forward neighbor pairs. The outside becomes westPair
   * and inside becomes eastPair.
   */
  void SwapHole(PairItr outside, PairItr inside) {
    VertItr tmp = outside->vEast;
    SetVEast(outside, inside->vEast);
    SetVEast(inside, tmp);
    inside->eastCertain = outside->eastCertain;

    activePairs_.splice(std::next(outside), activePairs_, inside);
    SetEastCertainty(outside, true);
  }

  /**
   * This is the key function for handling east-west degeneracies, and is the
   * purpose of running the sweep-line forwards and backwards. If the ordering
   * of inputPair is uncertain, this function uses the edge ahead of vert to
   * check if this new bit of geometric information is enough to place the pair
   * with certainty. It can also invert the pair if it is determined to be a
   * hole, in which case the inputPair becomes the eastPair while the pair it is
   * inside of becomes the westPair.
   *
   * This function normally returns false, but will instead return true if the
   * certainties conflict, indicating this vertex is not yet geometrically valid
   * and must be skipped.
   */
  bool ShiftEast(const VertItr vert, const PairItr inputPair,
                 const bool isHole) {
    ALWAYS_ASSERT(inputPair != activePairs_.end(), logicErr,
                  "input pair is not defined!");

    if (inputPair->eastCertain) return false;

    PairItr potentialPair = std::next(inputPair);
    while (potentialPair != activePairs_.end()) {
      const int EastOf = potentialPair->EastOf(vert, precision_);
      // This does not trigger a skip because ShiftWest may still succeed, and
      // if not it will mark the skip.
      if (EastOf > 0 && isHole) return false;

      if (EastOf >= 0 && !isHole) {  // in the right place
        activePairs_.splice(potentialPair, activePairs_, inputPair);
        SetEastCertainty(inputPair, EastOf != 0);
        return false;
      }

      const int outside = potentialPair->WestOf(vert, precision_);
      if (outside <= 0 && isHole) {  // certainly a hole
        SwapHole(potentialPair, inputPair);
        return false;
      }
      ++potentialPair;
    }
    if (isHole) return true;

    activePairs_.splice(activePairs_.end(), activePairs_, inputPair);
    inputPair->eastCertain = true;
    return false;
  }

  /**
   * Identical to the above function, but swapped to search westward instead.
   */
  bool ShiftWest(const VertItr vert, const PairItr inputPair,
                 const bool isHole) {
    ALWAYS_ASSERT(inputPair != activePairs_.end(), logicErr,
                  "input pair is not defined!");

    if (inputPair->westCertain) return false;

    PairItr potentialPair = inputPair;
    while (potentialPair != activePairs_.begin()) {
      --potentialPair;
      const int WestOf = potentialPair->WestOf(vert, precision_);
      if (WestOf > 0 && isHole) return true;

      if (WestOf >= 0 && !isHole) {  // in the right place
        SetEastCertainty(potentialPair, WestOf != 0);
        if (++potentialPair != inputPair)
          activePairs_.splice(potentialPair, activePairs_, inputPair);
        return false;
      }

      const int outside = potentialPair->EastOf(vert, precision_);
      if (outside <= 0 && isHole) {  // certainly a hole
        SwapHole(potentialPair, inputPair);
        return false;
      }
    }
    if (isHole) return true;

    if (inputPair != activePairs_.begin())
      activePairs_.splice(activePairs_.begin(), activePairs_, inputPair);
    inputPair->westCertain = true;
    return false;
  }

  /**
   * This function sweeps forward (South to North) keeping track of the
   * monotones and reordering degenerates (monotone ordering in the x-direction
   * and sweep line ordering in the y-direction). The input polygons (montones_)
   * is not changed during this process.
   */
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
      // fallback for completely degenerate polygons that have no starts.
      VertItr vert = insertAt;
      if (!nextAttached.empty() &&
          (starts.empty() ||
           !nextAttached.top()->IsPast(starts.back(), precision_))) {
        // Prefer neighbors, which may process starts without needing a new
        // pair.
        vert = nextAttached.top();
        nextAttached.pop();
      } else if (!starts.empty()) {
        // Create a new pair with the next vert from the sorted list of starts.
        vert = starts.back();
        starts.pop_back();
      }

      if (params.verbose)
        std::cout << "mesh_idx = " << vert->mesh_idx << std::endl;

      if (vert->Processed()) continue;

      ALWAYS_ASSERT(
          skipped.empty() || !vert->IsPast(skipped.back(), precision_),
          geometryErr,
          "Not Geometrically Valid! None of the skipped verts is valid.");

      VertType type = ProcessVert(vert);

      PairItr newPair = activePairs_.end();
      bool isHole = false;
      if (type == START) {
        newPair = activePairs_.insert(
            activePairs_.begin(), {vert, vert, monotones_.end(),
                                   activePairs_.end(), false, false, false});
        SetVWest(newPair, vert);
        SetVEast(newPair, vert);
        isHole = IsHole(vert);
      }

      const PairItr pair = GetPair(vert, type);
      if (type != SKIP && ShiftEast(vert, pair, isHole)) type = SKIP;
      if (type != SKIP && ShiftWest(vert, pair, isHole)) type = SKIP;

      if (type == SKIP) {
        ALWAYS_ASSERT(std::next(insertAt) != monotones_.end(), geometryErr,
                      "Not Geometrically Valid! Tried to skip final vert.");
        ALWAYS_ASSERT(
            !nextAttached.empty() || !starts.empty(), geometryErr,
            "Not Geometrically Valid! Tried to skip last queued vert.");
        skipped.push_back(vert);
        if (params.verbose) std::cout << "Skipping vert" << std::endl;
        // If a new pair was added, remove it.
        if (newPair != activePairs_.end()) {
          activePairs_.erase(newPair);
          vert->westPair = activePairs_.end();
          vert->eastPair = activePairs_.end();
        }
        continue;
      }

      if (vert == insertAt)
        ++insertAt;
      else
        monotones_.splice(insertAt, monotones_, vert);

      switch (type) {
        case WESTSIDE:
          nextAttached.push(vert->left);
          break;
        case EASTSIDE:
          nextAttached.push(vert->right);
          break;
        case START:
          nextAttached.push(vert->left);
          nextAttached.push(vert->right);
          break;
        case MERGE:
          // Mark merge as hole for sweep-back.
          pair->vMerge = vert;
        case END:
          RemovePair(pair);
          break;
        case SKIP:
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
  }  // namespace

  /**
   * This is the only function that actually changes monotones_; all the rest is
   * bookkeeping. This divides polygons by connecting two verts. It duplicates
   * these verts to break the polygons, then attaches them across to each other
   * with two new edges.
   */
  VertItr SplitVerts(VertItr north, VertItr south) {
    // at split events, add duplicate vertices to end of list and reconnect
    if (params.verbose)
      std::cout << "split from " << north->mesh_idx << " to " << south->mesh_idx
                << std::endl;

    VertItr northEast = monotones_.insert(north, *north);
    Link(north->left, northEast);
    northEast->SetProcessed(true);

    VertItr southEast = monotones_.insert(std::next(south), *south);
    Link(southEast, south->right);
    southEast->SetProcessed(true);

    Link(south, north);
    Link(northEast, southEast);

    return northEast;
  }

  /**
   * This function sweeps back, splitting the input polygons
   * into monotone polygons without doing a single geometric calculation.
   * Instead everything is based on the topology saved from the forward sweep,
   * primarily the relative ordering of new monotones. Even though the sweep is
   * going back, the polygon is considered rotated, so we still refer to
   * sweeping from South to North and the pairs as ordered from West to East
   * (though this is now the opposite order from the forward sweep).
   */
  bool SweepBack() {
    for (auto &vert : monotones_) vert.SetProcessed(false);
    VertItr vert = monotones_.end();
    while (vert != monotones_.begin()) {
      --vert;

      if (params.verbose)
        std::cout << "mesh_idx = " << vert->mesh_idx << std::endl;

      if (vert->Processed()) continue;

      VertType type = ProcessVert(vert);
      ALWAYS_ASSERT(type != SKIP, logicErr,
                    "SKIP should not happen on reverse sweep!");

      PairItr westPair = GetPair(vert, type);
      switch (type) {
        case MERGE: {
          PairItr eastPair = std::next(westPair);
          if (eastPair->vMerge != monotones_.end())
            vert = SplitVerts(vert, eastPair->vMerge);
          eastPair->vMerge = vert;
        }
        case END:
          RemovePair(westPair);
        case WESTSIDE:
        case EASTSIDE:
          if (westPair->vMerge != monotones_.end()) {
            VertItr eastVert = SplitVerts(vert, westPair->vMerge);
            if (type == WESTSIDE) westPair->vWest = eastVert;
            westPair->vMerge = monotones_.end();
          }
          break;
        case START: {
          // Due to sweeping in the opposite direction, east and west are
          // swapped and what was the next pair is now the previous pair and
          // begin and end are swapped.
          PairItr eastPair = westPair;
          westPair = eastPair->nextPair;
          activePairs_.splice(westPair == activePairs_.end()
                                  ? activePairs_.begin()
                                  : std::next(westPair),
                              inactivePairs_, eastPair);

          if (eastPair->vMerge == vert) {  // Hole
            VertItr split = westPair->vMerge != monotones_.end()
                                ? westPair->vMerge
                            : westPair->vWest->pos.y < westPair->vEast->pos.y
                                ? westPair->vWest
                                : westPair->vEast;
            VertItr eastVert = SplitVerts(vert, split);
            westPair->vMerge = monotones_.end();
            eastPair->vMerge = monotones_.end();
            SetVWest(eastPair, eastVert);
            SetVEast(eastPair, split == westPair->vEast ? eastVert->right
                                                        : westPair->vEast);
            SetVEast(westPair, vert);
          } else {  // Start
            SetVWest(eastPair, vert);
            SetVEast(eastPair, vert);
          }
          break;
        }
        case SKIP:
          break;
      }

      vert->SetProcessed(true);

      // Debug
      if (params.verbose) ListPairs();
    }
    return false;
  }

  void ListPairs() const {
    std::cout << "active edges:" << std::endl;
    for (const EdgePair &pair : activePairs_) {
      std::cout << (pair.westCertain ? "certain " : "uncertain ");
      std::cout << "edge West: S = " << pair.vWest->mesh_idx
                << ", N = " << pair.vWest->left->mesh_idx << std::endl;
      if (&*(pair.vWest->eastPair) != &pair)
        std::cout << "west does not point back!" << std::endl;

      std::cout << (pair.eastCertain ? "certain " : "uncertain ");
      std::cout << "edge East: S = " << pair.vEast->mesh_idx
                << ", N = " << pair.vEast->right->mesh_idx << std::endl;
      if (&*(pair.vEast->westPair) != &pair)
        std::cout << "east does not point back!" << std::endl;
    }
  }
};  // namespace

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

/**
 * @brief Triangulates a set of /epsilon-valid polygons.
 *
 * @param polys The set of polygons, wound CCW and representing multiple
 * polygons and/or holes. These have 2D-projected positions as well as
 * references back to the original vertices
 * @param precision The value of epsilon, bounding the uncertainty of the input
 * @return std::vector<glm::ivec3> The triangles, referencing the original
 * vertex indicies.
 */
std::vector<glm::ivec3> Triangulate(const Polygons &polys, float precision) {
  std::vector<glm::ivec3> triangles;
  try {
    Monotones monotones(polys, precision);
    monotones.Triangulate(triangles);
    if (params.intermediateChecks) {
      CheckTopology(triangles, polys);
      CheckGeometry(triangles, polys, precision);
    }
  } catch (const geometryErr &e) {
    if (!params.suppressErrors) {
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
      halfedges.push_back({poly[i - 1].idx, poly[i].idx, -1});
    }
    halfedges.push_back({poly.back().idx, poly[0].idx, -1});
  }
  return halfedges;
}

std::vector<Halfedge> Triangles2Edges(
    const std::vector<glm::ivec3> &triangles) {
  std::vector<Halfedge> halfedges;
  for (const glm::ivec3 &tri : triangles) {
    halfedges.push_back({tri[0], tri[1], -1});
    halfedges.push_back({tri[1], tri[2], -1});
    halfedges.push_back({tri[2], tri[0], -1});
  }
  return halfedges;
}

void CheckTopology(const std::vector<Halfedge> &halfedges) {
  ALWAYS_ASSERT(halfedges.size() % 2 == 0, topologyErr,
                "Odd number of halfedges.");
  size_t n_edges = halfedges.size() / 2;
  std::vector<Halfedge> forward(halfedges.size()), backward(halfedges.size());

  auto end = std::copy_if(halfedges.begin(), halfedges.end(), forward.begin(),
                          [](Halfedge e) { return e.endVert > e.startVert; });
  ALWAYS_ASSERT(std::distance(forward.begin(), end) == n_edges, topologyErr,
                "Half of halfedges should be forward.");
  forward.resize(n_edges);

  end = std::copy_if(halfedges.begin(), halfedges.end(), backward.begin(),
                     [](Halfedge e) { return e.endVert < e.startVert; });
  ALWAYS_ASSERT(std::distance(backward.begin(), end) == n_edges, topologyErr,
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
                  topologyErr, "Forward and backward edge do not match.");
    if (i > 0) {
      ALWAYS_ASSERT(forward[i - 1].startVert != forward[i].startVert ||
                        forward[i - 1].endVert != forward[i].endVert,
                    topologyErr, "Not a 2-manifold.");
      ALWAYS_ASSERT(backward[i - 1].startVert != backward[i].startVert ||
                        backward[i - 1].endVert != backward[i].endVert,
                    topologyErr, "Not a 2-manifold.");
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
                   const Polygons &polys, float precision) {
  std::map<int, glm::vec2> vertPos;
  for (const auto &poly : polys) {
    for (int i = 0; i < poly.size(); ++i) {
      vertPos[poly[i].idx] = poly[i].pos;
    }
  }
  ALWAYS_ASSERT(std::all_of(triangles.begin(), triangles.end(),
                            [&vertPos, precision](const glm::ivec3 &tri) {
                              return CCW(vertPos[tri[0]], vertPos[tri[1]],
                                         vertPos[tri[2]], precision) >= 0;
                            }),
                geometryErr, "triangulation is not entirely CCW!");
}

void Dump(const Polygons &polys) {
  for (auto poly : polys) {
    std::cout << "polys.push_back({" << std::setprecision(9) << std::endl;
    for (auto v : poly) {
      std::cout << "    {glm::vec2(" << v.pos.x << ", " << v.pos.y << "), "
                << v.idx << "},  //" << std::endl;
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
