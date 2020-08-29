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
struct ActiveEdge {
  VertAdj *vSouth, *vNorth;
};

struct EdgePair {
  ActiveEdge west, east;
  VertItr vMerge;
  bool westCertain, eastCertain;

  bool &getCertainty(bool westSide) {
    return westSide ? westCertain : eastCertain;
  }
  ActiveEdge &getEdge(bool westSide) { return westSide ? west : east; }
};

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
  int mesh_idx;  // This is a global index into the manifold.
  int edgeRight; // Cannot join identical edges with a triangle.
  VertAdj *left, *right;
  int index;
  PairItr leftPair, rightPair;

  bool Processed() const { return index < 0; }
  void SetProcessed(bool processed) { index = processed ? -1 : 0; }
  bool IsStart() const { return left->pos.y >= pos.y && right->pos.y > pos.y; }
  bool operator<(const VertAdj &other) const { return pos.y < other.pos.y; }
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
  Triangulator(const VertAdj *vert) {
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
  void ProcessVert(const VertAdj *vi, bool onRight, bool last,
                   std::vector<glm::ivec3> &triangles) {
    const VertAdj *v_top = reflex_chain_.top();
    if (reflex_chain_.size() < 2) {
      reflex_chain_.push(vi);
      onRight_ = onRight;
      return;
    }
    reflex_chain_.pop();
    const VertAdj *vj = reflex_chain_.top();
    if (onRight_ == onRight && !last) {
      // This only creates enough triangles to ensure the reflex chain is still
      // reflex.
      if (params.verbose)
        std::cout << "same chain" << std::endl;
      while (CCW(vi->pos, vj->pos, v_top->pos) == (onRight_ ? 1 : -1) ||
             (CCW(vi->pos, vj->pos, v_top->pos) == 0 && !SharesEdge(vi, vj))) {
        AddTriangle(triangles, vi->mesh_idx, vj->mesh_idx, v_top->mesh_idx);
        v_top = vj;
        reflex_chain_.pop();
        if (reflex_chain_.empty())
          break;
        vj = reflex_chain_.top();
      }
      reflex_chain_.push(v_top);
      reflex_chain_.push(vi);
    } else {
      // This branch empties the reflex chain and switches sides. It must be
      // used for the last point, as it will output all the triangles regardless
      // of geometry.
      if (params.verbose)
        std::cout << "different chain" << std::endl;
      onRight_ = !onRight_;
      const VertAdj *v_last = v_top;
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
  std::stack<const VertAdj *> reflex_chain_;
  const VertAdj *other_side_; // The end vertex across from the reflex chain
  bool onRight_;              // The side the reflex chain is on
  int triangles_output_ = 0;

  void AddTriangle(std::vector<glm::ivec3> &triangles, int v0, int v1, int v2) {
    // if (v0 == v1 || v1 == v2 || v2 == v0) return;
    if (onRight_)
      triangles.emplace_back(v0, v1, v2);
    else
      triangles.emplace_back(v0, v2, v1);
    ++triangles_output_;
    if (params.verbose)
      std::cout << triangles.back() << std::endl;
  }

  // This checks the extra edge constraint from the mesh Boolean.
  bool SharesEdge(const VertAdj *v0, const VertAdj *v1) {
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
    VertAdj *start, *last, *current;
    for (const SimplePolygon &poly : polys) {
      for (int i = 0; i < poly.size(); ++i) {

        monotones_.push_back({poly[i].pos,      //
                              poly[i].idx,      //
                              poly[i].nextEdge, //
                              nullptr,          //
                              nullptr,          //
                              0});
        current = &monotones_.back();
        if (i == 0)
          start = current;
        else
          Link(last, current);
        last = current;
      }
      Link(current, start);
    }

    // Reversed so that minimum element is at queue.top() / vector.back().
    auto cmp = [](VertItr a, VertItr b) { return *b < *a; };
    std::priority_queue<VertItr, std::vector<VertItr>, decltype(cmp)>
        nextAttached;

    std::vector<VertItr> starts;
    for (VertItr v = monotones_.begin(); v != monotones_.end(); v++) {
      if (v->IsStart()) {
        starts.push_back(v);
      }
    }
    std::sort(starts.begin(), starts.end(), cmp);

    Sweep(nextAttached, starts); // forward
    Check();
    for (auto &vert : monotones_) {
      vert.pos *= -1;
      vert.SetProcessed(false);
    }
    monotones_.reverse();
    Sweep(nextAttached, starts); // backward
    Check();
  }

  void Triangulate(std::vector<glm::ivec3> &triangles) {
    // Save the sweep-line order in the vert to check further down.
    int i = 1;
    for (auto &vert : monotones_) {
      vert.index = i++;
    }
    int triangles_left = monotones_.size();
    VertAdj *start = &monotones_.front();
    while (1) {
      if (params.verbose)
        std::cout << start->mesh_idx << std::endl;
      Triangulator triangulator(start);
      start->SetProcessed(true);
      VertAdj *vR = start->right;
      VertAdj *vL = start->left;
      while (vR != vL) {
        // Process the neighbor vert that is next in the sweep-line.
        if (vR->index < vL->index) {
          if (params.verbose)
            std::cout << vR->mesh_idx << std::endl;
          triangulator.ProcessVert(vR, true, false, triangles);
          vR->SetProcessed(true);
          vR = vR->right;
        } else {
          if (params.verbose)
            std::cout << vL->mesh_idx << std::endl;
          triangulator.ProcessVert(vL, false, false, triangles);
          vL->SetProcessed(true);
          vL = vL->left;
        }
      }
      if (params.verbose)
        std::cout << vR->mesh_idx << std::endl;
      triangulator.ProcessVert(vR, true, true, triangles);
      vR->SetProcessed(true);
      // validation
      ALWAYS_ASSERT(triangulator.NumTriangles() > 0, logicErr,
                    "Monotone produced no triangles.");
      triangles_left -= 2 + triangulator.NumTriangles();
      // Find next monotone
      VertItr itr =
          std::find_if(monotones_.begin(), monotones_.end(),
                       [](const VertAdj &v) { return !v.Processed(); });
      if (itr == monotones_.end())
        break;
      start = &*itr;
    }
    ALWAYS_ASSERT(triangles_left == 0, logicErr,
                  "Triangulation produced wrong number of triangles.");
  }

  // A variety of sanity checks on the data structure. Expensive checks are only
  // performed if params.intermediateChecks = true.
  void Check() {
    ALWAYS_ASSERT(activePairs_.empty(), logicErr,
                  "There are still active edges.");
    if (!params.intermediateChecks)
      return;
    std::vector<Halfedge> edges;
    for (auto &vert : monotones_) {
      vert.SetProcessed(false);
      edges.push_back({vert.mesh_idx, vert.right->mesh_idx, Edge::kNoIdx});
      ALWAYS_ASSERT(vert.right->right != &vert, logicErr, "two-edge monotone!");
      ALWAYS_ASSERT(vert.left->right == &vert, logicErr,
                    "monotone vert neighbors don't agree!");
    }
    if (params.verbose) {
      VertAdj *start = &monotones_.front();
      while (1) {
        start->SetProcessed(true);
        std::cout << "monotone start: " << start->mesh_idx << ", "
                  << start->pos.y << std::endl;
        VertAdj *v = start->right;
        while (v != start) {
          std::cout << v->mesh_idx << ", " << v->pos.y << std::endl;
          v->SetProcessed(true);
          v = v->right;
        }
        std::cout << std::endl;
        VertItr itr =
            std::find_if(monotones_.begin(), monotones_.end(),
                         [](const VertAdj &v) { return !v.Processed(); });
        if (itr == monotones_.end())
          break;
        start = &*itr;
      }
    }
  }

private:
  enum VertType { START, LEFTWARDS, RIGHTWARDS, MERGE, END, SKIP };
  std::list<VertAdj> monotones_;    // sweep-line list of verts
  std::list<EdgePair> activePairs_; // west to east list of monotone edge pairs

  void Link(VertAdj *left, VertAdj *right) {
    left->right = right;
    right->left = left;
  }

  /**
   * This is the only function that actually changes monotones_; all the rest is
   * bookkeeping. This divides or attaches polygons by connecting two verts. It
   * duplicates these verts to break the polygons, then attaches them across to
   * each other with two new edges.
   */
  VertItr SplitVerts(VertItr north, PairItr pair, bool NEisNofNW = false) {
    VertItr south = pair->vMerge;
    if (south == monotones_.end())
      return monotones_.end();

    // at split events, add duplicate vertices to end of list and reconnect
    if (params.verbose)
      std::cout << "split from " << north->mesh_idx << " to " << south->mesh_idx
                << std::endl;

    VertItr insertAt = NEisNofNW ? std::next(north) : north;
    VertItr northEast = monotones_.insert(insertAt, *north);
    northEast->SetProcessed(true);
    Link(north->left, &*northEast);

    VertItr southEast = monotones_.insert(south, *south);
    Link(&*southEast, south->right);

    Link(&*south, &*north);
    Link(&*northEast, &*southEast);

    pair->vMerge = monotones_.end();
    return northEast;
  }

  // If the first result is degenerate, its neighbor is used to attempt a
  // tie-break.
  int VertWestOfEdge(const VertAdj *vert, const ActiveEdge &edge) {
    glm::vec2 last = edge.vSouth->pos;
    glm::vec2 next = edge.vNorth->pos;
    int side = CCW(last, next, vert->pos);
    if (side == 0)
      side = CCW(vert->pos, next, vert->right->pos);
    return side;
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
  void Reorder(const VertAdj *vert, const PairItr inputPair, const bool west) {
    PairItr potentialPair = inputPair;
    if (!potentialPair->getCertainty(west)) {
      int sign = west ? -1 : 1;
      PairItr end = west ? activePairs_.begin() : std::prev(activePairs_.end());
      while (potentialPair != end) {
        potentialPair =
            west ? std::prev(potentialPair) : std::next(potentialPair);
        int eastOf = sign * VertWestOfEdge(vert, potentialPair->getEdge(!west));
        if (eastOf >= 0) {  // in the right place
          if (eastOf > 0) { // certain
            inputPair->getCertainty(west) = true;
          }
          PairItr loc = west ? std::next(potentialPair) : potentialPair;
          activePairs_.splice(loc, activePairs_, inputPair);
          break;
        }
        eastOf = sign * VertWestOfEdge(vert, potentialPair->getEdge(west));
        if (eastOf >= 0) { // in the right place
          PairItr loc = west ? potentialPair : std::next(potentialPair);
          activePairs_.splice(loc, activePairs_, inputPair);
          inputPair->getCertainty(!west) = false;
          if (eastOf > 0) { // certainly a hole
            std::swap(potentialPair->west, inputPair->west);
            potentialPair->west.vSouth->leftPair = potentialPair;
            inputPair->west.vSouth->leftPair = inputPair;
            inputPair->eastCertain = true;
            inputPair->westCertain = true;
          }
          break;
        }
      }
    }
  }

  template <typename T>
  void Sweep(T &nextAttached, std::vector<VertItr> &starts) {
    std::vector<VertItr> skipped;
    VertItr pos = monotones_.begin();
    while (pos != monotones_.end()) {
      VertItr vert = pos;
      if (!nextAttached.empty()) {
        vert = nextAttached.top();
        nextAttached.pop();
      } else if (!starts.empty()) {
        vert = starts.back();
        starts.pop_back();
      }

      if (vert->Processed())
        continue;

      if (!skipped.empty() &&
          vert->pos.y > skipped.back()->pos.y + kTolerance) {
        std::cout << "Not Geometrically Valid!" << std::endl;
        break;
      }

      VertType type = CategorizeVert(vert);

      auto loc = activePairs_.end();
      int isStart = false;
      if (type = START) {
        loc = std::find_if(activePairs_.begin(), activePairs_.end(),
                           [vert, this](const EdgePair &pair) {
                             return VertWestOfEdge(&*vert, pair.east) > 0;
                           });
        isStart =
            loc == activePairs_.end() ? 1 : VertWestOfEdge(&*vert, loc->west);
        int isStart2 = CCW(vert->left->pos, vert->pos, vert->right->pos);
        if (isStart * isStart2 < 0) {
          type = SKIP;
        }
      }

      if (type == SKIP) {
        skipped.push_back(vert);
        continue;
      }

      monotones_.splice(pos, monotones_, vert);

      switch (type) {
      case LEFTWARDS:
        Leftwards(vert);
        nextAttached.push(vert->left);
        break;
      case RIGHTWARDS:
        Rightwards(vert);
        nextAttached.push(vert->right);
        break;
      case START:
        if (isStart >= 0) {
          Start(vert, loc, isStart);
        } else {
          Hole(vert, loc);
        }
        nextAttached.push(vert->left);
        nextAttached.push(vert->right);
        break;
      case MERGE:
        Merge(vert);
        break;
      case END:
        End(vert);
        break;
      }

      vert->SetProcessed(true);
      while (!skipped.empty()) {
        starts.push_back(skipped.back());
        skipped.pop_back();
      }

      if (params.intermediateChecks)
        ListPairs();
    }
  }

  VertType CategorizeVert(VertItr vert) {
    if (vert->right->Processed()) {
      if (vert->left->Processed()) {
        const PairItr leftPair = vert->left->rightPair;
        const PairItr rightPair = vert->right->leftPair;
        if (leftPair == rightPair) { // facing in
          return END;
        } else if (leftPair != activePairs_.end() &&
                   std::next(leftPair) == rightPair) { // facing out
          return MERGE;
        } else {
          return SKIP;
        }
      } else {
        return LEFTWARDS;
      }
    } else {
      if (vert->left->Processed()) {
        return RIGHTWARDS;
      } else {
        return START;
      }
    }
  }

  void Leftwards(VertItr vert) {
    vert->leftPair = vert->right->leftPair;
    ActiveEdge &activeEdge = vert->leftPair->west;
    activeEdge.vSouth = &*vert;
    activeEdge.vNorth = vert->left;
    Reorder(&*vert, vert->leftPair, true);
    Reorder(&*vert, vert->leftPair, false);
    const VertItr newVert = SplitVerts(vert, vert->leftPair, true);
    if (newVert != monotones_.end())
      newVert->leftPair->west.vSouth = &*newVert;
  }

  void Rightwards(VertItr vert) {
    vert->rightPair = vert->left->rightPair;
    ActiveEdge &activeEdge = vert->rightPair->east;
    activeEdge.vSouth = &*vert;
    activeEdge.vNorth = vert->right;
    Reorder(&*vert, vert->rightPair, true);
    Reorder(&*vert, vert->rightPair, false);
    SplitVerts(vert, vert->rightPair, false);
  }

  void Start(VertItr vert, std::list<EdgePair>::iterator loc, int isStart) {
    bool westCertain = loc == activePairs_.begin() ||
                       VertWestOfEdge(&*vert, std::prev(loc)->east) < 0;
    bool eastCertain = isStart > 0;
    vert->leftPair = activePairs_.insert(loc, {{&*vert, vert->left},
                                               {&*vert, vert->right},
                                               monotones_.end(),
                                               westCertain,
                                               eastCertain});
    vert->rightPair = vert->leftPair;
  }

  void Hole(VertItr vert, std::list<EdgePair>::iterator loc) {
    if (loc == activePairs_.end())
      --loc;
    // hole-starting vert links earlier activePair
    vert->rightPair = activePairs_.insert(
        loc, {loc->west, {&*vert, vert->right}, loc->vMerge, true, true});
    loc->west.vSouth->leftPair = vert->rightPair;
    loc->vMerge = monotones_.end();
    VertItr newVert = SplitVerts(vert, vert->rightPair);
    if (newVert != monotones_.end())
      vert = newVert;
    vert->leftPair = loc;
    loc->west = {&*vert, vert->left};
  }

  void Merge(VertItr vert) {
    const PairItr leftPair = vert->left->rightPair;
    const PairItr rightPair = vert->right->leftPair;
    const VertItr newVert = SplitVerts(vert, rightPair, true);
    if (newVert != monotones_.end())
      vert = newVert;
    SplitVerts(vert, leftPair, false);
    leftPair->east = rightPair->east;
    leftPair->east.vSouth->rightPair = leftPair;
    if (std::next(leftPair) == rightPair) {
      leftPair->vMerge = vert;
    }
  }

  void End(VertItr vert) {
    const PairItr rightPair = vert->right->leftPair;
    SplitVerts(vert, rightPair);
  }

  // // Central function for processing each sweep-line vert.
  // VertItr ProcessVert(VertItr vert) {
  //   // Skip newly duplicated verts
  //   if (vert->Processed())
  //     return vert;
  //   if (params.verbose)
  //     std::cout << "mesh_idx = " << vert->mesh_idx << std::endl;
  //   if (vert->right->Processed()) {
  //     if (vert->left->Processed()) {
  //       const PairItr leftPair = vert->left->rightPair;
  //       const PairItr rightPair = vert->right->leftPair;
  //       if (leftPair == rightPair) { // facing in
  //         if (params.verbose)
  //           std::cout << "END" << std::endl;
  //         SplitVerts(vert, rightPair);
  //       } else if (leftPair != activePairs_.end() &&
  //                  std::next(leftPair) == rightPair) { // facing out
  //         if (params.verbose)
  //           std::cout << "MERGE" << std::endl;
  //         const VertItr newVert = SplitVerts(vert, rightPair, true);
  //         if (newVert != monotones_.end())
  //           vert = newVert; // TODO: causing erroneous sweep splice?

  //         SplitVerts(vert, leftPair, false);
  //         leftPair->east = rightPair->east;
  //         leftPair->east.vSouth->rightPair = leftPair;
  //         if (std::next(leftPair) == rightPair) {
  //           leftPair->vMerge = vert;
  //         }
  //       } else {
  //         const VertItr nextVert = std::next(vert);
  //         if (vert != monotones_.end() &&
  //             nextVert->pos.y - vert->pos.y < kTolerance) {
  //           if (params.verbose)
  //             std::cout << "SKIP" << std::endl;
  //           return ProcessVert(nextVert);
  //         } else {
  //           if (params.verbose)
  //             std::cout << "SELF-INTERSECTION" << std::endl;
  //         }
  //       }
  //       activePairs_.erase(rightPair);
  //     } else {
  //       if (params.verbose)
  //         std::cout << "LEFTWARDS" << std::endl;
  //       // update edge
  //       vert->leftPair = vert->right->leftPair;
  //       ActiveEdge &activeEdge = vert->leftPair->west;
  //       activeEdge.vSouth = &*vert;
  //       activeEdge.vNorth = vert->left;
  //       Reorder(&*vert, vert->leftPair, true);
  //       Reorder(&*vert, vert->leftPair, false);
  //       const VertItr newVert = SplitVerts(vert, vert->leftPair, true);
  //       if (newVert != monotones_.end())
  //         newVert->leftPair->west.vSouth = &*newVert;
  //     }
  //   } else {
  //     if (vert->left->Processed()) {
  //       if (params.verbose)
  //         std::cout << "RIGHTWARDS" << std::endl;
  //       // update edge
  //       vert->rightPair = vert->left->rightPair;
  //       ActiveEdge &activeEdge = vert->rightPair->east;
  //       activeEdge.vSouth = &*vert;
  //       activeEdge.vNorth = vert->right;
  //       Reorder(&*vert, vert->rightPair, true);
  //       Reorder(&*vert, vert->rightPair, false);
  //       SplitVerts(vert, vert->rightPair, false);
  //     } else {
  //       auto loc = std::find_if(activePairs_.begin(), activePairs_.end(),
  //                               [vert, this](const EdgePair &pair) {
  //                                 return VertWestOfEdge(&*vert, pair.east) >
  //                                 0;
  //                               });
  //       int isStart =
  //           loc == activePairs_.end() ? 1 : VertWestOfEdge(&*vert,
  //           loc->west);
  //       int isStart2 = CCW(vert->left->pos, vert->pos, vert->right->pos);
  //       if (isStart * isStart2 < 0) {
  //         if (vert != monotones_.end() &&
  //             std::next(vert)->pos.y - vert->pos.y < kTolerance) {
  //           if (params.verbose)
  //             std::cout << "SKIP" << std::endl;
  //           return ProcessVert(std::next(vert));
  //         } else {
  //           if (params.verbose)
  //             std::cout << "SELF-INTERSECTION" << std::endl;
  //         }
  //       }
  //       if (isStart >= 0) {
  //         if (params.verbose)
  //           std::cout << "START" << std::endl;
  //         bool westCertain = loc == activePairs_.begin() ||
  //                            VertWestOfEdge(&*vert, std::prev(loc)->east) <
  //                            0;
  //         bool eastCertain = isStart > 0;
  //         vert->leftPair = activePairs_.insert(loc, {{&*vert, vert->left},
  //                                                    {&*vert, vert->right},
  //                                                    monotones_.end(),
  //                                                    westCertain,
  //                                                    eastCertain});
  //         vert->rightPair = vert->leftPair;
  //       } else {
  //         if (params.verbose)
  //           std::cout << "HOLE" << std::endl;
  //         if (loc == activePairs_.end())
  //           --loc;
  //         // hole-starting vert links earlier activePair
  //         vert->rightPair = activePairs_.insert(
  //             loc, {loc->west, {&*vert, vert->right}, loc->vMerge, true,
  //             true});
  //         loc->west.vSouth->leftPair = vert->rightPair;
  //         loc->vMerge = monotones_.end();
  //         VertItr newVert = SplitVerts(vert, vert->rightPair);
  //         if (newVert != monotones_.end())
  //           vert = newVert;
  //         vert->leftPair = loc;
  //         loc->west = {&*vert, vert->left};
  //       }
  //     }
  //   }
  //   if (params.intermediateChecks)
  //     ListPairs();
  //   vert->SetProcessed(true);
  //   return vert;
  // }

  void ListPairs() {
    if (params.verbose)
      std::cout << "active edges:" << std::endl;
    for (EdgePair &pair : activePairs_) {
      ListPair(pair);
    }
  }

  void ListPair(EdgePair &pair) {
    ListEdge(pair.west);
    ALWAYS_ASSERT(&*(pair.west.vSouth->leftPair) == &pair, logicErr,
                  "west vSouth does not point back!");
    ALWAYS_ASSERT(pair.west.vSouth->left == pair.west.vNorth, logicErr,
                  "west edge does not go left!");
    ListEdge(pair.east);
    ALWAYS_ASSERT(&*(pair.east.vSouth->rightPair) == &pair, logicErr,
                  "east vSouth does not point back!");
    ALWAYS_ASSERT(pair.east.vSouth->right == pair.east.vNorth, logicErr,
                  "east edge does not go right!");
    if (params.verbose && pair.vMerge != monotones_.end())
      std::cout << "pair vMerge: " << pair.vMerge->mesh_idx << std::endl;
  }

  void ListEdge(ActiveEdge edge) {
    if (params.verbose)
      std::cout << "edge: S = " << edge.vSouth->mesh_idx
                << ", N = " << edge.vNorth->mesh_idx << std::endl;
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
} // namespace

namespace manifold {
// This is nearly the only function to do a floating point comparison in this
// whole triangulator (the other is the check for sweep-line degeneracies).
// This
// is done to maintain maximum consistency.
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
    if (params.checkGeometry)
      CheckGeometry(triangles, polys);
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

std::vector<Halfedge>
Triangles2Edges(const std::vector<glm::ivec3> &triangles) {
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
      continue; // only interested in polygon edges
    auto vert = vert2edges.emplace(halfedge.startVert,
                                   glm::ivec2(halfedge.face, Edge::kInvalid));
    if (!vert.second)
      (vert.first->second)[1] = halfedge.face;

    vert = vert2edges.emplace(halfedge.endVert,
                              glm::ivec2(halfedge.face, Edge::kInvalid));
    if (!vert.second)
      (vert.first->second)[1] = halfedge.face;
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

} // namespace manifold