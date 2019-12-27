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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <stack>

#include "polygon.h"

namespace {
using namespace manifold;

ExecutionParams params;

constexpr float kTolerance = 1e-5;
constexpr float kTolerance2 = kTolerance * kTolerance;

struct VertAdj;
typedef std::list<VertAdj>::iterator VertItr;

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

struct VertAdj {
  glm::vec2 pos;
  int mesh_idx;   // This is a global index into the manifold.
  int edgeRight;  // Cannot join identical edges with a triangle.
  VertAdj *left, *right;
  int degenerateRadius;
  int index;
  std::list<EdgePair>::iterator activePair;

  bool Processed() const { return index < 0; }
  void setProcessed(bool processed) { index = processed ? -1 : 0; }
};

int Next(int i, int n) { return ++i >= n ? 0 : i; }
int Prev(int i, int n) { return --i < 0 ? n - 1 : i; }

bool SharedEdge(glm::ivec2 edges0, glm::ivec2 edges1) {
  return (edges0[0] != Edge::kNoIdx &&
          (edges0[0] == edges1[0] || edges0[0] == edges1[1])) ||
         (edges0[1] != Edge::kNoIdx &&
          (edges0[1] == edges1[0] || edges0[1] == edges1[1]));
}

class Triangulator {
 public:
  Triangulator(const VertAdj *vert) {
    reflex_chain_.push(vert);
    other_side_ = vert;
  }
  int NumTriangles() { return triangles_output_; }

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
      if (params.verbose) std::cout << "different chain" << std::endl;
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
  const VertAdj *other_side_;
  int triangles_output_ = 0;
  bool onRight_;

  void AddTriangle(std::vector<glm::ivec3> &triangles, int v0, int v1, int v2) {
    // if (v0 == v1 || v1 == v2 || v2 == v0) return;
    if (onRight_)
      triangles.emplace_back(v0, v1, v2);
    else
      triangles.emplace_back(v0, v2, v1);
    ++triangles_output_;
    if (params.verbose) std::cout << triangles.back() << std::endl;
  }

  bool SharesEdge(const VertAdj *v0, const VertAdj *v1) {
    glm::ivec2 e0(v0->edgeRight, v0->left->edgeRight);
    glm::ivec2 e1(v1->edgeRight, v1->left->edgeRight);
    return SharedEdge(e0, e1);
  }
};

class Monotones {
 public:
  enum VertType { START, HOLE, LEFTWARDS, RIGHTWARDS, MERGE, END };

  Monotones(const Polygons &polys) {
    VertAdj *start, *last, *current;
    for (const SimplePolygon &poly : polys) {
      for (int i = 0; i < poly.size(); ++i) {
        monotones_.push_back({poly[i].pos,       //
                              poly[i].idx,       //
                              poly[i].nextEdge,  //
                              nullptr,           //
                              nullptr,           //
                              -1, false, activePairs_.begin()});
        current = &monotones_.back();
        if (i == 0)
          start = current;
        else
          Link(last, current);
        last = current;
      }
      Link(current, start);
    }
    monotones_.sort(
        [](const VertAdj &a, const VertAdj &b) { return a.pos.y < b.pos.y; });
    // Collapse degenerate sweep-line stops
    float yLast = -std::numeric_limits<float>::infinity();
    float yFirst = yLast;
    for (auto &vert : monotones_) {
      if (vert.pos.y - yLast < kTolerance) {
        yLast = vert.pos.y;
        vert.pos.y = yFirst;
      } else
        yFirst = yLast = vert.pos.y;
    }
    // Sort degenerates by degenerate radius
    for (auto &start : monotones_) {
      VertAdj *right = start.right;
      int radiusR = 0;
      while (right->pos.y == start.pos.y && right->mesh_idx != start.mesh_idx) {
        ++radiusR;
        right = right->right;
      }
      VertAdj *left = start.left;
      int radiusL = 0;
      while (left->pos.y == start.pos.y && left->mesh_idx != start.mesh_idx) {
        ++radiusL;
        left = left->left;
      }
      start.degenerateRadius =
          left->pos.y < start.pos.y
              ? (right->pos.y < start.pos.y ? std::min(radiusR, radiusL)
                                            : radiusL)
              : (right->pos.y < start.pos.y ? radiusR
                                            : std::numeric_limits<int>::max() -
                                                  std::min(radiusR, radiusL));
    }
    monotones_.sort([](const VertAdj &a, const VertAdj &b) {
      return a.pos.y != b.pos.y ? a.pos.y < b.pos.y
                                : a.degenerateRadius < b.degenerateRadius;
    });
    // Sweep forward
    VertType v_type = START;
    for (VertItr vItr = monotones_.begin(); vItr != monotones_.end(); ++vItr) {
      v_type = ProcessVert(vItr);
      if (params.verbose) std::cout << v_type << std::endl;
    }
    Check(v_type);
    // Sweep backward
    for (auto &vert : monotones_) {
      vert.pos *= -1;
      vert.setProcessed(false);
    }
    monotones_.reverse();
    for (VertItr vItr = monotones_.begin(); vItr != monotones_.end(); ++vItr) {
      v_type = ProcessVert(vItr);
      if (params.verbose) std::cout << v_type << std::endl;
    }
    Check(v_type);
  }

  void Triangulate(std::vector<glm::ivec3> &triangles) {
    int i = 1;
    for (auto &vert : monotones_) {
      vert.index = i++;
    }
    int triangles_left = monotones_.size();
    VertAdj *start = &monotones_.front();
    while (1) {
      if (params.verbose) std::cout << start->mesh_idx << std::endl;
      Triangulator triangulator(start);
      start->setProcessed(true);
      VertAdj *vR = start->right;
      VertAdj *vL = start->left;
      while (vR != vL) {
        if (vR->index < vL->index) {
          if (params.verbose) std::cout << vR->mesh_idx << std::endl;
          triangulator.ProcessVert(vR, true, false, triangles);
          vR->setProcessed(true);
          vR = vR->right;
        } else {
          if (params.verbose) std::cout << vL->mesh_idx << std::endl;
          triangulator.ProcessVert(vL, false, false, triangles);
          vL->setProcessed(true);
          vL = vL->left;
        }
      }
      if (params.verbose) std::cout << vR->mesh_idx << std::endl;
      triangulator.ProcessVert(vR, true, true, triangles);
      vR->setProcessed(true);
      // validation
      ALWAYS_ASSERT(triangulator.NumTriangles() > 0, logicErr,
                    "Monotone produced no triangles.");
      triangles_left -= 2 + triangulator.NumTriangles();
      // Find next monotone
      VertItr itr =
          std::find_if(monotones_.begin(), monotones_.end(),
                       [](const VertAdj &v) { return !v.Processed(); });
      if (itr == monotones_.end()) break;
      start = &*itr;
    }
    ALWAYS_ASSERT(triangles_left == 0, logicErr,
                  "Triangulation produced wrong number of triangles.");
  }

  void Check(VertType v_type) {
    ALWAYS_ASSERT(activePairs_.empty(), logicErr,
                  "There are still active edges.");
    ALWAYS_ASSERT(v_type == END, logicErr,
                  "Monotones did not finish with an END.");
    if (!params.intermediateChecks) return;
    std::vector<EdgeVerts> edges;
    for (auto &vert : monotones_) {
      vert.setProcessed(false);
      edges.push_back({vert.mesh_idx, vert.right->mesh_idx, Edge::kNoIdx});
      ALWAYS_ASSERT(vert.right->right != &vert, logicErr, "two-edge monotone!");
      ALWAYS_ASSERT(vert.left->right == &vert, logicErr,
                    "monotone vert neighbors don't agree!");
    }
    if (params.verbose) {
      VertAdj *start = &monotones_.front();
      while (1) {
        start->setProcessed(true);
        std::cout << "monotone start: " << start->mesh_idx << ", "
                  << start->pos.y << std::endl;
        VertAdj *v = start->right;
        while (v != start) {
          std::cout << v->mesh_idx << ", " << v->pos.y << std::endl;
          v->setProcessed(true);
          v = v->right;
        }
        std::cout << std::endl;
        VertItr itr =
            std::find_if(monotones_.begin(), monotones_.end(),
                         [](const VertAdj &v) { return !v.Processed(); });
        if (itr == monotones_.end()) break;
        start = &*itr;
      }
    }
  }

 private:
  std::list<VertAdj> monotones_;
  std::list<EdgePair> activePairs_;

  void Link(VertAdj *left, VertAdj *right) {
    left->right = right;
    right->left = left;
  }

  VertItr SplitVerts(VertItr north, EdgePair &pair, bool NEisNofNW = false) {
    VertItr south = pair.vMerge;
    if (south == monotones_.end()) return monotones_.end();

    // at split events, add duplicate vertices to end of list and reconnect
    if (params.verbose)
      std::cout << "split from " << north->mesh_idx << " to " << south->mesh_idx
                << std::endl;

    VertItr insertAt = NEisNofNW ? std::next(north) : north;
    VertItr northEast = monotones_.insert(insertAt, *north);
    Link(north->left, &*northEast);

    VertItr southEast = monotones_.insert(south, *south);
    Link(&*southEast, south->right);

    Link(&*south, &*north);
    Link(&*northEast, &*southEast);

    pair.vMerge = monotones_.end();
    return northEast;
  }

  auto LeftPair(VertAdj *vert) {
    auto pair = vert->left->activePair;
    if (pair->east.vNorth == vert) return pair;
    return std::find_if(
        activePairs_.begin(), activePairs_.end(),
        [vert](const EdgePair &pair) { return pair.east.vNorth == vert; });
  }

  auto RightPair(VertAdj *vert) {
    auto pair = vert->right->activePair;
    if (pair->west.vNorth == vert) return pair;
    return std::find_if(
        activePairs_.begin(), activePairs_.end(),
        [vert](const EdgePair &pair) { return pair.west.vNorth == vert; });
  }

  int VertWestOfEdge(const VertAdj *vert, const ActiveEdge &edge) {
    glm::vec2 last = edge.vSouth->pos;
    glm::vec2 next = edge.vNorth->pos;
    int side = CCW(last, next, vert->pos);
    if (side == 0) side = CCW(vert->pos, next, vert->right->pos);
    return side;
  }

  void Reorder(const VertAdj *vert, const bool west) {
    auto activePair = vert->activePair;
    if (!activePair->getCertainty(west)) {
      int sign = west ? -1 : 1;
      auto end = west ? activePairs_.begin() : std::prev(activePairs_.end());
      while (activePair != end) {
        activePair = west ? std::prev(activePair) : std::next(activePair);
        int eastOf = sign * VertWestOfEdge(&*vert, activePair->getEdge(!west));
        if (eastOf >= 0) {   // in the right place
          if (eastOf > 0) {  // certain
            vert->activePair->getCertainty(west) = true;
          }
          auto loc = west ? std::next(activePair) : activePair;
          activePairs_.splice(loc, activePairs_, vert->activePair);
          break;
        }
        eastOf = sign * VertWestOfEdge(&*vert, activePair->getEdge(west));
        if (eastOf >= 0) {  // in the right place
          auto loc = west ? activePair : std::next(activePair);
          activePairs_.splice(loc, activePairs_, vert->activePair);
          vert->activePair->getCertainty(!west) = false;
          if (eastOf > 0) {  // certainly a hole
            std::swap(activePair->west, vert->activePair->west);
            activePair->west.vSouth->activePair = activePair;
            vert->activePair->west.vSouth->activePair = vert->activePair;
            vert->activePair->east.vSouth->activePair = vert->activePair;
            vert->activePair->eastCertain = true;
            vert->activePair->westCertain = true;
          }
          break;
        }
      }
    }
  }

  VertType ProcessVert(VertItr vert) {
    // Skip newly duplicated verts
    if (vert->Processed()) return END;
    vert->setProcessed(true);
    if (params.verbose)
      std::cout << "mesh_idx = " << vert->mesh_idx << std::endl;
    VertType vertType;
    if (vert->right->Processed()) {
      if (vert->left->Processed()) {
        vertType = END;
        auto leftPair = LeftPair(&*vert);
        auto rightPair = RightPair(&*vert);
        if (leftPair == rightPair) {  // facing in
          SplitVerts(vert, *rightPair);
        } else {  // facing out
          vertType = MERGE;
          VertItr newVert = SplitVerts(vert, *rightPair, true);
          if (newVert != monotones_.end()) vert = newVert;
          SplitVerts(vert, *leftPair, false);
          leftPair->east = rightPair->east;
          leftPair->east.vSouth->activePair = leftPair;
          if (std::next(leftPair) == rightPair) {
            leftPair->vMerge = vert;
            vertType = MERGE;
          }
        }
        activePairs_.erase(rightPair);
      } else {
        vertType = LEFTWARDS;
        // update edge
        vert->activePair = RightPair(&*vert);
        ActiveEdge &activeEdge = vert->activePair->west;
        activeEdge.vSouth = &*vert;
        activeEdge.vNorth = vert->left;
        Reorder(&*vert, true);
        Reorder(&*vert, false);
        SplitVerts(vert, *vert->activePair, true);
      }
    } else {
      if (vert->left->Processed()) {
        vertType = RIGHTWARDS;
        // update edge
        vert->activePair = LeftPair(&*vert);
        ActiveEdge &activeEdge = vert->activePair->east;
        activeEdge.vSouth = &*vert;
        activeEdge.vNorth = vert->right;
        Reorder(&*vert, true);
        Reorder(&*vert, false);
        SplitVerts(vert, *vert->activePair, false);
      } else {
        auto loc = std::find_if(activePairs_.begin(), activePairs_.end(),
                                [vert, this](const EdgePair &pair) {
                                  return VertWestOfEdge(&*vert, pair.east) > 0;
                                });
        int isStart =
            loc == activePairs_.end() ? 1 : VertWestOfEdge(&*vert, loc->west);
        if (isStart >= 0) {
          vertType = START;
          bool westCertain = loc == activePairs_.begin() ||
                             VertWestOfEdge(&*vert, std::prev(loc)->east) < 0;
          bool eastCertain = isStart > 0;
          vert->activePair = activePairs_.insert(loc, {{&*vert, vert->left},
                                                       {&*vert, vert->right},
                                                       monotones_.end(),
                                                       westCertain,
                                                       eastCertain});
        } else {
          vertType = HOLE;
          if (loc == activePairs_.end()) --loc;
          // hole-starting vert links earlier activePair
          vert->activePair = activePairs_.insert(
              loc, {loc->west, {&*vert, vert->right}, loc->vMerge, true, true});
          loc->west.vSouth->activePair = vert->activePair;
          loc->west = {&*vert, vert->left};
          loc->vMerge = monotones_.end();
          VertItr newVert = SplitVerts(vert, *vert->activePair);
          if (newVert != monotones_.end()) newVert->activePair = loc;
        }
      }
    }
    if (params.verbose) ListPairs();
    return vertType;
  }

  void ListPairs() {
    std::cout << "active edges:" << std::endl;
    for (EdgePair pair : activePairs_) {
      ListPair(pair);
    }
  }

  void ListPair(EdgePair pair) {
    ListEdge(pair.west);
    ListEdge(pair.east);
    if (pair.vMerge != monotones_.end())
      std::cout << "pair vMerge: " << pair.vMerge->mesh_idx << std::endl;
  }

  void ListEdge(ActiveEdge edge) {
    std::cout << "edge: S = " << edge.vSouth->mesh_idx
              << ", N = " << edge.vNorth->mesh_idx << std::endl;
  }
};  // namespace
}  // namespace

namespace manifold {

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

Polygons Assemble(const std::vector<EdgeVerts> &halfedges) {
  Polygons polys;
  std::map<int, int> vert_edge;
  for (int i = 0; i < halfedges.size(); ++i) {
    ALWAYS_ASSERT(
        vert_edge.emplace(std::make_pair(halfedges[i].first, i)).second,
        runtimeErr, "polygon has duplicate vertices.");
  }
  auto startEdge = halfedges.begin();
  auto thisEdge = halfedges.begin();
  for (;;) {
    if (thisEdge == startEdge) {
      if (vert_edge.empty()) break;
      startEdge = std::next(halfedges.begin(), vert_edge.begin()->second);
      thisEdge = startEdge;
      polys.push_back({});
    }
    polys.back().push_back(
        {glm::vec2(1.0f / 0.0f), thisEdge->first, thisEdge->edge});
    auto result = vert_edge.find(thisEdge->second);
    ALWAYS_ASSERT(result != vert_edge.end(), runtimeErr, "nonmanifold edge");
    thisEdge = std::next(halfedges.begin(), result->second);
    vert_edge.erase(result);
  }
  return polys;
}

std::vector<glm::ivec3> Triangulate(const Polygons &polys) {
  std::vector<glm::ivec3> triangles;
  try {
    Monotones monotones(polys);
    monotones.Triangulate(triangles);
    CheckTopology(triangles, polys);
    if (params.checkGeometry) CheckGeometry(triangles, polys);
  } catch (const std::exception &e) {
    if (params.checkGeometry && !params.suppressErrors) {
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
    throw;
  }
  return triangles;
}

std::vector<EdgeVerts> Polygons2Edges(const Polygons &polys) {
  std::vector<EdgeVerts> halfedges;
  for (const auto &poly : polys) {
    for (int i = 1; i < poly.size(); ++i) {
      halfedges.push_back({poly[i - 1].idx, poly[i].idx, poly[i - 1].nextEdge});
    }
    halfedges.push_back({poly.back().idx, poly[0].idx, poly.back().nextEdge});
  }
  return halfedges;
}

std::vector<EdgeVerts> Triangles2Edges(
    const std::vector<glm::ivec3> &triangles) {
  std::vector<EdgeVerts> halfedges;
  for (const glm::ivec3 &tri : triangles) {
    // Differentiate edges of triangles by setting index to Edge::kInterior.
    halfedges.push_back({tri[0], tri[1], Edge::kInterior});
    halfedges.push_back({tri[1], tri[2], Edge::kInterior});
    halfedges.push_back({tri[2], tri[0], Edge::kInterior});
  }
  return halfedges;
}

void CheckTopology(const std::vector<EdgeVerts> &halfedges) {
  ALWAYS_ASSERT(halfedges.size() % 2 == 0, runtimeErr,
                "Odd number of halfedges.");
  size_t n_edges = halfedges.size() / 2;
  std::vector<EdgeVerts> forward(halfedges.size()), backward(halfedges.size());

  auto end = std::copy_if(halfedges.begin(), halfedges.end(), forward.begin(),
                          [](EdgeVerts e) { return e.second > e.first; });
  ALWAYS_ASSERT(std::distance(forward.begin(), end) == n_edges, runtimeErr,
                "Half of halfedges should be forward.");
  forward.resize(n_edges);

  end = std::copy_if(halfedges.begin(), halfedges.end(), backward.begin(),
                     [](EdgeVerts e) { return e.second < e.first; });
  ALWAYS_ASSERT(std::distance(backward.begin(), end) == n_edges, runtimeErr,
                "Half of halfedges should be backward.");
  backward.resize(n_edges);

  std::for_each(backward.begin(), backward.end(),
                [](EdgeVerts &e) { std::swap(e.first, e.second); });
  auto cmp = [](const EdgeVerts &a, const EdgeVerts &b) {
    return a.first < b.first || (a.first == b.first && a.second < b.second);
  };
  std::sort(forward.begin(), forward.end(), cmp);
  std::sort(backward.begin(), backward.end(), cmp);
  for (int i = 0; i < n_edges; ++i) {
    ALWAYS_ASSERT(forward[i].first == backward[i].first &&
                      forward[i].second == backward[i].second,
                  runtimeErr, "Forward and backward edge do not match.");
    if (i > 0) {
      ALWAYS_ASSERT(forward[i - 1].first != forward[i].first ||
                        forward[i - 1].second != forward[i].second,
                    runtimeErr, "Not a 2-manifold.");
      ALWAYS_ASSERT(backward[i - 1].first != backward[i].first ||
                        backward[i - 1].second != backward[i].second,
                    runtimeErr, "Not a 2-manifold.");
    }
  }
  // Check that no interior edges link vertices that share the same edge data.
  std::map<int, glm::ivec2> vert2edges;
  for (EdgeVerts halfedge : halfedges) {
    if (halfedge.edge == Edge::kInterior)
      continue;  // only interested in polygon edges
    auto vert = vert2edges.emplace(halfedge.first,
                                   glm::ivec2(halfedge.edge, Edge::kInvalid));
    if (!vert.second) (vert.first->second)[1] = halfedge.edge;

    vert = vert2edges.emplace(halfedge.second,
                              glm::ivec2(halfedge.edge, Edge::kInvalid));
    if (!vert.second) (vert.first->second)[1] = halfedge.edge;
  }
  for (int i = 0; i < n_edges; ++i) {
    if (forward[i].edge == Edge::kInterior &&
        backward[i].edge == Edge::kInterior) {
      glm::ivec2 TwoEdges0 = vert2edges.find(forward[i].first)->second;
      glm::ivec2 TwoEdges1 = vert2edges.find(forward[i].second)->second;
      ALWAYS_ASSERT(!SharedEdge(TwoEdges0, TwoEdges1), runtimeErr,
                    "Added an interface edge!");
    }
  }
}

void CheckTopology(const std::vector<glm::ivec3> &triangles,
                   const Polygons &polys) {
  std::vector<EdgeVerts> halfedges = Triangles2Edges(triangles);
  std::vector<EdgeVerts> openEdges = Polygons2Edges(polys);
  for (EdgeVerts e : openEdges) {
    halfedges.push_back({e.second, e.first, e.edge});
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