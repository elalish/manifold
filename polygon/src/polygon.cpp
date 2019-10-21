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

DebugControls debug;

constexpr float kTolerance = 1e-5;

struct ActiveEdge {
  int vLeft, vRight;
  int vMerge = -1;
  bool surePos, sureNeg;
};

struct VertAdj {
  glm::vec2 pos;
  int mesh_idx;     // This is a global index into the manifold.
  int edgeRight;    // Cannot join identical edges with a triangle.
  int left, right;  // These are local indices within this vector.
  int sweepOrder;
  bool processed, needsSplit;
  std::list<ActiveEdge>::iterator activeEdge;
};

int Next(int i, int n) { return ++i >= n ? 0 : i; }
int Prev(int i, int n) { return --i < 0 ? n - 1 : i; }

class Monotones {
 public:
  const std::vector<VertAdj> &GetMonotones() { return monotones_; }

  enum VertType { START, LEFTWARDS, RIGHTWARDS, MERGE, END };

  Monotones(const Polygons &polys) {
    std::vector<std::tuple<float, int, int>> sweepForward;
    for (const SimplePolygon &poly : polys) {
      int start = Num_verts();
      for (int i = 0; i < poly.size(); ++i) {
        monotones_.push_back({poly[i].pos,                   //
                              poly[i].idx,                   //
                              poly[i].nextEdge,              //
                              Prev(i, poly.size()) + start,  //
                              Next(i, poly.size()) + start,  //
                              -1, false, false, activeEdges_.begin()});
        sweepForward.push_back(
            std::make_tuple(monotones_.back().pos.y, 0, start + i));
      }
    }
    std::sort(sweepForward.begin(), sweepForward.end());
    // Collapse degenerate sweep-line stops
    float yLast = -std::numeric_limits<float>::infinity();
    float yFirst = yLast;
    for (auto &sweepPoint : sweepForward) {
      float &y = std::get<0>(sweepPoint);
      int idx = std::get<2>(sweepPoint);
      if (y - yLast < kTolerance) {
        yLast = y;
        y = yFirst;
        Vert(idx).pos.y = y;
      } else
        yFirst = yLast = y;
    }
    // Sort degenerates by degenerate radius
    for (auto &sweepPoint : sweepForward) {
      VertAdj start = Vert(std::get<2>(sweepPoint));
      VertAdj right = Right(start);
      int radiusR = 0;
      while (right.pos.y == start.pos.y && right.mesh_idx != start.mesh_idx) {
        ++radiusR;
        right = Right(right);
      }
      VertAdj left = Left(start);
      int radiusL = 0;
      while (left.pos.y == start.pos.y && left.mesh_idx != start.mesh_idx) {
        ++radiusL;
        left = Left(left);
      }
      std::get<1>(sweepPoint) =
          left.pos.y < start.pos.y
              ? (right.pos.y < start.pos.y ? std::min(radiusR, radiusL)
                                           : radiusL)
              : (right.pos.y < start.pos.y ? radiusR
                                           : std::numeric_limits<int>::max() -
                                                 std::min(radiusR, radiusL));
    }
    std::sort(sweepForward.begin(), sweepForward.end());
    // Sweep forward
    VertType v_type = START;
    for (int i = 0; i < sweepForward.size(); ++i) {
      int idx = std::get<2>(sweepForward[i]);
      Vert(idx).sweepOrder = i;
      v_type = ProcessVert(idx);
      if (debug.verbose) std::cout << v_type << std::endl;
    }
    ALWAYS_ASSERT(v_type == END, logicErr,
                  "Monotones did not finish with an END.");
    Check();
    // Sweep backward
    std::vector<std::tuple<int, int>> sweepBack;
    for (int i = 0; i < monotones_.size(); ++i) {
      VertAdj &v = Vert(i);
      v.pos *= -1;
      v.processed = false;
      sweepBack.push_back(std::make_tuple(-v.sweepOrder, i));
    }
    std::sort(sweepBack.begin(), sweepBack.end());
    ALWAYS_ASSERT(activeEdges_.empty(), logicErr,
                  "There are still active edges.");
    for (auto sweepPoint : sweepBack) {
      int idx = std::get<1>(sweepPoint);
      v_type = ProcessVert(idx);
      if (debug.verbose) std::cout << v_type << std::endl;
    }
    ALWAYS_ASSERT(activeEdges_.empty(), logicErr,
                  "There are still active edges.");
    ALWAYS_ASSERT(v_type == END, logicErr,
                  "Monotones did not finish with an END.");
    Check();
  }

  void Check() {
    std::vector<EdgeVerts> edges;
    for (int i = 0; i < monotones_.size(); ++i) {
      edges.push_back({i, monotones_[i].right, Edge::kNoIdx});
      ALWAYS_ASSERT(monotones_[monotones_[i].right].right != i, logicErr,
                    "two-edge monotone!");
      ALWAYS_ASSERT(monotones_[monotones_[i].right].left == i, logicErr,
                    "monotone vert neighbors don't agree!");
    }
    Polygons polys = Assemble(edges);
    if (debug.verbose) {
      for (auto &poly : polys)
        for (auto &v : poly) {
          auto vert = monotones_[v.idx];
          v.idx = vert.mesh_idx;
          v.pos = vert.pos;
        }
      Dump(polys);
    }
  }

 private:
  std::vector<VertAdj> monotones_;
  std::list<ActiveEdge> activeEdges_;

  VertAdj &Vert(int idx) { return monotones_[idx]; }
  VertAdj &Right(const VertAdj &v) { return monotones_[v.right]; }
  VertAdj &Left(const VertAdj &v) { return monotones_[v.left]; }
  int Num_verts() const { return monotones_.size(); }

  void Link(int left_idx, int right_idx) {
    Vert(left_idx).right = right_idx;
    Vert(right_idx).left = left_idx;
  }

  int SplitVerts(int v_idx, int merge_idx) {
    // at split events, add duplicate vertices to end of list and reconnect
    if (debug.verbose)
      std::cout << "split from " << v_idx << " to " << merge_idx << std::endl;

    Vert(merge_idx).needsSplit = false;

    int vLeft_idx = Num_verts();
    monotones_.push_back(Vert(v_idx));
    Link(Vert(v_idx).left, vLeft_idx);

    int mergeRight_idx = Num_verts();
    monotones_.push_back(Vert(merge_idx));
    Link(mergeRight_idx, Vert(merge_idx).right);

    Link(merge_idx, v_idx);
    Link(vLeft_idx, mergeRight_idx);
    return vLeft_idx;
  }

  auto LeftActiveEdge(int idx) {
    auto activeEdge = Left(Vert(idx)).activeEdge;
    if (activeEdge->vRight != idx) {
      activeEdge =
          std::find_if(activeEdges_.begin(), activeEdges_.end(),
                       [idx](ActiveEdge &edge) { return edge.vRight == idx; });
    }
    return activeEdge;
  }

  auto RightActiveEdge(int idx) {
    auto activeEdge = Right(Vert(idx)).activeEdge;
    if (activeEdge->vLeft != idx) {
      activeEdge =
          std::find_if(activeEdges_.begin(), activeEdges_.end(),
                       [idx](ActiveEdge &edge) { return edge.vLeft == idx; });
    }
    return activeEdge;
  }

  float DistanceX(VertAdj vert, ActiveEdge edge) {
    float a = (Vert(edge.vRight).pos.y - vert.pos.y) /
              (Vert(edge.vRight).pos.y - Vert(edge.vLeft).pos.y);
    float x;
    if (std::isnan(a)) {
      x = std::min(Vert(edge.vRight).pos.x, Vert(edge.vLeft).pos.x);
    } else {
      // a = std::max(std::min(a, 1.0f), 0.0f);
      x = (1.0f - a) * Vert(edge.vRight).pos.x + a * Vert(edge.vLeft).pos.x;
    }
    return vert.pos.x - x;
  }

  VertType ProcessVert(int idx) {
    auto &vert = Vert(idx);
    vert.processed = true;
    if (debug.verbose)
      std::cout << "idx = " << idx << ", mesh_idx = " << vert.mesh_idx
                << std::endl;
    VertType vertType;
    if (Right(vert).processed) {
      if (Left(vert).processed) {
        auto leftEdge = LeftActiveEdge(idx);
        auto rightEdge = RightActiveEdge(idx);
        if (std::next(rightEdge) == leftEdge) {  // facing in
          vertType = END;
          int mergeIdx = rightEdge->vMerge;
          if (mergeIdx >= 0 && Vert(mergeIdx).needsSplit)
            SplitVerts(idx, mergeIdx);
        } else {  // facing out
          vertType = MERGE;
          int mergeIdx = rightEdge->vMerge;
          if (mergeIdx >= 0 && Vert(mergeIdx).needsSplit)
            idx = SplitVerts(idx, mergeIdx);
          mergeIdx = leftEdge->vMerge;
          if (mergeIdx >= 0 && Vert(mergeIdx).needsSplit)
            SplitVerts(idx, mergeIdx);
          Vert(idx).needsSplit = true;
          std::prev(leftEdge)->vMerge = idx;
          std::next(rightEdge)->vMerge = idx;
        }
        activeEdges_.erase(leftEdge);
        activeEdges_.erase(rightEdge);
        if (debug.verbose) ListEdges();
        // early return because no active edge for further processing
        return vertType;
      } else {
        vertType = LEFTWARDS;
        // update edge
        vert.activeEdge = RightActiveEdge(idx);
        ActiveEdge &activeEdge = *vert.activeEdge;
        activeEdge.vRight = idx;
        activeEdge.vLeft = vert.left;
      }
    } else {
      if (Left(vert).processed) {
        vertType = RIGHTWARDS;
        // update edge
        vert.activeEdge = LeftActiveEdge(idx);
        ActiveEdge &activeEdge = *vert.activeEdge;
        activeEdge.vLeft = idx;
        activeEdge.vRight = vert.right;
      } else {
        vertType = START;
        // add edges, sorted appropriately
        auto loc = std::find_if(activeEdges_.begin(), activeEdges_.end(),
                                [vert, this](ActiveEdge &edge) {
                                  return DistanceX(vert, edge) <= 0;
                                });
        // TODO: record certainty
        int vMerge = loc == activeEdges_.end() ? -1 : loc->vMerge;
        activeEdges_.insert(loc, {vert.left, idx, vMerge, false, false});
        vert.activeEdge = std::prev(loc);  // left edge is active
        auto pos =
            DistanceX(Right(vert), *std::prev(loc)) > 0 ? loc : std::prev(loc);
        activeEdges_.insert(pos, {idx, vert.right, vMerge, false, false});
      }
    }
    if (debug.verbose) ListEdges();
    int &mergeIdx = vert.activeEdge->vMerge;
    if (mergeIdx >= 0) {
      if (Vert(mergeIdx).needsSplit)
        SplitVerts(idx, mergeIdx);
      else
        mergeIdx = -1;
    }
    // if edge order was uncertain, re-sort
    return vertType;
  }

  void ListEdges() {
    for (ActiveEdge edge : activeEdges_) {
      ListEdge(edge);
    }
  }
  void ListEdge(ActiveEdge edge) {
    std::cout << "edge: L = " << edge.vLeft << ", R = " << edge.vRight
              << ", M = " << edge.vMerge << std::endl;
  }
};

bool SharedEdge(glm::ivec2 edges0, glm::ivec2 edges1) {
  return (edges0[0] != Edge::kNoIdx &&
          (edges0[0] == edges1[0] || edges0[0] == edges1[1])) ||
         (edges0[1] != Edge::kNoIdx &&
          (edges0[1] == edges1[0] || edges0[1] == edges1[1]));
}

class Triangulator {
 public:
  Triangulator(const std::vector<VertAdj> &monotones, int v_idx)
      : monotones_(monotones) {
    reflex_chain_.push(v_idx);
    other_side_ = v_idx;
  }
  int NumTriangles() { return triangles_output; }

  bool ProcessVert(int vi_idx, std::vector<glm::ivec3> &triangles) {
    int attached = Attached(vi_idx);
    if (attached == 0)
      return 0;
    else {
      const VertAdj vi = monotones_[vi_idx];
      int v_top_idx = reflex_chain_.top();
      VertAdj v_top = monotones_[v_top_idx];
      if (reflex_chain_.size() < 2) {
        reflex_chain_.push(vi_idx);
        onRight_ = vi.left == v_top_idx;
        return 1;
      }
      reflex_chain_.pop();
      int vj_idx = reflex_chain_.top();
      VertAdj vj = monotones_[vj_idx];
      if (attached == 1) {
        if (debug.verbose) std::cout << "same chain" << std::endl;
        while (CCW(vi.pos, vj.pos, v_top.pos) == (onRight_ ? 1 : -1) ||
               (CCW(vi.pos, vj.pos, v_top.pos) == 0 &&
                !SharesEdge(vi, vj, v_top))) {
          AddTriangle(triangles, vi.mesh_idx, vj.mesh_idx, v_top.mesh_idx);
          v_top_idx = vj_idx;
          reflex_chain_.pop();
          if (reflex_chain_.size() == 0) break;
          v_top = vj;
          vj_idx = reflex_chain_.top();
          vj = monotones_[vj_idx];
        }
        reflex_chain_.push(v_top_idx);
        reflex_chain_.push(vi_idx);
      } else {
        if (debug.verbose) std::cout << "different chain" << std::endl;
        onRight_ = !onRight_;
        VertAdj v_last = v_top;
        while (!reflex_chain_.empty()) {
          vj = monotones_[reflex_chain_.top()];
          AddTriangle(triangles, vi.mesh_idx, v_last.mesh_idx, vj.mesh_idx);
          v_last = vj;
          reflex_chain_.pop();
        }
        reflex_chain_.push(v_top_idx);
        reflex_chain_.push(vi_idx);
        other_side_ = v_top_idx;
      }
      return 1;
    }
  }

 private:
  const std::vector<VertAdj> &monotones_;
  std::stack<int> reflex_chain_;
  int other_side_;
  int triangles_output = 0;
  bool onRight_;

  const VertAdj GetTop() { return monotones_[reflex_chain_.top()]; }
  const VertAdj GetOther() { return monotones_[other_side_]; }

  int Attached(int v_idx) {
    if (onRight_) {
      if (GetOther().left == v_idx)
        return -1;
      else if (GetTop().right == v_idx)
        return 1;
      else
        return 0;
    } else {
      if (GetOther().right == v_idx)
        return -1;
      else if (GetTop().left == v_idx)
        return 1;
      else
        return 0;
    }
  }

  void AddTriangle(std::vector<glm::ivec3> &triangles, int v0, int v1, int v2) {
    if (onRight_)
      triangles.emplace_back(v0, v1, v2);
    else
      triangles.emplace_back(v0, v2, v1);
    ++triangles_output;
  }

  bool SharesEdge(const VertAdj &v0, const VertAdj &v1, const VertAdj &v2) {
    glm::ivec2 e0(v0.edgeRight, monotones_[v0.left].edgeRight);
    glm::ivec2 e1(v1.edgeRight, monotones_[v1.left].edgeRight);
    glm::ivec2 e2(v2.edgeRight, monotones_[v2.left].edgeRight);
    return SharedEdge(e0, e1) || SharedEdge(e1, e2) || SharedEdge(e2, e0);
  }
};

void TriangulateMonotones(const std::vector<VertAdj> &monotones,
                          std::vector<glm::ivec3> &triangles) {
  // make sorted index list to traverse the sweep line.
  std::vector<std::tuple<int, int>> sweep_line;
  for (int i = 0; i < monotones.size(); ++i) {
    // Ensure sweep line is sorted identically here and in Monotones
    // above, including when the y-values are identical.
    sweep_line.push_back(std::make_tuple(monotones[i].sweepOrder, i));
  }
  std::sort(sweep_line.begin(), sweep_line.end());
  std::vector<Triangulator> triangulators;
  for (int i = 0; i < sweep_line.size(); ++i) {
    const int v_idx = std::get<1>(sweep_line[i]);
    if (debug.verbose)
      std::cout << "i = " << i << ", v_idx = " << v_idx
                << ", mesh_idx = " << monotones[v_idx].mesh_idx << std::endl;
    bool found = false;
    for (int j = 0; j < triangulators.size(); ++j) {
      if (triangulators[j].ProcessVert(v_idx, triangles)) {
        found = true;
        if (debug.verbose)
          std::cout << "in triangulator " << j << ", with "
                    << triangulators[j].NumTriangles() << " triangles so far"
                    << std::endl;
        break;
      }
    }
    if (!found) triangulators.emplace_back(monotones, v_idx);
  }
  // quick validation
  int triangles_left = monotones.size();
  for (auto &triangulator : triangulators) {
    triangles_left -= 2;
    ALWAYS_ASSERT(triangulator.NumTriangles() > 0, logicErr,
                  "Monotone produced no triangles.");
    triangles_left -= triangulator.NumTriangles();
  }
  ALWAYS_ASSERT(triangles_left == 0, logicErr,
                "Triangulation produced wrong number of triangles.");
}

void PrintTriangulationWarning(const std::string &triangulationType,
                               const Polygons &polys,
                               const std::vector<glm::ivec3> &triangles,
                               const std::exception &e) {
  if (debug.geometricWarnings) {
    std::cout << "-----------------------------------" << std::endl;
    std::cout << triangulationType
              << " triangulation failed, switching to backup! Warnings so far: "
              << ++debug.numWarnings << std::endl;
    std::cout << e.what() << std::endl;
    Dump(polys);
    std::cout << "produced this triangulation:" << std::endl;
    for (int j = 0; j < triangles.size(); ++j) {
      std::cout << triangles[j][0] << ", " << triangles[j][1] << ", "
                << triangles[j][2] << std::endl;
    }
  }
}
}  // namespace

namespace manifold {

int CCW(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2) {
  glm::vec2 v1 = p1 - p0;
  glm::vec2 v2 = p2 - p0;
  float result = v1.x * v2.y - v1.y * v2.x;
  p0 = glm::abs(p0);
  p1 = glm::abs(p1);
  p2 = glm::abs(p2);
  float norm = p0.x * p0.y + p1.x * p1.y + p2.x * p2.y;
  if (std::abs(result) <= norm * kTolerance)
    return 0;
  else
    return result > 0 ? 1 : -1;
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
    triangles = PrimaryTriangulate(polys);
    CheckTopology(triangles, polys);
    if (debug.geometricWarnings && !CheckGeometry(triangles, polys)) {
      std::cout << "-----------------------------------" << std::endl;
      std::cout << "Warning: triangulation is folded! Warnings so far: "
                << ++debug.numWarnings << std::endl;
      Dump(polys);
      std::cout << "produced this triangulation:" << std::endl;
      for (int j = 0; j < triangles.size(); ++j) {
        std::cout << triangles[j][0] << ", " << triangles[j][1] << ", "
                  << triangles[j][2] << std::endl;
      }
    };
  } catch (const std::exception &e) {
    // The primary triangulator has guaranteed manifold and non-overlapping
    // output for non-overlapping input. For overlapping input it occasionally
    // has trouble, and if so we switch to a simpler, toplogical backup
    // triangulator that has guaranteed manifold output, except in the presence
    // of certain edge constraints.
    PrintTriangulationWarning("Primary", polys, triangles, e);
    throw;
    // try {
    //   triangles = BackupTriangulate(polys);
    //   CheckTopology(triangles, polys);
    // } catch (const std::exception &e2) {
    //   PrintTriangulationWarning("Backup", polys, triangles, e2);
    //   throw;
    // }
  }
  return triangles;
}

std::vector<glm::ivec3> PrimaryTriangulate(const Polygons &polys) {
  std::vector<glm::ivec3> triangles;
  Monotones monotones(polys);
  TriangulateMonotones(monotones.GetMonotones(), triangles);
  return triangles;
}

std::vector<glm::ivec3> BackupTriangulate(const Polygons &polys) {
  std::vector<glm::ivec3> triangles;
  for (const auto &poly : polys) {
    if (poly.size() < 3) continue;
    int start = 1;
    int end = poly.size() - 1;
    glm::ivec3 tri{poly[end].idx, poly[0].idx, poly[start].idx};
    glm::ivec2 startEdges{poly[start - 1].nextEdge, poly[start].nextEdge};
    glm::ivec2 endEdges{poly[end - 1].nextEdge, poly[end].nextEdge};
    bool forward = false;
    for (;;) {
      if (start == end) break;
      if (SharedEdge(startEdges, endEdges)) {
        // Attempt to avoid shared edges by switching to the other side.
        if (forward) {
          start = Prev(start, poly.size());
          end = Prev(end, poly.size());
          tri = {poly[end].idx, tri[0], tri[1]};
        } else {
          start = Next(start, poly.size());
          end = Next(end, poly.size());
          tri = {tri[1], tri[2], poly[start].idx};
        }
        startEdges = {poly[start - 1].nextEdge, poly[start].nextEdge};
        endEdges = {poly[end - 1].nextEdge, poly[end].nextEdge};
        forward = !forward;
      }
      triangles.push_back(tri);
      // By default, alternate to avoid making a high degree vertex.
      forward = !forward;
      if (forward) {
        start = Next(start, poly.size());
        startEdges = {poly[Prev(start, poly.size())].nextEdge,
                      poly[start].nextEdge};
        tri = {tri[0], tri[2], poly[start].idx};
      } else {
        end = Prev(end, poly.size());
        endEdges = {poly[Prev(end, poly.size())].nextEdge, poly[end].nextEdge};
        tri = {poly[end].idx, tri[0], tri[2]};
      }
    }
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

bool CheckGeometry(const std::vector<glm::ivec3> &triangles,
                   const Polygons &polys) {
  std::map<int, glm::vec2> vertPos;
  for (const auto &poly : polys) {
    for (int i = 0; i < poly.size(); ++i) {
      vertPos[poly[i].idx] = poly[i].pos;
    }
  }
  return std::all_of(
      triangles.begin(), triangles.end(), [&vertPos](const glm::ivec3 &tri) {
        return CCW(vertPos[tri[0]], vertPos[tri[1]], vertPos[tri[2]]) >= 0;
      });
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

void SetPolygonWarnings(bool val) { debug.geometricWarnings = val; };
void SetPolygonVerbose(bool val) { debug.verbose = val; };

}  // namespace manifold