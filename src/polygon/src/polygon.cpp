// Copyright 2021 The Manifold Authors.
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
#if MANIFOLD_PAR == 'T'
#include "tbb/tbb.h"
#endif

#include <algorithm>
#include <numeric>
#if MANIFOLD_PAR == 'T' && TBB_INTERFACE_VERSION >= 10000 && \
    __has_include(<pstl/glue_execution_defs.h>)
#include <execution>
#endif
#include <list>
#include <map>
#if __has_include(<memory_resource>)
#include <memory_resource>
#endif
#include <queue>
#include <set>
#include <stack>

#include "optional_assert.h"

namespace {
using namespace manifold;

static ExecutionParams params;

constexpr float kBest = -std::numeric_limits<float>::infinity();

#ifdef MANIFOLD_DEBUG
struct PolyEdge {
  int startVert, endVert;
};

std::vector<PolyEdge> Polygons2Edges(const PolygonsIdx &polys) {
  std::vector<PolyEdge> halfedges;
  for (const auto &poly : polys) {
    for (int i = 1; i < poly.size(); ++i) {
      halfedges.push_back({poly[i - 1].idx, poly[i].idx});
    }
    halfedges.push_back({poly.back().idx, poly[0].idx});
  }
  return halfedges;
}

std::vector<PolyEdge> Triangles2Edges(
    const std::vector<glm::ivec3> &triangles) {
  std::vector<PolyEdge> halfedges;
  halfedges.reserve(triangles.size() * 3);
  for (const glm::ivec3 &tri : triangles) {
    halfedges.push_back({tri[0], tri[1]});
    halfedges.push_back({tri[1], tri[2]});
    halfedges.push_back({tri[2], tri[0]});
  }
  return halfedges;
}

void CheckTopology(const std::vector<PolyEdge> &halfedges) {
  ASSERT(halfedges.size() % 2 == 0, topologyErr, "Odd number of halfedges.");
  size_t n_edges = halfedges.size() / 2;
  std::vector<PolyEdge> forward(halfedges.size()), backward(halfedges.size());

  auto end = std::copy_if(halfedges.begin(), halfedges.end(), forward.begin(),
                          [](PolyEdge e) { return e.endVert > e.startVert; });
  ASSERT(std::distance(forward.begin(), end) == n_edges, topologyErr,
         "Half of halfedges should be forward.");
  forward.resize(n_edges);

  end = std::copy_if(halfedges.begin(), halfedges.end(), backward.begin(),
                     [](PolyEdge e) { return e.endVert < e.startVert; });
  ASSERT(std::distance(backward.begin(), end) == n_edges, topologyErr,
         "Half of halfedges should be backward.");
  backward.resize(n_edges);

  std::for_each(backward.begin(), backward.end(),
                [](PolyEdge &e) { std::swap(e.startVert, e.endVert); });
  auto cmp = [](const PolyEdge &a, const PolyEdge &b) {
    return a.startVert < b.startVert ||
           (a.startVert == b.startVert && a.endVert < b.endVert);
  };
  std::stable_sort(forward.begin(), forward.end(), cmp);
  std::stable_sort(backward.begin(), backward.end(), cmp);
  for (int i = 0; i < n_edges; ++i) {
    ASSERT(forward[i].startVert == backward[i].startVert &&
               forward[i].endVert == backward[i].endVert,
           topologyErr, "Forward and backward edge do not match.");
    if (i > 0) {
      ASSERT(forward[i - 1].startVert != forward[i].startVert ||
                 forward[i - 1].endVert != forward[i].endVert,
             topologyErr, "Not a 2-manifold.");
      ASSERT(backward[i - 1].startVert != backward[i].startVert ||
                 backward[i - 1].endVert != backward[i].endVert,
             topologyErr, "Not a 2-manifold.");
    }
  }
}

void CheckTopology(const std::vector<glm::ivec3> &triangles,
                   const PolygonsIdx &polys) {
  std::vector<PolyEdge> halfedges = Triangles2Edges(triangles);
  std::vector<PolyEdge> openEdges = Polygons2Edges(polys);
  for (PolyEdge e : openEdges) {
    halfedges.push_back({e.endVert, e.startVert});
  }
  CheckTopology(halfedges);
}

void CheckGeometry(const std::vector<glm::ivec3> &triangles,
                   const PolygonsIdx &polys, float precision) {
  std::unordered_map<int, glm::vec2> vertPos;
  for (const auto &poly : polys) {
    for (int i = 0; i < poly.size(); ++i) {
      vertPos[poly[i].idx] = poly[i].pos;
    }
  }
  ASSERT(std::all_of(triangles.begin(), triangles.end(),
                     [&vertPos, precision](const glm::ivec3 &tri) {
                       return CCW(vertPos[tri[0]], vertPos[tri[1]],
                                  vertPos[tri[2]], precision) >= 0;
                     }),
         geometryErr, "triangulation is not entirely CCW!");
}

void Dump(const PolygonsIdx &polys) {
  for (auto poly : polys) {
    std::cout << "polys.push_back({" << std::setprecision(9) << std::endl;
    for (auto v : poly) {
      std::cout << "    {" << v.pos.x << ", " << v.pos.y << "},  //"
                << std::endl;
    }
    std::cout << "});" << std::endl;
  }
  for (auto poly : polys) {
    std::cout << "show(array([" << std::endl;
    for (auto v : poly) {
      std::cout << "  [" << v.pos.x << ", " << v.pos.y << "]," << std::endl;
    }
    std::cout << "]))" << std::endl;
  }
}

void PrintFailure(const std::exception &e, const PolygonsIdx &polys,
                  std::vector<glm::ivec3> &triangles, float precision) {
  std::cout << "-----------------------------------" << std::endl;
  std::cout << "Triangulation failed! Precision = " << precision << std::endl;
  std::cout << e.what() << std::endl;
  Dump(polys);
  std::cout << "produced this triangulation:" << std::endl;
  for (int j = 0; j < triangles.size(); ++j) {
    std::cout << triangles[j][0] << ", " << triangles[j][1] << ", "
              << triangles[j][2] << std::endl;
  }
}

#define PRINT(msg) \
  if (params.verbose) std::cout << msg << std::endl;
#else
#define PRINT(msg)
#endif

/**
 * Ear-clipping triangulator based on David Eberly's approach from Geometric
 * Tools, but adjusted to handle epsilon-valid polygons, and including a
 * fallback that ensures a manifold triangulation even for overlapping polygons.
 * This is an O(n^2) algorithm, but hopefully this is not a big problem as the
 * number of edges in a given polygon is generally much less than the number of
 * triangles in a mesh, and relatively few faces even need triangulation.
 *
 * The main adjustments for robustness involve clipping the sharpest ears first
 * (a known technique to get higher triangle quality), and doing an exhaustive
 * search to determine ear convexity exactly if the first geometric result is
 * within precision.
 */

class EarClip {
 public:
  EarClip(const PolygonsIdx &polys, float precision) : precision_(precision) {
    int numVert = 0;
    for (const SimplePolygonIdx &poly : polys) {
      numVert += poly.size();
    }
    polygon_.reserve(numVert + 2 * polys.size());

    Initialize(polys);
  }

  std::vector<glm::ivec3> Triangulate() {
    for (VertItr v = polygon_.begin(); v != polygon_.end(); ++v) {
      ClipIfDegenerate(v);
    }

    FindStarts();

    CutKeyholes();

    for (const VertItr start : starts_) {
      TriangulatePoly(start);
    }

    return triangles_;
  }

  float GetPrecision() const { return precision_; }

 private:
  struct Vert;
  typedef std::vector<Vert>::iterator VertItr;
  struct MaxX {
    bool operator()(const VertItr &a, const VertItr &b) const {
      return a->pos.x > b->pos.x;
    }
  };
  struct MinCost {
    bool operator()(const VertItr &a, const VertItr &b) const {
      return a->cost < b->cost;
    }
  };
  typedef std::set<VertItr, MinCost>::iterator qItr;

  std::vector<Vert> polygon_;
  std::multiset<VertItr, MaxX> starts_;
  std::map<VertItr, Rect> start2BBox_;
  std::multiset<VertItr, MinCost> earsQueue_;
  std::vector<glm::ivec3> triangles_;
  float precision_;

  struct Vert {
    int mesh_idx;
    qItr ear;
    glm::vec2 pos, rightDir;
    VertItr left, right;
    float cost;

    bool IsShort(float precision) const {
      const glm::vec2 edge = right->pos - pos;
      return glm::dot(edge, edge) < precision * precision;
    }

    bool IsConvex(float precision) const {
      int convexity = CCW(left->pos, pos, right->pos, precision);
      if (convexity != 0) {
        return convexity > 0;
      }

      // Uncertain - walk the polygon to get certainty.
      VertItr nextL = left;
      VertItr nextR = right;
      VertItr center = left->right;

      while (nextL != nextR) {
        glm::vec2 vecL = center->pos - nextL->pos;
        glm::vec2 vecR = center->pos - nextR->pos;
        float L2 = glm::dot(vecL, vecL);
        float R2 = glm::dot(vecR, vecR);

        if (L2 > R2) {
          center = nextR;
          nextR = nextR->right;
        } else {
          center = nextL;
          nextL = nextL->left;
        }

        convexity = CCW(nextL->pos, center->pos, nextR->pos, precision);
        if (convexity != 0) {
          return convexity > 0;
        }
      }
      // The whole polygon is degenerate - consider this to be convex.
      return true;
    }

    std::pair<VertItr, float> InterpY2X(float y, int onTop,
                                        float precision) const {
      const float p2 = precision * precision;
      if (pos.y < right->pos.y) {  // Edge goes up
        if (glm::abs(pos.y - y) <= precision) {
          if (glm::abs(right->pos.y - y) > precision) {
            // Tail is at y
            VertItr prev = left;
            if (!(prev->pos.y > y + precision && IsConvex(precision)) &&
                !(onTop == 1 && prev->pos.y > y - precision)) {
              return std::make_pair(left->right, pos.x);
            }
          }  // Edges within the precision band are skipped
        } else {
          if (glm::abs(right->pos.y - y) <= precision) {
            // Head is at y
            VertItr next = right->right;
            if (!(next->pos.y < y - precision && right->IsConvex(precision)) &&
                !(onTop == -1 && next->pos.y <= y + precision)) {
              return std::make_pair(right, right->pos.x);
            }
          } else if (pos.y < y && right->pos.y > y) {
            // Edge crosses y
            float a =
                glm::clamp((y - pos.y) / (right->pos.y - pos.y), 0.0f, 1.0f);
            const float x = glm::mix(pos.x, right->pos.x, a);
            const VertItr p = pos.x < right->pos.x ? right : left->right;
            return std::make_pair(p, x);
          }
        }
      }
      // Edge does not cross y going up
      return std::make_pair(left, std::numeric_limits<float>::infinity());
    }

    float SignedDist(VertItr v, glm::vec2 unit, float precision) const {
      float d = glm::determinant(glm::mat2(unit, v->pos - pos));
      if (glm::abs(d) < precision) {
        d = glm::determinant(glm::mat2(unit, v->right->pos - pos));
        return d < precision ? kBest : 0;
      }
      return d;
    }

    float Cost(VertItr v, glm::vec2 openSide, float precision) const {
      const glm::vec2 offset = v->pos - pos;
      float cost = SignedDist(v, rightDir, precision);
      if (isfinite(cost)) {
        cost = glm::min(cost, SignedDist(v, left->rightDir, precision));
      }
      if (isfinite(cost)) {
        float openCost =
            glm::determinant(glm::mat2(openSide, v->pos - right->pos));
        if (cost == 0 && glm::abs(openCost) < precision) {
          return kBest;
        }
        cost = glm::min(cost, openCost);
      }
      return cost;
    }

    float DelaunayCost(glm::vec2 diff, float scale, float precision) const {
      return -precision - scale * glm::dot(diff, diff);
    }

    float EarCost(float precision) const {
      glm::vec2 openSide = left->pos - right->pos;
      const glm::vec2 center = 0.5f * (left->pos + right->pos);
      const float scale = 4 / glm::dot(openSide, openSide);
      openSide = glm::normalize(openSide);

      float totalCost = glm::dot(left->rightDir, rightDir) - 1 - precision;
      if (CCW(pos, left->pos, right->pos, precision) == 0) {
        return totalCost < -1 ? kBest : 0;
      }
      VertItr test = right;
      while (test != left) {
        float cost = Cost(test, openSide, precision);
        if (cost < -precision) {
          cost = DelaunayCost(test->pos - center, scale, precision);
        }
        totalCost = glm::max(totalCost, cost);

        test = test->right;
      }
      return totalCost;
    }

    void PrintVert() const {
#ifdef MANIFOLD_DEBUG
      if (!params.verbose) return;
      std::cout << "vert: " << mesh_idx << ", left: " << left->mesh_idx
                << ", right: " << right->mesh_idx << ", cost: " << cost
                << std::endl;
#endif
    }
  };

  void Link(VertItr left, VertItr right) const {
    left->right = right;
    right->left = left;
    left->rightDir = glm::normalize(right->pos - left->pos);
    if (!isfinite(left->rightDir.x)) left->rightDir = {0, 0};
  }

  bool Clipped(VertItr v) { return v->right->left != v; }

  void ClipEar(VertItr ear) {
    Link(ear->left, ear->right);
    if (ear->left->mesh_idx != ear->mesh_idx &&
        ear->mesh_idx != ear->right->mesh_idx &&
        ear->right->mesh_idx != ear->left->mesh_idx) {
      // Filter out topological degenerates, which can form in bad
      // triangulations of polygons with holes, due to vert duplication.
      triangles_.push_back(
          {ear->left->mesh_idx, ear->mesh_idx, ear->right->mesh_idx});
      if (params.verbose) {
        std::cout << "output tri: " << ear->mesh_idx << ", "
                  << ear->right->mesh_idx << ", " << ear->left->mesh_idx
                  << std::endl;
      }
    } else {
      PRINT("Topological degenerate!");
    }
  }

  void ClipIfDegenerate(VertItr ear) {
    if (Clipped(ear)) {
      return;
    }
    if (ear->left == ear->right) {
      return;
    }
    if (ear->IsShort(precision_) ||
        (CCW(ear->left->pos, ear->pos, ear->right->pos, precision_) == 0 &&
         glm::dot(ear->left->pos - ear->pos, ear->right->pos - ear->pos) > 0 &&
         ear->IsConvex(precision_))) {
      ClipEar(ear);
      ClipIfDegenerate(ear->left);
      ClipIfDegenerate(ear->right);
    }
  }

  void Initialize(const PolygonsIdx &polys) {
    float bound = 0;
    for (const SimplePolygonIdx &poly : polys) {
      auto vert = poly.begin();
      polygon_.push_back({vert->idx, earsQueue_.end(), vert->pos});
      const VertItr first = std::prev(polygon_.end());
      VertItr last = first;
      starts_.insert(first);

      for (++vert; vert != poly.end(); ++vert) {
        polygon_.push_back({vert->idx, earsQueue_.end(), vert->pos});
        VertItr next = std::prev(polygon_.end());

        bound = glm::max(
            bound, glm::max(glm::abs(next->pos.x), glm::abs(next->pos.y)));

        Link(last, next);
        last = next;
      }
      Link(last, first);
    }

    if (precision_ < 0) precision_ = bound * kTolerance;

    triangles_.reserve(polygon_.size());
  }

  void FindStarts() {
    std::multiset<VertItr, MaxX> starts;
    for (auto startItr = starts_.begin(); startItr != starts_.end();
         ++startItr) {
      VertItr first = *startItr;
      VertItr start = first;
      VertItr v = first;
      float maxX = -std::numeric_limits<float>::infinity();
      Rect bBox;
      do {
        if (Clipped(v)) {
          first = v->right->left;
        } else {
          bBox.Union(v->pos);
          if (v->pos.x > maxX) {
            maxX = v->pos.x;
            start = v;
          }
        }
        v = v->right;
      } while (v != first);

      if (isfinite(maxX)) {
        starts.insert(start);
        start2BBox_.insert({start, bBox});
      }
    }
    starts_ = starts;
  }

  void CutKeyholes() {
    auto startItr = starts_.begin();
    while (startItr != starts_.end()) {
      const VertItr start = *startItr;

      if (start->IsConvex(precision_)) {  // Outer
        ++startItr;
        continue;
      }

      // Hole
      const float startX = start->pos.x;
      const Rect bBox = start2BBox_[start];
      const int onTop = start->pos.y >= bBox.max.y - precision_   ? 1
                        : start->pos.y <= bBox.min.y + precision_ ? -1
                                                                  : 0;
      float minX = std::numeric_limits<float>::infinity();
      VertItr connector = polygon_.end();
      for (auto poly = starts_.begin(); poly != starts_.end(); ++poly) {
        if (poly == startItr) continue;
        VertItr edge = *poly;
        do {
          const std::pair<VertItr, float> pair =
              edge->InterpY2X(start->pos.y, onTop, precision_);
          const float x = pair.second;
          if (isfinite(x) && x > startX - precision_ &&
              (!isfinite(minX) || (x >= startX && x < minX) ||
               (minX < startX && x > minX))) {
            minX = x;
            connector = pair.first;
          }
          edge = edge->right;
        } while (edge != *poly);
      }

      if (connector == polygon_.end()) {
        PRINT("hole did not find an outer contour!");
        ++startItr;
        continue;
      }

      connector = FindBridge(start, connector, glm::vec2(minX, start->pos.y));

      JoinPolygons(start, connector);

      startItr = starts_.erase(startItr);
    }
  }

  VertItr FindBridge(VertItr start, VertItr guess,
                     glm::vec2 intersection) const {
    const float above = guess->pos.y > start->pos.y ? 1 : -1;
    VertItr best = guess;
    VertItr vert = guess->right;
    const glm::vec2 left = start->pos - guess->pos;
    const glm::vec2 right = intersection - guess->pos;
    float minD2 = glm::dot(left, left);
    while (vert != guess) {
      const glm::vec2 offset = vert->pos - guess->pos;
      const glm::vec2 diff = vert->pos - start->pos;
      const float d2 = glm::dot(diff, diff);
      if (d2 < minD2 && vert->pos.y * above > start->pos.y * above &&
          above * glm::determinant(glm::mat2(left, offset)) > 0 &&
          above * glm::determinant(glm::mat2(offset, right)) > 0 &&
          !vert->IsConvex(precision_)) {
        minD2 = d2;
        best = vert;
      }
      vert = vert->right;
    }
    if (params.verbose) {
      std::cout << "connected " << start->mesh_idx << " to " << best->mesh_idx
                << std::endl;
    }
    return best;
  }

  void JoinPolygons(VertItr start, VertItr connector) {
    polygon_.push_back(*start);
    const VertItr newStart = std::prev(polygon_.end());
    polygon_.push_back(*connector);
    const VertItr newConnector = std::prev(polygon_.end());

    start->right->left = newStart;
    connector->left->right = newConnector;
    Link(start, connector);
    Link(newConnector, newStart);

    ClipIfDegenerate(start);
    ClipIfDegenerate(newStart);
    ClipIfDegenerate(connector);
    ClipIfDegenerate(newConnector);
  }

  void ProcessEar(VertItr v) {
    if (v->ear != earsQueue_.end()) {
      earsQueue_.erase(v->ear);
      v->ear = earsQueue_.end();
    }
    if (v->IsShort(precision_)) {
      v->cost = kBest;
      v->ear = earsQueue_.insert(v);
    } else if (v->IsConvex(precision_)) {
      v->cost = v->EarCost(precision_);
      v->ear = earsQueue_.insert(v);
    }
  }

  void TriangulatePoly(VertItr start) {
    int numTri = -2;
    earsQueue_.clear();
    VertItr v = start;
    do {
      if (v->left == v->right) {
        return;
      }
      if (Clipped(v)) {
        start = v->right->left;
      } else {
        ProcessEar(v);
        v->PrintVert();
      }
      v = v->right;
      ++numTri;
    } while (v != start);
    // Dump(v);

    while (numTri > 0) {
      const qItr ear = earsQueue_.begin();
      if (ear != earsQueue_.end()) {
        v = *ear;
        v->PrintVert();
        earsQueue_.erase(ear);
      } else {
        PRINT("No ear found!");
      }

      ClipEar(v);
      --numTri;

      ProcessEar(v->left);
      ProcessEar(v->right);
      v = v->right;
    }

    ASSERT(v->right == v->left, logicErr, "Triangulator error!");
    PRINT("Finished poly");
  }

  void Dump(VertItr start) const {
    VertItr v = start;
    std::cout << "show(array([" << std::endl;
    do {
      std::cout << "  [" << v->pos.x << ", " << v->pos.y << "],# "
                << v->mesh_idx << ", cost: " << v->cost << std::endl;
      v = v->right;
    } while (v != start);
    std::cout << "  [" << v->pos.x << ", " << v->pos.y << "],# " << v->mesh_idx
              << std::endl;
    std::cout << "]))" << std::endl;
  }
};
}  // namespace

namespace manifold {

/**
 * @brief Triangulates a set of &epsilon;-valid polygons. If the input is not
 * &epsilon;-valid, the triangulation may overlap, but will always return a
 * manifold result that matches the input edge directions.
 *
 * @param polys The set of polygons, wound CCW and representing multiple
 * polygons and/or holes. These have 2D-projected positions as well as
 * references back to the original vertices.
 * @param precision The value of &epsilon;, bounding the uncertainty of the
 * input.
 * @return std::vector<glm::ivec3> The triangles, referencing the original
 * vertex indicies.
 */
std::vector<glm::ivec3> TriangulateIdx(const PolygonsIdx &polys,
                                       float precision) {
  std::vector<glm::ivec3> triangles;
  try {
    EarClip triangulator(polys, precision);
    triangles = triangulator.Triangulate();
#ifdef MANIFOLD_DEBUG
    if (params.intermediateChecks) {
      CheckTopology(triangles, polys);
      if (!params.processOverlaps) {
        CheckGeometry(triangles, polys, 2 * triangulator.GetPrecision());
      }
    }
  } catch (const geometryErr &e) {
    if (!params.suppressErrors) {
      PrintFailure(e, polys, triangles, precision);
    }
    throw;
  } catch (const std::exception &e) {
    PrintFailure(e, polys, triangles, precision);
    throw;
#else
  } catch (const std::exception &e) {
#endif
  }
  return triangles;
}

/**
 * @brief Triangulates a set of &epsilon;-valid polygons. If the input is not
 * &epsilon;-valid, the triangulation may overlap, but will always return a
 * manifold result that matches the input edge directions.
 *
 * @param polygons The set of polygons, wound CCW and representing multiple
 * polygons and/or holes.
 * @param precision The value of &epsilon;, bounding the uncertainty of the
 * input.
 * @return std::vector<glm::ivec3> The triangles, referencing the original
 * polygon points in order.
 */
std::vector<glm::ivec3> Triangulate(const Polygons &polygons, float precision) {
  int idx = 0;
  PolygonsIdx polygonsIndexed;
  for (const auto &poly : polygons) {
    SimplePolygonIdx simpleIndexed;
    for (const glm::vec2 &polyVert : poly) {
      simpleIndexed.push_back({polyVert, idx++});
    }
    polygonsIndexed.push_back(simpleIndexed);
  }
  return TriangulateIdx(polygonsIndexed, precision);
}

ExecutionParams &PolygonParams() { return params; }

}  // namespace manifold
