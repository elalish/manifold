#include <algorithm>

#include "par.h"
#include "polygon.h"
#include "test.h"
#include "utils.h"

namespace manifold {

inline float cross(glm::vec2 p, glm::vec2 q) { return p.x * q.y - p.y * q.x; }

// return true if p intersects with q
// note that we don't care about collinear, head-to-tail etc.
bool intersect(glm::vec2 p0, glm::vec2 p1, glm::vec2 q0, glm::vec2 q1) {
  glm::vec2 r = p1 - p0;
  glm::vec2 s = q1 - q0;
  float rxs = cross(r, s);
  if (rxs < 1e-6 && rxs > -1e-6) {
    // float qpxr = cross(q0 - p0, r);
    // return qpxr < 1e-6 && qpxr > -1e-6;
    //
    // ignore if they are not crossing
    return false;
  }
  float u = cross(p0 - q0, r) / rxs;
  float t = cross(q0 - p0, s) / rxs;
  // we ignore head-to-tail case
  return (1e-6 <= u && u <= (1 - 1e-6) && 1e-6 <= t && t <= (1 - 1e-6));
}

bool safeToRemove(Polygons &polys, int i, int j) {
  if (polys[i].size() == 3) return false;
  int prev = j == 0 ? polys[i].size() - 1 : (j - 1);
  int next = j == (polys[i].size() - 1) ? 0 : (j + 1);
  for (int k = 0; k < polys.size(); k++) {
    if (!std::all_of(countAt(0lu), countAt(polys[k].size()), [&](int l) {
          int ll = l == (polys[k].size() - 1) ? 0 : (l + 1);
          if (i == k && (l == j || ll == j)) return true;
          bool intersected = intersect(polys[i][prev], polys[i][next],
                                       polys[k][l], polys[k][ll]);
          return !intersected;
        }))
      return false;
  }
  return true;
}

std::pair<int, int> findIndex(const Polygons &polys, int i) {
  int outer = 0;
  while (i > polys[outer].size()) i -= polys[outer++].size();
  return std::make_pair(outer, i);
}

void Dump(const Polygons &polys) {
  for (auto poly : polys) {
    std::cout << "polys.push_back({" << std::setprecision(9) << std::endl;
    for (auto v : poly) {
      std::cout << "    {" << v.x << ", " << v.y << "},  //" << std::endl;
    }
    std::cout << "});" << std::endl;
  }
  for (auto poly : polys) {
    std::cout << "show(array([" << std::endl;
    for (auto v : poly) {
      std::cout << "  [" << v.x << ", " << v.y << "]," << std::endl;
    }
    std::cout << "]))" << std::endl;
  }
}

void DumpTriangulation(const Polygons &polys) {
  bool oldProcessOverlaps = manifold::PolygonParams().processOverlaps;
  manifold::PolygonParams().processOverlaps = true;
  auto result = Triangulate(polys);
  manifold::PolygonParams().processOverlaps = oldProcessOverlaps;
  for (auto &tri : result) {
    auto x = findIndex(polys, tri.x);
    auto y = findIndex(polys, tri.y);
    auto z = findIndex(polys, tri.z);
    printf("show(array([[%.7f, %.7f], [%.7f, %.7f], [%.7f, %.7f]]))\n",
           polys[x.first][x.second].x, polys[x.first][x.second].y,
           polys[y.first][y.second].x, polys[y.first][y.second].y,
           polys[z.first][z.second].x, polys[z.first][z.second].y);
  }
}

// we are assuming polys is valid
// we try to remove vertices from polys such that
// 1. the updated polys is still valid (no overlapping edges)
// 2. error in triangulation (either geometryErr or overlapping triangles)
void simplify(Polygons &polys, float precision = -1) {
  PolygonParams().intermediateChecks = true;
  PolygonParams().processOverlaps = false;
  PolygonParams().suppressErrors = true;

  bool removedSomething = true;
  while (removedSomething) {
    removedSomething = false;
    for (int i = 0; i < polys.size(); i++) {
      for (int j = 0; j < polys[i].size(); j++) {
        if (safeToRemove(polys, i, j)) {
          glm::vec2 removed = polys[i][j];
          polys[i].erase(polys[i].begin() + j);
          try {
            auto result = Triangulate(polys, precision);
            std::vector<std::pair<glm::vec2, glm::vec2>> triangleEdges(
                result.size() * 3);
            for (auto &tri : result) {
              auto x = findIndex(polys, tri.x);
              auto y = findIndex(polys, tri.y);
              auto z = findIndex(polys, tri.z);
              triangleEdges.push_back(std::make_pair(polys[x.first][x.second],
                                                     polys[y.first][y.second]));
              triangleEdges.push_back(std::make_pair(polys[y.first][y.second],
                                                     polys[z.first][z.second]));
              triangleEdges.push_back(std::make_pair(polys[z.first][z.second],
                                                     polys[x.first][x.second]));
            }
            // if triangles are non-overlapping, it is fine
            if (std::all_of(
                    triangleEdges.begin(), triangleEdges.end(), [&](auto &p) {
                      return all_of(ExecutionPolicy::Par, triangleEdges.begin(),
                                    triangleEdges.end(), [&](auto &q) {
                                      return !intersect(p.first, p.second,
                                                        q.first, q.second);
                                    });
                    })) {
              polys[i].insert(polys[i].begin() + j, removed);
            } else {
              removedSomething = true;
            }
          } catch (geometryErr &e) {
            removedSomething = true;
          }
        }
      }
    }
  }
}

}  // namespace manifold
