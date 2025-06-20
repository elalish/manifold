#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../src/collider.h"
#include "../src/impl.h"
#include "manifold/linalg.h"
#include "manifold/manifold.h"

using namespace manifold;

vec3 toVec3(vec2 in) { return vec3(in.x, in.y, 0); }

void Print(const manifold::Polygons& poly) {
  std::cout << "Poly = [" << std::endl;

  auto& loop = poly[0];
  for (size_t i = 0; i != loop.size(); i++) {
    std::cout << "[" << loop[i].x << ", " << loop[i].y << "]";

    if (i != loop.size() - 1) std::cout << " ,";
  }

  std::cout << std::endl << "]" << std::endl;
}

struct PolygonTest {
  PolygonTest(const manifold::Polygons& polygons)
      : polygons(polygons), name("Result"){};
  std::string name;
  int expectedNumTri = -1;
  double epsilon = -1;

  manifold::Polygons polygons;
};

void Save(const std::string& filename, const std::vector<PolygonTest>& result) {
  // Open a file stream for writing.
  std::ofstream outFile(filename);

  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }

  // Write each test case to the file.
  for (const auto& test : result) {
    // Write the header for the test.
    outFile << test.name << " " << test.expectedNumTri << " " << test.epsilon
            << " " << test.polygons.size() << "\n";

    // Write each polygon within the test.
    for (const auto& poly : test.polygons) {
      // Write the number of points for the current polygon.
      outFile << poly.size() << "\n";
      // Write the coordinates for each point in the polygon.
      for (const auto& point : poly) {
        outFile << point.x << " " << point.y << "\n";
      }
    }
  }

  outFile.close();
  std::cout << "Successfully saved " << result.size() << " tests to "
            << filename << std::endl;
}

manifold::Polygons VertexByVertex(const double radius,
                                  const manifold::Polygons& poly) {
  auto generateArc = [](const vec2& preP, const vec2& curP, const vec2& nextP,
                        double radius) -> std::vector<vec2> {
    vec2 norm1 = la::normalize(preP - curP),
         norm2 = la::normalize(nextP - curP);
    double theta = std::acos(la::dot(norm1, norm2));

    double convexity = la::cross((curP - preP), (nextP - curP));

    double dist = radius / std::tan(theta / 2.0);

    vec2 t1 = curP + norm1 * dist, t2 = curP + norm2 * dist;

    vec2 circleCenter = t1 + vec2(-norm1.y, norm1.x) * radius;

    double start = std::atan2(t1.y - circleCenter.y, t1.x - circleCenter.x),
           end = std::atan2(t2.y - circleCenter.y, t2.x - circleCenter.x);

    // Discrete, Merge

    double sweep_angle_total_rad = end - start;

    const double EPSILON = 1E-7;

    if (convexity > 0 && sweep_angle_total_rad < EPSILON) {
      sweep_angle_total_rad += 2 * M_PI;
    } else if (convexity < 0 && sweep_angle_total_rad > EPSILON) {
      sweep_angle_total_rad -= 2 * M_PI;
    }

    int num_arc_segments = 100;

    std::vector<vec2> arcPoints;
    arcPoints.push_back(t1);

    for (int i = 1; i < num_arc_segments; ++i) {
      double fraction = static_cast<double>(i) / num_arc_segments;
      double current_angle = start + sweep_angle_total_rad * fraction;
      double px = circleCenter.x + radius * std::cos(current_angle);
      double py = circleCenter.y + radius * std::sin(current_angle);
      arcPoints.push_back({px, py});
    }

    arcPoints.push_back(t2);

    return arcPoints;
  };

  manifold::Polygons newPoly{{}};

  for (size_t i = 0; i != poly[0].size(); i++) {
    auto& loop = poly[0];
    auto& newLoop = newPoly[0];

    const vec2 preP = loop[(i - 1 + loop.size()) % loop.size()], curP = loop[i],
               nextP = loop[(i + 1) % loop.size()];

    auto r = generateArc(preP, curP, nextP, radius);

    if (!r.empty()) {
      newLoop.insert(newLoop.end(), r.begin(), r.end());
    } else {
      newLoop.push_back(curP);
    }
  }

  return newPoly;
}

manifold::Polygons RollingBall(const double radius,
                               const manifold::Polygons& poly) {
  auto& loop = poly[0];

  manifold::Polygons newPoly = poly;
  auto& newLoop = newPoly[0];

  Vec<manifold::Box> boxVec;
  Vec<uint32_t> mortonVec;

  struct pair {
    manifold::Box box;
    uint32_t morton;
    size_t p1Ref;
    size_t p2Ref;
  };

  std::vector<pair> pairs;
  {
    for (size_t i = 0; i != loop.size(); i++) {
      const vec2 p1 = loop[i], p2 = loop[(i + 1) % loop.size()];

      vec3 center = toVec3(p1) + toVec3(p2);
      center /= 2;

      std::cout << center.x << " " << center.y << std::endl;

      manifold::Box bbox(toVec3(p1), toVec3(p2));

      pairs.push_back({bbox, manifold::Collider::MortonCode(center, bbox), i,
                       (i + 1) % loop.size()});
    }

    std::stable_sort(pairs.begin(), pairs.end(),
                     [](const pair& lhs, const pair& rhs) -> bool {
                       return rhs.morton > lhs.morton;
                     });

    for (auto it = pairs.begin(); it != pairs.end(); it++) {
      boxVec.push_back(it->box);
      mortonVec.push_back(it->morton);
    }

    boxVec.Dump();
    mortonVec.Dump();
  }

  const double EPSILON = 1E-7;

  manifold::Collider collider(boxVec, mortonVec);

  for (size_t i = 0; i != loop.size(); i++) {
    const vec2 p1 = loop[i], p2 = loop[(i + 1) % loop.size()];
    vec2 e = p2 - p1;
    vec2 perp = la::normalize(vec2(-e.y, e.x));

    // auto box = boxVec[i];

    vec2 extendP1 = p1 - e * radius, extendP2 = p2 + e * radius;

    vec2 offsetP1 = extendP1 + perp * 2.0 * radius,
         offsetP2 = extendP2 + perp * 2.0 * radius;

    manifold::Box box(toVec3(extendP1), toVec3(extendP2));
    box.Union(toVec3(offsetP1));
    box.Union(toVec3(offsetP2));

    auto r = collider.Collisions(manifold::Vec<manifold::Box>({box}).cview());
    r.Dump();

    r.Sort();

    // In Out Classify
    // AABB is too wide for collision test, this part can accelerate

    // Line intersect
    auto intersectLine = [&EPSILON](const vec2& p1, const vec2& p2,
                                    const vec2& p3, const vec2& p4,
                                    vec2& intersectionPoint) -> bool {
      double det = la::cross(p2 - p1, p4 - p3);

      if (std::abs(det) < EPSILON) {
        // Parallel
        return false;
      }

      double num_t = la::cross(p3 - p1, p4 - p3);
      double num_u = la::cross(p3 - p1, p2 - p1);

      double t = num_t / det;
      double u = num_u / det;

      // Check if the intersection point inside line segement
      if ((t >= 0.0 - EPSILON && t <= 1.0 + EPSILON) &&
          (u >= 0.0 - EPSILON && u <= 1.0 + EPSILON)) {
        // Inside
        double intersect_x = p1.x + t * (p2.x - p1.x);
        double intersect_y = p1.y + t * (p2.y - p1.y);

        intersectionPoint = {intersect_x, intersect_y};
        return true;
      } else {
        // Outside -> No intersection

        intersectionPoint = {};
        return false;
      }
    };

    // Cirle intersection
    auto intersectCircleDetermine = [&radius, &EPSILON](
                                        const vec2& p1, const vec2& p2,
                                        const vec2& center) -> bool {
      vec2 d = p2 - p1;

      if (la::length(d) < EPSILON)
        return la::dot(p1 - center, p1 - center) <= radius * radius;

      // Project vec p1 -> circle to line segement
      double t = la::dot(center - p1, d) / la::dot(d, d);

      vec2 closestPoint;

      if (t < 0) {
        closestPoint = p1;
      } else if (t > 1) {
        closestPoint = p2;
      } else {
        closestPoint = p1 + t * d;
      }

      // Calculate the distance from the closest point to the circle's center
      double distanceSquared =
          la::dot(closestPoint - center, closestPoint - center);

      return distanceSquared <= radius * radius;
    };

    // Result
    for (size_t j = 0; j != r.size(); j++) {
      auto ele = pairs[r.Get(j, true)];

      // if (ele.p1Ref == i || ele.p1Ref == (i + 1) % loop.size() ||
      //     ele.p2Ref == i || ele.p2Ref == (i + 1) % loop.size())
      //   continue;

      vec2 t;
      if (intersectLine(p1, p2, loop[ele.p1Ref], loop[ele.p2Ref], t) ||
          intersectCircleDetermine(loop[ele.p1Ref], loop[ele.p2Ref],
                                   extendP1 + perp * radius) ||
          intersectCircleDetermine(loop[ele.p1Ref], loop[ele.p2Ref],
                                   extendP2 + perp * radius)) {
        // Intersect
        std::cout << "Intersect\n";

        {
          vec2 norm1 = la::normalize(p1 - p2),
               norm2 = la::normalize(loop[ele.p2Ref] - loop[ele.p1Ref]);
          double theta = std::acos(la::dot(norm1, norm2));

          double convexity =
              la::cross((p2 - p1), (loop[ele.p2Ref] - loop[ele.p1Ref]));

          double dist = radius / std::tan(theta / 2.0);

          vec2 t1 = p2 + norm1 * dist, t2 = loop[ele.p2Ref] + norm2 * dist;

          vec2 circleCenter = t1 + vec2(-norm1.y, norm1.x) * radius;

          const uint32_t seg = 20;
          for (size_t k = 0; k != seg; k++) {
            newLoop.push_back(circleCenter +
                              vec2{radius * cos(M_PI * 2 / seg * k),
                                   radius * sin(M_PI * 2 / seg * k)});
          }
        }
      }
    }
  }

  return newPoly;
}

int main() {
  if (false) {
    auto obj = manifold::Manifold::Cube();

    auto mesh = obj.GetMeshGL();
    Manifold::Fillet(mesh, 5, {});
  }

  manifold::Polygons Rect{{vec2{0, 0}, vec2{0, 5}, vec2{5, 5}, vec2{5, 0}}};
  manifold::Polygons Tri{{vec2{0, 0}, vec2{0, 5}, vec2{5, 0}}};
  manifold::Polygons UShape{
      {vec2{0, 0}, vec2{-1, 5}, vec2{3, 2}, vec2{7, 5}, vec2{6, 0}}};

  const manifold::Polygons poly = UShape;

  std::vector<PolygonTest> result{PolygonTest(VertexByVertex(0.5, poly)),
                                  PolygonTest(RollingBall(0.5, poly))};

  Save("../project/result.txt", result);

  return 0;
}