#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

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
      // Concave, ignore
      return std::vector<vec2>();
    } else if (convexity < 0 && sweep_angle_total_rad > EPSILON) {
      sweep_angle_total_rad -= 2 * M_PI;
    }

    int num_arc_segments = 10;

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

  std::vector<std::optional<vec2>> adjustCircleCenter;

  manifold::Polygons newPoly = poly;
  auto& newLoop = newPoly[0];
  newLoop.push_back(loop[0]);

  Vec<manifold::Box> boxVec;
  Vec<uint32_t> mortonVec;

  struct pair {
    manifold::Box box;
    uint32_t morton;
    size_t p1Ref;
    size_t p2Ref;
  };

  std::vector<pair> pairs;

  // Build up collider
  {
    for (size_t i = 0; i != loop.size(); i++) {
      const vec2 p1 = loop[i], p2 = loop[(i + 1) % loop.size()];

      vec3 center = toVec3(p1) + toVec3(p2);
      center /= 2;

      // std::cout << center.x << " " << center.y << std::endl;

      manifold::Box bbox(toVec3(p1), toVec3(p2));

      std::cout << i << "\t" << bbox.min << " " << bbox.max << std::endl;

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

  std::vector<uint8_t> markEE(loop.size() * loop.size(), 0);
  std::vector<uint8_t> markEV(loop.size() * loop.size(), 0);

  struct Info {
    vec2 center;

    double t1, t2;
    vec2 p1E1, p2E2;
    size_t e1, e2;
    double startRad, endRad;
  };

  std::vector<std::vector<Info>> circleConnection(loop.size(),
                                                  std::vector<Info>());

  std::cout << "Collider BBox Testing" << std::endl;

  // create BBox for every line to find Collision
  for (size_t i = 0; i != loop.size(); i++) {
    // CCW, p1 p2 -> current edge start end
    const vec2 p1 = loop[i], p2 = loop[(i + 1) % loop.size()];
    vec2 e = p2 - p1;
    vec2 perp = la::normalize(vec2(e.y, -e.x));

    vec2 extendP1 = p1 - e * radius, extendP2 = p2 + e * radius;

    vec2 offsetExtendP1 = extendP1 + perp * 2.0 * radius,
         offsetExtendP2 = extendP2 + perp * 2.0 * radius;

    vec2 offsetP1 = p1 + perp * 2.0 * radius,
         offsetP2 = p2 + perp * 2.0 * radius;

    manifold::Box box(toVec3(extendP1), toVec3(extendP2));
    box.Union(toVec3(offsetExtendP1));
    box.Union(toVec3(offsetExtendP2));

    auto r = collider.Collisions(manifold::Vec<manifold::Box>({box}).cview());
    // r.Dump();

    r.Sort();

    // TODO: AABB is too wide for collision test, this part can accelerate with
    // OBB

#pragma region Utility
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

    // Handle if projection not on linesegment
    auto intersectEndpoint = [&](const vec2& line_normal, double line_c,
                                 const vec2& vertex_p1, const vec2& vertex_p2,
                                 const vec2& oldCenter, double filletRadius,
                                 vec2& newCenter) -> bool {
      // Closer endpoint
      vec2 endpoint = (la::length2(oldCenter - vertex_p1) <
                       la::length2(oldCenter - vertex_p2))
                          ? vertex_p1
                          : vertex_p2;

      double c_offset = line_c - filletRadius * la::length(line_normal);

      double signed_dist_to_offset_line =
          (la::dot(line_normal, endpoint) + c_offset) / la::length(line_normal);

      // No solution
      if (std::abs(signed_dist_to_offset_line) > filletRadius + EPSILON) {
        return false;
      }

      vec2 proj_point =
          endpoint -
          (signed_dist_to_offset_line / la::length(line_normal)) * line_normal;
      double chord_half_length = std::sqrt(std::max(
          0.0, filletRadius * filletRadius -
                   signed_dist_to_offset_line * signed_dist_to_offset_line));
      vec2 line_dir = la::normalize(vec2{-line_normal.y, line_normal.x});

      vec2 sol1 = proj_point + line_dir * chord_half_length;
      vec2 sol2 = proj_point - line_dir * chord_half_length;

      newCenter =
          (la::length2(sol1 - oldCenter) < la::length2(sol2 - oldCenter))
              ? sol1
              : sol2;
      return true;
    };

    // Projection
    auto isProjectionOnSegment = [](const vec2& c, const vec2& p1,
                                    const vec2& p2, double& t) -> bool {
      t = la::dot(c - p1, p2 - p1) / la::length2(p2 - p1);

      return t >= 0 && t <= 1;
    };

#pragma endregion

    // Intersection
    std::cout << "Now " << i << "->" << (i + 1) % loop.size() << std::endl;
    // std::cout << "BBox " << box << std::endl;
    // r.Dump();

    // In Out Classify
    for (size_t j = 0; j != r.size(); j++) {
      auto ele = pairs[r.Get(j, true)];

      // Skip neighbour because handled before
      // if (ele.p1Ref == i || ele.p1Ref == (i + 1) % loop.size() ||
      //     ele.p2Ref == i || ele.p2Ref == (i + 1) % loop.size())
      //   continue;

      // CCW, p3 p4 -> bbox hit edge start end
      vec2 p3 = loop[ele.p1Ref], p4 = loop[ele.p2Ref];
      vec2 t;

      std::cout << "Testing " << ele.p1Ref << "->" << ele.p2Ref << "\t";

      bool flag1 = intersectLine(offsetP1, offsetP2, p3, p4, t),
           flag2 = intersectCircleDetermine(p3, p4, p1 + perp * radius),
           flag3 = intersectCircleDetermine(p3, p4, p2 + perp * radius);

      std::cout << (flag1 ? "True" : "False") << " "
                << (flag2 ? "True" : "False") << " "
                << (flag3 ? "True" : "False") << " ";

      if (flag1 || flag2 || flag3) {
        // Intersect logical

        std::cout << "Intersect ";

        // Skip processed line
        markEE[i * loop.size() + ele.p1Ref] = 1;
        if (markEE[ele.p1Ref * loop.size() + i] != 0) {
          std::cout << "Skipped" << std::endl;
          continue;
        }

        {
          vec2 e1 = p2 - p1;
          vec2 e2 = p4 - p3;

          if (la::length(e1) < EPSILON || la::length(e2) < EPSILON) {
            // FIXME: Degenerate
            std::cout << "Degenerate" << std::endl;
            throw std::exception();
          }

          double angle =
              std::atan2(std::abs(la::cross(e1, e2)), la::dot(e1, e2));

          if (angle > M_PI - EPSILON) {
            std::cout << "Arc too long (>π)" << std::endl;
            throw std::exception();
          }

          vec2 normal1, normal2;

          if (la::cross(e1, e2) > 0) {
            // Concave
            normal1 = {e1.y, -e1.x};
            normal2 = {e2.y, -e2.x};
          } else {
            // Convex
            normal1 = {-e1.y, e1.x};
            normal2 = {-e2.y, e2.x};
          }

          normal1 = la::normalize(normal1);
          normal2 = la::normalize(normal2);

          double c1 = -la::dot(normal1, p1);
          double c2 = -la::dot(normal2, p3);

          mat2 A = {{normal1.x, normal2.x}, {normal1.y, normal2.y}};
          vec2 b = {radius - c1, radius - c2};

          if (std::abs(la::determinant(A)) < EPSILON) {
            std::cout << "Parallel" << std::endl;
            throw std::exception();
          }

          vec2 circleCenter = la::mul(la::inverse(A), b);

          double e1T = 0, e2T = 0;
          bool onE1 = isProjectionOnSegment(circleCenter, p1, p2, e1T),
               onE2 = isProjectionOnSegment(circleCenter, p3, p4, e2T);

          // Check is on line segment
          if (onE1 && onE2) {
          } else if (onE1) {
            // e2 endpoint
            if (!intersectEndpoint(normal1, c1, p3, p4, circleCenter, radius,
                                   circleCenter)) {
              continue;
            }
            if (e2T < 0) {
              // p3
              if (markEV[i * loop.size() + ele.p1Ref]) continue;

              markEV[i * loop.size() + ele.p1Ref] = 1;
            } else {
              // p4
              if (markEV[i * loop.size() + ele.p2Ref]) continue;

              markEV[i * loop.size() + ele.p2Ref] = 1;
            }

            std::cout << "Center (line-vertex on e2): " << circleCenter
                      << std::endl;

          } else if (onE2) {
            // e1 endpoint
            if (!intersectEndpoint(normal2, c2, p1, p2, circleCenter, radius,
                                   circleCenter)) {
              continue;
            }
            if (e1T < 0) {
              // p1
              if (markEV[i * loop.size() + i]) continue;

              markEV[i * loop.size() + i] = 1;
            } else {
              // p2
              if (markEV[i * loop.size() + (i + 1) % loop.size()]) continue;

              markEV[(i + 1) % loop.size()] = 1;
            }

            std::cout << "Center (line-vertex on e1): " << circleCenter
                      << std::endl;
          } else {
            // Not on line segemnt skip
            continue;
          }

          {
            vec2 tangent1 =
                circleCenter - (la::dot(normal1, circleCenter) + c1) /
                                   la::dot(normal1, normal1) * normal1;

            vec2 tangent2 =
                circleCenter - (la::dot(normal2, circleCenter) + c2) /
                                   la::dot(normal2, normal2) * normal2;

            vec2 v_start = tangent1 - circleCenter;
            vec2 v_end = tangent2 - circleCenter;

            double start_rad = atan2(v_start.y, v_start.x);
            double end_rad = atan2(v_end.y, v_end.x);

            // Normalize to [0, 2π]
            if (start_rad < 0) start_rad += 2 * M_PI;
            if (end_rad < 0) end_rad += 2 * M_PI;

            // Sort result by CCW
            double arcAngle = end_rad - start_rad;
            if (arcAngle < 0) arcAngle += 2 * M_PI;

            if (arcAngle <= M_PI) {
              circleConnection[i].emplace_back(
                  Info{circleCenter, e1T, e2T, tangent1, tangent2, i, ele.p1Ref,
                       start_rad, end_rad});
            } else {
              circleConnection[ele.p1Ref].emplace_back(
                  Info{circleCenter, e2T, e1T, tangent2, tangent1, ele.p1Ref, i,
                       end_rad, start_rad});
            }
          }

          std::cout << "Circle center " << circleCenter << " Vetex index "
                    << newLoop.size() << "~" << newLoop.size() + 20
                    << std::endl;

          // NOTE: inter result shown in upper figure
          const uint32_t seg = 20;
          for (size_t k = 0; k != seg; k++) {
            newLoop.push_back(circleCenter +
                              vec2{radius * cos(M_PI * 2 / seg * k),
                                   radius * sin(M_PI * 2 / seg * k)});
          }
        }
      } else {
        std::cout << std::endl;
      }
    }
  }

  // Construct Reuslt

  for (size_t i = 0; i != circleConnection.size(); i++) {
    std::cout << circleConnection[i].size();
    for (size_t j = 0; j != circleConnection[i].size(); j++) {
      std::cout << "\t" << circleConnection[i][j].e1 << " "
                << circleConnection[i][j].e2 << " " << circleConnection[i][j].t1
                << " " << circleConnection[i][j].t2 << std::endl;
    }

    std::cout << std::endl;
  }

  // Do tracing
  SimplePolygon rLoop{};

  while (true) {
    // Tracing to construct result
    size_t currentEdgeIndex = 0, endEdgeIndex = 0;

    double currentEdgeT = 0;

    auto it = circleConnection.begin();
    for (; it != circleConnection.end(); it++) {
      if (!it->empty()) {
        Info& info = *it->begin();

        double total_arc_angle = info.endRad - info.startRad;
        const uint32_t seg = 10;

        for (uint32_t seg_i = 0; seg_i < seg; ++seg_i) {
          double fraction = static_cast<double>(seg_i) / (seg - 1);
          double current_angle = info.startRad + fraction * total_arc_angle;

          vec2 point_on_arc = {info.center.x + radius * cos(current_angle),
                               info.center.y + radius * sin(current_angle)};

          rLoop.push_back(point_on_arc);
        }

        it->erase(it->begin());

        currentEdgeIndex = info.e2;
        endEdgeIndex = info.e1;
        currentEdgeT = info.t2;

        break;
      }
    }

    if (it == circleConnection.end()) break;

    while (currentEdgeIndex != endEdgeIndex) {
      auto it = std::find_if(circleConnection[currentEdgeIndex].begin(),
                             circleConnection[currentEdgeIndex].end(),
                             [&currentEdgeT](const Info& ele) -> bool {
                               return ele.t1 > currentEdgeT;
                             });

      if (it != circleConnection[currentEdgeIndex].end()) {
        // Found next circle fillet

        double total_arc_angle = it->endRad - it->startRad;
        const uint32_t seg = 10;

        for (uint32_t seg_i = 0; seg_i < seg; ++seg_i) {
          double fraction = static_cast<double>(seg_i) / (seg - 1);
          double current_angle = it->startRad + fraction * total_arc_angle;

          vec2 point_on_arc = {it->center.x + radius * cos(current_angle),
                               it->center.y + radius * sin(current_angle)};

          rLoop.push_back(point_on_arc);
        }

        currentEdgeIndex = it->e2;
        endEdgeIndex = it->e1;
        currentEdgeT = it->t1;

        circleConnection[currentEdgeIndex].erase(it);
      } else {
        // Not found, just add vertex

        rLoop.push_back(loop[(currentEdgeIndex + 1) % loop.size()]);
        currentEdgeIndex = (currentEdgeIndex + 1) % loop.size();
      }
    }
  }
  newPoly.push_back(rLoop);

  return newPoly;
}

int main() {
  if (false) {
    auto obj = manifold::Manifold::Cube();

    auto mesh = obj.GetMeshGL();
    Manifold::Fillet(mesh, 5, {});
  }

  manifold::Polygons Rect{{vec2{0, 0}, vec2{0, 5}, vec2{5, 5}, vec2{5, 0}}},
      Tri{{vec2{0, 0}, vec2{0, 5}, vec2{5, 0}}}, AShape{{vec2{}}},
      UShape{{vec2{0, 0}, vec2{-1, 5}, vec2{3, 1}, vec2{7, 5}, vec2{6, 0}}},
      ZShape{{vec2{0, 0}, vec2{4, 4}, vec2{0, 6}, vec2{6, 6}, vec2{3, 1},
              vec2{6, 0}}},
      WShape{{vec2{0, 0}, vec2{-2, 5}, vec2{0, 3}, vec2{2, 5}, vec2{4, 3},
              vec2{6, 5}, vec2{4, 0}, vec2{2, 3}}},
      TShape{{vec2{0, 0}, vec2{0, 5}, vec2{2, 5}, vec2{0, 8}, vec2{4, 8},
              vec2{3, 5}, vec2{5, 5}, vec2{5, 0}}};

  const manifold::Polygons poly = Rect;
  const double radius = 0.7;

  std::vector<PolygonTest> result{
      // poly,
      // PolygonTest(VertexByVertex(radius, poly)),
      PolygonTest(RollingBall(radius, poly)),
  };

  // UnionFind

  Save("../project/result.txt", result);

  return 0;
}