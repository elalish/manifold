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

#include <iostream>
#include <optional>
#include <vector>

#include "../collider.h"
#include "../vec.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"

namespace {
using namespace manifold;

const double EPSILON = 1e-9;

vec3 toVec3(vec2 in) { return vec3(in.x, in.y, 0); }

bool intersectLine(const vec2& p1, const vec2& p2, const vec2& p3,
                   const vec2& p4, vec2& intersectionPoint) {
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
bool intersectCircleDetermine(const vec2& p1, const vec2& p2,
                              const vec2& center, double radius) {
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
bool intersectEndpoint(const vec2& line_normal, double line_c,
                       const vec2& vertex_p1, const vec2& vertex_p2,
                       double filletRadius, vec2& center) {
  // Closer endpoint
  vec2 endpoint =
      (la::length2(center - vertex_p1) < la::length2(center - vertex_p2))
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

  center =
      (la::length2(sol1 - center) < la::length2(sol2 - center)) ? sol1 : sol2;
  return true;
};

// Projection
bool isProjectionOnSegment(const vec2& c, const vec2& p1, const vec2& p2,
                           double& t) {
  t = la::dot(c - p1, p2 - p1) / la::length2(p2 - p1);

  return t >= 0 && t <= 1;
};
}  // namespace

namespace manifold {

Polygons CrossSection::Fillet(const Polygons& polygons, double radius,
                              int circularSegments) {
  using namespace manifold;

  auto& loop = polygons[0];

  std::vector<std::optional<vec2>> adjustCircleCenter;

  manifold::Polygons newPoly = polygons;
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

#ifdef MANIFOLD_DEBUG
    if (ManifoldParams().verbose) {
      boxVec.Dump();
      mortonVec.Dump();
    }
#endif
  }

  const double EPSILON = 1E-7;

  manifold::Collider collider(boxVec, mortonVec);

  std::vector<uint8_t> markEE(loop.size() * loop.size(), 0);
  std::vector<uint8_t> markEV(loop.size() * loop.size(), 0);

  struct Info {
    vec2 center;

    double t1, t2;
    size_t e1, e2;
    double startRad, endRad;
  };

  std::vector<std::vector<Info>> circleConnection(loop.size(),
                                                  std::vector<Info>());

  std::cout << "Collider BBox Testing" << std::endl;

  // create BBox for every line to find Collision
  for (size_t i = 0; i != loop.size(); i++) {
    // CW, p1 p2 -> current edge start end
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

      // CW, p3 p4 -> bbox hit edge start end
      vec2 p3 = loop[ele.p1Ref], p4 = loop[ele.p2Ref];
      vec2 t;

      std::cout << "Testing " << ele.p1Ref << "->" << ele.p2Ref << "\t";

      bool flag1 = intersectLine(offsetP1, offsetP2, p3, p4, t),
           flag2 = intersectCircleDetermine(p3, p4, p1 + perp * radius, radius),
           flag3 = intersectCircleDetermine(p3, p4, p2 + perp * radius, radius);

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

        // Calc circle center
        {
          vec2 e1 = p2 - p1;
          vec2 normal1 = {e1.y, -e1.x};
          double c1 = -la::dot(normal1, p1);
          vec2 e2 = p4 - p3;
          vec2 normal2 = {e2.y, -e2.x};
          double c2 = -la::dot(normal2, p3);
          if (la::length(e1) < EPSILON || la::length(e2) < EPSILON) {
            // FIXME: Degenerate
            std::cout << "Degenerate" << std::endl;
            continue;
          }
          mat2 A = {{normal1.x, normal2.x}, {normal1.y, normal2.y}};
          vec2 b = {radius * la::length(normal1) - c1,
                    radius * la::length(normal2) - c2};
          if (std::abs(la::determinant(A)) < EPSILON) {
            // FIXME: Parallel line
            std::cout << "Parallel" << std::endl;
            continue;
          }
          vec2 circleCenter = la::mul(la::inverse(A), b);

          double e1T = 0, e2T = 0;
          bool onE1 = isProjectionOnSegment(circleCenter, p1, p2, e1T),
               onE2 = isProjectionOnSegment(circleCenter, p3, p4, e2T);

          // Check Circle center projection for tangent point status
          if (onE1 && onE2) {
            // Tangent point on both edge
          } else if ((!onE1) && (!onE2)) {
            // Not on both line, invalid result
            continue;
          } else if (onE1) {
            // Only on e1, tangent point might be e2 endpoint
            // Calc new circle center
            if (!intersectEndpoint(normal1, c1, p3, p4, radius, circleCenter)) {
              continue;
            }
            if (e2T < 0) {
              // p3
              if (markEV[i * loop.size() + ele.p1Ref]) continue;

              markEV[i * loop.size() + ele.p1Ref] = 1;

              e2T = 0;
            } else {
              // p4
              if (markEV[i * loop.size() + ele.p2Ref]) continue;

              markEV[i * loop.size() + ele.p2Ref] = 1;

              e2T = 1;
            }

            // Calc new tangent point's parameter value, and check is valid
            isProjectionOnSegment(circleCenter, p1, p2, e1T);
            if (e1T < EPSILON || e1T > (1 + EPSILON)) continue;

            std::cout << "Center (line-vertex on e2): ";

          } else if (onE2) {
            // tangent point might be e1 endpoint

            if (!intersectEndpoint(normal2, c2, p1, p2, radius, circleCenter)) {
              continue;
            }

            if (e1T < 0) {
              // p1
              if (markEV[ele.p1Ref * loop.size() + i]) continue;

              markEV[ele.p1Ref * loop.size() + i] = 1;

              e1T = 0;
            } else {
              // p2
              if (markEV[ele.p1Ref * loop.size() + (i + 1) % loop.size()])
                continue;

              markEV[ele.p1Ref * loop.size() + (i + 1) % loop.size()] = 1;

              e1T = 1;
            }

            isProjectionOnSegment(circleCenter, p3, p4, e2T);

            if (e2T < EPSILON || e2T > (1 + EPSILON)) continue;
            std::cout << "Center (line-vertex on e1): ";
          }

          {
            vec2 tangent1 = p1 + e1T * e1;
            vec2 tangent2 = p3 + e2T * e2;

            vec2 v_start = tangent1 - circleCenter;
            vec2 v_end = tangent2 - circleCenter;

            double start_rad = atan2(v_start.y, v_start.x);
            double end_rad = atan2(v_end.y, v_end.x);

            // Normalize to [0, 2π]
            if (start_rad < 0) start_rad += 2 * M_PI;
            if (end_rad < 0) end_rad += 2 * M_PI;

            // Sort result by CW
            double arcAngle = end_rad - start_rad;
            if (arcAngle < 0) arcAngle += 2 * M_PI;

            if (arcAngle <= M_PI) {
              circleConnection[ele.p1Ref].emplace_back(Info{
                  circleCenter, e2T, e1T, ele.p1Ref, i, end_rad, start_rad});
            } else {
              circleConnection[i].emplace_back(Info{
                  circleCenter, e1T, e2T, i, ele.p1Ref, start_rad, end_rad});
            }
          }

#ifdef MANIFOLD_DEBUG
          if (ManifoldParams().verbose) {
            std::cout << "Circle center " << circleCenter << " " << i << " "
                      << ele.p1Ref << " Vetex index " << newLoop.size() << "~"
                      << newLoop.size() + 20 << std::endl;
          }
#endif

          // NOTE: inter result shown in upper figure
          // const uint32_t seg = 20;
          // for (size_t k = 0; k != seg; k++) {
          //   newLoop.push_back(circleCenter +
          //                     vec2{radius * cos(M_PI * 2 / seg * k),
          //                          radius * sin(M_PI * 2 / seg * k)});
          // }
        }
      } else {
        std::cout << std::endl;
      }
    }
  }

  // Construct Reuslt

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    for (size_t i = 0; i != circleConnection.size(); i++) {
      std::cout << i << " " << circleConnection[i].size();
      for (size_t j = 0; j != circleConnection[i].size(); j++) {
        std::cout << "\t" << circleConnection[i][j].e1 << " "
                  << circleConnection[i][j].e2 << " "
                  << circleConnection[i][j].t1 << " "
                  << circleConnection[i][j].t2 << " "
                  << circleConnection[i][j].startRad << " "
                  << circleConnection[i][j].endRad << std::endl;
      }

      std::cout << std::endl;
    }
  }
#endif
  // Do tracing

  while (true) {
    SimplePolygon rLoop{};

    std::vector<size_t> tracingEList;
    std::vector<size_t> mapVV;

    // Tracing to construct result
    size_t currentEdgeIndex = 0, endEdgeIndex = 0;

    double currentEdgeT = 0;

    auto it = circleConnection.begin();
    for (; it != circleConnection.end(); it++) {
      if (!it->empty()) {
        Info& info = *it->begin();

        double total_arc_angle = info.endRad - info.startRad;

        if (total_arc_angle > 0) {
          total_arc_angle -= 2.0 * M_PI;
        }

        for (int i = 0; i < circularSegments; ++i) {
          double fraction = static_cast<double>(i) / (circularSegments - 1);
          double current_angle = info.startRad + fraction * total_arc_angle;

          vec2 point_on_arc = {info.center.x + radius * cos(current_angle),
                               info.center.y + radius * sin(current_angle)};

          rLoop.push_back(point_on_arc);
        }

        currentEdgeIndex = info.e2;
        endEdgeIndex = info.e1;
        currentEdgeT = info.t2;

        it->erase(it->begin());
        break;
      }
    }

    if (it == circleConnection.end()) break;

    // For detecting inner loop
    tracingEList.push_back(currentEdgeIndex);
    mapVV.push_back(rLoop.size());

    while (currentEdgeIndex != endEdgeIndex) {
      auto it = std::find_if(circleConnection[currentEdgeIndex].begin(),
                             circleConnection[currentEdgeIndex].end(),
                             [currentEdgeT, EPSILON](const Info& ele) -> bool {
                               return ele.t1 + EPSILON > currentEdgeT;
                             });

      if (it == circleConnection[currentEdgeIndex].end()) {
        // Not found, just add vertex
        // FIXME: shouldn't add vertex directly, should search for next edge
        // with fillet arc
        rLoop.push_back(loop[(currentEdgeIndex + 1) % loop.size()]);
        currentEdgeIndex = (currentEdgeIndex + 1) % loop.size();
        currentEdgeT = 0;

        tracingEList.push_back(currentEdgeIndex);
        mapVV.push_back(rLoop.size());
      } else {
        // Found next circle fillet

        Info t = *it;
        circleConnection[currentEdgeIndex].erase(it);

        double total_arc_angle = t.endRad - t.startRad;
        if (total_arc_angle > 0) {
          total_arc_angle -= 2.0 * M_PI;
        }

        const uint32_t seg = 10;

        for (uint32_t seg_i = 0; seg_i < seg; ++seg_i) {
          double fraction = static_cast<double>(seg_i) / (seg - 1);
          double current_angle = t.startRad + fraction * total_arc_angle;

          vec2 point_on_arc = {t.center.x + radius * cos(current_angle),
                               t.center.y + radius * sin(current_angle)};

          rLoop.push_back(point_on_arc);
        }

        // Check if current result contain inner loop

        auto itt = std::find(tracingEList.rbegin(), tracingEList.rend(), t.e2);

        if (itt != tracingEList.rend()) {
          size_t pos = tracingEList.size() -
                       std::distance(tracingEList.rbegin(), itt) - 1;

          SimplePolygon innerLoop{};
          innerLoop.insert(innerLoop.end(), rLoop.begin() + mapVV[pos],
                           rLoop.end());

          newPoly.push_back(innerLoop);

          rLoop.erase(rLoop.begin() + mapVV[pos], rLoop.end());

          currentEdgeIndex = (currentEdgeIndex + 1) % loop.size();
          currentEdgeT = 0;

          continue;
        }

        currentEdgeIndex = t.e2;
        currentEdgeT = t.t2;

        tracingEList.push_back(currentEdgeIndex);
        mapVV.push_back(rLoop.size());
      }
    }
    newPoly.push_back(rLoop);
  }

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    std::cout << "Result loop count:" << newPoly.size() << std::endl;
  }
#endif

  return newPoly;
}

}  // namespace manifold