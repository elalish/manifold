#include <array>
#include <cassert>
#include <iomanip>
#include <iostream>

#include "../src/collider.h"
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

void VertexByVertex(double radius) {
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

  manifold::Polygons poly{{vec2{0, 0}, vec2{0, 5}, vec2{5, 5}}};
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

  Print(newPoly);
}

void RollingBall(double radius) {
  manifold::Polygons poly{{vec2{0, 0}, vec2{1, 1}, vec2{0, 1}}};

  std::vector<manifold::Box> boxVec;
  std::vector<uint32_t> mortonVec;

  for (size_t i = 0; i != poly[0].size(); i++) {
    const vec2 p1 = poly[0][i], p2 = poly[0][(i + 1) % poly[0].size()];

    vec3 center = toVec3(p1) + toVec3(p2);
    center /= 2;

    manifold::Box bbox(toVec3(p1), toVec3(p2));

    mortonVec.push_back(manifold::Collider::MortonCode(center, bbox));

    boxVec.push_back(bbox);
  }

  manifold::Collider collider(boxVec, mortonVec);

  for (size_t i = 0; i != poly[0].size(); i++) {
    const vec2 p1 = poly[0][i], p2 = poly[0][(i + 1) % poly[0].size()];
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
  }
}

int main() {
  if (false) {
    auto obj = manifold::Manifold::Cube();

    auto mesh = obj.GetMeshGL();
    Manifold::Fillet(mesh, 5, {});
  }

  VertexByVertex(0.5);

  // RollingBall(0.5);
  // manifold::Collider co(poly);

  // co.Collisions(poly,);

  return 0;
}