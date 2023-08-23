// Copyright 2023 The Manifold Authors.
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
// From
// https://gist.github.com/ochafik/70a6b15e982b7ccd5a79ff9afd99dbcf#file-condensed-matter-scad

#include "samples.h"

namespace {
using namespace manifold;
using namespace glm;

constexpr float AtomicRadiusN2 = 0.65;
constexpr float BondPairN2 = 1.197;
constexpr float AtomicRadiusSi = 1.1;
constexpr float LatticeCellSizeSi = 5.4309;
constexpr float fccOffset = 0.25;
constexpr float AtomicRadiusC = 0.7;
constexpr float LatticeCellSizeC = 3.65;
constexpr float cellLenA = 2.464;
constexpr float cellLenB = cellLenA;
constexpr float cellLenC = 6.711;
constexpr float cellAngleA = 90;
constexpr float cellAngleB = cellAngleA;
constexpr float cellAngleC = 120;
constexpr float LayerSeperationC = 3.364;

Manifold bond(int fn, vec3 p1 = {0, 0, 0}, vec3 p2 = {1, 1, 1}, float ar1 = 1.0,
              float ar2 = 2.0) {
  float cyR = std::min(ar1, ar2) / 5.0;
  float dist = length(p1 - p2);
  vec3 cyC = (p1 + p2) / 2.0f;
  float beta = degrees(acos((p1.z - p2.z) / dist));
  float gamma = degrees(atan2(p1.y - p2.y, p1.x - p2.x));
  vec3 rot = {0.0, beta, gamma};
  return Manifold::Cylinder(dist, cyR, -1, fn, true)
      .Rotate(rot.x, rot.y, rot.z)
      .Translate(cyC);
}

Manifold bondPair(int fn, float d = 0.0, float ar = 1.0) {
  float axD = pow(d, 1.0 / 3.0);
  vec3 p1 = {+axD, -axD, -axD};
  vec3 p2 = {-axD, +axD, +axD};
  Manifold sphere = Manifold::Sphere(ar, fn);
  return sphere.Translate(p1) + sphere.Translate(p2) + bond(fn, p1, p2, ar, ar);
}

Manifold hexagonalClosePacked(int fn, vec3 dst = {1.0, 1.0, 1.0},
                              float ar = 1.0) {
  std::vector<Manifold> parts;
  vec3 p1 = {0, 0, 0};
  parts.push_back(Manifold::Sphere(ar, fn));
  float baseAg = 30;
  vec3 ag = {baseAg, baseAg + 120, baseAg + 240};
  vec3 points[] = {{cosd(ag.x) * dst.x, sind(ag.x) * dst.x, 0},
                   {cosd(ag.y) * dst.y, sind(ag.y) * dst.y, 0},
                   {cosd(ag.z) * dst.z, sind(ag.z) * dst.z, 0}};
  for (vec3 p2 : points) {
    parts.push_back(Manifold::Sphere(ar, fn).Translate(p2));
    parts.push_back(bond(fn, p1, p2, ar, ar));
  }
  return Manifold::BatchBoolean(parts, OpType::Add);
}

Manifold fccDiamond(int fn, float ar = 1.0, float unitCell = 2.0,
                    float fccOffset = 0.25) {
  std::vector<Manifold> parts;
  float huc = unitCell / 2.0;
  float od = fccOffset * unitCell;
  vec3 interstitial[] = {
      {+od, +od, +od}, {+od, -od, -od}, {-od, +od, -od}, {-od, -od, +od}};
  vec3 corners[] = {{+huc, +huc, +huc},
                    {+huc, -huc, -huc},
                    {-huc, +huc, -huc},
                    {-huc, -huc, +huc}};
  vec3 fcc[] = {{+huc, 0, 0}, {-huc, 0, 0}, {0, +huc, 0},
                {0, -huc, 0}, {0, 0, +huc}, {0, 0, -huc}};
  for (auto p : corners) parts.push_back(Manifold::Sphere(ar, fn).Translate(p));

  for (auto p : fcc) parts.push_back(Manifold::Sphere(ar, fn).Translate(p));
  for (auto p : interstitial)
    parts.push_back(Manifold::Sphere(ar, fn).Translate(p));

  vec3 bonds[][2] = {{interstitial[0], corners[0]}, {interstitial[0], fcc[0]},
                     {interstitial[0], fcc[2]},     {interstitial[0], fcc[4]},
                     {interstitial[1], corners[1]}, {interstitial[1], fcc[0]},
                     {interstitial[1], fcc[3]},     {interstitial[1], fcc[5]},
                     {interstitial[2], corners[2]}, {interstitial[2], fcc[1]},
                     {interstitial[2], fcc[2]},     {interstitial[2], fcc[5]},
                     {interstitial[3], corners[3]}, {interstitial[3], fcc[1]},
                     {interstitial[3], fcc[3]},     {interstitial[3], fcc[4]}};
  for (auto b : bonds) parts.push_back(bond(fn, b[0], b[1], ar, ar));

  return Manifold::BatchBoolean(parts, OpType::Add);
}

Manifold SiCell(int fn, float x = 1.0, float y = 1.0, float z = 1.0) {
  return fccDiamond(fn, AtomicRadiusSi, LatticeCellSizeSi, fccOffset)
      .Translate({LatticeCellSizeSi * x, LatticeCellSizeSi * y,
                  LatticeCellSizeSi * z});
}

Manifold SiN2Cell(int fn, float x = 1.0, float y = 1.0, float z = 1.0) {
  float n2Offset = LatticeCellSizeSi / 8;
  return bondPair(fn, BondPairN2, AtomicRadiusN2)
             .Translate({LatticeCellSizeSi * x - n2Offset,
                         LatticeCellSizeSi * y + n2Offset,
                         LatticeCellSizeSi * z + n2Offset}) +
         SiCell(fn, x, y, z);
}

Manifold GraphiteCell(int fn, vec3 xyz = {1.0, 1.0, 1.0}) {
  vec3 loc = {(cellLenA * xyz.x * cosd(30) * 2),
              ((cellLenB * sind(30)) + cellLenC) * xyz.y, xyz.z};
  return hexagonalClosePacked(fn, {cellLenA, cellLenB, cellLenC}, AtomicRadiusC)
      .Translate(loc);
}

}  // namespace

namespace manifold {
Manifold CondensedMatter(int fn) {
  std::vector<Manifold> parts;
  float siOffset = 3.0 * LatticeCellSizeSi / 8.0;
  for (int x = -3; x <= 3; x++)
    for (int y = -1; y <= 2; y++)
      parts.push_back(
          GraphiteCell(fn, {x + (y % 2 == 0 ? 0.0 : 0.5), y,
                            LayerSeperationC * 0.5 + LatticeCellSizeSi * 1.5})
              .Translate({0, -siOffset, 0})
              .Rotate(0, 0, 45));

  float xyPlane[] = {-2, -1, 0, +1, +2};
  for (float x : xyPlane)
    for (float y : xyPlane) parts.push_back(SiN2Cell(fn, x, y, 1));
  return Manifold::BatchBoolean(parts, OpType::Add);
}
}  // namespace manifold
