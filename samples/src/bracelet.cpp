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

#include "samples.h"

namespace {

using namespace manifold;

Manifold Base(float radius, float width, float bigRadius, float rr, int n) {
  Manifold center = Manifold::Cylinder(width, bigRadius + rr / 2);

  Polygons circle(1);
  int m = 20;
  float dPhi = 360.0f / m;
  for (int i = 0; i < m; ++i) {
    circle[0].push_back(
        {glm::vec2(radius * cosd(dPhi * i) + rr, radius * sind(dPhi * i)), 0,
         Edge::kNoIdx});
  }
  Manifold decor = std::move(Manifold::Extrude(circle, width, 10, 180)
                                 .Scale({1.0f, 0.5f, 1.0f})
                                 .Translate({0.0f, bigRadius, 0.0f}));

  Manifold base = center + decor;
  for (int i = 1; i < 6; ++i) {
    base = base + decor.Rotate(0, 0, 360.0f / n);
  }
  return base;
}
}  // namespace

// module base(r1,w){
// render()
// union(){
// 	cylinder(r=r2+rr*0.5,h=w);
// 	for(i=[1:n]){
// 		rotate([0,0,i*360/n])translate([0,-r2,0])
// 		scale([1,0.5,1])linear_extrude(height=w,twist=180,slices=10)
// 			translate([rr,0,0])circle(r=r1,$fn=20);
// 	}
// }}

namespace manifold {

Manifold StretchyBracelet(float radius, float height, float width,
                          float thickness, int n, int m) {
  float rr = glm::pi<float>() * radius / n;
  float r1 = rr * 1.5;
  float ro = radius + (r1 + rr) * 0.5;
  float ri = ro - height;
  float a = glm::pi<float>() * 2 * ri / m - thickness;

  return Base(r1, width, radius, rr, n);

  // module hollow(){
  // difference(){
  // 	base(r1=r1,w=w);
  // 	difference(){
  // 		translate([0,0,-0.01])base(r1=r1-t,w=w+0.02);
  // 		for(i=[1:m])rotate([0,0,i*360/m])
  // 			translate([0,0,-0.02])linear_extrude(height=w+0.04)
  // 				polygon(points=[[ri,a/2],[ri,-a/2],[ro+3*t*h/a,0]],paths=[[0,1,2]]);
  // 	}
  // 	for(i=[1:m])rotate([0,0,i*360/m])
  // 		translate([0,0,-0.03])linear_extrude(height=w+0.06)
  // 			polygon(points=[[ri+t,a/2-t],[ri+t,t-a/2],[ro+t*h/a,0]],paths=[[0,1,2]]);
  // }}
}
}  // namespace manifold
