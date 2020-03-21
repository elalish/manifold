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

#pragma once
#include "manifold.h"

// These are mostly 3D-printable designs I've invented over the years,
// translated from their original OpenSCAD to C++ to demonstrate the usage of
// this library. You can find the originals here:
// http://www.thingiverse.com/emmett
// These also each have tests you can find in test/samples_test.cpp, which have
// nice parameter choices for making some of the specific designs I print. While
// the source code is under the Apache License above, I license all of my
// designs (the output of those tests if you uncomment the export lines) under
// CC-BY-SA: https://creativecommons.org/licenses/by-sa/2.0/, which means you're
// welcome to print and sell them, so long as you attribute the design to
// Emmett Lalish and share any derivative works under the same license.

namespace manifold {

// p is the number of times the thread passes through the donut hole and q is
// the number of times the thread circles the donut. If p and q have a common
// factor then you will get multiple separate, interwoven knots. This is an
// example of using the Warp() method on a manifold, thus avoiding any
// handling of triangles. It also demonstrates copying meshes, since that is not
// automatic.
Manifold TorusKnot(int p, int q, float majorRadius, float minorRadius,
                   float threadRadius, int circularSegments = 0,
                   int linearSegments = 0);

// The overall size is given by radius; the radius left for your wrist is
// roughly radius - height. Its length along your arm (the height of the print)
// is given by width. The thickness parameter is the width of the material,
// which should be equal to your printer's nozzle diameter. The number of twisty
// shapes around the outside is nDecor, while the number of cuts that enable
// stretching is nCut. The resolution is given by nDivision, the number of
// divisions along the width.
Manifold StretchyBracelet(float radius = 30.0f, float height = 8.0f,
                          float width = 15.0f, float thickness = 0.4f,
                          int nDecor = 20, int nCut = 27, int nDivision = 30);
}  // namespace manifold