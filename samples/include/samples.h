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

#pragma once
#include "manifold.h"

namespace manifold {

/** @defgroup Samples
 * @brief Examples of usage and interesting designs
 *
 * These are mostly 3D-printable designs I've invented over the years,
 * translated from their original OpenSCAD to C++ to demonstrate the usage of
 * this library. You can find the originals here:
 * http://www.thingiverse.com/emmett These also each have tests you can find in
 * test/samples_test.cpp, which have nice parameter choices for making some of
 * the specific designs I print. While the source code is under the Apache
 * License above, I license all of my designs (the output of those tests if you
 * uncomment the export lines) under CC-BY-SA:
 * https://creativecommons.org/licenses/by-sa/2.0/, which means you're welcome
 * to print and sell them, so long as you attribute the design to Emmett Lalish
 * and share any derivative works under the same license.
 *  @{
 */

Manifold TorusKnot(int p, int q, float majorRadius, float minorRadius,
                   float threadRadius, int circularSegments = 0,
                   int linearSegments = 0);

Manifold StretchyBracelet(float radius = 30.0f, float height = 8.0f,
                          float width = 15.0f, float thickness = 0.4f,
                          int nDecor = 20, int nCut = 27, int nDivision = 30);

Manifold MengerSponge(int n = 3);

Manifold RoundedFrame(float edgeLength, float radius, int circularSegments = 0);

Manifold TetPuzzle(float edgeLength, float gap, int nDivisions);

Manifold Scallop();

Manifold GyroidModule(float size = 20, int n = 20);
/** @} */  // end of Samples
}  // namespace manifold
