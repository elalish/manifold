// Copyright 2022 The Manifold Authors.
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
#include <ostream>

#include "gtest/gtest.h"
#include "manifold/common.h"
#include "manifold/manifold.h"

#ifdef MANIFOLD_EXPORT
#include "manifold/meshIO.h"
#endif

using namespace manifold;

struct Options {
  bool exportModels = false;
  manifold::ExecutionParams params = {};
};

extern Options options;

struct MeshSize {
  int numVert, numTri;
  int numProp = 0;
  int numPropVert = numVert;
};

Polygons SquareHole(double xOffset = 0.0);
MeshGL Csaszar();
Manifold Gyroid();
MeshGL TetGL();
MeshGL CubeSTL();
MeshGL CubeUV();
MeshGL WithIndexColors(const MeshGL& in);
MeshGL WithPositionColors(const Manifold& in);
float GetMaxProperty(const MeshGL& mesh, int channel);
float GetMinProperty(const MeshGL& mesh, int channel);
void CheckFinite(const MeshGL& mesh);
void Identical(const MeshGL& mesh1, const MeshGL& mesh2);
void RelatedGL(const Manifold& out, const std::vector<MeshGL>& originals,
               bool checkNormals = false, bool updateNormals = false);
void ExpectMeshes(const Manifold& manifold,
                  const std::vector<MeshSize>& meshSize);
void CheckStrictly(const Manifold& manifold);
void CheckGL(const Manifold& manifold);
#ifdef MANIFOLD_EXPORT
Manifold ReadMesh(const std::string& filename);
#endif
void RegisterPolygonTests();
