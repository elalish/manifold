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
#include "gtest/gtest.h"
#include "manifold.h"
#include "public.h"

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

Polygons SquareHole(float xOffset = 0.0);
Mesh Csaszar();
Mesh Gyroid();
Mesh Tet();
MeshGL TetGL();
MeshGL CubeSTL();
MeshGL WithIndexColors(const Mesh& in);
MeshGL WithPositionColors(const Manifold& in);
MeshGL WithNormals(const Manifold& in);
MeshGL CubeUV();
float GetMaxProperty(const MeshGL& mesh, int channel);
float GetMinProperty(const MeshGL& mesh, int channel);
void Identical(const Mesh& mesh1, const Mesh& mesh2);
void RelatedGL(const Manifold& out, const std::vector<MeshGL>& originals,
               bool checkNormals = false, bool updateNormals = false);
void ExpectMeshes(const Manifold& manifold,
                  const std::vector<MeshSize>& meshSize);
void CheckNormals(const Manifold& manifold);
void CheckStrictly(const Manifold& manifold);
void CheckGL(const Manifold& manifold);
