// Copyright 2020 Emmett Lalish
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

#include "manifold.h"
#include "meshIO.h"

using namespace manifold;

int main(int argc, char **argv) {
  ALWAYS_ASSERT(argc > 1, userErr, "No filename given!");
  std::string filename = argv[1];

  Mesh mesh = ImportMesh(filename);

  std::cout << mesh.vertPos.size() << " vertices input" << std::endl;
  std::cout << mesh.triVerts.size() << " triangles input" << std::endl;

  Manifold manifold(mesh);

  std::cout << "Manifold is valid" << std::endl;
  std::cout << manifold.NumVert() << " vertices now" << std::endl;
  std::cout << manifold.NumTri() << " triangles now" << std::endl;
  std::cout << "Genus = " << manifold.Genus() << std::endl;
  std::cout << manifold.NumDegenerateTris() << " degenerate triangles"
            << std::endl;
}
