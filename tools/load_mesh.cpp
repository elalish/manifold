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

#include <iostream>
#include "manifold.h"
#include "meshIO.h"

using namespace manifold;

int main(int argc, char **argv) {
  std::string filename = "tetra.ply";
  if (argc > 1) filename = argv[1];

  Mesh mesh = ImportMesh(filename);
  Manifold manifold(mesh);
  std::cout << "Manifold is ";
  if (!manifold.IsValid()) std::cout << "NOT " << std::endl;
  std::cout << "valid" << std::endl;
}
