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

#include <iostream>

#include "manifold.h"
#include "meshIO.h"

using namespace manifold;

/*
 */
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Specify an input filename." << std::endl;
    return 1;
  }

  manifold::ManifoldParams().verbose = true;

  const std::string filename(argv[1]);

  MeshGL input = ImportMesh(filename);
  std::cout << input.NumVert() << " vertices, " << input.NumTri()
            << " triangles" << std::endl;

  if (input.Merge())
    std::cout << filename << " is not manifold, attempting to fix."
              << std::endl;

  const Manifold manifold(input);
  if (manifold.Status() != Manifold::Error::NoError) {
    std::cout << "Could not make a valid manifold, error: "
              << (int)manifold.Status() << std::endl;
    return 2;
  }

  const std::vector<Manifold> parts = manifold.Decompose();
  std::cout << parts.size() << " objects:" << std::endl;
  for (const Manifold& part : parts) {
    auto prop = part.GetProperties();
    std::cout << part.NumVert() << " vertices, " << part.NumTri()
              << " triangles, volume = " << prop.volume
              << ", surface area = " << prop.surfaceArea << std::endl;
  }

  if (argc == 3) {
    std::string outName = argv[2];

    std::cout << "Writing " << outName << std::endl;

    ExportMesh(outName, manifold.GetMeshGL(), {});
  }

  return 0;
}