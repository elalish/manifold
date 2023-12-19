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

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/convex_decomposition_3.h>

#include <chrono>
#include <iostream>

#include "manifold.h"

using namespace manifold;

using Kernel = CGAL::Epeck;
using Polyhedron = CGAL::Polyhedron_3<Kernel>;
using HalfedgeDS = Polyhedron::HalfedgeDS;
using NefPolyhedron = CGAL::Nef_polyhedron_3<Kernel>;

template <class HDS>
class BuildFromManifold : public CGAL::Modifier_base<HDS> {
 public:
  using Vertex = typename HDS::Vertex;
  using Point = typename Vertex::Point;
  BuildFromManifold(const Manifold manifold) : manifold(manifold) {}
  void operator()(HDS &hds) {
    // Postcondition: hds is a valid polyhedral surface.
    CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, true);
    auto mesh = manifold.GetMeshGL();
    B.begin_surface(mesh.NumVert(), mesh.NumTri(), mesh.NumTri() * 3);

    for (size_t i = 0; i < mesh.vertProperties.size(); i += mesh.numProp) {
      B.add_vertex(Point(mesh.vertProperties[i], mesh.vertProperties[i + 1],
                         mesh.vertProperties[i + 2]));
    }
    for (size_t i = 0; i < mesh.triVerts.size(); i += 3) {
      B.begin_facet();
      for (const int j : {0, 1, 2}) B.add_vertex_to_facet(mesh.triVerts[i + j]);
      B.end_facet();
    }
    B.end_surface();
  }

 private:
  const Manifold manifold;
};

int main(int argc, char **argv) {
  Manifold spherecube = Manifold::Cube(glm::vec3(1), true) - Manifold::Sphere(0.6, 100);
  BuildFromManifold<HalfedgeDS> build(spherecube);
  std::cout << "nTri = " << spherecube.NumTri() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  Polyhedron poly;
  poly.delegate(build);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "to Polyhedron took " << elapsed.count() << " sec"
            << std::endl;

  start = std::chrono::high_resolution_clock::now();
  NefPolyhedron np(poly);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "conversion to Nef Polyhedron took " << elapsed.count()
            << " sec" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  CGAL::convex_decomposition_3(np);
  std::cout << "decomposed into " << np.number_of_volumes() << " parts"
            << std::endl;
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "decomposition took " << elapsed.count() << " sec" << std::endl
            << std::endl;
  return 0;
}
