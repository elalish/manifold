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
#include <CGAL/minkowski_sum_3.h>

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
  Manifold spherecube =
      Manifold::Cube(glm::vec3(1), true) - Manifold::Sphere(0.6, 100);
  Manifold smallsphere = Manifold::Sphere(0.1, 20);

  BuildFromManifold<HalfedgeDS> build(spherecube);
  std::cout << "nTri = " << spherecube.NumTri() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  Polyhedron poly;
  poly.delegate(build);
  std::cout << "to Polyhedron took "
            << (std::chrono::high_resolution_clock::now() - start).count() / 1e9
            << " sec" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  NefPolyhedron np(poly);
  std::cout << "conversion to Nef Polyhedron took "
            << (std::chrono::high_resolution_clock::now() - start).count() / 1e9
            << " sec" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  auto convexDecomposition = spherecube.ConvexDecomposition();
  std::cout << "[MANIFOLD] decomposed into " << convexDecomposition.size()
            << " parts in "
            << (std::chrono::high_resolution_clock::now() - start).count() / 1e9
            << " sec" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  auto generalMinkowskiSum = Manifold::Minkowski(spherecube, smallsphere);
  std::cout << "[MANIFOLD] general minkowski summed in "
            << (std::chrono::high_resolution_clock::now() - start).count() / 1e9
            << " sec" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  auto naiveMinkowskiSum = Manifold::Minkowski(spherecube, smallsphere, true);
  std::cout << "[MANIFOLD] naive minkowski summed in "
            << (std::chrono::high_resolution_clock::now() - start).count() / 1e9
            << " sec" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  CGAL::convex_decomposition_3(np);
  std::cout << "[CGAL] decomposed into " << np.number_of_volumes()
            << " parts in "
            << (std::chrono::high_resolution_clock::now() - start).count() / 1e9
            << " sec" << std::endl;

  // Create the Small Sphere NEF Polyhedron for Minkowski Summing
  Polyhedron poly2;
  poly.delegate(BuildFromManifold<HalfedgeDS>(smallsphere));
  NefPolyhedron np2(poly);

  start = std::chrono::high_resolution_clock::now();
  CGAL::minkowski_sum_3(np, np2);
  std::cout << "[CGAL] minkowski summed in "
            << (std::chrono::high_resolution_clock::now() - start).count() / 1e9
            << " sec" << std::endl;

  return 0;
}
