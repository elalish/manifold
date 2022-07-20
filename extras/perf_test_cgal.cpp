// Copyright 2020 Emmett Lalish & contributors.
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
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Surface_mesh.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "manifold.h"

using namespace manifold;

// Epick = Exact predicates Inexact constructions. Seems fair to use to compare
// to Manifold, which seems to store double coordinates.
typedef CGAL::Epick Kernel;

// Epeck = Exact predicates Exact constructions. What OpenSCAD uses to guarantee
// geometry ends up where it should even after many operations. typedef
// CGAL::Epeck Kernel;

typedef CGAL::Point_3<Kernel> Point;
typedef CGAL::Surface_mesh<Point> TriangleMesh;
typedef CGAL::SM_Vertex_index Vertex;

void manifoldToCGALSurfaceMesh(Manifold &manifold, TriangleMesh &cgalMesh) {
  auto maniMesh = manifold.GetMesh();

  const int n = maniMesh.vertPos.size();
  std::vector<Vertex> vertices(n);
  for (size_t i = 0; i < n; i++) {
    auto &vert = maniMesh.vertPos[i];
    vertices[i] = cgalMesh.add_vertex(Point(vert.x, vert.y, vert.z));
  }

  for (auto &triVert : maniMesh.triVerts) {
    std::vector<Vertex> polygon{vertices[triVert[0]], vertices[triVert[1]],
                                vertices[triVert[2]]};
    cgalMesh.add_face(polygon);
  }
}

int main(int argc, char **argv) {
  for (int i = 0; i < 8; ++i) {
    Manifold sphere = Manifold::Sphere(1, (8 << i) * 4);
    Manifold sphere2 = sphere;
    sphere2.Translate(glm::vec3(0.5));

    TriangleMesh cgalSphere, cgalSphere2, cgalOut;
    manifoldToCGALSurfaceMesh(sphere, cgalSphere);
    manifoldToCGALSurfaceMesh(sphere2, cgalSphere2);

    auto start = std::chrono::high_resolution_clock::now();
    auto result =
        CGAL::Polygon_mesh_processing::corefine_and_compute_difference(
            // CGAL::Polygon_mesh_processing::corefine_and_compute_union(
            // CGAL::Polygon_mesh_processing::corefine_and_compute_intersection(
            cgalSphere, cgalSphere2, cgalSphere);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "nTri = " << sphere.NumTri() << ", time = " << elapsed.count()
              << " sec" << std::endl;
  }
}
