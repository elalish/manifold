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

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/convex_hull_3.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <algorithm>
#include <random>

#include "manifold.h"
#include "meshIO.h"
#include "samples.h"

#include <fstream>
#include <sstream> // For string manipulation
using namespace std;
using namespace glm;
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

// Converts Manfiold to CGAL Surface Mesh
void manifoldToCGALSurfaceMesh(Manifold &manifold, TriangleMesh &cgalMesh) {
  auto maniMesh = manifold.GetMesh();

  const size_t n = maniMesh.vertPos.size();
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

// Prints the volume and area of the manifold and the convex hull of our implementation,
void PrintVolArea(Manifold &manfiold_obj, Manifold &hullManifold) {
  std::cout << manfiold_obj.GetProperties().volume
            << ",";
  std::cout << hullManifold.GetProperties().volume
            << ",";
  std::cout << manfiold_obj.GetProperties().surfaceArea
            << ",";
  std::cout << hullManifold.GetProperties().surfaceArea
            << ",";
  std::cout << manfiold_obj.NumTri() << ",";
  std::cout << hullManifold.NumTri() << ",";
  return;
}

// Prints the volume and area of the manifold and Convex Hull of CGAL Implementation
void PrintVolAreaCGAL(TriangleMesh &cgalMesh,
                  TriangleMesh &cgalHull) {
  std::cout << CGAL::Polygon_mesh_processing::volume(cgalMesh) << ",";
  std::cout << CGAL::Polygon_mesh_processing::volume(cgalHull) << ",";
  std::cout << CGAL::Polygon_mesh_processing::area(cgalMesh) << ",";
  std::cout << CGAL::Polygon_mesh_processing::area(cgalHull) << ",";
  std::cout << std::distance(cgalMesh.faces_begin(),cgalMesh.faces_end())
            << ",";
  std::cout << std::distance(cgalHull.faces_begin(), cgalHull.faces_end())
            << ",";
  return;
}

void PrintManifold(Manifold input, Manifold (Manifold::*hull_func)() const) {
  auto start = std::chrono::high_resolution_clock::now();
  Manifold hullManifold = (input.*hull_func)();
  #ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportMesh("testHullManifold.glb", hullManifold.GetMesh(), {});
  }
  #endif
  auto end = std::chrono::high_resolution_clock::now();
  PrintVolArea(input, hullManifold);
  std::chrono::duration<double> elapsed = end - start;
  std::cout << elapsed.count() << " sec" ;
}

void PrintCGAL(Manifold input, Manifold (Manifold::*hull_func)() const){
  TriangleMesh cgalInput;
  manifoldToCGALSurfaceMesh(input, cgalInput);
  std::vector<Point> points;
  for (const auto &vert : cgalInput.vertices()) {
    points.push_back(cgalInput.point(vert));
  }
  // Convex Hull
  TriangleMesh cgalHull;
  auto start = std::chrono::high_resolution_clock::now();
  CGAL::convex_hull_3(points.begin(), points.end(), cgalHull);
  auto end = std::chrono::high_resolution_clock::now();
  PrintVolAreaCGAL(cgalInput, cgalHull);
  std::chrono::duration<double> elapsed = end - start;
  std::cout << elapsed.count() << " sec" ;
}

// Constructs a Menger Sponge, and tests the convex hull implementation on it
// (you can pass the specific hull implementation to be tested). Comparing the
// volume and surface area with CGAL implementation, for various values of
// rotation
void MengerTestHull(Manifold (Manifold::*hull_func)() const, float rx, float ry,
                    float rz,char* implementation) {
  Manifold sponge = MengerSponge(4);
  sponge = sponge.Rotate(rx, ry, rz);
  if (!strcmp(implementation,"Hull"))
    PrintManifold(sponge, hull_func);
  else if (!strcmp(implementation,"CGAL"))
    PrintCGAL(sponge, hull_func);
}

// Constructs a high quality sphere, and tests the convex hull implementation on
// it (you can pass the specific hull implementation to be tested). Comparing
// the volume and surface area with CGAL implementation
void SphereTestHull(Manifold (Manifold::*hull_func)() const, char* implementation) {
  Manifold sphere = Manifold::Sphere(1, 6000);
  sphere = sphere.Translate(glm::vec3(0.5));
  if (!strcmp(implementation,"Hull"))
    PrintManifold(sphere, hull_func);
  else if (!strcmp(implementation,"CGAL"))
    PrintCGAL(sphere, hull_func);
}


int main(int argc, char **argv) {
  if (argc<4)
  {
    std::cout << "Usage: ./test_hull_performance <Test (Sphere/Menger/Input)> <Implementation (Hull/Hull_CGAL)> <Print Header (1/0)> <Input Mesh (filename)> " << std::endl;
    return 0;
  }
  if (!strcmp(argv[3],"1"))
    std::cout << "VolManifold,VolHull,AreaManifold,AreaHull,ManifoldTri,HullTri,Time" << std::endl;
  if (!strcmp(argv[1],"Sphere"))
    SphereTestHull(&Manifold::Hull,argv[2]);
  else if (!strcmp(argv[1],"Menger"))
    MengerTestHull(&Manifold::Hull, 1, 2, 3,argv[2]);
  else if (!strcmp(argv[1],"Input"))
  {
    auto inputMesh = ImportMesh(argv[4], 1);
    Manifold inputManifold = Manifold(inputMesh);
    if (!strcmp(argv[2],"Hull"))
      PrintManifold(inputManifold, &Manifold::Hull);
    else if (!strcmp(argv[2],"Hull_CGAL"))
      PrintCGAL(inputManifold, &Manifold::Hull);
  }
}
