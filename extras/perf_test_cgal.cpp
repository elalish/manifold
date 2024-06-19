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

#include "manifold.h"
#include "meshIO.h"
#include "samples.h"

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

void computeCGALConvexHull(const std::vector<Point> &points,
                           TriangleMesh &hullMesh) {
  CGAL::convex_hull_3(points.begin(), points.end(), hullMesh);
}

// Performance test for CGAL (original main function)
void perfTestCGAL() {
  for (int i = 0; i < 8; ++i) {
    Manifold sphere = Manifold::Sphere(1, (8 << i) * 4);
    Manifold sphere2 = sphere.Translate(glm::vec3(0.5));

    TriangleMesh cgalSphere, cgalSphere2;
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

// Prints the volume and area of the manifold, the convex hull of the manifold,
// and the CGAL implementations.
void PrintVolArea(Manifold &manfiold_obj, TriangleMesh &manifoldMesh,
                  TriangleMesh &hullMesh, Manifold &hullManifold) {
  // std::cout << CGAL::Polygon_mesh_processing::volume(manifoldMesh) << ",";
  // std::cout << CGAL::Polygon_mesh_processing::volume(hullMesh) << ",";
  // std::cout << CGAL::Polygon_mesh_processing::area(manifoldMesh) << ",";
  // std::cout << CGAL::Polygon_mesh_processing::area(hullMesh) << ",";
  // std::cout << std::distance(manifoldMesh.faces_begin(),manifoldMesh.faces_end())
  //           << ",";
  // std::cout << std::distance(hullMesh.faces_begin(), hullMesh.faces_end())
  //           << ",";
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
}

void PrintManifold(Manifold input, Manifold (Manifold::*hull_func)() const) {
  TriangleMesh cgalInput;
  // manifoldToCGALSurfaceMesh(input, cgalInput);
  // std::vector<Point> points;
  // for (const auto &vert : cgalInput.vertices()) {
  //   points.push_back(cgalInput.point(vert));
  // }

  // Convex Hull
  TriangleMesh hullMesh;
  auto start = std::chrono::high_resolution_clock::now();
  // computeCGALConvexHull(points, hullMesh);
  Manifold hullManifold = (input.*hull_func)();
  auto end = std::chrono::high_resolution_clock::now();
  // Manifold hullManifold;
  PrintVolArea(input, cgalInput, hullMesh, hullManifold);
  std::chrono::duration<double> elapsed = end - start;
  std::cout << elapsed.count() << " sec" ;
}

// Constructs a Menger Sponge, and tests the convex hull implementation on it
// (you can pass the specific hull implementation to be tested). Comparing the
// volume and surface area with CGAL implementation, for various values of
// rotation
void MengerTestHull(Manifold (Manifold::*hull_func)() const, float rx, float ry,
                    float rz) {
  Manifold sponge = MengerSponge(4);
  sponge = sponge.Rotate(rx, ry, rz);
  PrintManifold(sponge, hull_func);
}

// Constructs a high quality sphere, and tests the convex hull implementation on
// it (you can pass the specific hull implementation to be tested). Comparing
// the volume and surface area with CGAL implementation
void SphereTestHull(Manifold (Manifold::*hull_func)() const) {
  Manifold sphere = Manifold::Sphere(1, 6000);
  sphere = sphere.Translate(glm::vec3(0.5));
  PrintManifold(sphere, hull_func);
}

void RunThingi10K(Manifold (Manifold::*hull_func)() const) {
  std::string folderPath = "../extras/Thingi10K/raw_meshes";
  std::string logFilePath = "../extras/output_log.txt";

  // Create an ofstream object to write to a temporary file
  std::ofstream logFile(logFilePath);
  if (!logFile) {
    std::cerr << "Error opening log file: " << logFilePath << std::endl;
    return;
  }

  // Redirect std::cout to the log file
  std::streambuf *originalCoutBuffer = std::cout.rdbuf();
  std::streambuf *originalCerrBuffer = std::cerr.rdbuf();
  std::cout.rdbuf(logFile.rdbuf());
  std::cerr.rdbuf(logFile.rdbuf());
  // Iterate through the directory
  for (const auto &entry : std::filesystem::directory_iterator(folderPath)) {
    if (entry.is_regular_file() && (entry.path().filename() == "74463.stl" ||
                                    entry.path().filename() == "286163.stl" ||
                                    entry.path().filename() == "49911.stl" ||
                                    entry.path().filename() == "81313.obj" ||
                                    entry.path().filename() == "77942.stl")) {
    } else if (entry.is_regular_file()) {
      std::cout << entry.path().filename() << std::endl;
      auto inputMesh = ImportMesh(entry.path(), 1);
      Manifold inputManifold = Manifold(inputMesh);
      PrintManifold(inputManifold, hull_func);
    }
  }
  std::cout.rdbuf(originalCoutBuffer);
  std::cerr.rdbuf(originalCerrBuffer);

  // Close the log file
  logFile.close();
}

int main(int argc, char **argv) {
  // perfTestCGAL();
  // SphereTestHull(Manifold::Hull);
  // MengerTestHull(Manifold::Hull, 1, 2, 3);
  
  // std::cout << argv[1] << std::endl;
  auto inputMesh = ImportMesh(argv[1], 1);
  Manifold inputManifold = Manifold(inputMesh);
  PrintManifold(inputManifold, &Manifold::Hull3);
  // RunThingi10K(&Manifold::Hull4);
}
