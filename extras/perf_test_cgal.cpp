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
  // Manifold hullManifold2 = hullManifold.Hull2();
  PrintVolArea(input, cgalInput, hullMesh, hullManifold);
  // PrintVolArea(input, cgalInput, hullMesh, hullManifold2);
  // Manifold Fin21 = hullManifold2 - hullManifold;
  // Manifold Fin12 = hullManifold - hullManifold2;
  std::chrono::duration<double> elapsed = end - start;
  // ExportMesh("1750623.glb", Fin.GetMesh(), {});
  // ExportMesh("39202_hull2_hull1.glb", Fin21.GetMesh(), {});
  // ExportMesh("39202_hull1_hull2.glb", Fin12.GetMesh(), {});
  // ExportMesh("39202_hull1.glb", hullManifold.GetMesh(), {});
  // ExportMesh("39202_hull2.glb", hullManifold2.GetMesh(), {});
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

bool runHullAlgorithm(std::vector<glm::vec3>& pts) {
    
    std::ostringstream errBuffer;
    std::streambuf* originalCerr = std::cerr.rdbuf(errBuffer.rdbuf());
    
    bool hasError = false;

    try {
        Manifold output = Manifold::Hull(pts);
    } catch (...) {
        hasError = true;
    }
    
    // Restore the original std::cerr
    std::cerr.rdbuf(originalCerr);

    if (!errBuffer.str().empty()) {
        std::cerr << "Captured error: " << errBuffer.str() << std::endl;
        hasError = true;
    }

    return hasError;
}

void savePointsToFile( const std::vector<glm::vec3>& points,const string& filename) {
    ofstream outfile(filename);

    if (outfile.is_open()) {
        outfile << points.size() << endl;
        outfile << fixed << setprecision(10); // Set precision to 15 decimal places
        for (const glm::vec3& point : points) {
            outfile << point.x << " " << point.y << " " << point.z << endl;
        }

        outfile.close();
        cout << "Points saved to file: " << filename << endl;
    } else {
        cerr << "Error opening file for writing: " << filename << endl;
    }
}


void saveBinaryData(const std::vector<glm::vec3>& data, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }
    for (const glm::vec3& vec : data) {
        outFile.write(reinterpret_cast<const char*>(&vec), sizeof(glm::vec3));
    }
    outFile.close();
}

std::vector<glm::vec3> readBinaryData(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading!" << std::endl;
        return {};
    }
    std::vector<glm::vec3> data;
    glm::vec3 vec;
    while (inFile.read(reinterpret_cast<char*>(&vec), sizeof(glm::vec3))) {
        data.push_back(vec);
    }
    inFile.close();
    return data;
}

std::vector<glm::vec3> loadPointsFromFile(const string& filename) {
    ifstream infile(filename);
    vector<glm::vec3> points;

    if (infile.is_open()) {
        int numPoints;
        infile >> numPoints;

        for (int i = 0; i < numPoints; ++i) {
            string line;
            getline(infile, line); // Read entire line

            stringstream ss(line);
            float x, y, z;

            ss >> x >> y >> z;

            points.push_back(glm::vec3(x, y, z));
        }

        infile.close();
        cout << "Points loaded from file: " << filename << endl;
    } else {
        cerr << "Error opening file for reading: " << filename << endl;
    }

    return points;
}


std::vector<glm::vec3> getRandomSubset(const std::vector<glm::vec3>& pts, int size) {
    std::vector<glm::vec3> subset;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::vector<glm::vec3> shuffled_points = pts;
    std::shuffle(shuffled_points.begin(), shuffled_points.end(), gen);
    
    subset.assign(shuffled_points.begin(), shuffled_points.begin() + size);
    
    return subset;
}

std::vector<glm::vec3> narrowDownPoints(const std::vector<glm::vec3>& points, int initialSize) {
    std::vector <glm::vec3> prev_subset = points;
    std::vector<glm::vec3> subset = getRandomSubset(points, initialSize);
    
    int step = 0;
    while (subset.size() > 5) {
        
        if (runHullAlgorithm(subset)) {
            // If errors occur, narrow down further
            // savePointsToFile(subset, "points_step_" + std::to_string(step) + ".txt");
            saveBinaryData(subset, "points_step_" + std::to_string(step) + ".bin");
            // Manifold temp = Manifold::Hull(subset);
            // ExportMesh("Horizon_hull_" + std::to_string(step) + ".glb", temp.GetMesh(), {});
            std::cout << "Step " << step << ": " << subset.size() << " points\n";
            prev_subset = subset;
            subset = getRandomSubset(subset, subset.size() / 2); // Halve the subset size
        }         
        else
        {
          subset=getRandomSubset(prev_subset, prev_subset.size() / 2);
        }
        step++;
    }
    
    return subset;
}


int main(int argc, char **argv) {
  // perfTestCGAL();
  // SphereTestHull(Manifold::Hull);
  // MengerTestHull(Manifold::Hull, 1, 2, 3);
  
  // std::cout << argv[1] << std::endl;

  // Narrowing down points
  // auto inputMesh = ImportMesh(argv[1], 1);
  // std::vector<glm::vec3> problematicPoints = narrowDownPoints(inputMesh.vertPos, inputMesh.vertPos.size());

  // // Print the problematic points (if any)
  // std::cout << "Problematic points causing errors:\n";
  // for (const auto& p : problematicPoints) {
  //     std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ")\n";
  // }

  // Rendering the points

  std::vector <glm::vec3> narrowed_points = readBinaryData("points_step_12409164.bin");
  Manifold HorizonMesh = Manifold::Hull(narrowed_points);
  ExportMesh("Horizon_hull.glb", HorizonMesh.GetMesh(), {});
  // auto inputMesh = ImportMesh(argv[1], 1);
  // Manifold temp = Manifold::Hull(inputMesh.vertPos);
  // Manifold inputManifold = Manifold(inputMesh);
  // PrintManifold(inputManifold, &Manifold::Hull6);


  // RunThingi10K(&Manifold::Hull4);
}
