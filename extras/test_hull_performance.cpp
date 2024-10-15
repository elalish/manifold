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

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>  // For string manipulation

#include "manifold/manifold.h"
#include "manifold/meshIO.h"
#include "samples.h"
using namespace std;
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

class HullImpl {
 public:
  // actual hull operation, we will measure the time needed to evaluate this
  // function.
  virtual void hull(const manifold::Manifold &input,
                    const std::vector<vec3> &pts) = 0;
  virtual ~HullImpl() = default;

#ifdef MANIFOLD_DEBUG
  // Check if the mesh remains convex after adding new faces
  bool isMeshConvex() {
    // Get the mesh from the manifold
    manifold::Mesh mesh = hullManifold.GetMesh();

    const auto &vertPos = mesh.vertPos;

    // Iterate over each triangle
    for (const auto &tri : mesh.triVerts) {
      // Get the vertices of the triangle
      vec3 v0 = vertPos[tri[0]];
      vec3 v1 = vertPos[tri[1]];
      vec3 v2 = vertPos[tri[2]];

      // Compute the normal of the triangle
      vec3 normal = la::normalize(la::cross(v1 - v0, v2 - v0));

      // Check all other vertices
      for (int i = 0; i < (int)vertPos.size(); ++i) {
        if (i == tri[0] || i == tri[2] || i == tri[3])
          continue;  // Skip vertices of the current triangle

        // Get the vertex
        vec3 v = vertPos[i];

        // Compute the signed distance from the plane
        double distance = la::dot(normal, v - v0);

        // If any vertex lies on the opposite side of the normal direction
        if (distance > 0) {
          // The manifold is not convex
          return false;
        }
      }
    }
    // If we didn't find any vertex on the opposite side for any triangle, it's
    // convex
    return true;
  }
#endif

 protected:
  manifold::Manifold hullManifold;
  manifold::Manifold inputManifold;
};

class HullImplOriginal : public HullImpl {
 public:
  void hull(const manifold::Manifold &input,
            const std::vector<vec3> &pts) override {
    inputManifold = input;
    auto start = std::chrono::high_resolution_clock::now();
    hullManifold = Manifold::Hull(pts);
    auto end = std::chrono::high_resolution_clock::now();
    PrintVolArea();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << " sec";
  }

 private:
  // Prints the volume and area of the manifold and the convex hull of our
  // implementation,
  void PrintVolArea() {
    std::cout << inputManifold.GetProperties().volume << ",";
    std::cout << hullManifold.GetProperties().volume << ",";
    std::cout << inputManifold.GetProperties().surfaceArea << ",";
    std::cout << hullManifold.GetProperties().surfaceArea << ",";
    std::cout << inputManifold.NumTri() << ",";
    std::cout << hullManifold.NumTri() << ",";
    return;
  }
};

class HullImplCGAL : public HullImpl {
 private:
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
  // Converts CGAL Surface Mesh to Manifold
  void CGALToManifoldSurfaceMesh(TriangleMesh &cgalMesh, Manifold &manifold) {
    Mesh mesh;
    mesh.triVerts.reserve(faces(cgalMesh).size() * 3);
    for (auto v : vertices(cgalMesh)) {
      auto pt = cgalMesh.point(v);
      mesh.vertPos.push_back({pt.x(), pt.y(), pt.z()});
    }
    for (auto f : faces(cgalMesh)) {
      vector<CGAL::SM_Index<CGAL::SM_Vertex_index>::size_type> verts;
      for (auto v : vertices_around_face(cgalMesh.halfedge(f), cgalMesh)) {
        verts.push_back(v.idx());
      }
      mesh.triVerts.push_back({verts[0], verts[1], verts[2]});
    }
    manifold = Manifold(mesh);
  }
  // Prints the volume and area of the manifold and Convex Hull of CGAL
  // Implementation
  void PrintVolAreaCGAL() {
    std::cout << CGAL::Polygon_mesh_processing::volume(cgalInput) << ",";
    std::cout << CGAL::Polygon_mesh_processing::volume(cgalHull) << ",";
    std::cout << CGAL::Polygon_mesh_processing::area(cgalInput) << ",";
    std::cout << CGAL::Polygon_mesh_processing::area(cgalHull) << ",";
    std::cout << std::distance(cgalInput.faces_begin(), cgalInput.faces_end())
              << ",";
    std::cout << std::distance(cgalHull.faces_begin(), cgalHull.faces_end())
              << ",";
    return;
  }
  TriangleMesh cgalHull, cgalInput;

 public:
  // This was the code  where I converted the CGAL Hull back to manifold, but I
  // thought that added overhead, so if needed we can also use the functions of
  // the library to print the output as well void hull(const
  // std::vector<vec3>& pts) {
  //   std::vector<Point> points;
  //   for (const auto &vert : pts) {
  //     points.push_back(Point(vert.x, vert.y, vert.z));
  //   }
  //   // Convex Hull
  //   CGAL::convex_hull_3(points.begin(), points.end(), cgalHull);
  //   CGALToManifoldSurfaceMesh(cgalHull, hullManifold);
  // }
  void hull(const manifold::Manifold &input,
            const std::vector<vec3> &pts) override {
    inputManifold = input;
    manifoldToCGALSurfaceMesh(inputManifold, cgalInput);
    std::vector<Point> points;
    for (const auto &vert : cgalInput.vertices()) {
      points.push_back(cgalInput.point(vert));
    }
    // Convex Hull
    auto start = std::chrono::high_resolution_clock::now();
    CGAL::convex_hull_3(points.begin(), points.end(), cgalHull);
    auto end = std::chrono::high_resolution_clock::now();
    PrintVolAreaCGAL();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << " sec";
  }
};

// Constructs a Menger Sponge, and tests the convex hull implementation on it
// (you can pass the specific hull implementation to be tested). Comparing the
// volume and surface area with CGAL implementation, for various values of
// rotation
void MengerTestHull(HullImpl *impl, double rx, double ry, double rz,
                    char *implementation) {
  if (impl == NULL) return;
  Manifold sponge = MengerSponge(4);
  sponge = sponge.Rotate(rx, ry, rz);
  impl->hull(sponge, sponge.GetMesh().vertPos);
}

// Constructs a high quality sphere, and tests the convex hull implementation on
// it (you can pass the specific hull implementation to be tested). Comparing
// the volume and surface area with CGAL implementation
void SphereTestHull(HullImpl *impl, char *implementation) {
  if (impl == NULL) return;
  Manifold sphere = Manifold::Sphere(1, 6000);
  sphere = sphere.Translate(vec3(0.5));
  impl->hull(sphere, sphere.GetMesh().vertPos);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "Usage: ./test_hull_performance <Test (Sphere/Menger/Input)> "
                 "<Implementation (Hull/Hull_CGAL)> <Print Header (1/0)> "
                 "<Input Mesh (filename)> "
              << std::endl;
    return 0;
  }
  if (!strcmp(argv[3], "1"))
    std::cout
        << "VolManifold,VolHull,AreaManifold,AreaHull,ManifoldTri,HullTri,Time"
        << std::endl;

  HullImpl *hullImpl = NULL;
  if (!strcmp(argv[2], "Hull"))
    hullImpl = new HullImplOriginal();
  else if (!strcmp(argv[2], "Hull_CGAL"))
    hullImpl = new HullImplCGAL();
  else {
    std::cout << "Invalid Implementation";
    return 0;
  }
  if (!strcmp(argv[1], "Sphere"))
    SphereTestHull(hullImpl, argv[2]);
  else if (!strcmp(argv[1], "Menger"))
    MengerTestHull(hullImpl, 1, 2, 3, argv[2]);
  else if (!strcmp(argv[1], "Input")) {
    auto inputMesh = ImportMesh(argv[4], 1);
    Manifold inputManifold = Manifold(inputMesh);
    hullImpl->hull(inputManifold, inputManifold.GetMesh().vertPos);
#ifdef MANIFOLD_DEBUG
    if (!hullImpl->isMeshConvex()) cout << "INVALID HULL" << endl;
#endif
  }
}
