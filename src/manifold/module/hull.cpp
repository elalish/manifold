// Copyright 2024 The Manifold Authors.
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

#include "QuickHull.hpp"
#include "manifold.h"
#include "par.h"

using namespace manifold;

/**
 * Compute the convex hull of a set of points. If the given points are fewer
 * than 4, or they are all coplanar, an empty Manifold will be returned.
 *
 * @param pts A vector of 3-dimensional points over which to compute a convex
 * hull.
 */
Manifold Manifold::Hull(const std::vector<glm::vec3>& pts) {
  const int numVert = pts.size();
  if (numVert < 4) return Manifold();

  std::vector<quickhull::Vector3<double>> vertices(numVert);
  for (int i = 0; i < numVert; i++) {
    vertices[i] = {pts[i].x, pts[i].y, pts[i].z};
  }

  quickhull::QuickHull<double> qh;
  // bools: correct triangle winding, and use original indices
  auto hull = qh.getConvexHull(vertices, false, true);
  const auto& triangles = hull.getIndexBuffer();
  const int numTris = triangles.size() / 3;

  Mesh mesh;
  mesh.vertPos = pts;
  mesh.triVerts.reserve(numTris);
  for (int i = 0; i < numTris; i++) {
    const int j = i * 3;
    mesh.triVerts.push_back({triangles[j], triangles[j + 1], triangles[j + 2]});
  }
  return Manifold(mesh);
}

/**
 * Compute the convex hull of this manifold.
 */
Manifold Manifold::Hull() const { return Hull(GetMesh().vertPos); }

/**
 * Compute the convex hull enveloping a set of manifolds.
 *
 * @param manifolds A vector of manifolds over which to compute a convex hull.
 */
Manifold Manifold::Hull(const std::vector<Manifold>& manifolds) {
  return Compose(manifolds).Hull();
}