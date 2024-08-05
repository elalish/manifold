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
//
// Derived from the public domain work of Antti Kuukka at
// https://github.com/akuukka/quickhull

#include "quickhull.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

double defaultEps() { return 0.0000001; }

/*
 * Implementation of the algorithm
 */

ConvexHull QuickHull::getConvexHull(const std::vector<glm::dvec3>& pointCloud,
                                    bool CCW, bool useOriginalIndices,
                                    double epsilon) {
  VertexDataSource vertexDataSource(pointCloud);
  return getConvexHull(vertexDataSource, CCW, useOriginalIndices, epsilon);
}

ConvexHull QuickHull::getConvexHull(const glm::dvec3* vertexData,
                                    size_t vertexCount, bool CCW,
                                    bool useOriginalIndices, double epsilon) {
  VertexDataSource vertexDataSource(vertexData, vertexCount);
  return getConvexHull(vertexDataSource, CCW, useOriginalIndices, epsilon);
}

ConvexHull QuickHull::getConvexHull(const double* vertexData,
                                    size_t vertexCount, bool CCW,
                                    bool useOriginalIndices, double epsilon) {
  VertexDataSource vertexDataSource((const vec3*)vertexData, vertexCount);
  return getConvexHull(vertexDataSource, CCW, useOriginalIndices, epsilon);
}

HalfEdgeMesh QuickHull::getConvexHullAsMesh(const double* vertexData,
                                            size_t vertexCount, bool CCW,
                                            double epsilon) {
  VertexDataSource vertexDataSource((const vec3*)vertexData, vertexCount);
  buildMesh(vertexDataSource, CCW, false, epsilon);
  return HalfEdgeMesh(mesh, originalVertexData);
}

void QuickHull::buildMesh(const VertexDataSource& pointCloud, bool CCW,
                          bool useOriginalIndices, double epsilon) {
  // CCW is unused for now
  (void)CCW;
  // useOriginalIndices is unused for now
  (void)useOriginalIndices;

  if (pointCloud.size() == 0) {
    mesh = MeshBuilder();
    return;
  }
  originalVertexData = pointCloud;

  // Very first: find extreme values and use them to compute the scale of the
  // point cloud.
  extremeValues = getExtremeValues();
  scale = getScale(extremeValues);

  // Epsilon we use depends on the scale
  m_epsilon = epsilon * scale;
  epsilonSquared = m_epsilon * m_epsilon;

  // Reset diagnostics
  diagnostics = DiagnosticsData();
  // The planar case happens when all the points appear to lie on a two
  // dimensional subspace of R^3.
  planar = false;
  createConvexHalfEdgeMesh();
  if (planar) {
    const size_t extraPointIndex = planarPointCloudTemp.size() - 1;
    for (auto& he : mesh.halfEdges) {
      if (he.endVertex == extraPointIndex) {
        he.endVertex = 0;
      }
    }
    originalVertexData = pointCloud;
    planarPointCloudTemp.clear();
  }
}

ConvexHull QuickHull::getConvexHull(const VertexDataSource& pointCloud,
                                    bool CCW, bool useOriginalIndices,
                                    double epsilon) {
  buildMesh(pointCloud, CCW, useOriginalIndices, epsilon);
  return ConvexHull(mesh, originalVertexData, CCW, useOriginalIndices);
}

void QuickHull::createConvexHalfEdgeMesh() {
  visibleFaces.clear();
  horizonEdgesData.clear();
  possiblyVisibleFaces.clear();

  // Compute base tetrahedron
  setupInitialTetrahedron();
  assert(mesh.faces.size() == 4);

  // Init face stack with those faces that have points assigned to them
  faceList.clear();
  for (size_t i = 0; i < 4; i++) {
    auto& f = mesh.faces[i];
    if (f.pointsOnPositiveSide && f.pointsOnPositiveSide->size() > 0) {
      faceList.push_back(i);
      f.inFaceStack = 1;
    }
  }

  // Process faces until the face list is empty.
  size_t iter = 0;
  while (!faceList.empty()) {
    iter++;
    if (iter == std::numeric_limits<size_t>::max()) {
      // Visible face traversal marks visited faces with iteration counter (to
      // mark that the face has been visited on this iteration) and the max
      // value represents unvisited faces. At this point we have to reset
      // iteration counter. This shouldn't be an issue on 64 bit machines.
      iter = 0;
    }

    const size_t topFaceIndex = faceList.front();
    faceList.pop_front();

    auto& tf = mesh.faces[topFaceIndex];
    tf.inFaceStack = 0;

    assert(!tf.pointsOnPositiveSide || tf.pointsOnPositiveSide->size() > 0);
    if (!tf.pointsOnPositiveSide || tf.isDisabled()) {
      continue;
    }

    // Pick the most distant point to this triangle plane as the point to which
    // we extrude
    const vec3& activePoint = originalVertexData[tf.mostDistantPoint];
    const size_t activePointIndex = tf.mostDistantPoint;

    // Find out the faces that have our active point on their positive side
    // (these are the "visible faces"). The face on top of the stack of course
    // is one of them. At the same time, we create a list of horizon edges.
    horizonEdgesData.clear();
    possiblyVisibleFaces.clear();
    visibleFaces.clear();
    possiblyVisibleFaces.emplace_back(topFaceIndex,
                                      std::numeric_limits<size_t>::max());
    while (possiblyVisibleFaces.size()) {
      const auto faceData = possiblyVisibleFaces.back();
      possiblyVisibleFaces.pop_back();
      auto& pvf = mesh.faces[faceData.faceIndex];
      assert(!pvf.isDisabled());

      if (pvf.visibilityCheckedOnIteration == iter) {
        if (pvf.isVisibleFaceOnCurrentIteration) {
          continue;
        }
      } else {
        const Plane& P = pvf.P;
        pvf.visibilityCheckedOnIteration = iter;
        const double d = glm::dot(P.N, activePoint) + P.D;
        if (d > 0) {
          pvf.isVisibleFaceOnCurrentIteration = 1;
          pvf.horizonEdgesOnCurrentIteration = 0;
          visibleFaces.push_back(faceData.faceIndex);
          for (auto heIndex : mesh.getHalfEdgeIndicesOfFace(pvf)) {
            if (mesh.halfEdges[heIndex].opp != faceData.enteredFromHalfEdge) {
              possiblyVisibleFaces.emplace_back(
                  mesh.halfEdges[mesh.halfEdges[heIndex].opp].face, heIndex);
            }
          }
          continue;
        }
        assert(faceData.faceIndex != topFaceIndex);
      }

      // The face is not visible. Therefore, the halfedge we came from is part
      // of the horizon edge.
      pvf.isVisibleFaceOnCurrentIteration = 0;
      horizonEdgesData.push_back(faceData.enteredFromHalfEdge);
      // Store which half edge is the horizon edge. The other half edges of the
      // face will not be part of the final mesh so their data slots can by
      // recycled.
      const auto halfEdgesMesh = mesh.getHalfEdgeIndicesOfFace(
          mesh.faces[mesh.halfEdges[faceData.enteredFromHalfEdge].face]);
      const std::int8_t ind =
          (halfEdgesMesh[0] == faceData.enteredFromHalfEdge)
              ? 0
              : (halfEdgesMesh[1] == faceData.enteredFromHalfEdge ? 1 : 2);
      mesh.faces[mesh.halfEdges[faceData.enteredFromHalfEdge].face]
          .horizonEdgesOnCurrentIteration |= (1 << ind);
    }
    const size_t horizonEdgeCount = horizonEdgesData.size();

    // Order horizon edges so that they form a loop. This may fail due to
    // numerical instability in which case we give up trying to solve horizon
    // edge for this point and accept a minor degeneration in the convex hull.
    if (!reorderHorizonEdges(horizonEdgesData)) {
      diagnostics.failedHorizonEdges++;
      std::cerr << "Failed to solve horizon edge." << std::endl;
      auto it = std::find(tf.pointsOnPositiveSide->begin(),
                          tf.pointsOnPositiveSide->end(), activePointIndex);
      tf.pointsOnPositiveSide->erase(it);
      if (tf.pointsOnPositiveSide->size() == 0) {
        reclaimToIndexVectorPool(tf.pointsOnPositiveSide);
      }
      continue;
    }

    // Except for the horizon edges, all half edges of the visible faces can be
    // marked as disabled. Their data slots will be reused. The faces will be
    // disabled as well, but we need to remember the points that were on the
    // positive side of them - therefore we save pointers to them.
    newFaceIndices.clear();
    newHalfEdgeIndices.clear();
    disabledFacePointVectors.clear();
    size_t disableCounter = 0;
    for (auto faceIndex : visibleFaces) {
      auto& disabledFace = mesh.faces[faceIndex];
      auto halfEdgesMesh = mesh.getHalfEdgeIndicesOfFace(disabledFace);
      for (size_t j = 0; j < 3; j++) {
        if ((disabledFace.horizonEdgesOnCurrentIteration & (1 << j)) == 0) {
          if (disableCounter < horizonEdgeCount * 2) {
            // Use on this iteration
            newHalfEdgeIndices.push_back(halfEdgesMesh[j]);
            disableCounter++;
          } else {
            // Mark for reusal on later iteration step
            mesh.disableHalfEdge(halfEdgesMesh[j]);
          }
        }
      }
      // Disable the face, but retain pointer to the points that were on the
      // positive side of it. We need to assign those points to the new faces we
      // create shortly.
      auto t = mesh.disableFace(faceIndex);
      if (t) {
        // Because we should not assign point vectors to faces unless needed...
        assert(t->size());
        disabledFacePointVectors.push_back(std::move(t));
      }
    }
    if (disableCounter < horizonEdgeCount * 2) {
      const size_t newHalfEdgesNeeded = horizonEdgeCount * 2 - disableCounter;
      for (size_t i = 0; i < newHalfEdgesNeeded; i++) {
        newHalfEdgeIndices.push_back(mesh.addHalfEdge());
      }
    }

    // Create new faces using the edgeloop
    for (size_t i = 0; i < horizonEdgeCount; i++) {
      const size_t AB = horizonEdgesData[i];

      auto horizonEdgeVertexIndices =
          mesh.getVertexIndicesOfHalfEdge(mesh.halfEdges[AB]);
      size_t A, B, C;
      A = horizonEdgeVertexIndices[0];
      B = horizonEdgeVertexIndices[1];
      C = activePointIndex;

      const size_t newFaceIndex = mesh.addFace();
      newFaceIndices.push_back(newFaceIndex);

      const size_t CA = newHalfEdgeIndices[2 * i + 0];
      const size_t BC = newHalfEdgeIndices[2 * i + 1];

      mesh.halfEdges[AB].next = BC;
      mesh.halfEdges[BC].next = CA;
      mesh.halfEdges[CA].next = AB;

      mesh.halfEdges[BC].face = newFaceIndex;
      mesh.halfEdges[CA].face = newFaceIndex;
      mesh.halfEdges[AB].face = newFaceIndex;

      mesh.halfEdges[CA].endVertex = A;
      mesh.halfEdges[BC].endVertex = C;

      auto& newFace = mesh.faces[newFaceIndex];

      const glm::dvec3 planeNormal = mathutils::getTriangleNormal(
          originalVertexData[A], originalVertexData[B], activePoint);
      newFace.P = Plane(planeNormal, activePoint);
      newFace.he = AB;

      mesh.halfEdges[CA].opp =
          newHalfEdgeIndices[i > 0 ? i * 2 - 1 : 2 * horizonEdgeCount - 1];
      mesh.halfEdges[BC].opp =
          newHalfEdgeIndices[((i + 1) * 2) % (horizonEdgeCount * 2)];
    }

    // Assign points that were on the positive side of the disabled faces to the
    // new faces.
    for (auto& disabledPoints : disabledFacePointVectors) {
      assert(disabledPoints);
      for (const auto& point : *(disabledPoints)) {
        if (point == activePointIndex) {
          continue;
        }
        for (size_t j = 0; j < horizonEdgeCount; j++) {
          if (addPointToFace(mesh.faces[newFaceIndices[j]], point)) {
            break;
          }
        }
      }
      // The points are no longer needed: we can move them to the vector pool
      // for reuse.
      reclaimToIndexVectorPool(disabledPoints);
    }

    // Increase face stack size if needed
    for (const auto newFaceIndex : newFaceIndices) {
      auto& newFace = mesh.faces[newFaceIndex];
      if (newFace.pointsOnPositiveSide) {
        assert(newFace.pointsOnPositiveSide->size() > 0);
        if (!newFace.inFaceStack) {
          faceList.push_back(newFaceIndex);
          newFace.inFaceStack = 1;
        }
      }
    }
  }

  // Cleanup
  indexVectorPool.clear();
}

/*
 * Private helper functions
 */

std::array<size_t, 6> QuickHull::getExtremeValues() {
  std::array<size_t, 6> outIndices{0, 0, 0, 0, 0, 0};
  double extremeVals[6] = {originalVertexData[0].x, originalVertexData[0].x,
                           originalVertexData[0].y, originalVertexData[0].y,
                           originalVertexData[0].z, originalVertexData[0].z};
  const size_t vCount = originalVertexData.size();
  for (size_t i = 1; i < vCount; i++) {
    const glm::dvec3& pos = originalVertexData[i];
    if (pos.x > extremeVals[0]) {
      extremeVals[0] = pos.x;
      outIndices[0] = i;
    } else if (pos.x < extremeVals[1]) {
      extremeVals[1] = pos.x;
      outIndices[1] = i;
    }
    if (pos.y > extremeVals[2]) {
      extremeVals[2] = pos.y;
      outIndices[2] = i;
    } else if (pos.y < extremeVals[3]) {
      extremeVals[3] = pos.y;
      outIndices[3] = i;
    }
    if (pos.z > extremeVals[4]) {
      extremeVals[4] = pos.z;
      outIndices[4] = i;
    } else if (pos.z < extremeVals[5]) {
      extremeVals[5] = pos.z;
      outIndices[5] = i;
    }
  }
  return outIndices;
}

bool QuickHull::reorderHorizonEdges(std::vector<size_t>& horizonEdges) {
  const size_t horizonEdgeCount = horizonEdges.size();
  for (size_t i = 0; i < horizonEdgeCount - 1; i++) {
    const size_t endVertexCheck = mesh.halfEdges[horizonEdges[i]].endVertex;
    bool foundNext = false;
    for (size_t j = i + 1; j < horizonEdgeCount; j++) {
      const size_t beginVertex =
          mesh.halfEdges[mesh.halfEdges[horizonEdges[j]].opp].endVertex;
      if (beginVertex == endVertexCheck) {
        std::swap(horizonEdges[i + 1], horizonEdges[j]);
        foundNext = true;
        break;
      }
    }
    if (!foundNext) {
      return false;
    }
  }
  assert(mesh.halfEdges[horizonEdges[horizonEdges.size() - 1]].endVertex ==
         mesh.halfEdges[mesh.halfEdges[horizonEdges[0]].opp].endVertex);
  return true;
}

double QuickHull::getScale(const std::array<size_t, 6>& extremeValuesInput) {
  double s = 0;
  for (size_t i = 0; i < 6; i++) {
    const double* v =
        (const double*)(&originalVertexData[extremeValuesInput[i]]);
    v += i / 2;
    auto a = std::abs(*v);
    if (a > s) {
      s = a;
    }
  }
  return s;
}

void QuickHull::setupInitialTetrahedron() {
  const size_t vertexCount = originalVertexData.size();

  // If we have at most 4 points, just return a degenerate tetrahedron:
  if (vertexCount <= 4) {
    size_t v[4] = {0, std::min((size_t)1, vertexCount - 1),
                   std::min((size_t)2, vertexCount - 1),
                   std::min((size_t)3, vertexCount - 1)};
    const glm::dvec3 N = mathutils::getTriangleNormal(originalVertexData[v[0]],
                                                      originalVertexData[v[1]],
                                                      originalVertexData[v[2]]);
    const Plane trianglePlane(N, originalVertexData[v[0]]);
    if (trianglePlane.isPointOnPositiveSide(originalVertexData[v[3]])) {
      std::swap(v[0], v[1]);
    }
    return mesh.setup(v[0], v[1], v[2], v[3]);
  }

  // Find two most distant extreme points.
  double maxD = epsilonSquared;
  std::pair<size_t, size_t> selectedPoints;
  for (size_t i = 0; i < 6; i++) {
    for (size_t j = i + 1; j < 6; j++) {
      // I found a function for squaredDistance but i can't seem to include it
      // like this for some reason
      const double d =
          mathutils::getSquaredDistance(originalVertexData[extremeValues[i]],
                                        originalVertexData[extremeValues[j]]);
      if (d > maxD) {
        maxD = d;
        selectedPoints = {extremeValues[i], extremeValues[j]};
      }
    }
  }
  if (maxD == epsilonSquared) {
    // A degenerate case: the point cloud seems to consists of a single point
    return mesh.setup(0, std::min((size_t)1, vertexCount - 1),
                      std::min((size_t)2, vertexCount - 1),
                      std::min((size_t)3, vertexCount - 1));
  }
  assert(selectedPoints.first != selectedPoints.second);

  // Find the most distant point to the line between the two chosen extreme
  // points.
  const Ray r(originalVertexData[selectedPoints.first],
              (originalVertexData[selectedPoints.second] -
               originalVertexData[selectedPoints.first]));
  maxD = epsilonSquared;
  size_t maxI = std::numeric_limits<size_t>::max();
  const size_t vCount = originalVertexData.size();
  for (size_t i = 0; i < vCount; i++) {
    const double distToRay = mathutils::getSquaredDistanceBetweenPointAndRay(
        originalVertexData[i], r);
    if (distToRay > maxD) {
      maxD = distToRay;
      maxI = i;
    }
  }
  if (maxD == epsilonSquared) {
    // It appears that the point cloud belongs to a 1 dimensional subspace of
    // R^3: convex hull has no volume => return a thin triangle Pick any point
    // other than selectedPoints.first and selectedPoints.second as the third
    // point of the triangle
    auto it =
        std::find_if(originalVertexData.begin(), originalVertexData.end(),
                     [&](const vec3& ve) {
                       return ve != originalVertexData[selectedPoints.first] &&
                              ve != originalVertexData[selectedPoints.second];
                     });
    const size_t thirdPoint =
        (it == originalVertexData.end())
            ? selectedPoints.first
            : std::distance(originalVertexData.begin(), it);
    it =
        std::find_if(originalVertexData.begin(), originalVertexData.end(),
                     [&](const vec3& ve) {
                       return ve != originalVertexData[selectedPoints.first] &&
                              ve != originalVertexData[selectedPoints.second] &&
                              ve != originalVertexData[thirdPoint];
                     });
    const size_t fourthPoint =
        (it == originalVertexData.end())
            ? selectedPoints.first
            : std::distance(originalVertexData.begin(), it);
    return mesh.setup(selectedPoints.first, selectedPoints.second, thirdPoint,
                      fourthPoint);
  }

  // These three points form the base triangle for our tetrahedron.
  assert(selectedPoints.first != maxI && selectedPoints.second != maxI);
  std::array<size_t, 3> baseTriangle{selectedPoints.first,
                                     selectedPoints.second, maxI};
  const glm::dvec3 baseTriangleVertices[] = {
      originalVertexData[baseTriangle[0]], originalVertexData[baseTriangle[1]],
      originalVertexData[baseTriangle[2]]};

  // Next step is to find the 4th vertex of the tetrahedron. We naturally choose
  // the point farthest away from the triangle plane.
  maxD = m_epsilon;
  maxI = 0;
  const glm::dvec3 N = mathutils::getTriangleNormal(baseTriangleVertices[0],
                                                    baseTriangleVertices[1],
                                                    baseTriangleVertices[2]);
  Plane trianglePlane(N, baseTriangleVertices[0]);
  for (size_t i = 0; i < vCount; i++) {
    const double d = std::abs(mathutils::getSignedDistanceToPlane(
        originalVertexData[i], trianglePlane));
    if (d > maxD) {
      maxD = d;
      maxI = i;
    }
  }
  if (maxD == m_epsilon) {
    // All the points seem to lie on a 2D subspace of R^3. How to handle this?
    // Well, let's add one extra point to the point cloud so that the convex
    // hull will have volume.
    planar = true;
    const vec3 N1 = mathutils::getTriangleNormal(baseTriangleVertices[1],
                                                 baseTriangleVertices[2],
                                                 baseTriangleVertices[0]);
    planarPointCloudTemp.clear();
    planarPointCloudTemp.insert(planarPointCloudTemp.begin(),
                                originalVertexData.begin(),
                                originalVertexData.end());
    const vec3 extraPoint = N1 + originalVertexData[0];
    planarPointCloudTemp.push_back(extraPoint);
    maxI = planarPointCloudTemp.size() - 1;
    originalVertexData = VertexDataSource(planarPointCloudTemp);
  }

  // Enforce CCW orientation (if user prefers clockwise orientation, swap two
  // vertices in each triangle when final mesh is created)
  const Plane triPlane(N, baseTriangleVertices[0]);
  if (triPlane.isPointOnPositiveSide(originalVertexData[maxI])) {
    std::swap(baseTriangle[0], baseTriangle[1]);
  }

  // Create a tetrahedron half edge mesh and compute planes defined by each
  // triangle
  mesh.setup(baseTriangle[0], baseTriangle[1], baseTriangle[2], maxI);
  for (auto& f : mesh.faces) {
    auto v = mesh.getVertexIndicesOfFace(f);
    const glm::dvec3& va = originalVertexData[v[0]];
    const glm::dvec3& vb = originalVertexData[v[1]];
    const glm::dvec3& vc = originalVertexData[v[2]];
    const glm::dvec3 N1 = mathutils::getTriangleNormal(va, vb, vc);
    const Plane plane(N1, va);
    f.P = plane;
  }

  // Finally we assign a face for each vertex outside the tetrahedron (vertices
  // inside the tetrahedron have no role anymore)
  for (size_t i = 0; i < vCount; i++) {
    for (auto& face : mesh.faces) {
      if (addPointToFace(face, i)) {
        break;
      }
    }
  }
}
