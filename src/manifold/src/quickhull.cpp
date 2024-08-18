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
#include <limits>

namespace manifold {

double defaultEps() { return 0.0000001; }

// MathUtils.hpp
inline double getSquaredDistanceBetweenPointAndRay(const glm::dvec3& p,
                                                   const Ray& r) {
  const glm::dvec3 s = p - r.S;
  double t = glm::dot(s, r.V);
  return glm::dot(s, s) - t * t * r.VInvLengthSquared;
}

inline double getSquaredDistance(const glm::dvec3& p1, const glm::dvec3& p2) {
  return glm::dot(p1 - p2, p1 - p2);
}
// Note that the unit of distance returned is relative to plane's normal's
// length (divide by N.getNormalized() if needed to get the "real" distance).
inline double getSignedDistanceToPlane(const glm::dvec3& v, const Plane& p) {
  return glm::dot(p.N, v) + p.D;
}

inline glm::dvec3 getTriangleNormal(const glm::dvec3& a, const glm::dvec3& b,
                                    const glm::dvec3& c) {
  // We want to get (a-c).crossProduct(b-c) without constructing temp vectors
  double x = a.x - c.x;
  double y = a.y - c.y;
  double z = a.z - c.z;
  double rhsx = b.x - c.x;
  double rhsy = b.y - c.y;
  double rhsz = b.z - c.z;
  double px = y * rhsz - z * rhsy;
  double py = z * rhsx - x * rhsz;
  double pz = x * rhsy - y * rhsx;
  return glm::normalize(glm::dvec3(px, py, pz));
}

size_t MeshBuilder::addFace() {
  if (disabledFaces.size()) {
    size_t index = disabledFaces.back();
    auto& f = faces[index];
    assert(f.isDisabled());
    assert(!f.pointsOnPositiveSide);
    f.mostDistantPointDist = 0;
    disabledFaces.pop_back();
    return index;
  }
  faces.emplace_back();
  return faces.size() - 1;
}

size_t MeshBuilder::addHalfEdge() {
  if (disabledHalfEdges.size()) {
    const size_t index = disabledHalfEdges.back();
    disabledHalfEdges.pop_back();
    return index;
  }
  halfEdges.emplace_back();
  return halfEdges.size() - 1;
}

void MeshBuilder::setup(size_t a, size_t b, size_t c, size_t d) {
  faces.clear();
  halfEdges.clear();
  disabledFaces.clear();
  disabledHalfEdges.clear();

  faces.reserve(4);
  halfEdges.reserve(12);

  // Create halfedges
  // AB
  halfEdges.emplace_back(b, 6, 0, 1);
  // BC
  halfEdges.emplace_back(c, 9, 0, 2);
  // CA
  halfEdges.emplace_back(a, 3, 0, 0);
  // AC
  halfEdges.emplace_back(c, 2, 1, 4);
  // CD
  halfEdges.emplace_back(d, 11, 1, 5);
  // DA
  halfEdges.emplace_back(a, 7, 1, 3);
  // BA
  halfEdges.emplace_back(a, 0, 2, 7);
  // AD
  halfEdges.emplace_back(d, 5, 2, 8);
  // DB
  halfEdges.emplace_back(b, 10, 2, 6);
  // CB
  halfEdges.emplace_back(b, 1, 3, 10);
  // BD
  halfEdges.emplace_back(d, 8, 3, 11);
  // DC
  halfEdges.emplace_back(c, 4, 3, 9);

  // Create faces
  faces.emplace_back(0);
  faces.emplace_back(3);
  faces.emplace_back(6);
  faces.emplace_back(9);
}

std::array<size_t, 3> MeshBuilder::getVertexIndicesOfFace(const Face& f) const {
  std::array<size_t, 3> v;
  const HalfEdge* he = &halfEdges[f.he];
  v[0] = he->endVertex;
  he = &halfEdges[he->next];
  v[1] = he->endVertex;
  he = &halfEdges[he->next];
  v[2] = he->endVertex;
  return v;
}

HalfEdgeMesh::HalfEdgeMesh(const MeshBuilder& builderObject,
                           const VecView<glm::dvec3>& vertexData) {
  std::unordered_map<size_t, size_t> faceMapping;
  std::unordered_map<size_t, size_t> halfEdgeMapping;
  std::unordered_map<size_t, size_t> vertexMapping;

  size_t i = 0;
  for (const auto& face : builderObject.faces) {
    if (!face.isDisabled()) {
      faces.emplace_back(static_cast<size_t>(face.he));
      faceMapping[i] = faces.size() - 1;

      const auto heIndices = builderObject.getHalfEdgeIndicesOfFace(face);
      for (const auto heIndex : heIndices) {
        const size_t vertexIndex = builderObject.halfEdges[heIndex].endVertex;
        if (vertexMapping.count(vertexIndex) == 0) {
          vertices.push_back(vertexData[vertexIndex]);
          vertexMapping[vertexIndex] = vertices.size() - 1;
        }
      }
    }
    i++;
  }

  i = 0;
  for (const auto& halfEdge : builderObject.halfEdges) {
    if (!halfEdge.isDisabled()) {
      halfEdges.emplace_back(halfEdge.endVertex, halfEdge.opp, halfEdge.face,
                             halfEdge.next);
      halfEdgeMapping[i] = halfEdges.size() - 1;
    }
    i++;
  }

  for (auto& face : faces) {
    assert(halfEdgeMapping.count(face.halfEdgeIndex) == 1);
    face.halfEdgeIndex = halfEdgeMapping[face.halfEdgeIndex];
  }

  for (auto& he : halfEdges) {
    he.face = faceMapping[he.face];
    he.opp = halfEdgeMapping[he.opp];
    he.next = halfEdgeMapping[he.next];
    he.endVertex = vertexMapping[he.endVertex];
  }
}

/*
 * Implementation of the algorithm
 */

ConvexHull QuickHull::buildMesh(const VecView<glm::dvec3>& pointCloud, bool CCW,
                                double epsilon) {
  if (pointCloud.size() == 0) {
    return ConvexHull();
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

  std::vector<size_t> halfEdgeStack;
  std::vector<bool> halfEdgeProcessed(mesh.halfEdges.size(), false);
  std::vector<bool> halfEdgeProcessedOpp(mesh.halfEdges.size(), false);
  // Iterates through the halfEdges and identifies the valid ones
  for (size_t i = 0; i < mesh.halfEdges.size(); i++) {
    if (!mesh.halfEdges[i].isDisabled() &&
        !mesh.faces[mesh.halfEdges[i].face].isDisabled()) {
      halfEdgeStack.push_back(i);
    }
  }
  if (halfEdgeStack.size() == 0) {
    return ConvexHull();
  }
  Vec<Halfedge> halfEdgeVec;
  std::vector<glm::dvec3> newVerts;
  // To map the index of the vertex in the original point cloud to the index in
  // the new point cloud
  std::unordered_map<size_t, size_t> vertexIndexMapping;
  Halfedge currHalfEdge;
  // Iterates through the valid half edges
  for (size_t index = 0; index < halfEdgeStack.size(); index++) {
    size_t top = halfEdgeStack[index];

    DEBUG_ASSERT(!(mesh.halfEdges[top].isDisabled() ||
                   mesh.faces[mesh.halfEdges[top].face].isDisabled()),
                 logicErr, "halfEdge is disabled");
    // If the half edge has already been processed, skip it
    if (halfEdgeProcessed[top]) {
      continue;
    } else {
      // Process the half edge in the same face till we reach the starting half
      // edge
      while (halfEdgeProcessed[top] != true) {
        DEBUG_ASSERT(
            mesh.halfEdges[top].opp != std::numeric_limits<size_t>::max(),
            logicErr, "halfEdge has no paired halfedge");
        halfEdgeProcessed[top] = true;
        // Maps the endVertex of this half edge to the new point cloud (Since,
        // we will do this for all halfEdges just mapping the end vertex is
        // enough)
        auto itV = vertexIndexMapping.find(mesh.halfEdges[top].endVertex);
        if (itV == vertexIndexMapping.end()) {
          newVerts.push_back(pointCloud[mesh.halfEdges[top].endVertex]);
          vertexIndexMapping[mesh.halfEdges[top].endVertex] =
              newVerts.size() - 1;
          mesh.halfEdges[top].endVertex = newVerts.size() - 1;
        } else {
          mesh.halfEdges[top].endVertex = itV->second;
        }
        // If this half edge has not processed it's opposite (i.e in the
        // opposite edge store this index as the opposite) process it and mark
        // it's opp as max so we can correct it later
        if (!halfEdgeProcessedOpp[top]) {
          size_t top2 = mesh.halfEdges[top].opp;
          DEBUG_ASSERT(!(mesh.halfEdges[top2].isDisabled() ||
                         mesh.faces[mesh.halfEdges[top2].face].isDisabled()),
                       logicErr, "paired halfEdge is disabled");
          DEBUG_ASSERT(
              !(halfEdgeProcessedOpp[top] || halfEdgeProcessedOpp[top2]),
              logicErr, "halfEdge has already been processed");
          // Set's curr.opp as max so we can correct it later
          mesh.halfEdges[top].opp = std::numeric_limits<size_t>::max();
          // Set's the opposite half edge's opp as the current half edge
          mesh.halfEdges[top2].opp = halfEdgeVec.size();
          // Marks both edges as processed (so we will be able to handle the
          // opposite half edge later)
          halfEdgeProcessedOpp[top] = true;
          halfEdgeProcessedOpp[top2] = true;
        }
        // If the opposite half edge has been handled, then just update the opp
        // of it's opposite half edge (since we set it to max before)
        else {
          size_t top2 = mesh.halfEdges[top].opp;
          DEBUG_ASSERT(top2 != std::numeric_limits<size_t>::max(), logicErr,
                       "halfEdge has no paired halfedge");
          // Updates the opposite half edge's opp as the current half edge
          halfEdgeVec[top2].pairedHalfedge = halfEdgeVec.size();
          // Updates the start vertex of the opposite half edge
          halfEdgeVec[top2].startVert = mesh.halfEdges[top].endVertex;
        }
        // Updates values of the current half edge
        currHalfEdge.endVert = mesh.halfEdges[top].endVertex;
        if (mesh.halfEdges[top].opp == std::numeric_limits<size_t>::max()) {
          currHalfEdge.pairedHalfedge = -1;
          currHalfEdge.startVert = -1;
        } else {
          currHalfEdge.pairedHalfedge = mesh.halfEdges[top].opp;
          currHalfEdge.startVert = halfEdgeVec[mesh.halfEdges[top].opp].endVert;
        }
        // Sets the face id to the index/3
        currHalfEdge.face = halfEdgeVec.size() / 3;
        halfEdgeVec.push_back(currHalfEdge);
        // Move to the next half edge
        top = mesh.halfEdges[top].next;
        DEBUG_ASSERT(!(mesh.halfEdges[top].isDisabled() ||
                       mesh.faces[mesh.halfEdges[top].face].isDisabled()),
                     logicErr, "next halfEdge is disabled");
      }
    }
  }
  // Handles counter clockwise if needed
  for (size_t i = 0; i < halfEdgeVec.size(); i += 3) {
    if (CCW) {
      std::swap(halfEdgeVec[i + 1], halfEdgeVec[i + 2]);
    }
  }

  return ConvexHull(halfEdgeVec, newVerts);
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
      int change_flag = 0;
      for (size_t index = 0; index < tf.pointsOnPositiveSide->size(); index++) {
        if ((*tf.pointsOnPositiveSide)[index] == activePointIndex) {
          change_flag = 1;
        } else if (change_flag == 1) {
          change_flag = 2;
          (*tf.pointsOnPositiveSide)[index - 1] =
              (*tf.pointsOnPositiveSide)[index];
        }
      }
      if (change_flag == 1)
        tf.pointsOnPositiveSide->resize(tf.pointsOnPositiveSide->size() - 1);

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

      const glm::dvec3 planeNormal = getTriangleNormal(
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
    const glm::dvec3 N =
        getTriangleNormal(originalVertexData[v[0]], originalVertexData[v[1]],
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
      const double d = getSquaredDistance(originalVertexData[extremeValues[i]],
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
    const double distToRay =
        getSquaredDistanceBetweenPointAndRay(originalVertexData[i], r);
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
  const glm::dvec3 N =
      getTriangleNormal(baseTriangleVertices[0], baseTriangleVertices[1],
                        baseTriangleVertices[2]);
  Plane trianglePlane(N, baseTriangleVertices[0]);
  for (size_t i = 0; i < vCount; i++) {
    const double d = std::abs(
        getSignedDistanceToPlane(originalVertexData[i], trianglePlane));
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
    const vec3 N1 =
        getTriangleNormal(baseTriangleVertices[1], baseTriangleVertices[2],
                          baseTriangleVertices[0]);
    planarPointCloudTemp.clear();
    planarPointCloudTemp.insert(planarPointCloudTemp.begin(),
                                originalVertexData.begin(),
                                originalVertexData.end());
    const vec3 extraPoint = N1 + originalVertexData[0];
    planarPointCloudTemp.push_back(extraPoint);
    maxI = planarPointCloudTemp.size() - 1;
    originalVertexData = Vec<glm::dvec3>(planarPointCloudTemp);
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
    const glm::dvec3 N1 = getTriangleNormal(va, vb, vc);
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

std::unique_ptr<Vec<size_t>> QuickHull::getIndexVectorFromPool() {
  auto r = indexVectorPool.get();
  r->resize(0);
  return r;
}

void QuickHull::reclaimToIndexVectorPool(std::unique_ptr<Vec<size_t>>& ptr) {
  const size_t oldSize = ptr->size();
  if ((oldSize + 1) * 128 < ptr->capacity()) {
    // Reduce memory usage! Huge vectors are needed at the beginning of
    // iteration when faces have many points on their positive side. Later on,
    // smaller vectors will suffice.
    ptr.reset(nullptr);
    return;
  }
  indexVectorPool.reclaim(ptr);
}

bool QuickHull::addPointToFace(typename MeshBuilder::Face& f,
                               size_t pointIndex) {
  const double D =
      getSignedDistanceToPlane(originalVertexData[pointIndex], f.P);
  if (D > 0 && D * D > epsilonSquared * f.P.sqrNLength) {
    if (!f.pointsOnPositiveSide) {
      f.pointsOnPositiveSide = getIndexVectorFromPool();
    }
    f.pointsOnPositiveSide->push_back(pointIndex);
    if (D > f.mostDistantPointDist) {
      f.mostDistantPointDist = D;
      f.mostDistantPoint = pointIndex;
    }
    return true;
  }
  return false;
}
}  // namespace manifold
