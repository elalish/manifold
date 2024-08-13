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

/*
 * INPUT:  a list of points in 3D space (for example, vertices of a 3D mesh)
 *
 * OUTPUT: a ConvexHull object which provides vertex and index buffers of the
 *generated convex hull as a triangle mesh.
 *
 *
 *
 * The implementation is thread-safe if each thread is using its own QuickHull
 *object.
 *
 *
 * SUMMARY OF THE ALGORITHM:
 *         - Create initial simplex (tetrahedron) using extreme points. We have
 *four faces now and they form a convex mesh M.
 *         - For each point, assign them to the first face for which they are on
 *the positive side of (so each point is assigned to at most one face). Points
 *inside the initial tetrahedron are left behind now and no longer affect the
 *calculations.
 *         - Add all faces that have points assigned to them to Face Stack.
 *         - Iterate until Face Stack is empty:
 *              - Pop topmost face F from the stack
 *              - From the points assigned to F, pick the point P that is
 *farthest away from the plane defined by F.
 *              - Find all faces of M that have P on their positive side. Let us
 *call these the "visible faces".
 *              - Because of the way M is constructed, these faces are
 *connected. Solve their horizon edge loop.
 *				- "Extrude to P": Create new faces by connecting
 *P with the points belonging to the horizon edge. Add the new faces to M and
 *remove the visible faces from M.
 *              - Each point that was assigned to visible faces is now assigned
 *to at most one of the newly created faces.
 *              - Those new faces that have points assigned to them are added to
 *the top of Face Stack.
 *          - M is now the convex hull.
 *
 * */
#pragma once
#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "par.h"
#include "shared.h"
#include "vec.h"
// Pool.hpp

class Pool {
  std::vector<std::unique_ptr<manifold::Vec<size_t>>> data;

 public:
  void clear() { data.clear(); }

  void reclaim(std::unique_ptr<manifold::Vec<size_t>>& ptr) {
    data.push_back(std::move(ptr));
  }

  std::unique_ptr<manifold::Vec<size_t>> get() {
    if (data.size() == 0) {
      return std::make_unique<manifold::Vec<size_t>>();
    }
    auto it = data.end() - 1;
    std::unique_ptr<manifold::Vec<size_t>> r = std::move(*it);
    data.erase(it);
    return r;
  }
};

// Plane.hpp

class Plane {
 public:
  glm::dvec3 N;

  // Signed distance (if normal is of length 1) to the plane from origin
  double D;

  // Normal length squared
  double sqrNLength;

  bool isPointOnPositiveSide(const glm::dvec3& Q) const {
    double d = glm::dot(N, Q) + D;
    if (d >= 0) return true;
    return false;
  }

  Plane() = default;

  // Construct a plane using normal N and any point P on the plane
  Plane(const glm::dvec3& N, const glm::dvec3& P)
      : N(N),
        D(glm::dot(-N, P)),
        sqrNLength(N.x * N.x + N.y * N.y + N.z * N.z) {}
};

// Ray.hpp

struct Ray {
  const glm::dvec3 S;
  const glm::dvec3 V;
  const double VInvLengthSquared;

  Ray(const glm::dvec3& S, const glm::dvec3& V)
      : S(S), V(V), VInvLengthSquared(1 / (glm::dot(V, V))) {}
};

// Mesh.hpp

class MeshBuilder {
 public:
  struct HalfEdge {
    // size_t endVertex;
    // size_t opp;
    // size_t face;
    manifold::Halfedge halfEdgeManifold;
    int next;

    void disable() {
      halfEdgeManifold.endVert = std::numeric_limits<int>::max();
    }

    bool isDisabled() const {
      return halfEdgeManifold.endVert == std::numeric_limits<int>::max();
    }
  };

  struct Face {
    int he;
    Plane P{};
    double mostDistantPointDist;
    size_t mostDistantPoint;
    size_t visibilityCheckedOnIteration;
    std::uint8_t isVisibleFaceOnCurrentIteration : 1;
    std::uint8_t inFaceStack : 1;
    std::uint8_t horizonEdgesOnCurrentIteration : 3;
    // Bit for each half edge assigned to this face, each being 0 or 1 depending
    // on whether the edge belongs to horizon edge
    std::unique_ptr<manifold::Vec<size_t>> pointsOnPositiveSide;

    Face()
        : he(std::numeric_limits<int>::max()),
          mostDistantPointDist(0),
          mostDistantPoint(0),
          visibilityCheckedOnIteration(0),
          isVisibleFaceOnCurrentIteration(0),
          inFaceStack(0),
          horizonEdgesOnCurrentIteration(0) {}

    void disable() { he = std::numeric_limits<int>::max(); }

    bool isDisabled() const { return he == std::numeric_limits<int>::max(); }
  };

  // Mesh data
  std::vector<Face> faces;
  std::vector<HalfEdge> halfEdges;

  // When the mesh is modified and faces and half edges are removed from it, we
  // do not actually remove them from the container vectors. Insted, they are
  // marked as disabled which means that the indices can be reused when we need
  // to add new faces and half edges to the mesh. We store the free indices in
  // the following vectors.
  std::vector<size_t> disabledFaces, disabledHalfEdges;

  size_t addFace() {
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

  size_t addHalfEdge() {
    if (disabledHalfEdges.size()) {
      const size_t index = disabledHalfEdges.back();
      disabledHalfEdges.pop_back();
      return index;
    }
    halfEdges.emplace_back();
    return halfEdges.size() - 1;
  }

  // Mark a face as disabled and return a pointer to the points that were on the
  // positive of it.
  std::unique_ptr<manifold::Vec<size_t>> disableFace(size_t faceIndex) {
    auto& f = faces[faceIndex];
    f.disable();
    disabledFaces.push_back(faceIndex);
    return std::move(f.pointsOnPositiveSide);
  }

  void disableHalfEdge(size_t heIndex) {
    auto& he = halfEdges[heIndex];
    he.disable();
    disabledHalfEdges.push_back(heIndex);
  }

  MeshBuilder() = default;

  // Create a mesh with initial tetrahedron ABCD. Dot product of AB with the
  // normal of triangle ABC should be negative.
  void setup(size_t a, size_t b, size_t c, size_t d) {
    faces.clear();
    halfEdges.clear();
    disabledFaces.clear();
    disabledHalfEdges.clear();

    faces.reserve(4);
    halfEdges.reserve(12);

    // Create halfedges
    HalfEdge AB;
    AB.halfEdgeManifold.endVert = b;
    AB.halfEdgeManifold.pairedHalfedge = 6;
    AB.halfEdgeManifold.face = 0;
    AB.next = 1;
    halfEdges.push_back(AB);

    HalfEdge BC;
    BC.halfEdgeManifold.endVert = c;
    BC.halfEdgeManifold.pairedHalfedge = 9;
    BC.halfEdgeManifold.face = 0;
    BC.next = 2;
    halfEdges.push_back(BC);

    HalfEdge CA;
    CA.halfEdgeManifold.endVert = a;
    CA.halfEdgeManifold.pairedHalfedge = 3;
    CA.halfEdgeManifold.face = 0;
    CA.next = 0;
    halfEdges.push_back(CA);

    HalfEdge AC;
    AC.halfEdgeManifold.endVert = c;
    AC.halfEdgeManifold.pairedHalfedge = 2;
    AC.halfEdgeManifold.face = 1;
    AC.next = 4;
    halfEdges.push_back(AC);

    HalfEdge CD;
    CD.halfEdgeManifold.endVert = d;
    CD.halfEdgeManifold.pairedHalfedge = 11;
    CD.halfEdgeManifold.face = 1;
    CD.next = 5;
    halfEdges.push_back(CD);

    HalfEdge DA;
    DA.halfEdgeManifold.endVert = a;
    DA.halfEdgeManifold.pairedHalfedge = 7;
    DA.halfEdgeManifold.face = 1;
    DA.next = 3;
    halfEdges.push_back(DA);

    HalfEdge BA;
    BA.halfEdgeManifold.endVert = a;
    BA.halfEdgeManifold.pairedHalfedge = 0;
    BA.halfEdgeManifold.face = 2;
    BA.next = 7;
    halfEdges.push_back(BA);

    HalfEdge AD;
    AD.halfEdgeManifold.endVert = d;
    AD.halfEdgeManifold.pairedHalfedge = 5;
    AD.halfEdgeManifold.face = 2;
    AD.next = 8;
    halfEdges.push_back(AD);

    HalfEdge DB;
    DB.halfEdgeManifold.endVert = b;
    DB.halfEdgeManifold.pairedHalfedge = 10;
    DB.halfEdgeManifold.face = 2;
    DB.next = 6;
    halfEdges.push_back(DB);

    HalfEdge CB;
    CB.halfEdgeManifold.endVert = b;
    CB.halfEdgeManifold.pairedHalfedge = 1;
    CB.halfEdgeManifold.face = 3;
    CB.next = 10;
    halfEdges.push_back(CB);

    HalfEdge BD;
    BD.halfEdgeManifold.endVert = d;
    BD.halfEdgeManifold.pairedHalfedge = 8;
    BD.halfEdgeManifold.face = 3;
    BD.next = 11;
    halfEdges.push_back(BD);

    HalfEdge DC;
    DC.halfEdgeManifold.endVert = c;
    DC.halfEdgeManifold.pairedHalfedge = 4;
    DC.halfEdgeManifold.face = 3;
    DC.next = 9;
    halfEdges.push_back(DC);

    // Create faces
    Face ABC;
    ABC.he = 0;
    faces.push_back(std::move(ABC));

    Face ACD;
    ACD.he = 3;
    faces.push_back(std::move(ACD));

    Face BAD;
    BAD.he = 6;
    faces.push_back(std::move(BAD));

    Face CBD;
    CBD.he = 9;
    faces.push_back(std::move(CBD));
  }

  std::array<size_t, 3> getVertexIndicesOfFace(const Face& f) const {
    std::array<size_t, 3> v;
    const HalfEdge* he = &halfEdges[f.he];
    v[0] = he->halfEdgeManifold.endVert;
    he = &halfEdges[he->next];
    v[1] = he->halfEdgeManifold.endVert;
    he = &halfEdges[he->next];
    v[2] = he->halfEdgeManifold.endVert;
    return v;
  }

  std::array<int, 2> getVertexIndicesOfHalfEdge(const HalfEdge& he) const {
    return {
        halfEdges[he.halfEdgeManifold.pairedHalfedge].halfEdgeManifold.endVert,
        he.halfEdgeManifold.endVert};
  }

  std::array<int, 3> getHalfEdgeIndicesOfFace(const Face& f) const {
    return {f.he, halfEdges[f.he].next, halfEdges[halfEdges[f.he].next].next};
  }
};

// ConvexHull.hpp

class ConvexHull {
  std::unique_ptr<manifold::Vec<glm::dvec3>> optimizedVertexBuffer;
  manifold::Vec<glm::dvec3> vertices;
  std::vector<size_t> indices;

 public:
  ConvexHull() {}

  // Copy constructor
  ConvexHull(const ConvexHull& o) {
    indices = o.indices;
    if (o.optimizedVertexBuffer) {
      optimizedVertexBuffer.reset(
          new manifold::Vec<glm::dvec3>(*o.optimizedVertexBuffer));
      vertices = *optimizedVertexBuffer;
    } else {
      vertices = o.vertices;
    }
  }

  ConvexHull& operator=(const ConvexHull& o) {
    if (&o == this) {
      return *this;
    }
    indices = o.indices;
    if (o.optimizedVertexBuffer) {
      optimizedVertexBuffer.reset(
          new manifold::Vec<glm::dvec3>(*o.optimizedVertexBuffer));
      vertices = manifold::Vec<glm::dvec3>(*optimizedVertexBuffer);
    } else {
      vertices = o.vertices;
    }
    return *this;
  }

  ConvexHull(ConvexHull&& o) {
    indices = std::move(o.indices);
    if (o.optimizedVertexBuffer) {
      optimizedVertexBuffer = std::move(o.optimizedVertexBuffer);
      o.vertices = manifold::Vec<glm::dvec3>();
      vertices = manifold::Vec<glm::dvec3>(*optimizedVertexBuffer);
    } else {
      vertices = o.vertices;
    }
  }

  ConvexHull& operator=(ConvexHull&& o) {
    if (&o == this) {
      return *this;
    }
    indices = std::move(o.indices);
    if (o.optimizedVertexBuffer) {
      optimizedVertexBuffer = std::move(o.optimizedVertexBuffer);
      o.vertices = manifold::Vec<glm::dvec3>();
      vertices = manifold::Vec<glm::dvec3>(*optimizedVertexBuffer);
    } else {
      vertices = o.vertices;
    }
    return *this;
  }

  // Construct vertex and index buffers from half edge mesh and pointcloud
  ConvexHull(const MeshBuilder& mesh_input,
             const manifold::Vec<glm::dvec3>& pointCloud, bool CCW,
             bool useOriginalIndices) {
    if (!useOriginalIndices) {
      optimizedVertexBuffer.reset(new manifold::Vec<glm::dvec3>());
    }

    std::vector<bool> faceProcessed(mesh_input.faces.size(), false);
    std::vector<size_t> faceStack;
    // Map vertex indices from original point cloud to the new mesh vertex
    // indices
    std::unordered_map<size_t, size_t> vertexIndexMapping;
    for (size_t i = 0; i < mesh_input.faces.size(); i++) {
      if (!mesh_input.faces[i].isDisabled()) {
        faceStack.push_back(i);
        break;
      }
    }
    if (faceStack.size() == 0) {
      return;
    }

    const size_t iCCW = CCW ? 1 : 0;
    const size_t finalMeshFaceCount =
        mesh_input.faces.size() - mesh_input.disabledFaces.size();
    indices.reserve(finalMeshFaceCount * 3);

    while (faceStack.size()) {
      auto it = faceStack.end() - 1;
      size_t top = *it;
      assert(!mesh_input.faces[top].isDisabled());
      faceStack.erase(it);
      if (faceProcessed[top]) {
        continue;
      } else {
        faceProcessed[top] = true;
        auto halfEdgesMesh =
            mesh_input.getHalfEdgeIndicesOfFace(mesh_input.faces[top]);
        int adjacent[] = {mesh_input
                              .halfEdges[mesh_input.halfEdges[halfEdgesMesh[0]]
                                             .halfEdgeManifold.pairedHalfedge]
                              .halfEdgeManifold.face,
                          mesh_input
                              .halfEdges[mesh_input.halfEdges[halfEdgesMesh[1]]
                                             .halfEdgeManifold.pairedHalfedge]
                              .halfEdgeManifold.face,
                          mesh_input
                              .halfEdges[mesh_input.halfEdges[halfEdgesMesh[2]]
                                             .halfEdgeManifold.pairedHalfedge]
                              .halfEdgeManifold.face};
        for (auto a : adjacent) {
          if (!faceProcessed[a] && !mesh_input.faces[a].isDisabled()) {
            faceStack.push_back(a);
          }
        }
        auto MeshVertices =
            mesh_input.getVertexIndicesOfFace(mesh_input.faces[top]);
        if (!useOriginalIndices) {
          for (auto& v : MeshVertices) {
            auto itV = vertexIndexMapping.find(v);
            if (itV == vertexIndexMapping.end()) {
              optimizedVertexBuffer->push_back(pointCloud[v]);
              vertexIndexMapping[v] = optimizedVertexBuffer->size() - 1;
              v = optimizedVertexBuffer->size() - 1;
            } else {
              v = itV->second;
            }
          }
        }
        indices.push_back(MeshVertices[0]);
        indices.push_back(MeshVertices[1 + iCCW]);
        indices.push_back(MeshVertices[2 - iCCW]);
      }
    }

    if (!useOriginalIndices) {
      vertices = manifold::Vec<glm::dvec3>(*optimizedVertexBuffer);
    } else {
      vertices = pointCloud;
    }
  }

  std::vector<size_t>& getIndexBuffer() { return indices; }

  const std::vector<size_t>& getIndexBuffer() const { return indices; }

  manifold::VecView<glm::dvec3>& getVertexBuffer() { return vertices; }

  const manifold::VecView<glm::dvec3>& getVertexBuffer() const {
    return vertices;
  }
};

// HalfEdgeMesh.hpp

class HalfEdgeMesh {
 public:
  struct HalfEdge {
    // size_t endVertex;
    // size_t opp;
    // size_t face;
    manifold::Halfedge halfEdgeManifold;
    int next;
  };

  struct Face {
    // Index of one of the half edges of this face
    size_t halfEdgeIndex;
  };

  std::vector<glm::dvec3> vertices;
  std::vector<Face> faces;
  std::vector<HalfEdge> halfEdges;

  HalfEdgeMesh(const MeshBuilder& builderObject,
               const manifold::VecView<glm::dvec3>& vertexData) {
    std::unordered_map<size_t, size_t> faceMapping;
    std::unordered_map<size_t, size_t> halfEdgeMapping;
    std::unordered_map<size_t, size_t> vertexMapping;

    size_t i = 0;
    for (const auto& face : builderObject.faces) {
      if (!face.isDisabled()) {
        faces.push_back({static_cast<size_t>(face.he)});
        faceMapping[i] = faces.size() - 1;

        const auto heIndices = builderObject.getHalfEdgeIndicesOfFace(face);
        for (const auto heIndex : heIndices) {
          const size_t vertexIndex =
              builderObject.halfEdges[heIndex].halfEdgeManifold.endVert;
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
        halfEdges.push_back(
            {{static_cast<int>(halfEdge.halfEdgeManifold.endVert),
              static_cast<int>(halfEdge.halfEdgeManifold.pairedHalfedge),
              static_cast<int>(halfEdge.halfEdgeManifold.face),
              static_cast<int>(halfEdge.next)}});
        halfEdgeMapping[i] = halfEdges.size() - 1;
      }
      i++;
    }

    for (auto& face : faces) {
      assert(halfEdgeMapping.count(face.halfEdgeIndex) == 1);
      face.halfEdgeIndex = halfEdgeMapping[face.halfEdgeIndex];
    }

    for (auto& he : halfEdges) {
      he.halfEdgeManifold.face = faceMapping[he.halfEdgeManifold.face];
      he.halfEdgeManifold.pairedHalfedge =
          halfEdgeMapping[he.halfEdgeManifold.pairedHalfedge];
      he.next = halfEdgeMapping[he.next];
      he.halfEdgeManifold.endVert = vertexMapping[he.halfEdgeManifold.endVert];
    }
  }
};

// MathUtils.hpp

namespace mathutils {

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

}  // namespace mathutils

// QuickHull.hpp

struct DiagnosticsData {
  // How many times QuickHull failed to solve the  horizon edge. Failures lead
  // to degenerated convex hulls.
  size_t failedHorizonEdges;

  DiagnosticsData() : failedHorizonEdges(0) {}
};

double defaultEps();

// Computes convex hull for a given point cloud.
// Params:
//   pointCloud: a vector of of 3D points
//   CCW: whether the output mesh triangles should have CCW orientation
//   useOriginalIndices: should the output mesh use same vertex indices as the
//   original point cloud. If this is false,
//      then we generate a new vertex buffer which contains only the vertices
//      that are part of the convex hull.
//   eps: minimum distance to a plane to consider a point being on positive of
//   it (for a point cloud with scale 1)
ConvexHull getConvexHull(const std::vector<glm::dvec3>& pointCloud, bool CCW,
                         bool useOriginalIndices, double eps = defaultEps());

class QuickHull {
  using vec3 = glm::dvec3;

  double m_epsilon, epsilonSquared, scale;
  bool planar;
  std::vector<vec3> planarPointCloudTemp;
  manifold::VecView<glm::dvec3> originalVertexData;
  MeshBuilder mesh;
  std::array<size_t, 6> extremeValues;
  DiagnosticsData diagnostics;

  // Temporary variables used during iteration process
  std::vector<size_t> newFaceIndices;
  std::vector<size_t> newHalfEdgeIndices;
  std::vector<std::unique_ptr<manifold::Vec<size_t>>> disabledFacePointVectors;
  std::vector<size_t> visibleFaces;
  std::vector<size_t> horizonEdgesData;
  struct FaceData {
    size_t faceIndex;
    // If the face turns out not to be visible, this half edge will be marked as
    // horizon edge
    int enteredFromHalfEdge;
    FaceData(size_t fi, size_t he) : faceIndex(fi), enteredFromHalfEdge(he) {}
  };
  std::vector<FaceData> possiblyVisibleFaces;
  std::deque<size_t> faceList;

  // Create a half edge mesh representing the base tetrahedron from which the
  // QuickHull iteration proceeds. extremeValues must be properly set up when
  // this is called.
  void setupInitialTetrahedron();

  // Given a list of half edges, try to rearrange them so that they form a loop.
  // Return true on success.
  bool reorderHorizonEdges(std::vector<size_t>& horizonEdges);

  // Find indices of extreme values (max x, min x, max y, min y, max z, min z)
  // for the given point cloud
  std::array<size_t, 6> getExtremeValues();

  // Compute scale of the vertex data.
  double getScale(const std::array<size_t, 6>& extremeValuesInput);

  // Each face contains a unique pointer to a vector of indices. However, many -
  // often most - faces do not have any points on the positive side of them
  // especially at the the end of the iteration. When a face is removed from the
  // mesh, its associated point vector, if such exists, is moved to the index
  // vector pool, and when we need to add new faces with points on the positive
  // side to the mesh, we reuse these vectors. This reduces the amount of
  // std::vectors we have to deal with, and impact on performance is remarkable.
  Pool indexVectorPool;
  inline std::unique_ptr<manifold::Vec<size_t>> getIndexVectorFromPool();
  inline void reclaimToIndexVectorPool(
      std::unique_ptr<manifold::Vec<size_t>>& ptr);

  // Associates a point with a face if the point resides on the positive side of
  // the plane. Returns true if the points was on the positive side.
  inline bool addPointToFace(typename MeshBuilder::Face& f, size_t pointIndex);

  // This will update mesh from which we create the ConvexHull object that
  // getConvexHull function returns
  void createConvexHalfEdgeMesh();

  // Constructs the convex hull into a MeshBuilder object which can be converted
  // to a ConvexHull or Mesh object
  void buildMesh(const manifold::VecView<glm::dvec3>& pointCloud, bool CCW,
                 bool useOriginalIndices, double eps);

 public:
  QuickHull(const manifold::Vec<glm::dvec3>& pointCloudVec)
      : originalVertexData(manifold::VecView(pointCloudVec)) {}
  // The getConvexHull functions will setup a VertexDataSource object and
  // call this
  ConvexHull getConvexHull(const manifold::Vec<glm::dvec3>& pointCloud,
                           bool CCW, bool useOriginalIndices, double eps);
  // Computes convex hull for a given point cloud. This function assumes that
  // the vertex data resides in memory in the following format:
  // x_0,y_0,z_0,x_1,y_1,z_1,... Params:
  //   vertexData: pointer to the X component of the first point of the point
  //   cloud. vertexCount: number of vertices in the point cloud CCW: whether
  //   the output mesh triangles should have CCW orientation eps: minimum
  //   distance to a plane to consider a point being on positive side of it (for
  //   a point cloud with scale 1)
  // Returns:
  //   Convex hull of the point cloud as a mesh object with half edge structure.
  HalfEdgeMesh getConvexHullAsMesh(const double* vertexData, size_t vertexCount,
                                   bool CCW, double eps = defaultEps());

  // Get diagnostics about last generated convex hull
  const DiagnosticsData& getDiagnostics() { return diagnostics; }
};

/*
 * Inline function definitions
 */

std::unique_ptr<manifold::Vec<size_t>> QuickHull::getIndexVectorFromPool() {
  auto r = indexVectorPool.get();
  // r->clear();
  r->resize(0);
  return r;
}

void QuickHull::reclaimToIndexVectorPool(
    std::unique_ptr<manifold::Vec<size_t>>& ptr) {
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
      mathutils::getSignedDistanceToPlane(originalVertexData[pointIndex], f.P);
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
