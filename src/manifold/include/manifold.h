// Copyright 2021 The Manifold Authors.
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

#pragma once
#include <functional>
#include <memory>

#include "cross_section.h"
#include "public.h"

namespace manifold {

/**
 * @ingroup Debug
 *
 * Allows modification of the assertions checked in MANIFOLD_DEBUG mode.
 *
 * @return ExecutionParams&
 */
ExecutionParams& ManifoldParams();

class CsgNode;
class CsgLeafNode;

/** @defgroup Core
 *  @brief The central classes of the library
 *  @{
 */
class Manifold {
 public:
  /** @name Creation
   *  Constructors
   */
  ///@{
  Manifold();
  ~Manifold();
  Manifold(const Manifold& other);
  Manifold& operator=(const Manifold& other);
  Manifold(Manifold&&) noexcept;
  Manifold& operator=(Manifold&&) noexcept;

  Manifold(const MeshGL&, const std::vector<float>& propertyTolerance = {});
  Manifold(const Mesh&);

  static Manifold Smooth(const MeshGL&,
                         const std::vector<Smoothness>& sharpenedEdges = {});
  static Manifold Smooth(const Mesh&,
                         const std::vector<Smoothness>& sharpenedEdges = {});
  static Manifold Tetrahedron();
  static Manifold Cube(glm::vec3 size = glm::vec3(1.0f), bool center = false);
  static Manifold Cylinder(float height, float radiusLow,
                           float radiusHigh = -1.0f, int circularSegments = 0,
                           bool center = false);
  static Manifold Sphere(float radius, int circularSegments = 0);
  static Manifold Extrude(const CrossSection& crossSection, float height,
                          int nDivisions = 0, float twistDegrees = 0.0f,
                          glm::vec2 scaleTop = glm::vec2(1.0f));
  static Manifold Revolve(const CrossSection& crossSection,
                          int circularSegments = 0);
  ///@}

  /** @name Topological
   *  No geometric calculations.
   */
  ///@{
  static Manifold Compose(const std::vector<Manifold>&);
  std::vector<Manifold> Decompose() const;
  ///@}

  /** @name Information
   *  Details of the manifold
   */
  ///@{
  Mesh GetMesh() const;
  MeshGL GetMeshGL(glm::ivec3 normalIdx = glm::ivec3(0)) const;
  bool IsEmpty() const;
  enum class Error {
    NoError,
    NonFiniteVertex,
    NotManifold,
    VertexOutOfBounds,
    PropertiesWrongLength,
    MissingPositionProperties,
    MergeVectorsDifferentLengths,
    MergeIndexOutOfBounds,
    TransformWrongLength,
    RunIndexWrongLength,
    FaceIDWrongLength,
    InvalidConstruction,
  };
  Error Status() const;
  int NumVert() const;
  int NumEdge() const;
  int NumTri() const;
  int NumProp() const;
  int NumPropVert() const;
  Box BoundingBox() const;
  float Precision() const;
  int Genus() const;
  Properties GetProperties() const;
  Curvature GetCurvature() const;
  ///@}

  /** @name Mesh ID
   *  Details of the manifold's relation to its input meshes, for the purposes
   * of reapplying mesh properties.
   */
  ///@{
  int OriginalID() const;
  Manifold AsOriginal() const;
  static uint32_t ReserveIDs(uint32_t);
  ///@}

  /** @name Modification
   */
  ///@{
  Manifold Translate(glm::vec3) const;
  Manifold Scale(glm::vec3) const;
  Manifold Rotate(float xDegrees, float yDegrees = 0.0f,
                  float zDegrees = 0.0f) const;
  Manifold Transform(const glm::mat4x3&) const;
  Manifold Mirror(glm::vec3) const;
  Manifold Warp(std::function<void(glm::vec3&)>) const;
  Manifold Refine(int) const;
  // Manifold RefineToLength(float);
  // Manifold RefineToPrecision(float);
  ///@}

  /** @name Boolean
   *  Combine two manifolds
   */
  ///@{
  Manifold Boolean(const Manifold& second, OpType op) const;
  static Manifold BatchBoolean(const std::vector<Manifold>& manifolds,
                               OpType op);
  // Boolean operation shorthand
  Manifold operator+(const Manifold&) const;  // Add (Union)
  Manifold& operator+=(const Manifold&);
  Manifold operator-(const Manifold&) const;  // Subtract (Difference)
  Manifold& operator-=(const Manifold&);
  Manifold operator^(const Manifold&) const;  // Intersect
  Manifold& operator^=(const Manifold&);
  std::pair<Manifold, Manifold> Split(const Manifold&) const;
  std::pair<Manifold, Manifold> SplitByPlane(glm::vec3 normal,
                                             float originOffset) const;
  Manifold TrimByPlane(glm::vec3 normal, float originOffset) const;
  ///@}

  /** @name Testing hooks
   *  These are just for internal testing.
   */
  ///@{
  bool IsManifold() const;
  bool MatchesTriNormals() const;
  int NumDegenerateTris() const;
  int NumOverlaps(const Manifold& second) const;
  ///@}

  struct Impl;

 private:
  Manifold(std::shared_ptr<CsgNode> pNode_);
  Manifold(std::shared_ptr<Impl> pImpl_);

  mutable std::shared_ptr<CsgNode> pNode_;

  CsgLeafNode& GetCsgLeafNode() const;

  static int circularSegments_;
  static float circularAngle_;
  static float circularEdgeLength_;
};
/** @} */
}  // namespace manifold
