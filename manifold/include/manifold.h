// Copyright 2021 Emmett Lalish
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

#include "structs.h"

namespace manifold {

class Manifold {
 public:
  // Creation
  Manifold();
  Manifold(const Mesh&);
  struct SmoothOptions {
    bool distributeVertAngles;
    const std::vector<glm::vec3>& triSharpness;
  };
  static Manifold Smooth(const Mesh&, const SmoothOptions& = {false, {}});
  static Manifold Tetrahedron();
  static Manifold Cube(glm::vec3 size = glm::vec3(1.0f), bool center = false);
  static Manifold Cylinder(float height, float radiusLow,
                           float radiusHigh = -1.0f, int circularSegments = 0,
                           bool center = false);
  static Manifold Sphere(float radius, int circularSegments = 0);
  static Manifold Extrude(Polygons crossSection, float height,
                          int nDivisions = 0, float twistDegrees = 0.0f,
                          glm::vec2 scaleTop = glm::vec2(1.0f));
  static Manifold Revolve(const Polygons& crossSection,
                          int circularSegments = 0);

  // Topological
  static Manifold Compose(const std::vector<Manifold>&);
  std::vector<Manifold> Decompose() const;

  // Defaults for construction
  static void SetMinCircularAngle(float degrees);
  static void SetMinCircularEdgeLength(float length);
  static void SetCircularSegments(int number);
  static int GetCircularSegments(float radius);

  // Information
  Mesh Extract(bool includeNormals = false) const;
  bool IsEmpty() const;
  int NumVert() const;
  int NumEdge() const;
  int NumTri() const;
  Box BoundingBox() const;
  float Precision() const;
  int Genus() const;
  struct Properties {
    float surfaceArea, volume;
  };
  Properties GetProperties() const;
  MeshRelation GetMeshRelation() const;

  // Modification
  Manifold& Translate(glm::vec3);
  Manifold& Scale(glm::vec3);
  Manifold& Rotate(float xDegrees, float yDegrees = 0.0f,
                   float zDegrees = 0.0f);
  Manifold& Transform(const glm::mat4x3&);
  Manifold& Warp(std::function<void(glm::vec3&)>);

  // Refinement
  Manifold Refine(int) const;
  // Manifold RefineToLength(float) const;
  // Manifold RefineToPrecision(float) const;

  // Boolean
  enum class OpType { ADD, SUBTRACT, INTERSECT };
  Manifold Boolean(const Manifold& second, OpType op) const;
  // Boolean operation shorthand
  Manifold operator+(const Manifold&) const;  // ADD (Union)
  Manifold& operator+=(const Manifold&);
  Manifold operator-(const Manifold&) const;  // SUBTRACT (Difference)
  Manifold& operator-=(const Manifold&);
  Manifold operator^(const Manifold&) const;  // INTERSECT
  Manifold& operator^=(const Manifold&);
  // First result is the intersection, second is the difference. This is more
  // efficient than doing them separately.
  std::pair<Manifold, Manifold> Split(const Manifold&) const;
  // First is in the direction of the normal, second is opposite.
  std::pair<Manifold, Manifold> SplitByPlane(glm::vec3 normal,
                                             float originOffset) const;
  // Returns only the first of the above pair.
  Manifold TrimByPlane(glm::vec3 normal, float originOffset) const;

  // Testing hooks
  bool IsManifold() const;
  bool MatchesTriNormals() const;
  int NumOverlaps(const Manifold& second) const;

  ~Manifold();
  Manifold(const Manifold& other);
  Manifold& operator=(const Manifold& other);
  Manifold(Manifold&&) noexcept;
  Manifold& operator=(Manifold&&) noexcept;
  struct Impl;

 private:
  std::unique_ptr<Impl> pImpl_;
  static int circularSegments;
  static float circularAngle;
  static float circularEdgeLength;
};
}  // namespace manifold