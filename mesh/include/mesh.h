// Copyright 2019 Emmett Lalish
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
#include <memory>
#include "mesh_host.h"

namespace manifold {

class Mesh {
 public:
  Mesh();
  Mesh(const MeshHost&);
  static Mesh Tetrahedron();
  static Mesh Cube();
  static Mesh Octahedron();
  static Mesh Sphere(int circularSegments);
  Mesh(const std::vector<Mesh>&);
  std::vector<Mesh> Decompose() const;
  void Append2Host(MeshHost&) const;
  Mesh Copy() const;
  void Refine(int n);

  int NumVert() const;
  int NumEdge() const;
  int NumTri() const;
  Box BoundingBox() const;
  bool IsValid() const;

  void Translate(glm::vec3);
  void Scale(glm::vec3);
  void Rotate(glm::mat3);

  int NumOverlaps(const Mesh& B) const;
  enum class OpType { ADD, SUBTRACT, INTERSECT };
  Mesh Boolean(const Mesh& second, OpType op) const;

  ~Mesh();
  Mesh(Mesh&&);
  Mesh& operator=(Mesh&&);
  struct Impl;

 private:
  mutable std::unique_ptr<Impl> pImpl_;
  mutable glm::mat4 transform_ = glm::mat4(1.0f);

  void ApplyTransform() const;
  Mesh(const Mesh& other);
  Mesh& operator=(const Mesh& other);
};
}  // namespace manifold