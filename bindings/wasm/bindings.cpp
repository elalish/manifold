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

#include <emscripten/bind.h>

using namespace emscripten;

#include <manifold.h>

using namespace manifold;

Manifold Union(Manifold& a, Manifold& b) { return a + b; }
void Add(Manifold& a, Manifold& b) { a += b; }

Manifold Difference(Manifold& a, Manifold& b) { return a - b; }
void Subtract(Manifold& a, Manifold& b) { a -= b; }

Manifold Intersection(Manifold& a, Manifold& b) { return a ^ b; }
void Intersect(Manifold& a, Manifold& b) { a ^= b; }

std::vector<SimplePolygon> ToPolygon(
    std::vector<std::vector<glm::vec2>>& polygons) {
  std::vector<SimplePolygon> simplePolygons(polygons.size());
  for (int i = 0; i < polygons.size(); i++) {
    std::vector<PolyVert> vertices(polygons[i].size());
    for (int j = 0; j < polygons[i].size(); j++) {
      vertices[j] = {polygons[i][j], j};
    }
    simplePolygons[i] = {vertices};
  }
  return simplePolygons;
}

Manifold Extrude(std::vector<std::vector<glm::vec2>>& polygons, float height,
                 int nDivisions, float twistDegrees, glm::vec2 scaleTop) {
  return Manifold::Extrude(ToPolygon(polygons), height, nDivisions,
                           twistDegrees, scaleTop);
}

Manifold Revolve(std::vector<std::vector<glm::vec2>>& polygons,
                 int circularSegments) {
  return Manifold::Revolve(ToPolygon(polygons), circularSegments);
}

Manifold Transform(Manifold &manifold, std::vector<float> &mat) {
  glm::mat4x3 matrix;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 3; j++)
      matrix[i][j] = mat[i*3+j];
  return manifold.Transform(matrix);
}

EMSCRIPTEN_BINDINGS(whatever) {
  value_object<glm::vec2>("vec2")
      .field("x", &glm::vec2::x)
      .field("y", &glm::vec2::y);

  value_object<glm::ivec3>("ivec3")
      .field("0", &glm::ivec3::x)
      .field("1", &glm::ivec3::y)
      .field("2", &glm::ivec3::z);

  register_vector<glm::ivec3>("Vector_ivec3");

  value_object<glm::vec3>("vec3")
      .field("x", &glm::vec3::x)
      .field("y", &glm::vec3::y)
      .field("z", &glm::vec3::z);

  register_vector<glm::vec3>("Vector_vec3");
  register_vector<glm::vec2>("Vector_vec2");
  register_vector<std::vector<glm::vec2>>("Vector2_vec2");
  register_vector<float>("Vector_f32");

  value_object<glm::vec4>("vec4")
      .field("x", &glm::vec4::x)
      .field("y", &glm::vec4::y)
      .field("z", &glm::vec4::z)
      .field("w", &glm::vec4::w);

  register_vector<glm::vec4>("Vector_vec4");

  value_object<Mesh>("Mesh")
      .field("vertPos", &Mesh::vertPos)
      .field("triVerts", &Mesh::triVerts)
      .field("vertNormal", &Mesh::vertNormal)
      .field("halfedgeTangent", &Mesh::halfedgeTangent);

  class_<Manifold>("Manifold")
      .constructor<Mesh>()
      .function("add", &Add)
      .function("subtract", &Subtract)
      .function("intersect", &Intersect)
      .function("getMesh", &Manifold::GetMesh)
      .function("refine", &Manifold::Refine)
      .function("_Transform", &Transform)
      .function("_Translate", &Manifold::Translate)
      .function("_Rotate", &Manifold::Rotate)
      .function("_Scale", &Manifold::Scale);
      // .function("Warp", &Manifold::Warp);

  function("_Cube", &Manifold::Cube);
  function("_Cylinder", &Manifold::Cylinder);
  function("_Sphere", &Manifold::Sphere);
  function("_Extrude", &Extrude);
  function("_Revolve", &Revolve);

  function("union", &Union);
  function("difference", &Difference);
  function("intersection", &Intersection);
}
