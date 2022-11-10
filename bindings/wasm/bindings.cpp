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
#include <emscripten/val.h>

using namespace emscripten;

#include <manifold.h>
#include <sdf.h>

using namespace manifold;

Manifold Union(const Manifold& a, const Manifold& b) { return a + b; }

Manifold Difference(const Manifold& a, const Manifold& b) { return a - b; }

Manifold Intersection(const Manifold& a, const Manifold& b) { return a ^ b; }

Manifold UnionN(const std::vector<Manifold>& manifolds) {
  return Manifold::BatchBoolean(manifolds, Manifold::OpType::ADD);
  ;
}

Manifold DifferenceN(const std::vector<Manifold>& manifolds) {
  return Manifold::BatchBoolean(manifolds, Manifold::OpType::SUBTRACT);
  ;
}

Manifold IntersectionN(const std::vector<Manifold>& manifolds) {
  return Manifold::BatchBoolean(manifolds, Manifold::OpType::INTERSECT);
  ;
}

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

val GetMeshJS(const Manifold& manifold) {
  MeshGL mesh = manifold.GetMeshGL();
  val meshJS = val::object();

  meshJS.set("triVerts",
             val(typed_memory_view(mesh.triVerts.size(), mesh.triVerts.data()))
                 .call<val>("slice"));
  meshJS.set("vertPos",
             val(typed_memory_view(mesh.vertPos.size(), mesh.vertPos.data()))
                 .call<val>("slice"));
  meshJS.set("vertNormal", val(typed_memory_view(mesh.vertNormal.size(),
                                                 mesh.vertNormal.data()))
                               .call<val>("slice"));
  meshJS.set("halfedgeTangent",
             val(typed_memory_view(mesh.halfedgeTangent.size(),
                                   mesh.halfedgeTangent.data()))
                 .call<val>("slice"));

  return meshJS;
}

MeshGL MeshJS2GL(const val& mesh) {
  MeshGL out;
  out.triVerts = convertJSArrayToNumberVector<uint32_t>(mesh["triVerts"]);
  out.vertPos = convertJSArrayToNumberVector<float>(mesh["vertPos"]);
  if (mesh["vertNormal"] != val::undefined()) {
    out.vertNormal = convertJSArrayToNumberVector<float>(mesh["vertNormal"]);
  }
  if (mesh["halfedgeTangent"] != val::undefined()) {
    out.halfedgeTangent =
        convertJSArrayToNumberVector<float>(mesh["halfedgeTangent"]);
  }
  return out;
}

Manifold FromMeshJS(const val& mesh) { return Manifold(MeshJS2GL(mesh)); }

Manifold Smooth(const val& mesh,
                const std::vector<Smoothness>& sharpenedEdges = {}) {
  return Manifold::Smooth(MeshJS2GL(mesh), sharpenedEdges);
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

Manifold Transform(Manifold& manifold, std::vector<float>& mat) {
  glm::mat4x3 matrix;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 3; j++) matrix[i][j] = mat[i * 3 + j];
  return manifold.Transform(matrix);
}

Manifold Warp(Manifold& manifold, uintptr_t funcPtr) {
  void (*f)(glm::vec3&) = reinterpret_cast<void (*)(glm::vec3&)>(funcPtr);
  return manifold.Warp(f);
}

Manifold LevelSetJs(uintptr_t funcPtr, Box bounds, float edgeLength,
                    float level) {
  float (*f)(const glm::vec3&) =
      reinterpret_cast<float (*)(const glm::vec3&)>(funcPtr);
  Mesh m = LevelSet(f, bounds, edgeLength, level);
  return Manifold(m);
}

EMSCRIPTEN_BINDINGS(whatever) {
  value_object<glm::vec2>("vec2")
      .field("x", &glm::vec2::x)
      .field("y", &glm::vec2::y);

  value_object<glm::ivec3>("ivec3")
      .field("0", &glm::ivec3::x)
      .field("1", &glm::ivec3::y)
      .field("2", &glm::ivec3::z);

  value_object<glm::vec3>("vec3")
      .field("x", &glm::vec3::x)
      .field("y", &glm::vec3::y)
      .field("z", &glm::vec3::z);

  value_object<glm::vec4>("vec4")
      .field("x", &glm::vec4::x)
      .field("y", &glm::vec4::y)
      .field("z", &glm::vec4::z)
      .field("w", &glm::vec4::w);

  enum_<Manifold::Error>("status")
      .value("NO_ERROR", Manifold::Error::NO_ERROR)
      .value("NON_FINITE_VERTEX", Manifold::Error::NON_FINITE_VERTEX)
      .value("NOT_MANIFOLD", Manifold::Error::NOT_MANIFOLD)
      .value("VERTEX_INDEX_OUT_OF_BOUNDS",
             Manifold::Error::VERTEX_INDEX_OUT_OF_BOUNDS)
      .value("PROPERTIES_WRONG_LENGTH",
             Manifold::Error::PROPERTIES_WRONG_LENGTH)
      .value("TRI_PROPERTIES_WRONG_LENGTH",
             Manifold::Error::TRI_PROPERTIES_WRONG_LENGTH)
      .value("TRI_PROPERTIES_OUT_OF_BOUNDS",
             Manifold::Error::TRI_PROPERTIES_OUT_OF_BOUNDS);

  value_object<Box>("box").field("min", &Box::min).field("max", &Box::max);

  value_object<Smoothness>("smoothness")
      .field("halfedge", &Smoothness::halfedge)
      .field("smoothness", &Smoothness::smoothness);

  value_object<Properties>("properties")
      .field("surfaceArea", &Properties::surfaceArea)
      .field("volume", &Properties::volume);

  value_object<BaryRef>("baryRef")
      .field("meshID", &BaryRef::meshID)
      .field("originalID", &BaryRef::originalID)
      .field("tri", &BaryRef::tri)
      .field("vertBary", &BaryRef::vertBary);

  value_object<MeshRelation>("meshRelation")
      .field("barycentric", &MeshRelation::barycentric)
      .field("triBary", &MeshRelation::triBary);

  value_object<Curvature>("curvature")
      .field("maxMeanCurvature", &Curvature::maxMeanCurvature)
      .field("minMeanCurvature", &Curvature::minMeanCurvature)
      .field("maxGaussianCurvature", &Curvature::maxGaussianCurvature)
      .field("minGaussianCurvature", &Curvature::minGaussianCurvature)
      .field("vertMeanCurvature", &Curvature::vertMeanCurvature)
      .field("vertGaussianCurvature", &Curvature::vertGaussianCurvature);

  register_vector<glm::ivec3>("Vector_ivec3");
  register_vector<glm::vec3>("Vector_vec3");
  register_vector<glm::vec2>("Vector_vec2");
  register_vector<std::vector<glm::vec2>>("Vector2_vec2");
  register_vector<float>("Vector_f32");
  register_vector<Manifold>("Vector_manifold");
  register_vector<Smoothness>("Vector_smoothness");
  register_vector<BaryRef>("Vector_baryRef");
  register_vector<glm::vec4>("Vector_vec4");

  class_<Manifold>("Manifold")
      .constructor(&FromMeshJS)
      .function("add", &Union)
      .function("subtract", &Difference)
      .function("intersect", &Intersection)
      .function("_GetMeshJS", &GetMeshJS)
      .function("refine", &Manifold::Refine)
      .function("_Warp", &Warp)
      .function("_Transform", &Transform)
      .function("_Translate", &Manifold::Translate)
      .function("_Rotate", &Manifold::Rotate)
      .function("_Scale", &Manifold::Scale)
      .function("_Decompose", &Manifold::Decompose)
      .function("isEmpty", &Manifold::IsEmpty)
      .function("status", &Manifold::Status)
      .function("numVert", &Manifold::NumVert)
      .function("numEdge", &Manifold::NumEdge)
      .function("numTri", &Manifold::NumTri)
      .function("_boundingBox", &Manifold::BoundingBox)
      .function("precision", &Manifold::Precision)
      .function("genus", &Manifold::Genus)
      .function("getProperties", &Manifold::GetProperties)
      .function("_getCurvature", &Manifold::GetCurvature)
      .function("originalID", &Manifold::OriginalID)
      .function("asOriginal", &Manifold::AsOriginal)
      .function("_getMeshRelation", &Manifold::GetMeshRelation);

  function("_Cube", &Manifold::Cube);
  function("_Cylinder", &Manifold::Cylinder);
  function("_Sphere", &Manifold::Sphere);
  function("tetrahedron", &Manifold::Tetrahedron);
  function("_Smooth", &Smooth);
  function("_Extrude", &Extrude);
  function("_Revolve", &Revolve);
  function("_LevelSet", &LevelSetJs);

  function("_unionN", &UnionN);
  function("_differenceN", &DifferenceN);
  function("_intersectionN", &IntersectionN);
  function("_Compose", &Manifold::Compose);

  function("setMinCircularAngle", &Manifold::SetMinCircularAngle);
  function("setMinCircularEdgeLength", &Manifold::SetMinCircularEdgeLength);
  function("setCircularSegments", &Manifold::SetCircularSegments);
  function("getCircularSegments", &Manifold::GetCircularSegments);
}
