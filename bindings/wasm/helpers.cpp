#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <vector>

#include "manifold/cross_section.h"
#include "manifold/manifold.h"
#include "manifold/polygon.h"

using namespace emscripten;
using namespace manifold;

namespace js {
val MeshGL2JS(const MeshGL& mesh) {
  val meshJS = val::object();

  meshJS.set("numProp", mesh.numProp);
  meshJS.set("triVerts",
             val(typed_memory_view(mesh.triVerts.size(), mesh.triVerts.data()))
                 .call<val>("slice"));
  meshJS.set("vertProperties",
             val(typed_memory_view(mesh.vertProperties.size(),
                                   mesh.vertProperties.data()))
                 .call<val>("slice"));
  meshJS.set("mergeFromVert", val(typed_memory_view(mesh.mergeFromVert.size(),
                                                    mesh.mergeFromVert.data()))
                                  .call<val>("slice"));
  meshJS.set("mergeToVert", val(typed_memory_view(mesh.mergeToVert.size(),
                                                  mesh.mergeToVert.data()))
                                .call<val>("slice"));
  meshJS.set("runIndex",
             val(typed_memory_view(mesh.runIndex.size(), mesh.runIndex.data()))
                 .call<val>("slice"));
  meshJS.set("runOriginalID", val(typed_memory_view(mesh.runOriginalID.size(),
                                                    mesh.runOriginalID.data()))
                                  .call<val>("slice"));
  meshJS.set("faceID",
             val(typed_memory_view(mesh.faceID.size(), mesh.faceID.data()))
                 .call<val>("slice"));
  meshJS.set("halfedgeTangent",
             val(typed_memory_view(mesh.halfedgeTangent.size(),
                                   mesh.halfedgeTangent.data()))
                 .call<val>("slice"));
  meshJS.set("runTransform", val(typed_memory_view(mesh.runTransform.size(),
                                                   mesh.runTransform.data()))
                                 .call<val>("slice"));

  return meshJS;
}

MeshGL MeshJS2GL(const val& mesh) {
  MeshGL out;
  out.numProp = mesh["numProp"].as<int>();
  out.triVerts = convertJSArrayToNumberVector<uint32_t>(mesh["triVerts"]);
  out.vertProperties =
      convertJSArrayToNumberVector<float>(mesh["vertProperties"]);
  if (mesh["mergeFromVert"] != val::undefined()) {
    out.mergeFromVert =
        convertJSArrayToNumberVector<uint32_t>(mesh["mergeFromVert"]);
  }
  if (mesh["mergeToVert"] != val::undefined()) {
    out.mergeToVert =
        convertJSArrayToNumberVector<uint32_t>(mesh["mergeToVert"]);
  }
  if (mesh["runIndex"] != val::undefined()) {
    out.runIndex = convertJSArrayToNumberVector<uint32_t>(mesh["runIndex"]);
  }
  if (mesh["runOriginalID"] != val::undefined()) {
    out.runOriginalID =
        convertJSArrayToNumberVector<uint32_t>(mesh["runOriginalID"]);
  }
  if (mesh["faceID"] != val::undefined()) {
    out.faceID = convertJSArrayToNumberVector<uint32_t>(mesh["faceID"]);
  }
  if (mesh["halfedgeTangent"] != val::undefined()) {
    out.halfedgeTangent =
        convertJSArrayToNumberVector<float>(mesh["halfedgeTangent"]);
  }
  if (mesh["runTransform"] != val::undefined()) {
    out.runTransform =
        convertJSArrayToNumberVector<float>(mesh["runTransform"]);
  }
  return out;
}

val GetMeshJS(const Manifold& manifold, int normalIdx) {
  MeshGL mesh = manifold.GetMeshGL(normalIdx);
  return MeshGL2JS(mesh);
}

val Merge(const val& mesh) {
  val out = val::object();
  MeshGL meshGL = MeshJS2GL(mesh);
  bool changed = meshGL.Merge();
  out.set("changed", changed);
  out.set("mesh", changed ? MeshGL2JS(meshGL) : mesh);
  return out;
}

Manifold Smooth(const val& mesh,
                const std::vector<Smoothness>& sharpenedEdges = {}) {
  return Manifold::Smooth(MeshJS2GL(mesh), sharpenedEdges);
}

}  // namespace js

namespace cross_js {
CrossSection OfPolygons(std::vector<std::vector<vec2>> polygons,
                        int fill_rule) {
  auto fr = fill_rule == 0   ? CrossSection::FillRule::EvenOdd
            : fill_rule == 1 ? CrossSection::FillRule::NonZero
            : fill_rule == 2 ? CrossSection::FillRule::Positive
                             : CrossSection::FillRule::Negative;
  return CrossSection(polygons, fr);
}

CrossSection Union(const CrossSection& a, const CrossSection& b) {
  return a + b;
}

CrossSection Difference(const CrossSection& a, const CrossSection& b) {
  return a - b;
}

CrossSection Intersection(const CrossSection& a, const CrossSection& b) {
  return a ^ b;
}

CrossSection UnionN(const std::vector<CrossSection>& cross_sections) {
  return CrossSection::BatchBoolean(cross_sections, OpType::Add);
}

CrossSection DifferenceN(const std::vector<CrossSection>& cross_sections) {
  return CrossSection::BatchBoolean(cross_sections, OpType::Subtract);
}

CrossSection IntersectionN(const std::vector<CrossSection>& cross_sections) {
  return CrossSection::BatchBoolean(cross_sections, OpType::Intersect);
}

CrossSection Transform(CrossSection& cross_section, const val& mat) {
  std::vector<double> array = convertJSArrayToNumberVector<double>(mat);
  mat2x3 matrix;
  for (const int col : {0, 1, 2})
    for (const int row : {0, 1}) matrix[col][row] = array[col * 3 + row];
  return cross_section.Transform(matrix);
}

CrossSection Warp(CrossSection& cross_section, uintptr_t funcPtr) {
  void (*f)(vec2&) = reinterpret_cast<void (*)(vec2&)>(funcPtr);
  return cross_section.Warp(f);
}

CrossSection Offset(CrossSection& cross_section, double delta, int join_type,
                    double miter_limit, double arc_tolerance) {
  auto jt = join_type == 0   ? CrossSection::JoinType::Square
            : join_type == 1 ? CrossSection::JoinType::Round
            : join_type == 2 ? CrossSection::JoinType::Miter
                             : CrossSection::JoinType::Bevel;
  return cross_section.Offset(delta, jt, miter_limit, arc_tolerance);
}

void CollectVertices(std::vector<vec2>& verts, const CrossSection& cs) {
  auto polys = cs.ToPolygons();
  for (auto poly : polys) verts.insert(verts.end(), poly.begin(), poly.end());
}
}  // namespace cross_js

namespace man_js {
Manifold FromMeshJS(const val& mesh) { return Manifold(js::MeshJS2GL(mesh)); }

Manifold Union(const Manifold& a, const Manifold& b) { return a + b; }

Manifold Difference(const Manifold& a, const Manifold& b) { return a - b; }

Manifold Intersection(const Manifold& a, const Manifold& b) { return a ^ b; }

Manifold UnionN(const std::vector<Manifold>& manifolds) {
  return Manifold::BatchBoolean(manifolds, OpType::Add);
}

Manifold DifferenceN(const std::vector<Manifold>& manifolds) {
  return Manifold::BatchBoolean(manifolds, OpType::Subtract);
}

Manifold IntersectionN(const std::vector<Manifold>& manifolds) {
  return Manifold::BatchBoolean(manifolds, OpType::Intersect);
}

Manifold Transform(Manifold& manifold, const val& mat) {
  std::vector<double> array = convertJSArrayToNumberVector<double>(mat);
  mat3x4 matrix;
  for (const int col : {0, 1, 2, 3})
    for (const int row : {0, 1, 2}) matrix[col][row] = array[col * 4 + row];
  return manifold.Transform(matrix);
}

Manifold Warp(Manifold& manifold, uintptr_t funcPtr) {
  void (*f)(vec3&) = reinterpret_cast<void (*)(vec3&)>(funcPtr);
  return manifold.Warp(f);
}

Manifold SetProperties(Manifold& manifold, int numProp, uintptr_t funcPtr) {
  void (*f)(double*, vec3, const double*) =
      reinterpret_cast<void (*)(double*, vec3, const double*)>(funcPtr);
  return manifold.SetProperties(numProp, f);
}

Manifold LevelSet(uintptr_t funcPtr, Box bounds, double edgeLength,
                  double level, double tolerance) {
  double (*f)(const vec3&) = reinterpret_cast<double (*)(const vec3&)>(funcPtr);
  return Manifold::LevelSet(f, bounds, edgeLength, level, tolerance, false);
}

std::string Status(Manifold& manifold) {
  switch (manifold.Status()) {
    case Manifold::Error::NoError:
      return "NoError";
    case Manifold::Error::NonFiniteVertex:
      return "NonFiniteVertex";
    case Manifold::Error::NotManifold:
      return "NotManifold";
    case Manifold::Error::VertexOutOfBounds:
      return "VertexOutOfBounds";
    case Manifold::Error::PropertiesWrongLength:
      return "PropertiesWrongLength";
    case Manifold::Error::MissingPositionProperties:
      return "MissingPositionProperties";
    case Manifold::Error::MergeVectorsDifferentLengths:
      return "MergeVectorsDifferentLengths";
    case Manifold::Error::MergeIndexOutOfBounds:
      return "MergeIndexOutOfBounds";
    case Manifold::Error::TransformWrongLength:
      return "TransformWrongLength";
    case Manifold::Error::RunIndexWrongLength:
      return "RunIndexWrongLength";
    case Manifold::Error::FaceIDWrongLength:
      return "FaceIDWrongLength";
    case Manifold::Error::InvalidConstruction:
      return "InvalidConstruction";
    default:
      return "UnknownError";
  }
}

std::vector<Manifold> Split(Manifold& a, Manifold& b) {
  auto [r1, r2] = a.Split(b);
  return {r1, r2};
}

std::vector<Manifold> SplitByPlane(Manifold& m, vec3 normal,
                                   double originOffset) {
  auto [a, b] = m.SplitByPlane(normal, originOffset);
  return {a, b};
}

void CollectVertices(std::vector<vec3>& verts, const Manifold& manifold) {
  const MeshGL64 mesh = manifold.GetMeshGL64();
  const auto numVert = mesh.NumVert();
  for (size_t v = 0; v < numVert; ++v) {
    verts.push_back(mesh.GetVertPos(v));
  }
}
}  // namespace man_js
