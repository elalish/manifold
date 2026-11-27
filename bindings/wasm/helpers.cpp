#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <type_traits>
#include <vector>

#include "manifold/cross_section.h"
#include "manifold/manifold.h"
#include "manifold/polygon.h"

using namespace emscripten;
using namespace manifold;

namespace js {
// JS indices are always Uint32Array: wasm32 memory cannot hold more verts.
template <typename T>
val ToJS(const std::vector<T>& v) {
  if constexpr (std::is_same_v<T, uint64_t>) {
    return ToJS(std::vector<uint32_t>(v.begin(), v.end()));
  } else {
    return val(typed_memory_view(v.size(), v.data())).call<val>("slice");
  }
}

// Leaves out unchanged if v is undefined.
template <typename T>
void FromJS(std::vector<T>& out, const val& v) {
  if (v == val::undefined()) return;
  if constexpr (std::is_same_v<T, uint64_t>) {
    const auto u32 = convertJSArrayToNumberVector<uint32_t>(v);
    out.assign(u32.begin(), u32.end());
  } else {
    out = convertJSArrayToNumberVector<T>(v);
  }
}

template <typename P, typename I>
val MeshGL2JS(const MeshGLP<P, I>& mesh) {
  val meshJS = val::object();
  meshJS.set("numProp", static_cast<uint32_t>(mesh.numProp));
  meshJS.set("triVerts", ToJS(mesh.triVerts));
  meshJS.set("vertProperties", ToJS(mesh.vertProperties));
  meshJS.set("mergeFromVert", ToJS(mesh.mergeFromVert));
  meshJS.set("mergeToVert", ToJS(mesh.mergeToVert));
  meshJS.set("runIndex", ToJS(mesh.runIndex));
  meshJS.set("runOriginalID", ToJS(mesh.runOriginalID));
  meshJS.set("faceID", ToJS(mesh.faceID));
  meshJS.set("halfedgeTangent", ToJS(mesh.halfedgeTangent));
  meshJS.set("runTransform", ToJS(mesh.runTransform));
  meshJS.set("runFlags", ToJS(mesh.runFlags));
  return meshJS;
}

template <typename P, typename I>
MeshGLP<P, I> MeshJS2GL(const val& mesh) {
  MeshGLP<P, I> out;
  out.numProp = mesh["numProp"].as<uint32_t>();
  FromJS(out.triVerts, mesh["triVerts"]);
  FromJS(out.vertProperties, mesh["vertProperties"]);
  FromJS(out.mergeFromVert, mesh["mergeFromVert"]);
  FromJS(out.mergeToVert, mesh["mergeToVert"]);
  FromJS(out.runIndex, mesh["runIndex"]);
  FromJS(out.runOriginalID, mesh["runOriginalID"]);
  FromJS(out.faceID, mesh["faceID"]);
  FromJS(out.halfedgeTangent, mesh["halfedgeTangent"]);
  FromJS(out.runTransform, mesh["runTransform"]);
  FromJS(out.runFlags, mesh["runFlags"]);
  return out;
}

val GetMeshJS(const Manifold& manifold) {
  return MeshGL2JS(manifold.GetMeshGL());
}

val GetMesh64JS(const Manifold& manifold) {
  return MeshGL2JS(manifold.GetMeshGL64());
}

template <typename P, typename I>
val Merge(const val& mesh, P tolerance) {
  return MeshGL2JS(MeshJS2GL<P, I>(mesh).Merge(tolerance));
}

CircularQuality DefaultCircularQuality() { return {}; }

int SegmentsFor(CircularQuality quality, double radius) {
  return quality.SegmentsFor(radius);
}

ivec3 Version() {
  const auto v = manifold::Version();
  return {v[0], v[1], v[2]};
}
}  // namespace js

namespace cross_js {
CrossSection OfPolygons(std::vector<std::vector<vec2>> polygons) {
  return CrossSection(polygons);
}

CrossSection EvenOdd(std::vector<std::vector<vec2>> polygons) {
  return CrossSection::EvenOdd(polygons);
}

CrossSection NonZero(std::vector<std::vector<vec2>> polygons) {
  return CrossSection::NonZero(polygons);
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

CrossSection WarpBatch(CrossSection& cross_section, uintptr_t funcPtr) {
  void (*f)(uintptr_t, size_t) =
      reinterpret_cast<void (*)(uintptr_t, size_t)>(funcPtr);

  return cross_section.WarpBatch([&](manifold::VecView<manifold::vec2> vecs) {
    const size_t n = vecs.size();

    // Flatten
    std::vector<double> flat;
    flat.resize(n * 2);
    for (size_t i = 0; i < n; ++i) {
      const manifold::vec2& v = vecs[i];
      flat[i * 2 + 0] = v[0];
      flat[i * 2 + 1] = v[1];
    }

    f(reinterpret_cast<uintptr_t>(flat.data()), n);

    // Copy back to vecs
    for (size_t i = 0; i < n; ++i) {
      manifold::vec2& v = vecs[i];
      v[0] = flat[i * 2 + 0];
      v[1] = flat[i * 2 + 1];
    }
  });
}

CrossSection Offset(CrossSection& cross_section, double delta, int join_type,
                    double miter_limit, CircularQuality quality) {
  auto jt = join_type == 0   ? CrossSection::JoinType::Square
            : join_type == 1 ? CrossSection::JoinType::Round
            : join_type == 2 ? CrossSection::JoinType::Miter
                             : CrossSection::JoinType::Bevel;
  return cross_section.Offset(delta, jt, miter_limit, quality);
}

void CollectVertices(std::vector<vec2>& verts, const CrossSection& cs) {
  auto polys = cs.ToPolygons();
  for (auto poly : polys) verts.insert(verts.end(), poly.begin(), poly.end());
}
}  // namespace cross_js

namespace man_js {
template <typename P, typename I>
Manifold FromMeshJS(const val& mesh) {
  return Manifold(js::MeshJS2GL<P, I>(mesh));
}

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

Manifold WarpBatch(Manifold& manifold, uintptr_t funcPtr) {
  void (*f)(uintptr_t, size_t) =
      reinterpret_cast<void (*)(uintptr_t, size_t)>(funcPtr);

  return manifold.WarpBatch([&](manifold::VecView<manifold::vec3> vecs) {
    const size_t n = vecs.size();

    // Flatten
    std::vector<double> flat;
    flat.resize(n * 3);
    for (size_t i = 0; i < n; ++i) {
      const manifold::vec3& v = vecs[i];
      flat[i * 3 + 0] = v[0];
      flat[i * 3 + 1] = v[1];
      flat[i * 3 + 2] = v[2];
    }

    f(reinterpret_cast<uintptr_t>(flat.data()), n);

    // Copy back to vecs
    for (size_t i = 0; i < n; ++i) {
      manifold::vec3& v = vecs[i];
      v[0] = flat[i * 3 + 0];
      v[1] = flat[i * 3 + 1];
      v[2] = flat[i * 3 + 2];
    }
  });
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

// ctx-aware static factories: like the plain factories above, but run under an
// ExecutionContext so progress / cancellation are observed (these ops have no
// source Manifold to attach via withContext). Bound as ExecutionContext methods
// (first arg is the receiver).
template <typename P, typename I>
Manifold ExecutionContextFromMesh(ExecutionContext& ctx, const val& mesh) {
  return ctx.FromMeshGL(js::MeshJS2GL<P, I>(mesh));
}

Manifold ExecutionContextLevelSet(ExecutionContext& ctx, uintptr_t funcPtr,
                                  Box bounds, double edgeLength, double level,
                                  double tolerance) {
  double (*f)(const vec3&) = reinterpret_cast<double (*)(const vec3&)>(funcPtr);
  return ctx.LevelSet(f, bounds, edgeLength, level, tolerance, false);
}

std::string Status(Manifold& manifold) { return ErrorName(manifold.Status()); }

std::vector<Manifold> Split(Manifold& a, Manifold& b) {
  auto [r1, r2] = a.Split(b);
  return {r1, r2};
}

std::vector<Manifold> SplitByPlane(Manifold& m, vec3 normal,
                                   double originOffset) {
  auto [a, b] = m.SplitByPlane(normal, originOffset);
  return {a, b};
}

std::vector<RayHit> RayCast(const Manifold& m, vec3 origin, vec3 endpoint) {
  return m.RayCast(origin, endpoint);
}

void CollectVertices(std::vector<vec3>& verts, const Manifold& manifold) {
  const MeshGL64 mesh = manifold.GetMeshGL64();
  const auto numVert = mesh.NumVert();
  for (size_t v = 0; v < numVert; ++v) {
    verts.push_back(mesh.GetVertPos(v));
  }
}
}  // namespace man_js
