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

#include <vector>

#include "helpers.cpp"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"
#include "manifold/polygon.h"

#if (MANIFOLD_PAR == 1)
#include <tbb/parallel_for.h>

#include <atomic>
#endif

// https://github.com/oneapi-src/oneTBB/blob/master/WASM_Support.md#limitations
void initTBB() {
#if (MANIFOLD_PAR == 1)
  int num_threads = tbb::this_task_arena::max_concurrency();
  std::atomic<int> barrier{num_threads};
  tbb::parallel_for(
      0, num_threads,
      [&barrier](int) {
        barrier--;
        while (barrier > 0) {
          // Send browser thread to event loop
          std::this_thread::yield();
        }
      },
      tbb::static_partitioner{});
#endif
}

using namespace emscripten;
using namespace manifold;

EMSCRIPTEN_BINDINGS(whatever) {
  value_object<vec2>("vec2").field("x", &vec2::x).field("y", &vec2::y);

  value_object<ivec3>("ivec3")
      .field("0", &ivec3::x)
      .field("1", &ivec3::y)
      .field("2", &ivec3::z);

  value_object<vec3>("vec3")
      .field("x", &vec3::x)
      .field("y", &vec3::y)
      .field("z", &vec3::z);

  value_object<vec4>("vec4")
      .field("x", &vec4::x)
      .field("y", &vec4::y)
      .field("z", &vec4::z)
      .field("w", &vec4::w);

  enum_<Manifold::Error>("status")
      .value("NoError", Manifold::Error::NoError)
      .value("NonFiniteVertex", Manifold::Error::NonFiniteVertex)
      .value("NotManifold", Manifold::Error::NotManifold)
      .value("VertexOutOfBounds", Manifold::Error::VertexOutOfBounds)
      .value("PropertiesWrongLength", Manifold::Error::PropertiesWrongLength)
      .value("MissingPositionProperties",
             Manifold::Error::MissingPositionProperties)
      .value("MergeVectorsDifferentLengths",
             Manifold::Error::MergeVectorsDifferentLengths)
      .value("MergeIndexOutOfBounds", Manifold::Error::MergeIndexOutOfBounds)
      .value("TransformWrongLength", Manifold::Error::TransformWrongLength)
      .value("RunIndexWrongLength", Manifold::Error::RunIndexWrongLength)
      .value("FaceIDWrongLength", Manifold::Error::FaceIDWrongLength)
      .value("InvalidConstruction", Manifold::Error::InvalidConstruction);

  enum_<CrossSection::FillRule>("fillrule")
      .value("EvenOdd", CrossSection::FillRule::EvenOdd)
      .value("NonZero", CrossSection::FillRule::NonZero)
      .value("Positive", CrossSection::FillRule::Positive)
      .value("Negative", CrossSection::FillRule::Negative);

  enum_<CrossSection::JoinType>("jointype")
      .value("Square", CrossSection::JoinType::Square)
      .value("Round", CrossSection::JoinType::Round)
      .value("Miter", CrossSection::JoinType::Miter);

  value_object<Rect>("rect").field("min", &Rect::min).field("max", &Rect::max);
  value_object<Box>("box").field("min", &Box::min).field("max", &Box::max);

  value_object<Smoothness>("smoothness")
      .field("halfedge", &Smoothness::halfedge)
      .field("smoothness", &Smoothness::smoothness);

  register_vector<ivec3>("Vector_ivec3");
  register_vector<vec3>("Vector_vec3");
  register_vector<vec2>("Vector_vec2");
  register_vector<std::vector<vec2>>("Vector2_vec2");
  register_vector<double>("Vector_f64");
  register_vector<CrossSection>("Vector_crossSection");
  register_vector<Manifold>("Vector_manifold");
  register_vector<Smoothness>("Vector_smoothness");
  register_vector<vec4>("Vector_vec4");

  class_<CrossSection>("CrossSection")
      .constructor(&cross_js::OfPolygons)
      .function("_add", &cross_js::Union)
      .function("_subtract", &cross_js::Difference)
      .function("_intersect", &cross_js::Intersection)
      .function("_Warp", &cross_js::Warp)
      .function("transform", &cross_js::Transform)
      .function("_Translate", &CrossSection::Translate)
      .function("rotate", &CrossSection::Rotate)
      .function("_Scale", &CrossSection::Scale)
      .function("_Mirror", &CrossSection::Mirror)
      .function("_Decompose", &CrossSection::Decompose)
      .function("isEmpty", &CrossSection::IsEmpty)
      .function("area", &CrossSection::Area)
      .function("numVert", &CrossSection::NumVert)
      .function("numContour", &CrossSection::NumContour)
      .function("_Bounds", &CrossSection::Bounds)
      .function("simplify", &CrossSection::Simplify)
      .function("_Offset", &cross_js::Offset)
      .function("_ToPolygons", &CrossSection::ToPolygons)
      .function("hull",
                select_overload<CrossSection() const>(&CrossSection::Hull));

  // CrossSection Static Methods
  function("_Square", &CrossSection::Square);
  function("_Circle", &CrossSection::Circle);
  function("_crossSectionCompose", &CrossSection::Compose);
  function("_crossSectionUnionN", &cross_js::UnionN);
  function("_crossSectionDifferenceN", &cross_js::DifferenceN);
  function("_crossSectionIntersectionN", &cross_js::IntersectionN);
  function("_crossSectionCollectVertices", &cross_js::CollectVertices);
  function(
      "_crossSectionHullPoints",
      select_overload<CrossSection(std::vector<vec2>)>(&CrossSection::Hull));

  class_<Manifold>("Manifold")
      .constructor(&man_js::FromMeshJS)
      .function("add", &man_js::Union)
      .function("subtract", &man_js::Difference)
      .function("intersect", &man_js::Intersection)
      .function("_Split", &man_js::Split)
      .function("_SplitByPlane", &man_js::SplitByPlane)
      .function("_TrimByPlane", &Manifold::TrimByPlane)
      .function("_Slice", &Manifold::Slice)
      .function("_Project", &Manifold::Project)
      .function("hull", select_overload<Manifold() const>(&Manifold::Hull))
      .function("_GetMeshJS", &js::GetMeshJS)
      .function("refine", &Manifold::Refine)
      .function("refineToLength", &Manifold::RefineToLength)
      .function("refineToTolerance", &Manifold::RefineToTolerance)
      .function("smoothByNormals", &Manifold::SmoothByNormals)
      .function("_SmoothOut", &Manifold::SmoothOut)
      .function("_Warp", &man_js::Warp)
      .function("_SetProperties", &man_js::SetProperties)
      .function("transform", &man_js::Transform)
      .function("_Translate", &Manifold::Translate)
      .function("_Rotate", &Manifold::Rotate)
      .function("_Scale", &Manifold::Scale)
      .function("_Mirror", &Manifold::Mirror)
      .function("_Decompose", select_overload<std::vector<Manifold>() const>(
                                  &Manifold::Decompose))
      .function("isEmpty", &Manifold::IsEmpty)
      .function("status", &Manifold::Status)
      .function("numVert", &Manifold::NumVert)
      .function("numEdge", &Manifold::NumEdge)
      .function("numTri", &Manifold::NumTri)
      .function("numProp", &Manifold::NumProp)
      .function("numPropVert", &Manifold::NumPropVert)
      .function("_boundingBox", &Manifold::BoundingBox)
      .function("tolerance", &Manifold::GetTolerance)
      .function("setTolerance", &Manifold::SetTolerance)
      .function("genus", &Manifold::Genus)
      .function("volume", &Manifold::Volume)
      .function("surfaceArea", &Manifold::SurfaceArea)
      .function("minGap", &Manifold::MinGap)
      .function("calculateCurvature", &Manifold::CalculateCurvature)
      .function("_CalculateNormals", &Manifold::CalculateNormals)
      .function("originalID", &Manifold::OriginalID)
      .function("asOriginal", &Manifold::AsOriginal);

  // Manifold Static Methods
  function("_Cube", &Manifold::Cube);
  function("_Cylinder", &Manifold::Cylinder);
  function("_Sphere", &Manifold::Sphere);
  function("_Tetrahedron", &Manifold::Tetrahedron);
  function("_Smooth", &js::Smooth);
  function("_Extrude", &Manifold::Extrude);
  function("_Triangulate", &Triangulate);
  function("_Revolve", &Manifold::Revolve);
  function("_LevelSet", &man_js::LevelSet);
  function("_Merge", &js::Merge);
  function("_ReserveIDs", &Manifold::ReserveIDs);
  function("_manifoldCompose", &Manifold::Compose);
  function("_manifoldUnionN", &man_js::UnionN);
  function("_manifoldDifferenceN", &man_js::DifferenceN);
  function("_manifoldIntersectionN", &man_js::IntersectionN);
  function("_manifoldCollectVertices", &man_js::CollectVertices);
  function(
      "_manifoldHullPoints",
      select_overload<Manifold(const std::vector<vec3>&)>(&Manifold::Hull));

  // Quality Globals
  function("setMinCircularAngle", &Quality::SetMinCircularAngle);
  function("setMinCircularEdgeLength", &Quality::SetMinCircularEdgeLength);
  function("setCircularSegments", &Quality::SetCircularSegments);
  function("getCircularSegments", &Quality::GetCircularSegments);
  function("resetToCircularDefaults", &Quality::ResetToDefaults);

  function("initTBB", &initTBB);
}
