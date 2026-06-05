// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// CrossSection implementation backed by the Boolean2 algorithm. Public
// methods route through `boolean2/` headers for polygon boolean operations,
// containment grouping, offsets, and polygon utilities.

#include <algorithm>
#include <cassert>

#include "../utils.h"
#include "boolean2/boolean2.h"
#include "manifold/cross_section.h"

namespace b2 = manifold::boolean2;

using namespace manifold;

// Linker fingerprint for tests/fuzzers that require the boolean2 backend.
extern "C" int ManifoldCrossSectionBackendIsBoolean2() { return 1; }

namespace manifold {
struct PathImpl {
  PathImpl(Polygons paths) : paths_(std::move(paths)) {}
  operator const Polygons&() const { return paths_; }
  const Polygons paths_;
};
}  // namespace manifold

namespace {

Polygons FilterSmallContours(const Polygons& paths, double epsilon) {
  Polygons filtered;
  filtered.reserve(paths.size());
  for (const auto& path : paths) {
    const double area = b2::SignedArea(path);
    Rect box;
    for (const vec2& v : path) box.Union(v);
    const vec2 size = box.Size();
    if (std::fabs(area) > std::max(size.x, size.y) * epsilon) {
      filtered.push_back(path);
    }
  }
  return filtered;
}

b2::OffsetJoinType JoinTypeOf(CrossSection::JoinType jointype) {
  switch (jointype) {
    case CrossSection::JoinType::Square:
      return b2::OffsetJoinType::Square;
    case CrossSection::JoinType::Round:
      return b2::OffsetJoinType::Round;
    case CrossSection::JoinType::Miter:
      return b2::OffsetJoinType::Miter;
    case CrossSection::JoinType::Bevel:
      return b2::OffsetJoinType::Bevel;
  }
  return b2::OffsetJoinType::Square;
}

// ||M||_2; callers pass the linear part of an affine 2D transform.
double Mat2SpectralNorm(const mat2& m) {
  const vec2 c0 = m[0];
  const vec2 c1 = m[1];
  const double mtm00 = la::dot(c0, c0);
  const double mtm11 = la::dot(c1, c1);
  const double mtm01 = la::dot(c0, c1);
  const double trace = mtm00 + mtm11;
  const double det = mtm00 * mtm11 - mtm01 * mtm01;
  const double disc = std::sqrt(std::max(0.0, trace * trace - 4.0 * det));
  return std::sqrt(0.5 * (trace + disc));
}

Polygons TransformPolygons(const Polygons& paths, const mat2x3& transform) {
  const bool invert = la::determinant(mat2(transform)) < 0;
  Polygons out;
  out.reserve(paths.size());
  for (const auto& path : paths) {
    const size_t size = path.size();
    SimplePolygon transformed(size);
    for (size_t i = 0; i < size; ++i) {
      const size_t index = invert ? size - 1 - i : i;
      transformed[index] = transform * vec3(path[i].x, path[i].y, 1.0);
    }
    out.push_back(std::move(transformed));
  }
  return out;
}

std::shared_ptr<const PathImpl> shared_paths(Polygons paths) {
  return std::make_shared<const PathImpl>(std::move(paths));
}

void HullBacktrack(const vec2& point, std::vector<vec2>& stack) {
  size_t size = stack.size();
  while (size >= 2 &&
         CCW(stack[size - 2], stack[size - 1], point, 0.0) <= 0.0) {
    stack.pop_back();
    size = stack.size();
  }
}

SimplePolygon HullImpl(SimplePolygon& points) {
  if (points.size() < 3) return {};
  manifold::stable_sort(points.begin(), points.end(), [](vec2 a, vec2 b) {
    if (a.x == b.x) return a.y < b.y;
    return a.x < b.x;
  });

  std::vector<vec2> lower;
  for (const vec2& point : points) {
    HullBacktrack(point, lower);
    lower.push_back(point);
  }

  std::vector<vec2> upper;
  for (auto it = points.rbegin(); it != points.rend(); ++it) {
    HullBacktrack(*it, upper);
    upper.push_back(*it);
  }

  upper.pop_back();
  lower.pop_back();

  SimplePolygon path;
  path.reserve(lower.size() + upper.size());
  for (const vec2& point : lower) path.push_back(point);
  for (const vec2& point : upper) path.push_back(point);
  return path;
}

}  // namespace

namespace manifold {

CrossSection::CrossSection() : paths_(shared_paths({})) {}

CrossSection::~CrossSection() = default;

// Explicit move ops keep `other.pathsMutex_` intact so a subsequent copy-
// or move-assign into the moved-from object can still lock it.
CrossSection::CrossSection(CrossSection&& other) noexcept {
  std::lock_guard<std::mutex> lock(*other.pathsMutex_);
  paths_ = std::move(other.paths_);
  transform_ = other.transform_;
  tolerance_ = other.tolerance_;
}

CrossSection& CrossSection::operator=(CrossSection&& other) noexcept {
  if (this == &other) return *this;
  std::scoped_lock lock(*pathsMutex_, *other.pathsMutex_);
  paths_ = std::move(other.paths_);
  transform_ = other.transform_;
  tolerance_ = other.tolerance_;
  return *this;
}

CrossSection::CrossSection(const CrossSection& other) {
  std::lock_guard<std::mutex> lock(*other.pathsMutex_);
  paths_ = other.paths_;
  transform_ = other.transform_;
  tolerance_ = other.tolerance_;
}

CrossSection& CrossSection::operator=(const CrossSection& other) {
  if (this == &other) return *this;
  std::scoped_lock lock(*pathsMutex_, *other.pathsMutex_);
  paths_ = other.paths_;
  transform_ = other.transform_;
  tolerance_ = other.tolerance_;
  return *this;
}

CrossSection::CrossSection(std::shared_ptr<const PathImpl> paths)
    : paths_(std::move(paths)) {}

CrossSection::CrossSection(const SimplePolygon& contour, FillRule fillrule) {
  assert(fillrule == FillRule::Positive);
  (void)fillrule;
  Polygons input{contour};
  tolerance_ = b2::InferEps(input, {});
  paths_ = shared_paths(b2::Simplify(input, 0.0, tolerance_));
}

CrossSection::CrossSection(const Polygons& contours, FillRule fillrule) {
  assert(fillrule == FillRule::Positive);
  (void)fillrule;
  tolerance_ = b2::InferEps(contours, {});
  paths_ = shared_paths(b2::Simplify(contours, 0.0, tolerance_));
}

CrossSection::CrossSection(const Rect& rect) {
  if (rect.IsEmpty()) {
    paths_ = shared_paths({});
    return;
  }
  paths_ = shared_paths({{{rect.min.x, rect.min.y},
                          {rect.max.x, rect.min.y},
                          {rect.max.x, rect.max.y},
                          {rect.min.x, rect.max.y}}});
}

std::shared_ptr<const PathImpl> CrossSection::GetPaths() const {
  std::lock_guard<std::mutex> lock(*pathsMutex_);
  if (transform_ == mat2x3(la::identity)) return paths_;
  paths_ = shared_paths(TransformPolygons(paths_->paths_, transform_));
  transform_ = mat2x3(la::identity);
  return paths_;
}

CrossSection CrossSection::Square(const vec2 size, bool center) {
  if (size.x < 0.0 || size.y < 0.0 || la::length(size) == 0.0) {
    return CrossSection();
  }
  const vec2 half = size * 0.5;
  if (center) return CrossSection(Rect(-half, half));
  return CrossSection(Rect({0.0, 0.0}, size));
}

CrossSection CrossSection::Circle(double radius, int circularSegments) {
  if (radius <= 0.0) return CrossSection();
  const int n = circularSegments > 2 ? circularSegments
                                     : Quality::GetCircularSegments(radius);
  SimplePolygon circle(n);
  const double dPhi = 360.0 / n;
  for (int i = 0; i < n; ++i)
    circle[i] = {radius * cosd(dPhi * i), radius * sind(dPhi * i)};
  return CrossSection(shared_paths({std::move(circle)}));
}

CrossSection CrossSection::Boolean(const CrossSection& second,
                                   OpType op) const {
  // GetPaths() returns a snapshot aliasing each source's paths_, which stays
  // alive for this call, so bind by reference instead of deep-copying.
  const Polygons& a = GetPaths()->paths_;
  const Polygons& b = second.GetPaths()->paths_;
  const double eps = b2::InferEps(a, b);
  const double tolerance = std::max({tolerance_, second.tolerance_, eps});
  CrossSection result(shared_paths(b2::Boolean2D(a, b, op, eps, tolerance)));
  result.tolerance_ = tolerance;
  return result;
}

CrossSection CrossSection::BatchBoolean(
    const std::vector<CrossSection>& crossSections, OpType op) {
  if (crossSections.empty()) return CrossSection();
  if (crossSections.size() == 1) return crossSections[0];

  if (op == OpType::Intersect) {
    Polygons result = crossSections[0].GetPaths()->paths_;
    double tol = crossSections[0].tolerance_;
    for (size_t i = 1; i < crossSections.size(); ++i) {
      const auto& clip = crossSections[i].GetPaths()->paths_;
      const double eps = b2::InferEps(result, clip);
      const double tolerance =
          std::max({tol, crossSections[i].tolerance_, eps});
      result = b2::Boolean2D(result, clip, OpType::Intersect, eps, tolerance);
      tol = tolerance;
    }
    CrossSection out(shared_paths(std::move(result)));
    out.tolerance_ = tol;
    return out;
  }

  Polygons clips;
  double clipsTol = 0.0;
  for (size_t i = 1; i < crossSections.size(); ++i) {
    const auto& paths = crossSections[i].GetPaths()->paths_;
    clips.insert(clips.end(), paths.begin(), paths.end());
    clipsTol = std::max(clipsTol, crossSections[i].tolerance_);
  }
  const auto& subject = crossSections[0].GetPaths()->paths_;
  const double eps = b2::InferEps(subject, clips);
  const double tolerance =
      std::max({crossSections[0].tolerance_, clipsTol, eps});
  CrossSection out(
      shared_paths(b2::Boolean2D(subject, clips, op, eps, tolerance)));
  out.tolerance_ = tolerance;
  return out;
}

CrossSection CrossSection::operator+(const CrossSection& other) const {
  return Boolean(other, OpType::Add);
}

CrossSection& CrossSection::operator+=(const CrossSection& other) {
  *this = *this + other;
  return *this;
}

CrossSection CrossSection::operator-(const CrossSection& other) const {
  return Boolean(other, OpType::Subtract);
}

CrossSection& CrossSection::operator-=(const CrossSection& other) {
  *this = *this - other;
  return *this;
}

CrossSection CrossSection::operator^(const CrossSection& other) const {
  return Boolean(other, OpType::Intersect);
}

CrossSection& CrossSection::operator^=(const CrossSection& other) {
  *this = *this ^ other;
  return *this;
}

CrossSection CrossSection::Compose(
    const std::vector<CrossSection>& crossSections) {
  return BatchBoolean(crossSections, OpType::Add);
}

std::vector<CrossSection> CrossSection::Decompose() const {
  if (NumContour() < 2) return {*this};
  const auto& paths = GetPaths()->paths_;
  std::vector<Polygons> components = b2::DecomposeByContainment(paths);
  std::vector<CrossSection> out;
  out.reserve(components.size());
  for (auto& component : components) {
    CrossSection piece(shared_paths(std::move(component)));
    piece.tolerance_ = tolerance_;
    out.push_back(std::move(piece));
  }
  return out;
}

CrossSection CrossSection::Translate(const vec2 v) const {
  mat2x3 m({1.0, 0.0}, {0.0, 1.0}, {v.x, v.y});
  return Transform(m);
}

CrossSection CrossSection::Rotate(double degrees) const {
  const double s = sind(degrees);
  const double c = cosd(degrees);
  mat2x3 m({c, s}, {-s, c}, {0.0, 0.0});
  return Transform(m);
}

CrossSection CrossSection::Scale(const vec2 scale) const {
  mat2x3 m({scale.x, 0.0}, {0.0, scale.y}, {0.0, 0.0});
  return Transform(m);
}

CrossSection CrossSection::Mirror(const vec2 ax) const {
  if (la::length(ax) == 0.0) return CrossSection();
  const vec2 n = la::normalize(ax);
  return Transform(
      mat2x3(mat2(la::identity) - 2.0 * la::outerprod(n, n), vec2(0.0)));
}

CrossSection CrossSection::Transform(const mat2x3& m) const {
  std::lock_guard<std::mutex> lock(*pathsMutex_);
  CrossSection transformed;
  transformed.transform_ = m * Mat3(transform_);
  transformed.paths_ = paths_;
  // Floor at the translated scale so large translations don't leave
  // tolerance below the post-materialization FP-noise bound.
  const double translationScale =
      std::max(std::fabs(m[2][0]), std::fabs(m[2][1]));
  transformed.tolerance_ =
      std::max(Mat2SpectralNorm(mat2(m[0], m[1])) * tolerance_,
               b2::EpsilonFromScale(translationScale));
  return transformed;
}

CrossSection CrossSection::Warp(std::function<void(vec2&)> warpFunc) const {
  return WarpBatch([&warpFunc](VecView<vec2> points) {
    for (vec2& point : points) warpFunc(point);
  });
}

CrossSection CrossSection::WarpBatch(
    std::function<void(VecView<vec2>)> warpFunc) const {
  Polygons paths = GetPaths()->paths_;
  std::vector<vec2> points;
  for (const auto& path : paths) {
    points.insert(points.end(), path.begin(), path.end());
  }
  warpFunc(VecView<vec2>(points.data(), points.size()));
  auto point = points.begin();
  for (auto& path : paths) {
    for (auto& v : path) v = *point++;
  }
  const double eps = b2::InferEps(paths, {});
  const double tolerance = std::max(tolerance_, eps);
  CrossSection out(shared_paths(b2::Simplify(paths, eps, tolerance)));
  out.tolerance_ = tolerance;
  return out;
}

CrossSection CrossSection::Simplify(double epsilon) const {
  const auto& paths = GetPaths()->paths_;
  const Polygons filtered = FilterSmallContours(paths, epsilon);
  const double tolerance = std::max(tolerance_, epsilon);
  CrossSection out(shared_paths(FilterSmallContours(
      b2::Simplify(filtered, epsilon, tolerance), epsilon)));
  out.tolerance_ = tolerance;
  return out;
}

CrossSection CrossSection::Offset(double delta, JoinType jt, double miterLimit,
                                  int circularSegments) const {
  Polygons offset = b2::Offset(GetPaths()->paths_, delta, JoinTypeOf(jt),
                               miterLimit, circularSegments, tolerance_);
  CrossSection out(shared_paths(std::move(offset)));
  // Round-join faceting comes from circularSegments inside b2::Offset; it is
  // not folded into tolerance_. The faceting sagitta scales with delta and is a
  // quality choice, not a drift budget, so folding it would over-merge features
  // in every downstream op. Max, not sum, keeps chained Offset bounded.
  out.tolerance_ =
      std::max(tolerance_, b2::InferEps(out.GetPaths()->paths_, {}));
  return out;
}

CrossSection CrossSection::Hull(
    const std::vector<CrossSection>& crossSections) {
  size_t numPoints = 0;
  double maxTol = 0.0;
  for (const auto& cs : crossSections) {
    numPoints += cs.NumVert();
    maxTol = std::max(maxTol, cs.tolerance_);
  }

  SimplePolygon points;
  points.reserve(numPoints);
  for (const auto& cs : crossSections) {
    const auto& paths = cs.GetPaths()->paths_;
    for (const auto& path : paths)
      for (const vec2& point : path) points.push_back(point);
  }

  SimplePolygon hull = HullImpl(points);
  if (hull.empty()) return CrossSection();
  CrossSection out(shared_paths({std::move(hull)}));
  out.tolerance_ = std::max(maxTol, b2::InferEps(out.GetPaths()->paths_, {}));
  return out;
}

CrossSection CrossSection::Hull() const { return Hull({*this}); }

CrossSection CrossSection::Hull(SimplePolygon pts) {
  SimplePolygon hull = HullImpl(pts);
  if (hull.empty()) return CrossSection();
  Polygons inputForEps{pts};
  CrossSection out(shared_paths({std::move(hull)}));
  out.tolerance_ = b2::InferEps(inputForEps, {});
  return out;
}

CrossSection CrossSection::Hull(const Polygons polys) {
  SimplePolygon points;
  for (const auto& path : polys)
    for (const vec2& point : path) points.push_back(point);
  return Hull(points);
}

double CrossSection::Area() const {
  return b2::TotalSignedArea(GetPaths()->paths_);
}

size_t CrossSection::NumVert() const {
  size_t numVert = 0;
  for (const auto& path : GetPaths()->paths_) numVert += path.size();
  return numVert;
}

size_t CrossSection::NumContour() const { return GetPaths()->paths_.size(); }

bool CrossSection::IsEmpty() const { return GetPaths()->paths_.empty(); }

Rect CrossSection::Bounds() const {
  Rect box;
  for (const auto& path : GetPaths()->paths_) {
    for (const vec2& v : path) box.Union(v);
  }
  return box;
}

Polygons CrossSection::ToPolygons() const { return GetPaths()->paths_; }

}  // namespace manifold
