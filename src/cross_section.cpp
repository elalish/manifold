// Copyright 2023 The Manifold Authors.
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

#include "manifold/cross_section.h"

#include <algorithm>
#include <cmath>
#include <queue>

#include "boolean2.h"
#include "manifold/optional_assert.h"
#include "svd.h"
#include "utils.h"

// CrossSection implementation. Public methods route through `boolean2/` for
// polygon boolean operations, containment grouping, offsets, and polygon
// utilities.
using namespace manifold;

namespace manifold {
struct PathImpl {
  PathImpl(Polygons paths) : paths_(std::move(paths)) {}
  operator const Polygons&() const { return paths_; }
  const Polygons paths_;
};
}  // namespace manifold

namespace {

bool AllFinite(const mat2x3& m) {
  for (const int c : {0, 1, 2}) {
    if (!la::all(la::isfinite(m[c]))) return false;
  }
  return true;
}

Polygons FilterSmallContours(const Polygons& paths, double epsilon) {
  Polygons filtered;
  filtered.reserve(paths.size());
  for (const auto& path : paths) {
    const double area = SignedArea(path);
    Rect box;
    for (const vec2& v : path) box.Union(v);
    const vec2 size = box.Size();
    if (std::fabs(area) > std::max(size.x, size.y) * epsilon) {
      filtered.push_back(path);
    }
  }
  return filtered;
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
  for (const vec2& p : points) {
    if (!la::all(la::isfinite(p))) return {};
  }
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

// Drop the vertex closest to the line through its current neighbors, repeating
// while under tol (Clipper2 SimplifyPaths semantics). Re-checking current, not
// original, neighbors each step is what keeps a curve from collapsing. A
// min-heap over (deviation, index) with stamp-based lazy invalidation makes
// each removal re-key only the two neighbors, so the pass is O(n log n).
// Exactly-tied deviations remove the lowest original index first.
SimplePolygon SimplifyRing(SimplePolygon ring, double tol) {
  const int n = static_cast<int>(ring.size());
  if (n <= 3) return ring;
  const double tol2 = tol * tol;

  std::vector<int> prev(n), next(n), stamp(n, 0);
  std::vector<char> alive(n, 1);
  for (int i = 0; i < n; ++i) {
    prev[i] = (i + n - 1) % n;
    next[i] = (i + 1) % n;
  }

  auto deviation2 = [&](int i) {
    const vec2 P = ring[prev[i]];
    const vec2 N = ring[next[i]];
    const vec2 pn = N - P;
    const double pnLen2 = la::dot(pn, pn);
    const double cross = la::cross(ring[i] - P, pn);
    return pnLen2 > 0.0 ? cross * cross / pnLen2 : 0.0;
  };

  struct Entry {
    double d2;
    int stamp;
    int idx;
  };
  auto worse = [](const Entry& a, const Entry& b) {
    return a.d2 > b.d2 || (a.d2 == b.d2 && a.idx > b.idx);
  };
  std::priority_queue<Entry, std::vector<Entry>, decltype(worse)> heap(worse);
  for (int i = 0; i < n; ++i) heap.push({deviation2(i), 0, i});

  int numAlive = n;
  while (numAlive > 3 && !heap.empty()) {
    const Entry top = heap.top();
    heap.pop();
    // Each live vertex always has exactly one current-stamp entry, so the
    // first non-stale pop is the true global minimum.
    if (!alive[top.idx] || top.stamp != stamp[top.idx]) continue;
    if (top.d2 >= tol2) break;  // every vertex deviates by >= tol
    alive[top.idx] = 0;
    --numAlive;
    const int p = prev[top.idx];
    const int nx = next[top.idx];
    next[p] = nx;
    prev[nx] = p;
    for (const int j : {p, nx}) {
      ++stamp[j];
      heap.push({deviation2(j), stamp[j], j});
    }
  }

  SimplePolygon out;
  out.reserve(numAlive);
  for (int i = 0; i < n; ++i) {
    if (alive[i]) out.push_back(ring[i]);
  }
  return out;
}

}  // namespace

namespace manifold {

/**
 * The default constructor is an empty cross-section (containing no contours).
 */
CrossSection::CrossSection() : paths_(shared_paths({})) {}

CrossSection::~CrossSection() = default;

// Explicit move ops keep `other.pathsMutex_` intact so a subsequent copy-
// or move-assign into the moved-from object can still lock it.
CrossSection::CrossSection(CrossSection&& other) noexcept {
  std::lock_guard<std::mutex> lock(other.pathsMutex_);
  paths_ = std::move(other.paths_);
  transform_ = other.transform_;
  tolerance_ = other.tolerance_;
}

CrossSection& CrossSection::operator=(CrossSection&& other) noexcept {
  if (this == &other) return *this;
  std::scoped_lock lock(pathsMutex_, other.pathsMutex_);
  paths_ = std::move(other.paths_);
  transform_ = other.transform_;
  tolerance_ = other.tolerance_;
  return *this;
}

/**
 * The copy constructor avoids copying the underlying paths vector (sharing
 * with its parent via shared_ptr), however subsequent transformations, and
 * their application will not be shared. It is generally recommended to avoid
 * this, opting instead to simply create CrossSections with the available
 * const methods.
 */
CrossSection::CrossSection(const CrossSection& other) {
  std::lock_guard<std::mutex> lock(other.pathsMutex_);
  paths_ = other.paths_;
  transform_ = other.transform_;
  tolerance_ = other.tolerance_;
}

CrossSection& CrossSection::operator=(const CrossSection& other) {
  if (this == &other) return *this;
  std::scoped_lock lock(pathsMutex_, other.pathsMutex_);
  paths_ = other.paths_;
  transform_ = other.transform_;
  tolerance_ = other.tolerance_;
  return *this;
}

CrossSection::CrossSection(std::shared_ptr<const PathImpl> paths)
    : paths_(std::move(paths)) {}

/**
 * Create a 2d cross-section from a single contour. A boolean union operation
 * (with the Positive filling rule) is performed to ensure the resulting
 * CrossSection is free of self-intersections.
 *
 * @param contour A closed path outlining the desired cross-section.
 * @param fillrule Only FillRule::Positive is supported; other values are
 * ignored (and, with MANIFOLD_ASSERT, rejected).
 */
CrossSection::CrossSection(const SimplePolygon& contour, FillRule fillrule)
    : CrossSection(Polygons{contour}, fillrule) {}

/**
 * Create a 2d cross-section from a set of contours (complex polygons). A
 * boolean union operation (with the Positive filling rule) is performed to
 * combine overlapping polygons and ensure the resulting CrossSection is free
 * of intersections.
 *
 * @param contours A set of closed paths describing zero or more complex
 * polygons.
 * @param fillrule Only FillRule::Positive is supported; other values are
 * ignored (and, with MANIFOLD_ASSERT, rejected).
 */
CrossSection::CrossSection(const Polygons& contours, FillRule fillrule) {
  DEBUG_ASSERT(fillrule == FillRule::Positive, logicErr,
               "Boolean2 CrossSection supports only Positive fill");
  (void)fillrule;
  tolerance_ = InferEps(contours, {});
  paths_ = shared_paths(ApplyFillRule(contours, tolerance_));
}

/**
 * Create a 2d cross-section from an axis-aligned rectangle (bounding box).
 *
 * @param rect An axis-aligned rectangular bounding box.
 */
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
  std::lock_guard<std::mutex> lock(pathsMutex_);
  if (transform_ == mat2x3(la::identity)) return paths_;
  // Scale tolerance once from the composed transform, not per Transform call:
  // spectral norm is sub-multiplicative, so per-step scaling would inflate it
  // super-linearly under chained shears. Floor at the translated scale so large
  // translations stay above post-materialization FP noise.
  const double translationScale =
      std::max(std::fabs(transform_[2][0]), std::fabs(transform_[2][1]));
  tolerance_ =
      std::max(SpectralNorm(mat2(transform_[0], transform_[1])) * tolerance_,
               EpsilonFromScale(translationScale));
  paths_ = shared_paths(TransformPolygons(paths_->paths_, transform_));
  transform_ = mat2x3(la::identity);
  return paths_;
}

/**
 * Constructs a square with the given XY dimensions. By default it is
 * positioned in the first quadrant, touching the origin. If any dimensions in
 * size are negative, or if all are zero, an empty Manifold will be returned.
 *
 * @param size The X, and Y dimensions of the square.
 * @param center Set to true to shift the center to the origin.
 */
CrossSection CrossSection::Square(const vec2 size, bool center) {
  if (size.x < 0.0 || size.y < 0.0 || la::length(size) == 0.0) {
    return CrossSection();
  }
  const vec2 half = size * 0.5;
  if (center) return CrossSection(Rect(-half, half));
  return CrossSection(Rect({0.0, 0.0}, size));
}

/**
 * Constructs a circle of a given radius.
 *
 * @param radius Radius of the circle. Must be positive.
 * @param circularSegments Number of segments along its diameter. Default is
 * calculated by the static Quality defaults according to the radius.
 */
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

/**
 * Perform the given boolean operation between this and another CrossSection.
 */
CrossSection CrossSection::Boolean(const CrossSection& second,
                                   OpType op) const {
  // GetPaths() returns a snapshot aliasing each source's paths_, which stays
  // alive for this call, so bind by reference instead of deep-copying.
  const Polygons& a = GetPaths()->paths_;
  const Polygons& b = second.GetPaths()->paths_;
  const double eps = InferEps(a, b);
  const double tolerance = std::max({tolerance_, second.tolerance_, eps});
  CrossSection result(shared_paths(Boolean2D(a, b, op, eps)));
  result.tolerance_ = tolerance;
  return result;
}

/**
 * Perform the given boolean operation on a list of CrossSections. In case of
 * Subtract, all CrossSections in the tail are differenced from the head.
 */
CrossSection CrossSection::BatchBoolean(
    const std::vector<CrossSection>& crossSections, OpType op) {
  if (crossSections.empty()) return CrossSection();
  if (crossSections.size() == 1) return crossSections[0];

  if (op == OpType::Intersect) {
    Polygons result = crossSections[0].GetPaths()->paths_;
    double tol = crossSections[0].tolerance_;
    for (size_t i = 1; i < crossSections.size(); ++i) {
      const auto& clip = crossSections[i].GetPaths()->paths_;
      const double eps = InferEps(result, clip);
      const double tolerance =
          std::max({tol, crossSections[i].tolerance_, eps});
      result = Boolean2D(result, clip, OpType::Intersect, eps);
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
  const double eps = InferEps(subject, clips);
  const double tolerance =
      std::max({crossSections[0].tolerance_, clipsTol, eps});
  CrossSection out(shared_paths(Boolean2D(subject, clips, op, eps)));
  out.tolerance_ = tolerance;
  return out;
}

/**
 * Compute the boolean union between two cross-sections.
 */
CrossSection CrossSection::operator+(const CrossSection& Q) const {
  return Boolean(Q, OpType::Add);
}

/**
 * Compute the boolean union between two cross-sections, assigning the result
 * to the first.
 */
CrossSection& CrossSection::operator+=(const CrossSection& Q) {
  *this = *this + Q;
  return *this;
}

/**
 * Compute the boolean difference of a (clip) cross-section from another
 * (subject).
 */
CrossSection CrossSection::operator-(const CrossSection& Q) const {
  return Boolean(Q, OpType::Subtract);
}

/**
 * Compute the boolean difference of a (clip) cross-section from a another
 * (subject), assigning the result to the subject.
 */
CrossSection& CrossSection::operator-=(const CrossSection& Q) {
  *this = *this - Q;
  return *this;
}

/**
 * Compute the boolean intersection between two cross-sections.
 */
CrossSection CrossSection::operator^(const CrossSection& Q) const {
  return Boolean(Q, OpType::Intersect);
}

/**
 * Compute the boolean intersection between two cross-sections, assigning the
 * result to the first.
 */
CrossSection& CrossSection::operator^=(const CrossSection& Q) {
  *this = *this ^ Q;
  return *this;
}

/**
 * Construct a CrossSection from a vector of other CrossSections (batch
 * boolean union).
 */
CrossSection CrossSection::Compose(
    const std::vector<CrossSection>& crossSections) {
  return BatchBoolean(crossSections, OpType::Add);
}

/**
 * This operation returns a vector of CrossSections that are topologically
 * disconnected, each containing one outline contour with zero or more
 * holes.
 */
std::vector<CrossSection> CrossSection::Decompose() const {
  if (NumContour() < 2) return {*this};
  const auto& paths = GetPaths()->paths_;
  std::vector<Polygons> components = DecomposeByContainment(paths);
  std::vector<CrossSection> out;
  out.reserve(components.size());
  for (auto& component : components) {
    CrossSection piece(shared_paths(std::move(component)));
    piece.tolerance_ = tolerance_;
    out.push_back(std::move(piece));
  }
  return out;
}

/**
 * Move this CrossSection in space. This operation can be chained. Transforms
 * are combined and applied lazily.
 *
 * @param v The vector to add to every vertex.
 */
CrossSection CrossSection::Translate(const vec2 v) const {
  mat2x3 m({1.0, 0.0}, {0.0, 1.0}, {v.x, v.y});
  return Transform(m);
}

/**
 * Applies a (Z-axis) rotation to the CrossSection, in degrees. This operation
 * can be chained. Transforms are combined and applied lazily.
 *
 * @param degrees degrees about the Z-axis to rotate.
 */
CrossSection CrossSection::Rotate(double degrees) const {
  const double s = sind(degrees);
  const double c = cosd(degrees);
  mat2x3 m({c, s}, {-s, c}, {0.0, 0.0});
  return Transform(m);
}

/**
 * Scale this CrossSection in space. This operation can be chained. Transforms
 * are combined and applied lazily.
 *
 * @param scale The vector to multiply every vertex by per component.
 */
CrossSection CrossSection::Scale(const vec2 scale) const {
  mat2x3 m({scale.x, 0.0}, {0.0, scale.y}, {0.0, 0.0});
  return Transform(m);
}

/**
 * Mirror this CrossSection over the arbitrary axis whose normal is described by
 * the unit form of the given vector. If the length of the vector is zero, an
 * empty CrossSection is returned. This operation can be chained. Transforms are
 * combined and applied lazily.
 *
 * @param ax the axis to be mirrored over
 */
CrossSection CrossSection::Mirror(const vec2 ax) const {
  if (la::length(ax) == 0.0) return CrossSection();
  const vec2 n = la::normalize(ax);
  return Transform(
      mat2x3(mat2(la::identity) - 2.0 * la::outerprod(n, n), vec2(0.0)));
}

/**
 * Transform this CrossSection in space. The first two columns form a 2x2
 * matrix transform and the last is a translation vector. This operation can
 * be chained. Transforms are combined and applied lazily.
 *
 * @param m The affine transform matrix to apply to all the vertices.
 */
CrossSection CrossSection::Transform(const mat2x3& m) const {
  // A non-finite transform is a no-op rather than poisoning coords/tolerance_.
  if (!AllFinite(m)) return *this;
  std::lock_guard<std::mutex> lock(pathsMutex_);
  CrossSection transformed;
  transformed.transform_ = m * Mat3(transform_);
  transformed.paths_ = paths_;
  // Carry tolerance unscaled; GetPaths() scales it once from the composed
  // transform at materialization.
  transformed.tolerance_ = tolerance_;
  return transformed;
}

/**
 * Move the vertices of this CrossSection (creating a new one) according to
 * any arbitrary input function, followed by a union operation (with a
 * Positive fill rule) that ensures any introduced intersections are not
 * included in the result.
 *
 * @param warpFunc A function that modifies a given vertex position.
 */
CrossSection CrossSection::Warp(std::function<void(vec2&)> warpFunc) const {
  return WarpBatch([&warpFunc](VecView<vec2> points) {
    for (vec2& point : points) warpFunc(point);
  });
}

/**
 * Same as CrossSection::Warp but calls warpFunc with
 * a VecView which is roughly equivalent to std::span
 * pointing to all vec2 elements to be modified in-place
 *
 * @param warpFunc A function that modifies multiple vertex positions.
 */
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
  // Warping can self-intersect the rings, so re-apply the fill rule at machine
  // eps (not the drift tolerance - this is regularization, not decimation).
  const double eps = InferEps(paths, {});
  CrossSection out(shared_paths(ApplyFillRule(paths, eps)));
  out.tolerance_ = std::max(tolerance_, eps);
  return out;
}

/**
 * Remove vertices from the contours in this CrossSection that are less than
 * the specified distance tolerance from an imaginary line that passes through
 * its two adjacent vertices. Near duplicate vertices and collinear points
 * will be removed at lower tolerances, with elimination of line segments
 * becoming increasingly aggressive with larger tolerances.
 *
 * It is recommended to apply this function following Offset, in order to
 * clean up any spurious tiny line segments introduced that do not improve
 * quality in any meaningful way. This is particularly important if further
 * offseting operations are to be performed, which would compound the issue.
 */
CrossSection CrossSection::Simplify(double epsilon) const {
  // Stored paths are already fill-rule-regularized, so Simplify only decimates:
  // drop tiny contours, then remove near-collinear verts per ring at `epsilon`
  // (clipper2 SimplifyPaths style, not Ramer-Douglas-Peucker). Like clipper2,
  // the result is NOT re-regularized - a coarse `epsilon` can self-intersect a
  // ring, so healing is left to a later boolean.
  const Polygons& paths = GetPaths()->paths_;
  const Polygons filtered = FilterSmallContours(paths, epsilon);
  Polygons out;
  out.reserve(filtered.size());
  for (const auto& ring : filtered) {
    SimplePolygon simplified = SimplifyRing(ring, epsilon);
    if (simplified.size() >= 3) out.push_back(std::move(simplified));
  }
  CrossSection result(shared_paths(std::move(out)));
  result.tolerance_ = std::max(tolerance_, epsilon);
  return result;
}

/**
 * Inflate the contours in CrossSection by the specified delta, handling
 * corners according to the given JoinType.
 *
 * @param delta Positive deltas will cause the expansion of outlining contours
 * to expand, and retraction of inner (hole) contours. Negative deltas will
 * have the opposite effect.
 * @param jointype The join type specifying the treatment of contour joins
 * (corners). Defaults to Round.
 * @param miter_limit The maximum distance in multiples of delta that vertices
 * can be offset from their original positions with before squaring is
 * applied, <B>when the join type is Miter</B> (default is 2, which is the
 * minimum allowed). See the [Clipper2
 * MiterLimit](http://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/Properties/MiterLimit.htm)
 * page for a visual example.
 * @param circularSegments Number of segments per 360 degrees of
 * <B>JoinType::Round</B> corners (roughly, the number of vertices that
 * will be added to each contour). Default is calculated by the static Quality
 * defaults according to the radius.
 */
CrossSection CrossSection::Offset(double delta, JoinType jointype,
                                  double miterLimit,
                                  int circularSegments) const {
  // Qualified: unqualified lookup inside the class finds this member and
  // hides the namespace-scope polygon offset.
  Polygons offset = manifold::Offset(GetPaths()->paths_, delta, jointype,
                                     miterLimit, circularSegments);
  CrossSection out(shared_paths(std::move(offset)));
  // Round-join faceting (circularSegments) is a quality choice, not a drift
  // budget, so it is not folded into tolerance_ - doing so would over-merge
  // features downstream. Max, not sum, keeps chained Offset bounded.
  out.tolerance_ = std::max(tolerance_, InferEps(out.GetPaths()->paths_, {}));
  return out;
}

/**
 * Compute the convex hull enveloping a set of cross-sections.
 *
 * @param crossSections A vector of cross-sections over which to compute a
 * convex hull.
 */
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
  if (hull.size() < 3) return CrossSection();
  CrossSection out(shared_paths({std::move(hull)}));
  out.tolerance_ = std::max(maxTol, InferEps(out.GetPaths()->paths_, {}));
  return out;
}

/**
 * Compute the convex hull of this cross-section.
 */
CrossSection CrossSection::Hull() const { return Hull({*this}); }

/**
 * Compute the convex hull of a set of points. If the given points are fewer
 * than 3, an empty CrossSection will be returned.
 *
 * @param pts A vector of 2-dimensional points over which to compute a convex
 * hull.
 */
CrossSection CrossSection::Hull(SimplePolygon pts) {
  SimplePolygon hull = HullImpl(pts);
  if (hull.size() < 3) return CrossSection();
  Polygons inputForEps{pts};
  CrossSection out(shared_paths({std::move(hull)}));
  out.tolerance_ = InferEps(inputForEps, {});
  return out;
}

/**
 * Compute the convex hull of a set of points/polygons. If the given points are
 * fewer than 3, an empty CrossSection will be returned.
 *
 * @param polys A vector of vectors of 2-dimensional points over which to
 * compute a convex hull.
 */
CrossSection CrossSection::Hull(const Polygons polys) {
  SimplePolygon points;
  for (const auto& path : polys)
    for (const vec2& point : path) points.push_back(point);
  return Hull(points);
}

/**
 * Return the total area covered by complex polygons making up the
 * CrossSection.
 */
double CrossSection::Area() const {
  return TotalSignedArea(GetPaths()->paths_);
}

/**
 * Return the number of vertices in the CrossSection.
 */
size_t CrossSection::NumVert() const {
  size_t numVert = 0;
  for (const auto& path : GetPaths()->paths_) numVert += path.size();
  return numVert;
}

/**
 * Return the number of contours (both outer and inner paths) in the
 * CrossSection.
 */
size_t CrossSection::NumContour() const { return GetPaths()->paths_.size(); }

/**
 * Does the CrossSection contain any contours?
 */
bool CrossSection::IsEmpty() const { return GetPaths()->paths_.empty(); }

/**
 * Returns the axis-aligned bounding rectangle of all the CrossSections'
 * vertices.
 */
Rect CrossSection::Bounds() const {
  Rect box;
  for (const auto& path : GetPaths()->paths_) {
    for (const vec2& v : path) box.Union(v);
  }
  return box;
}

/**
 * Return the contours of this CrossSection as a Polygons.
 */
Polygons CrossSection::ToPolygons() const { return GetPaths()->paths_; }

}  // namespace manifold
