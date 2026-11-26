// Copyright 2026 The Manifold Authors.
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

#include <cmath>
#include <limits>

#include "csg_tree.h"
#include "disjoint_sets.h"
#include "impl.h"
#include "manifold/manifold.h"
#include "manifold/polygon.h"
#include "parallel.h"

namespace manifold {
namespace {
// Divisions a revolve of degrees, already in (0, 360], makes from a positive
// full-circle segment count, rounded up, and at least one for an angle so small
// the product underflows. The multiply and divide each round by at most half
// an ulp, so for an exact angle a whole count n comes out below n + 2 ulp(n),
// at most one ulp above n. Stepping down one ulp before rounding up removes
// that, and also absorbs the rounding of a computed angle such as k * 3.6.
int RevolveDivisions(int segments, double degrees) {
  return std::max(1, static_cast<int>(std::ceil(
                         std::nextafter(segments * degrees / 360, 0.0))));
}

std::shared_ptr<Manifold::Impl> ExtrudeImpl(const Polygons& crossSection,
                                            double height, int nDivisions,
                                            double twistDegrees, vec2 scaleTop,
                                            double zOffset = 0.0,
                                            bool flipZ = false) {
  auto pImpl_ = std::make_shared<Manifold::Impl>();
  ++nDivisions;
  auto& vertPos = pImpl_->vertPos_;
  Vec<ivec3> triVertsDH;
  auto& triVerts = triVertsDH;
  int nCrossSection = 0;
  bool isCone = scaleTop.x == 0.0 && scaleTop.y == 0.0;
  size_t idx = 0;
  PolygonsIdx polygonsIndexed;
  for (auto& poly : crossSection) {
    nCrossSection += poly.size();
    SimplePolygonIdx simpleIndexed;
    for (const vec2& polyVert : poly) {
      vertPos.push_back({polyVert.x, polyVert.y, 0.0});
      simpleIndexed.push_back({polyVert, static_cast<int>(idx++)});
    }
    polygonsIndexed.push_back(simpleIndexed);
  }
  for (int i = 1; i < nDivisions + 1; ++i) {
    double alpha = i / double(nDivisions);
    double phi = alpha * twistDegrees;
    vec2 scale = la::lerp(vec2(1.0), scaleTop, alpha);
    mat2 rotation({cosd(phi), sind(phi)}, {-sind(phi), cosd(phi)});
    mat2 transform = mat2({scale.x, 0.0}, {0.0, scale.y}) * rotation;
    size_t j = 0;
    size_t idx = 0;
    for (const auto& poly : crossSection) {
      for (size_t vert = 0; vert < poly.size(); ++vert) {
        size_t offset = idx + nCrossSection * i;
        size_t thisVert = vert + offset;
        size_t lastVert = (vert == 0 ? poly.size() : vert) - 1 + offset;
        if (i == nDivisions && isCone) {
          triVerts.push_back(ivec3(nCrossSection * i + j,
                                   lastVert - nCrossSection,
                                   thisVert - nCrossSection));
        } else {
          vec2 pos = transform * poly[vert];
          vertPos.push_back({pos.x, pos.y, height * alpha});
          triVerts.push_back(
              ivec3(thisVert, lastVert, thisVert - nCrossSection));
          triVerts.push_back(ivec3(lastVert, lastVert - nCrossSection,
                                   thisVert - nCrossSection));
        }
      }
      ++j;
      idx += poly.size();
    }
  }
  if (isCone)
    for (size_t j = 0; j < crossSection.size();
         ++j)  // Duplicate vertex for Genus
      vertPos.push_back({0.0, 0.0, height});
  std::vector<ivec3> top = TriangulateIdx(polygonsIndexed);
  for (const ivec3& tri : top) {
    triVerts.push_back({tri[0], tri[2], tri[1]});
    if (!isCone) triVerts.push_back(tri + nCrossSection * nDivisions);
  }

  if (flipZ) {
    for (vec3& v : vertPos) v.z = zOffset - v.z;
    for (ivec3& tri : triVerts) std::swap(tri[1], tri[2]);
  } else if (zOffset != 0.0) {
    for (vec3& v : vertPos) v.z += zOffset;
  }

  pImpl_->CreateHalfedges(triVertsDH);
  pImpl_->CalculateBBox();
  pImpl_->SortGeometry();
  pImpl_->SetFaceAndVertNormals();
  pImpl_->InitializeOriginal();
  return pImpl_;
}
}  // namespace

/**
 * Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
 * and the rest at similarly symmetric points.
 */
Manifold Manifold::Tetrahedron() {
  return Manifold(std::make_shared<Impl>(Impl::Shape::Tetrahedron));
}

/**
 * Constructs a unit cube (edge lengths all one), by default in the first
 * octant, touching the origin. If any dimensions in size are negative or
 * non-finite, or if all are zero, an empty Manifold with InvalidConstruction
 * status will be returned.
 *
 * @param size The X, Y, and Z dimensions of the box.
 * @param center Set to true to shift the center to the origin.
 */
Manifold Manifold::Cube(vec3 size, bool center) {
  if (!la::all(la::isfinite(size)) || size.x < 0.0 || size.y < 0.0 ||
      size.z < 0.0 || la::length(size) == 0.) {
    return Invalid();
  }
  mat3x4 m({{size.x, 0.0, 0.0}, {0.0, size.y, 0.0}, {0.0, 0.0, size.z}},
           center ? (-size / 2.0) : vec3(0.0));
  return Manifold(std::make_shared<Impl>(Manifold::Impl::Shape::Cube, m));
}

/**
 * A convenience constructor for the common case of extruding a circle. Can also
 * form cones if both radii are specified.
 *
 * Non-finite dimensions return an empty Manifold with InvalidConstruction
 * status.
 *
 * @param height Z-extent. Must be positive and finite.
 * @param radiusLow Radius of bottom circle. Must be non-negative and finite. If
 * zero, radiusHigh must be positive and a cone with apex at the bottom is
 * created.
 * @param radiusHigh Radius of top circle. Must be finite and can equal zero.
 * Negative values, including the default, use radiusLow.
 * @param quality How many line segments to use around the circle, see
 * CircularQuality.
 * @param center Set to true to shift the center to the origin. Default is
 * origin at the bottom.
 */
Manifold Manifold::Cylinder(double height, double radiusLow, double radiusHigh,
                            CircularQuality quality, bool center) {
  if (!std::isfinite(height) || !std::isfinite(radiusLow) ||
      !std::isfinite(radiusHigh) || height <= 0.0 || radiusLow < 0.0) {
    return Invalid();
  }
  if (radiusLow == 0.0 && radiusHigh <= 0.0) {
    return Invalid();
  }
  // Preserve the usual construction's floating-point coordinates. Reverse only
  // when scaling from the requested bottom radius would overflow.
  const bool flip =
      radiusLow == 0.0 ||
      (radiusHigh > radiusLow && !std::isfinite(radiusHigh / radiusLow));
  if (flip) std::swap(radiusLow, radiusHigh);
  const double scale = radiusHigh >= 0.0 ? radiusHigh / radiusLow : 1.0;
  const double radius = fmax(radiusLow, radiusHigh);
  const int n = quality.SegmentsFor(radius);

  SimplePolygon circle(n);
  const double dPhi = 360.0 / n;
  for (int i = 0; i < n; ++i) {
    circle[i] = {radiusLow * cosd(dPhi * i), radiusLow * sind(dPhi * i)};
  }

  const double zOffset = (flip ? height : 0.0) - (center ? height / 2.0 : 0.0);
  return Manifold(
      ExtrudeImpl({circle}, height, 0, 0.0, vec2(scale), zOffset, flip));
}

/**
 * Constructs a geodesic sphere of a given radius.
 *
 * @param radius Radius of the sphere. Must be positive and finite; otherwise
 * returns an empty Manifold with InvalidConstruction status.
 * @param quality Number of segments in a full circle, see CircularQuality.
 * This number will always be rounded up to the nearest factor of four, as this
 * sphere is constructed by refining an octahedron. This means there are a
 * circle of vertices on all three of the axis planes.
 */
Manifold Manifold::Sphere(double radius, CircularQuality quality) {
  if (radius <= 0.0 || !std::isfinite(radius)) {
    return Invalid();
  }
  // Round an explicit count up to a multiple of four without overflowing.
  const int segments = quality.segments;
  int n = segments > 0 ? segments / 4 + (segments % 4 != 0)
                       : quality.SegmentsFor(radius) / 4;
  auto pImpl_ = std::make_shared<Impl>(Impl::Shape::Octahedron);
  pImpl_->Subdivide([n](vec3, vec4, vec4) { return n - 1; });
  if (pImpl_->status_ != Error::NoError) return Manifold(pImpl_);
  for_each_n(autoPolicy(pImpl_->NumVert(), 1e5), pImpl_->vertPos_.begin(),
             pImpl_->NumVert(), [radius](vec3& v) {
               v = vec3(la::cos(kHalfPi * (1.0 - v.x)),
                        la::cos(kHalfPi * (1.0 - v.y)),
                        la::cos(kHalfPi * (1.0 - v.z)));
               v = radius * la::normalize(v);
               if (std::isnan(v.x)) v = vec3(0.0);
             });
  // Ignore preceding octahedron.
  pImpl_->CalculateBBox();
  pImpl_->SortGeometry();
  pImpl_->SetFaceAndVertNormals();
  pImpl_->InitializeOriginal();
  return Manifold(pImpl_);
}

/**
 * Constructs a manifold from a set of polygons by extruding them along the
 * Z-axis.
 * Note that high twistDegrees with small nDivisions may cause
 * self-intersection. This is not checked here and it is up to the user to
 * choose the correct parameters.
 *
 * Contours with fewer than three vertices are ignored. If none remain, the
 * result is an empty Manifold with no error. An invalid parameter or a
 * non-finite vertex in a remaining contour gives an empty Manifold with
 * InvalidConstruction status.
 *
 * @param crossSection A set of non-overlapping polygons to extrude.
 * @param height Z-extent of extrusion. Must be positive and finite.
 * @param nDivisions Number of extra copies of the cross-section to insert into
 * the shape vertically; especially useful in combination with twistDegrees to
 * avoid interpolation artifacts. Must be non-negative. Default is none.
 * @param twistDegrees Amount to twist the top cross-section relative to the
 * bottom, interpolated linearly for the divisions in between. Must be finite.
 * @param scaleTop Amount to scale the top (independently in X and Y). Must be
 * non-negative and finite. If the scale is {0, 0}, a pure cone is formed with
 * only a single vertex at the top. Note that scale is applied after twist.
 * Default {1, 1}.
 */
Manifold Manifold::Extrude(const Polygons& crossSection, double height,
                           int nDivisions, double twistDegrees, vec2 scaleTop) {
  ZoneScoped;
  if (!(height > 0.0) || !std::isfinite(height) || nDivisions < 0 ||
      !std::isfinite(twistDegrees) || !la::all(la::isfinite(scaleTop)) ||
      scaleTop.x < 0.0 || scaleTop.y < 0.0) {
    return Invalid();
  }
  // Contours with fewer than 3 verts have no area.
  Polygons polygons;
  for (const SimplePolygon& poly : crossSection) {
    if (poly.size() < 3) continue;
    for (const vec2& v : poly) {
      if (!la::all(la::isfinite(v))) return Invalid();
    }
    polygons.push_back(poly);
  }
  if (polygons.empty()) return Manifold();
  // Vert indices are int.
  int64_t numVert = 0;
  for (const SimplePolygon& poly : polygons) numVert += poly.size();
  if (numVert * (nDivisions + int64_t(2)) > std::numeric_limits<int>::max()) {
    return PropagateStatus(Error::ResultTooLarge);
  }
  return Manifold(
      ExtrudeImpl(polygons, height, nDivisions, twistDegrees, scaleTop));
}

/**
 * Constructs a manifold from a set of polygons by revolving this cross-section
 * around its Y-axis and then setting this as the Z-axis of the resulting
 * manifold. If the polygons cross the Y-axis, only the part on the positive X
 * side is used. Geometrically valid input will result in geometrically valid
 * output.
 *
 * Contours with fewer than three vertices, and contours entirely on the
 * negative X side, are ignored. If none remain, the result is an empty Manifold
 * with no error. An invalid angle, or a non-finite vertex in a contour of at
 * least three vertices, gives an empty Manifold with InvalidConstruction
 * status.
 *
 * @param crossSection A set of non-overlapping polygons to revolve.
 * @param quality Number of segments in a full circle, see CircularQuality.
 * A partial revolve uses the matching fraction of it, rounded up, ignoring an
 * excess of one unit in the last place. For an exact number of divisions, use
 * CircularQuality::ForDivisions().
 * @param revolveDegrees Number of degrees to revolve. Must be finite and
 * nonzero. A negative value revolves the other way. Default is 360 degrees.
 */
Manifold Manifold::Revolve(const Polygons& crossSection,
                           CircularQuality quality, double revolveDegrees) {
  ZoneScoped;
  if (!std::isfinite(revolveDegrees) || revolveDegrees == 0.0) {
    return Invalid();
  }
  // A negative angle builds the positive sweep and mirrors it in y below.
  const bool negative = revolveDegrees < 0;
  revolveDegrees = std::abs(revolveDegrees);

  Polygons polygons;
  double radius = 0;
  for (const SimplePolygon& poly : crossSection) {
    if (poly.size() < 3) continue;
    // Check before clipping so invalid coordinates cannot be discarded
    // silently.
    for (const vec2& v : poly) {
      if (!la::all(la::isfinite(v))) return Invalid();
    }
    size_t i = 0;
    while (i < poly.size() && poly[i].x < 0) {
      ++i;
    }
    if (i == poly.size()) {
      continue;
    }
    polygons.push_back({});
    const size_t start = i;
    do {
      if (poly[i].x >= 0) {
        polygons.back().push_back(poly[i]);
        radius = std::max(radius, poly[i].x);
      }
      const size_t next = i + 1 == poly.size() ? 0 : i + 1;
      if ((poly[next].x < 0) != (poly[i].x < 0)) {
        const double y = poly[next].y - poly[next].x *
                                            (poly[i].y - poly[next].y) /
                                            (poly[i].x - poly[next].x);
        polygons.back().push_back({0, y});
      }
      i = next;
    } while (i != start);
  }

  if (polygons.empty()) return Manifold();

  if (revolveDegrees > 360.0) {
    revolveDegrees = 360.0;
  }
  const bool isFullRevolution = revolveDegrees == 360.0;

  const int nDivisions =
      RevolveDivisions(quality.SegmentsFor(radius), revolveDegrees);

  auto pImpl_ = std::make_shared<Impl>();
  auto& vertPos = pImpl_->vertPos_;
  Vec<ivec3> triVertsDH;
  auto& triVerts = triVertsDH;

  std::vector<int> startPoses;
  std::vector<int> endPoses;

  const double dPhi = revolveDegrees / nDivisions;
  // first and last slice are distinguished if not a full revolution.
  const int64_t slices = isFullRevolution ? nDivisions : nDivisions + 1LL;
  // A partial revolve closes each end with the triangulated cross-section.
  const std::vector<ivec3> capTriangles =
      isFullRevolution ? std::vector<ivec3>()
                       : Triangulate(polygons, pImpl_->epsilon_);
  // Vertices on the axis are not repeated per slice. Each vertex off the axis
  // adds two triangles per division.
  int64_t numVert = 0;
  int64_t numTri = 2 * static_cast<int64_t>(capTriangles.size());
  for (const auto& poly : polygons) {
    for (const vec2& pt : poly) {
      numVert += pt.x > 0 ? slices : 1;
      if (pt.x > 0) numTri += 2LL * nDivisions;
    }
  }
  if (numVert > std::numeric_limits<int>::max() ||
      3 * numTri > std::numeric_limits<int>::max())
    return PropagateStatus(Error::ResultTooLarge);
  const int nSlices = static_cast<int>(slices);

  for (const auto& poly : polygons) {
    size_t nPosVerts = 0;
    size_t nRevolveAxisVerts = 0;
    for (auto& pt : poly) {
      if (pt.x > 0) {
        nPosVerts++;
      } else {
        nRevolveAxisVerts++;
      }
    }

    for (size_t polyVert = 0; polyVert < poly.size(); ++polyVert) {
      const size_t startPosIndex = vertPos.size();

      if (!isFullRevolution) startPoses.push_back(startPosIndex);

      const vec2 currPolyVertex = poly[polyVert];
      const vec2 prevPolyVertex =
          poly[polyVert == 0 ? poly.size() - 1 : polyVert - 1];

      const int prevStartPosIndex =
          startPosIndex +
          (polyVert == 0 ? nRevolveAxisVerts + (nSlices * nPosVerts) : 0) +
          (prevPolyVertex.x == 0.0 ? -1 : -nSlices);

      for (int slice = 0; slice < nSlices; ++slice) {
        const double phi = slice * dPhi;
        if (slice == 0 || currPolyVertex.x > 0) {
          vertPos.push_back({currPolyVertex.x * cosd(phi),
                             currPolyVertex.x * sind(phi), currPolyVertex.y});
        }

        if (isFullRevolution || slice > 0) {
          const int lastSlice = (slice == 0 ? nDivisions : slice) - 1;
          if (currPolyVertex.x > 0.0) {
            triVerts.push_back(ivec3(
                startPosIndex + slice, startPosIndex + lastSlice,
                // "Reuse" vertex of first slice if it lies on the revolve axis
                (prevPolyVertex.x == 0.0 ? prevStartPosIndex
                                         : prevStartPosIndex + lastSlice)));
          }

          if (prevPolyVertex.x > 0.0) {
            triVerts.push_back(
                ivec3(prevStartPosIndex + lastSlice, prevStartPosIndex + slice,
                      (currPolyVertex.x == 0.0 ? startPosIndex
                                               : startPosIndex + slice)));
          }
        }
      }
      if (!isFullRevolution) endPoses.push_back(vertPos.size() - 1);
    }
  }

  for (auto& t : capTriangles) {
    triVerts.push_back({startPoses[t.x], startPoses[t.y], startPoses[t.z]});
  }
  for (auto& t : capTriangles) {
    triVerts.push_back({endPoses[t.z], endPoses[t.y], endPoses[t.x]});
  }

  // Mirror in y for a negative angle, and flip the winding to keep the volume
  // positive.
  if (negative) {
    for (vec3& v : vertPos) v.y = -v.y;
    for (ivec3& tri : triVerts) std::swap(tri.y, tri.z);
  }

  pImpl_->CreateHalfedges(triVertsDH);
  pImpl_->CalculateBBox();
  pImpl_->SortGeometry();
  pImpl_->SetFaceAndVertNormals();
  pImpl_->InitializeOriginal();
  return Manifold(pImpl_);
}

/**
 * Returns a quality that makes Manifold::Revolve() divide the given angle into
 * exactly this many divisions. Like Revolve, it uses the angle's magnitude,
 * clamped to 360 degrees. A revolve of more than 120 degrees needs at least two
 * divisions, and of more than 240 degrees at least three, and a tiny angle
 * cannot get more divisions than the largest int count allows. Past those
 * limits it returns the closest count Revolve allows. A non-finite or zero
 * angle, which Revolve rejects, gives the default quality.
 *
 * @param divisions Number of divisions. Values below 1 are treated as 1.
 * @param degrees Angle Revolve will sweep.
 */
CircularQuality CircularQuality::ForDivisions(int divisions, double degrees) {
  if (!std::isfinite(degrees) || degrees == 0) return {};
  degrees = std::min(std::abs(degrees), 360.0);
  divisions = std::max(divisions, 1);
  // Aim for the middle of the range of counts that give these divisions, then
  // step once if rounding put it just outside.
  const double target = std::round((divisions - 0.5) * 360 / degrees);
  int segments = static_cast<int>(
      std::min(std::max(target, 3.0), double(std::numeric_limits<int>::max())));
  const int got = RevolveDivisions(segments, degrees);
  if (got > divisions && segments > 3) --segments;
  if (got < divisions && segments < std::numeric_limits<int>::max()) {
    ++segments;
  }
  return {segments};
}

/**
 * This operation returns a vector of topologically disconnected Manifolds,
 * keeping enclosed negative-volume shells with their enclosing
 * positive-volume shell. If everything is connected, the vector is length
 * one, containing a copy of the original.
 */
std::vector<Manifold> Manifold::Decompose() const {
  ZoneScoped;
  auto pImpl_ = GetCsgLeafNode().GetImpl();
  if (pImpl_->status_ != Error::NoError) {
    return {PropagateStatus(pImpl_->status_)};
  }
  DisjointSets uf(NumVert());
  for (size_t edge = 0; edge < pImpl_->halfedge_.size(); ++edge) {
    if (pImpl_->halfedge_.IsForward(edge)) {
      uf.unite(pImpl_->halfedge_.Start(edge), pImpl_->halfedge_.End(edge));
    }
  }
  std::vector<int> componentIndices;
  const int numComponents = uf.connectedComponents(componentIndices);

  if (numComponents == 1) {
    return {Manifold(LoadPNode())};
  }
  const auto& vertLabel = componentIndices;
  const int numVert = NumVert();
  const size_t numTri = pImpl_->NumTri();
  std::vector<Vec<int>> componentVerts(numComponents);
  std::vector<Vec<int>> componentFaces(numComponents);
  for (int v = 0; v < numVert; ++v) {
    componentVerts[componentIndices[v]].push_back(v);
  }
  const auto& halfedge = pImpl_->halfedge_;
  for (size_t face = 0; face < numTri; ++face) {
    const int v = halfedge.Start(static_cast<int>(3 * face));
    componentFaces[componentIndices[v]].push_back(static_cast<int>(face));
  }
  std::vector<std::shared_ptr<Impl>> components;
  std::vector<int> componentLabels;
  components.reserve(numComponents);
  componentLabels.reserve(numComponents);
  for (int i = 0; i < numComponents; ++i) {
    const Vec<int>& vertNew2Old = componentVerts[i];
    const Vec<int>& faceNew2Old = componentFaces[i];
    if (faceNew2Old.empty()) continue;

    auto impl = std::make_shared<Impl>();
    impl->relPrecision_ = pImpl_->relPrecision_;
    const int nVert = vertNew2Old.size();
    impl->vertPos_.resize(nVert);
    impl->vertNormal_.resize(nVert);
    gather(vertNew2Old.begin(), vertNew2Old.end(), pImpl_->vertPos_.begin(),
           impl->vertPos_.begin());
    gather(vertNew2Old.begin(), vertNew2Old.end(), pImpl_->vertNormal_.begin(),
           impl->vertNormal_.begin());

    impl->GatherFaces(*pImpl_, faceNew2Old);
    impl->ReindexVerts(vertNew2Old, pImpl_->NumVert());
    impl->CalculateBBox();
    impl->SortGeometry();

    components.push_back(std::move(impl));
    componentLabels.push_back(i);
  }

  const int numMeshes = static_cast<int>(components.size());
  std::vector<bool> isVug(numMeshes);
  bool hasVug = false;
  for (int i = 0; i < numMeshes; ++i) {
    isVug[i] = components[i]->IsVug();
    hasVug |= isVug[i];
  }
  if (!hasVug) {
    std::vector<Manifold> meshes;
    meshes.reserve(numMeshes);
    for (auto& component : components) meshes.push_back(Manifold(component));
    return meshes;
  }

  std::vector<int> parent(numMeshes, -1);
  for (int i = 0; i < numMeshes; ++i) {
    if (!isVug[i]) continue;
    const auto& childVerts = components[i]->vertPos_;
    Box parentBox;
    for (int j = 0; j < numMeshes; ++j) {
      if (i == j || isVug[j]) continue;
      const Box& childBox = components[i]->bBox_;
      const Box& currentBox = components[j]->bBox_;
      const vec3 eps(components[j]->epsilon_);
      const Box expandedBox(currentBox.min - eps, currentBox.max + eps);
      if (!expandedBox.Contains(childBox)) continue;

      if (parent[i] >= 0) {
        if (!parentBox.Contains(currentBox) ||
            components[parent[i]]->PointWinding(VecView<const vec3>(
                components[j]->vertPos_.data(), 1))[0] != 1) {
          continue;
        }
      }
      if (components[j]->PointWinding(
              VecView<const vec3>(childVerts.data(), 1))[0] == 1) {
        parent[i] = j;
        parentBox = expandedBox;
      }
    }
  }

  std::vector<int> outputByComponent(numComponents, -1);
  std::vector<int> outputRoot;
  std::vector<int> outputComponentCount;
  int numOutputs = 0;
  for (int i = 0; i < numMeshes; ++i) {
    if (!isVug[i]) {
      outputByComponent[componentLabels[i]] = numOutputs++;
      outputRoot.push_back(i);
      outputComponentCount.push_back(1);
    }
  }
  for (int i = 0; i < numMeshes; ++i) {
    if (!isVug[i]) continue;
    if (parent[i] >= 0) {
      const int output = outputByComponent[componentLabels[parent[i]]];
      outputByComponent[componentLabels[i]] = output;
      ++outputComponentCount[output];
    } else {
      outputByComponent[componentLabels[i]] = numOutputs++;
      outputRoot.push_back(i);
      outputComponentCount.push_back(1);
    }
  }

  std::vector<Manifold> meshes;
  meshes.reserve(numOutputs);
  for (int i = 0; i < numOutputs; ++i) {
    if (outputComponentCount[i] == 1) {
      meshes.push_back(Manifold(components[outputRoot[i]]));
      continue;
    }

    auto impl = std::make_shared<Impl>();
    impl->relPrecision_ = pImpl_->relPrecision_;

    Vec<int> vertNew2Old(numVert);
    const int nVert = copy_if(countAt(0), countAt(numVert), vertNew2Old.begin(),
                              [i, &vertLabel, &outputByComponent](int v) {
                                return outputByComponent[vertLabel[v]] == i;
                              }) -
                      vertNew2Old.begin();
    impl->vertPos_.resize(nVert);
    impl->vertNormal_.resize(nVert);
    vertNew2Old.resize(nVert);
    gather(vertNew2Old.begin(), vertNew2Old.end(), pImpl_->vertPos_.begin(),
           impl->vertPos_.begin());
    gather(vertNew2Old.begin(), vertNew2Old.end(), pImpl_->vertNormal_.begin(),
           impl->vertNormal_.begin());

    Vec<int> faceNew2Old;
    faceNew2Old.reserve(NumTri());
    const auto& halfedge = pImpl_->halfedge_;
    for (size_t face = 0; face < NumTri(); ++face) {
      const int component =
          vertLabel[halfedge.Start(static_cast<int>(3 * face))];
      if (outputByComponent[component] == i) {
        faceNew2Old.push_back(static_cast<int>(face));
      }
    }

    if (faceNew2Old.empty()) continue;

    impl->GatherFaces(*pImpl_, faceNew2Old);
    impl->ReindexVerts(vertNew2Old, pImpl_->NumVert());
    impl->CalculateBBox();
    impl->SortGeometry();
    meshes.push_back(Manifold(impl));
  }
  return meshes;
}
}  // namespace manifold
