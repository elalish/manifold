// Copyright 2020 Emmett Lalish
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

#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

#include "boolean3.cuh"
#include "connected_components.cuh"
#include "manifold_impl.cuh"
#include "polygon.h"

namespace {
using namespace manifold;
using namespace thrust::placeholders;

struct ToSphere {
  float length;
  __host__ __device__ void operator()(glm::vec3& v) {
    v = glm::cos(glm::half_pi<float>() * (1.0f - v));
    v = length * glm::normalize(v);
    if (isnan(v.x)) v = glm::vec3(0.0);
  }
};

struct UpdateHalfedge {
  const int nextVert;
  const int nextEdge;
  const int nextFace;

  __host__ __device__ Halfedge operator()(Halfedge edge) {
    edge.startVert += nextVert;
    edge.endVert += nextVert;
    edge.pairedHalfedge += nextEdge;
    edge.face += nextFace;
    return edge;
  }
};

struct Equals {
  int val;
  __host__ __device__ bool operator()(int x) { return x == val; }
};

struct RemoveFace {
  const Halfedge* halfedge;
  const int* vertLabel;
  const int keepLabel;

  __host__ __device__ bool operator()(int face) {
    return vertLabel[halfedge[3 * face].startVert] != keepLabel;
  }
};

struct MakeTri {
  const Halfedge* halfedges;

  __host__ __device__ void operator()(thrust::tuple<glm::ivec3&, int> inOut) {
    glm::ivec3& tri = thrust::get<0>(inOut);
    const int face = 3 * thrust::get<1>(inOut);

    for (int i : {0, 1, 2}) {
      tri[i] = halfedges[face + i].startVert;
    }
  }
};
}  // namespace

namespace manifold {

Manifold::Manifold() : pImpl_{std::make_unique<Impl>()} {}
Manifold::Manifold(const Mesh& manifold)
    : pImpl_{std::make_unique<Impl>(manifold)} {}
Manifold::~Manifold() = default;
Manifold::Manifold(Manifold&&) noexcept = default;
Manifold& Manifold::operator=(Manifold&&) noexcept = default;

Manifold::Manifold(const Manifold& other) : pImpl_(new Impl(*other.pImpl_)) {}

Manifold& Manifold::operator=(const Manifold& other) {
  if (this != &other) {
    pImpl_.reset(new Impl(*other.pImpl_));
  }
  return *this;
}

/**
 * Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
 * and the rest at similarly symmetric points.
 */
Manifold Manifold::Tetrahedron() {
  Manifold tetrahedron;
  tetrahedron.pImpl_ = std::make_unique<Impl>(Impl::Shape::TETRAHEDRON);
  return tetrahedron;
}

/**
 * Constructs an octahedron centered at the origin with vertices one unit out
 * along each axis.
 */
Manifold Manifold::Octahedron() {
  Manifold octahedron;
  octahedron.pImpl_ = std::make_unique<Impl>(Impl::Shape::OCTAHEDRON);
  return octahedron;
}

/**
 * Constructs a unit cube (edge lengths all one), by default in the first
 * octant, touching the origin. Set center to true to shift the center to the
 * origin.
 */
Manifold Manifold::Cube(glm::vec3 size, bool center) {
  Manifold cube;
  cube.pImpl_ = std::make_unique<Impl>(Impl::Shape::CUBE);
  cube.Scale(size);
  if (center) cube.Translate(-size / 2.0f);
  return cube;
}

/**
 * A convenience constructor for the common case of extruding a circle. Can also
 * form cones if both radii are specified. Set center to true to center the
 * manifold vertically on the origin (default places the bottom on the origin).
 */
Manifold Manifold::Cylinder(float height, float radiusLow, float radiusHigh,
                            int circularSegments, bool center) {
  float scale = radiusHigh >= 0.0f ? radiusHigh / radiusLow : 1.0f;
  float radius = max(radiusLow, radiusHigh);
  int n = circularSegments > 2 ? circularSegments : GetCircularSegments(radius);
  Polygons circle(1);
  float dPhi = 360.0f / n;
  for (int i = 0; i < n; ++i) {
    circle[0].push_back(
        {radiusLow * glm::vec2(cosd(dPhi * i), sind(dPhi * i)), 0});
  }
  Manifold cylinder =
      Manifold::Extrude(circle, height, 0, 0.0f, glm::vec2(scale));
  if (center) cylinder.Translate(glm::vec3(0.0f, 0.0f, -height / 2.0f));
  return cylinder;
}

/**
 * Constructs a sphere of a given radius and number of segments along its
 * diameter. This number will always be rounded up to the nearest factor of
 * four, as this sphere is constructed by refining an octahedron. This means
 * there are a circle of vertices on all three of the axis planes.
 */
Manifold Manifold::Sphere(float radius, int circularSegments) {
  int n = circularSegments > 0 ? (circularSegments + 3) / 4
                               : GetCircularSegments(radius) / 4;
  Manifold sphere;
  sphere.pImpl_ = std::make_unique<Impl>(Impl::Shape::OCTAHEDRON);
  sphere.pImpl_->Refine(n);
  thrust::for_each_n(sphere.pImpl_->vertPos_.beginD(), sphere.NumVert(),
                     ToSphere({radius}));
  sphere.pImpl_->Finish();
  return sphere;
}

/**
 * Constructs a manifold from a set of polygons by extruding them along the
 * Z-axis. The overall height and the scale at the top (X and Y independently)
 * can be specified, as can a twist, to be applied linearly. In the case of
 * twist, it can also be helpful to specify nDivisions, which specifies the
 * quantization of the triangles vertically. If the scale is {0,0}, a pure cone
 * is formed with only a single vertex at the top.
 */
Manifold Manifold::Extrude(Polygons crossSection, float height, int nDivisions,
                           float twistDegrees, glm::vec2 scaleTop) {
  ALWAYS_ASSERT(scaleTop.x >= 0 && scaleTop.y >= 0, userErr,
                "scale values cannot be negative");
  Manifold extrusion;
  ++nDivisions;
  auto& vertPos = extrusion.pImpl_->vertPos_.H();
  VecDH<glm::ivec3> triVertsDH;
  auto& triVerts = triVertsDH.H();
  int nCrossSection = 0;
  bool isCone = scaleTop.x == 0.0 && scaleTop.y == 0.0;
  int idx = 0;
  for (auto& poly : crossSection) {
    nCrossSection += poly.size();
    for (PolyVert& polyVert : poly) {
      vertPos.push_back({polyVert.pos.x, polyVert.pos.y, 0.0f});
      polyVert.idx = idx++;
    }
  }
  for (int i = 1; i < nDivisions + 1; ++i) {
    float alpha = i / float(nDivisions);
    float phi = alpha * twistDegrees;
    glm::mat2 transform(cosd(phi), sind(phi), -sind(phi), cosd(phi));
    glm::vec2 scale = glm::mix(glm::vec2(1.0f), scaleTop, alpha);
    transform = transform * glm::mat2(scale.x, 0.0f, 0.0f, scale.y);
    int j = 0;
    int idx = 0;
    for (const auto& poly : crossSection) {
      for (int vert = 0; vert < poly.size(); ++vert) {
        int offset = idx + nCrossSection * i;
        int thisVert = vert + offset;
        int lastVert = (vert == 0 ? poly.size() : vert) - 1 + offset;
        if (i == nDivisions && isCone) {
          triVerts.push_back({nCrossSection * i + j, lastVert - nCrossSection,
                              thisVert - nCrossSection});
        } else {
          glm::vec2 pos = transform * poly[vert].pos;
          vertPos.push_back({pos.x, pos.y, height * alpha});
          triVerts.push_back({thisVert, lastVert, thisVert - nCrossSection});
          triVerts.push_back(
              {lastVert, lastVert - nCrossSection, thisVert - nCrossSection});
        }
      }
      ++j;
      idx += poly.size();
    }
  }
  if (isCone)
    for (int j = 0; j < crossSection.size(); ++j)  // Duplicate vertex for Genus
      vertPos.push_back({0.0f, 0.0f, height});
  std::vector<glm::ivec3> top = Triangulate(crossSection);
  for (const glm::ivec3& tri : top) {
    triVerts.push_back({tri[0], tri[2], tri[1]});
    if (!isCone) triVerts.push_back(tri + nCrossSection * nDivisions);
  }

  extrusion.pImpl_->CreateHalfedges(triVertsDH);
  extrusion.pImpl_->Finish();
  return extrusion;
}

/**
 * Constructs a manifold from a set of polygons by revolving this cross-section
 * around its Y-axis and then setting this as the Z-axis of the resulting
 * manifold. If the polygons cross the Y-axis, only the part on the positive X
 * side is used. Geometrically valid input will result in geometrically valid
 * output.
 */
Manifold Manifold::Revolve(const Polygons& crossSection, int circularSegments) {
  float radius = 0.0f;
  for (const auto& poly : crossSection) {
    for (const auto& vert : poly) {
      radius = max(radius, vert.pos.x);
    }
  }
  int nDivisions =
      circularSegments > 2 ? circularSegments : GetCircularSegments(radius);
  Manifold revoloid;
  auto& vertPos = revoloid.pImpl_->vertPos_.H();
  VecDH<glm::ivec3> triVertsDH;
  auto& triVerts = triVertsDH.H();
  float dPhi = 360.0f / nDivisions;
  for (const auto& poly : crossSection) {
    int start = -1;
    for (int polyVert = 0; polyVert < poly.size(); ++polyVert) {
      if (poly[polyVert].pos.x <= 0) {
        start = polyVert;
        break;
      }
    }
    if (start == -1) {  // poly all positive
      for (int polyVert = 0; polyVert < poly.size(); ++polyVert) {
        int startVert = vertPos.size();
        int lastStart =
            startVert +
            (polyVert == 0 ? nDivisions * (poly.size() - 1) : -nDivisions);
        for (int slice = 0; slice < nDivisions; ++slice) {
          int lastSlice = (slice == 0 ? nDivisions : slice) - 1;
          float phi = slice * dPhi;
          glm::vec2 pos = poly[polyVert].pos;
          vertPos.push_back({pos.x * cosd(phi), pos.x * sind(phi), pos.y});
          triVerts.push_back({startVert + slice, startVert + lastSlice,
                              lastStart + lastSlice});
          triVerts.push_back(
              {lastStart + lastSlice, lastStart + slice, startVert + slice});
        }
      }
    } else {  // poly crosses zero
      int polyVert = start;
      glm::vec2 pos = poly[polyVert].pos;
      do {
        glm::vec2 lastPos = pos;
        polyVert = (polyVert + 1) % poly.size();
        pos = poly[polyVert].pos;
        if (pos.x > 0) {
          if (lastPos.x <= 0) {
            float a = pos.x / (pos.x - lastPos.x);
            vertPos.push_back({0.0f, 0.0f, glm::mix(pos.y, lastPos.y, a)});
          }
          int startVert = vertPos.size();
          for (int slice = 0; slice < nDivisions; ++slice) {
            int lastSlice = (slice == 0 ? nDivisions : slice) - 1;
            float phi = slice * dPhi;
            glm::vec2 pos = poly[polyVert].pos;
            vertPos.push_back({pos.x * cosd(phi), pos.x * sind(phi), pos.y});
            if (lastPos.x > 0) {
              triVerts.push_back({startVert + slice, startVert + lastSlice,
                                  startVert - nDivisions + lastSlice});
              triVerts.push_back({startVert - nDivisions + lastSlice,
                                  startVert - nDivisions + slice,
                                  startVert + slice});
            } else {
              triVerts.push_back(
                  {startVert - 1, startVert + slice, startVert + lastSlice});
            }
          }
        } else if (lastPos.x > 0) {
          int startVert = vertPos.size();
          float a = pos.x / (pos.x - lastPos.x);
          vertPos.push_back({0.0f, 0.0f, glm::mix(pos.y, lastPos.y, a)});
          for (int slice = 0; slice < nDivisions; ++slice) {
            int lastSlice = (slice == 0 ? nDivisions : slice) - 1;
            triVerts.push_back({startVert, startVert - nDivisions + lastSlice,
                                startVert - nDivisions + slice});
          }
        }
      } while (polyVert != start);
    }
  }

  revoloid.pImpl_->CreateHalfedges(triVertsDH);
  revoloid.pImpl_->Finish();
  return revoloid;
}

/**
 * Constructs a new manifold from a vector of other manifolds. This is a purely
 * topological operation, so care should be taken to avoid creating
 * geometrically-invalid results.
 */
Manifold Manifold::Compose(const std::vector<Manifold>& manifolds) {
  int numVert = 0;
  int numEdge = 0;
  int NumTri = 0;
  for (const Manifold& manifold : manifolds) {
    numVert += manifold.NumVert();
    numEdge += manifold.NumEdge();
    NumTri += manifold.NumTri();
  }

  Manifold out;
  Impl& combined = *(out.pImpl_);
  combined.vertPos_.resize(numVert);
  combined.halfedge_.resize(2 * numEdge);
  combined.faceNormal_.resize(NumTri);

  int nextVert = 0;
  int nextEdge = 0;
  int nextFace = 0;
  for (const Manifold& manifold : manifolds) {
    const Impl& impl = *(manifold.pImpl_);
    impl.ApplyTransform();

    thrust::copy(impl.vertPos_.beginD(), impl.vertPos_.endD(),
                 combined.vertPos_.beginD() + nextVert);
    thrust::copy(impl.faceNormal_.beginD(), impl.faceNormal_.endD(),
                 combined.faceNormal_.beginD() + nextFace);
    thrust::transform(impl.halfedge_.beginD(), impl.halfedge_.endD(),
                      combined.halfedge_.beginD() + nextEdge,
                      UpdateHalfedge({nextVert, nextEdge, nextFace}));

    nextVert += manifold.NumVert();
    nextEdge += 2 * manifold.NumEdge();
    nextFace += manifold.NumTri();
  }

  combined.Finish();
  return out;
}

/**
 * This operation returns a copy of this manifold, but as a vector of meshes
 * that are topologically disconnected.
 */
std::vector<Manifold> Manifold::Decompose() const {
  VecDH<int> vertLabel;
  int numLabel = ConnectedComponents(vertLabel, NumVert(), pImpl_->halfedge_);

  if (numLabel == 1) {
    std::vector<Manifold> meshes(1);
    meshes[0] = *this;
    return meshes;
  }

  std::vector<Manifold> meshes(numLabel);
  for (int i = 0; i < numLabel; ++i) {
    meshes[i].pImpl_->vertPos_.resize(NumVert());
    VecDH<int> vertNew2Old(NumVert());
    int nVert =
        thrust::copy_if(
            zip(pImpl_->vertPos_.beginD(), countAt(0)),
            zip(pImpl_->vertPos_.endD(), countAt(NumVert())),
            vertLabel.beginD(),
            zip(meshes[i].pImpl_->vertPos_.beginD(), vertNew2Old.beginD()),
            Equals({i})) -
        zip(meshes[i].pImpl_->vertPos_.beginD(), countAt(0));
    meshes[i].pImpl_->vertPos_.resize(nVert);

    VecDH<int> faceNew2Old(NumTri());
    thrust::sequence(faceNew2Old.beginD(), faceNew2Old.endD());

    int nFace =
        thrust::remove_if(
            faceNew2Old.beginD(), faceNew2Old.endD(),
            RemoveFace({pImpl_->halfedge_.cptrD(), vertLabel.cptrD(), i})) -
        faceNew2Old.beginD();
    faceNew2Old.resize(nFace);

    meshes[i].pImpl_->GatherFaces(pImpl_->halfedge_, faceNew2Old);
    meshes[i].pImpl_->ReindexVerts(vertNew2Old, pImpl_->NumVert());

    meshes[i].pImpl_->Finish();
    meshes[i].pImpl_->transform_ = pImpl_->transform_;
  }
  return meshes;
}

/**
 * This returns a Mesh of simple vectors of vertices and triangles suitable for
 * saving or other operations outside of the context of this library.
 */
Mesh Manifold::Extract(bool includeNormals) const {
  pImpl_->ApplyTransform();

  Mesh result;
  result.vertPos.insert(result.vertPos.end(), pImpl_->vertPos_.begin(),
                        pImpl_->vertPos_.end());
  if (includeNormals) {
    result.vertNormal.insert(result.vertNormal.end(),
                             pImpl_->vertNormal_.begin(),
                             pImpl_->vertNormal_.end());
  }

  result.triVerts.resize(NumTri());
  thrust::for_each_n(zip(result.triVerts.begin(), countAt(0)), NumTri(),
                     MakeTri({pImpl_->halfedge_.cptrH()}));

  return result;
}

/**
 * These static properties control how circular shapes are quantized by default
 * on construction. If circularSegments is specified, it takes precedence. If it
 * is zero, then instead the minimum is used of the segments calculated based on
 * edge length and angle, rounded up to the nearest multiple of four. To get
 * numbers not divisible by four, circularSegements must be specified.
 */
int Manifold::circularSegments = 0;
float Manifold::circularAngle = 10.0f;
float Manifold::circularEdgeLength = 1.0f;

void Manifold::SetMinCircularAngle(float angle) {
  ALWAYS_ASSERT(angle > 0.0f, userErr, "angle must be positive!");
  Manifold::circularAngle = angle;
}

void Manifold::SetMinCircularEdgeLength(float length) {
  ALWAYS_ASSERT(length > 0.0f, userErr, "length must be positive!");
  Manifold::circularEdgeLength = length;
}

void Manifold::SetCircularSegments(int number) {
  ALWAYS_ASSERT(number > 2 || number == 0, userErr,
                "must have at least three segments in circle!");
  Manifold::circularSegments = number;
}

int Manifold::GetCircularSegments(float radius) {
  if (Manifold::circularSegments > 0) return Manifold::circularSegments;
  int nSegA = 360.0f / Manifold::circularAngle;
  int nSegL = 2.0f * radius * glm::pi<float>() / Manifold::circularEdgeLength;
  int nSeg = min(nSegA, nSegL) + 3;
  nSeg -= nSeg % 4;
  return nSeg;
}

bool Manifold::IsEmpty() const { return pImpl_->IsEmpty(); }
int Manifold::NumVert() const { return pImpl_->NumVert(); }
int Manifold::NumEdge() const { return pImpl_->NumEdge(); }
int Manifold::NumTri() const { return pImpl_->NumTri(); }

Box Manifold::BoundingBox() const {
  return pImpl_->bBox_.Transform(pImpl_->transform_);
}

float Manifold::Precision() const {
  pImpl_->ApplyTransform();
  return pImpl_->precision_;
}

/**
 * The genus is a topological property of the manifold, representing the number
 * of "handles". A sphere is 0, torus 1, etc. It is only meaningful for a single
 * mesh, so it is best to call Decompose() first.
 */
int Manifold::Genus() const {
  int chi = NumVert() - NumEdge() + NumTri();
  return 1 - chi / 2;
}

Manifold::Properties Manifold::GetProperties() const {
  return pImpl_->GetProperties();
}

bool Manifold::IsManifold() const { return pImpl_->IsManifold(); }

bool Manifold::MatchesTriNormals() const { return pImpl_->MatchesTriNormals(); }

Manifold& Manifold::Translate(glm::vec3 v) {
  pImpl_->transform_[3] += v;
  return *this;
}

Manifold& Manifold::Scale(glm::vec3 v) {
  glm::mat3 s(1.0f);
  for (int i : {0, 1, 2}) s[i] *= v;
  pImpl_->transform_ = s * pImpl_->transform_;
  return *this;
}

/**
 * Applys an Euler angle rotation to the manifold, first about the X axis, then
 * Y, then Z, in degrees. We use degrees so that we can minimize rounding error,
 * and elimiate it completely for any multiples of 90 degrees. Addtionally, more
 * efficient code paths are used to update the manifold when the transforms only
 * rotate by multiples of 90 degrees.
 */
Manifold& Manifold::Rotate(float xDegrees, float yDegrees, float zDegrees) {
  glm::mat3 rX(1.0f, 0.0f, 0.0f,                      //
               0.0f, cosd(xDegrees), sind(xDegrees),  //
               0.0f, -sind(xDegrees), cosd(xDegrees));
  glm::mat3 rY(cosd(yDegrees), 0.0f, -sind(yDegrees),  //
               0.0f, 1.0f, 0.0f,                       //
               sind(yDegrees), 0.0f, cosd(yDegrees));
  glm::mat3 rZ(cosd(zDegrees), sind(zDegrees), 0.0f,   //
               -sind(zDegrees), cosd(zDegrees), 0.0f,  //
               0.0f, 0.0f, 1.0f);
  pImpl_->transform_ = rZ * rY * rX * pImpl_->transform_;
  return *this;
}

/**
 * This function does not change the topology, but allows the vertices to be
 * moved according to any arbitrary input function. It is easy to create a
 * function that warps a geometrically valid object into one with is not, but
 * that is not checked here, so it is up to the user to choose their function
 * with discretion.
 */
Manifold& Manifold::Warp(std::function<void(glm::vec3&)> warpFunc) {
  pImpl_->ApplyTransform();
  thrust::for_each_n(pImpl_->vertPos_.begin(), NumVert(), warpFunc);
  pImpl_->Update();
  pImpl_->faceNormal_.resize(0);  // force recalculation of triNormal
  pImpl_->CalculateNormals();
  pImpl_->SetPrecision();
  return *this;
}

/**
 * This is a checksum-style verification of the collider, simply returning the
 * total number of edge-face bounding box overlaps between this and other.
 */
int Manifold::NumOverlaps(const Manifold& other) const {
  pImpl_->ApplyTransform();
  other.pImpl_->ApplyTransform();

  SparseIndices overlaps = pImpl_->EdgeCollisions(*other.pImpl_);
  int num_overlaps = overlaps.size();

  overlaps = other.pImpl_->EdgeCollisions(*pImpl_);
  return num_overlaps += overlaps.size();
}

Manifold Manifold::Boolean(const Manifold& second, OpType op) const {
  pImpl_->ApplyTransform();
  second.pImpl_->ApplyTransform();
  Boolean3 boolean(*pImpl_, *second.pImpl_, op);
  Manifold result;
  result.pImpl_ = std::make_unique<Impl>(boolean.Result(op));
  return result;
}

Manifold Manifold::operator+(const Manifold& Q) const {
  return Boolean(Q, OpType::ADD);
}

Manifold& Manifold::operator+=(const Manifold& Q) {
  *this = *this + Q;
  return *this;
}

Manifold Manifold::operator-(const Manifold& Q) const {
  return Boolean(Q, OpType::SUBTRACT);
}

Manifold& Manifold::operator-=(const Manifold& Q) {
  *this = *this - Q;
  return *this;
}

Manifold Manifold::operator^(const Manifold& Q) const {
  return Boolean(Q, OpType::INTERSECT);
}

Manifold& Manifold::operator^=(const Manifold& Q) {
  *this = *this ^ Q;
  return *this;
}

std::pair<Manifold, Manifold> Manifold::Split(const Manifold& cutter) const {
  pImpl_->ApplyTransform();
  cutter.pImpl_->ApplyTransform();
  Boolean3 boolean(*pImpl_, *cutter.pImpl_, OpType::SUBTRACT);
  std::pair<Manifold, Manifold> result;
  result.first.pImpl_ =
      std::make_unique<Impl>(boolean.Result(OpType::INTERSECT));
  result.second.pImpl_ =
      std::make_unique<Impl>(boolean.Result(OpType::SUBTRACT));
  return result;
}

std::pair<Manifold, Manifold> Manifold::SplitByPlane(glm::vec3 normal,
                                                     float originOffset) const {
  normal = glm::normalize(normal);
  Manifold cutter =
      Manifold::Cube(glm::vec3(2.0f), true).Translate({1.0f, 0.0f, 0.0f});
  float size = glm::length(BoundingBox().Center() - normal * originOffset) +
               0.5f * glm::length(BoundingBox().Size());
  cutter.Scale(glm::vec3(size)).Translate({originOffset, 0.0f, 0.0f});
  float yDeg = glm::degrees(-glm::asin(normal.z));
  float zDeg = glm::degrees(glm::atan(normal.y, normal.x));
  cutter.Rotate(0.0f, yDeg, zDeg);
  return Split(cutter);
}
}  // namespace manifold