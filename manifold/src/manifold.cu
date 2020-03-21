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

#include <thrust/transform_reduce.h>

#include "boolean3.cuh"
#include "connected_components.cuh"
#include "manifold.cuh"
#include "polygon.h"

namespace {
using namespace manifold;

struct NormalizeTo {
  float length;
  __host__ __device__ void operator()(glm::vec3& v) {
    v = length * glm::normalize(v);
    if (isnan(v.x)) v = glm::vec3(0.0);
  }
};

struct Positive {
  __host__ __device__ bool operator()(int x) { return x >= 0; }
};

struct Equals {
  int val;
  __host__ __device__ bool operator()(int x) { return x == val; }
};

struct KeepTri {
  int val;
  const int* components;
  __host__ __device__ bool operator()(glm::ivec3 tri) {
    return components[tri[0]] == val;
  }
};

struct Reindex {
  const int* indexInv_;

  __host__ __device__ void operator()(glm::ivec3& triVerts) {
    for (int i : {0, 1, 2}) triVerts[i] = indexInv_[triVerts[i]];
  }
};

struct TetVolume {
  const glm::vec3* vertPos;

  __host__ __device__ float operator()(const glm::ivec3& triVerts) {
    return glm::dot(vertPos[triVerts[0]],
                    glm::cross(vertPos[triVerts[1]], vertPos[triVerts[2]])) /
           6;
  }
};

struct TriArea {
  const glm::vec3* vertPos;

  __host__ __device__ float operator()(const glm::ivec3& triVerts) {
    return 0.5 *
           glm::length(glm::cross(vertPos[triVerts[1]] - vertPos[triVerts[0]],
                                  vertPos[triVerts[2]] - vertPos[triVerts[0]]));
  }
};
}

namespace manifold {

Manifold::Manifold() : pImpl_{std::make_unique<Impl>()} {}
Manifold::Manifold(const Mesh& manifold)
    : pImpl_{std::make_unique<Impl>(manifold)} {}
Manifold::~Manifold() = default;
Manifold::Manifold(Manifold&&) = default;
Manifold& Manifold::operator=(Manifold&&) = default;

Manifold::Manifold(const Manifold& other) : pImpl_(new Impl(*other.pImpl_)) {}

Manifold& Manifold::operator=(const Manifold& other) {
  if (this != &other) {
    pImpl_.reset(new Impl(*other.pImpl_));
  }
  return *this;
}

Manifold Manifold::DeepCopy() const { return *this; }

Manifold Manifold::Tetrahedron() {
  Manifold tetrahedron;
  tetrahedron.pImpl_ = std::make_unique<Impl>(Impl::Shape::TETRAHEDRON);
  return tetrahedron;
}

Manifold Manifold::Octahedron() {
  Manifold octahedron;
  octahedron.pImpl_ = std::make_unique<Impl>(Impl::Shape::OCTAHEDRON);
  return octahedron;
}

Manifold Manifold::Cube(glm::vec3 size, bool center) {
  Manifold cube;
  cube.pImpl_ = std::make_unique<Impl>(Impl::Shape::CUBE);
  cube.Scale(size);
  if (center) cube.Translate(-size / 2.0f);
  return cube;
}

Manifold Manifold::Cylinder(float height, float radiusLow, float radiusHigh,
                            int circularSegments, bool center) {
  float scale = radiusHigh >= 0.0f ? radiusHigh / radiusLow : 1.0f;
  float radius = max(radiusLow, radiusHigh);
  int n = circularSegments > 2 ? circularSegments : GetCircularSegments(radius);
  Polygons circle(1);
  float dPhi = 360.0f / n;
  for (int i = 0; i < n; ++i) {
    circle[0].push_back({radiusLow * glm::vec2(cosd(dPhi * i), sind(dPhi * i)),
                         0, Edge::kNoIdx});
  }
  Manifold cylinder =
      Manifold::Extrude(circle, height, 0, 0.0f, glm::vec2(scale));
  if (center) cylinder.Translate(glm::vec3(0.0f, 0.0f, -height / 2.0f));
  return cylinder;
}

Manifold Manifold::Sphere(float radius, int circularSegments) {
  int n = circularSegments > 0 ? (circularSegments + 3) / 4
                               : GetCircularSegments(radius) / 4;
  Manifold sphere;
  sphere.pImpl_ = std::make_unique<Impl>(Impl::Shape::OCTAHEDRON);
  sphere.pImpl_->Refine(n);
  thrust::for_each_n(sphere.pImpl_->vertPos_.beginD(), sphere.NumVert(),
                     NormalizeTo({radius}));
  sphere.pImpl_->Finish();
  return sphere;
}

Manifold Manifold::Extrude(Polygons crossSection, float height, int nDivisions,
                           float twistDegrees, glm::vec2 scaleTop) {
  ALWAYS_ASSERT(scaleTop.x >= 0 && scaleTop.y >= 0, runtimeErr, "");
  Manifold extrusion;
  ++nDivisions;
  auto& vertPos = extrusion.pImpl_->vertPos_.H();
  auto& triVerts = extrusion.pImpl_->triVerts_.H();
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
  extrusion.pImpl_->Finish();
  return extrusion;
}

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
  auto& triVerts = revoloid.pImpl_->triVerts_.H();
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
  revoloid.pImpl_->Finish();
  return revoloid;
}

Manifold Manifold::Compose(const std::vector<Manifold>& manifolds) {
  Manifold combined;
  for (Manifold manifold : manifolds) {
    manifold.pImpl_->ApplyTransform();
    const int startIdx = combined.NumVert();
    combined.pImpl_->vertPos_.H().insert(combined.pImpl_->vertPos_.end(),
                                         manifold.pImpl_->vertPos_.begin(),
                                         manifold.pImpl_->vertPos_.end());
    for (auto tri : manifold.pImpl_->triVerts_.H())
      combined.pImpl_->triVerts_.H().push_back(tri + startIdx);
  }
  combined.pImpl_->Finish();
  return combined;
}

std::vector<Manifold> Manifold::Decompose() const {
  VecDH<int> components;
  int nManifolds =
      ConnectedComponents(components, NumVert(), pImpl_->edgeVerts_);
  std::vector<Manifold> meshes(nManifolds);
  VecDH<int> vertOld2New(NumVert(), -1);
  for (int i = 0; i < nManifolds; ++i) {
    int compVert =
        thrust::find_if(components.beginD(), components.endD(), Positive()) -
        components.beginD();
    int compLabel = components.H()[compVert];

    meshes[i].pImpl_->vertPos_.resize(NumVert());
    VecDH<int> vertNew2Old(NumVert());
    int nVert =
        thrust::copy_if(
            zip(pImpl_->vertPos_.beginD(), thrust::make_counting_iterator(0)),
            zip(pImpl_->vertPos_.endD(),
                thrust::make_counting_iterator(NumVert())),
            components.beginD(),
            zip(meshes[i].pImpl_->vertPos_.beginD(), vertNew2Old.beginD()),
            Equals({compLabel})) -
        zip(meshes[i].pImpl_->vertPos_.beginD(),
            thrust::make_counting_iterator(0));
    thrust::scatter(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(nVert), vertNew2Old.beginD(),
                    vertOld2New.beginD());
    meshes[i].pImpl_->vertPos_.resize(nVert);

    meshes[i].pImpl_->triVerts_.resize(NumTri());
    int nTri =
        thrust::copy_if(pImpl_->triVerts_.beginD(), pImpl_->triVerts_.endD(),
                        meshes[i].pImpl_->triVerts_.beginD(),
                        KeepTri({compLabel, components.ptrD()})) -
        meshes[i].pImpl_->triVerts_.beginD();
    meshes[i].pImpl_->triVerts_.resize(nTri);

    thrust::for_each_n(meshes[i].pImpl_->triVerts_.beginD(), nTri,
                       Reindex({vertOld2New.ptrD()}));

    meshes[i].pImpl_->Finish();
    meshes[i].pImpl_->transform_ = pImpl_->transform_;
    thrust::replace(components.beginD(), components.endD(), compLabel, -1);
  }
  return meshes;
}

Mesh Manifold::Extract() const {
  pImpl_->ApplyTransform();
  Mesh result;
  result.vertPos.insert(result.vertPos.end(), pImpl_->vertPos_.begin(),
                        pImpl_->vertPos_.end());
  result.triVerts.insert(result.triVerts.end(), pImpl_->triVerts_.begin(),
                         pImpl_->triVerts_.end());
  return result;
}

int Manifold::circularSegments = 0;
float Manifold::circularAngle = 10.0f;
float Manifold::circularEdgeLength = 1.0f;

void Manifold::SetMinCircularAngle(float angle) {
  ALWAYS_ASSERT(angle > 0.0f, runtimeErr, "angle must be positive!");
  Manifold::circularAngle = angle;
}

void Manifold::SetMinCircularEdgeLength(float length) {
  ALWAYS_ASSERT(length > 0.0f, runtimeErr, "length must be positive!");
  Manifold::circularEdgeLength = length;
}

void Manifold::SetCircularSegments(int number) {
  ALWAYS_ASSERT(number > 2, runtimeErr,
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

bool Manifold::IsEmpty() const { return NumVert() == 0; }
int Manifold::NumVert() const { return pImpl_->NumVert(); }
int Manifold::NumEdge() const { return pImpl_->NumEdge(); }
int Manifold::NumTri() const { return pImpl_->NumTri(); }

Box Manifold::BoundingBox() const {
  return pImpl_->bBox_.Transform(pImpl_->transform_);
}

float Manifold::Volume() const {
  pImpl_->ApplyTransform();
  return thrust::transform_reduce(
      pImpl_->triVerts_.beginD(), pImpl_->triVerts_.endD(),
      TetVolume({pImpl_->vertPos_.ptrD()}), 0.0f, thrust::plus<float>());
}

float Manifold::SurfaceArea() const {
  pImpl_->ApplyTransform();
  return thrust::transform_reduce(
      pImpl_->triVerts_.beginD(), pImpl_->triVerts_.endD(),
      TriArea({pImpl_->vertPos_.ptrD()}), 0.0f, thrust::plus<float>());
}

int Manifold::Genus() const {
  int chi = NumVert() - NumTri() / 2;
  return 1 - chi / 2;
}

bool Manifold::IsValid() const { return pImpl_->IsValid(); }

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

Manifold& Manifold::Warp(std::function<void(glm::vec3&)> warpFunc) {
  pImpl_->ApplyTransform();
  thrust::for_each_n(pImpl_->vertPos_.begin(), NumVert(), warpFunc);
  pImpl_->Update();
  pImpl_->triNormal_.resize(0);  // force recalculation of triNormal
  pImpl_->CalculateNormals();
  return *this;
}

int Manifold::NumOverlaps(const Manifold& B) const {
  pImpl_->ApplyTransform();
  B.pImpl_->ApplyTransform();

  SparseIndices overlaps = pImpl_->EdgeCollisions(*B.pImpl_);
  int num_overlaps = overlaps.size();

  overlaps = B.pImpl_->EdgeCollisions(*pImpl_);
  return num_overlaps += overlaps.size();
}

void Manifold::SetExpectGeometry(bool val) {
  PolygonParams().checkGeometry = val;
  PolygonParams().intermediateChecks = false;
}

void Manifold::SetSuppressErrors(bool val) {
  PolygonParams().suppressErrors = val;
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
}