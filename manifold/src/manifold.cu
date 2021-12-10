// Copyright 2021 Emmett Lalish
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

#include "boolean3.cuh"
#include "impl.cuh"

namespace {
using namespace manifold;
using namespace thrust::placeholders;

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

struct GetMeshID {
  __host__ __device__ void operator()(thrust::tuple<int&, BaryRef> inOut) {
    thrust::get<0>(inOut) = thrust::get<1>(inOut).meshID;
  }
};

Manifold Halfspace(Box bBox, glm::vec3 normal, float originOffset) {
  normal = glm::normalize(normal);
  Manifold cutter =
      Manifold::Cube(glm::vec3(2.0f), true).Translate({1.0f, 0.0f, 0.0f});
  float size = glm::length(bBox.Center() - normal * originOffset) +
               0.5f * glm::length(bBox.Size());
  cutter.Scale(glm::vec3(size)).Translate({originOffset, 0.0f, 0.0f});
  float yDeg = glm::degrees(-glm::asin(normal.z));
  float zDeg = glm::degrees(glm::atan(normal.y, normal.x));
  return cutter.Rotate(0.0f, yDeg, zDeg);
}
}  // namespace

namespace manifold {

Manifold::Manifold() : pImpl_{std::make_unique<Impl>()} {}
Manifold::Manifold(const Mesh& mesh) : pImpl_{std::make_unique<Impl>(mesh)} {}
Manifold::~Manifold() = default;
Manifold::Manifold(Manifold&&) noexcept = default;
Manifold& Manifold::operator=(Manifold&&) noexcept = default;

Manifold::Manifold(const Manifold& other) : pImpl_(new Impl(*other.pImpl_)) {
  pImpl_->DuplicateMeshIDs();
}

Manifold& Manifold::operator=(const Manifold& other) {
  if (this != &other) {
    pImpl_.reset(new Impl(*other.pImpl_));
    pImpl_->DuplicateMeshIDs();
  }
  return *this;
}

/**
 * This returns a Mesh of simple vectors of vertices and triangles suitable for
 * saving or other operations outside of the context of this library.
 */
Mesh Manifold::GetMesh() const {
  pImpl_->ApplyTransform();

  Mesh result;
  result.vertPos.insert(result.vertPos.end(), pImpl_->vertPos_.begin(),
                        pImpl_->vertPos_.end());
  result.vertNormal.insert(result.vertNormal.end(), pImpl_->vertNormal_.begin(),
                           pImpl_->vertNormal_.end());
  result.halfedgeTangent.insert(result.halfedgeTangent.end(),
                                pImpl_->halfedgeTangent_.begin(),
                                pImpl_->halfedgeTangent_.end());

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
int Manifold::circularSegments_ = 0;
float Manifold::circularAngle_ = 10.0f;
float Manifold::circularEdgeLength_ = 1.0f;

void Manifold::SetMinCircularAngle(float angle) {
  ALWAYS_ASSERT(angle > 0.0f, userErr, "angle must be positive!");
  Manifold::circularAngle_ = angle;
}

void Manifold::SetMinCircularEdgeLength(float length) {
  ALWAYS_ASSERT(length > 0.0f, userErr, "length must be positive!");
  Manifold::circularEdgeLength_ = length;
}

void Manifold::SetCircularSegments(int number) {
  ALWAYS_ASSERT(number > 2 || number == 0, userErr,
                "must have at least three segments in circle!");
  Manifold::circularSegments_ = number;
}

int Manifold::GetCircularSegments(float radius) {
  if (Manifold::circularSegments_ > 0) return Manifold::circularSegments_;
  int nSegA = 360.0f / Manifold::circularAngle_;
  int nSegL = 2.0f * radius * glm::pi<float>() / Manifold::circularEdgeLength_;
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

/**
 * Returns the surface area and volume of the manifold in a Properties
 * structure. These properties are clamped to zero for a given face if they are
 * within rounding tolerance. This means degenerate manifolds can by identified
 * by testing these properties as == 0.
 */
Properties Manifold::GetProperties() const { return pImpl_->GetProperties(); }

/**
 * Curvature is the inverse of the radius of curvature, and signed such that
 * positive is convex and negative is concave. There are two orthogonal
 * principal curvatures at any point on a manifold, with one maximum and the
 * other minimum. Gaussian curvature is their product, while mean
 * curvature is their sum. This approximates them for every vertex (returned as
 * vectors in the structure) and also returns their minimum and maximum values.
 */
Curvature Manifold::GetCurvature() const { return pImpl_->GetCurvature(); }

/**
 * Gets the relationship to the previous mesh, for the purpose of assinging
 * properties like texture coordinates. The triBary vector is the same length as
 * Mesh.triVerts and BaryRef.face gives a unique identifier of the original mesh
 * face to which this triangle belongs. BaryRef.verts gives the three original
 * mesh vertex indices to which its barycentric coordinates refer.
 * BaryRef.vertBary gives an index for each vertex into the barycentric vector
 * if that index is >= 0, indicating it is a new vertex. If the index is < 0,
 * this indicates it is an original vertex of the triangle, found as the
 * corresponding element of BaryRef.verts.
 */
MeshRelation Manifold::GetMeshRelation() const {
  MeshRelation out;
  const auto& relation = pImpl_->meshRelation_;
  out.triBary.insert(out.triBary.end(), relation.triBary.begin(),
                     relation.triBary.end());
  out.barycentric.insert(out.barycentric.end(), relation.barycentric.begin(),
                         relation.barycentric.end());
  return out;
}

/**
 * Returns a vector of unique meshIDs that are referenced by this manifold's
 * meshRelation. If this manifold has been newly constructed then there will
 * only be a single meshID, which can be associated with the input mesh for
 * future reference.
 */
std::vector<int> Manifold::GetMeshIDs() const {
  VecDH<int> meshIDs(NumTri());
  thrust::for_each_n(
      zip(meshIDs.beginD(), pImpl_->meshRelation_.triBary.beginD()), NumTri(),
      GetMeshID());

  thrust::sort(meshIDs.beginD(), meshIDs.endD());
  int n = thrust::unique(meshIDs.beginD(), meshIDs.endD()) - meshIDs.beginD();
  meshIDs.resize(n);

  std::vector<int> out;
  out.insert(out.end(), meshIDs.begin(), meshIDs.end());
  return out;
}

/**
 * If you copy a manifold, but you want this new copy to have new properties
 * (e.g. a different UV mapping), you can reset its meshID as an original,
 * meaning it will now be referenced by its descendents instead of the mesh it
 * was copied from, allowing you to differentiate the copies when applying your
 * properties to the final result. Its new meshID is returned.
 */
int Manifold::SetAsOriginal(bool mergeCoplanarRelations) {
  int meshID = pImpl_->InitializeNewReference();
  if (mergeCoplanarRelations) pImpl_->MergeCoplanarRelations();
  return meshID;
}

std::vector<int> Manifold::MeshID2Original() {
  return Manifold::Impl::meshID2Original_;
}

bool Manifold::IsManifold() const { return pImpl_->IsManifold(); }

bool Manifold::MatchesTriNormals() const { return pImpl_->MatchesTriNormals(); }

int Manifold::NumDegenerateTris() const { return pImpl_->NumDegenerateTris(); }

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

Manifold& Manifold::Transform(const glm::mat4x3& m) {
  glm::mat4 old(pImpl_->transform_);
  pImpl_->transform_ = m * old;
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

Manifold& Manifold::Refine(int n) {
  pImpl_->Refine(n);
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

/**
 * Split cuts this manifold in two using the input manifold. The first result is
 * the intersection, second is the difference. This is more efficient than doing
 * them separately.
 */
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

/**
 * Convient version of Split for a half-space. The first result is in the
 * direction of the normal, second is opposite. Origin offset is the distance of
 * the plane from the origin in the direction of the normal vector. The length
 * of the normal is not important, as it is normalized internally.
 */
std::pair<Manifold, Manifold> Manifold::SplitByPlane(glm::vec3 normal,
                                                     float originOffset) const {
  return Split(Halfspace(BoundingBox(), normal, originOffset));
}

/**
 * Identical to SplitbyPlane, but calculating and returning only the first
 * result.
 */
Manifold Manifold::TrimByPlane(glm::vec3 normal, float originOffset) const {
  pImpl_->ApplyTransform();
  return *this ^ Halfspace(BoundingBox(), normal, originOffset);
}
}  // namespace manifold