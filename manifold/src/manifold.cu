// Copyright 2019 Emmett Lalish
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

#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <algorithm>

#include "boolean3.cuh"
#include "connected_components.cuh"
#include "manifold.cuh"

namespace {
using namespace manifold;

struct NormalizeTo {
  float length;
  __host__ __device__ void operator()(glm::vec3& v) {
    v = length * glm::normalize(v);
  }
};

struct Positive {
  __host__ __device__ bool operator()(int x) { return x >= 0; }
};

struct Equals {
  int val;
  __host__ __device__ bool operator()(int x) { return x == val; }
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

struct KeepTri {
  int val;
  const int* components;
  __host__ __device__ bool operator()(glm::ivec3 tri) {
    return components[tri[0]] == val;
  }
};

struct SplitEdges {
  glm::vec3* vertPos;
  const int startIdx;
  const int n;

  __host__ __device__ void operator()(thrust::tuple<int, EdgeVertsD> in) {
    int edge = thrust::get<0>(in);
    EdgeVertsD edgeVerts = thrust::get<1>(in);

    float invTotal = 1.0f / n;
    for (int i = 1; i < n; ++i)
      vertPos[startIdx + (n - 1) * edge + i - 1] =
          (float(n - i) * vertPos[edgeVerts.first] +
           float(i) * vertPos[edgeVerts.second]) *
          invTotal;
  }
};

struct InteriorVerts {
  glm::vec3* vertPos;
  const int startIdx;
  const int n;

  __host__ __device__ void operator()(thrust::tuple<int, glm::ivec3> in) {
    int tri = thrust::get<0>(in);
    glm::ivec3 triVerts = thrust::get<1>(in);

    int vertsPerTri = ((n - 2) * (n - 2) + (n - 2)) / 2;
    float invTotal = 1.0f / n;
    int pos = startIdx + vertsPerTri * tri;
    for (int i = 1; i < n - 1; ++i)
      for (int j = 1; j < n - i; ++j)
        vertPos[pos++] = (float(i) * vertPos[triVerts[2]] +  //
                          float(j) * vertPos[triVerts[0]] +  //
                          float(n - i - j) * vertPos[triVerts[1]]) *
                         invTotal;
  }
};

struct SplitTris {
  glm::ivec3* triVerts;
  const int edgeIdx;
  const int triIdx;
  const int n;

  __host__ __device__ int EdgeVert(int i, EdgeIdx edge) const {
    return edgeIdx + (n - 1) * edge.Idx() +
           (edge.Dir() > 0 ? i - 1 : n - 1 - i);
  }

  __host__ __device__ int TriVert(int i, int j, int tri) const {
    --i;
    --j;
    int m = n - 2;
    int vertsPerTri = (m * m + m) / 2;
    int vertOffset = (i * (2 * m - i + 1)) / 2 + j;
    return triIdx + vertsPerTri * tri + vertOffset;
  }

  __host__ __device__ int Vert(int i, int j, int tri, glm::ivec3 triVert,
                               TriEdges triEdge) const {
    bool edge0 = i == 0;
    bool edge1 = j == 0;
    bool edge2 = j == n - i;
    if (edge0) {
      if (edge1)
        return triVert[1];
      else if (edge2)
        return triVert[0];
      else
        return EdgeVert(n - j, triEdge[0]);
    } else if (edge1) {
      if (edge2)
        return triVert[2];
      else
        return EdgeVert(i, triEdge[1]);
    } else if (edge2)
      return EdgeVert(j, triEdge[2]);
    else
      return TriVert(i, j, tri);
  }

  __host__ __device__ void operator()(
      thrust::tuple<int, glm::ivec3, TriEdges> in) {
    int tri = thrust::get<0>(in);
    glm::ivec3 triVert = thrust::get<1>(in);
    TriEdges triEdge = thrust::get<2>(in);

    int pos = n * n * tri;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n - i; ++j) {
        int a = Vert(i, j, tri, triVert, triEdge);
        int b = Vert(i + 1, j, tri, triVert, triEdge);
        int c = Vert(i, j + 1, tri, triVert, triEdge);
        triVerts[pos++] = glm::ivec3(a, b, c);
        if (j < n - 1 - i) {
          int d = Vert(i + 1, j + 1, tri, triVert, triEdge);
          triVerts[pos++] = glm::ivec3(b, d, c);
        }
      }
    }
  }
};

struct IdxMin
    : public thrust::binary_function<glm::ivec3, glm::ivec3, glm::ivec3> {
  __host__ __device__ int min3(glm::ivec3 a) {
    return glm::min(a.x, glm::min(a.y, a.z));
  }
  __host__ __device__ glm::ivec3 operator()(glm::ivec3 a, glm::ivec3 b) {
    return glm::ivec3(glm::min(min3(a), min3(b)));
  }
};

struct IdxMax
    : public thrust::binary_function<glm::ivec3, glm::ivec3, glm::ivec3> {
  __host__ __device__ int max3(glm::ivec3 a) {
    return glm::max(a.x, glm::max(a.y, a.z));
  }
  __host__ __device__ glm::ivec3 operator()(glm::ivec3 a, glm::ivec3 b) {
    return glm::ivec3(glm::max(max3(a), max3(b)));
  }
};

struct PosMin
    : public thrust::binary_function<glm::vec3, glm::vec3, glm::vec3> {
  __host__ __device__ glm::vec3 operator()(glm::vec3 a, glm::vec3 b) {
    return glm::min(a, b);
  }
};

struct PosMax
    : public thrust::binary_function<glm::vec3, glm::vec3, glm::vec3> {
  __host__ __device__ glm::vec3 operator()(glm::vec3 a, glm::vec3 b) {
    return glm::max(a, b);
  }
};

struct Transform {
  const glm::mat4x3 transform;

  __host__ __device__ void operator()(glm::vec3& position) {
    position = transform * glm::vec4(position, 1.0f);
  }
};

__host__ __device__ uint32_t SpreadBits3(uint32_t v) {
  v = 0xFF0000FFu & (v * 0x00010001u);
  v = 0x0F00F00Fu & (v * 0x00000101u);
  v = 0xC30C30C3u & (v * 0x00000011u);
  v = 0x49249249u & (v * 0x00000005u);
  return v;
}

__host__ __device__ uint32_t MortonCode(glm::vec3 position, Box bBox) {
  glm::vec3 xyz = (position - bBox.min) / (bBox.max - bBox.min);
  xyz = glm::min(glm::vec3(1023.0f), glm::max(glm::vec3(0.0f), 1024.0f * xyz));
  uint32_t x = SpreadBits3(static_cast<uint32_t>(xyz.x));
  uint32_t y = SpreadBits3(static_cast<uint32_t>(xyz.y));
  uint32_t z = SpreadBits3(static_cast<uint32_t>(xyz.z));
  return x * 4 + y * 2 + z;
}

struct Morton {
  const Box bBox;

  __host__ __device__ void operator()(
      thrust::tuple<uint32_t&, const glm::vec3&> inout) {
    glm::vec3 position = thrust::get<1>(inout);
    thrust::get<0>(inout) = MortonCode(position, bBox);
  }
};

struct Reindex {
  const int* indexInv_;

  __host__ __device__ void operator()(glm::ivec3& triVerts) {
    for (int i : {0, 1, 2}) triVerts[i] = indexInv_[triVerts[i]];
  }
};

struct TriMortonBox {
  const glm::vec3* vertPos;
  const Box bBox;

  __host__ __device__ void operator()(
      thrust::tuple<uint32_t&, Box&, const glm::ivec3&> inout) {
    uint32_t& mortonCode = thrust::get<0>(inout);
    Box& triBox = thrust::get<1>(inout);
    const glm::ivec3& triVerts = thrust::get<2>(inout);

    glm::vec3 center =
        (vertPos[triVerts[0]] + vertPos[triVerts[1]] + vertPos[triVerts[2]]) /
        3.0f;
    mortonCode = MortonCode(center, bBox);
    triBox = Box(vertPos[triVerts[0]], vertPos[triVerts[1]]);
    triBox.Union(vertPos[triVerts[2]]);
  }
};

struct EdgeBox {
  const glm::vec3* vertPos;

  __host__ __device__ void operator()(thrust::tuple<Box&, EdgeVertsD> inout) {
    EdgeVertsD edgeVerts = thrust::get<1>(inout);
    thrust::get<0>(inout) =
        Box(vertPos[edgeVerts.first], vertPos[edgeVerts.second]);
  }
};

struct MakeHalfedges {
  int i, j;

  __host__ __device__ void operator()(
      thrust::tuple<TriEdges&, int&, EdgeVertsD&, const glm::ivec3&> inout) {
    const glm::ivec3& in = thrust::get<3>(inout);
    int V1 = in[i];
    int V2 = in[j];
    TriEdges& triEdges = thrust::get<0>(inout);
    int& dir = thrust::get<1>(inout);
    EdgeVertsD& edgeVerts = thrust::get<2>(inout);
    if (V1 < V2) {  // forward
      dir = 1;
      edgeVerts = thrust::make_pair(V1, V2);
    } else {  // backward
      dir = -1;
      edgeVerts = thrust::make_pair(V2, V1);
    }
    triEdges[i] = EdgeIdx(0, dir);
  }
};

struct AssignEdges {
  int i;

  __host__ __device__ void operator()(thrust::tuple<TriEdges&, int> inout) {
    int idx2 = thrust::get<1>(inout);
    TriEdges& triEdges = thrust::get<0>(inout);
    triEdges[i] = EdgeIdx(idx2 / 2, triEdges[i].Dir());
  }
};

struct OpposedDir {
  __host__ __device__ bool operator()(int a, int b) const { return a + b == 0; }
};

struct LinkEdges2Tris {
  EdgeTrisD* edgeTris;

  __host__ __device__ void operator()(thrust::tuple<int, TriEdges> in) {
    const int tri = thrust::get<0>(in);
    const TriEdges triEdges = thrust::get<1>(in);
    for (int i : {0, 1, 2}) {
      if (triEdges[i].Dir() > 0)
        edgeTris[triEdges[i].Idx()].left = tri;
      else
        edgeTris[triEdges[i].Idx()].right = tri;
    }
  }
};

struct CheckTris {
  const EdgeVertsD* edgeVerts;

  __host__ __device__ bool operator()(thrust::tuple<glm::ivec3, TriEdges> in) {
    const glm::ivec3& triVerts = thrust::get<0>(in);
    const TriEdges& triEdges = thrust::get<1>(in);
    bool good = true;
    for (int i : {0, 1, 2}) {
      int j = (i + 1) % 3;
      if (triEdges[i].Dir() > 0) {
        good &= triVerts[i] == edgeVerts[triEdges[i].Idx()].first;
        good &= triVerts[j] == edgeVerts[triEdges[i].Idx()].second;
      } else {
        good &= triVerts[i] == edgeVerts[triEdges[i].Idx()].second;
        good &= triVerts[j] == edgeVerts[triEdges[i].Idx()].first;
      }
    }
    return good;
  }
};
}  // namespace

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
    float alpha = i / nDivisions;
    float phi = alpha * twistDegrees;
    glm::mat2 transform(cos(phi), sin(phi), -sin(phi), cos(phi));
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
float Manifold::circularAngle = 12.0f;
float Manifold::circularEdgeLength = 2.0f;

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

Manifold Manifold::Boolean(const Manifold& second, OpType op) const {
  pImpl_->ApplyTransform();
  second.pImpl_->ApplyTransform();
  Boolean3 boolean(*pImpl_, *second.pImpl_);
  Manifold result;
  result.pImpl_ = std::make_unique<Impl>(boolean.Result(op));
  return result;
}

Manifold Manifold::operator+(const Manifold& Q) const {
  return Boolean(Q, OpType::ADD);
}

Manifold Manifold::operator-(const Manifold& Q) const {
  return Boolean(Q, OpType::SUBTRACT);
}

Manifold Manifold::operator^(const Manifold& Q) const {
  return Boolean(Q, OpType::INTERSECT);
}

std::pair<Manifold, Manifold> Manifold::Split(const Manifold& cutter) const {
  pImpl_->ApplyTransform();
  cutter.pImpl_->ApplyTransform();
  Boolean3 boolean(*pImpl_, *cutter.pImpl_);
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

Manifold::Impl::Impl(const Mesh& manifold)
    : vertPos_(manifold.vertPos), triVerts_(manifold.triVerts) {
  CheckDevice();
  Finish();
}

Manifold::Impl::Impl(Shape shape) {
  std::vector<glm::vec3> vertPos;
  std::vector<glm::ivec3> triVerts;
  switch (shape) {
    case Shape::TETRAHEDRON:
      vertPos = {{-1.0f, -1.0f, 1.0f},
                 {-1.0f, 1.0f, -1.0f},
                 {1.0f, -1.0f, -1.0f},
                 {1.0f, 1.0f, 1.0f}};
      triVerts = {{2, 0, 1}, {0, 3, 1}, {2, 3, 0}, {3, 2, 1}};
      break;
    case Shape::CUBE:
      vertPos = {{0.0f, 0.0f, 0.0f},  //
                 {1.0f, 0.0f, 0.0f},  //
                 {1.0f, 1.0f, 0.0f},  //
                 {0.0f, 1.0f, 0.0f},  //
                 {0.0f, 0.0f, 1.0f},  //
                 {1.0f, 0.0f, 1.0f},  //
                 {1.0f, 1.0f, 1.0f},  //
                 {0.0f, 1.0f, 1.0f}};
      triVerts = {{0, 2, 1}, {0, 3, 2},  //
                  {4, 5, 6}, {4, 6, 7},  //
                  {0, 1, 5}, {0, 5, 4},  //
                  {1, 2, 6}, {1, 6, 5},  //
                  {2, 3, 7}, {2, 7, 6},  //
                  {3, 0, 4}, {3, 4, 7}};
      break;
    case Shape::OCTAHEDRON:
      vertPos = {{1.0f, 0.0f, 0.0f},   //
                 {-1.0f, 0.0f, 0.0f},  //
                 {0.0f, 1.0f, 0.0f},   //
                 {0.0f, -1.0f, 0.0f},  //
                 {0.0f, 0.0f, 1.0f},   //
                 {0.0f, 0.0f, -1.0f}};
      triVerts = {{0, 2, 4}, {1, 5, 3},  //
                  {2, 1, 4}, {3, 5, 0},  //
                  {1, 3, 4}, {0, 5, 2},  //
                  {3, 0, 4}, {2, 5, 1}};
      break;
    default:
      throw logicErr("Unrecognized shape!");
  }
  vertPos_ = vertPos;
  triVerts_ = triVerts;
  Finish();
}

void Manifold::Impl::Finish() {
  ALWAYS_ASSERT(thrust::reduce(triVerts_.beginD(), triVerts_.endD(),
                               glm::ivec3(std::numeric_limits<int>::max()),
                               IdxMin())[0] >= 0,
                runtimeErr, "Negative vertex index!");
  ALWAYS_ASSERT(thrust::reduce(triVerts_.beginD(), triVerts_.endD(),
                               glm::ivec3(-1), IdxMax())[0] < NumVert(),
                runtimeErr, "Vertex index exceeds number of verts!");
  CalculateBBox();
  SortVerts();
  VecDH<Box> triBox;
  VecDH<uint32_t> triMorton;
  GetTriBoxMorton(triBox, triMorton);
  SortTris(triBox, triMorton);
  CreateEdges();
  collider_ = Collider(triBox, triMorton);
}

void Manifold::Impl::Update() {
  CalculateBBox();
  VecDH<Box> triBox;
  VecDH<uint32_t> triMorton;
  GetTriBoxMorton(triBox, triMorton);
  collider_.UpdateBoxes(triBox);
}

void Manifold::Impl::ApplyTransform() const {
  // This const_cast is here because these operations cancel out, leaving the
  // state conceptually unchanged. This enables lazy transformation evaluation.
  const_cast<Impl*>(this)->ApplyTransform();
}

void Manifold::Impl::ApplyTransform() {
  if (transform_ == glm::mat4x3(1.0f)) return;
  thrust::for_each(vertPos_.beginD(), vertPos_.endD(), Transform({transform_}));
  // This optimization does a cheap collider update if the transform is
  // axis-aligned.
  if (!collider_.Transform(transform_)) Update();
  transform_ = glm::mat4x3(1.0f);
  CalculateBBox();
}

void Manifold::Impl::Refine(int n) {
  // This function doesn't run Finish(), as that is expensive and it'll need to
  // be run after the new vertices have moved, which is a likely scenario after
  // refinement (smoothing).
  int numVert = NumVert();
  int numEdge = NumEdge();
  int numTri = NumTri();
  // Append new verts
  int vertsPerEdge = n - 1;
  int vertsPerTri = ((n - 2) * (n - 2) + (n - 2)) / 2;
  int triVertStart = numVert + numEdge * vertsPerEdge;
  vertPos_.resize(triVertStart + numTri * vertsPerTri);
  thrust::for_each_n(
      zip(thrust::make_counting_iterator(0), edgeVerts_.beginD()), numEdge,
      SplitEdges({vertPos_.ptrD(), numVert, n}));
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), triVerts_.beginD()),
                     numTri, InteriorVerts({vertPos_.ptrD(), triVertStart, n}));
  // Create subtriangles
  VecDH<glm::ivec3> inTri(triVerts_);
  triVerts_.resize(n * n * numTri);
  thrust::for_each_n(zip(thrust::make_counting_iterator(0), inTri.beginD(),
                         triEdges_.beginD()),
                     numTri,
                     SplitTris({triVerts_.ptrD(), numVert, triVertStart, n}));
}

bool Manifold::Impl::IsValid() const {
  return thrust::all_of(zip(triVerts_.beginD(), triEdges_.beginD()),
                        zip(triVerts_.endD(), triEdges_.endD()),
                        CheckTris({edgeVerts_.ptrD()}));
}

glm::vec3 Manifold::Impl::GetTriNormal(int tri) const {
  glm::vec3 normal = glm::normalize(glm::cross(
      vertPos_.H()[triVerts_.H()[tri][1]] - vertPos_.H()[triVerts_.H()[tri][0]],
      vertPos_.H()[triVerts_.H()[tri][2]] -
          vertPos_.H()[triVerts_.H()[tri][0]]));
  return std::isfinite(normal.x) ? normal : glm::vec3(0.0f);
}

void Manifold::Impl::CalculateBBox() {
  bBox_.min = thrust::reduce(vertPos_.begin(), vertPos_.end(),
                             glm::vec3(1 / 0.0f), PosMin());
  bBox_.max = thrust::reduce(vertPos_.begin(), vertPos_.end(),
                             glm::vec3(-1 / 0.0f), PosMax());
  ALWAYS_ASSERT(bBox_.isFinite(), runtimeErr,
                "Input vertices are not all finite!");
}

void Manifold::Impl::SortVerts() {
  VecDH<uint32_t> vertMorton(NumVert());
  thrust::for_each_n(zip(vertMorton.beginD(), vertPos_.cbeginD()), NumVert(),
                     Morton({bBox_}));

  VecDH<int> vertNew2Old(NumVert());
  thrust::sequence(vertNew2Old.beginD(), vertNew2Old.endD());
  thrust::sort_by_key(vertMorton.beginD(), vertMorton.endD(),
                      zip(vertPos_.beginD(), vertNew2Old.beginD()));

  VecDH<int> vertOld2New(NumVert());
  thrust::scatter(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(NumVert()),
                  vertNew2Old.beginD(), vertOld2New.beginD());
  thrust::for_each(triVerts_.beginD(), triVerts_.endD(),
                   Reindex({vertOld2New.cptrD()}));
}

void Manifold::Impl::CreateEdges() {
  VecDH<EdgeVertsD> halfEdgeVerts(NumTri() * 3);
  VecDH<int> dir(NumTri() * 3);
  edgeVerts_.resize(halfEdgeVerts.size() / 2);
  triEdges_.resize(NumTri());
  edgeTris_.resize(NumEdge());
  for (int i : {0, 1, 2}) {
    int j = (i + 1) % 3;
    int start = i * NumTri();
    thrust::for_each_n(zip(triEdges_.beginD(), dir.beginD() + start,
                           halfEdgeVerts.beginD() + start, triVerts_.cbeginD()),
                       NumTri(), MakeHalfedges({i, j}));
  }
  SortHalfedges(halfEdgeVerts, dir);
  strided_range<VecDH<EdgeVertsD>::IterD> edgeVerts(halfEdgeVerts.beginD(),
                                                    halfEdgeVerts.endD(), 2);
  thrust::copy(edgeVerts.begin(), edgeVerts.end(), edgeVerts_.beginD());

  thrust::for_each_n(zip(thrust::make_counting_iterator(0), triEdges_.beginD()),
                     NumTri(), LinkEdges2Tris({edgeTris_.ptrD()}));
  // verify
  strided_range<VecDH<EdgeVertsD>::IterD> edgesOdd(halfEdgeVerts.beginD() + 1,
                                                   halfEdgeVerts.endD(), 2);
  ALWAYS_ASSERT(
      thrust::equal(edgeVerts.begin(), edgeVerts.end(), edgesOdd.begin()),
      runtimeErr, "Manifold is not manifold!");
  strided_range<VecDH<int>::IterD> dir1(dir.beginD(), dir.endD(), 2);
  strided_range<VecDH<int>::IterD> dir2(dir.beginD() + 1, dir.endD(), 2);
  ALWAYS_ASSERT(
      thrust::equal(dir1.begin(), dir1.end(), dir2.begin(), OpposedDir()),
      runtimeErr, "Manifold is not oriented!");
}

void Manifold::Impl::SortHalfedges(VecDH<EdgeVertsD>& halfEdgeVerts,
                                   VecDH<int>& dir) {
  VecDH<int> halfedgeNew2Old(NumTri() * 3);
  thrust::sequence(halfedgeNew2Old.beginD(), halfedgeNew2Old.endD());
  thrust::sort_by_key(halfEdgeVerts.beginD(), halfEdgeVerts.endD(),
                      zip(dir.beginD(), halfedgeNew2Old.beginD()));

  VecDH<int> halfedgeOld2New(NumTri() * 3);
  thrust::scatter(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator((int)halfedgeNew2Old.size()),
                  halfedgeNew2Old.beginD(), halfedgeOld2New.beginD());
  // assign edge idx to triEdges_ (assumes edge dir is already assigned)
  for (int i : {0, 1, 2}) {
    int start = i * NumTri();
    thrust::for_each_n(
        zip(triEdges_.beginD(), halfedgeOld2New.cbeginD() + start), NumTri(),
        AssignEdges({i}));
  }
}

VecDH<Box> Manifold::Impl::GetEdgeBox() const {
  VecDH<Box> edgeBox(NumEdge());
  thrust::for_each_n(zip(edgeBox.beginD(), edgeVerts_.cbeginD()), NumEdge(),
                     EdgeBox({vertPos_.cptrD()}));
  return edgeBox;
}

void Manifold::Impl::GetTriBoxMorton(VecDH<Box>& triBox,
                                     VecDH<uint32_t>& triMorton) const {
  triBox.resize(NumTri());
  triMorton.resize(NumTri());
  thrust::for_each_n(
      zip(triMorton.beginD(), triBox.beginD(), triVerts_.cbeginD()), NumTri(),
      TriMortonBox({vertPos_.cptrD(), bBox_}));
}

void Manifold::Impl::SortTris(VecDH<Box>& triBox, VecDH<uint32_t>& triMorton) {
  thrust::sort_by_key(triMorton.beginD(), triMorton.endD(),
                      zip(triBox.beginD(), triVerts_.beginD()));
}

SparseIndices Manifold::Impl::EdgeCollisions(const Impl& B) const {
  VecDH<Box> BedgeBB = B.GetEdgeBox();
  return collider_.Collisions(BedgeBB);
}

SparseIndices Manifold::Impl::VertexCollisionsZ(
    const VecDH<glm::vec3>& vertsIn) const {
  return collider_.Collisions(vertsIn);
}
}  // namespace manifold