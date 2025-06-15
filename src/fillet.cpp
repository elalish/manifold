// Copyright 2021 The Manifold Authors.
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

#include "./impl.h"

namespace {
using namespace manifold;

// Get range one neighbour vertex except endVert
std::vector<int> getNeighbour(const size_t& halfedge,
                              const Vec<Halfedge>& vec) {
  std::vector<int> r;
  int tri = halfedge / 3;
  int idx = halfedge % 3;

  while (true) {
    const int next = vec[tri * 3 + (idx + 1) % 3].pairedHalfedge;
    if (next == halfedge) break;

    r.push_back(vec[next].startVert);

    tri = next / 3;
    idx = next % 3;
  };

  return r;
};

}  // namespace

namespace manifold {

void Manifold::Impl::Fillet(double radius,
                            const std::vector<size_t>& selectedEdges) {
  // Map to sorted idx
  std::unordered_map<int, int> oldHalfedge2New;
  for (size_t tri = 0; tri < NumTri(); ++tri) {
    int oldTri = meshRelation_.triRef[tri].tri;
    for (int i : {0, 1, 2}) oldHalfedge2New[3 * oldTri + i] = 3 * tri + i;
  }
  std::vector<size_t> newEdge = selectedEdges;
  for (size_t& edge : newEdge) {
    edge = oldHalfedge2New[edge];
  }

  for (const size_t& halfedge : newEdge) {
    auto r1 = getNeighbour(halfedge, halfedge_);
    auto r2 = getNeighbour(halfedge_[halfedge].pairedHalfedge, halfedge_);

    std::vector<ivec3> f;
    for (int i = 0; i != r1.size() - 1; i++) {
      f.emplace_back(r1[0], r1[1], halfedge_[halfedge].endVert);
    }

    for (int i = 0; i != r2.size() - 1; i++) {
      f.emplace_back(r2[0], r2[1], halfedge_[halfedge].endVert);
    }

    const auto& edge = halfedge_[halfedge];
    const vec3 target = vertPos_[edge.startVert] - vertPos_[edge.endVert];

    // FIXME: The cylinder length should be determine by r1 neighbour, current
    // set manually
    float len = la::length(target) * 2;

    /*
            v4
           / \
          /   \    <- f2
         /  n2 \
        v1 --- v2  <- Half edge
         \  n1 /
          \   /    <- f1
           \ /
            v3
    */

    vec3 v1 = vertPos_[edge.startVert], v2 = vertPos_[edge.endVert],
         v3 = vertPos_[*r1.begin()], v4 = vertPos_[*r1.end()];

    vec3 v1v2 = v2 - v1, v1v3 = v3 - v1, v1v4 = v4 - v1;

    // Inv normal
    vec3 n1 = la::normalize(la::cross(v1v2, v1v3)),
         n2 = la::normalize(la::cross(v1v4, v1v2));

    // Cross line
    vec3 n3 = la::normalize(v1v2);
    float A1 = n1.x, B1 = n1.y, C1 = n1.z,
          D1 = A1 * -v3.x + B1 * -v3.y + C1 * -v3.z -
               radius * std::sqrt(A1 * A1 + B1 * B1 * +C1 * C1),
          D3 = A1 * -v3.x + B1 * -v3.y + C1 * -v3.z;

    float A2 = n2.x, B2 = n2.y, C2 = n2.z,
          D2 = A2 * -v4.x + B2 * -v4.y + C2 * -v4.z -
               radius * std::sqrt(A2 * A2 + B2 * B2 * +C2 * C2),
          D4 = A2 * -v4.x + B2 * -v4.y + C2 * -v4.z;

    vec3 p;
    if ((A1 * B2 - A2 * B1) > epsilon_) {
      p = vec3((B2 * D1 - B1 * D2) / (A1 * B2 - A2 * B1),
               (A1 * D2 - A2 * D1) / (A1 * B2 - A2 * B1), 0);
    }

    vec3 p1 = p + n3;

    double det1 = 0, det2 = 0;

    // Determine whether intersect with original tri
    {
      // Project p, p1 to f1, f2

      vec3 PF1p = p - (n1 * p + D3) / (la::length2(n1)) * n1,
           PF1p1 = p1 - (n1 * p1 + D3) / (la::length2(n1)) * n1,
           PF2p = p - (n2 * p + D4) / (la::length2(n2)) * n2,
           PF2p1 = p1 - (n2 * p1 + D4) / (la::length2(n2)) * n2;

      vec3 e1 = la::cross(PF1p - PF1p1, PF1p - v1);
      det1 = la::dot(e1, la::cross(PF1p - PF1p1, PF1p - v2)) *
             la::dot(e1, la::cross(PF1p - PF1p1, PF1p - v3));

      vec3 e2 = la::cross(PF2p - PF2p1, PF2p - v1);
      det2 = la::dot(e2, la::cross(PF2p - PF2p1, PF1p - v2)) *
             la::dot(e2, la::cross(PF2p - PF2p1, PF1p - v4));
    }

    if (det1 <= epsilon_ && det2 <= epsilon_) {
      // Inside tri

      // Create cylinder
      Manifold cylinder = Manifold::Cylinder(len, radius);

      cylinder.Rotate(la::angle(vec3(1, 0, 0), target));

      // v1 project to cross line

      vec3 pv1 = (v1 - p);
      double t = -la::dot(pv1, n3) / la::dot(n3, n3);
      vec3 origin = p + t * n3;

      // Offset origin by len

      origin -= len / 4 * n3;

      cylinder.Translate(origin);

      // Cut with r1 neighbour

      for (ivec3 v : f) {
        vec3 ab = vertPos_[v.x] - vertPos_[v.y];
        vec3 ac = vertPos_[v.z] - vertPos_[v.y];
        vec3 normal = la::cross(ab, ac);
        double off = la::dot(normal, vertPos_[v.x]);

        // Transform to origin and normal
        cylinder = cylinder.SplitByPlane(normal, off).second;
      }

      // TODO: determine cross edge and union boundary
      // m.Split(cylinder);

    } else {
      // Radius out range
      throw std::exception();
    }
  }
}

}  // namespace manifold