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

#include "sdf.h"

namespace {
using namespace manifold;

struct MarkVerts {
  int* vert;

  __host__ __device__ void operator()(glm::ivec3 triVerts) {
    for (int i : {0, 1, 2}) {
      vert[triVerts[i]] = 1;
    }
  }
};

struct ReindexVerts {
  const int* old2new;

  __host__ __device__ void operator()(glm::ivec3& triVerts) {
    for (int i : {0, 1, 2}) {
      triVerts[i] = old2new[triVerts[i]];
    }
  }
};

}  // namespace

namespace manifold {
/**
 * Sorts the vertices according to their Morton code.
 */
void RemoveUnreferencedVerts(VecDH<glm::vec3>& vertPos,
                             VecDH<glm::ivec3>& triVerts) {
  const int numVert = vertPos.size();
  VecDH<int> vertOld2New(numVert + 1, 0);
  auto policy = autoPolicy(numVert);
  for_each(policy, triVerts.cbegin(), triVerts.cend(),
           MarkVerts({vertOld2New.ptrD() + 1}));

  const VecDH<glm::vec3> oldVertPos = vertPos;
  vertPos.resize(copy_if<decltype(vertPos.begin())>(
                     policy, oldVertPos.cbegin(), oldVertPos.cend(),
                     vertOld2New.cbegin() + 1, vertPos.begin(),
                     thrust::identity<int>()) -
                 vertPos.begin());

  inclusive_scan(policy, vertOld2New.begin() + 1, vertOld2New.end(),
                 vertOld2New.begin() + 1);

  for_each(policy, triVerts.begin(), triVerts.end(),
           ReindexVerts({vertOld2New.cptrD()}));
}

}  // namespace manifold