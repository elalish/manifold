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

#pragma once
#include "manifold_impl.cuh"

namespace manifold {

class Boolean3 {
 public:
  Boolean3(const Manifold::Impl& inP, const Manifold::Impl& inQ,
           Manifold::OpType op);
  Manifold::Impl Result(Manifold::OpType op) const;

 private:
  const Manifold::Impl &inP_, &inQ_;
  const float expandP_;
  SparseIndices p1q2_, p2q1_, p2q2_;
  VecDH<int> dir12_, dir21_, w03_, w30_;
  VecDH<glm::vec3> v12_, v21_;
};
}  // namespace manifold