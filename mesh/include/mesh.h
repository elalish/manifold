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
#include <glm/glm.hpp>
#include <vector>
#include "structs.h"

namespace manifold {

struct Mesh {
  std::vector<glm::vec3> vertPos;
  std::vector<glm::ivec3> triVerts;
};

Mesh ImportMesh(const std::string& filename);
void ExportMesh(const std::string& filename, const Mesh&);

}  // namespace manifold