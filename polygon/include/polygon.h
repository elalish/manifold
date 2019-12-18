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
#include "structs.h"

namespace manifold {

int CCW(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2);
Polygons Assemble(const std::vector<EdgeVerts> &edges);
std::vector<glm::ivec3> Triangulate(const Polygons &polys);

std::vector<EdgeVerts> Polygons2Edges(const Polygons &polys);
std::vector<EdgeVerts> Triangles2Edges(
    const std::vector<glm::ivec3> &triangles);
void CheckTopology(const std::vector<EdgeVerts> &halfedges);
void CheckTopology(const std::vector<glm::ivec3> &triangles,
                   const Polygons &polys);
bool CheckGeometry(const std::vector<glm::ivec3> &triangles,
                   const Polygons &polys);
void Dump(const Polygons &polys);
void SetPolygonWarnings(bool);
void SetPolygonVerbose(bool);

}  // namespace manifold