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

#pragma once
#include "structs.h"

namespace manifold {

/** @addtogroup Connections
 *  @{
 */
struct Material {
  float roughness = 1;
  float metalness = 0;
  glm::vec4 color = glm::vec4(1.0f);
  std::vector<glm::vec4> vertColor;
};

struct ExportOptions {
  bool faceted = true;
  Material mat = {};
};

/**
 * @brief Read mesh file.
 *
 * @param filename Handles any files the Assimp library can import.
 * @return Mesh The mesh should be checked for manifoldness.
 */
Mesh ImportMesh(const std::string& filename, bool forceCleanup = false);

/**
 *
 * @brief Write mesh file.
 *
 * @param filename The file extension must be one that Assimp supports for
 * export. GLB & 3MF are recommended.
 * @param mesh The mesh to export, likely from Manifold.GetMesh().
 * @param options The options currently only affect an exported GLB's material.
 * Pass {} for defaults.
 */
void ExportMesh(const std::string& filename, const Mesh& mesh,
                const ExportOptions& options);
/** @} */
}  // namespace manifold