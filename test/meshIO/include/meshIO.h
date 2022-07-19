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

/**
 * PBR material properties for GLB/glTF files.
 */
struct Material {
  /// Roughness value between 0 (shiny) and 1 (matte).
  float roughness = 1;
  /// Metalness value, generally either 0 (dielectric) or 1 (metal).
  float metalness = 0;
  /// Color (RGBA) multiplier to apply to the whole mesh (each value between 0
  /// and 1).
  glm::vec4 color = glm::vec4(1.0f);
  /// Optional: If non-empty, must match Mesh.vertPos. Provides an RGBA color
  /// for each vertex, linearly interpolated across triangles. Colors are
  /// linear, not sRGB.
  std::vector<glm::vec4> vertColor;
};

/**
 * These options only currently affect .glb and .gltf files.
 */
struct ExportOptions {
  /// When false, vertex normals are exported, causing the mesh to appear smooth
  /// through normal interpolation.
  bool faceted = true;
  /// PBR material properties.
  Material mat = {};
};

Mesh ImportMesh(const std::string& filename, bool forceCleanup = false);

void ExportMesh(const std::string& filename, const Mesh& mesh,
                const ExportOptions& options);
/** @} */
}  // namespace manifold