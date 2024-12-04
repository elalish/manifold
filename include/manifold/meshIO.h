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

#pragma once
#include <string>

#include "manifold/manifold.h"

namespace manifold {

/** @addtogroup MeshIO
 *  @ingroup Optional
 *  @brief 3D model file I/O based on Assimp
 * @{
 */

/**
 * PBR material properties for GLB/glTF files.
 */
struct Material {
  /// Roughness value between 0 (shiny) and 1 (matte).
  double roughness = 0.2;
  /// Metalness value, generally either 0 (dielectric) or 1 (metal).
  double metalness = 1;
  /// Color (RGB) multiplier to apply to the whole mesh (each value between 0
  /// and 1).
  vec3 color = vec3(1.0);
  /// Alpha multiplier to apply to the whole mesh (each value between 0
  /// and 1).
  double alpha = 1.0;
  /// Gives the property index where the first normal channel
  /// can be found. 0 indicates the first three property channels following
  /// position. A negative value does not save normals.
  int normalIdx = -1;
  /// Gives the property index where the first color channel
  /// can be found. 0 indicates the first three property channels following
  /// position. A negative value does not save vertex colors.
  int colorIdx = -1;
  /// Gives the property index where the alpha channel
  /// can be found. 0 indicates the first property channel following
  /// position. A negative value does not save vertex alpha.
  int alphaIdx = -1;
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

MeshGL ImportMesh(const std::string& filename, bool forceCleanup = false);

void ExportMesh(const std::string& filename, const MeshGL& mesh,
                const ExportOptions& options);
/** @} */
}  // namespace manifold
