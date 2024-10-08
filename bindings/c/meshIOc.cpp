// Copyright 2023 The Manifold Authors.
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

#include "./conv.h"
#include "manifold/meshIO.h"
#include "manifold/types.h"

// C <-> C++ conversions

ManifoldMaterial *to_c(manifold::Material *m) {
  return reinterpret_cast<ManifoldMaterial *>(m);
}

ManifoldExportOptions *to_c(manifold::ExportOptions *m) {
  return reinterpret_cast<ManifoldExportOptions *>(m);
}

manifold::Material *from_c(ManifoldMaterial *mat) {
  return reinterpret_cast<manifold::Material *>(mat);
}

manifold::ExportOptions *from_c(ManifoldExportOptions *options) {
  return reinterpret_cast<manifold::ExportOptions *>(options);
}

#ifdef __cplusplus
extern "C" {
#endif

// material

ManifoldMaterial *manifold_material(void *mem) {
  return to_c(new (mem) manifold::Material());
}

void manifold_material_set_roughness(ManifoldMaterial *mat, double roughness) {
  from_c(mat)->roughness = roughness;
}

void manifold_material_set_metalness(ManifoldMaterial *mat, double metalness) {
  from_c(mat)->metalness = metalness;
}

void manifold_material_set_color(ManifoldMaterial *mat, ManifoldVec4 color) {
  from_c(mat)->color = from_c(color);
}

void manifold_material_set_vert_color(ManifoldMaterial *mat,
                                      ManifoldVec4 *vert_color, size_t n_vert) {
  from_c(mat)->vertColor = vector_of_vec_array(vert_color, n_vert);
}

// export options

ManifoldExportOptions *manifold_export_options(void *mem) {
  return to_c(new (mem) manifold::ExportOptions());
}

void manifold_export_options_set_faceted(ManifoldExportOptions *options,
                                         int faceted) {
  from_c(options)->faceted = faceted;
}

void manifold_export_options_set_material(ManifoldExportOptions *options,
                                          ManifoldMaterial *mat) {
  from_c(options)->mat = *from_c(mat);
}

// mesh IO

void manifold_export_meshgl(const char *filename, ManifoldMeshGL *mesh,
                            ManifoldExportOptions *options) {
  manifold::ExportMesh(std::string(filename), *from_c(mesh), *from_c(options));
}

ManifoldMeshGL *manifold_import_meshgl(void *mem, const char *filename,
                                       int force_cleanup) {
  auto m = manifold::ImportMesh(std::string(filename), force_cleanup);
  return to_c(new (mem) MeshGL(m));
}

// memory size
size_t manifold_material_size() { return sizeof(manifold::Material); }

size_t manifold_export_options_size() {
  return sizeof(manifold::ExportOptions);
}

// memory free + destruction
void manifold_delete_material(ManifoldMaterial *m) { delete (from_c(m)); }

void manifold_delete_export_options(ManifoldExportOptions *m) {
  delete (from_c(m));
}

// destruction
void manifold_destruct_material(ManifoldMaterial *m) { from_c(m)->~Material(); }

void manifold_destruct_export_options(ManifoldExportOptions *m) {
  from_c(m)->~ExportOptions();
}

#ifdef __cplusplus
}
#endif
