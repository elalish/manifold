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

#include "manifold/meshIO.h"

#include <iostream>

#include "assimp/Exporter.hpp"
#include "assimp/Importer.hpp"
#include "assimp/material.h"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "manifold/optional_assert.h"

#ifndef AI_MATKEY_ROUGHNESS_FACTOR
#include "assimp/pbrmaterial.h"
#define AI_MATKEY_METALLIC_FACTOR \
  AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR
#define AI_MATKEY_ROUGHNESS_FACTOR \
  AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR
#endif

namespace {
using namespace manifold;

aiScene* CreateScene(const ExportOptions& options) {
  aiScene* scene = new aiScene();

  scene->mNumMaterials = 1;
  scene->mMaterials = new aiMaterial*[scene->mNumMaterials];
  scene->mMaterials[0] = new aiMaterial();

  aiMaterial* material = scene->mMaterials[0];
  material->AddProperty(&options.mat.roughness, 1, AI_MATKEY_ROUGHNESS_FACTOR);
  material->AddProperty(&options.mat.metalness, 1, AI_MATKEY_METALLIC_FACTOR);
  const vec3& color = options.mat.color;
  aiColor4D baseColor(color.x, color.y, color.z, options.mat.alpha);
  material->AddProperty(&baseColor, 1, AI_MATKEY_COLOR_DIFFUSE);

  scene->mNumMeshes = 1;
  scene->mMeshes = new aiMesh*[scene->mNumMeshes];
  scene->mMeshes[0] = new aiMesh();
  scene->mMeshes[0]->mMaterialIndex = 0;

  scene->mRootNode = new aiNode();
  scene->mRootNode->mNumMeshes = 1;
  scene->mRootNode->mMeshes = new uint32_t[scene->mRootNode->mNumMeshes];
  scene->mRootNode->mMeshes[0] = 0;

  scene->mMeshes[0]->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;
  return scene;
}

std::string GetType(const std::string& filename) {
  std::string ext = filename.substr(filename.find_last_of(".") + 1);
  if (ext == "glb") ext = "glb2";
  if (ext == "gltf") ext = "gltf2";
  return ext;
}

void ExportScene(aiScene* scene, const std::string& filename) {
  Assimp::Exporter exporter;

  // int n = exporter.GetExportFormatCount();
  // for (int i = 0; i < n; ++i) {
  //   auto desc = exporter.GetExportFormatDescription(i);
  //   std::cout << i << ", id = " << desc->id << ", " << desc->description
  //             << std::endl;
  // }
  auto ext = filename.substr(filename.length() - 4, 4);
  if (ext == ".3mf") {
    // Workaround https://github.com/assimp/assimp/issues/3816
    aiNode* old_root = scene->mRootNode;
    scene->mRootNode = new aiNode();
    scene->mRootNode->addChildren(1, &old_root);
  }

  auto result = exporter.Export(scene, GetType(filename), filename);

  delete scene;

  DEBUG_ASSERT(result == AI_SUCCESS, userErr, exporter.GetErrorString());
}
}  // namespace

namespace manifold {

/**
 * Imports the given file as a Mesh structure, which can be converted to a
 * Manifold if the mesh is a proper oriented 2-manifold. Any supported polygon
 * format will be automatically triangulated.
 *
 * This is a very simple import function and is intended primarily as a
 * demonstration. Generally users of this library will need to modify this to
 * read all the important properties for their application and set up any custom
 * data structures.
 *
 * @param filename Supports any format the Assimp library supports.
 * @param forceCleanup This merges identical vertices, which can break
 * manifoldness. However it is always done for STLs, as they cannot possibly be
 * manifold without this step.
 */
MeshGL ImportMesh(const std::string& filename, bool forceCleanup) {
  std::string ext = filename.substr(filename.find_last_of(".") + 1);
  const bool isYup = ext == "glb" || ext == "gltf";

  Assimp::Importer importer;
  importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,                    //
                              aiComponent_NORMALS |                      //
                                  aiComponent_TANGENTS_AND_BITANGENTS |  //
                                  aiComponent_COLORS |                   //
                                  aiComponent_TEXCOORDS |                //
                                  aiComponent_BONEWEIGHTS |              //
                                  aiComponent_ANIMATIONS |               //
                                  aiComponent_TEXTURES |                 //
                                  aiComponent_LIGHTS |                   //
                                  aiComponent_CAMERAS |                  //
                                  aiComponent_MATERIALS);
  importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE,
                              aiPrimitiveType_POINT | aiPrimitiveType_LINE);

  unsigned int flags = aiProcess_Triangulate |           //
                       aiProcess_RemoveComponent |       //
                       aiProcess_PreTransformVertices |  //
                       aiProcess_SortByPType;
  if (forceCleanup || ext == "stl") {
    flags = flags | aiProcess_JoinIdenticalVertices | aiProcess_OptimizeMeshes;
  }

  const aiScene* scene = importer.ReadFile(filename, flags);

  DEBUG_ASSERT(scene, userErr, importer.GetErrorString());

  MeshGL mesh_out;
  mesh_out.numProp = 3;
  for (size_t i = 0; i < scene->mNumMeshes; ++i) {
    const aiMesh* mesh_i = scene->mMeshes[i];
    for (size_t j = 0; j < mesh_i->mNumVertices; ++j) {
      const aiVector3D vert = mesh_i->mVertices[j];
      if (isYup)
        mesh_out.vertProperties.insert(mesh_out.vertProperties.end(),
                                       {vert.z, vert.x, vert.y});
      else
        mesh_out.vertProperties.insert(mesh_out.vertProperties.end(),
                                       {vert.x, vert.y, vert.z});
    }
    for (size_t j = 0; j < mesh_i->mNumFaces; ++j) {
      const aiFace face = mesh_i->mFaces[j];
      DEBUG_ASSERT(face.mNumIndices == 3, userErr,
                   "Non-triangular face in " + filename);
      mesh_out.triVerts.insert(
          mesh_out.triVerts.end(),
          {face.mIndices[0], face.mIndices[1], face.mIndices[2]});
    }
  }
  return mesh_out;
}

/**
 * Saves the Mesh to the desired file type, determined from the extension
 * specified. In the case of .glb/.gltf, this will save in version 2.0.
 *
 * This is a very simple export function and is intended primarily as a
 * demonstration. Generally users of this library will need to modify this to
 * write all the important properties for their application and read any custom
 * data structures.
 *
 * @param filename The file extension must be one that Assimp supports for
 * export. GLB & 3MF are recommended.
 * @param mesh The mesh to export, likely from Manifold.GetMeshGL().
 * @param options The options currently only affect an exported GLB's material.
 * Pass {} for defaults.
 */
void ExportMesh(const std::string& filename, const MeshGL& mesh,
                const ExportOptions& options) {
  if (mesh.triVerts.size() == 0) {
    std::cout << filename << " was not saved because the input mesh was empty."
              << std::endl;
    return;
  }

  std::string type = GetType(filename);
  const bool isYup = type == "glb2" || type == "gltf2";

  aiScene* scene = CreateScene(options);
  aiMesh* mesh_out = scene->mMeshes[0];

  mesh_out->mNumVertices = mesh.NumVert();
  mesh_out->mVertices = new aiVector3D[mesh_out->mNumVertices];
  if (!options.faceted) {
    const bool validChannels = options.mat.normalIdx >= 0 &&
                               options.mat.normalIdx + 6 <= (int)mesh.numProp;
    DEBUG_ASSERT(
        validChannels, userErr,
        "When faceted is false, valid normalChannels must be supplied.");
    mesh_out->mNormals = new aiVector3D[mesh_out->mNumVertices];
  }

  const bool hasColor = options.mat.colorIdx >= 0;
  const bool validChannels =
      !hasColor || options.mat.colorIdx + 6 <= (int)mesh.numProp;
  DEBUG_ASSERT(validChannels, userErr, "Invalid colorChannels.");
  const bool hasAlpha = options.mat.alphaIdx >= 0;
  const bool validAlpha =
      !hasAlpha || options.mat.alphaIdx + 4 <= (int)mesh.numProp;
  DEBUG_ASSERT(validAlpha, userErr, "Invalid colorChannels.");
  if (hasColor) mesh_out->mColors[0] = new aiColor4D[mesh_out->mNumVertices];

  for (size_t i = 0; i < mesh_out->mNumVertices; ++i) {
    vec3 v;
    for (int j : {0, 1, 2}) v[j] = mesh.vertProperties[i * mesh.numProp + j];
    mesh_out->mVertices[i] =
        isYup ? aiVector3D(v.y, v.z, v.x) : aiVector3D(v.x, v.y, v.z);
    if (!options.faceted) {
      vec3 n;
      for (int j : {0, 1, 2})
        n[j] = mesh.vertProperties[i * mesh.numProp + 3 +
                                   options.mat.normalIdx + j];
      mesh_out->mNormals[i] =
          isYup ? aiVector3D(n.y, n.z, n.x) : aiVector3D(n.x, n.y, n.z);
    }
    if (hasColor) {
      vec4 c(1.0);
      for (int j : {0, 1, 2})
        c[j] = mesh.vertProperties[i * mesh.numProp + 3 + options.mat.colorIdx +
                                   j];
      if (hasAlpha)
        c[3] = mesh.vertProperties[i * mesh.numProp + 3 + options.mat.alphaIdx];
      c = la::clamp(c, 0, 1);
      mesh_out->mColors[0][i] = aiColor4D(c.x, c.y, c.z, c.w);
    }
  }

  mesh_out->mNumFaces = mesh.NumTri();
  mesh_out->mFaces = new aiFace[mesh_out->mNumFaces];

  for (size_t i = 0; i < mesh_out->mNumFaces; ++i) {
    aiFace& face = mesh_out->mFaces[i];
    face.mNumIndices = 3;
    face.mIndices = new uint32_t[face.mNumIndices];
    for (int j : {0, 1, 2}) face.mIndices[j] = mesh.triVerts[3 * i + j];
  }

  ExportScene(scene, filename);
}
}  // namespace manifold
