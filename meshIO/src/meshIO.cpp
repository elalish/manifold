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

#include "meshIO.h"

#include <algorithm>

#include "assimp/Exporter.hpp"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"

namespace manifold {

Mesh ImportMesh(const std::string& filename) {
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
  const aiScene* scene =
      importer.ReadFile(filename,                             //
                        aiProcess_JoinIdenticalVertices |     //
                            aiProcess_Triangulate |           //
                            aiProcess_RemoveComponent |       //
                            aiProcess_PreTransformVertices |  //
                            aiProcess_SortByPType |           //
                            aiProcess_OptimizeMeshes);

  ALWAYS_ASSERT(scene, userErr, importer.GetErrorString());

  Mesh mesh_out;
  for (int i = 0; i < scene->mNumMeshes; ++i) {
    const aiMesh* mesh_i = scene->mMeshes[i];
    for (int j = 0; j < mesh_i->mNumVertices; ++j) {
      const aiVector3D vert = mesh_i->mVertices[j];
      mesh_out.vertPos.emplace_back(vert.x, vert.y, vert.z);
    }
    for (int j = 0; j < mesh_i->mNumFaces; ++j) {
      const aiFace face = mesh_i->mFaces[j];
      ALWAYS_ASSERT(face.mNumIndices == 3, userErr,
                    "Non-triangular face in " + filename);
      mesh_out.triVerts.emplace_back(face.mIndices[0], face.mIndices[1],
                                     face.mIndices[2]);
    }
  }
  return mesh_out;
}

void ExportMesh(const std::string& filename, const Mesh& manifold) {
  if (manifold.triVerts.size() == 0) {
    std::cout << filename << " was not saved because the input mesh was empty."
              << std::endl;
    return;
  }

  aiScene* scene = new aiScene();

  scene->mNumMaterials = 1;
  scene->mMaterials = new aiMaterial*[scene->mNumMaterials];
  scene->mMaterials[0] = new aiMaterial();

  scene->mNumMeshes = 1;
  scene->mMeshes = new aiMesh*[scene->mNumMeshes];
  scene->mMeshes[0] = new aiMesh();
  scene->mMeshes[0]->mMaterialIndex = 0;

  scene->mRootNode = new aiNode();
  scene->mRootNode->mNumMeshes = 1;
  scene->mRootNode->mMeshes = new uint[scene->mRootNode->mNumMeshes];
  scene->mRootNode->mMeshes[0] = 0;

  aiMesh* mesh_out = scene->mMeshes[0];

  mesh_out->mNumVertices = manifold.vertPos.size();
  mesh_out->mVertices = new aiVector3D[mesh_out->mNumVertices];

  for (int i = 0; i < mesh_out->mNumVertices; ++i) {
    const glm::vec3& v = manifold.vertPos[i];
    mesh_out->mVertices[i] = aiVector3D(v.x, v.y, v.z);
  }

  mesh_out->mNumFaces = manifold.triVerts.size();
  mesh_out->mFaces = new aiFace[mesh_out->mNumFaces];

  for (int i = 0; i < mesh_out->mNumFaces; ++i) {
    aiFace& face = mesh_out->mFaces[i];
    face.mNumIndices = 3;
    face.mIndices = new uint[face.mNumIndices];
    for (int j : {0, 1, 2}) face.mIndices[j] = manifold.triVerts[i][j];
  }

  Assimp::Exporter exporter;
  auto result = exporter.Export(
      scene, filename.substr(filename.find_last_of(".") + 1), filename);

  delete scene;

  ALWAYS_ASSERT(result == AI_SUCCESS, userErr, exporter.GetErrorString());
}

}  // namespace manifold