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

#include "mesh.h"
#include "gtest/gtest.h"

namespace {

using namespace manifold;

void Identical(MeshHost& mesh1, MeshHost& mesh2) {
  ASSERT_EQ(mesh1.vertPos.size(), mesh2.vertPos.size());
  for (int i = 0; i < mesh1.vertPos.size(); ++i) {
    glm::vec3 v_in = mesh1.vertPos[i];
    glm::vec3 v_out = mesh2.vertPos[i];
    for (int j : {0, 1, 2}) ASSERT_NEAR(v_in[j], v_out[j], 0.0001);
  }
  ASSERT_EQ(mesh1.triVerts.size(), mesh2.triVerts.size());
  for (int i = 0; i < mesh1.triVerts.size(); ++i) {
    ASSERT_EQ(mesh1.triVerts[i], mesh2.triVerts[i]);
  }
}

void CheckIdx(int idx, int dir) {
  EdgeIdx edge(idx, dir);
  ASSERT_EQ(edge.Idx(), idx);
  ASSERT_EQ(edge.Dir(), dir);
}

void ExpectMeshes(const Mesh& mesh,
                  const std::vector<std::pair<int, int>>& numVertTri) {
  ASSERT_TRUE(mesh.IsValid());
  std::vector<Mesh> meshes = mesh.Decompose();
  ASSERT_EQ(meshes.size(), numVertTri.size());
  std::sort(meshes.begin(), meshes.end(), [](const Mesh& a, const Mesh& b) {
    return a.NumVert() != b.NumVert() ? a.NumVert() > b.NumVert()
                                      : a.NumTri() > b.NumTri();
  });
  for (int i = 0; i < meshes.size(); ++i) {
    EXPECT_TRUE(meshes[i].IsValid());
    EXPECT_EQ(meshes[i].NumVert(), numVertTri[i].first);
    EXPECT_EQ(meshes[i].NumTri(), numVertTri[i].second);
  }
}
}  // namespace

TEST(MeshHost, EdgeIdx) {
  CheckIdx(0, 1);
  CheckIdx(0, -1);
  CheckIdx(1, 1);
  CheckIdx(1, -1);
}

TEST(MeshHost, ReadWrite) {
  MeshHost mesh_host = ImportMesh("data/gyroidpuzzle.ply");
  ExportMesh("data/gyroidpuzzle1.ply", mesh_host);
  MeshHost mesh_out = ImportMesh("data/gyroidpuzzle1.ply");
  Identical(mesh_host, mesh_out);
}

TEST(Mesh, Regression) {
  Mesh mesh(ImportMesh("data/gyroidpuzzle.ply"));
  ASSERT_TRUE(mesh.IsValid());

  Mesh mesh1 = mesh.Copy();
  mesh1.Translate(glm::vec3(5.0f));
  int num_overlaps = mesh.NumOverlaps(mesh1);
  ASSERT_EQ(num_overlaps, 237472);

  MeshHost mesh_out, mesh_out2;
  mesh.Append2Host(mesh_out);
  Mesh mesh2(mesh_out);
  mesh2.Append2Host(mesh_out2);
  Identical(mesh_out, mesh_out2);
}

TEST(Mesh, Decompose) {
  std::vector<Mesh> meshList;
  meshList.push_back(Mesh::Tetrahedron());
  meshList.push_back(Mesh::Cube());
  meshList.push_back(Mesh::Octahedron());
  Mesh meshes(meshList);
  ASSERT_TRUE(meshes.IsValid());

  ExpectMeshes(meshes, {{8, 12}, {6, 8}, {4, 4}});
}

TEST(Mesh, Refine) {
  int n = 5;
  Mesh tetra = Mesh::Tetrahedron();
  ASSERT_TRUE(tetra.IsValid());
  tetra.Refine(n);
  ASSERT_TRUE(tetra.IsValid());
  ASSERT_EQ(tetra.NumTri(), n * n * 4);
}

TEST(Mesh, Sphere) {
  int n = 25;
  Mesh sphere = Mesh::Sphere(4 * n);
  ASSERT_TRUE(sphere.IsValid());
  ASSERT_EQ(sphere.NumTri(), n * n * 8);
}

TEST(Mesh, BooleanTetra) {
  Mesh tetra = Mesh::Tetrahedron();
  ASSERT_TRUE(tetra.IsValid());

  Mesh tetra2 = tetra.Copy();
  tetra2.Translate(glm::vec3(0.5f));
  Mesh result = tetra2.Boolean(tetra, Mesh::OpType::SUBTRACT);

  ExpectMeshes(result, {{8, 12}});
}

TEST(Mesh, BooleanSphere) {
  Mesh sphere = Mesh::Sphere(12);
  Mesh sphere2 = sphere.Copy();
  sphere2.Translate(glm::vec3(0.5));
  Mesh result = sphere.Boolean(sphere2, Mesh::OpType::SUBTRACT);

  ExpectMeshes(result, {{74, 144}});
}

TEST(Mesh, Boolean3) {
  Mesh gyroid(ImportMesh("data/gyroidpuzzle.ply"));
  ASSERT_TRUE(gyroid.IsValid());

  Mesh gyroid2 = gyroid.Copy();
  gyroid2.Translate(glm::vec3(5.0f));
  Mesh result = gyroid.Boolean(gyroid2, Mesh::OpType::ADD);

  ExpectMeshes(result, {{31733, 63602}});
}

TEST(Mesh, BooleanSelfIntersecting) {
  std::vector<Mesh> meshList;
  meshList.push_back(Mesh::Tetrahedron());
  meshList.push_back(Mesh::Tetrahedron());
  meshList[1].Translate(glm::vec3(0, 0, 0.25));
  Mesh tetras(meshList);
  ASSERT_TRUE(tetras.IsValid());

  meshList[0].Translate(glm::vec3(0, 0, 0.5f));
  Mesh result = meshList[0].Boolean(tetras, Mesh::OpType::SUBTRACT);

  ExpectMeshes(result, {{8, 12}, {4, 4}});
}

TEST(Mesh, BooleanSelfIntersectingAlt) {
  std::vector<Mesh> meshList;
  meshList.push_back(Mesh::Tetrahedron());
  meshList.push_back(Mesh::Tetrahedron());
  meshList[1].Translate(glm::vec3(0, 0, 0.5));
  Mesh tetras(meshList);
  ASSERT_TRUE(tetras.IsValid());

  meshList[0].Translate(glm::vec3(0, 0, -0.5f));
  Mesh result = meshList[0].Boolean(tetras, Mesh::OpType::SUBTRACT);

  ExpectMeshes(result, {{8, 12}, {4, 4}});
}

TEST(Mesh, BooleanWinding) {
  std::vector<Mesh> meshList;
  meshList.push_back(Mesh::Tetrahedron());
  meshList.push_back(Mesh::Tetrahedron());
  meshList[1].Translate(glm::vec3(0.25));
  Mesh tetras(meshList);
  ASSERT_TRUE(tetras.IsValid());

  meshList[0].Translate(glm::vec3(-0.25f));
  Mesh result = tetras.Boolean(meshList[0], Mesh::OpType::SUBTRACT);

  ExpectMeshes(result, {{8, 12}, {8, 12}});
}