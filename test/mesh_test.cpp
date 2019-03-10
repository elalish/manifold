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
  MeshHost mesh_host = ImportMesh("data/gyroidpuzzle.ply");
  Mesh mesh(mesh_host);
  ASSERT_TRUE(mesh.IsValid());

  Mesh mesh1 = mesh.Copy();
  mesh1.Translate(glm::vec3(5.0f));
  int expect_num_overlaps = 237472;
  int num_overlaps = mesh.NumOverlaps(mesh1, 1000000);
  ASSERT_EQ(expect_num_overlaps, num_overlaps);

  MeshHost mesh_check = ImportMesh("data/gyroidpuzzle_check.ply");
  MeshHost mesh_out;
  mesh.Append2Host(mesh_out);
  Identical(mesh_out, mesh_check);
}