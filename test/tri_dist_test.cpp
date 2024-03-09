// Copyright 2024 The Manifold Authors.
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

#include "tri_dist.h"

#include "manifold.h"
#include "public.h"
#include "test.h"

TEST(TriangleDistance, ClosestPointsOnVertices) {
  glm::vec3 cp;
  glm::vec3 cq;

  glm::vec3 p[3] = {
      {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};

  glm::vec3 q[3] = {{2.0f, 0.0f, 0.0f}, {4.0f, 0.0f, 0.0f}, {3.0f, 1.0f, 0.0f}};
  float distance = DistanceTriangleTriangleSquared(cp, cq, p, q);

  EXPECT_FLOAT_EQ(distance, 1.0f);
}

TEST(TriangleDistance, ClosestPointOnEdge) {
  glm::vec3 cp;
  glm::vec3 cq;

  glm::vec3 p[3] = {
      {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};

  glm::vec3 q[3] = {
      {-1.0f, 2.0f, 0.0f}, {1.0f, 2.0f, 0.0f}, {0.0f, 3.0f, 0.0f}};

  float distance = DistanceTriangleTriangleSquared(cp, cq, p, q);

  EXPECT_FLOAT_EQ(distance, 1.0f);
}

TEST(TriangleDistance, ClosestPointOnEdge2) {
  glm::vec3 cp;
  glm::vec3 cq;

  glm::vec3 p[3] = {
      {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};

  glm::vec3 q[3] = {{1.0f, 1.0f, 0.0f}, {3.0f, 1.0f, 0.0f}, {2.0f, 2.0f, 0.0f}};

  float distance = DistanceTriangleTriangleSquared(cp, cq, p, q);

  EXPECT_FLOAT_EQ(distance, 0.5f);
}

TEST(TriangleDistance, ClosestPointOnFace) {
  glm::vec3 cp;
  glm::vec3 cq;

  glm::vec3 p[3] = {
      {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};

  glm::vec3 q[3] = {
      {-1.0f, 2.0f, -0.5f}, {1.0f, 2.0f, -0.5f}, {0.0f, 2.0f, 1.5f}};

  float distance = DistanceTriangleTriangleSquared(cp, cq, p, q);

  EXPECT_FLOAT_EQ(distance, 1.0f);
}
