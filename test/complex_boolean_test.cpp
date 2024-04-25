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

#include "manifold.h"
#include "polygon.h"
#include "test.h"

using namespace manifold;

/**
 * These tests verify correct topology and geometry for complex boolean
 * operations between valid shapes with many faces.
 */

TEST(Boolean, Sphere) {
  Manifold sphere = Manifold::Sphere(1.0f, 12);
  MeshGL sphereGL = WithPositionColors(sphere);
  sphere = Manifold(sphereGL);

  Manifold sphere2 = sphere.Translate(glm::vec3(0.5));
  Manifold result = sphere - sphere2;

  ExpectMeshes(result, {{74, 144, 3, 110}});
  EXPECT_EQ(result.NumDegenerateTris(), 0);

  RelatedGL(result, {sphereGL});
  result = result.Refine(4);
  RelatedGL(result, {sphereGL});

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorChannels = glm::ivec4(3, 4, 5, -1);
  if (options.exportModels)
    ExportMesh("sphereUnion.glb", result.GetMeshGL(), opt);
#endif
}

TEST(Boolean, MeshRelation) {
  Mesh gyroidMesh = Gyroid();
  MeshGL gyroidMeshGL = WithPositionColors(gyroidMesh);
  Manifold gyroid(gyroidMeshGL);

  Manifold gyroid2 = gyroid.Translate(glm::vec3(2.0f));

  EXPECT_FALSE(gyroid.IsEmpty());
  EXPECT_TRUE(gyroid.MatchesTriNormals());
  EXPECT_LE(gyroid.NumDegenerateTris(), 0);
  Manifold result = gyroid + gyroid2;
  result = result.RefineToLength(0.1);

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorChannels = glm::ivec4(3, 4, 5, -1);
  if (options.exportModels)
    ExportMesh("gyroidUnion.glb", result.GetMeshGL(), opt);
#endif

  EXPECT_TRUE(result.MatchesTriNormals());
  EXPECT_LE(result.NumDegenerateTris(), 1);
  EXPECT_EQ(result.Decompose().size(), 1);
  auto prop = result.GetProperties();
  EXPECT_NEAR(prop.volume, 226, 1);
  EXPECT_NEAR(prop.surfaceArea, 387, 1);

  RelatedGL(result, {gyroidMeshGL});
}

TEST(Boolean, Cylinders) {
  Manifold rod = Manifold::Cylinder(1.0, 0.4, -1.0, 12);
  float arrays1[][12] = {
      {0, 0, 1, 3,    //
       -1, 0, 0, 3,   //
       0, -1, 0, 6},  //
      {0, 0, 1, 2,    //
       -1, 0, 0, 3,   //
       0, -1, 0, 8},  //

      {0, 0, 1, 1,    //
       -1, 0, 0, 2,   //
       0, -1, 0, 7},  //
      {1, 0, 0, 3,    //
       0, 1, 0, 2,    //
       0, 0, 1, 6},   //
      {0, 0, 1, 3,    //
       -1, 0, 0, 3,   //
       0, -1, 0, 7},  //
      {0, 0, 1, 1,    //
       -1, 0, 0, 3,   //
       0, -1, 0, 7},  //
      {1, 0, 0, 3,    //
       0, 0, 1, 4,    //
       0, -1, 0, 6},  //
      {1, 0, 0, 4,    //
       0, 0, 1, 4,    //
       0, -1, 0, 6},  //
  };
  float arrays2[][12] = {
      {1, 0, 0, 3,    //
       0, 0, 1, 2,    //
       0, -1, 0, 6},  //
      {1, 0, 0, 4,    //
       0, 1, 0, 3,    //
       0, 0, 1, 6},   //

      {0, 0, 1, 2,    //
       -1, 0, 0, 2,   //
       0, -1, 0, 7},  //
      {1, 0, 0, 3,    //
       0, 1, 0, 3,    //
       0, 0, 1, 7},   //
      {1, 0, 0, 2,    //
       0, 1, 0, 3,    //
       0, 0, 1, 7},   //
      {1, 0, 0, 1,    //
       0, 1, 0, 3,    //
       0, 0, 1, 7},   //
      {1, 0, 0, 3,    //
       0, 1, 0, 4,    //
       0, 0, 1, 7},   //
      {1, 0, 0, 3,    //
       0, 1, 0, 5,    //
       0, 0, 1, 6},   //
      {0, 0, 1, 3,    //
       -1, 0, 0, 4,   //
       0, -1, 0, 6},  //
  };

  Manifold m1;
  for (auto& array : arrays1) {
    glm::mat4x3 mat;
    for (const int i : {0, 1, 2, 3}) {
      for (const int j : {0, 1, 2}) {
        mat[i][j] = array[j * 4 + i];
      }
    }
    m1 += rod.Transform(mat);
  }

  Manifold m2;
  for (auto& array : arrays2) {
    glm::mat4x3 mat;
    for (const int i : {0, 1, 2, 3}) {
      for (const int j : {0, 1, 2}) {
        mat[i][j] = array[j * 4 + i];
      }
    }
    m2 += rod.Transform(mat);
  }
  m1 += m2;

  EXPECT_TRUE(m1.MatchesTriNormals());
  EXPECT_LE(m1.NumDegenerateTris(), 12);
}

TEST(Boolean, Subtract) {
  Mesh firstMesh;
  firstMesh.vertPos = {{0, 0, 0},           {1540, 0, 0},
                       {1540, 70, 0},       {0, 70, 0},
                       {0, 0, -278.282},    {1540, 70, -278.282},
                       {1540, 0, -278.282}, {0, 70, -278.282}};
  firstMesh.triVerts = {
      {0, 1, 2}, {2, 3, 0}, {4, 5, 6}, {5, 4, 7}, {6, 2, 1}, {6, 5, 2},
      {5, 3, 2}, {5, 7, 3}, {7, 0, 3}, {7, 4, 0}, {4, 1, 0}, {4, 6, 1},
  };

  Mesh secondMesh;
  secondMesh.vertPos = {
      {2.04636e-12, 70, 50000},       {2.04636e-12, -1.27898e-13, 50000},
      {1470, -1.27898e-13, 50000},    {1540, 70, 50000},
      {2.04636e-12, 70, -28.2818},    {1470, -1.27898e-13, 0},
      {2.04636e-12, -1.27898e-13, 0}, {1540, 70, -28.2818}};
  secondMesh.triVerts = {{0, 1, 2}, {2, 3, 0}, {4, 5, 6}, {5, 4, 7},
                         {6, 2, 1}, {6, 5, 2}, {5, 3, 2}, {5, 7, 3},
                         {7, 0, 3}, {7, 4, 0}, {4, 1, 0}, {4, 6, 1}};

  Manifold first(firstMesh);
  Manifold second(secondMesh);

  first -= second;
  first.GetMesh();
}

TEST(Boolean, Close) {
  PolygonParams().processOverlaps = true;

  const float r = 10;
  Manifold a = Manifold::Sphere(r, 256);
  Manifold result = a;
  for (int i = 0; i < 10; i++) {
    // std::cout << i << std::endl;
    result ^= a.Translate({a.Precision() / 10 * i, 0.0, 0.0});
  }
  auto prop = result.GetProperties();
  const float tol = 0.004;
  EXPECT_NEAR(prop.volume, (4.0f / 3.0f) * glm::pi<float>() * r * r * r,
              tol * r * r * r);
  EXPECT_NEAR(prop.surfaceArea, 4 * glm::pi<float>() * r * r, tol * r * r);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("close.glb", result.GetMesh(), {});
#endif

  PolygonParams().processOverlaps = false;
}

TEST(Boolean, BooleanVolumes) {
  glm::mat4 m = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f));

  // Define solids which volumes are easy to compute w/ bit arithmetics:
  // m1, m2, m4 are unique, non intersecting "bits" (of volume 1, 2, 4)
  // m3 = m1 + m2
  // m7 = m1 + m2 + m3
  auto m1 = Manifold::Cube({1, 1, 1});
  auto m2 = Manifold::Cube({2, 1, 1}).Transform(
      glm::mat4x3(glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0, 0))));
  auto m4 = Manifold::Cube({4, 1, 1}).Transform(
      glm::mat4x3(glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 0, 0))));
  auto m3 = Manifold::Cube({3, 1, 1});
  auto m7 = Manifold::Cube({7, 1, 1});

  EXPECT_FLOAT_EQ((m1 ^ m2).GetProperties().volume, 0);
  EXPECT_FLOAT_EQ((m1 + m2 + m4).GetProperties().volume, 7);
  EXPECT_FLOAT_EQ((m1 + m2 - m4).GetProperties().volume, 3);
  EXPECT_FLOAT_EQ((m1 + (m2 ^ m4)).GetProperties().volume, 1);
  EXPECT_FLOAT_EQ((m7 ^ m4).GetProperties().volume, 4);
  EXPECT_FLOAT_EQ((m7 ^ m3 ^ m1).GetProperties().volume, 1);
  EXPECT_FLOAT_EQ((m7 ^ (m1 + m2)).GetProperties().volume, 3);
  EXPECT_FLOAT_EQ((m7 - m4).GetProperties().volume, 3);
  EXPECT_FLOAT_EQ((m7 - m4 - m2).GetProperties().volume, 1);
  EXPECT_FLOAT_EQ((m7 - (m7 - m1)).GetProperties().volume, 1);
  EXPECT_FLOAT_EQ((m7 - (m1 + m2)).GetProperties().volume, 4);
}

TEST(Boolean, Spiral) {
  ManifoldParams().deterministic = true;
  const int d = 2;
  std::function<Manifold(const int, const float, const float)> spiral =
      [&](const int rec, const float r, const float add) {
        const float rot = 360.0f / (glm::pi<float>() * r * 2) * d;
        const float rNext = r + add / 360 * rot;
        const Manifold cube =
            Manifold::Cube(glm::vec3(1), true).Translate({0, r, 0});
        if (rec > 0)
          return spiral(rec - 1, rNext, add).Rotate(0, 0, rot) + cube;
        return cube;
      };
  const Manifold result = spiral(120, 25, 2);
  EXPECT_EQ(result.Genus(), -120);
}

TEST(Boolean, Sweep) {
  PolygonParams().processOverlaps = true;

  // generate the minimum equivalent positive angle
  auto minPosAngle = [](float angle) {
    float div = angle / glm::two_pi<float>();
    float wholeDiv = floor(div);
    return angle - wholeDiv * glm::two_pi<float>();
  };

  // calculate determinant
  auto det = [](glm::vec2 v1, glm::vec2 v2) {
    return v1.x * v2.y - v1.y * v2.x;
  };

  // generate sweep profile
  auto generateProfile = []() {
    float filletRadius = 2.5;
    float filletWidth = 5;
    int numberOfArcPoints = 10;
    glm::vec2 arcCenterPoint =
        glm::vec2(filletWidth - filletRadius, filletRadius);
    std::vector<glm::vec2> arcPoints;

    for (int i = 0; i < numberOfArcPoints; i++) {
      float angle = i * glm::pi<float>() / numberOfArcPoints;
      float y = arcCenterPoint.y - cos(angle) * filletRadius;
      float x = arcCenterPoint.x + sin(angle) * filletRadius;
      arcPoints.push_back(glm::vec2(x, y));
    }

    std::vector<glm::vec2> profile;
    profile.push_back(glm::vec2(0, 0));
    profile.push_back(glm::vec2(filletWidth - filletRadius, 0));
    for (int i = 0; i < numberOfArcPoints; i++) {
      profile.push_back(arcPoints[i]);
    }
    profile.push_back(glm::vec2(0, filletWidth));

    CrossSection profileCrossSection = CrossSection(profile);
    return profileCrossSection;
  };

  CrossSection profile = generateProfile();

  auto partialRevolve = [minPosAngle, profile](float startAngle, float endAngle,
                                               int nSegmentsPerRotation) {
    float posEndAngle = minPosAngle(endAngle);
    float totalAngle = 0;
    if (startAngle < 0 && endAngle < 0 && startAngle < endAngle) {
      totalAngle = endAngle - startAngle;
    } else {
      totalAngle = posEndAngle - startAngle;
    }

    int nSegments =
        ceil(totalAngle / glm::two_pi<float>() * nSegmentsPerRotation + 1);
    if (nSegments < 2) {
      nSegments = 2;
    }

    float angleStep = totalAngle / (nSegments - 1);
    auto warpFunc = [nSegments, angleStep, startAngle](glm::vec3& vertex) {
      float zIndex = nSegments - 1 - vertex.z;
      float angle = zIndex * angleStep + startAngle;

      // transform
      vertex.z = vertex.y;
      vertex.y = vertex.x * sin(angle);
      vertex.x = vertex.x * cos(angle);
    };

    return Manifold::Extrude(profile, nSegments - 1, nSegments - 2)
        .Warp(warpFunc);
  };

  auto cutterPrimitives = [det, partialRevolve, profile](
                              glm::vec2 p1, glm::vec2 p2, glm::vec2 p3) {
    glm::vec2 diff = p2 - p1;
    glm::vec2 vec1 = p1 - p2;
    glm::vec2 vec2 = p3 - p2;
    float determinant = det(vec1, vec2);

    float startAngle = atan2(vec1.x, -vec1.y);
    float endAngle = atan2(-vec2.x, vec2.y);

    Manifold round = partialRevolve(startAngle, endAngle, 20)
                         .Translate(glm::vec3(p2.x, p2.y, 0));

    float distance = sqrt(diff.x * diff.x + diff.y * diff.y);
    float angle = atan2(diff.y, diff.x);
    Manifold extrusionPrimitive =
        Manifold::Extrude(profile, distance)
            .Rotate(90, 0, -90)
            .Translate(glm::vec3(distance, 0, 0))
            .Rotate(0, 0, angle * 180 / glm::pi<float>())
            .Translate(glm::vec3(p1.x, p1.y, 0));

    std::vector<Manifold> result;

    if (determinant < 0) {
      result.push_back(round);
      result.push_back(extrusionPrimitive);
    } else {
      result.push_back(extrusionPrimitive);
    }

    return result;
  };

  auto scalePath = [](std::vector<glm::vec2> path, float scale) {
    std::vector<glm::vec2> newPath;
    for (glm::vec2 point : path) {
      newPath.push_back(scale * point);
    }
    return newPath;
  };

  std::vector<glm::vec2> pathPoints = {
      glm::vec2(-21.707751473606564, 10.04202769267855),
      glm::vec2(-21.840846948218307, 9.535474475521578),
      glm::vec2(-21.940954413815387, 9.048287386171369),
      glm::vec2(-22.005569458385835, 8.587741145234093),
      glm::vec2(-22.032187669917704, 8.16111047331591),
      glm::vec2(-22.022356960178296, 7.755456475810721),
      glm::vec2(-21.9823319178086, 7.356408291345673),
      glm::vec2(-21.91208498286602, 6.964505631629036),
      glm::vec2(-21.811437268778267, 6.579251589515578),
      glm::vec2(-21.68020988897306, 6.200149257860059),
      glm::vec2(-21.51822395687812, 5.82670172951726),
      glm::vec2(-21.254086890521585, 5.336709200579579),
      glm::vec2(-21.01963533308061, 4.974523796623895),
      glm::vec2(-20.658228140926262, 4.497743844638198),
      glm::vec2(-20.350337020134603, 4.144115181723373),
      glm::vec2(-19.9542029967, 3.7276501717684054),
      glm::vec2(-20.6969129296381, 3.110639833377638),
      glm::vec2(-21.026318197401537, 2.793796378245609),
      glm::vec2(-21.454710558515973, 2.3418076758544806),
      glm::vec2(-21.735944543382722, 2.014266362004704),
      glm::vec2(-21.958999535447845, 1.7205197644485681),
      glm::vec2(-22.170169612837164, 1.3912359628761894),
      glm::vec2(-22.376940405634056, 1.0213515348242117),
      glm::vec2(-22.62545385249271, 0.507889651991388),
      glm::vec2(-22.77620002102207, 0.13973666928102288),
      glm::vec2(-22.8689989640578, -0.135962138067232),
      glm::vec2(-22.974385239894364, -0.5322784681448909),
      glm::vec2(-23.05966775687304, -0.9551466941218276),
      glm::vec2(-23.102914137841445, -1.2774406685179822),
      glm::vec2(-23.14134824916783, -1.8152432718003662),
      glm::vec2(-23.152085124298473, -2.241104719188421),
      glm::vec2(-23.121576743285054, -2.976332948223073),
      glm::vec2(-23.020491352156856, -3.6736813934577914),
      glm::vec2(-22.843552165110886, -4.364810769710428),
      glm::vec2(-22.60334013490563, -5.033012850282157),
      glm::vec2(-22.305015243491663, -5.67461444847819),
      glm::vec2(-21.942709324216615, -6.330962778427178),
      glm::vec2(-21.648491707764062, -6.799117771996025),
      glm::vec2(-21.15330508818782, -7.496539096945377),
      glm::vec2(-21.10687739725184, -7.656798276710632),
      glm::vec2(-21.01253055778545, -8.364144493707382),
      glm::vec2(-20.923211927856293, -8.782280691344269),
      glm::vec2(-20.771325204062215, -9.258087073404687),
      glm::vec2(-20.554404009259198, -9.72613360625344),
      glm::vec2(-20.384050989017144, -9.985885743112847),
      glm::vec2(-20.134404839253612, -10.263023004626703),
      glm::vec2(-19.756998832033442, -10.613109670467736),
      glm::vec2(-18.83161393127597, -15.68768837402245),
      glm::vec2(-19.155593463785983, -17.65410871259763),
      glm::vec2(-17.930304365744544, -19.005810988385562),
      glm::vec2(-16.893408103100064, -19.50558228186199),
      glm::vec2(-16.27514960757635, -19.8288501942628),
      glm::vec2(-15.183033464853374, -20.47781203017123),
      glm::vec2(-14.906850387751492, -20.693472553142833),
      glm::vec2(-14.585198957236713, -21.015257964547136),
      glm::vec2(-11.013839210807205, -34.70394287828328),
      glm::vec2(-8.79778020674896, -36.17434400175442),
      glm::vec2(-7.850491148257242, -36.48835987119041),
      glm::vec2(-6.982497182376991, -36.74546968896842),
      glm::vec2(-6.6361688522576, -36.81653354539242),
      glm::vec2(-6.0701080598244035, -36.964332993204),
      glm::vec2(-5.472439187922815, -37.08824838436714),
      glm::vec2(-4.802871164820756, -37.20127157090685),
      glm::vec2(-3.6605994233344745, -37.34427653957914),
      glm::vec2(-1.7314396363710867, -37.46415201430501),
      glm::vec2(-0.7021130485987349, -37.5),
      glm::vec2(0.01918509410483974, -37.49359541901704),
      glm::vec2(1.2107837650065625, -37.45093992812552),
      glm::vec2(3.375529069920302, 32.21823383780513),
      glm::vec2(1.9041980552754056, 32.89839543047101),
      glm::vec2(1.4107184651094313, 33.16556804736585),
      glm::vec2(1.1315552947605065, 33.34344755450097),
      glm::vec2(0.8882931135353977, 33.52377699790175),
      glm::vec2(0.6775397019893341, 33.708817857198056),
      glm::vec2(0.49590284067753837, 33.900831612019715),
      glm::vec2(0.2291596803839543, 34.27380625039597),
      glm::vec2(0.03901816126171688, 34.66402375075138),
      glm::vec2(-0.02952797094655369, 34.8933309389416),
      glm::vec2(-0.0561772851849209, 35.044928843125824),
      glm::vec2(-0.067490756643705, 35.27129875796868),
      glm::vec2(-0.05587453990569748, 35.42204271802184),
      glm::vec2(0.013497378362074697, 35.72471438137191),
      glm::vec2(0.07132375113026912, 35.877348797053145),
      glm::vec2(0.18708820875448923, 36.108917464873215),
      glm::vec2(0.39580614140195136, 36.424415957998825),
      glm::vec2(0.8433687814267005, 36.964365016108914),
      glm::vec2(0.7078417131710703, 37.172455373435916),
      glm::vec2(0.5992848016685662, 37.27482757003058),
      glm::vec2(0.40594743344375905, 37.36664006036318),
      glm::vec2(0.1397973410299913, 37.434752779117005)};

  int numPoints = pathPoints.size();
  pathPoints = scalePath(pathPoints, 0.9);

  std::vector<Manifold> result;

  for (int i = 0; i < numPoints; i++) {
    std::vector<Manifold> primitives =
        cutterPrimitives(pathPoints[i], pathPoints[(i + 1) % numPoints],
                         pathPoints[(i + 2) % numPoints]);

    for (Manifold primitive : primitives) {
      result.push_back(primitive);
    }
  }

  // all primitives should be valid
  for (Manifold primitive : result) {
    manifold::Properties properties = primitive.GetProperties();
    if (properties.volume < 0) {
      std::cerr << "INVALID PRIMITIVE" << std::endl;
    }
  }

  Manifold shape = Manifold::BatchBoolean(result, OpType::Add);
  auto prop = shape.GetProperties();

  EXPECT_NEAR(prop.volume, 3757, 1);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("unionError.glb", shape.GetMesh(), {});
#endif

  PolygonParams().processOverlaps = false;
}

TEST(Boolean, InterpolatedNormals) {
  MeshGL a;
  a.numProp = 8;
  a.vertProperties = {
      // 0
      -409.0570983886719, -300, -198.83624267578125, 0, -1, 0,
      590.9429321289062, 301.1637268066406,
      // 1
      -1000, -300, 500, 0, -1, 0, 0, 1000,
      // 2
      -1000, -300, -500, 0, -1, 0, 0, 0,
      // 3
      -1000, -300, -500, -1, 0, 0, 600, 0,
      // 4
      -1000, -300, 500, -1, 0, 0, 600, 1000,
      // 5
      -1000, 300, -500, -1, 0, 0, 0, 0,
      // 6
      7.179656982421875, -300, -330.03717041015625, 0, -1, 0,
      1007.1796264648438, 169.9628448486328,
      // 7
      1000, 300, 500, 0, 0, 1, 2000, 600,
      // 8
      403.5837097167969, 300, 500, 0, 0, 1, 1403.583740234375, 600,
      // 9
      564.2904052734375, 21.64801025390625, 500, 0, 0, 1, 1564.29052734375,
      321.64801025390625,
      // 10
      1000, -300, -500, 0, 0, -1, 2000, 600,
      // 11
      -1000, -300, -500, 0, 0, -1, 0, 600,
      // 12
      -1000, 300, -500, 0, 0, -1, 0, 0,
      // 13
      1000, 300, 500, 0, 1, 0, 0, 1000,
      // 14
      1000, 300, -500, 0, 1, 0, 0, 0,
      // 15
      724.5271606445312, 300, 398.83624267578125, 0, 1, 0, 275.47283935546875,
      898.8362426757812,
      // 16
      -115.35255432128906, -300, 500, 0, -1, 0, 884.6475219726562,
      1000.0001220703125,
      // 17
      -384.7195129394531, 166.55722045898438, 500, 0, 0, 1, 615.280517578125,
      466.5572509765625,
      // 18
      -1000, -300, 500, 0, 0, 1, 0, 0,
      // 19
      -161.6136932373047, -219.87335205078125, 500, 0, 0, 1, 838.3862915039062,
      80.12664794921875,
      // 20
      1000, -300, 500, 0, 0, 1, 2000, 0,
      // 21
      -115.35255432128906, -300, 500, 0, 0, 1, 884.6475219726562, 0,
      // 22
      1000, 300, 500, 1, 0, 0, 600, 1000,
      // 23
      1000, -300, 500, 1, 0, 0, 0, 1000,
      // 24
      1000, 300, -500, 1, 0, 0, 600, 0,
      // 25
      566.6257934570312, 300, 23.1280517578125, 0, 1, 0, 433.3742370605469,
      523.1281127929688,
      // 26
      411.5867004394531, -66.51548767089844, -500, 0, 0, -1, 1411.586669921875,
      366.5155029296875,
      // 27
      375.7498779296875, -4.444300651550293, -500, 0, 0, -1, 1375.7498779296875,
      304.4443054199219,
      // 28
      346.7673034667969, 300, -500, 0, 1, 0, 653.2326049804688, 0,
      // 29
      -153.58984375, 300, -388.552490234375, 0, 1, 0, 1153.58984375,
      111.447509765625,
      // 30
      199.9788818359375, 300, -500, 0, 1, 0, 800.0211791992188, 0,
      // 31
      -1000, 300, -500, 0, 1, 0, 2000, 0,
      // 32
      -153.58987426757812, 300, 44.22247314453125, 0, 1, 0, 1153.58984375,
      544.2224731445312,
      // 33
      199.9788818359375, 300, -500, 0, 0, -1, 1199.9791259765625, 0,
      // 34
      521.6780395507812, -2.9542479515075684, -500, 0, 0, -1, 1521.677978515625,
      302.9542541503906,
      // 35
      346.7673034667969, 300, -500, 0, 0, -1, 1346.767333984375, 0,
      // 36
      1000, 300, -500, 0, 0, -1, 2000, 0,
      // 37
      -1000, 300, 500, -1, 0, 0, 0, 1000,
      // 38
      -1000, 300, 500, 0, 0, 1, 0, 600,
      // 39
      -1000, 300, 500, 0, 1, 0, 2000, 1000,
      // 40
      -153.58985900878906, 300, 500, 0, 0, 1, 846.4102172851562, 600,
      // 41
      88.46627807617188, -253.06915283203125, 500, 0, 0, 1, 1088.4664306640625,
      46.93084716796875,
      // 42
      -153.58985900878906, 300, 500, 0, 1, 0, 1153.58984375, 1000,
      // 43
      7.1796698570251465, -300, 500, 0, 0, 1, 1007.1797485351562, 0,
      // 44
      1000, -300, -500, 0, -1, 0, 2000, 0,
      // 45
      1000, -300, 500, 0, -1, 0, 2000, 1000,
      // 46
      7.1796698570251465, -300, 500, 0, -1, 0, 1007.1796264648438, 1000,
      // 47
      403.5837097167969, 300, 500, 0, 1, 0, 596.4163208007812, 1000,
      // 48
      1000, -300, -500, 1, 0, 0, 0, 0,
      // 49
      492.3005676269531, -19.915321350097656, -500, 0, 0, -1, 1492.300537109375,
      319.91534423828125,
      // 50
      411.5867004394531, -66.51548767089844, -500, -0.5, 0.8660253882408142, 0,
      880.5439453125, 0,
      // 51
      7.179656982421875, -300, -330.03717041015625, -0.5000000596046448,
      0.866025447845459, 0, 383.6058654785156, 0,
      // 52
      492.3005676269531, -19.915321350097656, -500, -0.5, 0.8660253882408142, 0,
      968.1235961914062, 31.876384735107422,
      // 53
      7.1796698570251465, -300, 500, -0.5000000596046448, 0.866025447845459, 0,
      99.71644592285156, 779.979736328125,
      // 54
      88.46627807617188, -253.06915283203125, 500, -0.5, 0.8660253882408142, 0,
      187.91758728027344, 812.0823974609375,
      // 55
      -153.58985900878906, 300, 500, 0.5, -0.866025447845459, 0,
      749.2095947265625, 834.9661865234375,
      // 56
      -384.7195129394531, 166.55722045898438, 500, 0.5000000596046448,
      -0.866025447845459, 0, 1000, 743.6859741210938,
      // 57
      -153.58987426757812, 300, 44.22247314453125, 0.5, -0.8660253882408142, 0,
      593.3245239257812, 406.6754455566406,
      // 58
      564.2904052734375, 21.64801025390625, 500, -0.5, 0.866025447845459, 0,
      704.217041015625, 1000.0000610351562,
      // 59
      -604.9979248046875, 39.37942886352539, -198.83624267578125, 0.5,
      -0.8660253882408142, 0, 1000, 0,
      // 60
      199.9788818359375, 300, -500, 0.29619815945625305, 0.1710100919008255,
      0.9396927356719971, 880.5438842773438, 176.7843475341797,
      // 61
      -153.58984375, 300, -388.552490234375, 0.29619815945625305,
      0.1710100919008255, 0.9396927356719971, 554.6932373046875, 0,
      // 62
      375.7498779296875, -4.444300651550293, -500, 0.29619812965393066,
      0.1710100919008255, 0.9396926760673523, 880.5438842773438,
      528.3263549804688,
      // 63
      566.6257934570312, 300, 23.1280517578125, -0.8137977123260498,
      -0.46984636783599854, 0.342020183801651, 239.89218139648438, 600.1796875,
      // 64
      346.7673034667969, 300, -500, -0.8137977719306946, -0.46984633803367615,
      0.342020183801651, 349.8214111328125, 43.478458404541016,
      // 65
      521.6780395507812, -2.9542479515075684, -500, -0.8137977719306946,
      -0.46984633803367615, 0.342020183801651, 0, 43.478458404541016,
      // 66
      804.9979248046875, 160.62057495117188, 398.83624267578125, -0.5,
      0.8660253882408142, 0, 1000, 1000,
      // 67
      521.6780395507812, -2.9542479515075684, -500, -0.5, 0.8660253882408142, 0,
      1000, 43.47837829589844,
      // 68
      -153.58984375, 300, -388.552490234375, 0.5, -0.8660253882408142, 0,
      445.3067626953125, 0,
      // 69
      -604.9979248046875, 39.37942886352539, -198.83624267578125,
      0.29619815945625305, 0.1710100919008255, 0.9396927356719971, 0, 0,
      // 70
      804.9979248046875, 160.62057495117188, 398.83624267578125,
      -0.813797652721405, -0.46984630823135376, 0.3420201539993286, 0, 1000,
      // 71
      -161.6136932373047, -219.87335205078125, 500, 0.8137977123260498,
      0.46984630823135376, -0.3420201539993286, 446.21160888671875,
      743.68603515625,
      // 72
      -604.9979248046875, 39.37942886352539, -198.83624267578125,
      0.813797652721405, 0.46984630823135376, -0.3420201539993286, 0, 0,
      // 73
      -384.7195129394531, 166.55722045898438, 500, 0.8137977123260498,
      0.46984636783599854, -0.342020183801651, 0, 743.6859741210938,
      // 74
      -115.35255432128906, -300, 500, 0.813797652721405, 0.46984633803367615,
      -0.3420201539993286, 538.73388671875, 743.68603515625,
      // 75
      -409.0570983886719, -300, -198.83624267578125, 0.813797652721405,
      0.46984630823135376, -0.3420201539993286, 391.8816223144531, 0,
      // 76
      7.179656982421875, -300, -330.03717041015625, 0.29619815945625305,
      0.1710100919008255, 0.9396927356719971, 383.6058654785156, 600,
      // 77
      564.2904052734375, 21.64801025390625, 500, -0.29619812965393066,
      -0.1710100919008255, -0.9396926164627075, 704.2169189453125,
      -0.000030517578125,
      // 78
      403.5837097167969, 300, 500, -0.29619812965393066, -0.1710100919008255,
      -0.9396926164627075, 704.2169799804688, 321.4132385253906,
      // 79
      724.5271606445312, 300, 398.83624267578125, -0.29619815945625305,
      -0.1710100919008255, -0.9396926760673523, 1000, 160.94149780273438,
      // 80
      804.9979248046875, 160.62057495117188, 398.83624267578125,
      -0.29619815945625305, -0.1710100919008255, -0.9396927356719971, 1000, 0,
      // 81
      -409.0570983886719, -300, -198.83624267578125, 0.29619815945625305,
      0.1710100919008255, 0.9396927356719971, 0, 391.88165283203125,
      // 82
      724.5271606445312, 300, 398.83624267578125, -0.813797652721405,
      -0.46984630823135376, 0.342020183801651, 160.94149780273438,
      1000.0000610351562,
      // 83
      411.5867004394531, -66.51548767089844, -500, 0.29619815945625305,
      0.1710100769996643, 0.9396926164627075, 880.5440063476562,
      600.0000610351562};
  a.triVerts = {// 0
                0, 1, 2,
                // 1
                3, 4, 5,
                // 2
                6, 0, 2,
                // 3
                7, 8, 9,
                // 4
                10, 11, 12,
                // 5
                13, 14, 15,
                // 6
                0, 16, 1,
                // 7
                17, 18, 19,
                // 8
                9, 20, 7,
                // 9
                18, 21, 19,
                // 10
                22, 23, 24,
                // 11
                14, 25, 15,
                // 12
                26, 12, 27,
                // 13
                14, 28, 25,
                // 14
                29, 30, 31,
                // 15
                29, 31, 32,
                // 16
                12, 33, 27,
                // 17
                34, 35, 36,
                // 18
                5, 4, 37,
                // 19
                17, 38, 18,
                // 20
                31, 39, 32,
                // 21
                40, 38, 17,
                // 22
                9, 41, 20,
                // 23
                39, 42, 32,
                // 24
                41, 43, 20,
                // 25
                6, 2, 44,
                // 26
                6, 45, 46,
                // 27
                26, 10, 12,
                // 28
                47, 13, 15,
                // 29
                48, 24, 23,
                // 30
                6, 44, 45,
                // 31
                26, 49, 10,
                // 32
                49, 34, 10,
                // 33
                34, 36, 10,
                // 34
                50, 51, 52,
                // 35
                51, 53, 54,
                // 36
                51, 54, 52,
                // 37
                55, 56, 57,
                // 38
                52, 54, 58,
                // 39
                59, 57, 56,
                // 40
                60, 61, 62,
                // 41
                63, 64, 65,
                // 42
                52, 66, 67,
                // 43
                59, 68, 57,
                // 44
                69, 62, 61,
                // 45
                65, 70, 63,
                // 46
                71, 72, 73,
                // 47
                52, 58, 66,
                // 48
                74, 72, 71,
                // 49
                74, 75, 72,
                // 50
                62, 69, 76,
                // 51
                77, 78, 79,
                // 52
                79, 80, 77,
                // 53
                69, 81, 76,
                // 54
                63, 70, 82,
                // 55
                76, 83, 62};
  a.mergeFromVert = {3,  4,  11, 12, 13, 18, 21, 22, 23, 24, 31, 33, 35, 36,
                     38, 39, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55,
                     56, 57, 58, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71,
                     72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83};
  a.mergeToVert = {2,  1,  2,  5,  7,  1,  16, 7,  20, 14, 5,  30, 28, 14,
                   37, 37, 40, 10, 20, 43, 8,  10, 26, 6,  49, 43, 41, 40,
                   17, 32, 9,  30, 29, 27, 25, 28, 34, 34, 29, 59, 66, 19,
                   59, 17, 16, 0,  6,  9,  8,  15, 66, 0,  15, 26};

  MeshGL b;
  b.numProp = 8;
  b.vertProperties = {// 0
                      -1700, -600, -1000, -1, 0, 0, 1200, 0,
                      // 1
                      -1700, -600, 1000, -1, 0, 0, 1200, 2000,
                      // 2
                      -1700, 600, -1000, -1, 0, 0, 0, 0,
                      // 3
                      -1700, -600, -1000, 0, -1, 0, 0, 0,
                      // 4
                      300, -600, -1000, 0, -1, 0, 2000, 0,
                      // 5
                      -1700, -600, 1000, 0, -1, 0, 0, 2000,
                      // 6
                      -1700, -600, -1000, 0, 0, -1, 0, 1200,
                      // 7
                      -1700, 600, -1000, 0, 0, -1, 0, 0,
                      // 8
                      300, -600, -1000, 0, 0, -1, 2000, 1200,
                      // 9
                      -1700, -600, 1000, 0, 0, 1, 0, 0,
                      // 10
                      300, -600, 1000, 0, 0, 1, 2000, 0,
                      // 11
                      -1700, 600, 1000, 0, 0, 1, 0, 1200,
                      // 12
                      -1700, 600, 1000, -1, 0, 0, 0, 2000,
                      // 13
                      -1700, 600, -1000, 0, 1, 0, 2000, 0,
                      // 14
                      -1700, 600, 1000, 0, 1, 0, 2000, 2000,
                      // 15
                      300, 600, 1000, 0, 1, 0, 0, 2000,
                      // 16
                      300, -600, -1000, 1, 0, 0, 0, 0,
                      // 17
                      300, 600, -1000, 1, 0, 0, 1200, 0,
                      // 18
                      300, -600, 1000, 1, 0, 0, 0, 2000,
                      // 19
                      300, -600, 1000, 0, -1, 0, 2000, 2000,
                      // 20
                      300, 600, -1000, 0, 0, -1, 2000, 0,
                      // 21
                      300, 600, -1000, 0, 1, 0, 0, 0,
                      // 22
                      300, 600, 1000, 0, 0, 1, 2000, 1200,
                      // 23
                      300, 600, 1000, 1, 0, 0, 1200, 2000};
  b.triVerts = {// 0
                0, 1, 2,
                // 1
                3, 4, 5,
                // 2
                6, 7, 8,
                // 3
                9, 10, 11,
                // 4
                1, 12, 2,
                // 5
                13, 14, 15,
                // 6
                16, 17, 18,
                // 7
                4, 19, 5,
                // 8
                7, 20, 8,
                // 9
                21, 13, 15,
                // 10
                10, 22, 11,
                // 11
                17, 23, 18};
  b.mergeFromVert = {3, 5, 6, 7, 8, 9, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23};
  b.mergeToVert = {0, 1, 0, 2, 4, 1, 11, 2, 11, 4, 10, 10, 17, 17, 15, 15};

  a.runOriginalID = {Manifold::ReserveIDs(1)};
  b.runOriginalID = {Manifold::ReserveIDs(1)};

  Manifold aManifold(a);
  Manifold bManifold(b);

  auto aMinusB = aManifold - bManifold;

  std::vector<MeshGL> meshList;
  meshList.emplace_back(a);
  meshList.emplace_back(b);

  RelatedGL(aMinusB, meshList, false, false);
}
