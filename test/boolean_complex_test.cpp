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

#ifdef MANIFOLD_CROSS_SECTION
#include "manifold/cross_section.h"
#endif
#include "manifold/manifold.h"
#include "manifold/polygon.h"
#include "test.h"

using namespace manifold;

/**
 * These tests verify correct topology and geometry for complex boolean
 * operations between valid shapes with many faces.
 */

TEST(BooleanComplex, Sphere) {
  Manifold sphere = WithPositionColors(Manifold::Sphere(1.0, 12));
  MeshGL sphereGL = sphere.GetMeshGL();

  Manifold sphere2 = sphere.Translate(vec3(0.5));
  Manifold result = sphere - sphere2;

  ExpectMeshes(result, {{74, 144, 3, 110}});
  EXPECT_EQ(result.NumDegenerateTris(), 0);

  RelatedGL(result, {sphereGL});
  result = result.Refine(4);
  RelatedGL(result, {sphereGL});

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorIdx = 0;
  if (options.exportModels)
    ExportMesh("sphereUnion.glb", result.GetMeshGL(), opt);
#endif
}

TEST(BooleanComplex, MeshRelation) {
  Manifold gyroid = WithPositionColors(Gyroid()).AsOriginal();
  MeshGL gyroidMeshGL = gyroid.GetMeshGL();

  Manifold gyroid2 = gyroid.Translate(vec3(2.0));

  EXPECT_FALSE(gyroid.IsEmpty());
  EXPECT_TRUE(gyroid.MatchesTriNormals());
  EXPECT_LE(gyroid.NumDegenerateTris(), 0);
  Manifold result = gyroid + gyroid2;
  result = result.RefineToLength(0.1);

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorIdx = 0;
  if (options.exportModels)
    ExportMesh("gyroidUnion.glb", result.GetMeshGL(), opt);
#endif

  EXPECT_TRUE(result.MatchesTriNormals());
  EXPECT_LE(result.NumDegenerateTris(), 12);
  EXPECT_EQ(result.Decompose().size(), 1);
  EXPECT_NEAR(result.Volume(), 226, 1);
  EXPECT_NEAR(result.SurfaceArea(), 387, 1);

  RelatedGL(result, {gyroidMeshGL});
}

TEST(BooleanComplex, Cylinders) {
  Manifold rod = Manifold::Cylinder(1.0, 0.4, -1.0, 12);
  double arrays1[][12] = {
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
  double arrays2[][12] = {
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
    mat3x4 mat;
    for (const int i : {0, 1, 2, 3}) {
      for (const int j : {0, 1, 2}) {
        mat[i][j] = array[j * 4 + i];
      }
    }
    m1 += rod.Transform(mat);
  }

  Manifold m2;
  for (auto& array : arrays2) {
    mat3x4 mat;
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

TEST(BooleanComplex, Subtract) {
  MeshGL firstMesh;
  firstMesh.vertProperties = {
      0,    0,  0,         //
      1540, 0,  0,         //
      1540, 70, 0,         //
      0,    70, 0,         //
      0,    0,  -278.282,  //
      1540, 70, -278.282,  //
      1540, 0,  -278.282,  //
      0,    70, -278.282   //
  };
  firstMesh.triVerts = {
      0, 1, 2,  //
      2, 3, 0,  //
      4, 5, 6,  //
      5, 4, 7,  //
      6, 2, 1,  //
      6, 5, 2,  //
      5, 3, 2,  //
      5, 7, 3,  //
      7, 0, 3,  //
      7, 4, 0,  //
      4, 1, 0,  //
      4, 6, 1,  //
  };

  MeshGL secondMesh;
  secondMesh.vertProperties = {
      2.04636e-12, 70,           50000,     //
      2.04636e-12, -1.27898e-13, 50000,     //
      1470,        -1.27898e-13, 50000,     //
      1540,        70,           50000,     //
      2.04636e-12, 70,           -28.2818,  //
      1470,        -1.27898e-13, 0,         //
      2.04636e-12, -1.27898e-13, 0,         //
      1540,        70,           -28.2818   //
  };
  secondMesh.triVerts = {
      0, 1, 2,  //
      2, 3, 0,  //
      4, 5, 6,  //
      5, 4, 7,  //
      6, 2, 1,  //
      6, 5, 2,  //
      5, 3, 2,  //
      5, 7, 3,  //
      7, 0, 3,  //
      7, 4, 0,  //
      4, 1, 0,  //
      4, 6, 1   //
  };

  Manifold first(firstMesh);
  Manifold second(secondMesh);

  first -= second;
  first.GetMeshGL();
  EXPECT_EQ(first.Status(), Manifold::Error::NoError);
}

TEST(BooleanComplex, Close) {
  PolygonParams().processOverlaps = true;

  const double r = 10;
  Manifold a = Manifold::Sphere(r, 256);
  Manifold result = a;
  for (int i = 0; i < 10; i++) {
    // std::cout << i << std::endl;
    result ^= a.Translate({a.GetEpsilon() / 10 * i, 0.0, 0.0});
  }
  const double tol = 0.004;
  EXPECT_NEAR(result.Volume(), (4.0 / 3.0) * kPi * r * r * r, tol * r * r * r);
  EXPECT_NEAR(result.SurfaceArea(), 4 * kPi * r * r, tol * r * r);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("close.glb", result.GetMeshGL(), {});
#endif

  PolygonParams().processOverlaps = false;
}

TEST(BooleanComplex, BooleanVolumes) {
  // Define solids which volumes are easy to compute w/ bit arithmetics:
  // m1, m2, m4 are unique, non intersecting "bits" (of volume 1, 2, 4)
  // m3 = m1 + m2
  // m7 = m1 + m2 + m3
  auto m1 = Manifold::Cube({1, 1, 1});
  auto m2 = Manifold::Cube({2, 1, 1}).Translate({1, 0, 0});
  auto m4 = Manifold::Cube({4, 1, 1}).Translate({3, 0, 0});
  auto m3 = Manifold::Cube({3, 1, 1});
  auto m7 = Manifold::Cube({7, 1, 1});

  EXPECT_FLOAT_EQ((m1 ^ m2).Volume(), 0);
  EXPECT_FLOAT_EQ((m1 + m2 + m4).Volume(), 7);
  EXPECT_FLOAT_EQ((m1 + m2 - m4).Volume(), 3);
  EXPECT_FLOAT_EQ((m1 + (m2 ^ m4)).Volume(), 1);
  EXPECT_FLOAT_EQ((m7 ^ m4).Volume(), 4);
  EXPECT_FLOAT_EQ((m7 ^ m3 ^ m1).Volume(), 1);
  EXPECT_FLOAT_EQ((m7 ^ (m1 + m2)).Volume(), 3);
  EXPECT_FLOAT_EQ((m7 - m4).Volume(), 3);
  EXPECT_FLOAT_EQ((m7 - m4 - m2).Volume(), 1);
  EXPECT_FLOAT_EQ((m7 - (m7 - m1)).Volume(), 1);
  EXPECT_FLOAT_EQ((m7 - (m1 + m2)).Volume(), 4);
}

TEST(BooleanComplex, Spiral) {
  const int d = 2;
  std::function<Manifold(const int, const double, const double)> spiral =
      [&](const int rec, const double r, const double add) {
        const double rot = 360.0 / (kPi * r * 2) * d;
        const double rNext = r + add / 360 * rot;
        const Manifold cube =
            Manifold::Cube(vec3(1), true).Translate({0, r, 0});
        if (rec > 0)
          return spiral(rec - 1, rNext, add).Rotate(0, 0, rot) + cube;
        return cube;
      };
  const Manifold result = spiral(120, 25, 2);
  EXPECT_EQ(result.Genus(), -120);
}

#ifdef MANIFOLD_CROSS_SECTION
TEST(BooleanComplex, Sweep) {
  PolygonParams().processOverlaps = true;

  // generate the minimum equivalent positive angle
  auto minPosAngle = [](double angle) {
    double div = angle / kTwoPi;
    double wholeDiv = floor(div);
    return angle - wholeDiv * kTwoPi;
  };

  // calculate determinant
  auto det = [](vec2 v1, vec2 v2) { return v1.x * v2.y - v1.y * v2.x; };

  // generate sweep profile
  auto generateProfile = []() {
    double filletRadius = 2.5;
    double filletWidth = 5;
    int numberOfArcPoints = 10;
    vec2 arcCenterPoint = vec2(filletWidth - filletRadius, filletRadius);
    std::vector<vec2> arcPoints;

    for (int i = 0; i < numberOfArcPoints; i++) {
      double angle = i * kPi / numberOfArcPoints;
      double y = arcCenterPoint.y - cos(angle) * filletRadius;
      double x = arcCenterPoint.x + sin(angle) * filletRadius;
      arcPoints.push_back(vec2(x, y));
    }

    std::vector<vec2> profile;
    profile.push_back(vec2(0, 0));
    profile.push_back(vec2(filletWidth - filletRadius, 0));
    for (int i = 0; i < numberOfArcPoints; i++) {
      profile.push_back(arcPoints[i]);
    }
    profile.push_back(vec2(0, filletWidth));

    CrossSection profileCrossSection = CrossSection(profile);
    return profileCrossSection;
  };

  CrossSection profile = generateProfile();

  auto partialRevolve = [minPosAngle, profile](double startAngle,
                                               double endAngle,
                                               int nSegmentsPerRotation) {
    double posEndAngle = minPosAngle(endAngle);
    double totalAngle = 0;
    if (startAngle < 0 && endAngle < 0 && startAngle < endAngle) {
      totalAngle = endAngle - startAngle;
    } else {
      totalAngle = posEndAngle - startAngle;
    }

    int nSegments = ceil(totalAngle / kTwoPi * nSegmentsPerRotation + 1);
    if (nSegments < 2) {
      nSegments = 2;
    }

    double angleStep = totalAngle / (nSegments - 1);
    auto warpFunc = [nSegments, angleStep, startAngle](vec3& vertex) {
      double zIndex = nSegments - 1 - vertex.z;
      double angle = zIndex * angleStep + startAngle;

      // transform
      vertex.z = vertex.y;
      vertex.y = vertex.x * sin(angle);
      vertex.x = vertex.x * cos(angle);
    };

    return Manifold::Extrude(profile.ToPolygons(), nSegments - 1, nSegments - 2)
        .Warp(warpFunc);
  };

  auto cutterPrimitives = [det, partialRevolve, profile](vec2 p1, vec2 p2,
                                                         vec2 p3) {
    vec2 diff = p2 - p1;
    vec2 vec1 = p1 - p2;
    vec2 vec2 = p3 - p2;
    double determinant = det(vec1, vec2);

    double startAngle = atan2(vec1.x, -vec1.y);
    double endAngle = atan2(-vec2.x, vec2.y);

    Manifold round =
        partialRevolve(startAngle, endAngle, 20).Translate(vec3(p2.x, p2.y, 0));

    double distance = sqrt(diff.x * diff.x + diff.y * diff.y);
    double angle = atan2(diff.y, diff.x);
    Manifold extrusionPrimitive =
        Manifold::Extrude(profile.ToPolygons(), distance)
            .Rotate(90, 0, -90)
            .Translate(vec3(distance, 0, 0))
            .Rotate(0, 0, angle * 180 / kPi)
            .Translate(vec3(p1.x, p1.y, 0));

    std::vector<Manifold> result;

    if (determinant < 0) {
      result.push_back(round);
      result.push_back(extrusionPrimitive);
    } else {
      result.push_back(extrusionPrimitive);
    }

    return result;
  };

  auto scalePath = [](std::vector<vec2> path, double scale) {
    std::vector<vec2> newPath;
    for (vec2 point : path) {
      newPath.push_back(scale * point);
    }
    return newPath;
  };

  std::vector<vec2> pathPoints = {{-21.707751473606564, 10.04202769267855},
                                  {-21.840846948218307, 9.535474475521578},
                                  {-21.940954413815387, 9.048287386171369},
                                  {-22.005569458385835, 8.587741145234093},
                                  {-22.032187669917704, 8.16111047331591},
                                  {-22.022356960178296, 7.755456475810721},
                                  {-21.9823319178086, 7.356408291345673},
                                  {-21.91208498286602, 6.964505631629036},
                                  {-21.811437268778267, 6.579251589515578},
                                  {-21.68020988897306, 6.200149257860059},
                                  {-21.51822395687812, 5.82670172951726},
                                  {-21.254086890521585, 5.336709200579579},
                                  {-21.01963533308061, 4.974523796623895},
                                  {-20.658228140926262, 4.497743844638198},
                                  {-20.350337020134603, 4.144115181723373},
                                  {-19.9542029967, 3.7276501717684054},
                                  {-20.6969129296381, 3.110639833377638},
                                  {-21.026318197401537, 2.793796378245609},
                                  {-21.454710558515973, 2.3418076758544806},
                                  {-21.735944543382722, 2.014266362004704},
                                  {-21.958999535447845, 1.7205197644485681},
                                  {-22.170169612837164, 1.3912359628761894},
                                  {-22.376940405634056, 1.0213515348242117},
                                  {-22.62545385249271, 0.507889651991388},
                                  {-22.77620002102207, 0.13973666928102288},
                                  {-22.8689989640578, -0.135962138067232},
                                  {-22.974385239894364, -0.5322784681448909},
                                  {-23.05966775687304, -0.9551466941218276},
                                  {-23.102914137841445, -1.2774406685179822},
                                  {-23.14134824916783, -1.8152432718003662},
                                  {-23.152085124298473, -2.241104719188421},
                                  {-23.121576743285054, -2.976332948223073},
                                  {-23.020491352156856, -3.6736813934577914},
                                  {-22.843552165110886, -4.364810769710428},
                                  {-22.60334013490563, -5.033012850282157},
                                  {-22.305015243491663, -5.67461444847819},
                                  {-21.942709324216615, -6.330962778427178},
                                  {-21.648491707764062, -6.799117771996025},
                                  {-21.15330508818782, -7.496539096945377},
                                  {-21.10687739725184, -7.656798276710632},
                                  {-21.01253055778545, -8.364144493707382},
                                  {-20.923211927856293, -8.782280691344269},
                                  {-20.771325204062215, -9.258087073404687},
                                  {-20.554404009259198, -9.72613360625344},
                                  {-20.384050989017144, -9.985885743112847},
                                  {-20.134404839253612, -10.263023004626703},
                                  {-19.756998832033442, -10.613109670467736},
                                  {-18.83161393127597, -15.68768837402245},
                                  {-19.155593463785983, -17.65410871259763},
                                  {-17.930304365744544, -19.005810988385562},
                                  {-16.893408103100064, -19.50558228186199},
                                  {-16.27514960757635, -19.8288501942628},
                                  {-15.183033464853374, -20.47781203017123},
                                  {-14.906850387751492, -20.693472553142833},
                                  {-14.585198957236713, -21.015257964547136},
                                  {-11.013839210807205, -34.70394287828328},
                                  {-8.79778020674896, -36.17434400175442},
                                  {-7.850491148257242, -36.48835987119041},
                                  {-6.982497182376991, -36.74546968896842},
                                  {-6.6361688522576, -36.81653354539242},
                                  {-6.0701080598244035, -36.964332993204},
                                  {-5.472439187922815, -37.08824838436714},
                                  {-4.802871164820756, -37.20127157090685},
                                  {-3.6605994233344745, -37.34427653957914},
                                  {-1.7314396363710867, -37.46415201430501},
                                  {-0.7021130485987349, -37.5},
                                  {0.01918509410483974, -37.49359541901704},
                                  {1.2107837650065625, -37.45093992812552},
                                  {3.375529069920302, 32.21823383780513},
                                  {1.9041980552754056, 32.89839543047101},
                                  {1.4107184651094313, 33.16556804736585},
                                  {1.1315552947605065, 33.34344755450097},
                                  {0.8882931135353977, 33.52377699790175},
                                  {0.6775397019893341, 33.708817857198056},
                                  {0.49590284067753837, 33.900831612019715},
                                  {0.2291596803839543, 34.27380625039597},
                                  {0.03901816126171688, 34.66402375075138},
                                  {-0.02952797094655369, 34.8933309389416},
                                  {-0.0561772851849209, 35.044928843125824},
                                  {-0.067490756643705, 35.27129875796868},
                                  {-0.05587453990569748, 35.42204271802184},
                                  {0.013497378362074697, 35.72471438137191},
                                  {0.07132375113026912, 35.877348797053145},
                                  {0.18708820875448923, 36.108917464873215},
                                  {0.39580614140195136, 36.424415957998825},
                                  {0.8433687814267005, 36.964365016108914},
                                  {0.7078417131710703, 37.172455373435916},
                                  {0.5992848016685662, 37.27482757003058},
                                  {0.40594743344375905, 37.36664006036318},
                                  {0.1397973410299913, 37.434752779117005}};

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
    if (primitive.Volume() < 0) {
      std::cerr << "INVALID PRIMITIVE" << std::endl;
    }
  }

  Manifold shape = Manifold::BatchBoolean(result, OpType::Add);

  EXPECT_NEAR(shape.Volume(), 3757, 1);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("unionError.glb", shape.GetMeshGL(), {});
#endif

  PolygonParams().processOverlaps = false;
}
#endif

TEST(BooleanComplex, InterpolatedNormals) {
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
  a.triVerts = {0,  1,  2,   //
                3,  4,  5,   //
                6,  0,  2,   //
                7,  8,  9,   //
                10, 11, 12,  //
                13, 14, 15,  //
                0,  16, 1,   //
                17, 18, 19,  //
                9,  20, 7,   //
                18, 21, 19,  //
                22, 23, 24,  //
                14, 25, 15,  //
                26, 12, 27,  //
                14, 28, 25,  //
                29, 30, 31,  //
                29, 31, 32,  //
                12, 33, 27,  //
                34, 35, 36,  //
                5,  4,  37,  //
                17, 38, 18,  //
                31, 39, 32,  //
                40, 38, 17,  //
                9,  41, 20,  //
                39, 42, 32,  //
                41, 43, 20,  //
                6,  2,  44,  //
                6,  45, 46,  //
                26, 10, 12,  //
                47, 13, 15,  //
                48, 24, 23,  //
                6,  44, 45,  //
                26, 49, 10,  //
                49, 34, 10,  //
                34, 36, 10,  //
                50, 51, 52,  //
                51, 53, 54,  //
                51, 54, 52,  //
                55, 56, 57,  //
                52, 54, 58,  //
                59, 57, 56,  //
                60, 61, 62,  //
                63, 64, 65,  //
                52, 66, 67,  //
                59, 68, 57,  //
                69, 62, 61,  //
                65, 70, 63,  //
                71, 72, 73,  //
                52, 58, 66,  //
                74, 72, 71,  //
                74, 75, 72,  //
                62, 69, 76,  //
                77, 78, 79,  //
                79, 80, 77,  //
                69, 81, 76,  //
                63, 70, 82,  //
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
  b.vertProperties = {-1700, -600, -1000, -1, 0,  0,  1200, 0,     //
                      -1700, -600, 1000,  -1, 0,  0,  1200, 2000,  //
                      -1700, 600,  -1000, -1, 0,  0,  0,    0,     //
                      -1700, -600, -1000, 0,  -1, 0,  0,    0,     //
                      300,   -600, -1000, 0,  -1, 0,  2000, 0,     //
                      -1700, -600, 1000,  0,  -1, 0,  0,    2000,  //
                      -1700, -600, -1000, 0,  0,  -1, 0,    1200,  //
                      -1700, 600,  -1000, 0,  0,  -1, 0,    0,     //
                      300,   -600, -1000, 0,  0,  -1, 2000, 1200,  //
                      -1700, -600, 1000,  0,  0,  1,  0,    0,     //
                      300,   -600, 1000,  0,  0,  1,  2000, 0,     //
                      -1700, 600,  1000,  0,  0,  1,  0,    1200,  //
                      -1700, 600,  1000,  -1, 0,  0,  0,    2000,  //
                      -1700, 600,  -1000, 0,  1,  0,  2000, 0,     //
                      -1700, 600,  1000,  0,  1,  0,  2000, 2000,  //
                      300,   600,  1000,  0,  1,  0,  0,    2000,  //
                      300,   -600, -1000, 1,  0,  0,  0,    0,     //
                      300,   600,  -1000, 1,  0,  0,  1200, 0,     //
                      300,   -600, 1000,  1,  0,  0,  0,    2000,  //
                      300,   -600, 1000,  0,  -1, 0,  2000, 2000,  //
                      300,   600,  -1000, 0,  0,  -1, 2000, 0,     //
                      300,   600,  -1000, 0,  1,  0,  0,    0,     //
                      300,   600,  1000,  0,  0,  1,  2000, 1200,  //
                      300,   600,  1000,  1,  0,  0,  1200, 2000};
  b.triVerts = {0,  1,  2,   //
                3,  4,  5,   //
                6,  7,  8,   //
                9,  10, 11,  //
                1,  12, 2,   //
                13, 14, 15,  //
                16, 17, 18,  //
                4,  19, 5,   //
                7,  20, 8,   //
                21, 13, 15,  //
                10, 22, 11,  //
                17, 23, 18};
  b.mergeFromVert = {3, 5, 6, 7, 8, 9, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23};
  b.mergeToVert = {0, 1, 0, 2, 4, 1, 11, 2, 11, 4, 10, 10, 17, 17, 15, 15};

  a.runOriginalID = {Manifold::ReserveIDs(1)};
  b.runOriginalID = {Manifold::ReserveIDs(1)};

  Manifold aManifold(a);
  Manifold bManifold(b);

  auto aMinusB = aManifold - bManifold;

  RelatedGL(aMinusB, {a, b}, false, false);
}

#ifdef MANIFOLD_EXPORT
TEST(BooleanComplex, SelfIntersect) {
  manifold::PolygonParams().processOverlaps = true;
  Manifold m1 = ReadMesh("self_intersectA.glb");
  Manifold m2 = ReadMesh("self_intersectB.glb");
  Manifold res = m1 + m2;
  res.GetMeshGL();  // test crash
  manifold::PolygonParams().processOverlaps = false;
}

TEST(BooleanComplex, GenericTwinBooleanTest7081) {
  Manifold m1 = ReadMesh("Generic_Twin_7081.1.t0_left.glb");
  Manifold m2 = ReadMesh("Generic_Twin_7081.1.t0_right.glb");
  Manifold res = m1 + m2;  // Union
  res.GetMeshGL();         // test crash
}

TEST(BooleanComplex, GenericTwinBooleanTest7863) {
  manifold::PolygonParams().processOverlaps = true;
  Manifold m1 = ReadMesh("Generic_Twin_7863.1.t0_left.glb");
  Manifold m2 = ReadMesh("Generic_Twin_7863.1.t0_right.glb");
  Manifold res = m1 + m2;  // Union
  res.GetMeshGL();         // test crash
  manifold::PolygonParams().processOverlaps = false;
}

TEST(BooleanComplex, Havocglass8Bool) {
  manifold::PolygonParams().processOverlaps = true;
  Manifold m1 = ReadMesh("Havocglass8_left.glb");
  Manifold m2 = ReadMesh("Havocglass8_right.glb");
  Manifold res = m1 + m2;  // Union
  res.GetMeshGL();         // test crash
  manifold::PolygonParams().processOverlaps = false;
}

TEST(BooleanComplex, CraycloudBool) {
  Manifold m1 = ReadMesh("Cray_left.glb");
  Manifold m2 = ReadMesh("Cray_right.glb");
  Manifold res = m1 - m2;
  EXPECT_EQ(res.Status(), Manifold::Error::NoError);
  EXPECT_TRUE(res.IsEmpty());
}

TEST(BooleanComplex, HullMask) {
  Manifold body = ReadMesh("hull-body.glb");
  Manifold mask = ReadMesh("hull-mask.glb");
  Manifold ret = body - mask;
  MeshGL mesh = ret.GetMeshGL();
}

// Note - For the moment, the Status() checks are included in the loops to
// (more or less) mimic the BRL-CAD behavior of checking the mesh for
// unexpected output after each iteration.  Doing so is not ideal - it
// *massively* slows the overall evaluation - but it also seems to be
// triggering behavior that avoids a triangulation failure.
//
// Eventually, once other issues are resolved, the in-loop checks should be
// removed in favor of the top level checks.
TEST(BooleanComplex, SimpleOffset) {
  std::string file = __FILE__;
  std::string dir = file.substr(0, file.rfind('/'));
  MeshGL seeds = ImportMesh(dir + "/models/" + "Generic_Twin_91.1.t0.glb");
  EXPECT_TRUE(seeds.NumTri() > 10);
  EXPECT_TRUE(seeds.NumVert() > 10);
  // Unique edges
  std::vector<std::pair<int, int>> edges;
  for (size_t i = 0; i < seeds.NumTri(); i++) {
    const int k[3] = {1, 2, 0};
    for (const int j : {0, 1, 2}) {
      int v1 = seeds.triVerts[i * 3 + j];
      int v2 = seeds.triVerts[i * 3 + k[j]];
      if (v2 > v1) edges.push_back(std::make_pair(v1, v2));
    }
  }
  manifold::Manifold c;
  // Vertex Spheres
  Manifold sph = Manifold::Sphere(1, 8);
  for (size_t i = 0; i < seeds.NumVert(); i++) {
    vec3 vpos(seeds.vertProperties[3 * i + 0], seeds.vertProperties[3 * i + 1],
              seeds.vertProperties[3 * i + 2]);
    Manifold vsph = sph.Translate(vpos);
    c += vsph;
  }
  // Edge Cylinders
  for (size_t i = 0; i < edges.size(); i++) {
    vec3 ev1 = vec3(seeds.vertProperties[3 * edges[i].first + 0],
                    seeds.vertProperties[3 * edges[i].first + 1],
                    seeds.vertProperties[3 * edges[i].first + 2]);
    vec3 ev2 = vec3(seeds.vertProperties[3 * edges[i].second + 0],
                    seeds.vertProperties[3 * edges[i].second + 1],
                    seeds.vertProperties[3 * edges[i].second + 2]);
    vec3 edge = ev2 - ev1;
    double len = la::length(edge);
    if (len < std::numeric_limits<float>::min()) continue;
    manifold::Manifold origin_cyl = manifold::Manifold::Cylinder(len, 1, 1, 8);
    vec3 evec(-1 * edge.x, -1 * edge.y, edge.z);
    quat q = rotation_quat(normalize(evec), vec3(0, 0, 1));
    manifold::Manifold right = origin_cyl.Transform({la::qmat(q), ev1});
    c += right;
  }
  // Triangle Volumes
  for (size_t i = 0; i < seeds.NumTri(); i++) {
    int eind[3];
    for (int j = 0; j < 3; j++) eind[j] = seeds.triVerts[i * 3 + j];
    std::vector<vec3> ev;
    for (int j = 0; j < 3; j++) {
      ev.push_back(vec3(seeds.vertProperties[3 * eind[j] + 0],
                        seeds.vertProperties[3 * eind[j] + 1],
                        seeds.vertProperties[3 * eind[j] + 2]));
    }
    vec3 a = ev[0] - ev[2];
    vec3 b = ev[1] - ev[2];
    vec3 n = la::normalize(la::cross(a, b));
    if (!all(isfinite(n))) continue;
    // Extrude the points above and below the plane of the triangle
    vec3 pnts[6];
    for (int j = 0; j < 3; j++) pnts[j] = ev[j] + n;
    for (int j = 3; j < 6; j++) pnts[j] = ev[j - 3] - n;
    // Construct the points and faces of the new manifold
    double pts[3 * 6] = {pnts[4].x, pnts[4].y, pnts[4].z, pnts[3].x, pnts[3].y,
                         pnts[3].z, pnts[0].x, pnts[0].y, pnts[0].z, pnts[1].x,
                         pnts[1].y, pnts[1].z, pnts[5].x, pnts[5].y, pnts[5].z,
                         pnts[2].x, pnts[2].y, pnts[2].z};
    int faces[24] = {
        faces[0] = 0,  faces[1] = 1,  faces[2] = 4,   // 1 2 5
        faces[3] = 2,  faces[4] = 3,  faces[5] = 5,   // 3 4 6
        faces[6] = 1,  faces[7] = 0,  faces[8] = 3,   // 2 1 4
        faces[9] = 3,  faces[10] = 2, faces[11] = 1,  // 4 3 2
        faces[12] = 3, faces[13] = 0, faces[14] = 4,  // 4 1 5
        faces[15] = 4, faces[16] = 5, faces[17] = 3,  // 5 6 4
        faces[18] = 5, faces[19] = 4, faces[20] = 1,  // 6 5 2
        faces[21] = 1, faces[22] = 2, faces[23] = 5   // 2 3 6
    };
    manifold::MeshGL64 tri_m;
    for (int j = 0; j < 18; j++)
      tri_m.vertProperties.insert(tri_m.vertProperties.end(), pts[j]);
    for (int j = 0; j < 24; j++)
      tri_m.triVerts.insert(tri_m.triVerts.end(), faces[j]);
    manifold::Manifold right(tri_m);
    c += right;
    // See above discussion
    EXPECT_EQ(c.Status(), Manifold::Error::NoError);
  }
  // See above discussion
  EXPECT_EQ(c.Status(), Manifold::Error::NoError);
}

#endif
