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

#include "cross_section.h"
#include "test.h"

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

namespace {

using namespace manifold;

template <typename T>
int NumUnique(const std::vector<T>& in) {
  std::set<int> unique;
  for (const T& v : in) {
    unique.emplace(v);
  }
  return unique.size();
}

}  // namespace

/**
 * This tests that turning a mesh into a manifold and returning it to a mesh
 * produces a consistent result.
 */
TEST(Manifold, GetMesh) {
  Manifold manifold = Manifold::Sphere(1);
  Mesh mesh_out = manifold.GetMesh();
  Manifold manifold2(mesh_out);
  Mesh mesh_out2 = manifold2.GetMesh();
  Identical(mesh_out, mesh_out2);
}

TEST(Manifold, GetMeshGL) {
  Manifold manifold = Manifold::Sphere(1);
  Mesh mesh_out = manifold.GetMesh();
  MeshGL meshGL_out = manifold.GetMeshGL();
  ASSERT_EQ(meshGL_out.NumVert(), mesh_out.vertPos.size());
  ASSERT_EQ(meshGL_out.NumTri(), mesh_out.triVerts.size());
  for (int i = 0; i < meshGL_out.NumVert(); ++i) {
    for (const int j : {0, 1, 2}) {
      ASSERT_EQ(meshGL_out.vertProperties[3 * i + j], mesh_out.vertPos[i][j]);
    }
  }
  for (int i = 0; i < meshGL_out.NumTri(); ++i) {
    for (const int j : {0, 1, 2})
      ASSERT_EQ(meshGL_out.triVerts[3 * i + j], mesh_out.triVerts[i][j]);
  }
}

TEST(Manifold, Empty) {
  Mesh emptyMesh;
  Manifold empty(emptyMesh);

  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_EQ(empty.Status(), Manifold::Error::NoError);
}

TEST(Manifold, ValidInput) {
  std::vector<float> propTol = {0.1, 0.2};
  MeshGL tetGL = TetGL();
  Manifold tet(tetGL, propTol);
  EXPECT_FALSE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NoError);
}

TEST(Manifold, InvalidInput1) {
  Mesh in = Tet();
  in.vertPos[2][1] = NAN;
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, InvalidInput2) {
  Mesh in = Tet();
  std::swap(in.triVerts[2][1], in.triVerts[2][2]);
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NotManifold);
}

TEST(Manifold, InvalidInput3) {
  Mesh in = Tet();
  for (glm::ivec3& tri : in.triVerts) {
    for (int i : {0, 1, 2}) {
      if (tri[i] == 2) tri[i] = -2;
    }
  }
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
}

TEST(Manifold, InvalidInput4) {
  Mesh in = Tet();
  for (glm::ivec3& tri : in.triVerts) {
    for (int i : {0, 1, 2}) {
      if (tri[i] == 2) tri[i] = 4;
    }
  }
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
}

TEST(Manifold, InvalidInput5) {
  MeshGL tetGL = TetGL();
  tetGL.mergeFromVert[tetGL.mergeFromVert.size() - 1] = 7;
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::MergeIndexOutOfBounds);
}

TEST(Manifold, InvalidInput7) {
  MeshGL tetGL = TetGL();
  tetGL.triVerts[tetGL.triVerts.size() - 1] = 7;
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
}

/**
 * ExpectMeshes performs a decomposition, so this test ensures that compose and
 * decompose are inverse operations.
 */
TEST(Manifold, Decompose) {
  std::vector<Manifold> manifoldList;
  manifoldList.emplace_back(Manifold::Tetrahedron());
  manifoldList.emplace_back(Manifold::Cube());
  manifoldList.emplace_back(Manifold::Sphere(1, 4));
  Manifold manifolds = Manifold::Compose(manifoldList);

  ExpectMeshes(manifolds, {{8, 12}, {6, 8}, {4, 4}});

  std::vector<MeshGL> input;

  for (const Manifold& manifold : manifoldList) {
    EXPECT_GE(manifold.OriginalID(), 0);
    input.emplace_back(manifold.GetMeshGL());
  }

  RelatedGL(manifolds, input);
}

/**
 * These tests check the various manifold constructors.
 */
TEST(Manifold, Sphere) {
  int n = 25;
  Manifold sphere = Manifold::Sphere(1.0f, 4 * n);
  EXPECT_EQ(sphere.NumTri(), n * n * 8);
}

TEST(Manifold, Normals) {
  Mesh cube = Manifold::Cube(glm::vec3(1), true).GetMesh();
  const int nVert = cube.vertPos.size();
  for (int i = 0; i < nVert; ++i) {
    glm::vec3 v = glm::normalize(cube.vertPos[i]);
    glm::vec3& n = cube.vertNormal[i];
    EXPECT_FLOAT_EQ(v.x, n.x);
    EXPECT_FLOAT_EQ(v.y, n.y);
    EXPECT_FLOAT_EQ(v.z, n.z);
  }
}

TEST(Manifold, Extrude) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0f, 3);
  EXPECT_EQ(donut.Genus(), 1);
  auto prop = donut.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 12.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 48.0f);
}

TEST(Manifold, ExtrudeCone) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0f, 0, 0, glm::vec2(0.0f));
  EXPECT_EQ(donut.Genus(), 0);
  EXPECT_FLOAT_EQ(donut.GetProperties().volume, 4.0f);
}

TEST(Manifold, Revolve) {
  Polygons polys = SquareHole();
  Manifold vug = Manifold::Revolve(polys, 48);
  EXPECT_EQ(vug.Genus(), -1);
  auto prop = vug.GetProperties();
  EXPECT_NEAR(prop.volume, 14.0f * glm::pi<float>(), 0.2f);
  EXPECT_NEAR(prop.surfaceArea, 30.0f * glm::pi<float>(), 0.2f);
}

TEST(Manifold, Revolve2) {
  Polygons polys = SquareHole(2.0f);
  Manifold donutHole = Manifold::Revolve(polys, 48);
  EXPECT_EQ(donutHole.Genus(), 0);
  auto prop = donutHole.GetProperties();
  EXPECT_NEAR(prop.volume, 48.0f * glm::pi<float>(), 1.0f);
  EXPECT_NEAR(prop.surfaceArea, 96.0f * glm::pi<float>(), 1.0f);
}

TEST(Manifold, Warp) {
  CrossSection square = CrossSection::Square({1, 1});
  Manifold shape = Manifold::Extrude(square, 2, 10).Warp([](glm::vec3& v) {
    v.x += v.z * v.z;
  });
  auto propBefore = shape.GetProperties();

  Manifold simplified = Manifold::Compose({shape});
  auto propAfter = simplified.GetProperties();

  EXPECT_NEAR(propBefore.volume, propAfter.volume, 0.0001);
  EXPECT_NEAR(propBefore.surfaceArea, propAfter.surfaceArea, 0.0001);
  EXPECT_NEAR(propBefore.volume, 2, 0.0001);
}

TEST(Manifold, Warp2) {
  CrossSection circle =
      CrossSection::Circle(5, 20).Translate(glm::vec2(10.0, 10.0));

  Manifold shape = Manifold::Extrude(circle, 2, 10).Warp([](glm::vec3& v) {
    int nSegments = 10;
    double angleStep = 2.0 / 3.0 * glm::pi<float>() / nSegments;
    int zIndex = nSegments - 1 - std::round(v.z);
    double angle = zIndex * angleStep;
    v.z = v.y;
    v.y = v.x * sin(angle);
    v.x = v.x * cos(angle);
  });

  auto propBefore = shape.GetProperties();

  Manifold simplified = Manifold::Compose({shape});
  auto propAfter = simplified.GetProperties();

  EXPECT_NEAR(propBefore.volume, propAfter.volume, 0.0001);
  EXPECT_NEAR(propBefore.surfaceArea, propAfter.surfaceArea, 0.0001);
  EXPECT_NEAR(propBefore.volume, 321, 1);
}

TEST(Manifold, DISABLED_UnionError) {
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
      glm::vec2(-0.12489098552494904, 37.4823871569106),
      glm::vec2(-0.35268401789311693, 37.49999999999999),
      glm::vec2(-1.0329383665594345, 37.49736832454696),
      glm::vec2(-1.7857460661542182, 37.44480548672453),
      glm::vec2(-2.341285800658867, 37.35505516104463),
      glm::vec2(-2.7408413518578136, 37.25251732873489),
      glm::vec2(-3.1866639886956873, 37.08515010697008),
      glm::vec2(-3.575655171293931, 36.87337410959638),
      glm::vec2(-3.9107595345872275, 36.6162691381967),
      glm::vec2(-4.12193071621465, 36.40304987095235),
      glm::vec2(-4.263464103578334, 36.229665839817535),
      glm::vec2(-4.444411184156673, 35.93749327232712),
      glm::vec2(-4.551492342784406, 35.704274259319284),
      glm::vec2(-4.755819579357982, 35.09300730867552),
      glm::vec2(-5.305202572986668, 32.93227370078094),
      glm::vec2(-5.546452832392646, 32.1332543728385),
      glm::vec2(-5.68346448742785, 31.775892004846316),
      glm::vec2(-5.8696795659142875, 31.382975470513728),
      glm::vec2(-6.081884069017544, 31.058848460870728),
      glm::vec2(-6.289867039189264, 30.810164654611793),
      glm::vec2(-6.645018845197344, 30.460931687515572),
      glm::vec2(-6.978446576301429, 30.178809343096965),
      glm::vec2(-7.846778009614522, 29.539372665043544),
      glm::vec2(-8.91665745023197, 28.834768547309896),
      glm::vec2(-10.998916782503146, 27.545966863748244),
      glm::vec2(-11.790922726189875, 27.037610977892186),
      glm::vec2(-12.42998727600341, 26.590951083187882),
      glm::vec2(-13.012629675833251, 26.136022653586952),
      glm::vec2(-13.635369169568841, 25.602861163041016),
      glm::vec2(-14.452677717110518, 24.86485922592765),
      glm::vec2(-15.106583226994921, 24.23958256111645),
      glm::vec2(-15.533420391437897, 23.803874223833486),
      glm::vec2(-16.023415742317397, 23.25989304029938),
      glm::vec2(-16.335436437545443, 22.87460531892385),
      glm::vec2(-16.603219145990813, 22.504466640049234),
      glm::vec2(-16.829745310524892, 22.14560112794273),
      glm::vec2(-17.017996374019063, 21.79413290687155),
      glm::vec2(-17.170953779344703, 21.446186101102906),
      glm::vec2(-17.29159896937319, 21.097884834903994),
      glm::vec2(-17.40145676537794, 20.681459715550417),
      glm::vec2(-17.459807651243878, 20.39236484368582),
      glm::vec2(-17.5210089437266, 19.954529191702218),
      glm::vec2(-17.5506201926299, 19.51192654383872),
      glm::vec2(-17.548641397953787, 19.064855044382455),
      glm::vec2(-17.515072559698257, 18.613612837620554),
      glm::vec2(-17.4499136778633, 18.15849806784016),
      glm::vec2(-17.353164752448937, 17.699808879328415),
      glm::vec2(-17.27111544462856, 17.3921775305499),
      glm::vec2(-17.00456709033987, 16.617308349964848),
      glm::vec2(-16.620481546849707, 15.742342516114123),
      glm::vec2(-16.640871303375647, 15.718060320283884),
      glm::vec2(-16.733821649724252, 15.679933187149437),
      glm::vec2(-17.179281728385796, 15.569061547202017),
      glm::vec2(-17.577997813242117, 15.437864810004104),
      glm::vec2(-17.9734011837969, 15.272798881787319),
      glm::vec2(-18.228331480950736, 15.146922609143909),
      glm::vec2(-18.587471672284742, 14.938835564531207),
      glm::vec2(-18.85663154968114, 14.751609540727115),
      glm::vec2(-19.161771307878972, 14.498256831739537),
      glm::vec2(-19.438325531675755, 14.232175202088003),
      glm::vec2(-19.79859965284586, 13.84004994637427),
      glm::vec2(-20.041019668479642, 13.54144187923408),
      glm::vec2(-20.2603086258297, 13.230658206738129),
      glm::vec2(-20.544170145256615, 12.770388025325978),
      glm::vec2(-20.810904931744606, 12.283410741214864),
      glm::vec2(-21.058008573281707, 11.777001075010961),
      glm::vec2(-21.282976657855965, 11.258433747320426),
      glm::vec2(-21.48330477345543, 10.734983478749434),
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
      glm::vec2(-19.66622555244656, -10.675037306234351),
      glm::vec2(-19.561203797876278, -10.712349143571252),
      glm::vec2(-19.490038842319517, -10.8201375053558),
      glm::vec2(-19.413804575029562, -11.01911200908371),
      glm::vec2(-19.25465196453413, -11.533819472493908),
      glm::vec2(-18.97167990889718, -12.580312545776795),
      glm::vec2(-18.850876260701273, -13.157656811200383),
      glm::vec2(-18.777669691589793, -13.679188466071738),
      glm::vec2(-18.74633583124969, -14.179571752842104),
      glm::vec2(-18.752336506474116, -14.731663565224466),
      glm::vec2(-18.780667820725814, -15.21146630766375),
      glm::vec2(-18.83161393127597, -15.68768837402245),
      glm::vec2(-19.155593463785983, -17.65410871259763),
      glm::vec2(-19.220073116496003, -18.149447348952368),
      glm::vec2(-19.23857033161078, -18.448921629535842),
      glm::vec2(-19.193497786210358, -18.482765300385278),
      glm::vec2(-18.92833341034099, -18.569108867485422),
      glm::vec2(-17.930304365744544, -19.005810988385562),
      glm::vec2(-16.893408103100064, -19.50558228186199),
      glm::vec2(-16.27514960757635, -19.8288501942628),
      glm::vec2(-15.183033464853374, -20.47781203017123),
      glm::vec2(-14.906850387751492, -20.693472553142833),
      glm::vec2(-14.585198957236713, -21.015257964547136),
      glm::vec2(-14.43765618343693, -21.23395244304647),
      glm::vec2(-14.160352181969897, -21.728643382408702),
      glm::vec2(-13.865189337703312, -22.39453870958846),
      glm::vec2(-13.43188961977709, -23.618417695575545),
      glm::vec2(-13.363561574593666, -23.76300111126595),
      glm::vec2(-13.30544797722622, -23.824551956454048),
      glm::vec2(-13.254935709556444, -23.939905085385835),
      glm::vec2(-13.1367400263677, -24.314418479555155),
      glm::vec2(-13.066427051852049, -24.448916643287358),
      glm::vec2(-12.98903419340841, -24.660146104147866),
      glm::vec2(-12.703702994416272, -25.61223688067309),
      glm::vec2(-12.355362497091518, -27.14919399256258),
      glm::vec2(-12.046798128464633, -29.205139085512332),
      glm::vec2(-11.897908552261589, -29.834756019524157),
      glm::vec2(-11.74188216252183, -30.902716217639725),
      glm::vec2(-11.543445177362626, -32.06537587525509),
      glm::vec2(-11.329259303027674, -33.05536508493485),
      glm::vec2(-11.013839210807205, -34.70394287828328),
      glm::vec2(-10.946186223182565, -34.93493869191542),
      glm::vec2(-10.842605008560968, -35.15608200244232),
      glm::vec2(-10.779643560546031, -35.232643246903145),
      glm::vec2(-10.673294757165172, -35.28178981742884),
      glm::vec2(-10.317818427848673, -35.54829890991839),
      glm::vec2(-10.231599516960774, -35.58646137867205),
      glm::vec2(-10.123291917882744, -35.675530696176736),
      glm::vec2(-10.036437947393916, -35.719043566606736),
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
      glm::vec2(2.0834533203906367, -37.39769896541626),
      glm::vec2(2.89493336183173, -37.32603636655554),
      glm::vec2(3.562376662125365, -37.24718303053085),
      glm::vec2(4.285158778142906, -37.13742421720249),
      glm::vec2(4.974519287651649, -37.00545107792679),
      glm::vec2(5.640116593237398, -36.84994441625301),
      glm::vec2(6.326176949153003, -36.659235073360854),
      glm::vec2(6.778064552411368, -36.51073124433429),
      glm::vec2(7.339847894522547, -36.302171585814555),
      glm::vec2(7.949878098493745, -36.02534645560377),
      glm::vec2(8.61115869195835, -35.69816119719486),
      glm::vec2(9.69836257014464, -35.05893395630116),
      glm::vec2(9.832487746785112, -34.94377517327507),
      glm::vec2(9.953998107365342, -34.784093510036385),
      glm::vec2(10.088693561638165, -34.51511877656149),
      glm::vec2(10.190817918125825, -34.156656587421224),
      glm::vec2(10.421166893617317, -32.67292178888639),
      glm::vec2(10.478929834041292, -32.3599836558365),
      glm::vec2(10.536856692476633, -32.15758122896524),
      glm::vec2(10.599812987380433, -31.787758638246892),
      glm::vec2(10.877059200386869, -29.94392835972801),
      glm::vec2(11.032319003498634, -29.22221502349864),
      glm::vec2(11.379131235292286, -26.928606481879605),
      glm::vec2(11.643805820217526, -25.537418614707356),
      glm::vec2(11.844030198782095, -24.69360500936585),
      glm::vec2(11.993774700661174, -24.285635654613472),
      glm::vec2(12.102180454237029, -23.89791066503716),
      glm::vec2(12.149176827794777, -23.807751832606638),
      glm::vec2(12.210028690265252, -23.77533532815555),
      glm::vec2(12.25331089762537, -23.7132467190125),
      glm::vec2(12.537179594599936, -22.99223715053253),
      glm::vec2(12.827037679031678, -22.382185352571806),
      glm::vec2(13.028835474196352, -22.010073798964218),
      glm::vec2(13.184649850220817, -21.759277646565565),
      glm::vec2(13.448447915480497, -21.433382751723652),
      glm::vec2(13.745141479140644, -21.102566758440243),
      glm::vec2(14.05339567955441, -20.799167156795694),
      glm::vec2(14.197115458357096, -20.713424417915363),
      glm::vec2(14.517653694173436, -20.464265114461025),
      glm::vec2(14.740577281904601, -20.31555319789905),
      glm::vec2(15.38803477744076, -19.981228181318478),
      glm::vec2(16.01445248828915, -19.70628945786274),
      glm::vec2(17.120449885480532, -19.362853863487313),
      glm::vec2(17.90598131310459, -19.172595111086743),
      glm::vec2(18.110745337193485, -19.0962260040554),
      glm::vec2(18.164943551642764, -19.039562025927676),
      glm::vec2(18.151398967101894, -18.80962763049636),
      glm::vec2(17.848474555919932, -16.761928496908745),
      glm::vec2(17.779940631677327, -16.01936518225177),
      glm::vec2(17.769694160956444, -15.560110838545521),
      glm::vec2(17.787593860565725, -15.13539878032697),
      glm::vec2(17.843409415752888, -14.56914351929497),
      glm::vec2(17.920190771675088, -14.11831794770305),
      glm::vec2(17.966227148234204, -13.962708990638555),
      glm::vec2(18.024301652505567, -13.88892573317909),
      glm::vec2(18.084441894866824, -13.740371845358606),
      glm::vec2(18.345699721722838, -12.955500997504899),
      glm::vec2(18.674426987292566, -12.069919708704678),
      glm::vec2(19.20859357989621, -10.749839867480434),
      glm::vec2(19.829675612242934, -10.10905871819308),
      glm::vec2(20.2356332905624, -9.612447508774054),
      glm::vec2(20.528057624339727, -9.144988185208536),
      glm::vec2(20.6336883508858, -8.919048509135585),
      glm::vec2(20.719140644472418, -8.688499329141813),
      glm::vec2(20.78490547696503, -8.4552701632781),
      glm::vec2(20.842001334883175, -8.141194051519088),
      glm::vec2(20.86417971174569, -7.824909463559035),
      glm::vec2(20.851487721711543, -7.5078769383255715),
      glm::vec2(20.803972478939677, -7.191557014746313),
      glm::vec2(20.745511629362895, -6.9556633223460755),
      glm::vec2(20.667509078121373, -6.721608186383179),
      glm::vec2(20.515459862623487, -6.360846774141305),
      glm::vec2(20.491816407187834, -6.27704246389986),
      glm::vec2(20.49436842413124, -6.221977790621039),
      glm::vec2(21.170307226261073, -5.386888525125412),
      glm::vec2(21.378132566443952, -5.086855990835558),
      glm::vec2(21.63881864335015, -4.663575515967764),
      glm::vec2(21.878845161554516, -4.217538902324637),
      glm::vec2(22.09613909806303, -3.7528921958942334),
      glm::vec2(22.288627429881664, -3.273781442664601),
      glm::vec2(22.415467947773067, -2.9074504528200746),
      glm::vec2(22.629042258240005, -2.121117225641919),
      glm::vec2(22.807840491474753, -1.166490136892936),
      glm::vec2(22.918836664941285, -0.4536663314619614),
      glm::vec2(23.081609213316433, 0.869866216916318),
      glm::vec2(23.128690490514803, 1.4656159843457806),
      glm::vec2(23.15208512429847, 2.022498042954719),
      glm::vec2(23.144451817705804, 2.7318330368633283),
      glm::vec2(23.08185863309739, 3.441945577346499),
      glm::vec2(22.960630405351836, 4.098238104929806),
      glm::vec2(22.77870000074499, 4.713729586818291),
      glm::vec2(22.579230276902276, 5.2048642315238105),
      glm::vec2(22.28064374240505, 5.7793669472403835),
      glm::vec2(21.948519579715263, 6.3022384320948985),
      glm::vec2(21.400465766690147, 7.02967288726958),
      glm::vec2(20.83601397186252, 7.67191555702952),
      glm::vec2(20.207316132029696, 8.29555402327205),
      glm::vec2(19.508765662276012, 8.904875674462087),
      glm::vec2(18.935639310336857, 9.35505063509984),
      glm::vec2(18.10135589326836, 9.949634388441723),
      glm::vec2(17.181801737760928, 10.541691645009639),
      glm::vec2(16.17137025889891, 11.135509793268499),
      glm::vec2(15.115818000902786, 11.704140101751664),
      glm::vec2(15.324545012841291, 12.276054583048785),
      glm::vec2(15.52936204035901, 12.945190480974182),
      glm::vec2(15.662321141565341, 13.46939660052497),
      glm::vec2(15.808433927024959, 14.180911684430514),
      glm::vec2(15.9038180141471, 14.799854439497183),
      glm::vec2(15.943065090101706, 15.19531891122686),
      glm::vec2(15.956582927541918, 15.54137240875234),
      glm::vec2(15.951476930565157, 16.268932808417038),
      glm::vec2(15.919890567779406, 17.23703675510339),
      glm::vec2(15.881980356085151, 17.795601611230946),
      glm::vec2(15.80287111288067, 18.46129403465974),
      glm::vec2(15.722566255872861, 18.888758590383492),
      glm::vec2(15.59121682708427, 19.378549841254074),
      glm::vec2(15.401852722258068, 19.889133987577118),
      glm::vec2(15.21324633564762, 20.294860604074056),
      glm::vec2(14.936655461114714, 20.77715392487577),
      glm::vec2(14.557439668994373, 21.326547715499213),
      glm::vec2(14.260873092715784, 21.705226786556143),
      glm::vec2(13.930139917289916, 22.093069561529852),
      glm::vec2(13.16904773269261, 22.8936078303261),
      glm::vec2(12.279911042594982, 23.722885736084983),
      glm::vec2(11.268477774389558, 24.57562649300353),
      glm::vec2(10.140495855468865, 25.446553315278752),
      glm::vec2(9.221482101818513, 26.108508720915893),
      glm::vec2(7.606574889997275, 27.21119229956763),
      glm::vec2(7.443590450599947, 27.95729592426797),
      glm::vec2(7.32770507890372, 28.353763780351645),
      glm::vec2(7.129753958508734, 28.85689686537853),
      glm::vec2(6.944796285149339, 29.22399198798861),
      glm::vec2(6.650521861781168, 29.702550376781744),
      glm::vec2(6.394298869890994, 30.037317824761264),
      glm::vec2(6.075054801128455, 30.396758368858602),
      glm::vec2(5.602482118998823, 30.862999104333657),
      glm::vec2(5.251206311416662, 31.16437439215373),
      glm::vec2(5.0349552664475175, 31.32576934215193),
      glm::vec2(4.476172507334794, 31.65925280660681),
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
  pathPoints = scalePath(pathPoints, 1.1);

  std::vector<Manifold> result;

  for (int i = 0; i < numPoints; i++) {
    // std::cerr << i << std::endl;
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

  std::cerr << "Shape volume: " << prop.volume << std::endl;
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("unionError.glb", shape.GetMesh(), {});
#endif
}

TEST(Manifold, Smooth) {
  Manifold tet = Manifold::Tetrahedron();
  Manifold smooth = Manifold::Smooth(tet.GetMesh());
  smooth = smooth.Refine(100);
  ExpectMeshes(smooth, {{20002, 40000}});
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 17.38, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 33.38, 0.1);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("smoothTet.glb", smooth.GetMesh(), {});
#endif
}

TEST(Manifold, SmoothSphere) {
  int n[5] = {4, 8, 16, 32, 64};
  float precision[5] = {0.03, 0.003, 0.003, 0.0005, 0.00006};
  for (int i = 0; i < 5; ++i) {
    Manifold sphere = Manifold::Sphere(1, n[i]);
    // Refine(odd) puts a center point in the triangle, which is the worst case.
    Manifold smoothed = Manifold::Smooth(sphere.GetMesh()).Refine(7);
    Mesh out = smoothed.GetMesh();
    auto bounds =
        std::minmax_element(out.vertPos.begin(), out.vertPos.end(),
                            [](const glm::vec3& a, const glm::vec3& b) {
                              return glm::dot(a, a) < glm::dot(b, b);
                            });
    float min = glm::length(*bounds.first);
    float max = glm::length(*bounds.second);
    EXPECT_NEAR(min, 1, precision[i]);
    EXPECT_NEAR(max, 1, precision[i]);
  }
}

TEST(Manifold, ManualSmooth) {
  // Unit Octahedron
  const Mesh oct = Manifold::Sphere(1, 4).GetMesh();
  Mesh smooth = Manifold::Smooth(oct).GetMesh();
  // Sharpen the edge from vert 4 to 5
  smooth.halfedgeTangent[6] = {0, 0, 0, 1};
  smooth.halfedgeTangent[22] = {0, 0, 0, 1};
  smooth.halfedgeTangent[16] = {0, 0, 0, 1};
  smooth.halfedgeTangent[18] = {0, 0, 0, 1};
  Manifold interp(smooth);
  interp = interp.Refine(100);

  ExpectMeshes(interp, {{40002, 80000}});
  auto prop = interp.GetProperties();
  EXPECT_NEAR(prop.volume, 3.53, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 11.39, 0.01);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    const Mesh out = interp.GetMesh();
    ExportOptions options;
    options.faceted = false;
    options.mat.roughness = 0.1;

    options.mat.vertColor.resize(interp.NumVert());
    const glm::vec4 red(1, 0, 0, 1);
    const glm::vec4 purple(1, 0, 1, 1);
    for (int tri = 0; tri < interp.NumTri(); ++tri) {
      for (int i : {0, 1, 2}) {
        const glm::vec3& uvw = {0.5, 0.5, 0.0};
        const float alpha = glm::min(uvw[0], glm::min(uvw[1], uvw[2]));
        options.mat.vertColor[out.triVerts[tri][i]] =
            glm::mix(purple, red, glm::smoothstep(0.0f, 0.2f, alpha));
      }
    }
    ExportMesh("sharpenedSphere.glb", out, options);
  }
#endif
}

TEST(Manifold, SmoothMirrored) {
  const Mesh tet = Manifold::Tetrahedron().GetMesh();
  Manifold smooth = Manifold::Smooth(tet);
  Manifold mirror = smooth.Scale({-1, 1, 2}).Refine(10);
  smooth = smooth.Refine(10).Scale({1, 1, 2});

  auto prop0 = smooth.GetProperties();
  auto prop1 = mirror.GetProperties();
  EXPECT_NEAR(prop0.volume, prop1.volume, 0.1);
  EXPECT_NEAR(prop0.surfaceArea, prop1.surfaceArea, 0.1);
}

TEST(Manifold, Csaszar) {
  Manifold csaszar = Manifold::Smooth(Csaszar());
  csaszar = csaszar.Refine(100);
  ExpectMeshes(csaszar, {{70000, 140000}});
  auto prop = csaszar.GetProperties();
  EXPECT_NEAR(prop.volume, 84699, 10);
  EXPECT_NEAR(prop.surfaceArea, 14796, 10);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    const Mesh out = csaszar.GetMesh();
    ExportOptions options;
    options.faceted = false;
    options.mat.roughness = 0.1;

    options.mat.vertColor.resize(csaszar.NumVert());
    const glm::vec4 blue(0, 0, 1, 1);
    const glm::vec4 yellow(1, 1, 0, 1);
    for (int tri = 0; tri < csaszar.NumTri(); ++tri) {
      for (int i : {0, 1, 2}) {
        const glm::vec3& uvw = {0.5, 0.5, 0.0};
        const float alpha = glm::min(uvw[0], glm::min(uvw[1], uvw[2]));
        options.mat.vertColor[out.triVerts[tri][i]] =
            glm::mix(yellow, blue, glm::smoothstep(0.0f, 0.2f, alpha));
      }
    }
    ExportMesh("smoothCsaszar.glb", out, options);
  }
#endif
}

/**
 * These tests verify the calculation of a manifold's geometric properties.
 */
TEST(Manifold, GetProperties) {
  Manifold cube = Manifold::Cube();
  auto prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 1.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0f);

  cube = cube.Scale(glm::vec3(-1.0f));
  prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 1.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0f);
}

TEST(Manifold, Precision) {
  Manifold cube = Manifold::Cube();
  EXPECT_FLOAT_EQ(cube.Precision(), kTolerance);
  cube = cube.Scale({0.1, 1, 10});
  EXPECT_FLOAT_EQ(cube.Precision(), 10 * kTolerance);
  cube = cube.Translate({-100, -10, -1});
  EXPECT_FLOAT_EQ(cube.Precision(), 100 * kTolerance);
}

TEST(Manifold, Precision2) {
  Manifold cube = Manifold::Cube();
  cube = cube.Translate({-0.5, 0, 0}).Scale({2, 1, 1});
  EXPECT_FLOAT_EQ(cube.Precision(), 2 * kTolerance);
}

TEST(Manifold, Precision3) {
  Manifold cylinder = Manifold::Cylinder(1, 1, 1, 1000);
  const auto prop = cylinder.GetProperties();

  MeshGL mesh = cylinder.GetMeshGL();
  mesh.precision = 0.001;
  mesh.faceID.clear();
  Manifold cylinder2(mesh);

  const auto prop2 = cylinder2.GetProperties();
  EXPECT_NEAR(prop.volume, prop2.volume, 0.001);
  EXPECT_NEAR(prop.surfaceArea, prop2.surfaceArea, 0.001);
}

/**
 * Curvature is the inverse of the radius of curvature, and signed such that
 * positive is convex and negative is concave. There are two orthogonal
 * principal curvatures at any point on a manifold, with one maximum and the
 * other minimum. Gaussian curvature is their product, while mean
 * curvature is their sum. Here we check our discrete approximations calculated
 * at each vertex against the constant expected values of spheres of different
 * radii and at different mesh resolutions.
 */
TEST(Manifold, GetCurvature) {
  const float precision = 0.015;
  for (int n = 4; n < 100; n *= 2) {
    Manifold sphere = Manifold::Sphere(1, 64);
    Curvature curvature = sphere.GetCurvature();
    EXPECT_NEAR(curvature.minMeanCurvature, 2, 2 * precision);
    EXPECT_NEAR(curvature.maxMeanCurvature, 2, 2 * precision);
    EXPECT_NEAR(curvature.minGaussianCurvature, 1, precision);
    EXPECT_NEAR(curvature.maxGaussianCurvature, 1, precision);

    sphere = sphere.Scale(glm::vec3(2.0f));
    curvature = sphere.GetCurvature();
    EXPECT_NEAR(curvature.minMeanCurvature, 1, precision);
    EXPECT_NEAR(curvature.maxMeanCurvature, 1, precision);
    EXPECT_NEAR(curvature.minGaussianCurvature, 0.25, 0.25 * precision);
    EXPECT_NEAR(curvature.maxGaussianCurvature, 0.25, 0.25 * precision);
  }
}

/**
 * Testing more advanced Manifold operations.
 */

TEST(Manifold, Transform) {
  Manifold cube = Manifold::Cube({1, 2, 3});
  Manifold cube2 = cube;
  cube = cube.Rotate(30, 40, 50).Scale({6, 5, 4}).Translate({1, 2, 3});

  glm::mat3 rX(1.0f, 0.0f, 0.0f,          //
               0.0f, cosd(30), sind(30),  //
               0.0f, -sind(30), cosd(30));
  glm::mat3 rY(cosd(40), 0.0f, -sind(40),  //
               0.0f, 1.0f, 0.0f,           //
               sind(40), 0.0f, cosd(40));
  glm::mat3 rZ(cosd(50), sind(50), 0.0f,   //
               -sind(50), cosd(50), 0.0f,  //
               0.0f, 0.0f, 1.0f);
  glm::mat3 s = glm::mat3(1.0f);
  s[0][0] = 6;
  s[1][1] = 5;
  s[2][2] = 4;
  glm::mat4x3 transform = glm::mat4x3(s * rZ * rY * rX);
  transform[3] = glm::vec3(1, 2, 3);
  cube2 = cube2.Transform(transform);

  Identical(cube.GetMesh(), cube2.GetMesh());
}

TEST(Manifold, MeshRelation) {
  Mesh gyroidMesh = Gyroid();
  MeshGL gyroidMeshGL = WithIndexColors(gyroidMesh);
  Manifold gyroid(gyroidMeshGL);

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorChannels = glm::ivec4(3, 4, 5, -1);
  if (options.exportModels) ExportMesh("gyroid.glb", gyroid.GetMeshGL(), opt);
#endif

  RelatedGL(gyroid, {gyroidMeshGL});
}

TEST(Manifold, MeshRelationTransform) {
  const Manifold cube = Manifold::Cube();
  const MeshGL cubeGL = cube.GetMeshGL();
  const Manifold turned = cube.Rotate(45, 90);

  RelatedGL(turned, {cubeGL});
}

TEST(Manifold, MeshRelationRefine) {
  const Mesh in = Csaszar();
  MeshGL inGL = WithIndexColors(in);
  Manifold csaszar(inGL);

  RelatedGL(csaszar, {inGL});
  csaszar.Refine(4);
  RelatedGL(csaszar, {inGL});
}

TEST(Manifold, MeshGLRoundTrip) {
  const Manifold cylinder = Manifold::Cylinder(2, 1);
  ASSERT_GE(cylinder.OriginalID(), 0);
  MeshGL inGL = cylinder.GetMeshGL();
  const Manifold cylinder2(inGL);
  const MeshGL outGL = cylinder2.GetMeshGL();

  ASSERT_EQ(inGL.runOriginalID.size(), 1);
  ASSERT_EQ(outGL.runOriginalID.size(), 1);
  ASSERT_EQ(outGL.runOriginalID[0], inGL.runOriginalID[0]);

  RelatedGL(cylinder2, {inGL});
}

void CheckCube(const MeshGL& cubeSTL) {
  Manifold cube(cubeSTL);
  EXPECT_EQ(cube.NumTri(), 12);
  EXPECT_EQ(cube.NumVert(), 8);
  EXPECT_EQ(cube.NumPropVert(), 24);

  auto prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 1.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0f);
}

TEST(Manifold, Merge) {
  MeshGL cubeSTL = CubeSTL();
  EXPECT_EQ(cubeSTL.NumTri(), 12);
  EXPECT_EQ(cubeSTL.NumVert(), 36);

  Manifold cubeBad(cubeSTL);
  EXPECT_TRUE(cubeBad.IsEmpty());
  EXPECT_EQ(cubeBad.Status(), Manifold::Error::NotManifold);

  EXPECT_TRUE(cubeSTL.Merge());
  CheckCube(cubeSTL);

  EXPECT_FALSE(cubeSTL.Merge());
  EXPECT_EQ(cubeSTL.mergeFromVert.size(), 28);
  cubeSTL.mergeFromVert.resize(14);
  cubeSTL.mergeToVert.resize(14);

  EXPECT_TRUE(cubeSTL.Merge());
  EXPECT_EQ(cubeSTL.mergeFromVert.size(), 28);
  CheckCube(cubeSTL);
}

TEST(Manifold, FaceIDRoundTrip) {
  const Manifold cube = Manifold::Cube();
  ASSERT_GE(cube.OriginalID(), 0);
  MeshGL inGL = cube.GetMeshGL();
  ASSERT_EQ(NumUnique(inGL.faceID), 6);
  inGL.faceID = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  const Manifold cube2(inGL);
  const MeshGL outGL = cube2.GetMeshGL();
  ASSERT_EQ(NumUnique(outGL.faceID), 12);
}

TEST(Manifold, MirrorUnion) {
  auto a = Manifold::Cube({5., 5., 5.}, true);
  auto b = a.Translate({2.5, 2.5, 2.5});
  auto result = a + b + b.Mirror({1, 1, 0});

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("manifold_mirror_union.glb", result.GetMesh(), {});
#endif

  auto vol_a = a.GetProperties().volume;
  EXPECT_FLOAT_EQ(vol_a * 2.75, result.GetProperties().volume);
  EXPECT_TRUE(a.Mirror(glm::vec3(0)).IsEmpty());
}

TEST(Manifold, MirrorUnion2) {
  auto a = Manifold::Cube();
  auto result = Manifold::Compose({a.Mirror({1, 0, 0})});
  EXPECT_TRUE(result.MatchesTriNormals());
}

TEST(Manifold, Invalid) {
  auto invalid = Manifold::Error::InvalidConstruction;
  auto circ = CrossSection::Circle(10.);
  auto empty_circ = CrossSection::Circle(-2.);
  auto empty_sq = CrossSection::Square(glm::vec3(0.0f));

  EXPECT_EQ(Manifold::Sphere(0).Status(), invalid);
  EXPECT_EQ(Manifold::Cylinder(0, 5).Status(), invalid);
  EXPECT_EQ(Manifold::Cylinder(2, -5).Status(), invalid);
  EXPECT_EQ(Manifold::Cube(glm::vec3(0.0f)).Status(), invalid);
  EXPECT_EQ(Manifold::Cube({-1, 1, 1}).Status(), invalid);
  EXPECT_EQ(Manifold::Extrude(circ, 0.).Status(), invalid);
  EXPECT_EQ(Manifold::Extrude(empty_circ, 10.).Status(), invalid);
  EXPECT_EQ(Manifold::Revolve(empty_sq).Status(), invalid);
}

TEST(Manifold, MultiCompose) {
  auto legAssembly = Manifold::Compose({Manifold::Cube({100, 140, 38})});
  auto finalAssembly = Manifold::Compose(
      {legAssembly, legAssembly.Translate({0, 300, 0}),
       legAssembly.Mirror({1, 0, 0}).Translate({300, 0, 0}),
       legAssembly.Mirror({1, 0, 0}).Translate({300, 300, 0})});
  finalAssembly.GetMesh();
}
