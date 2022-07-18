#include "manifold.h"
#include "meshIO.h"

using namespace manifold;

int main(int argc, char** argv) {
  const int n = 24;
  const int m = 1000;
  const float r0 = 10;
  const float r1 = 50;
  const float h = 300;
  const int k = 10;

  Polygons circle(1);
  float dPhi = 360.0f / n;
  for (int i = 0; i < n; ++i) {
    circle[0].push_back(
        {r0 * glm::vec2(cosd(dPhi * i) + 2.0f, sind(dPhi * i)), 0});
  }

  Manifold spring = Manifold::Extrude(circle, 1, m);

  spring.Warp([r1, h, k](glm::vec3& v) {
    const float phi = v.z * k * 2 * glm::pi<float>();
    v.x += v.z * h;
    const float r = v.y + r1;
    v.y = r * glm::cos(phi);
    v.z = r * glm::sin(phi);
  });

  ExportOptions opts;
  opts.mat.roughness = 0.1;
  opts.mat.metalness = 1;
  opts.mat.color = glm::vec4(1.0, 1.0, 0.0, 1.0);
  ExportMesh("spring.glb", spring.GetMesh(), opts);
}