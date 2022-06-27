#include "meshIO.h"
#include "sdf.h"

using namespace manifold;

struct Test {
  __host__ __device__ float operator()(glm::vec3 point) const {
    return glm::dot(point, {1, 1, 1});
  }
};

int main(int argc, char **argv) {
  Test func;
  SDF<Test> a(func);
  Box box({-1, -1, -1}, {1, 2, 3});
  Mesh b = a.LevelSet(box, 0.5, -0.1);
  // Dump(b.vertPos);
  // Dump(b.triVerts);
  ExportMesh("sdf.gltf", b, {});
}