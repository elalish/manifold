#include "sdf.h"

using namespace manifold;

struct Test {
  __host__ __device__ float operator()(glm::vec3 point) const {
    return point[1];
  }
};

int main(int argc, char **argv) {
  Test func;
  SDF<Test> a(func);
  Box box;
  Mesh b = a.LevelSet(box, 1);
}