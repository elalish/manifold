#include <chrono>
#include <iostream>

#include "manifold.h"
#include "manifold_set.h"

using namespace manifold;

int main(int argc, char **argv) {
  const Manifold m = Manifold::Sphere(2.5f, 60);
  for (int separation = 0; separation < 10; separation++) {
    std::cout << "Separation: " << separation << std::endl;
    {
      auto start = std::chrono::high_resolution_clock::now();
      Manifold result;
      for (int i = 0; i < 100; i++) {
        Manifold tmp = m;
        result += tmp.Translate(glm::vec3(separation*(i/10), separation*(i%10), 0));
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "Union:        took " << elapsed.count() << " seconds" << std::endl;
    }

    {
      ManifoldSet mset(m);
      auto start = std::chrono::high_resolution_clock::now();
      ManifoldSet resultSet;
      for (int i = 0; i < 100; i++) {
        glm::mat4x3 t(1.0f);
        t[3] = glm::vec3(separation*(i/10), separation*(i%10), 0);
        resultSet += mset.Transform(t);
      }
      Manifold result = resultSet.ToManifold();
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "Manifold Set: took " << elapsed.count() << " seconds" << std::endl;
    }
    std::cout << "----------------------" << std::endl;
  }
}

