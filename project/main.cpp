#include <array>
#include <cassert>
#include <iomanip>
#include <iostream>

#include "manifold/manifold.h"

using namespace manifold;

int main() {
  auto obj = manifold::Manifold::Cube();
  auto mesh = obj.GetMeshGL();
  Manifold::Fillet(mesh,5,{});
  
  return 0;
}
