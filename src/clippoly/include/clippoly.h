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

#pragma once

#include <clipper2/clipper.h>

#include "glm/ext/vector_float2.hpp"
#include "public.h"

namespace C2 = Clipper2Lib;

namespace manifold {
class Clippoly {
 public:
  Clippoly();
  ~Clippoly();
  Clippoly(const Clippoly& other);
  Clippoly& operator=(const Clippoly& other);
  Clippoly(Clippoly&&) noexcept;
  Clippoly& operator=(Clippoly&&) noexcept;
  Clippoly(std::vector<glm::vec2> contour);
  Clippoly(std::vector<std::vector<glm::vec2>> contours);

  static Clippoly Square(glm::vec2 dims, bool center = true);

  enum class OpType { Add, Subtract, Intersect, Xor };
  Clippoly Boolean(const Clippoly& second, OpType op) const;
  static Clippoly BatchBoolean(const std::vector<Clippoly>& clippolys,
                               OpType op);
  // Boolean operation shorthand
  Clippoly operator+(const Clippoly&) const;  // Add (Union)
  Clippoly& operator+=(const Clippoly&);
  Clippoly operator-(const Clippoly&) const;  // Subtract (Difference)
  Clippoly& operator-=(const Clippoly&);
  Clippoly operator^(const Clippoly&) const;  // Intersect
  Clippoly& operator^=(const Clippoly&);

  Clippoly Translate(glm::vec2 v);

  Polygons ToPolygons() const;

 private:
  C2::PathsD paths_;
  // if not clean, paths_ need to be unioned before extraction
  bool clean_ = false;
  Clippoly(C2::PathsD paths, bool clean);
};

}  // namespace manifold
