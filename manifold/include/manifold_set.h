#pragma once

#include "manifold.h"

namespace manifold {
class ManifoldSet {
public:
  ManifoldSet();
  ManifoldSet(const std::vector<manifold::Manifold> &manifolds);
  ManifoldSet(const manifold::Manifold &manifold);

  ManifoldSet(const ManifoldSet &other);

  ManifoldSet operator+(const ManifoldSet &other) const;

  ManifoldSet &operator+=(const ManifoldSet &other);

  ManifoldSet operator-(ManifoldSet &other);

  ManifoldSet &operator-=(ManifoldSet &other);

  ManifoldSet operator^(ManifoldSet &other);

  ManifoldSet &operator^=(ManifoldSet &other);

  ManifoldSet Transform(const glm::mat4x3 &transform) const;

  manifold::Manifold ToManifold();

private:
  std::shared_ptr<std::vector<manifold::Manifold>> manifolds;
  glm::mat4x3 transform_ = glm::mat4x3(1.0f);
};
} // namespace manifold
