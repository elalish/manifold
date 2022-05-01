#include "manifold_set.h"
#include <algorithm>
#include "utils.cuh"
#include <thrust/execution_policy.h>

namespace manifold {

ManifoldSet::ManifoldSet()
    : manifolds(std::make_shared<std::vector<Manifold>>()) {}

ManifoldSet::ManifoldSet(const std::vector<Manifold> &manifolds)
    : manifolds(std::make_shared<std::vector<Manifold>>(manifolds)) {}

ManifoldSet::ManifoldSet(const Manifold &manifold)
    : manifolds(std::make_shared<std::vector<Manifold>>(
          std::vector<Manifold>({manifold}))) {}

ManifoldSet::ManifoldSet(const ManifoldSet &other)
    : transform_(other.transform_), manifolds(other.manifolds) {}

ManifoldSet ManifoldSet::operator+(const ManifoldSet &other) const {
  ManifoldSet result;
  result.manifolds->reserve(manifolds->size() + other.manifolds->size());
  result.manifolds->insert(result.manifolds->end(), manifolds->begin(),
                           manifolds->end());
  if (transform_ != glm::mat4x3(1.0f))
    for (auto &m : *result.manifolds) {
      m.Transform(transform_);
    }
  int index = result.manifolds->size();
  result.manifolds->insert(result.manifolds->end(), other.manifolds->begin(),
                           other.manifolds->end());
  for (int i = index; i < result.manifolds->size(); i++) {
    result.manifolds->at(i).Transform(other.transform_);
  }
  return result;
}

ManifoldSet &ManifoldSet::operator+=(const ManifoldSet &other) {
  if (manifolds.use_count() != 1) {
    auto old = manifolds;
    manifolds = std::make_shared<std::vector<Manifold>>(old->size());
    manifolds->insert(manifolds->end(), old->begin(), old->end());
  }
  if (transform_ != glm::mat4x3(1.0f))
    for (auto &m : *manifolds)
      m.Transform(transform_);
  transform_ = glm::mat4x3(1.0f);

  manifolds->reserve(other.manifolds->size());
  int index = manifolds->size();
  manifolds->insert(manifolds->end(), other.manifolds->begin(),
                    other.manifolds->end());
  for (int i = index; i < manifolds->size(); i++) {
    manifolds->at(i).Transform(other.transform_);
  }
  return *this;
}

ManifoldSet ManifoldSet::operator-(ManifoldSet &other) {
  return ManifoldSet({this->ToManifold() - other.ToManifold()});
}

ManifoldSet &ManifoldSet::operator-=(ManifoldSet &other) {
  auto result = this->ToManifold() - other.ToManifold();
  transform_ = glm::mat4x3(1.0f);
  manifolds = std::make_shared<std::vector<Manifold>>(1);
  manifolds->push_back(result);
  return *this;
}

ManifoldSet ManifoldSet::operator^(ManifoldSet &other) {
  return ManifoldSet({this->ToManifold() ^ other.ToManifold()});
}

ManifoldSet &ManifoldSet::operator^=(ManifoldSet &other) {
  auto result = this->ToManifold() ^ other.ToManifold();
  transform_ = glm::mat4x3(1.0f);
  manifolds = std::make_shared<std::vector<Manifold>>(1);
  manifolds->push_back(result);
  return *this;
}

ManifoldSet ManifoldSet::Transform(const glm::mat4x3 &transform) const {
  ManifoldSet result;
  result.manifolds = manifolds;
  result.transform_ = transform * glm::mat4(transform_);
  return result;
}

Manifold ManifoldSet::ToManifold() {
  if (manifolds->size() == 0) {
    return Manifold();
  }
  std::vector<std::pair<std::vector<size_t>, Box>> disjointSets;
  for (size_t i = 0; i < manifolds->size(); i++) {
    Box box = manifolds->at(i).BoundingBox();
    auto it =
        std::find_if(disjointSets.begin(), disjointSets.end(),
                     [&box](const std::pair<std::vector<size_t>, Box> &p) {
                       return !p.second.DoesOverlap(box);
                     });
    if (it == disjointSets.end()) {
      disjointSets.push_back(std::make_pair(std::vector<size_t>{i}, box));
    } else {
      it->first.push_back(i);
      it->second = it->second.Union(box);
    }
  }
  std::vector<Manifold> results(disjointSets.size());
  thrust::for_each_n(thrust::host, countAt(0), disjointSets.size(),
                     [&](size_t i) {
                       if (disjointSets[i].first.size() == 1) {
                         results[i] = manifolds->at(disjointSets[i].first[0]);
                         return;
                       }
                       std::vector<Manifold> buffer;
                       buffer.reserve(disjointSets[i].first.size());
                       for (auto &i : disjointSets[i].first) {
                         buffer.push_back(std::move(manifolds->at(i)));
                       }
                       results[i] = Manifold::Compose(buffer);
                     });
  Manifold result;
  for (auto &m : results) {
    result += m;
  }

  manifolds->clear();
  manifolds->push_back(result);

  if (transform_ != glm::mat4x3(1.0f)) {
    manifolds = std::make_shared<std::vector<Manifold>>();
    manifolds->push_back(result);
    manifolds->back().Transform(transform_);
    transform_ = glm::mat4x3(1.0f);
  }
  return manifolds->back();
}

} // namespace manifold
