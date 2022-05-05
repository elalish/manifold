#include <algorithm>
#include <thrust/execution_policy.h>
#include "manifold_set.h"
#include "utils.cuh"
#include "vec_dh.cuh"

namespace manifold {

ManifoldSet::ManifoldSet()
    : manifolds(std::make_shared<std::vector<Manifold>>()) {}

ManifoldSet::ManifoldSet(const std::vector<Manifold> &manifolds)
    : manifolds(std::make_shared<std::vector<Manifold>>(manifolds)) {}

ManifoldSet::ManifoldSet(const Manifold &manifold)
    : manifolds(std::make_shared<std::vector<Manifold>>(
          std::vector<Manifold>({manifold}))) {}

ManifoldSet ManifoldSet::operator+(const ManifoldSet &other) const {
  if (manifolds == other.manifolds && transform_ == other.transform_) {
    return *this;
  }
  if (manifolds->empty()) {
    return other;
  }
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
  if (manifolds == other.manifolds && transform_ == other.transform_) {
      return *this;
  }
  if (manifolds->empty()) {
    manifolds = other.manifolds;
    transform_ = other.transform_;
    return *this;
  }
  if (manifolds.use_count() != 1) {
    auto old = manifolds;
    manifolds = std::make_shared<std::vector<Manifold>>();
    manifolds->insert(manifolds->end(), old->begin(), old->end());
  }
  if (transform_ != glm::mat4x3(1.0f))
    for (auto &m : *manifolds)
      m.Transform(transform_);
  transform_ = glm::mat4x3(1.0f);

  manifolds->reserve(other.manifolds->size());
  int end = other.manifolds->size();
  for (int i = 0; i < end; ++i) {
    manifolds->push_back(other.manifolds->at(i));
    if (other.transform_ != glm::mat4x3(1.0f))
      manifolds->back().Transform(other.transform_);
  }
  return *this;
}

ManifoldSet ManifoldSet::operator-(ManifoldSet &other) {
  return ManifoldSet({this->ToManifold() - other.ToManifold()});
}

ManifoldSet &ManifoldSet::operator-=(ManifoldSet &other) {
  auto result = this->ToManifold() - other.ToManifold();
  manifolds = std::make_shared<std::vector<Manifold>>();
  manifolds->push_back(result);
  return *this;
}

ManifoldSet ManifoldSet::operator^(ManifoldSet &other) {
  return ManifoldSet({this->ToManifold() ^ other.ToManifold()});
}

ManifoldSet &ManifoldSet::operator^=(ManifoldSet &other) {
  auto result = this->ToManifold() ^ other.ToManifold();
  manifolds = std::make_shared<std::vector<Manifold>>();
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

  // sort the manifolds by their distance to a point, hopefully
  // the combined bounding box of the first few manifolds will be
  // smaller, so we can compose more manifolds together cheaply
  glm::vec3 min_pt(0);
  for (auto &m : *manifolds) {
    Box b = m.BoundingBox();
    min_pt = glm::min(min_pt, b.min);
  }

  std::sort(manifolds->begin(), manifolds->end(), 
          [&min_pt](const Manifold &a, const Manifold &b) {
            return glm::distance(min_pt, a.BoundingBox().min) < glm::distance(min_pt, b.BoundingBox().min);
          });

  std::vector<std::pair<std::vector<size_t>, Box>> disjointSets;
  for (size_t i = 0; i < manifolds->size(); i++) {
    Box box = manifolds->at(i).BoundingBox();
    auto it = std::find_if(disjointSets.begin(), disjointSets.end(),
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

  std::vector<std::unique_ptr<Manifold>> results(disjointSets.size());
  thrust::for_each_n(thrust::host, countAt(0), disjointSets.size(),
                     [&](size_t i) {
                       if (disjointSets[i].first.size() == 1) {
                         results[i] = std::make_unique<Manifold>(manifolds->at(disjointSets[i].first[0]));
                         return;
                       }
                       std::vector<Manifold> buffer;
                       buffer.reserve(disjointSets[i].first.size());
                       for (auto &i : disjointSets[i].first) {
                         buffer.push_back(std::move(manifolds->at(i)));
                       }
                       results[i] = std::make_unique<Manifold>(Manifold::Compose(buffer));
                     });

  auto cmp_fn = [](const std::unique_ptr<Manifold> &a, const std::unique_ptr<Manifold> &b) {
                  // invert the order because we want a min heap
                  return a->NumVert() > b->NumVert();
                };

  // union smaller manifolds first to avoid copying large manifolds repeatedly
  std::make_heap(results.begin(), results.end(), cmp_fn);
  while (results.size() > 1) {
    std::pop_heap(results.begin(), results.end(), cmp_fn);
    auto a = std::move(results.back());
    results.pop_back();
    std::pop_heap(results.begin(), results.end(), cmp_fn);
    *results.back() += *a;
    std::push_heap(results.begin(), results.end(), cmp_fn);
  }

  manifolds->clear();
  manifolds->push_back(*results.back());

  if (transform_ != glm::mat4x3(1.0f)) {
    manifolds = std::make_shared<std::vector<Manifold>>();
    manifolds->push_back(*results.back());
    manifolds->back().Transform(transform_);
    transform_ = glm::mat4x3(1.0f);
  }
  return manifolds->back();
}

} // namespace manifold
