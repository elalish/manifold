// Copyright 2022 The Manifold Authors.
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

#include "csg_tree.h"

#include <algorithm>

#include "boolean3.h"
#include "impl.h"
#include "par.h"

namespace {
using namespace manifold;
struct Transform4x3 {
  const glm::mat4x3 transform;

  __host__ __device__ glm::vec3 operator()(glm::vec3 position) {
    return transform * glm::vec4(position, 1.0f);
  }
};

struct TransformNormals {
  const glm::mat3 transform;

  __host__ __device__ glm::vec3 operator()(glm::vec3 normal) {
    normal = glm::normalize(transform * normal);
    if (isnan(normal.x)) normal = glm::vec3(0.0f);
    return normal;
  }
};

struct UpdateHalfedge {
  const int nextVert;
  const int nextEdge;
  const int nextFace;

  __host__ __device__ Halfedge operator()(Halfedge edge) {
    edge.startVert += nextVert;
    edge.endVert += nextVert;
    edge.pairedHalfedge += nextEdge;
    edge.face += nextFace;
    return edge;
  }
};

struct UpdateMeshIDs {
  const int offset;

  __host__ __device__ TriRef operator()(TriRef ref) {
    ref.meshID += offset;
    return ref;
  }
};

struct CheckOverlap {
  const Box *boxes;
  const size_t i;
  __host__ __device__ bool operator()(int j) {
    return boxes[i].DoesOverlap(boxes[j]);
  }
};
}  // namespace
namespace manifold {

std::shared_ptr<CsgNode> CsgNode::Translate(const glm::vec3 &t) const {
  glm::mat4x3 transform(1.0f);
  transform[3] += t;
  return Transform(transform);
}

std::shared_ptr<CsgNode> CsgNode::Scale(const glm::vec3 &v) const {
  glm::mat4x3 transform(1.0f);
  for (int i : {0, 1, 2}) transform[i] *= v;
  return Transform(transform);
}

std::shared_ptr<CsgNode> CsgNode::Rotate(float xDegrees, float yDegrees,
                                         float zDegrees) const {
  glm::mat3 rX(1.0f, 0.0f, 0.0f,                      //
               0.0f, cosd(xDegrees), sind(xDegrees),  //
               0.0f, -sind(xDegrees), cosd(xDegrees));
  glm::mat3 rY(cosd(yDegrees), 0.0f, -sind(yDegrees),  //
               0.0f, 1.0f, 0.0f,                       //
               sind(yDegrees), 0.0f, cosd(yDegrees));
  glm::mat3 rZ(cosd(zDegrees), sind(zDegrees), 0.0f,   //
               -sind(zDegrees), cosd(zDegrees), 0.0f,  //
               0.0f, 0.0f, 1.0f);
  glm::mat4x3 transform(rZ * rY * rX);
  return Transform(transform);
}

CsgLeafNode::CsgLeafNode() : pImpl_(std::make_shared<Manifold::Impl>()) {}

CsgLeafNode::CsgLeafNode(std::shared_ptr<const Manifold::Impl> pImpl_)
    : pImpl_(pImpl_) {}

CsgLeafNode::CsgLeafNode(std::shared_ptr<const Manifold::Impl> pImpl_,
                         glm::mat4x3 transform_)
    : pImpl_(pImpl_), transform_(transform_) {}

std::shared_ptr<const Manifold::Impl> CsgLeafNode::GetImpl() const {
  if (transform_ == glm::mat4x3(1.0f)) return pImpl_;
  pImpl_ =
      std::make_shared<const Manifold::Impl>(pImpl_->Transform(transform_));
  transform_ = glm::mat4x3(1.0f);
  return pImpl_;
}

glm::mat4x3 CsgLeafNode::GetTransform() const { return transform_; }

std::shared_ptr<CsgLeafNode> CsgLeafNode::ToLeafNode() const {
  return std::make_shared<CsgLeafNode>(*this);
}

std::shared_ptr<CsgNode> CsgLeafNode::Transform(const glm::mat4x3 &m) const {
  return std::make_shared<CsgLeafNode>(pImpl_, m * glm::mat4(transform_));
}

CsgNodeType CsgLeafNode::GetNodeType() const { return CsgNodeType::LEAF; }

/**
 * Efficient union of a set of pairwise disjoint meshes.
 */
Manifold::Impl CsgLeafNode::Compose(
    const std::vector<std::shared_ptr<CsgLeafNode>> &nodes) {
  float precision = -1;
  int numVert = 0;
  int numEdge = 0;
  int numTri = 0;
  for (auto &node : nodes) {
    float nodeOldScale = node->pImpl_->bBox_.Scale();
    float nodeNewScale =
        node->pImpl_->bBox_.Transform(node->transform_).Scale();
    float nodePrecision = node->pImpl_->precision_;
    nodePrecision *= glm::max(1.0f, nodeNewScale / nodeOldScale);
    nodePrecision = glm::max(nodePrecision, kTolerance * nodeNewScale);
    if (!glm::isfinite(nodePrecision)) nodePrecision = -1;
    precision = glm::max(precision, nodePrecision);

    numVert += node->pImpl_->NumVert();
    numEdge += node->pImpl_->NumEdge();
    numTri += node->pImpl_->NumTri();
  }

  Manifold::Impl combined;
  combined.precision_ = precision;
  combined.vertPos_.resize(numVert);
  combined.halfedge_.resize(2 * numEdge);
  combined.faceNormal_.resize(numTri);
  combined.halfedgeTangent_.resize(2 * numEdge);
  combined.meshRelation_.triRef.resize(numTri);
  auto policy = autoPolicy(numTri);

  int nextVert = 0;
  int nextEdge = 0;
  int nextTri = 0;
  int i = 0;
  for (auto &node : nodes) {
    if (node->transform_ == glm::mat4x3(1.0f)) {
      copy(policy, node->pImpl_->vertPos_.begin(), node->pImpl_->vertPos_.end(),
           combined.vertPos_.begin() + nextVert);
      copy(policy, node->pImpl_->faceNormal_.begin(),
           node->pImpl_->faceNormal_.end(),
           combined.faceNormal_.begin() + nextTri);
    } else {
      // no need to apply the transform to the node, just copy the vertices and
      // face normals and apply transform on the fly
      auto vertPosBegin = thrust::make_transform_iterator(
          node->pImpl_->vertPos_.begin(), Transform4x3({node->transform_}));
      glm::mat3 normalTransform =
          glm::inverse(glm::transpose(glm::mat3(node->transform_)));
      auto faceNormalBegin =
          thrust::make_transform_iterator(node->pImpl_->faceNormal_.begin(),
                                          TransformNormals({normalTransform}));
      copy_n(policy, vertPosBegin, node->pImpl_->vertPos_.size(),
             combined.vertPos_.begin() + nextVert);
      copy_n(policy, faceNormalBegin, node->pImpl_->faceNormal_.size(),
             combined.faceNormal_.begin() + nextTri);
    }
    copy(policy, node->pImpl_->halfedgeTangent_.begin(),
         node->pImpl_->halfedgeTangent_.end(),
         combined.halfedgeTangent_.begin() + nextEdge);
    transform(policy, node->pImpl_->halfedge_.begin(),
              node->pImpl_->halfedge_.end(),
              combined.halfedge_.begin() + nextEdge,
              UpdateHalfedge({nextVert, nextEdge, nextTri}));
    // Since the nodes may be copies containing the same meshIDs, it is
    // important to add an offset so that each node instance gets
    // unique meshIDs.
    const int offset = i++ * Manifold::Impl::meshIDCounter_;
    transform(policy, node->pImpl_->meshRelation_.triRef.begin(),
              node->pImpl_->meshRelation_.triRef.end(),
              combined.meshRelation_.triRef.begin() + nextTri,
              UpdateMeshIDs({offset}));

    nextVert += node->pImpl_->NumVert();
    nextEdge += 2 * node->pImpl_->NumEdge();
    nextTri += node->pImpl_->NumTri();
  }

  // required to remove parts that are smaller than the precision
  combined.SimplifyTopology();
  combined.Finish();
  combined.IncrementMeshIDs();
  return combined;
}

CsgOpNode::CsgOpNode() {}

CsgOpNode::CsgOpNode(const std::vector<std::shared_ptr<CsgNode>> &children,
                     Manifold::OpType op)
    : impl_(std::make_shared<Impl>()) {
  impl_->children_ = children;
  SetOp(op);
  // opportunistically flatten the tree without costly evaluation
  GetChildren(false);
}

CsgOpNode::CsgOpNode(std::vector<std::shared_ptr<CsgNode>> &&children,
                     Manifold::OpType op)
    : impl_(std::make_shared<Impl>()) {
  impl_->children_ = children;
  SetOp(op);
  // opportunistically flatten the tree without costly evaluation
  GetChildren(false);
}

std::shared_ptr<CsgNode> CsgOpNode::Transform(const glm::mat4x3 &m) const {
  auto node = std::make_shared<CsgOpNode>();
  node->impl_ = impl_;
  node->transform_ = m * glm::mat4(transform_);
  return node;
}

std::shared_ptr<CsgLeafNode> CsgOpNode::ToLeafNode() const {
  if (cache_ != nullptr) return cache_;
  if (impl_->children_.empty()) return nullptr;
  // turn the children into leaf nodes
  GetChildren();
  auto &children_ = impl_->children_;
  if (children_.size() > 1) {
    switch (impl_->op_) {
      case CsgNodeType::UNION:
        BatchUnion();
        break;
      case CsgNodeType::INTERSECTION: {
        std::vector<std::shared_ptr<const Manifold::Impl>> impls;
        for (auto &child : children_) {
          impls.push_back(
              std::dynamic_pointer_cast<CsgLeafNode>(child)->GetImpl());
        }
        BatchBoolean(Manifold::OpType::INTERSECT, impls);
        children_.clear();
        children_.push_back(std::make_shared<CsgLeafNode>(impls.front()));
        break;
      };
      case CsgNodeType::DIFFERENCE: {
        // take the lhs out and treat the remaining nodes as the rhs, perform
        // union optimization for them
        auto lhs = std::dynamic_pointer_cast<CsgLeafNode>(children_.front());
        children_.erase(children_.begin());
        BatchUnion();
        auto rhs = std::dynamic_pointer_cast<CsgLeafNode>(children_.front());
        children_.clear();
        Boolean3 boolean(*lhs->GetImpl(), *rhs->GetImpl(),
                         Manifold::OpType::SUBTRACT);
        children_.push_back(
            std::make_shared<CsgLeafNode>(std::make_shared<Manifold::Impl>(
                boolean.Result(Manifold::OpType::SUBTRACT))));
      };
      case CsgNodeType::LEAF:
        // unreachable
        break;
    }
  }
  // children_ must contain only one CsgLeafNode now, and its Transform will
  // give CsgLeafNode as well
  cache_ = std::dynamic_pointer_cast<CsgLeafNode>(
      children_.front()->Transform(transform_));
  return cache_;
}

/**
 * Efficient boolean operation on a set of nodes utilizing commutativity of the
 * operation. Only supports union and intersection.
 */
void CsgOpNode::BatchBoolean(
    Manifold::OpType operation,
    std::vector<std::shared_ptr<const Manifold::Impl>> &results) {
  ASSERT(operation != Manifold::OpType::SUBTRACT, logicErr,
         "BatchBoolean doesn't support Difference.");
  auto cmpFn = [](std::shared_ptr<const Manifold::Impl> a,
                  std::shared_ptr<const Manifold::Impl> b) {
    // invert the order because we want a min heap
    return a->NumVert() > b->NumVert();
  };

  // apply boolean operations starting from smaller meshes
  // the assumption is that boolean operations on smaller meshes is faster,
  // due to less data being copied and processed
  std::make_heap(results.begin(), results.end(), cmpFn);
  while (results.size() > 1) {
    std::pop_heap(results.begin(), results.end(), cmpFn);
    auto a = std::move(results.back());
    results.pop_back();
    std::pop_heap(results.begin(), results.end(), cmpFn);
    auto b = std::move(results.back());
    results.pop_back();
    // boolean operation
    Boolean3 boolean(*a, *b, operation);
    results.push_back(
        std::make_shared<const Manifold::Impl>(boolean.Result(operation)));
    std::push_heap(results.begin(), results.end(), cmpFn);
  }
}

/**
 * Efficient union operation on a set of nodes by doing Compose as much as
 * possible.
 * Note: Due to some unknown issues with `Compose`, we are now doing
 * `BatchBoolean` instead of using `Compose` for non-intersecting manifolds.
 */
void CsgOpNode::BatchUnion() const {
  // INVARIANT: children_ is a vector of leaf nodes
  // this kMaxUnionSize is a heuristic to avoid the pairwise disjoint check
  // with O(n^2) complexity to take too long.
  // If the number of children exceeded this limit, we will operate on chunks
  // with size kMaxUnionSize.
  constexpr int kMaxUnionSize = 1000;
  auto &children_ = impl_->children_;
  while (children_.size() > 1) {
    const int start = (children_.size() > kMaxUnionSize)
                          ? (children_.size() - kMaxUnionSize)
                          : 0;
    VecDH<Box> boxes;
    boxes.reserve(children_.size() - start);
    for (int i = start; i < children_.size(); i++) {
      boxes.push_back(std::dynamic_pointer_cast<CsgLeafNode>(children_[i])
                          ->GetImpl()
                          ->bBox_);
    }
    const Box *boxesD = boxes.cptrD();
    // partition the children into a set of disjoint sets
    // each set contains a set of children that are pairwise disjoint
    std::vector<VecDH<size_t>> disjointSets;
    for (size_t i = 0; i < boxes.size(); i++) {
      auto lambda = [boxesD, i](const VecDH<size_t> &set) {
        return find_if<decltype(set.end())>(
                   autoPolicy(set.size()), set.begin(), set.end(),
                   CheckOverlap({boxesD, i})) == set.end();
      };
      auto it = std::find_if(disjointSets.begin(), disjointSets.end(), lambda);
      if (it == disjointSets.end()) {
        disjointSets.push_back(std::vector<size_t>{i});
      } else {
        it->push_back(i);
      }
    }
    // compose each set of disjoint children
    std::vector<std::shared_ptr<const Manifold::Impl>> impls;
    for (const auto &set : disjointSets) {
      if (set.size() == 1) {
        impls.push_back(
            std::dynamic_pointer_cast<CsgLeafNode>(children_[start + set[0]])
                ->GetImpl());
      } else {
        std::vector<std::shared_ptr<CsgLeafNode>> tmp;
        for (size_t j : set) {
          tmp.push_back(
              std::dynamic_pointer_cast<CsgLeafNode>(children_[start + j]));
        }
        impls.push_back(
            std::make_shared<const Manifold::Impl>(CsgLeafNode::Compose(tmp)));
      }
    }
    BatchBoolean(Manifold::OpType::ADD, impls);
    children_.erase(children_.begin() + start, children_.end());
    children_.push_back(std::make_shared<CsgLeafNode>(impls.back()));
    // move it to the front as we process from the back, and the newly added
    // child should be quite complicated
    std::swap(children_.front(), children_.back());
  }
}

/**
 * Flatten the children to a list of leaf nodes and return them.
 * If finalize is true, the list will be guaranteed to be a list of leaf nodes
 * (i.e. no ops). Otherwise, the list may contain ops.
 * Note that this function will not apply the transform to children, as they may
 * be shared with other nodes.
 */
std::vector<std::shared_ptr<CsgNode>> &CsgOpNode::GetChildren(
    bool finalize) const {
  auto &children_ = impl_->children_;
  if (children_.empty() || (impl_->simplified_ && !finalize) ||
      impl_->flattened_)
    return children_;
  impl_->simplified_ = true;
  impl_->flattened_ = finalize;
  std::vector<std::shared_ptr<CsgNode>> newChildren;

  CsgNodeType op = impl_->op_;
  for (auto &child : children_) {
    if (child->GetNodeType() == op && child.use_count() == 1 &&
        std::dynamic_pointer_cast<CsgOpNode>(child)->impl_.use_count() == 1) {
      auto grandchildren =
          std::dynamic_pointer_cast<CsgOpNode>(child)->GetChildren(finalize);
      int start = children_.size();
      for (auto &grandchild : grandchildren) {
        newChildren.push_back(grandchild->Transform(child->GetTransform()));
      }
    } else {
      if (!finalize || child->GetNodeType() == CsgNodeType::LEAF) {
        newChildren.push_back(child);
      } else {
        newChildren.push_back(child->ToLeafNode());
      }
    }
    // special handling for difference: we treat it as first - (second + third +
    // ...) so op = UNION after the first node
    if (op == CsgNodeType::DIFFERENCE) op = CsgNodeType::UNION;
  }
  children_ = newChildren;
  return children_;
}

void CsgOpNode::SetOp(Manifold::OpType op) {
  switch (op) {
    case Manifold::OpType::ADD:
      impl_->op_ = CsgNodeType::UNION;
      break;
    case Manifold::OpType::SUBTRACT:
      impl_->op_ = CsgNodeType::DIFFERENCE;
      break;
    case Manifold::OpType::INTERSECT:
      impl_->op_ = CsgNodeType::INTERSECTION;
      break;
  }
}

glm::mat4x3 CsgOpNode::GetTransform() const { return transform_; }

}  // namespace manifold
