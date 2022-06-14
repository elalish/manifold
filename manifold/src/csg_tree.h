#pragma once
#include "manifold.h"

namespace manifold {

enum class CsgNodeType { UNION, INTERSECTION, DIFFERENCE, LEAF };

class CsgLeafNode;

class CsgNode {
 public:
  virtual std::shared_ptr<CsgLeafNode> ToLeafNode() const = 0;
  virtual std::shared_ptr<CsgNode> Transform(const glm::mat4x3 &m) const = 0;
  virtual CsgNodeType GetNodeType() const = 0;
  virtual glm::mat4x3 GetTransform() const = 0;

  std::shared_ptr<CsgNode> Translate(const glm::vec3 &t) const;
  std::shared_ptr<CsgNode> Scale(const glm::vec3 &s) const;
  std::shared_ptr<CsgNode> Rotate(float xDegrees = 0, float yDegrees = 0,
                                  float zDegrees = 0) const;
};

class CsgLeafNode final : public CsgNode {
 public:
  CsgLeafNode();
  CsgLeafNode(std::shared_ptr<const Manifold::Impl> pImpl_);
  CsgLeafNode(std::shared_ptr<const Manifold::Impl> pImpl_,
              glm::mat4x3 transform_);

  std::shared_ptr<const Manifold::Impl> GetImpl() const;

  Box GetBoundingBox() const;

  std::shared_ptr<CsgLeafNode> ToLeafNode() const override;

  std::shared_ptr<CsgNode> Transform(const glm::mat4x3 &m) const override;

  CsgNodeType GetNodeType() const override;

  glm::mat4x3 GetTransform() const override;

  static Manifold::Impl Compose(
      const std::vector<std::shared_ptr<CsgLeafNode>> &nodes);

 private:
  mutable std::shared_ptr<const Manifold::Impl> pImpl_;
  mutable glm::mat4x3 transform_ = glm::mat4x3(1.0f);
};

class CsgOpNode final : public CsgNode {
 public:
  CsgOpNode();

  CsgOpNode(const std::vector<std::shared_ptr<CsgNode>> &children,
            Manifold::OpType op);

  CsgOpNode(std::vector<std::shared_ptr<CsgNode>> &&children,
            Manifold::OpType op);

  std::shared_ptr<CsgNode> Transform(const glm::mat4x3 &m) const override;

  std::shared_ptr<CsgLeafNode> ToLeafNode() const override;

  CsgNodeType GetNodeType() const override { return op_; }

  glm::mat4x3 GetTransform() const override;

 private:
  CsgNodeType op_;
  glm::mat4x3 transform_ = glm::mat4x3(1.0f);
  // the following fields are for lazy evaluation, so they are mutable
  mutable std::vector<std::shared_ptr<CsgNode>> children_;
  mutable std::shared_ptr<CsgLeafNode> cache_ = nullptr;
  mutable bool simplified_ = false;
  mutable bool flattened_ = false;

  void SetOp(Manifold::OpType);

  static void BatchBoolean(
      Manifold::OpType operation,
      std::vector<std::shared_ptr<const Manifold::Impl>> &results);

  void BatchUnion() const;

  std::vector<std::shared_ptr<CsgNode>> &GetChildren(
      bool finalize = true) const;
};

}  // namespace manifold
