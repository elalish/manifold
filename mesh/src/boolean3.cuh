#include "mesh.cuh"
#include "polygon.h"

namespace manifold {

class Boolean3 {
 public:
  Boolean3(const Mesh::Impl& inP, const Mesh::Impl& inQ);
  Mesh::Impl Result(Mesh::OpType op) const;

 private:
  const Mesh::Impl &inP_, &inQ_;
  SparseIndices p1q2_, p2q1_, p2q2_;
  VecDH<int> dir12_, dir21_, w03_, w30_;
  VecDH<glm::vec3> v12_, v21_;
};
}  // namespace manifold