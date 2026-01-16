// Copyright 2024 The Manifold Authors.
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

#include "impl.h"
#include "parallel.h"

namespace manifold {

/**
 * Compute the minkowski sum of two manifolds.
 *
 * @param other The other Impl to minkowski sum/diff with this one.
 * @param inset Whether it should subtract (erode) rather than add (dilate).
 */
Manifold Manifold::Impl::Minkowski(const Impl& other, bool inset) const {
  const Impl* aImpl = this;
  const Impl* bImpl = &other;

  bool aConvex = aImpl->IsConvex();
  bool bConvex = bImpl->IsConvex();

  constexpr size_t BATCH_SIZE = 1000;

  // If the convex manifold was supplied first, swap them!
  if (aConvex && !bConvex) {
    std::swap(aImpl, bImpl);
    std::swap(aConvex, bConvex);
  }

  // Early-exit if either input is empty
  if (bImpl->IsEmpty()) {
    std::shared_ptr<Impl> result = std::make_shared<Impl>(*aImpl);
    return Manifold(result);
  }
  if (aImpl->IsEmpty()) {
    std::shared_ptr<Impl> result = std::make_shared<Impl>(*bImpl);
    return Manifold(result);
  }

  std::shared_ptr<Impl> aImplCopy = std::make_shared<Impl>(*aImpl);
  Manifold a(aImplCopy);
  std::vector<Manifold> composedHulls({a});

  // Convex-Convex Minkowski: Very Fast
  if (!inset && aConvex && bConvex) {
    const VecView<vec3> verts = bImpl->vertPos_;
    std::vector<vec3> simpleHull;
    simpleHull.reserve(verts.size() * aImpl->vertPos_.size());
    for (const vec3& vertex : aImpl->vertPos_) {
      auto t = [vertex](vec3 v) { return v + vertex; };
      simpleHull.insert(simpleHull.end(), TransformIterator(verts.begin(), t),
                        TransformIterator(verts.end(), t));
    }
    composedHulls.push_back(Manifold::Hull(simpleHull));
    // Convex - Non-Convex Minkowski: Slower
  } else if ((inset || !aConvex) && bConvex) {
    const size_t numTri = aImpl->NumTri();
    const VecView<vec3> verts = bImpl->vertPos_;

    // do it in batches of 1000 meshes
    for (size_t offset = 0; offset < numTri; offset += BATCH_SIZE) {
      size_t numIter = std::min(numTri - offset, BATCH_SIZE);
      std::vector<Manifold> newHulls(numIter);
      for_each_n(
          autoPolicy(numIter, 100), countAt(0), numIter,
          [&newHulls, &aImpl, &verts, offset](const int iter) {
            std::vector<vec3> simpleHull;
            for (int i : {0, 1, 2}) {
              const auto vertex =
                  aImpl->vertPos_[aImpl->halfedge_[((offset + iter) * 3) + i]
                                      .startVert];
              auto t = [vertex](vec3 v) { return v + vertex; };
              simpleHull.insert(simpleHull.end(),
                                TransformIterator(verts.begin(), t),
                                TransformIterator(verts.end(), t));
            }
            newHulls[iter] = Manifold::Hull(simpleHull);
          });
      composedHulls.push_back(Manifold::BatchBoolean(newHulls, OpType::Add));
    }
    // Non-Convex - Non-Convex Minkowski: Very Slow
  } else if (!aConvex && !bConvex) {
    const size_t numTriA = aImpl->NumTri();
    const size_t numTriB = bImpl->NumTri();
    const size_t totalPairs = numTriA * numTriB;

    // Process face pairs in batches with parallelization
    for (size_t offset = 0; offset < totalPairs; offset += BATCH_SIZE) {
      size_t numIter = std::min(totalPairs - offset, BATCH_SIZE);
      std::vector<Manifold> newHulls(numIter);
      std::vector<bool> validHull(numIter, false);

      for_each_n(
          autoPolicy(numIter, 100), countAt(0), numIter, [&](const int iter) {
            size_t pairIdx = offset + iter;
            size_t aFace = pairIdx / numTriB;
            size_t bFace = pairIdx % numTriB;

            // Use tolerance-based coplanarity check instead of exact equality
            // to handle floating-point precision issues from scaling
            constexpr double kCoplanarTol = 1e-15;
            vec3 nA = aImpl->faceNormal_[aFace];
            vec3 nB = bImpl->faceNormal_[bFace];
            double dotSame = linalg::dot(nA, nB);
            double dotOpp = linalg::dot(nA, -nB);
            const bool coplanar = (std::abs(dotSame - 1.0) < kCoplanarTol) ||
                                  (std::abs(dotOpp - 1.0) < kCoplanarTol);
            if (coplanar) return;  // Skip Coplanar Triangles

            vec3 a1 =
                aImpl->vertPos_[aImpl->halfedge_[(aFace * 3) + 0].startVert];
            vec3 a2 =
                aImpl->vertPos_[aImpl->halfedge_[(aFace * 3) + 1].startVert];
            vec3 a3 =
                aImpl->vertPos_[aImpl->halfedge_[(aFace * 3) + 2].startVert];
            vec3 b1 =
                bImpl->vertPos_[bImpl->halfedge_[(bFace * 3) + 0].startVert];
            vec3 b2 =
                bImpl->vertPos_[bImpl->halfedge_[(bFace * 3) + 1].startVert];
            vec3 b3 =
                bImpl->vertPos_[bImpl->halfedge_[(bFace * 3) + 2].startVert];
            newHulls[iter] =
                Manifold::Hull({a1 + b1, a1 + b2, a1 + b3, a2 + b1, a2 + b2,
                                a2 + b3, a3 + b1, a3 + b2, a3 + b3});
            validHull[iter] = true;
          });

      // Collect valid (non-coplanar) hulls
      std::vector<Manifold> batchHulls;
      for (size_t i = 0; i < numIter; ++i) {
        if (validHull[i]) {
          batchHulls.push_back(std::move(newHulls[i]));
        }
      }
      if (!batchHulls.empty()) {
        composedHulls.push_back(
            Manifold::BatchBoolean(batchHulls, OpType::Add));
      }
    }
  }
  return Manifold::BatchBoolean(composedHulls, inset
                                                   ? manifold::OpType::Subtract
                                                   : manifold::OpType::Add)
      .AsOriginal();
}

}  // namespace manifold
