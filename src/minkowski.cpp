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

namespace {

// Convex-convex Minkowski sum using Impl vertex data directly.
Manifold ConvexConvexMinkowski(const Manifold::Impl* aImpl,
                               const Manifold::Impl* bImpl) {
  const VecView<vec3> bVerts = bImpl->vertPos_;
  std::vector<vec3> simpleHull;
  simpleHull.reserve(bVerts.size() * aImpl->vertPos_.size());
  for (const vec3& vertex : aImpl->vertPos_) {
    auto t = [vertex](vec3 v) { return v + vertex; };
    simpleHull.insert(simpleHull.end(), TransformIterator(bVerts.begin(), t),
                      TransformIterator(bVerts.end(), t));
  }
  return Manifold::Hull(simpleHull);
}

// Per-triangle Minkowski for non-convex × convex (original algorithm).
void MinkowskiTriangles(const Manifold::Impl* aImpl,
                        const Manifold::Impl* bImpl,
                        std::vector<Manifold>& composedHulls) {
  constexpr size_t BATCH_SIZE = 1000;
  const size_t numTri = aImpl->NumTri();
  const VecView<vec3> verts = bImpl->vertPos_;

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
}

// Per-triangle pair Minkowski for non-convex × non-convex (original algorithm).
void MinkowskiTrianglePairs(const Manifold::Impl* aImpl,
                            const Manifold::Impl* bImpl,
                            std::vector<Manifold>& composedHulls) {
  const size_t numTriA = aImpl->NumTri();
  const size_t numTriB = bImpl->NumTri();
  constexpr size_t REDUCE_THRESHOLD = 200;

  std::vector<Manifold> accumulated;
  accumulated.reserve(std::min(numTriA, REDUCE_THRESHOLD));

  for (size_t aFace = 0; aFace < numTriA; ++aFace) {
    vec3 a1 = aImpl->vertPos_[aImpl->halfedge_[(aFace * 3) + 0].startVert];
    vec3 a2 = aImpl->vertPos_[aImpl->halfedge_[(aFace * 3) + 1].startVert];
    vec3 a3 = aImpl->vertPos_[aImpl->halfedge_[(aFace * 3) + 2].startVert];
    vec3 nA = aImpl->faceNormal_[aFace];

    std::vector<Manifold> faceHulls(numTriB);
    for_each_n(
        autoPolicy(numTriB, 100), countAt(0), numTriB, [&](const int bFace) {
          constexpr double kCoplanarTol = 1e-12;
          vec3 nB = bImpl->faceNormal_[bFace];
          double dotSame = linalg::dot(nA, nB);
          double dotOpp = linalg::dot(nA, -nB);
          const bool coplanar = (std::abs(dotSame - 1.0) < kCoplanarTol) ||
                                (std::abs(dotOpp - 1.0) < kCoplanarTol);
          if (coplanar) return;

          vec3 b1 =
              bImpl->vertPos_[bImpl->halfedge_[(bFace * 3) + 0].startVert];
          vec3 b2 =
              bImpl->vertPos_[bImpl->halfedge_[(bFace * 3) + 1].startVert];
          vec3 b3 =
              bImpl->vertPos_[bImpl->halfedge_[(bFace * 3) + 2].startVert];
          faceHulls[bFace] =
              Manifold::Hull({a1 + b1, a1 + b2, a1 + b3, a2 + b1, a2 + b2,
                              a2 + b3, a3 + b1, a3 + b2, a3 + b3});
        });

    std::vector<Manifold> validFaceHulls;
    for (auto& hull : faceHulls) {
      if (!hull.IsEmpty()) validFaceHulls.push_back(std::move(hull));
    }
    if (!validFaceHulls.empty()) {
      accumulated.push_back(
          Manifold::BatchBoolean(validFaceHulls, OpType::Add));
    }
    if (accumulated.size() >= REDUCE_THRESHOLD) {
      Manifold reduced = Manifold::BatchBoolean(accumulated, OpType::Add);
      accumulated.clear();
      accumulated.push_back(std::move(reduced));
    }
  }
  if (!accumulated.empty()) {
    composedHulls.push_back(Manifold::BatchBoolean(accumulated, OpType::Add));
  }
}

}  // namespace

/**
 * Compute the minkowski sum of two manifolds.
 *
 * When either input is non-convex and decompose=true, the shape is first
 * decomposed into convex pieces using ConvexDecomposition, then each convex
 * piece uses the fast vertex-addition path. This is much faster than the
 * per-triangle approach for complex shapes but only works for Minkowski sum
 * (not difference/erosion, which requires surface-level triangle processing).
 *
 * @param other The other Impl to minkowski sum/diff with this one.
 * @param inset Whether it should subtract (erode) rather than add (dilate).
 * @param decompose If true and not insetting, use convex decomposition for
 *                  non-convex inputs (faster for complex shapes).
 */
Manifold Manifold::Impl::Minkowski(const Impl& other, bool inset,
                                   bool decompose) const {
  const Impl* aImpl = this;
  const Impl* bImpl = &other;

  bool aConvex = aImpl->IsConvex();
  bool bConvex = bImpl->IsConvex();

  // If the convex manifold was supplied first, swap them!
  if (aConvex && !bConvex) {
    std::swap(aImpl, bImpl);
    std::swap(aConvex, bConvex);
  }

  // Early-exit if either input is empty
  if (bImpl->IsEmpty()) {
    return Manifold(std::make_shared<Impl>(*aImpl));
  }
  if (aImpl->IsEmpty()) {
    return Manifold(std::make_shared<Impl>(*bImpl));
  }

  Manifold a(std::make_shared<Impl>(*aImpl));
  std::vector<Manifold> composedHulls;
  composedHulls.push_back(a);

  // Convex × Convex: vertex-addition hull (very fast)
  if (!inset && aConvex && bConvex) {
    composedHulls.push_back(ConvexConvexMinkowski(aImpl, bImpl));

    // Decomposition-based path: only for sum (not erosion).
    // Uses hullSnap=false to preserve exact topology, then dispatches each
    // piece to convex-convex (fast) or per-triangle (exact) based on IsConvex.
  } else if (decompose && !inset) {
    // Decompose non-convex shape(s) without hull-snapping for exact tiling
    auto aPieces = aImpl->ConvexDecomposition(3);

    if (bConvex) {
      // Non-Convex × Convex: per-piece Minkowski
      std::vector<Manifold> pieceResults(aPieces.size());
      for_each_n(autoPolicy(aPieces.size(), 4), countAt(0_uz), aPieces.size(),
                 [&](size_t i) {
                   MeshGL64 mesh = aPieces[i].GetMeshGL64();
                   size_t nv = mesh.vertProperties.size() / mesh.numProp;
                   std::vector<vec3> combined;
                   combined.reserve(nv * bImpl->vertPos_.size());
                   for (size_t vi = 0; vi < nv; vi++) {
                     vec3 vertex(mesh.vertProperties[vi * mesh.numProp],
                                 mesh.vertProperties[vi * mesh.numProp + 1],
                                 mesh.vertProperties[vi * mesh.numProp + 2]);
                     auto t = [vertex](vec3 v) { return v + vertex; };
                     combined.insert(
                         combined.end(),
                         TransformIterator(bImpl->vertPos_.begin(), t),
                         TransformIterator(bImpl->vertPos_.end(), t));
                   }
                   pieceResults[i] = Manifold::Hull(combined);
                 });
      composedHulls.push_back(
          Manifold::BatchBoolean(pieceResults, OpType::Add));

    } else {
      // Non-Convex × Non-Convex: decompose both, pairwise vertex-addition
      auto bPieces = bImpl->ConvexDecomposition(3);
      std::vector<Manifold> pairResults(aPieces.size() * bPieces.size());
      for_each_n(autoPolicy(pairResults.size(), 4), countAt(0_uz),
                 pairResults.size(), [&](size_t idx) {
                   size_t ai = idx / bPieces.size();
                   size_t bi = idx % bPieces.size();
                   MeshGL64 meshA = aPieces[ai].GetMeshGL64();
                   MeshGL64 meshB = bPieces[bi].GetMeshGL64();
                   size_t nvA = meshA.vertProperties.size() / meshA.numProp;
                   size_t nvB = meshB.vertProperties.size() / meshB.numProp;
                   std::vector<vec3> combined;
                   combined.reserve(nvA * nvB);
                   for (size_t va = 0; va < nvA; va++) {
                     vec3 vertA(meshA.vertProperties[va * meshA.numProp],
                                meshA.vertProperties[va * meshA.numProp + 1],
                                meshA.vertProperties[va * meshA.numProp + 2]);
                     for (size_t vb = 0; vb < nvB; vb++) {
                       vec3 vertB(meshB.vertProperties[vb * meshB.numProp],
                                  meshB.vertProperties[vb * meshB.numProp + 1],
                                  meshB.vertProperties[vb * meshB.numProp + 2]);
                       combined.push_back(vertA + vertB);
                     }
                   }
                   pairResults[idx] = Manifold::Hull(combined);
                 });
      composedHulls.push_back(Manifold::BatchBoolean(pairResults, OpType::Add));
    }

    // Original per-triangle path (used for erosion/inset or when
    // decompose=false)
  } else if ((inset || !aConvex) && bConvex) {
    MinkowskiTriangles(aImpl, bImpl, composedHulls);
  } else if (!aConvex && !bConvex) {
    MinkowskiTrianglePairs(aImpl, bImpl, composedHulls);
  }

  return Manifold::BatchBoolean(composedHulls, inset
                                                   ? manifold::OpType::Subtract
                                                   : manifold::OpType::Add)
      .AsOriginal();
}

}  // namespace manifold
