// Copyright 2026 The Manifold Authors.
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

#include "execution_impl.h"

#include "impl.h"
#include "manifold/manifold.h"
#include "manifold/mesh.h"

namespace {
using namespace manifold;

// Reset progress counters at the start of an eager static-factory call.
// Done counters reset before totals so an observer never reads
// Progress > 1.0 (it may briefly read 0). Cancel is intentionally
// preserved - sticky across ops per the ExecutionContext contract.
void ResetForStaticFactory(ExecutionContext::Impl* ctx, int totalPhases) {
  ctx->doneBooleans.store(0, std::memory_order_relaxed);
  ctx->donePhases.store(0, std::memory_order_relaxed);
  ctx->totalBooleans.store(0, std::memory_order_relaxed);
  ctx->totalPhases.store(totalPhases, std::memory_order_relaxed);
}

}  // namespace
namespace manifold {

ExecutionContext::ExecutionContext() : impl_(std::make_shared<Impl>()) {}
ExecutionContext::~ExecutionContext() = default;
ExecutionContext::ExecutionContext(const ExecutionContext&) = default;
ExecutionContext::ExecutionContext(ExecutionContext&&) noexcept = default;
ExecutionContext& ExecutionContext::operator=(const ExecutionContext&) =
    default;
ExecutionContext& ExecutionContext::operator=(ExecutionContext&&) noexcept =
    default;

void ExecutionContext::Cancel() {
  impl_->cancel.store(true, std::memory_order_relaxed);
}

bool ExecutionContext::Cancelled() const { return IsCancelled(impl_.get()); }

double ExecutionContext::Progress() const {
  const int total = impl_->totalPhases.load(std::memory_order_relaxed);
  // Zero-work case: no phases scheduled (single-leaf manifold, or
  // pre-evaluation). Treat as complete -- "no work to do" maps to 100%
  // more naturally than 0%, and matches the user expectation that a
  // returned `Status() == NoError` means the operation is done.
  if (total == 0) return 1.0;
  return double(impl_->donePhases.load(std::memory_order_relaxed)) / total;
}

Manifold ExecutionContext::FromMeshGL(const MeshGL& mesh) {
  ResetForStaticFactory(impl_.get(), kPhasesPerFromMesh);
  return Manifold::FromImpl(
      std::make_shared<Manifold::Impl>(mesh, impl_.get()));
}

Manifold ExecutionContext::FromMeshGL(const MeshGL64& mesh) {
  ResetForStaticFactory(impl_.get(), kPhasesPerFromMesh);
  return Manifold::FromImpl(
      std::make_shared<Manifold::Impl>(mesh, impl_.get()));
}

Manifold ExecutionContext::Smooth(
    const MeshGL& mesh, const std::vector<Smoothness>& sharpenedEdges) {
  ResetForStaticFactory(impl_.get(), kPhasesPerSmooth);
  return Manifold::FromImpl(MakeSmoothImpl(mesh, sharpenedEdges, impl_.get()));
}

Manifold ExecutionContext::Smooth(
    const MeshGL64& mesh, const std::vector<Smoothness>& sharpenedEdges) {
  ResetForStaticFactory(impl_.get(), kPhasesPerSmooth);
  return Manifold::FromImpl(MakeSmoothImpl(mesh, sharpenedEdges, impl_.get()));
}

Manifold ExecutionContext::LevelSet(std::function<double(vec3)> sdf, Box bounds,
                                    double edgeLength, double level,
                                    double tolerance, bool canParallel) {
  ResetForStaticFactory(impl_.get(), kPhasesPerLevelSet);
  auto pImpl = std::make_shared<Manifold::Impl>();
  pImpl->CreateLevelSet(sdf, bounds, edgeLength, level, tolerance, canParallel,
                        impl_.get());
  return Manifold::FromImpl(pImpl);
}

}  // namespace manifold
