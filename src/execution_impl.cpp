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

namespace manifold {

ExecutionContext::ExecutionContext() : impl_(std::make_shared<Impl>()) {}
ExecutionContext::ExecutionContext(std::shared_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
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

}  // namespace manifold
