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
//
// Prints per-phase wall-clock timings on representative workloads for the
// eager ops that carry phase instrumentation: Boolean3 (its own Timers) and
// the ADVANCE_PHASE_OR_RETURN ops driven through an ExecutionContext -
// LevelSet, FromMesh ingest, and Smooth. Requires MANIFOLD_TIMING=ON (or
// MANIFOLD_DEBUG=ON). Used for cancel-latency analysis and identifying dominant
// phases.

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "manifold/manifold.h"
#include "samples.h"

using namespace manifold;

namespace {
// Runs one boolean op of each kind (Add, Subtract, Intersect) against the
// same input pair. Each is a single boolean so Timer output is clean.
void BenchAll(const std::string& workload, const Manifold& a,
              const Manifold& b) {
  struct Op {
    const char* name;
    Manifold (*apply)(const Manifold&, const Manifold&);
  };
  const Op ops[] = {
      {"add", [](const Manifold& x, const Manifold& y) { return x + y; }},
      {"subtract", [](const Manifold& x, const Manifold& y) { return x - y; }},
      {"intersect", [](const Manifold& x, const Manifold& y) { return x ^ y; }},
  };
  for (const auto& op : ops) {
    std::cout << "=== " << workload << ": " << op.name
              << ", nTri(a) = " << a.NumTri() << " ===" << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    Manifold result = op.apply(a, b);
    result.NumTri();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "total = " << std::chrono::duration<double>(t1 - t0).count()
              << " sec\n"
              << std::endl;
  }
}

// Runs one LevelSet through an ExecutionContext. The wall-clock total is
// printed here; ADVANCE_PHASE_OR_RETURN prints the per-phase breakdown under
// verbose >= 2 (it only records phases when a ctx is attached). One op per
// call keeps that breakdown readable.
void BenchLevelSet(const std::string& workload,
                   const std::function<double(vec3)>& sdf, Box bounds,
                   double edgeLength, double tolerance, bool canParallel) {
  std::cout << "=== " << workload << ": edge=" << edgeLength
            << ", tol=" << tolerance << ", " << (canParallel ? "Par" : "Seq")
            << " ===" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  ExecutionContext ctx;
  Manifold result =
      ctx.LevelSet(sdf, bounds, edgeLength, 0, tolerance, canParallel);
  result.NumTri();
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "total = " << std::chrono::duration<double>(t1 - t0).count()
            << " sec\n"
            << std::endl;
}

// Re-ingests a MeshGL through ExecutionContext::FromMeshGL (kPhasesPerFromMesh
// = 7: CreateHalfedges, CleanupTopology, DedupePropVerts,
// SetNormalsAndCoplanar, RemoveDegenerates, RemoveUnreferencedVerts,
// SortGeometry).
void BenchFromMesh(const std::string& workload, const MeshGL& gl) {
  std::cout << "=== FromMesh " << workload << ": nTri = " << gl.NumTri()
            << " ===" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  ExecutionContext ctx;
  Manifold result = ctx.FromMeshGL(gl);
  result.NumTri();
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "total = " << std::chrono::duration<double>(t1 - t0).count()
            << " sec\n"
            << std::endl;
}

// Smooths a control-cage MeshGL through ExecutionContext::Smooth
// (kPhasesPerSmooth = 14: the 7 FromMesh ingest phases plus 7 tangent/smoothing
// phases), then refines to materialize the tessellation. The phase lines come
// from the Smooth call; Refine is just to make it a realistic workload.
void BenchSmooth(const std::string& workload, const MeshGL& gl, int refine) {
  std::cout << "=== Smooth " << workload << ": nTri = " << gl.NumTri()
            << ", refine = " << refine << " ===" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  ExecutionContext ctx;
  Manifold result = ctx.Smooth(gl, {}).Refine(refine);
  result.NumTri();
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "total = " << std::chrono::duration<double>(t1 - t0).count()
            << " sec\n"
            << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
  ManifoldParams().verbose = 2;

  // Sphere at escalating scales — uniform geometry, single boolean per op
  // so Timer output is readable.
  for (int i = 0; i < 8; ++i) {
    const int segments = (8 << i) * 4;
    Manifold sphere = Manifold::Sphere(1, segments);
    Manifold sphere2 = sphere.Translate(vec3(0.5));
    BenchAll("Sphere(" + std::to_string(segments) + ")", sphere, sphere2);
  }

  // Triangulator-heavy: Menger sponge. Forcing sponge.NumTri() first
  // separates the sponge's internal CSG construction from the measured
  // ops, so `total` reflects just each boolean.
  for (int level : {3, 4}) {
    Manifold sponge = MengerSponge(level);
    Manifold sponge2 = sponge.Translate(vec3(0.3));
    sponge.NumTri();  // force construction before timing the ops
    BenchAll("MengerSponge(" + std::to_string(level) + ")", sponge, sponge2);
  }

  // ---- LevelSet phase timings ----
  // Three SDF regimes chosen to move the dominant phase around:
  //  - cheap sphere: sparse near-surface set, trivial per-call cost.
  //  - cheap gyroid: same trivial cost but the surface threads most cells, so
  //    NearSurface touches a large fraction of the grid (Emmett's hypothesis).
  //  - expensive sphere: identical surface to the cheap sphere, but each sdf
  //    call carries a vanishing transcendental sum (no sleep - that would
  //    distort parallel timing), so sampling should dominate.
  // The tolerance axis (off vs a small positive value) toggles FindSurface
  // root-finding in NearSurface/ComputeVerts, the other lever on SDF cost.
  auto sphereSdf = [](vec3 p) { return 1.0 - la::length(p); };
  auto gyroidSdf = [](vec3 p) {
    return std::cos(p.x) * std::sin(p.y) + std::cos(p.y) * std::sin(p.z) +
           std::cos(p.z) * std::sin(p.x);
  };
  auto expensiveSphereSdf = [](vec3 p) {
    double noise = 0;
    for (int k = 1; k <= 40; ++k)
      noise += std::sin(k * p.x) * std::cos(k * p.y) * std::sin(k * p.z);
    return (1.0 - la::length(p)) + 1e-12 * noise;  // surface unmoved
  };

  struct SdfCase {
    const char* name;
    std::function<double(vec3)> sdf;
    Box bounds;
    bool expensive;
  };
  const Box sphereBounds(vec3(-1.1), vec3(1.1));
  const Box gyroidBounds(vec3(-kTwoPi), vec3(kTwoPi));
  const SdfCase cases[] = {
      {"sphere(cheap)", sphereSdf, sphereBounds, false},
      {"gyroid(cheap,high-area)", gyroidSdf, gyroidBounds, false},
      {"sphere(expensive)", expensiveSphereSdf, sphereBounds, true},
  };

  for (const auto& c : cases) {
    const vec3 s = c.bounds.Size();
    const double maxDim = std::max({s.x, s.y, s.z});
    // Cap the expensive SDF at coarser grids; the per-call cost makes fine
    // grids slow without changing which phase wins.
    const std::vector<int> resolutions =
        c.expensive ? std::vector<int>{32, 64} : std::vector<int>{32, 64, 128};
    for (int n : resolutions) {
      const double edge = maxDim / n;
      for (bool par : {false, true}) {
        for (double tol : {-1.0, edge * 0.01}) {
          BenchLevelSet(std::string(c.name) + " n=" + std::to_string(n), c.sdf,
                        c.bounds, edge, tol, par);
        }
      }
    }
  }

  // ---- FromMesh ingest phase timings ----
  // Round-trip representative meshes back through the MeshGL ingest pipeline.
  for (int segments : {64, 128, 256}) {
    const MeshGL gl = Manifold::Sphere(1, segments).GetMeshGL();
    BenchFromMesh("Sphere(" + std::to_string(segments) + ")", gl);
  }
  {
    Manifold sponge = MengerSponge(4);
    sponge.NumTri();  // force construction before measuring the ingest
    BenchFromMesh("MengerSponge(4)", sponge.GetMeshGL());
  }

  // ---- Smooth phase timings ----
  // A coarse control cage smoothed (FromMesh ingest + tangent phases) then
  // refined; the phase breakdown comes from the Smooth call.
  for (int refine : {16, 64}) {
    BenchSmooth("Sphere(32)", Manifold::Sphere(1, 32).GetMeshGL(), refine);
  }
}
