# GSoC 2026 Final Work Product

**Contributor:** Anshul Patil
**Project:** Testing, Benchmarking, and CI Infrastructure for Manifold
**Mentors:** [@elalish](https://github.com/elalish), [@pca006132](https://github.com/pca006132)
**Repository:** [elalish/manifold](https://github.com/elalish/manifold)

## Summary

Manifold is a geometry library for topologically robust mesh Boolean operations, used by Blender, Godot Engine, OpenSCAD, BRL-CAD, and around thirty other projects. It ships roughly 2.1M monthly downloads on PyPI and 217K on npm.

The goal of this project was to strengthen the testing and performance infrastructure around that library: catching cross-platform nondeterminism, surfacing performance regressions before they merge, tracking long-term performance trends, and expanding sanitizer coverage.

## Deliverables

### 1. Cross-platform determinism checking

Manifold must produce byte-identical output on Linux, macOS, and Windows. Divergence between platforms had previously gone unnoticed.

Added a CI stage that exports meshes from each platform and compares them, plus fixes for the sources of divergence found once the check was in place.

- [#1594](https://github.com/elalish/manifold/pull/1594) cross-platform determinism check
- [#1606](https://github.com/elalish/manifold/pull/1606) deterministic trigonometric functions
- [#1623](https://github.com/elalish/manifold/pull/1623) deterministic non-trigonometric functions

### 2. Sanitizer coverage

Added dedicated CI lanes running the test suite under AddressSanitizer and UndefinedBehaviorSanitizer with Clang, giving memory and UB errors a place to surface that the normal build did not cover.

- [#1666](https://github.com/elalish/manifold/pull/1666) Linux Clang ASan+UBSan lane
- [#1728](https://github.com/elalish/manifold/pull/1728) ASan+UBSan lane, part 2

### 3. Per-PR performance guard

Every pull request now builds both its base and head commits, runs `perfTest` against each, and reports timing and peak memory deltas as a non-blocking comment. Regressions are visible at review time rather than discovered later.

- [#1680](https://github.com/elalish/manifold/pull/1680) base-vs-head perfTest guard
- [#1762](https://github.com/elalish/manifold/pull/1762) memory guard and guard fixes

### 4. Weekly benchmark suite and dashboard

A scheduled workflow runs three benchmark suites (Ember Boolean phase timings, a `perfTest` size sweep, and selected regression tests), stores structured results in a dedicated `benchmark-data` branch, and renders them as a public dashboard.

Live at [manifoldcad.org/dashboard](https://manifoldcad.org/dashboard/). Historical results are in the [`benchmark-data`](https://github.com/elalish/manifold/tree/benchmark-data) branch, organized by date and run ID.

- [#1758](https://github.com/elalish/manifold/pull/1758) weekly benchmarks

## Deferred work

**A curated OpenSCAD-derived benchmark set**, split into a fast PR subset and a broader weekly suite with documented case categories.

This was scoped to build on [#1725](https://github.com/elalish/manifold/pull/1725), an OpenSCAD-to-TypeScript compiler being developed in parallel, which would supply the benchmark cases. That PR is still in progress and its output is not yet reliable enough to derive stable benchmark measurements from. Building cases on it now would produce a suite that has to be rewritten once the compiler stabilizes.

After discussion with the mentors, this was deliberately deferred rather than rushed. The benchmark harness from #1758 is already structured to accept additional suites, so adding an OpenSCAD suite is a matter of contributing cases rather than new infrastructure.

## Current state

Everything listed above is merged and running in production CI. The determinism check, sanitizer lanes, and performance guard run on every pull request; the benchmark suite runs on schedule and publishes to the live dashboard.

## Other contributions

Work merged upstream during the program but outside the scope of the original proposal.

- [#1787](https://github.com/elalish/manifold/pull/1787) versioned documentation deploy, replacing a scheme that wiped the docs site on every push with per-release versioned subdirectories
- [#1541](https://github.com/elalish/manifold/pull/1541) `Cylinder(h, 0, r)` cone with apex at bottom
- [#1547](https://github.com/elalish/manifold/pull/1547) integer divide-by-zero in `RefineToTolerance`
- [#1554](https://github.com/elalish/manifold/pull/1554) verbose level handling
- [#1556](https://github.com/elalish/manifold/pull/1556) C bindings mutating temporary copies
- [#1575](https://github.com/elalish/manifold/pull/1575) 3MF importer and test coverage

### Documentation

- [#1564](https://github.com/elalish/manifold/pull/1564) precision handling documentation
- [#1565](https://github.com/elalish/manifold/pull/1565) degrees vs radians in the JS/TS rotation API
- [#1573](https://github.com/elalish/manifold/pull/1573) `importModel` example

## Acknowledgements

Thanks to [@elalish](https://github.com/elalish) and [@pca006132](https://github.com/pca006132) for review and direction throughout, and to the Manifold community for feedback on the benchmark dashboard.
