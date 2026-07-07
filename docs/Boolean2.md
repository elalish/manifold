# Boolean2 CrossSection Core

Boolean2 is the sole `CrossSection` implementation. There is no
`MANIFOLD_CROSS_SECTION_BACKEND` selector and no Clipper2 dependency; Boolean2 is
always built in-tree.

Boolean2 is a manifold-native 2D arrangement pipeline for polygon fill and
Boolean operations. The core library lives in `src/boolean2*.{h,cpp}`, and the
public `CrossSection` methods dispatch to it. Sibling utilities provide the
decomposition and offset pieces that cover the rest of the public `CrossSection`
API.

## Goals

- Reuse manifold's geometric primitives where practical: BVH broad-phase
  queries, `DisjointSets` for vertex equality, and the shared `Interpolate`
  segment helper.
- Make robustness testable through deterministic regression tests and FuzzTest
  targets that exercise the public `CrossSection` API and
  `CrossSection`/`Manifold` round trips.

## Algorithm Outline

Boolean2 builds a planar arrangement with a sweep line and classifies it by
winding:

1. Merge vertices within the operation epsilon.
2. Collapse edges whose endpoints merge together.
3. Collect eps-padded AABB candidate edge pairs with a broad phase (an x-sorted
   interval sweep, or a BVH above ~1024 edges).
4. Incidence pre-split: with a narrow phase over those pairs, find the input
   vertices lying within eps of each edge and split every edge at them into
   two-vertex sub-edges. This is the one eps-scale quantization the pipeline
   applies to input features; the sweep adds no more.
5. Sweep-line arrangement. A Bentley-Ottmann sweep over the sub-edges maintains
   the edges crossing the sweep line in sorted order, so two edges are tested for
   a crossing exactly when they become adjacent, and every crossing is
   discovered. Near-concurrent events are grouped by Smith's block rule (section
   7.6.2): the contiguous run of status edges bracketing an event point is
   treated as passing through it. The result is a topologically valid
   arrangement - each vertex acts as start-vertex to as many sub-edges as it acts
   as end-vertex (section 7.6.3), the balance the winding walk needs. Because the
   arithmetic is rounded, sub-eps residual crossings can remain; the next pass
   clears them.
6. Forced-through winding. A second sweep classifies the arrangement and cleans
   up those residuals: it splits each block's edges at the event in status order
   (so a residual sub-eps crossing is split out rather than left as a
   self-intersection), accumulates the winding number through the block, and
   emits a boundary sub-edge wherever the requested rule's fill differs across it.
   Deciding membership by the maintained order, rather than an independent
   per-vertex ray cast, keeps the classification consistent across the residuals
   a dense arrangement leaves.
7. Turn the retained directed sub-edges back into regularized
   `manifold::Polygons`.

Steps 5 and 6 run as two lexicographic passes of the same sweep, in
`boolean2_sweep.cpp`. Offset and containment helpers live in
`boolean2_offset.cpp`, declared alongside the core API in `boolean2.h`.

## Architecture

The main dataflow is:

`boolean2.h` -> `boolean2.cpp` (conversion -> merge/incidence pre-split ->
sweep driver) -> `boolean2_sweep.cpp` (sweep-line arrangement + winding) ->
regularized `Polygons`.

| Layer | Files | Role |
| --- | --- | --- |
| Public core API | `boolean2.h`, `boolean2.cpp` (entry points) | Converts `Polygons` to local vertices plus directed edges, runs one arrangement build, and turns retained edges back into regularized output. |
| Arrangement coordinator | `boolean2.cpp` (driver section) | Runs vertex merge, edge collapse, the broad phase, and the incidence pre-split, then hands the sub-edges to the sweep. |
| Sweep-line engine | `boolean2_sweep.cpp` | The maintained-status Bentley-Ottmann sweep: a collect pass that discovers crossings and applies the block rule, and a measure pass that forces residual crossings through and classifies the fill by winding. |
| Geometry leaves | `boolean2.cpp` (broad phase, vertex merge, edge-vertex list sections), `boolean2_predicates.cpp` | Local geometric operations and the epsilon predicates used before the sweep. |
| Sibling helpers | `boolean2_offset.cpp` | Offset and decomposition support for the rest of the `CrossSection` API. |

Debug and performance tracing live in `boolean2_diagnostics.{h,cpp}`.

## Relationship To Smith, And To The Sketch

This engine is Julian Smith's 2D simplification sweep
([RobustBoolean.pdf](RobustBoolean.pdf), chapter 7): a Bentley-Ottmann sweep in
rounded arithmetic, with the section 7.6.2 block rule handling degenerate
near-concurrent events by the status order, followed by a winding classification
of the resulting arrangement. A computed intersection point lies within
`alpha = sqrt(153)*u*L = 12.37*u*L` of both source segments (section 8.2), and
`EpsilonFromScale(L, k)` is the `(k+1)*alpha` budget the eps predicates use (the
input feature merge and incidence pre-split use budget 1000).

The architecture differs from the original sketch (issue #289), which used a
broad phase with point-order crossing insertion and exact-distinct vertex
identity - chosen so the broad phase recovers the sweep's `O(n log n)` and
parallelizes, on the argument that Smith's rounding analysis keeps it robust
without the sweep. An earlier version of this engine followed that sketch. It
handles ordinary inputs, but does not stay consistent across dense
near-concurrent clusters: independent pairwise crossing decisions need not
compose, so the arrangement can become non-manifold and construction trips the
closed-walk assert (or, with asserts off, drops the region). A maintained sweep
order decides those clusters globally, which is why this engine uses the sweep.
Smith reports the same trade-off (section 7.8, where his non-sweep alternatives
either iterate excessively or give too much geometric error) and ties
arrangement validity to the sweep (section 7.6.3).

The remaining implementation notes:

- Vertex merging uses deterministic union-find over all pairs within epsilon,
  then chooses the source vertex nearest each cluster centroid as the
  representative. Boolean2 treats the first arrangement pass as the robustness
  boundary rather than relying on a production fixed-point cleanup loop.
- The broad phase feeds the incidence pre-split; the sweep discovers crossings
  itself from the sorted status, so there is no separate crossing broad phase.
- The incidence pre-split (step 4) applies eps-scale vertex-on-edge
  quantization to input features that the sweep does not reproduce; it is kept
  for that reason.

## Winding Rules

The winding pass keeps a sub-edge iff the requested rule classifies its two
sides differently: one side inside the result, the other outside. The internal
Boolean2 predicates are:

- `Add`: `w > 0`, used for union/fill under the default positive-winding rule.
- `Subtract`: implemented by appending the second input with negative
  multiplicity, then using `Add`.
- `Intersect`: `w > 1`, which corresponds to both operands covering a side for
  normalized unit-winding operands.

Boolean2 construction is Positive-only.

## Regularization And Epsilon

The core operates on `manifold::Polygons`, which cannot encode isolated
one-dimensional features. Output is therefore regularized: zero-area loops,
collapsed edges, and cancelled opposing sub-edges are dropped.

Epsilon quantization applies to input features - the vertex merge and the
incident-vertex pre-split. The sweep itself is tolerance-free: it runs in rounded
double arithmetic but with no eps band - crossings are constructed and split by
exact sign predicates, and near-concurrent clusters are resolved by the block
rule's status order rather than by a distance-based snap. Output vertices around
a resolved sub-epsilon tangle may consequently sit closer than epsilon to each
other; a subsequent pass fuses them as input features on re-ingest.

Callers may pass an explicit epsilon. A non-positive epsilon asks the core to
infer an operation scale and apply the local floating-point budget used by the
Boolean2 predicates. Inputs are translated into a local frame before the
arrangement is built, then translated back on output.

The core runs one arrangement pass and returns that regularized output. Repeated
`Simplify()` calls are not part of the public contract: tiny perturbations from
floating-point arithmetic, transforms, or serialization can legitimately change
future cleanup decisions within the epsilon regime.

## Validation

Build and run the `CrossSection` regression tests:

```sh
cmake -S . -B build -DMANIFOLD_TEST=ON
cmake --build build -j4 --target manifold_test
ctest --test-dir build -R '^CrossSection\.' --output-on-failure
```
