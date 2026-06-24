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

Boolean2 builds a planar arrangement and filters it by winding:

1. Merge vertices within the operation epsilon.
2. Collapse edges whose endpoints merge together.
3. Collect eps-padded AABB candidate edge pairs with a BVH broad phase.
4. With a narrow phase over those pairs (serial or TBB internally), find the
   vertices incident to each edge, then split every edge at its incident vertices
   into two-vertex sub-edges.
5. Repeat the broad phase over the sub-edges, find proper crossings among them
   with a second narrow phase, and insert them. A crossing reuses an existing
   vertex id when one lies within epsilon. Because each edge is already split at
   its incident vertices, a crossing near such a vertex lands on a sub-edge that
   does not share it and is kept rather than collapsed onto it. Endpoint,
   T-junction, and coincident-overlap degeneracies become ordinary sub-edge
   endpoints. There is no separate post-hoc merge.
6. Canonicalize sub-edges and cancel opposing multiplicities.
7. Filter by winding: for each canonical sub-edge, evaluate the winding of the
   face just left of vMin->vMax at the start vertex (a +x ray cast under a
   symbolic perturbation into that face), take the right side as leftW - mult,
   and keep the edge iff the requested rule disagrees across it. The ray cast
   reuses one BVH built over the canonical sub-edges, so the pass is roughly
   O(E log E) rather than O(E^2).

The high-level fill/Boolean core API is in `src/boolean2.h`. The lower-level
driver returns retained directed sub-edges plus the merged vertex map, and the
wrapper turns those edges back into regularized `manifold::Polygons`. Offset and
containment helpers live in `boolean2_offset.cpp`, declared alongside the core
API in `boolean2.h`.

## Architecture

The main dataflow is:

`boolean2.h` -> `boolean2.cpp` (conversion -> arrangement driver ->
canonicalization) -> `boolean2_winding.cpp` -> regularized `Polygons`.

| Layer | Files | Role |
| --- | --- | --- |
| Public core API | `boolean2.h`, `boolean2.cpp` (entry points) | Converts `Polygons` to local vertices plus directed edges, invokes one arrangement pass, and turns retained edges back into regularized output. |
| Arrangement coordinator | `boolean2.cpp` (driver section) | Runs merge, edge-pair discovery, incidence split into sub-edges, crossing insertion, canonicalization, and winding filtering in one arrangement build. |
| Geometry leaves | `boolean2.cpp` (BVH, vertex merge, edge-vertex list, intersection sections), `boolean2_predicates.cpp` | Provide the local geometric operations and the projected segment-order predicates used by the arrangement pass. |
| Output filter | `boolean2.cpp` (canonicalize section), `boolean2_winding.cpp` | Splits directed edges into canonical sub-edges, then evaluates the winding at each sub-edge's start vertex and keeps the boundary edges the rule selects. |
| Sibling helpers | `boolean2_offset.cpp` | Offset and decomposition support for the rest of the `CrossSection` API. |

Debug and performance tracing live in `boolean2_diagnostics.{h,cpp}`.

## Relationship To The Sketch

This implementation follows the six-step 2D overlap-removal sketch from upstream
issue #289: epsilon-based vertex merge, collapsed-edge removal, ordered edge
vertex lists, snapped proper crossings, multiplicity-based sub-edge
canonicalization, and positive-winding output. The current code generalizes the
final filter only as far as the Boolean2 operations need: positive-winding
add/subtract/fill and intersection.

The main implementation differences are:

- Vertex merging uses deterministic union-find over all pairs within epsilon,
  then chooses the source vertex nearest each cluster centroid as the
  representative. Boolean2 treats the first arrangement pass as the robustness
  boundary rather than relying on a production fixed-point cleanup loop.
- Broad phases use the local boolean2 sweep and BVH helpers. This keeps the core
  independent from the 3D `Collider` surface while preserving the intended
  sub-quadratic candidate search.
- Proper crossings are discovered from broad-phase pairs rather than a
  Bentley-Ottmann sweep. Edges are split at their incident vertices first, so
  endpoint-on-edge and collinear degeneracies become ordinary sub-edge endpoints
  and a crossing near such a vertex is kept on its own sub-edge; crossings are
  then inserted, snapped to a neighboring vertex when one is within epsilon.
  Splitting first can make the sub-edge broad phase super-quadratic on dense
  near-collinear input, which has not mattered in practice.

## Winding Rules

The per-edge filter keeps a canonical sub-edge iff the requested rule classifies
its two sides differently: one side inside the result, the other outside. The
internal Boolean2 predicates are:

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

Whether two segments cross is a sign decision: a crossing exists where each
strictly straddles the other over a positive-width shared projection interval,
with no epsilon band on nearness to an endpoint. A crossing that lands within
epsilon of an endpoint is kept and snapped to that endpoint at insertion, not
rejected. Orthogonal-coordinate ties within epsilon are treated as symbolic
ties, not raw CCW fallbacks: the tie policy first uses canonical segment
geometry, then falls back to stable edge ID for geometrically identical ties.
Splitting an edge at a vertex that lies on it stays an epsilon (bounded-distance)
decision, distinct from this sign-based crossing test.

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
