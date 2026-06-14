#Boolean2 CrossSection Core

Boolean2 is the sole `CrossSection` implementation.There is no
`MANIFOLD_CROSS_SECTION_BACKEND` selector and no Clipper2 dependency; Boolean2
is always built in-tree.

Boolean2 is a manifold-native 2D arrangement pipeline for polygon fill and
Boolean operations. The core library lives in `src/boolean2*.{
  h, cpp
}`,
and the public `CrossSection` methods dispatch to it. Sibling utilities provide
the decomposition and offset pieces that cover the rest of the public
`CrossSection` API.

## Goals

- Reuse manifold's geometric primitives where practical:
  BVH broad phase queries, `DisjointSets` for vertex equality, and
  symbolic predicates from `shared.h`.
- Make robustness testable through deterministic regression tests and
  FuzzTest targets that exercise the public `CrossSection` API and
  `CrossSection`/`Manifold` round trips.

## Algorithm Outline

Boolean2 builds a planar arrangement and filters it by per-face winding:

1. Merge vertices within the operation epsilon.
2. Collapse edges whose endpoints merge together.
3. Collect eps-padded AABB candidate edge pairs with a BVH broad phase.
4. Build per-edge lists of vertices that lie on each edge and precompute
   proper crossings from the same pair set; the helper chooses serial or TBB
   execution internally.
5. Insert strict proper edge-edge crossings using Boolean2's projected
   graph-order oracle, with shared symbolic `Interpolate`/`Shadows` helpers.
   Endpoint, T-junction, and coincident-overlap degeneracies stay in the
   vertex-on-edge/canonicalization path.
6. Merge duplicate generated intersection vertices that share an incident edge
   and are within epsilon.
7. Canonicalize sub-edges and cancel opposing multiplicities.
8. Traverse halfedge faces, propagate winding numbers, and retain boundary edges
   whose adjacent faces disagree under the requested rule.

The high-level fill/Boolean core API is in
`src/boolean2.h`. The lower-level driver returns
retained directed sub-edges plus the merged vertex map, and the wrapper turns
those edges back into regularized `manifold::Polygons`. Offset and containment
helpers live in `boolean2_offset.cpp`, declared alongside the core API in
`boolean2.h`.

## Architecture

The main dataflow is:

`boolean2.h` -> `boolean2.cpp` (conversion -> arrangement driver ->
canonicalization) -> `boolean2_winding.cpp` -> regularized `Polygons`.

| Layer | Files | Role |
| --- | --- | --- |
| Public core API | `boolean2.h`, `boolean2.cpp` (entry points) | Converts `Polygons` to local vertices plus directed edges, invokes one arrangement pass, and turns retained edges back into regularized output. |
| Arrangement coordinator | `boolean2.cpp` (driver section) | Runs one pass of merge, edge-pair discovery, edge vertex insertion, crossing insertion, canonicalization, and winding filtering. |
| Geometry leaves | `boolean2.cpp` (BVH, vertex merge, edge-vertex lists, intersections sections) | Provide the local geometric operations used by the arrangement pass. |
| Output filter | `boolean2.cpp` (canonicalize section), `boolean2_winding.cpp` | Splits directed edges into canonical sub-edges, builds halfedges, propagates winding, and keeps result boundary edges. |
| Sibling helpers | `boolean2_offset.cpp` | Offset and decomposition support for the rest of the `CrossSection` API. |

Shared leaf utilities live in `boolean2_predicates.cpp`, and debug/performance
tracing lives in `boolean2_diagnostics.h`.

## Relationship To The Sketch

This implementation follows the six-step 2D overlap-removal sketch from
upstream issue #289: epsilon-based vertex merge, collapsed-edge removal,
ordered edge vertex lists, snapped proper crossings, multiplicity-based
sub-edge canonicalization, and positive-winding output. The current code
generalizes the final filter only as far as the Boolean2 operations need:
positive-winding add/subtract/fill and intersection.

The main implementation differences are:

- Vertex merging uses deterministic union-find over all pairs within epsilon,
  then chooses the source vertex nearest each cluster centroid as the
  representative. The sketch called out weighted, up-to-date positions for
  chains of nearby vertices;
Boolean2 treats the first arrangement pass as the robustness boundary rather
        than relying on a production fixed -
    point cleanup loop.-
    Broad phases use the local boolean2 sweep /
        BVH helpers.This keeps the core independent from
            the 3D `Collider` surface while preserving the intended sub -
    quadratic candidate search.- Proper crossings are discovered from broad -
    phase edge pairs rather than a Bentley - Ottmann sweep.Endpoint - on -
    edge and collinear degeneracies are handled by the edge vertex lists; isolated crossings are inserted or snapped
  to neighboring list vertices.
- Output filtering uses a halfedge face traversal and winding propagation
  instead of independent midpoint ray casts for every sub-edge. This makes one
  winding decision per face, avoids per-edge disagreement around high-valence
  intersection vertices, and still retains exactly the edges whose adjacent
  faces differ under the selected Boolean2 predicate.

## Winding Rules

The halfedge filter keeps an edge iff the face on one side is inside the result
and the face on the other side is outside. The internal Boolean2 predicates are:

- `Add`: `w > 0`, used for union/fill under the default positive-winding rule.
- `Subtract`: implemented by appending the second input with negative
  multiplicity, then using `Add`.
- `Intersect`: `w > 1`, which corresponds to both operands covering the face
  for normalized unit-winding operands.

Boolean2 construction is Positive-only.

## Regularization And Epsilon

The core operates on `manifold::Polygons`, which cannot encode isolated
one-dimensional features. Output is therefore regularized: zero-area loops,
collapsed edges, and cancelled opposing sub-edges are dropped.

Segment crossings are decided over a positive-width shared projection interval.
Orthogonal-coordinate ties within epsilon are treated as symbolic ties, not raw
CCW fallbacks. The current tie policy first uses canonical segment geometry,
then falls back to stable edge ID for geometrically identical ties.

Callers may pass an explicit epsilon. A non-positive epsilon asks the core
to infer an operation scale and apply the local floating-point budget used by
the Boolean2 predicates. Inputs are translated into a local frame before the
arrangement is built, then translated back on output.

The core runs one arrangement pass and returns that regularized output. Repeated
`Simplify()` calls are not part of the public contract: tiny perturbations from
floating-point arithmetic, transforms, or serialization can legitimately change
future cleanup decisions within the epsilon regime.

## Validation

Build and run the `CrossSection` regression tests:

```sh
cmake -S . -B build \
  -DMANIFOLD_TEST=ON
cmake --build build -j4 --target manifold_test
ctest --test-dir build -R '^CrossSection\.' --output-on-failure
```
