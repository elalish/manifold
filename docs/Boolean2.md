# Boolean2 CrossSection Core

Status: experimental core library for a future `CrossSection` backend. The
default backend remains `clipper2`; this branch only adds the Boolean2 library
and a staged, empty-returning `boolean2` backend translation unit. A later PR
wires that translation unit into the public `CrossSection` methods.

Boolean2 is a manifold-native 2D arrangement pipeline for polygon fill and
Boolean operations. Sibling utilities in
`src/cross_section/boolean2/` provide the decomposition and offset pieces that
the follow-up backend wiring uses to cover the rest of the public
`CrossSection` API. The current branch keeps the implementation reviewable by
landing the core library before that public backend wiring is enabled.

## Goals

- Keep Clipper2 available while the new backend bakes upstream.
- Reuse manifold's geometric primitives where practical:
  BVH broad phase queries, `DisjointSets` for vertex equality, and
  symbolic predicates from `shared.h`.
- Keep the core independent from Clipper2 so the follow-up backend can compile
  either implementation from the same public API surface.
- Make robustness testable through deterministic regression tests and later
  FuzzTest targets that exercise the public `CrossSection` API and
  `CrossSection`/`Manifold` round trips.

## Build Selection

This core can be built behind the experimental backend selector:

```sh
cmake -S . -B build/clipper \
  -DMANIFOLD_CROSS_SECTION=ON \
  -DMANIFOLD_CROSS_SECTION_BACKEND=clipper2

cmake -S . -B build/boolean2 \
  -DMANIFOLD_CROSS_SECTION=ON \
  -DMANIFOLD_CROSS_SECTION_BACKEND=boolean2
```

`clipper2` is the default. Invalid backend names fail at configure time when
`MANIFOLD_CROSS_SECTION` is enabled. In this branch, selecting `boolean2`
builds the core library and an empty-returning placeholder backend, but usable
public `CrossSection` dispatch still lands in the follow-up backend-wiring PR.

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
`src/cross_section/boolean2/boolean2.h`. The lower-level driver returns
retained directed sub-edges plus the merged vertex map, and the wrapper turns
those edges back into regularized `manifold::Polygons`. Offset and containment
helpers live in `offset.h` and `containment.h`; they are included by the
follow-up backend wiring rather than by `boolean2.h`.

## Architecture

The main dataflow is:

`boolean2.h` -> `boolean2.cpp` -> `driver.cpp` ->
`canonicalize.cpp` -> `winding_filter.cpp` -> regularized `Polygons`.

| Layer | Files | Role |
| --- | --- | --- |
| Public core API | `boolean2.h`, `boolean2.cpp` | Converts `Polygons` to local vertices plus directed edges, invokes one arrangement pass, and turns retained edges back into regularized output. |
| Arrangement coordinator | `driver.cpp` | Runs one pass of merge, edge-pair discovery, edge vertex insertion, crossing insertion, canonicalization, and winding filtering. |
| Geometry leaves | `vertex_merge.cpp`, `bvh.cpp`, `edge_vert_lists.cpp`, `intersections.cpp` | Provide the local geometric operations used by the arrangement pass. |
| Output filter | `canonicalize.cpp`, `winding_filter.cpp` | Splits directed edges into canonical sub-edges, builds halfedges, propagates winding, and keeps result boundary edges. |
| Sibling helpers | `offset.cpp`, `containment.cpp` | Used by the later backend-wiring PR for offset and decomposition support in the rest of the `CrossSection` API. |

Shared leaf utilities live in `predicates.*`, and debug/performance tracing
lives in `diagnostics.h`.

## Relationship To The Sketch

This implementation follows the six-step 2D overlap-removal sketch from
upstream issue #289: epsilon-based vertex merge, collapsed-edge removal,
ordered edge vertex lists, snapped proper crossings, multiplicity-based
sub-edge canonicalization, and positive-winding output. The current code
generalizes the final filter so the same arrangement can serve union,
subtract, intersect, and construction-time fill rules.

The main implementation differences are:

- Vertex merging uses deterministic union-find over all pairs within epsilon,
  then chooses the source vertex nearest each cluster centroid as the
  representative. The sketch called out weighted, up-to-date positions for
  chains of nearby vertices; Boolean2 treats the first arrangement pass as the
  robustness boundary rather than relying on a production fixed-point cleanup
  loop.
- Broad phases use the local boolean2 sweep/BVH helpers. This keeps the core
  independent from the 3D `Collider` surface while preserving the intended
  sub-quadratic candidate search.
- Proper crossings are discovered from broad-phase edge pairs rather than a
  Bentley-Ottmann sweep. Endpoint-on-edge and collinear degeneracies are
  handled by the edge vertex lists; isolated crossings are inserted or snapped
  to neighboring list vertices.
- Output filtering uses a halfedge face traversal and winding propagation instead
  of independent midpoint ray casts for every sub-edge. This makes one winding
  decision per face, avoids per-edge disagreement around high-valence
  intersection vertices, and still retains exactly the edges whose adjacent
  faces differ under the selected rule.

## Winding Rules

The halfedge filter keeps an edge iff the face on one side is inside the
result and the face on the other side is outside. The built-in predicates are:

- `Add`: `w > 0`, used for union/fill under the default positive-winding rule.
- `Subtract`: implemented by appending the second input with negative
  multiplicity, then using `Add`.
- `Intersect`: `w > 1`, which corresponds to both operands covering the face
  for normalized unit-winding operands.
- `EvenOdd`: `w & 1`, available for construction-time fill.
- `NonZero`: `w != 0`, available for construction-time fill and offset cleanup.
- `Negative`: `w < 0`, available for construction-time fill and negative-offset
  fallback.

Construction-time fill rules are exposed through `FillByRule`. The public
`CrossSection` backend mapping from `FillRule`, `OpType`, decomposition, and
offset is staged in the backend-wiring branch.

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

Useful local checks:

```sh
cmake -S . -B build/boolean2-core \
  -DMANIFOLD_CROSS_SECTION=ON \
  -DMANIFOLD_CROSS_SECTION_BACKEND=boolean2
cmake --build build/boolean2-core -j4 --target manifold

cmake -S . -B build/clipper-test \
  -DMANIFOLD_CROSS_SECTION=ON \
  -DMANIFOLD_CROSS_SECTION_BACKEND=clipper2 \
  -DMANIFOLD_TEST=ON
cmake --build build/clipper-test -j4 --target manifold_test
ctest --test-dir build/clipper-test -R '^CrossSection\\.' --output-on-failure
```

The `CrossSection` regression command above intentionally targets `clipper2` in
this branch. `MANIFOLD_CROSS_SECTION_BACKEND=boolean2` only builds the core and
empty placeholder backend until the follow-up backend-wiring PR.

Focused backend regression tests, FuzzTest targets, and corpus tooling are
staged in later branches so this core-library branch stays reviewable.
