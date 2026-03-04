---
title: Tips and Tricks
---
## Handling Precision Issues

> [!WARNING]
> Geometry where two manifolds are meant to share a face exactly is called **marginal geometry** and should generally be avoided. Manifold uses symbolic perturbation, which means multiple topologically valid results may be within rounding error of each other — so even small imprecisions can produce unexpected output.

Two common sources of trouble:

- **Floating point drift** — JavaScript uses IEEE 754 binary floating point, so `0.1 + 0.2` evaluates to `0.30000000000000004`. Manifold does not compensate for this in its transform pipeline. A bounding box maximum of `0.30000000000000004` instead of `0.3` is enough to prevent a union from closing properly.
- **Different triangulations** — even with floating-point identical coordinates, if two coplanar faces are triangulated differently, that can also lead to gaps and slivers.

Where marginal geometry is unavoidable, the recommended workaround for floating point drift is to **work in larger, integer-friendly units**:

1. Scale coordinates up by a round factor (e.g., 1000) so your values are whole numbers
2. Use `Math.round()` after any arithmetic to eliminate accumulated drift
3. Perform all Manifold operations
4. Scale the result back down if needed

```js
const scale = 1000;

// Instead of cube([0.1, 0.1, 0.1]) translated by [0.2, 0.2, 0.2]:
const size = Math.round(0.1 * scale);   // 100
const offset = Math.round(0.2 * scale); // 200

const { cube } = Manifold;
const box = cube([size, size, size]).translate([offset, offset, offset]);
```

See [discussion #1135](https://github.com/elalish/manifold/discussions/1135) for more context.
