---
title: Using Manifold
group: none
---
# Using Manifold

## Installation

In your project root folder run:

```bash
npm install --save manifold-3d
```

To start using Manifold, import it and initialize it:

```js
import Module from 'manifold-3d';

const wasm = await Module();
wasm.setup();
const { Manifold } = wasm;
```

Intro example

```js
const {cube, sphere} = Manifold;
const box = cube([100, 100, 100], true);
const ball = sphere(60, 100);
const result = box.subtract(ball);
```

## Memory Management

> [!WARNING]
> Since Manifold is a WASM module, it does not automatically garbage-collect like regular JavaScript.

You must manually `delete()` each object constructed by your scripts (both `Manifold` and `CrossSection`).
See [discussion](https://github.com/elalish/manifold/discussions/256#discussioncomment-3944287).

## Handling Precision Issues

> [!WARNING]
> JavaScript uses IEEE 754 binary floating point, which means expressions like `0.1 + 0.2` evaluate to `0.30000000000000004`. Manifold intentionally follows standard floating point math and does not compensate for this in its transform pipeline.

This becomes a problem when two manifolds are meant to share a face. For example, translating a cube by `[0.2, 0.2, 0.2]` with size `[0.1, 0.1, 0.1]` produces a bounding box maximum of `0.30000000000000004` instead of `0.3`. That tiny epsilon gap means an adjacent manifold at exactly `0.3` will not properly union with it.

The recommended workaround is to **work in larger, integer-friendly units**:

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

## Examples

See [Examples](./bindings-examples.md).