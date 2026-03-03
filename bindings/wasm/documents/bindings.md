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

## Degrees vs. Radians

Manifold's rotation API takes **degrees**, not radians. This is a deliberate design choice: Manifold can eliminate floating point error entirely for multiples of 90°, and uses more efficient code paths when it detects such rotations.

```js
// Correct — pass degrees directly
box.rotate([90, 0, 0]);
box.rotate([0, 45, 0]);

// Wrong — Math.PI / 2 introduces floating point error
box.rotate([Math.PI / 2, 0, 0]); // ≈ 1.5707963..., not 90
```

If your own code works in radians (e.g. when using `Math.atan2`), convert to degrees before passing to Manifold:

```js
const toDeg = (rad) => rad * (180 / Math.PI);

const angle = Math.atan2(y, x); // radians
box.rotate([0, 0, toDeg(angle)]);
```

For common angles, always prefer the exact degree value (`90`, `45`, `30`, etc.) over a computed approximation to get the precision guarantee. See [issue #1262](https://github.com/elalish/manifold/issues/1262) for a real-world example where a rotation of `-89.999999999999` instead of `-90` caused mesh cracks.

## Examples

See [Examples](./bindings-examples.md).