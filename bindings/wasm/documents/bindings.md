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

## Examples

See [Examples](./bindings-examples.md).