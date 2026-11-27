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

`manifold-3d` contains only the geometry library and has no dependencies.  The ManifoldCAD runtime, worker and command line tool, along with glTF and 3MF import and export, are in the separate [manifold-cad](https://www.npmjs.com/package/manifold-cad) package.  See [Embedding ManifoldCAD](./embedding-manifoldcad.md).

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

You must manually `delete()` each object constructed by your scripts (`Manifold`, `CrossSection` and `ExecutionContext`).
See [discussion](https://github.com/elalish/manifold/discussions/256#discussioncomment-3944287).

## Errors

A call that evaluates a Manifold with an error throws a `ManifoldError`, whose `code` is the status that `status()` returns.
Constructors and other calls that return Manifolds check their results, and queries such as `volume()`, `isEmpty()` and `getMesh()` check their inputs.
Booleans, transforms, `trimByPlane()`, `withContext()`, `status()` and CrossSection operations other than `extrude()` and `revolve()` are lazy or have no status, and never throw.
An error in a lazy result throws at the next call that evaluates it, or you can check it with `status()` without an exception.
Invalid parameters, such as a negative size, give `'InvalidConstruction'`, while empty input, such as an empty CrossSection, gives an empty Manifold with no error.
When an eager op that takes an `ExecutionContext`, or one of the context's factories, detects a cancel, it throws with code `'Cancelled'`.

```js
const {ManifoldError} = wasm;
try {
  Manifold.cube([-1, 1, 1]);
} catch (e) {
  if (!(e instanceof ManifoldError)) throw e;
  console.log(e.code);  // 'InvalidConstruction'
}
```

A call that throws a `ManifoldError` deletes the objects it created, including every element of an array result.

## Examples

See [Examples](./bindings-examples.md).