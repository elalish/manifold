---
title: Embedding manifoldCAD
group: manifoldCAD Runtime
category: Core
---
# Embedding manifoldCAD

There are three ways to use manifold in your own projects:

 1. Using the manifoldCAD worker.  The worker adds overhead, but it will sandbox scripts.  It also makes it possible to include packages at runtime.
 1. Using manifoldCAD libraries directly.  ManifoldCAD implements importers, exporters, and garbage collection amongst other utilities.  These modules are reasonably light and can be used outside of manifoldCAD.
 1. Using the [manifold WASM module directly](./bindings.md) without any of the above.

## Using the manifoldCAD worker

The worker is a code evaluation sandbox intended to run as a [web worker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API).  It will bundle a script, include any dependencies, and even fetch those dependencies from a CDN if required.  At run time, it will evaluate the script as a [dynamically created function](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Function/Function).

See:

  * {@link lib/worker | worker API}
  * [manifold-cad](https://github.com/elalish/manifold/blob/master/bindings/wasm/bin/manifold-cad) CLI

### Detecting the worker

> [!NOTE]
> There are slight behaviour differences between scripts running inside the manifoldCAD worker, and those outside.
>
> See our [import integration tests](https://github.com/elalish/manifold/blob/master/bindings/wasm/test/import.test.ts) for examples.

When evaluated through the worker, {@link manifoldCAD!isManifoldCAD | isManifoldCAD()} will return true.  Otherwise it returns false.

## Using the manifoldCAD runtime directly

