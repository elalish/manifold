# About Manifold

[![codecov](https://codecov.io/github/elalish/manifold/branch/master/graph/badge.svg?token=IIA8G5HVS7)](https://codecov.io/github/elalish/manifold)
[![npm version](https://badge.fury.io/js/manifold-3d.svg)](https://badge.fury.io/js/manifold-3d)
[![twitter](https://img.shields.io/twitter/follow/manifoldcad?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=manifoldcad)

[**TS Documentation**](https://manifoldcad.org/jsdocs) | [**C++ Documentation**](https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html) |  [**Algorithm Documentation**](https://github.com/elalish/manifold/wiki/Manifold-Library) | [**Blog Posts**](https://elalish.blogspot.com/search/label/Manifold)

A [manifold mesh](https://github.com/elalish/manifold/wiki/Manifold-Library#manifoldness) is a mesh that represents a solid object, and so is very important in manufacturing, CAD, structural analysis, etc.  'Manifold' implies that there are no gaps or tears, and that all faces are oriented outwards.

## Manifold Library

[ [Using Manifold](https://manifoldcad.org/jsdocs/documents/Using_Manifold.html) |  [Examples](https://manifoldcad.org/jsdocs/documents/Manifold_Examples.html) | [API Reference](https://manifoldcad.org/jsdocs/modules/manifold.html) ]

[Manifold](https://github.com/elalish/manifold) is a geometry library dedicated to creating and operating on manifold triangle meshes.  It is available as a WASM module that runs in any modern browser.

Our primary goal is reliability: guaranteed manifold output without caveats or edge cases. Our secondary goal is performance: efficient algorithms that make extensive use of parallelization, or pipelining when only a single thread is available.


## [ManifoldCAD.org](https://manifoldcad.org)

<img alt="A metallic Menger sponge" src="https://manifoldcad.org/samples/models/mengerSponge192.png" style="float:right">

[ [ManifoldCAD](https://manifoldcad.org) | [ManifoldCAD CLI](https://manifoldcad.org/jsdocs/documents/Using_manifoldCAD.html#on-the-command-line) | [API Reference](https://manifoldcad.org/jsdocs/modules/manifoldCAD.html) ]

If you like OpenSCAD / JSCAD, you might also like ManifoldCAD - our own solid modelling web app where you script in JS/TS and save a GLB or 3MF file. It contains several examples showing how to use our API to make interesting shapes. You may notice that some of these examples bare a certain resemblance to my OpenSCAD designs on [Thingiverse](https://www.thingiverse.com/emmett), which is no accident. Much as I love OpenSCAD, this library is dramatically faster and the code is more flexible.

manifoldCAD = manifold + [TypeScript](https://www.typescriptlang.org/) + [glTF](https://www.khronos.org/gltf/)


### ManifoldCAD Runtime

The ManifoldCAD runtime extends manifold, adding:

  * glTF import and export, via [glTF Transform](https://gltf-transform.dev).
  * Support for glTF materials and animations.
  * Utilities for instantiating the WASM object, and garbage collection for objects created in WASM.
  * Bundling and sandboxing user scripts, including npm packages.

These modules are written in typescript and [can be used outside of ManifoldCAD](https://manifoldcad.org/jsdocs/documents/Embedding_manifoldCAD.html).

<div style="clear:both"></div>

## 3D Formats

[ [EXT_mesh_manifold](https://manifoldcad.org/jsdocs/modules/manifold-gltf.html) ]

Please avoid saving to STL files! They are lossy and inefficient - when saving a manifold mesh to STL there is no guarantee that the re-imported mesh will still be manifold, as the topology is lost. Please consider using [3MF](https://3mf.io/) instead, as this format was designed from the beginning for manifold meshes representing solid objects. 

If you use vertex properties for things like interpolated normals or texture UV coordinates, [glTF](https://www.khronos.org/Gltf) is recommended, specifically using the [`EXT_mesh_manifold`](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_mesh_manifold/README.md) extension. This allows for the lossless and efficient transmission of manifoldness even with property boundaries. Try our [make-manifold](https://manifoldcad.org/make-manifold) page to add this extension to your existing glTF/GLB models. 

## About the author

This library was started by [Emmett Lalish](https://elalish.blogspot.com/), currently a senior rendering engineer at Wētā FX. This was my 20% project when I was a Google employee, though my day job was maintaining [\<model-viewer\>](https://modelviewer.dev/). I was the first employee at a 3D video startup, [Omnivor](https://www.omnivor.io/), and before that I worked on 3D printing at Microsoft, including [3D Builder](https://www.microsoft.com/en-us/p/3d-builder/9wzdncrfj3t6?activetab=pivot%3Aoverviewtab). Originally an aerospace engineer, I started at a small DARPA contractor doing seedling projects, one of which became [Sea Hunter](https://en.wikipedia.org/wiki/Sea_Hunter). I earned my doctorate from the University of Washington in control theory and published some [papers](https://www.researchgate.net/scientific-contributions/75011026_Emmett_Lalish).
