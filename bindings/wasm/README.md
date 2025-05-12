# About Manifold

[![codecov](https://codecov.io/github/elalish/manifold/branch/master/graph/badge.svg?token=IIA8G5HVS7)](https://codecov.io/github/elalish/manifold)
[![npm version](https://badge.fury.io/js/manifold-3d.svg)](https://badge.fury.io/js/manifold-3d)
[![twitter](https://img.shields.io/twitter/follow/manifoldcad?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=manifoldcad)

[**TS Documentation**](https://manifoldcad.org/jsdocs) | [**C++ Documentation**](https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html) |  [**Algorithm Documentation**](https://github.com/elalish/manifold/wiki/Manifold-Library) | [**Blog Posts**](https://elalish.blogspot.com/search/label/Manifold) | [**Web Examples**](https://manifoldcad.org/model-viewer.html)

[Manifold](https://github.com/elalish/manifold) is a geometry library dedicated to creating and operating on manifold triangle meshes. A [manifold mesh](https://github.com/elalish/manifold/wiki/Manifold-Library#manifoldness) is a mesh that represents a solid object, and so is very important in manufacturing, CAD, structural analysis, etc. Manifold also supports arbitrary vertex properties and enables mapping of materials for rendering use-cases. Our primary goal is reliability: guaranteed manifold output without caveats or edge cases. Our secondary goal is performance: efficient algorithms that make extensive use of parallelization, or pipelining when only a single thread is available.

## [ManifoldCAD.org](https://manifoldcad.org)

If you like OpenSCAD / JSCAD, you might also like ManifoldCAD - our own solid modelling web app where you script in JS/TS and save a GLB or 3MF file. It contains several examples showing how to use our API to make interesting shapes. You may notice that some of these examples bare a certain resemblance to my OpenSCAD designs on [Thingiverse](https://www.thingiverse.com/emmett), which is no accident. Much as I love OpenSCAD, this library is dramatically faster and the code is more flexible.

![A metallic Menger sponge](https://manifoldcad.org/samples/models/mengerSponge192.png "A metallic Menger sponge")

## Manifold Library

This library is fast with guaranteed manifold output. As such you need manifold meshes as input, which this library can create using constructors inspired by the OpenSCAD API, as well as a level set function for evaluating signed-distance functions (SDF) that improves significantly over Marching Cubes. You can also pass in your own mesh data, but you'll get an error status if the imported mesh isn't manifold. We provide a [`Merge`](https://manifoldcad.org/docs/html/structmanifold_1_1_mesh_g_l_p.html) function to fix slightly non-manifold meshes, but in general you may need one of the automated repair tools that exist mostly for 3D printing.

The most significant contribution here is a guaranteed-manifold [mesh Boolean](https://github.com/elalish/manifold/wiki/Manifold-Library#mesh-boolean) algorithm, which I believe is the first of its kind. If you know of another, please open a discussion - a mesh Boolean algorithm robust to edge cases has been an open problem for many years. Likewise, if the Boolean here ever fails you, please submit an issue! This Boolean forms the basis of a CAD kernel, as it allows simple shapes to be combined into more complex ones.

Manifold has full support for arbitrary vertex properties, and also has IDs that make it easy to keep track of materials and what surfaces belong to what input objects or faces. See our [web example](https://manifoldcad.org/model-viewer.html) for a simple demonstration of combining objects with unique textures.

Also included are a novel and powerful suite of refining functions for smooth mesh interpolation. They handle smoothing both triangles and quads, as well as keeping polygonal faces flat. You can easily create sharp or small-radius edges where desired, or even drive the curvature by normal vectors.

## Note on memory management

Since Manifold is a WASM module, it does not automatically garbage-collect like regular JavaScript. You must manually `delete()` each object constructed by your scripts (both `Manifold` and `CrossSection`), see [discussion](https://github.com/elalish/manifold/discussions/256#discussioncomment-3944287).

## Examples

Please see our usage [examples](https://github.com/elalish/manifold/tree/master/bindings/wasm/examples) to see how to interface this library with `three.js`, `<model-viewer>`, and `glTF`. Of particular note are the included libraries for lossless roundtrip of manifold meshes through glTF files, via a new extension: [EXT_mesh_manifold](https://github.com/KhronosGroup/glTF/pull/2286). 

## 3D Formats

Please avoid saving to STL files! They are lossy and inefficient - when saving a manifold mesh to STL there is no guarantee that the re-imported mesh will still be manifold, as the topology is lost. Please consider using [3MF](https://3mf.io/) instead, as this format was designed from the beginning for manifold meshes representing solid objects. 

If you use vertex properties for things like interpolated normals or texture UV coordinates, [glTF](https://www.khronos.org/Gltf) is recommended, specifically using the [`EXT_mesh_manifold`](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_mesh_manifold/README.md) extension. This allows for the lossless and efficient transmission of manifoldness even with property boundaries. Try our [make-manifold](https://manifoldcad.org/make-manifold) page to add this extension to your existing glTF/GLB models. 

## About the author

This library was started by [Emmett Lalish](https://elalish.blogspot.com/), currently a senior rendering engineer at Wētā FX. This was my 20% project when I was a Google employee, though my day job was maintaining [\<model-viewer\>](https://modelviewer.dev/). I was the first employee at a 3D video startup, [Omnivor](https://www.omnivor.io/), and before that I worked on 3D printing at Microsoft, including [3D Builder](https://www.microsoft.com/en-us/p/3d-builder/9wzdncrfj3t6?activetab=pivot%3Aoverviewtab). Originally an aerospace engineer, I started at a small DARPA contractor doing seedling projects, one of which became [Sea Hunter](https://en.wikipedia.org/wiki/Sea_Hunter). I earned my doctorate from the University of Washington in control theory and published some [papers](https://www.researchgate.net/scientific-contributions/75011026_Emmett_Lalish).
