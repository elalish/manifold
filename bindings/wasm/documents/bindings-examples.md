---
title: Manifold Examples
group: none
---
# Manifold Examples

## Interoperation with three.js

[ [example](https://manifoldcad.org/three.html) | [source](https://github.com/elalish/manifold/blob/master/bindings/wasm/examples/three.ts) ]

This example shows how to marshal models between [three.js](https://threejs.org/) and Manifold.

## Passing glTF properties

[ [example](https://manifoldcad.org/model-viewer.html) | [source](https://github.com/elalish/manifold/blob/master/bindings/wasm/examples/model-viewer-script.ts) ]

Manifold has full support for arbitrary vertex properties, and also has IDs that make it easy to keep track of materials and what surfaces belong to what input objects or faces.  This example is a simple demonstration of combining objects with unique textures.  The final result is shown using [\<model-viewer\>](https://modelviewer.dev/).

## Manifoldness and EXT_mesh_manifold

[ [example](https://manifoldcad.org/make-manifold.html) | [source](https://github.com/elalish/manifold/blob/master/bindings/wasm/examples/make-manifold.ts) ]

This example will test a glTF model for manifoldness.  Manifold models will be extended with [EXT_mesh_manifold](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_mesh_manifold/README.md), a glTF extension for lossless transmission of manifold meshes.  

Our implementation of `EXT_mesh_manifold` is [manifold-gltf](https://manifoldcad.org/jsdocs/modules/manifold-gltf.html), an extension for [glTF Transform](https://gltf-transform.dev).  Although this example uses manifold to test for manifoldness, `manifold-gltf` depends only on glTF Transform.