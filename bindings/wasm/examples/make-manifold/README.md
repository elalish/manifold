## Manifoldness and EXT_mesh_manifold

<!-- #region summary -->
<!-- This region will be included in the examples list at https://manifoldcad.org/docs/jsapi/documents/Manifold_Examples.html -->

[ [example](https://manifoldcad.org/make-manifold.html) | [source](https://github.com/elalish/manifold/blob/master/bindings/wasm/examples/make-manifold/) ]

This example will test a glTF model for manifoldness.  Manifold models will be extended with [EXT_mesh_manifold](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_mesh_manifold/README.md), a glTF extension for lossless transmission of manifold meshes.  

Our implementation of `EXT_mesh_manifold` is [manifold-gltf](https://manifoldcad.org/jsdocs/modules/manifold-gltf.html), an extension for [glTF Transform](https://gltf-transform.dev).  Although this example uses manifold to test for manifoldness, `manifold-gltf` depends only on glTF Transform.

<!-- #endregion summary -->
