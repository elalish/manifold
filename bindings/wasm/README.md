[![codecov](https://codecov.io/github/elalish/manifold/branch/master/graph/badge.svg?token=IIA8G5HVS7)](https://codecov.io/github/elalish/manifold)
[![twitter](https://img.shields.io/twitter/follow/manifoldcad?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=manifoldcad)

## [ManifoldCAD.org](https://manifoldcad.org)
If you like OpenSCAD / OpenJSCAD, you might also like ManifoldCAD - our own solid modelling web app based on this package. 

![A metallic Menger sponge](https://elalish.github.io/manifold/samples/models/mengerSponge3.webp "A metallic Menger sponge")

# Manifold

[**High-level Documentation**](https://elalish.blogspot.com/search/label/Manifold) | [**API Documentation**](https://elalish.github.io/manifold/docs/html/modules.html) | [**Algorithm Documentation**](https://github.com/elalish/manifold/wiki/Manifold-Library) | [**Web Examples**](https://elalish.github.io/manifold/model-viewer.html)

[Manifold](https://github.com/elalish/manifold) is a geometry library dedicated to creating and operating on manifold triangle meshes. A [manifold mesh](https://github.com/elalish/manifold/wiki/Manifold-Library#manifoldness) is a mesh that represents a solid object, and so is very important in manufacturing, CAD, structural analysis, etc. Further information can be found on the [wiki](https://github.com/elalish/manifold/wiki/Manifold-Library).

## What's here

This library is fast with guaranteed manifold output. As such you need manifold meshes as input, which this library can create using constructors inspired by the OpenSCAD API, as well as more advanced features like smoothing and signed-distance function (SDF) level sets. You can also pass in your own mesh data, but you'll get an error status if the imported mesh isn't manifold. Various automated repair tools exist online for fixing non manifold models, usually for 3D printing. 

The most significant contribution here is a guaranteed-manifold [mesh Boolean](https://github.com/elalish/manifold/wiki/Manifold-Library#mesh-boolean) algorithm, which I believe is the first of its kind. If you know of another, please open a discussion - a mesh Boolean algorithm robust to edge cases has been an open problem for many years. Likewise, if the Boolean here ever fails you, please submit an issue! This Boolean forms the basis of a CAD kernel, as it allows simple shapes to be combined into more complex ones.

## Note on memory management

Since Manifold is a WASM module, it does not automatically garbage-collect like regular JavaScript. You must manually `delete()` each manifold object constructed by your scripts, see [discussion](https://github.com/elalish/manifold/discussions/256#discussioncomment-3944287).

## Local development

First, follow the directions in the root README to get your C++ build environment set up and working for WASM. From this directory (`bindings/wasm/`) you can test the JS bindings by running:

```
npm install
npm test
```

You can also test the manifoldCAD.org editor as well as our other example pages by serving from `bindings/wasm/examples/` with e.g. `npx http-server`.

Note that the `emcmake` command automatically copies your WASM build into `examples/built/` - these are checked into our repo in order to make sharing repro cases much easier. Note that you can test manifoldCAD.org on anyone's branch by simply going to: `https://raw.githack.com/<user>/manifold/<branch>/bindings/wasm/examples/index.html` e.g. https://raw.githack.com/elalish/manifold/glTFextension/bindings/wasm/examples/index.html

Of course these built files may easily end up with conflicts, but there's no need to address them; simply overwrite them with your newer build. These files are also not used for our deployed pages, as the deployment process overwrites them with current builds. Never edit anything in the `built` directory by hand. 

When testing [manifoldCAD.org](https://manifoldcad.org/) (either locally or the deployed version) note that it uses a service worker for faster loading. This means you need to open the page twice to see updates (the first time loads the old version and caches the new one, the second time loads the new version from cache). To see changes on each reload, open Chrome dev tools, go to the Application tab and check "update on reload".

## About the author

This library was started by [Emmett Lalish](https://elalish.blogspot.com/). I am currently a Google employee and this is my 20% project, not an official Google project. At my day job I'm the maintainer of [\<model-viewer\>](https://modelviewer.dev/). I was the first employee at a 3D video startup, [Omnivor](https://www.omnivor.io/), and before that I worked on 3D printing at Microsoft, including [3D Builder](https://www.microsoft.com/en-us/p/3d-builder/9wzdncrfj3t6?activetab=pivot%3Aoverviewtab). Originally an aerospace engineer, I started at a small DARPA contractor doing seedling projects, one of which became [Sea Hunter](https://en.wikipedia.org/wiki/Sea_Hunter). I earned my doctorate from the University of Washington in control theory and published some [papers](https://www.researchgate.net/scientific-contributions/75011026_Emmett_Lalish).
