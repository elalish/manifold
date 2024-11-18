# About Manifold

[![codecov](https://codecov.io/github/elalish/manifold/branch/master/graph/badge.svg?token=IIA8G5HVS7)](https://codecov.io/github/elalish/manifold)
[![PyPI version](https://badge.fury.io/py/manifold3d.svg)](https://badge.fury.io/py/manifold3d)
[![npm version](https://badge.fury.io/js/manifold-3d.svg)](https://badge.fury.io/js/manifold-3d)
[![twitter](https://img.shields.io/twitter/follow/manifoldcad?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=manifoldcad)

[**C++ Documentation**](https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html) | [**TS Documentation**](https://manifoldcad.org/jsdocs) | [**Algorithm Documentation**](https://github.com/elalish/manifold/wiki/Manifold-Library) | [**Blog Posts**](https://elalish.blogspot.com/search/label/Manifold) | [**Web Examples**](https://manifoldcad.org/model-viewer.html)

[Manifold](https://github.com/elalish/manifold) is a geometry library dedicated to creating and operating on manifold triangle meshes. A [manifold mesh](https://github.com/elalish/manifold/wiki/Manifold-Library#manifoldness) is a mesh that represents a solid object, and so is very important in manufacturing, CAD, structural analysis, etc. Manifold also supports arbitrary vertex properties and enables mapping of materials for rendering use-cases. Our primary goal is reliability: guaranteed manifold output without caveats or edge cases. Our secondary goal is performance: efficient algorithms that make extensive use of parallelization, or pipelining when only a single thread is available.

## Users

Here is an incomplete list of our users, whose integrations may be anywhere from in-progress to released. Please feel free to send a PR to update this list with your own project - it's quite difficult for us to keep track.

| | | |
| --- | --- | --- |
| [OpenSCAD](https://openscad.org/) | [IFCjs](https://ifcjs.github.io/info/) | [Dactyl Web Configurator](https://github.com/rianadon/dactyl-configurator) |
| [Nomad Sculpt](https://apps.apple.com/us/app/id1519508653?mt=8&platform=ipad) | [Grid.Space](https://grid.space/) | [badcad](https://github.com/wrongbad/badcad) |
| [Godot Engine](https://godotengine.org/) | [OCADml](https://github.com/OCADml/OManifold) | [Flitter](https://flitter.readthedocs.io/en/latest/) |
| [BRL-CAD](https://brlcad.org/) | [PolygonJS](https://polygonjs.com/) | [Spherene](https://spherene.ch/) |
| [trimesh](https://trimesh.org/) | [Gypsum](https://github.com/playkostudios/gypsum) | |

### Bindings & Packages

If C++ isn't your cup of tea, Manifold also has bindings to many other languages, some maintained in this repository, and others elsewhere.

| Language | Packager | Name | Maintenance |
| --- | --- | --- | --- |
| C | N/A | N/A | internal |
| TS/JS | npm | [manifold-3d](https://www.npmjs.com/package/manifold-3d) | internal |
| Python | PyPI | [manifold3d](https://pypi.org/project/manifold3d/) | internal |
| Java | N/A | [manifold](https://github.com/SovereignShop/manifold) | external |
| Clojure | N/A | [clj-manifold3d](https://github.com/SovereignShop/clj-manifold3d) | external |
| C# | NuGet | [ManifoldNET](https://www.nuget.org/packages/ManifoldNET) | external |
| Julia | Packages | [ManifoldBindings.jl](https://juliapackages.com/p/manifoldbindings) | external |
| OCaml | N/A | [OManifold](https://ocadml.github.io/OManifold/OManifold/index.html) | external |

## Frontend Sandboxes

### [ManifoldCAD.org](https://manifoldcad.org)

If you like OpenSCAD / JSCAD, you might also like ManifoldCAD - our own solid modelling web app where you script in JS/TS. This uses our npm package, [manifold-3d](https://www.npmjs.com/package/manifold-3d), built via WASM. It's not quite as fast as our raw C++, but it's hard to beat for interoperability.

### [Python Colab Example](https://colab.research.google.com/drive/1VxrFYHPSHZgUbl9TeWzCeovlpXrPQ5J5?usp=sharing)

If you prefer Python to JS/TS, make your own copy of the example notebook above. It demonstrates interop between our [`manifold3d`](https://pypi.org/project/manifold3d/) PyPI library and the popular [`trimesh`](https://pypi.org/project/trimesh/) library, including showing the interactive model right in the notebook and saving 3D model output.

![A metallic Menger sponge](https://manifoldcad.org/samples/models/mengerSponge192.png "A metallic Menger sponge")

## Manifold Library

This library is fast with guaranteed manifold output. As such you need manifold meshes as input, which this library can create using constructors inspired by the OpenSCAD API, as well as a level set function for evaluating signed-distance functions (SDF) that improves significantly over Marching Cubes. You can also pass in your own mesh data, but you'll get an error status if the imported mesh isn't manifold. We provide a [`Merge`](https://manifoldcad.org/docs/html/structmanifold_1_1_mesh_g_l_p.html) function to fix slightly non-manifold meshes, but in general you may need one of the automated repair tools that exist mostly for 3D printing.

The most significant contribution here is a guaranteed-manifold [mesh Boolean](https://github.com/elalish/manifold/wiki/Manifold-Library#mesh-boolean) algorithm, which I believe is the first of its kind. If you know of another, please open a discussion - a mesh Boolean algorithm robust to edge cases has been an open problem for many years. Likewise, if the Boolean here ever fails you, please submit an issue! This Boolean forms the basis of a CAD kernel, as it allows simple shapes to be combined into more complex ones.

Manifold has full support for arbitrary vertex properties, and also has IDs that make it easy to keep track of materials and what surfaces belong to what input objects or faces. See our [web example](https://manifoldcad.org/model-viewer.html) for a simple demonstration of combining objects with unique textures.

Also included are a novel and powerful suite of refining functions for smooth mesh interpolation. They handle smoothing both triangles and quads, as well as keeping polygonal faces flat. You can easily create sharp or small-radius edges where desired, or even drive the curvature by normal vectors.

To aid in speed, this library makes extensive use of parallelization through TBB, if enabled. Not everything is so parallelizable, for instance a [polygon triangulation](https://github.com/elalish/manifold/wiki/Manifold-Library#polygon-triangulation) algorithm is included which is serial. Even if compiled with parallel backend, the code will still fall back to the serial version of the algorithms if the problem size is small. The WASM build is serial-only for now, but still fast.

Look in the [samples](https://github.com/elalish/manifold/tree/master/samples) directory for examples of how to use this library to make interesting 3D models. You may notice that some of these examples bare a certain resemblance to my OpenSCAD designs on [Thingiverse](https://www.thingiverse.com/emmett), which is no accident. Much as I love OpenSCAD, my library is dramatically faster and the code is more flexible.

### Dependencies

Manifold no longer has **any** required dependencies! However, we do have several optional dependencies, of which the first two are strongly encouraged:
| Name | CMake Flag | Provides |
| --- | --- | --- |
| [`TBB`](https://github.com/oneapi-src/oneTBB/) |`MANIFOLD_PAR=ON` | Parallel acceleration |
| [`Clipper2`](https://github.com/AngusJohnson/Clipper2) | `MANIFOLD_CROSS_SECTION=ON` | 2D: [`CrossSection`](https://manifoldcad.org/docs/html/classmanifold_1_1_cross_section.html) |
| [`Assimp`](https://github.com/assimp/assimp) | `MANIFOLD_EXPORT=ON` | Basic Mesh I/O |
| [`Nanobind`](https://github.com/wjakob/nanobind) | `MANIFOLD_PYBIND=ON` | Python bindings |
| [`Emscripten`](https://github.com/emscripten-core/emscripten) | `MANIFOLD_JSBIND=ON` | JS bindings via WASM |
| [`GTest`](https://github.com/google/googletest/) | `MANIFOLD_TEST=ON` | Testing framework |

### 3D Formats

Please avoid saving to STL files! They are lossy and inefficient - when saving a manifold mesh to STL there is no guarantee that the re-imported mesh will still be manifold, as the topology is lost. Please consider using [3MF](https://3mf.io/) instead, as this format was designed from the beginning for manifold meshes representing solid objects. 

If you use vertex properties for things like interpolated normals or texture UV coordinates, [glTF](https://www.khronos.org/Gltf) is recommended, specifically using the [`EXT_mesh_manifold`](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_mesh_manifold/README.md) extension. This allows for the lossless and efficient transmission of manifoldness even with property boundaries. Try our [make-manifold](https://manifoldcad.org/make-manifold) page to add this extension to your existing glTF/GLB models. 

Manifold provides an optional [`MeshIO`](https://manifoldcad.org/docs/html/group___mesh_i_o.html) component based on [Assimp](https://assimp.org/), but it is limited in functionality and is primarily to aid in testing. If you are using our npm module, we have a much more capable [gltf-io.ts](https://github.com/elalish/manifold/blob/master/bindings/wasm/examples/gltf-io.ts) you can use instead. For other languages we strongly recommend using existing packages that focus on 3D file I/O, e.g. [trimesh](https://trimesh.org/) for Python, particularly when using vertex properties or materials.  

## Building

Only CMake, a C++ compiler, and Python are required to be installed and set up to build this library (it has been tested with GCC, LLVM, MSVC). However, a variety of optional dependencies can bring in more functionality, see below.

Build and test (Ubuntu or similar):
```
git clone --recurse-submodules https://github.com/elalish/manifold.git
cd manifold
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON .. && make
test/manifold_test
```

CMake flags (usage e.g. `-DMANIFOLD_DEBUG=ON`):
- `MANIFOLD_JSBIND=[OFF, <ON>]`: Build js binding (when using the emscripten toolchain).
- `MANIFOLD_CBIND=[<OFF>, ON]`: Build C FFI binding.
- `MANIFOLD_PYBIND=[OFF, <ON>]`: Build python binding, requires `nanobind`.
- `MANIFOLD_PAR=[<OFF>, ON]`: Enables multi-thread parallelization, requires `tbb`.
- `MANIFOLD_CROSS_SECTION=[OFF, <ON>]`: Build CrossSection for 2D support (needed by language bindings), requires `Clipper2`.
- `MANIFOLD_EXPORT=[<OFF>, ON]`: Enables `MeshIO` and GLB export of 3D models from the tests, requires `assimp`.
- `MANIFOLD_DEBUG=[<OFF>, ON]`: Enables internal assertions and exceptions. This incurs around 20% runtime overhead.
- `MANIFOLD_TEST=[OFF, <ON>]`: Build unit tests, requires `GTest`.
- `TRACY_ENABLE=[<OFF>, ON]`: Enable integration with tracy profiler. 
  See profiling section below.

Dependency version override:
- `MANIFOLD_USE_BUILTIN_TBB=[<OFF>, ON]`: Use builtin version of tbb.
- `MANIFOLD_USE_BUILTIN_CLIPPER2=[<OFF>, ON]`: Use builtin version of clipper2.
- `MANIFOLD_USE_BUILTIN_NANOBIND=[<OFF>, ON]`: Use builtin version of nanobind.

> Note: These three options can force the build to avoid using the system
> version of the dependency. This will either use the provided source directory
> via `FETCHCONTENT_SOURCE_DIR_*` (see below), or fetch the source from GitHub.
> Note that the dependency will be built as static dependency to avoid dynamic
> library conflict. When the system package is unavailable, the option will be
> automatically set to true.

Offline building (with missing dependencies/dependency version override):
- `MANIFOLD_DOWNLOADS=[OFF, <ON>]`: Automatically download missing dependencies.
  Need to set `FETCHCONTENT_SOURCE_DIR_*` if the dependency `*` is missing.
- `FETCHCONTENT_SOURCE_DIR_TBB`: path to tbb source (if `MANIFOLD_PAR` is enabled).
- `FETCHCONTENT_SOURCE_DIR_CLIPPER2`: path to tbb source (if `MANIFOLD_CROSS_SECTION` is enabled).
- `FETCHCONTENT_SOURCE_DIR_NANOBIND`: path to nanobind source (if `MANIFOLD_PYBIND` is enabled).
- `FETCHCONTENT_SOURCE_DIR_GOOGLETEST`: path to googletest source (if `MANIFOLD_TEST` is enabled).

> Note: When `FETCHCONTENT_SOURCE_DIR_*` is set, CMake will use the provided
> source directly without downloading regardless of the value of
> `MANIFOLD_DOWNLOADS`.

The build instructions used by our CI are in [manifold.yml](https://github.com/elalish/manifold/blob/master/.github/workflows/manifold.yml), which is a good source to check if something goes wrong and for instructions specific to other platforms, like Windows.

### WASM

To build the JS WASM library, first install NodeJS and set up emscripten:

(on Mac):
```
brew install nodejs
brew install emscripten
```
(on Linux):
```
sudo apt install nodejs
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk/emsdk_env.sh
```
Then build:
```
cd manifold
mkdir buildWASM
cd buildWASM
emcmake cmake -DCMAKE_BUILD_TYPE=Release .. && emmake make
node test/manifold_test.js
```

### Python

The CMake script will build the python binding `manifold3d` automatically. To
use the extension, please add `$BUILD_DIR/bindings/python` to your `PYTHONPATH`, where
`$BUILD_DIR` is the build directory for CMake. Examples using the python binding
can be found in `bindings/python/examples`. To see exported samples, run:
```
sudo apt install pkg-config libpython3-dev python3 python3-distutils python3-pip
pip install trimesh pytest
python3 run_all.py -e
```

Run the following code in the interpreter for
python binding documentation:

```
>>> import manifold3d
>>> help(manifold3d)
```

For more detailed documentation, please refer to the C++ API.

### Windows Shenanigans

Windows users should build with `-DBUILD_SHARED_LIBS=OFF`, as enabling shared
libraries in general makes things very complicated.

The DLL file for manifoldc (C FFI bindings) when built with msvc is in `${CMAKE_BINARY_DIR}/bin/${BUILD_TYPE}/manifoldc.dll`.
For example, for the following command, the path relative to the project root directory is `build/bin/Release/manifoldc.dll`.
```sh
cmake . -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DMANIFOLD_DEBUG=ON -DMANIFOLD_PAR=${{matrix.parallel_backend}} -A x64 -B build
```

## Contributing

Contributions are welcome! A lower barrier contribution is to simply make a PR that adds a test, especially if it repros an issue you've found. Simply name it prepended with DISABLED_, so that it passes the CI. That will be a very strong signal to me to fix your issue. However, if you know how to fix it yourself, then including the fix in your PR would be much appreciated!

### Formatting

There is a formatting script `format.sh` that automatically formats everything.
It requires clang-format, black formatter for python and [gersemi](https://github.com/BlankSpruce/gersemi) for formatting cmake files.

Note that our script can run with clang-format older than 18, but the GitHub
action check may fail due to slight differences between different versions of
clang-format. In that case, either update your clang-format version or apply the
patch from the GitHub action log.

### Profiling

There is now basic support for the [Tracy profiler](https://github.com/wolfpld/tracy) for our tests.
To enable tracing, compile with `-DTRACY_ENABLE=on` cmake option, and run the test with Tracy server running.
To enable memory profiling in addition to tracing, compile with `-DTRACY_MEMORY_USAGE=ON` in addition to `-DTRACY_ENABLE=ON`.

## About the author

This library was started by [Emmett Lalish](https://elalish.blogspot.com/). I am currently a Google employee and this is my 20% project, not an official Google project. At my day job I'm the maintainer of [\<model-viewer\>](https://modelviewer.dev/). I was the first employee at a 3D video startup, [Omnivor](https://www.omnivor.io/), and before that I worked on 3D printing at Microsoft, including [3D Builder](https://www.microsoft.com/en-us/p/3d-builder/9wzdncrfj3t6?activetab=pivot%3Aoverviewtab). Originally an aerospace engineer, I started at a small DARPA contractor doing seedling projects, one of which became [Sea Hunter](https://en.wikipedia.org/wiki/Sea_Hunter). I earned my doctorate from the University of Washington in control theory and published some [papers](https://www.researchgate.net/scientific-contributions/75011026_Emmett_Lalish).
