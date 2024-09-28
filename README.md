[![codecov](https://codecov.io/github/elalish/manifold/branch/master/graph/badge.svg?token=IIA8G5HVS7)](https://codecov.io/github/elalish/manifold)
[![PyPI version](https://badge.fury.io/py/manifold3d.svg)](https://badge.fury.io/py/manifold3d)
[![npm version](https://badge.fury.io/js/manifold-3d.svg)](https://badge.fury.io/js/manifold-3d)
[![twitter](https://img.shields.io/twitter/follow/manifoldcad?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=manifoldcad)

## Users

[OpenSCAD](https://openscad.org/), [IFCjs](https://ifcjs.github.io/info/), [Grid.Space](https://grid.space/), and [OCADml](https://github.com/OCADml/OManifold) have all integrated our Manifold geometry kernel! Why? Because its reliability is guaranteed and it's 1,000 times faster than other libraries. See our [usage](https://github.com/elalish/manifold/discussions/340) and [performance](https://github.com/elalish/manifold/discussions/383) discussions for all the latest and to add your own projects & analyses.

## Manifold Frontend Sandboxes

### [ManifoldCAD.org](https://manifoldcad.org)

If you like OpenSCAD / JSCAD, you might also like ManifoldCAD - our own solid modelling web app where you script in JS/TS. This uses our npm package, [manifold-3d](https://www.npmjs.com/package/manifold-3d), built via WASM. It's not quite as fast as our raw C++, but it's hard to beat for interoperability.

*Note for Firefox users: If you find the editor is stuck on **Loading...**, setting
`dom.workers.modules.enabled: true` in your `about:config`, as mentioned in
[issue#328](https://github.com/elalish/manifold/issues/328#issuecomment-1473847102)
may solve the problem.*

### [Python Colab Example](https://colab.research.google.com/drive/1VxrFYHPSHZgUbl9TeWzCeovlpXrPQ5J5?usp=sharing)

If you prefer Python to JS/TS, make your own copy of the example notebook above. It demonstrates interop between our [`manifold3d`](https://pypi.org/project/manifold3d/) PyPI library and the popular [`trimesh`](https://pypi.org/project/trimesh/) library, including showing the interactive model right in the notebook and saving 3D model output.

![A metallic Menger sponge](https://manifoldcad.org/samples/models/mengerSponge3.webp "A metallic Menger sponge")

# Manifold

[**C++ Documentation**](https://manifoldcad.org/docs/html/topics.html) | [**TS Documentation**](https://manifoldcad.org/jsdocs) | [**Algorithm Documentation**](https://github.com/elalish/manifold/wiki/Manifold-Library) | [**Blog Posts**](https://elalish.blogspot.com/search/label/Manifold) | [**Web Examples**](https://manifoldcad.org/model-viewer.html)

[Manifold](https://github.com/elalish/manifold) is a geometry library dedicated to creating and operating on manifold triangle meshes. A [manifold mesh](https://github.com/elalish/manifold/wiki/Manifold-Library#manifoldness) is a mesh that represents a solid object, and so is very important in manufacturing, CAD, structural analysis, etc. Further information can be found on the [wiki](https://github.com/elalish/manifold/wiki/Manifold-Library).

This is a modern C++ library that Github's CI verifies builds and runs on a variety of platforms. Additionally, we build bindings for JavaScript ([manifold-3d](https://www.npmjs.com/package/manifold-3d) on npm), Python ([manifold3d](https://pypi.org/project/manifold3d/)), and C to make this library more portable and easy to use.

System Dependencies (note that we will automatically download the dependency if there is no such package on the system):
- [`GLM`](https://github.com/g-truc/glm/): A compact header-only vector library.
- [`tbb`](https://github.com/oneapi-src/oneTBB/): Intel's thread building blocks library. (only when `MANIFOLD_PAR=ON` is enabled)
- [`gtest`](https://github.com/google/googletest/): Google test library (only when test is enabled, i.e. `MANIFOLD_TEST=ON`)

Other dependencies:
- [`Clipper2`](https://github.com/AngusJohnson/Clipper2): provides our 2D subsystem

## What's here

This library is fast with guaranteed manifold output. As such you need manifold meshes as input, which this library can create using constructors inspired by the OpenSCAD API, as well as more advanced features like smoothing and signed-distance function (SDF) level sets. You can also pass in your own mesh data, but you'll get an error status if the imported mesh isn't manifold. Various automated repair tools exist online for fixing non manifold models, usually for 3D printing.

The most significant contribution here is a guaranteed-manifold [mesh Boolean](https://github.com/elalish/manifold/wiki/Manifold-Library#mesh-boolean) algorithm, which I believe is the first of its kind. If you know of another, please open a discussion - a mesh Boolean algorithm robust to edge cases has been an open problem for many years. Likewise, if the Boolean here ever fails you, please submit an issue! This Boolean forms the basis of a CAD kernel, as it allows simple shapes to be combined into more complex ones.

To aid in speed, this library makes extensive use of parallelization, generally through PSTL. You can switch between the TBB, and serial C++ backends by setting a CMake flag. Not everything is so parallelizable, for instance a [polygon triangulation](https://github.com/elalish/manifold/wiki/Manifold-Library#polygon-triangulation) algorithm is included which is serial. Even if compiled with parallel backend, the code will still fall back to the serial version of the algorithms if the problem size is small. The WASM build is serial-only for now, but still fast.

> Note: OMP and CUDA backends are now removed

Look in the [samples](https://github.com/elalish/manifold/tree/master/samples) directory for examples of how to use this library to make interesting 3D models. You may notice that some of these examples bare a certain resemblance to my OpenSCAD designs on [Thingiverse](https://www.thingiverse.com/emmett), which is no accident. Much as I love OpenSCAD, my library is dramatically faster and the code is more flexible.

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
- `MANIFOLD_JSBIND=[OFF, <ON>]`: Build js binding when using emscripten.
- `MANIFOLD_CBIND=[<OFF>, ON]`: Build C FFI binding.
- `MANIFOLD_PYBIND=[OFF, <ON>]`: Build python binding.
- `MANIFOLD_PAR=[<OFF>, ON]`: Provides multi-thread parallelization, requires `libtbb-dev` enabled.
- `MANIFOLD_CROSS_SECTION=[OFF, <ON>]`: Build CrossSection for 2D support (needed by language bindings).
- `MANIFOLD_EXPORT=[<OFF>, ON]`: Enables GLB export of 3D models from the tests, requires `libassimp-dev`.
- `MANIFOLD_DEBUG=[<OFF>, ON]`: Enables internal assertions and exceptions.
- `MANIFOLD_TEST=[OFF, <ON>]`: Build unittests.
- `TRACY_ENABLE=[<OFF>, ON]`: Enable integration with tracy profiler. 
  See profiling section below.
- `BUILD_TEST_CGAL=[<OFF>, ON]`: Builds a CGAL-based performance [comparison](https://github.com/elalish/manifold/tree/master/extras), requires `libcgal-dev`.

Offline building:
- `FETCHCONTENT_SOURCE_DIR_GLM`: path to glm source.
- `FETCHCONTENT_SOURCE_DIR_GOOGLETEST`: path to googletest source.

The build instructions used by our CI are in [manifold.yml](https://github.com/elalish/manifold/blob/master/.github/workflows/manifold.yml), which is a good source to check if something goes wrong and for instructions specific to other platforms, like Windows.

### WASM

> Note that we have only tested emscripten version 3.1.45. It is known that
  3.1.48 has some issues compiling manifold.

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

### Java / Clojure

Unofficial java bindings are currently maintained in [a fork](https://github.com/SovereignShop/manifold).

There is also a Clojure [library](https://github.com/SovereignShop/clj-manifold3d).

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
It requires clang-format 11 and black formatter for python.

If you have clang-format installed but without clang-11, you can specify the
clang-format executable by setting the `CLANG_FORMAT` environment variable.

### Profiling

There is now basic support for the [Tracy profiler](https://github.com/wolfpld/tracy) for our tests.
To enable tracing, compile with `-DTRACY_ENABLE=on` cmake option, and run the test with Tracy server running.
To enable memory profiling in addition to tracing, compile with `-DTRACY_MEMORY_USAGE=ON` in addition to `-DTRACY_ENABLE=ON`.

### Fuzzing Support

We use https://github.com/google/fuzztest for fuzzing the triangulator.

To enable fuzzing, make sure that you are using clang compiler (`-DCMAKE_CXX_COMPILER=clang -DCMAKE_C_COMPILER=clang`), running Linux, and enable fuzzing support by setting `-DMANIFOLD_FUZZ=ON`.

To run the fuzzer and minimize testcase, do
```
../minimizer.sh ./test/polygon_fuzz --fuzz=PolygonFuzz.TriangulationNoCrash
```

## About the author

This library was started by [Emmett Lalish](https://elalish.blogspot.com/). I am currently a Google employee and this is my 20% project, not an official Google project. At my day job I'm the maintainer of [\<model-viewer\>](https://modelviewer.dev/). I was the first employee at a 3D video startup, [Omnivor](https://www.omnivor.io/), and before that I worked on 3D printing at Microsoft, including [3D Builder](https://www.microsoft.com/en-us/p/3d-builder/9wzdncrfj3t6?activetab=pivot%3Aoverviewtab). Originally an aerospace engineer, I started at a small DARPA contractor doing seedling projects, one of which became [Sea Hunter](https://en.wikipedia.org/wiki/Sea_Hunter). I earned my doctorate from the University of Washington in control theory and published some [papers](https://www.researchgate.net/scientific-contributions/75011026_Emmett_Lalish).
