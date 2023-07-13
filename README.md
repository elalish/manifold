[![codecov](https://codecov.io/github/elalish/manifold/branch/master/graph/badge.svg?token=IIA8G5HVS7)](https://codecov.io/github/elalish/manifold)
[![twitter](https://img.shields.io/twitter/follow/manifoldcad?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=manifoldcad)

## Users

[OpenSCAD](https://openscad.org/), [IFCjs](https://ifcjs.github.io/info/), [Grid.Space](https://grid.space/), and [OCADml](https://github.com/OCADml/OManifold) have all integrated our Manifold geometry kernel! Why? Because its reliability is guaranteed and it's 1,000 times faster than other libraries. See our [usage](https://github.com/elalish/manifold/discussions/340) and [performance](https://github.com/elalish/manifold/discussions/383) discussions for all the latest and to add your own projects & analyses.

For example, here is a log-log plot of Manifold's performance vs. earlier OpenSCAD geometry backends:

<img src="https://elalish.github.io/manifold/samples/models/perfSpheres.png" width="350px"/>

## [ManifoldCAD.org](https://manifoldcad.org)

If you like OpenSCAD / JSCAD, you might also like ManifoldCAD - our own solid modelling web app. Our WASM is not GPU-accelerated, but it's still quite fast and a good way to test out our Manifold library.

![A metallic Menger sponge](https://elalish.github.io/manifold/samples/models/mengerSponge3.webp "A metallic Menger sponge")

### Note for Firefox users

If you find the editor is stuck on **Loading...**, setting
`dom.workers.modules.enabled: true` in your `about:config`, as mentioned in the
discussion of the
[issue#328](https://github.com/elalish/manifold/issues/328#issuecomment-1473847102)
of this repository may solve the problem.

# Manifold

[**API Documentation**](https://elalish.github.io/manifold/docs/html/modules.html) | [**Algorithm Documentation**](https://github.com/elalish/manifold/wiki/Manifold-Library) | [**Blog Posts**](https://elalish.blogspot.com/search/label/Manifold) | [**Web Examples**](https://elalish.github.io/manifold/model-viewer.html)

[Manifold](https://github.com/elalish/manifold) is a geometry library dedicated to creating and operating on manifold triangle meshes. A [manifold mesh](https://github.com/elalish/manifold/wiki/Manifold-Library#manifoldness) is a mesh that represents a solid object, and so is very important in manufacturing, CAD, structural analysis, etc. Further information can be found on the [wiki](https://github.com/elalish/manifold/wiki/Manifold-Library).

This is a modern C++ library that Github's CI verifies builds and runs on a variety of platforms. Additionally, we build bindings for JavaScript ([manifold-3d](https://www.npmjs.com/package/manifold-3d) on npm), Python, and C to make this library more portable and easy to use.

We have four core dependencies, making use of submodules to ensure compatibility:
- `graphlite`: connected components algorithm
- `Clipper2`: provides our 2D subsystem
- `GLM`: a compact vector library
- `Thrust`: Nvidia's parallel algorithms library (basically a superset of C++17 std::parallel_algorithms)

## What's here

This library is fast with guaranteed manifold output. As such you need manifold meshes as input, which this library can create using constructors inspired by the OpenSCAD API, as well as more advanced features like smoothing and signed-distance function (SDF) level sets. You can also pass in your own mesh data, but you'll get an error status if the imported mesh isn't manifold. Various automated repair tools exist online for fixing non manifold models, usually for 3D printing.

The most significant contribution here is a guaranteed-manifold [mesh Boolean](https://github.com/elalish/manifold/wiki/Manifold-Library#mesh-boolean) algorithm, which I believe is the first of its kind. If you know of another, please open a discussion - a mesh Boolean algorithm robust to edge cases has been an open problem for many years. Likewise, if the Boolean here ever fails you, please submit an issue! This Boolean forms the basis of a CAD kernel, as it allows simple shapes to be combined into more complex ones.

To aid in speed, this library makes extensive use of parallelization, generally through Nvidia's Thrust library. You can switch between the CUDA, OMP, TBB, and serial C++ backends by setting a CMake flag. Not everything is so parallelizable, for instance a [polygon triangulation](https://github.com/elalish/manifold/wiki/Manifold-Library#polygon-triangulation) algorithm is included which is serial. Even if compiled for CUDA, the code will still run without a GPU, falling back to the serial version of the algorithms. The WASM build is serial-only for now, but still fast.

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

CMake flags (usage e.g. `-DMANIFOLD_USE_CUDA=ON`):
- `MANIFOLD_USE_CUDA=[<OFF>, ON]`: Provides Nvidia GPU parallelization, requires CUDA Toolkit and its nvcc compiler.
- `MANIFOLD_PAR=[<NONE>, OMP, TBB]`: Provides multi-thread parallelization, requires `libomp-dev` or `libtbb-dev`.
- `MANIFOLD_EXPORT=[<OFF>, ON]`: Enables GLB export of 3D models from the tests, requires `libassimp-dev`.
- `MANIFOLD_DEBUG=[<OFF>, ON]`: Enables internal assertions and exceptions.
- `BUILD_TEST_CGAL=[<OFF>, ON]`: Builds a CGAL-based performance [comparison](https://github.com/elalish/manifold/tree/master/extras), requires `libcgal-dev`.

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
use the extension, please add `$BUILD_DIR/tools` to your `PYTHONPATH`, where
`$BUILD_DIR` is the build directory for CMake. Examples using the python binding
can be found in `bindings/python/examples`. To see exported samples, run:
```
sudo apt install pkg-config libpython3-dev python3 python3-distutils python3-pip
pip install trimesh
python3 run_all.py -e
```

Run the following code in the interpreter for
python binding documentation:

```
>>> import manifold3d
>>> help(manifold3d)
```

For more detailed documentation, please refer to the C++ API.

## Contributing

Contributions are welcome! A lower barrier contribution is to simply make a PR that adds a test, especially if it repros an issue you've found. Simply name it prepended with DISABLED_, so that it passes the CI. That will be a very strong signal to me to fix your issue. However, if you know how to fix it yourself, then including the fix in your PR would be much appreciated!

## About the author

This library was started by [Emmett Lalish](https://elalish.blogspot.com/). I am currently a Google employee and this is my 20% project, not an official Google project. At my day job I'm the maintainer of [\<model-viewer\>](https://modelviewer.dev/). I was the first employee at a 3D video startup, [Omnivor](https://www.omnivor.io/), and before that I worked on 3D printing at Microsoft, including [3D Builder](https://www.microsoft.com/en-us/p/3d-builder/9wzdncrfj3t6?activetab=pivot%3Aoverviewtab). Originally an aerospace engineer, I started at a small DARPA contractor doing seedling projects, one of which became [Sea Hunter](https://en.wikipedia.org/wiki/Sea_Hunter). I earned my doctorate from the University of Washington in control theory and published some [papers](https://www.researchgate.net/scientific-contributions/75011026_Emmett_Lalish).
