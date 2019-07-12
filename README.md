# Manifold

This is a geometry library dedicated to creating and operating on manifold meshes. A [manifold mesh](https://github.com/elalish/manifold/wiki/Manifoldness) is a mesh that represents a solid object, and so is very important in manufacturing, CAD, structural analysis, etc. Further information can be found on the [wiki](https://github.com/elalish/manifold/wiki).

## What's here

This library is intended to be fast with guaranteed manifold output. As such you need manifold meshes to start, which can be hard to come by since it doesn't matter at all for 3D graphics. This library can create simple primitive meshes but also links in Assimp, which will import many kinds of 3D files, but you'll get an error if the imported mesh isn't manifold. Various automated repair tools exist online for fixing non manifold models, usually for 3D printing. 

The most significant contribution here is a guaranteed manifold mesh Boolean algorithm, which I believe is the first of its kind. If anyone knows of another, please tell me. Likewise, if the Boolean here ever fails you, please submit an issue! This Boolean forms the basis of CAD kernel, as it allows simple shapes to be combined into more complex ones.

To aid in speed, this library makes extensive use of parallelization, generally through Nvidia's thrust library. You can switch between the CUDA, OMP and serial C++ backends by setting a CMake flag. 

Not everything is so parallelizable, for instance a polygon triangulation algorithm is included which is serial. 

## Building

The canonical build instructions are in the [.travis.yml file](https://github.com/elalish/manifold/blob/master/.travis.yml), as that is what this project's continuous integration server uses to build and test. I only build under Ubuntu Linux; I used to build in OSX too, but a combination of iffy CUDA and OMP support made me abandon it for the moment. 

Look in the [tools](https://github.com/elalish/manifold/tree/master/tools) directory for examples of how to use this library in your own code.

## Contributing

Contributions are welcome! A lower barrier contribution is to simply make a PR that adds a test, especially if it repros an issue you've found. Simply name it prepended with DISABLED_, so that it passes the CI. That will be a very strong signal to me to fix your issue. However, if you know how to fix it yourself, then including the fix in your PR would be much appreciated!

## About the author

This library is by [Emmett Lalish](https://www.thingiverse.com/emmett). I am currently a Google employee, but this is not a Google project. At my day job I work on [\<model-viewer\>](https://github.com/GoogleWebComponents/model-viewer). I was the first employee at a 3D video startup, [Omnivor](https://www.omnivor.io/), and before that I worked on 3D printing at Microsoft, including [3D Builder](https://www.microsoft.com/en-us/p/3d-builder/9wzdncrfj3t6?activetab=pivot%3Aoverviewtab). Originally an aerospace engineer, I started at a small DARPA contractor doing seedling projects, one of which became [Sea Hunter](https://en.wikipedia.org/wiki/Sea_Hunter). I earned my doctorate from the University of Washington in control theory and published some [papers](https://www.researchgate.net/scientific-contributions/75011026_Emmett_Lalish).
