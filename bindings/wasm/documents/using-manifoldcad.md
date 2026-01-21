---
title: Using ManifoldCAD
---
## Running ManifoldCAD

### With a Browser

Just go to [ManifoldCAD.org](https://manifoldcad.org), and check out the examples!

### On the Command Line

ManifoldCAD has a [command line interface](https://github.com/elalish/manifold/blob/master/bindings/wasm/bin/manifold-cad).  It can be run directly as `./bin/manifold-cad`.  It can also be run via `npx`.  If Manifold is not already present, use `npx manifold-3d manifold-cad` or `npx manifold-3d`.  If Manifold is already installed, `npx manifold-cad` will suffice.

```
Usage: manifold-cad [options] <infile.js> <outfile>
```

The output file can be in either `.glb` or `.3mf` format, determined by extension.