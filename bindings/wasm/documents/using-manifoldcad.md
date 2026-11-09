---
title: Using ManifoldCAD
---
## Running ManifoldCAD

### With a Browser

Just go to [ManifoldCAD.org](https://manifoldcad.org), and check out the examples!

### On the Command Line

ManifoldCAD has a [command line interface](https://github.com/elalish/manifold/blob/master/bindings/wasm/manifold-cad/bin/manifold-cad), published in the [manifold-cad](https://www.npmjs.com/package/manifold-cad) npm package.  Run it with `npx manifold-cad`, or install it with `npm install manifold-cad`.  In a checkout of this repository, it can be run directly as `./bindings/wasm/manifold-cad/bin/manifold-cad`.

```
Usage: manifold-cad [options] <infile.[js|ts]> <outfile>
```

The output file can be in either `.glb` or `.3mf` format, determined by extension.

## Coordinate Systems

ManifoldCAD uses a right-handed coordinate system where '+Z' is up.  Length units are assumed to be millimetres.  This is slightly at odds with glTF, which also uses a right handed coordinate system, but specifies '+Y' as up, and length units are in metres.

ManifoldCAD will scale and rotate automatically when importing or exporting glTF models.  This means that an arrow pointing in the '+Z' direction in will point in the '+Y' direction after export.  However, that arrow will still point up, and will still be the same size.