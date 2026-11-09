[![npm version](https://badge.fury.io/js/manifold-cad.svg)](https://badge.fury.io/js/manifold-cad)

[**ManifoldCAD User Guide**](https://manifoldcad.org/docs/jsuser/) | [**JS/TS/WASM API**](https://manifoldcad.org/docs/jsapi/) | [**ManifoldCAD.org**](https://manifoldcad.org)

# manifold-cad

ManifoldCAD is scripted solid modelling in JavaScript or TypeScript, built on
the [manifold-3d](https://www.npmjs.com/package/manifold-3d) geometry library.
This package holds everything around the geometry engine:

- the `manifold-cad` command line tool, which turns a model script into a GLB
  or 3MF file;
- the ManifoldCAD runtime (`manifold-cad/manifoldCAD`) that model scripts
  import;
- the worker that bundles and evaluates scripts, as used by
  [ManifoldCAD.org](https://manifoldcad.org);
- glTF and 3MF import and export, including the
  [`EXT_mesh_manifold`](https://manifoldcad.org/docs/jsapi/modules/manifold-gltf.html)
  extension for [glTF Transform](https://gltf-transform.dev/)
  (`manifold-cad/manifold-gltf`).

If you only need the geometry library, install `manifold-3d` instead. It has
no dependencies.

## Installation

```bash
npm install manifold-cad
```

This also installs the matching version of `manifold-3d`.

## Writing a model

```js
import {Manifold} from 'manifold-cad/manifoldCAD';

const box = Manifold.cube([100, 100, 100], true);
const ball = Manifold.sphere(60, 100);
export default box.subtract(ball);
```

Models written for earlier versions import from `manifold-3d/manifoldCAD`.
The command line tool and ManifoldCAD.org still accept that name.

## Command line

```bash
npx manifold-cad model.js model.glb
```

```
Usage: manifold-cad [options] <infile.[js|ts]> <outfile>
```

The output format, `.glb` or `.3mf`, is chosen by the file extension.

## Documentation

- [ManifoldCAD User Guide](https://manifoldcad.org/docs/jsuser/): writing models, the command line, coordinate systems
- [Embedding ManifoldCAD](https://manifoldcad.org/docs/jsapi/documents/Embedding_ManifoldCAD.html)
  in your own application, with or without the worker
- [Examples](https://manifoldcad.org) to open and run in the browser

## License

Apache-2.0. See [LICENSE](./LICENSE).
