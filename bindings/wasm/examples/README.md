# WASM/JS Examples

This is the home of our editor, [ManifoldCAD.org](https://manifoldcad.org/), as well as our other public examples of integrations with `three.js`, `<model-viewer>`, and `glTF`. Included are `manifold-gltf.js` and `gltf-io.js`, which are intended to be fairly general-purpose libraries for interfacing between manifolds and glTF. We should probably make them into their own npm package at some point.

## Local development

First, follow the directions in the root README to get your C++ build environment set up and working for WASM. From this directory (`bindings/wasm/examples/`) you can test the JS bindings by running:

```
npm install
npm test
```

To develop the manifoldCAD.org editor as well as our other example pages, run
```
npm run dev
```
which will serve the pages, watch for changes, and automatically rebuild and refresh. This build step doesn't do TS type-checking, so to verify everything is correct (beyond VSCode's TS linting), run
```
npm run build
```
See `package.json` for other useful scripts.

Note that the `emcmake` command automatically copies your WASM build into `built/`, (here, not just under the `buildWASM` directory) which is then packaged by Vite into `dist/assets/`.

To debug the WASM build directly in Chrome dev tools, simply build in debug mode:
```
emcmake cmake -DCMAKE_BUILD_TYPE=Debug .. && emmake make
```
and install the [DWARF](goo.gle/wasm-debugging-extension) Chrome extension.

When testing [ManifoldCAD.org](https://manifoldcad.org/) (either locally or the
deployed version) note that it uses a service worker for faster loading. This
means you need to open the page twice to see updates (the first time loads the
old version and caches the new one, the second time loads the new version from
cache). To see changes on each reload, open Chrome dev tools, go to the
Application tab and check "update on reload".

### Note for firefox users

To use the manifoldCAD.org editor (`npm run dev`), you'll likely have to set
`dom.workers.modules.enabled: true` in your `about:config`, as mentioned in the
discussion of the
[issue#328](https://github.com/elalish/manifold/issues/328#issuecomment-1473847102)
of this repository.
