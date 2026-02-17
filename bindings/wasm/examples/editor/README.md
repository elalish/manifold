## Local development

First, follow the directions in the root README to get your C++ build environment set up and working for WASM.

To develop the manifoldCAD.org editor as well as our other example pages, run
```
npm run dev
```
which will serve the pages, watch for changes, and automatically rebuild and refresh. This build step doesn't do TS type-checking, so to verify everything is correct (beyond VSCode's TS linting), run
```
npm run build
```
See `package.json` for other useful scripts.

