// vite.config.js
import {resolve} from 'path'
import {defineConfig} from 'vite'
import {viteStaticCopy} from 'vite-plugin-static-copy'

import emscriptenStaticWorkerOptions from './vite-fixup-plugin.js'

export default defineConfig({
  worker: {format: 'es', plugins: () => ([emscriptenStaticWorkerOptions])},
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
    fs: {allow: [resolve(__dirname, '..')]}
  },
  plugins: [viteStaticCopy({
    targets: [
      {
        src: '../dist/manifoldCAD.d.ts',
        dest: './',  // Targets are relative to 'dist'.
      },
      {
        src: '../dist/manifoldCADGlobals.d.ts',
        dest: './',
      },
      // If 'dist/manifold-encapsulated-types.d.ts' and
      // 'dist/manifold-global-types.d.ts' are missing, the web editor can't
      // load them, and type validation won't work for our core modules.
      //
      // They are generated when building the wasm module.  But we don't
      // really need to build the whole thing when working up here in JS/TS
      // territory.  Additionally, they rarely change.  In the case where the
      // binding files are not present, grab the checked in copies, and use
      // those instead.  But if they _are_ present, don't overwrite them.
      {
        src: '../manifold-encapsulated-types.d.ts',
        dest: './',
        overwrite: false,
      },
      {
        src: '../manifold-global-types.d.ts',
        dest: './',
        overwrite: false,
      }
    ],
  })],
  resolve: {
    alias: {
      path: resolve(
          __dirname,
          './node_modules/rollup-plugin-node-polyfills/polyfills/path.js')
    }
  },
  build: {
    target: 'esnext',
    sourcemap: 'hidden',
    rollupOptions: {
      input: {
        manifoldCAD: resolve(__dirname, 'index.html'),
        makeManifold: resolve(__dirname, 'make-manifold.html'),
        modelViewer: resolve(__dirname, 'model-viewer.html'),
        three: resolve(__dirname, 'three.html'),
      },
      output: {
        entryFileNames: `assets/[name].js`,
        chunkFileNames: `assets/[name].js`,
        assetFileNames: `assets/[name].[ext]`
      }
    }
  },
})
