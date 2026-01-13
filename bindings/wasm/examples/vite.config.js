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
      // If type declaration files are missing, the web editor can't
      // load them, and type validation won't work for our core modules.
      {
        src: '../dist/manifoldCAD.d.ts',
        dest: './',  // Targets are relative to 'dist'.
      },
      {
        src: '../dist/manifoldCADGlobals.d.ts',
        dest: './',
      },
      // Copy API documentation over.
      // This gives us /docs/jsuser and /docs/jsapi
      /*
      {
        src: '../docs/',
        dest: './docs/',
        silent: true // Fail quiet if docs have not been built.
      }
        */
      // FIXME also copy C++ docs to ./docs/api
      // FIXME create redirects.
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
