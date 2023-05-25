// vite.config.js
import {resolve} from 'path'
import {defineConfig} from 'vite'

export default defineConfig({
  test: {testTimeout: 15000},
  worker: {
    format: 'es',
  },
  build: {
    target: 'esnext',
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
    },
  },
})
