// vite.config.js
import {resolve} from 'path'
import {defineConfig} from 'vite'

export default defineConfig({
  worker: {format: 'es'},
  build: {
    target: 'esnext',
    sourcemap: 'hidden',
    rollupOptions: {
      input: {
        three: resolve(__dirname, 'make-manifold.html'),
      },
      output: {
        entryFileNames: `assets/make-manifold/[name].js`,
        chunkFileNames: `assets/make-manifold/[name].js`,
        assetFileNames: `assets/make-manifold/[name].[ext]`
      }
    }
  },
})
