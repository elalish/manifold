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
        three: resolve(__dirname, 'model-viewer.html'),
      },
      output: {
        entryFileNames: `assets/model-viewer/[name].js`,
        chunkFileNames: `assets/model-viewer/[name].js`,
        assetFileNames: `assets/model-viewer/[name].[ext]`
      }
    }
  },
})
