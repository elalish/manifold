// vite.config.js
import {resolve} from 'path'
import {defineConfig} from 'vite'

export default defineConfig({
  worker: {format: 'es'},
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
    fs: {allow: [resolve(__dirname, '../..')]}
  },
  build: {
    target: 'esnext',
    sourcemap: false,
    rollupOptions: {
      input: {
        three: resolve(__dirname, 'model-viewer.html'),
      },
      output: {
        entryFileNames: `assets/[name].js`,
        chunkFileNames: `assets/[name].js`,
        assetFileNames: `assets/[name].[ext]`
      }
    }
  },
})
