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
        three: resolve(__dirname, 'three.html'),
      },
      output: {
        entryFileNames: `assets/three/[name].js`,
        chunkFileNames: `assets/three/[name].js`,
        assetFileNames: `assets/three/[name].[ext]`
      }
    }
  },
})
