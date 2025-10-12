// vite.config.js
import {defineConfig} from 'vite'

export default defineConfig({
  test: {
    testTimeout: 15000,
    exclude: ['node_modules', 'examples/node_modules', 'lib/*.test.js'],
  }
})
