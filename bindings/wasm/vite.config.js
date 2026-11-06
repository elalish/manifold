// vite.config.js
import {defineConfig} from 'vite';

export default defineConfig({
  test: {
    testTimeout: 15000,
    // manifold-cad is a workspace with its own vite.config.js and tests.
    exclude: ['**/node_modules/**', 'manifold-cad/**', 'test/fixtures/**']
  }
});
