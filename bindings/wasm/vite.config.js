// vite.config.js
import {defineConfig} from 'vite';
import {coverageConfigDefaults} from 'vitest/config';

export default defineConfig({
  test: {
    testTimeout: 15000,
    exclude: ['**/node_modules/**', 'lib/*.test.js', 'test/fixtures/**'],
    coverage: {
      include: ['lib/**', 'test/**'],
      exclude: [
        'lib/*.js', 'lib/*.test.js', 'lib/node-http-import-hook.mjs',
        'test/fixtures/**', 'examples/**', ...coverageConfigDefaults.exclude
      ],
      reporters: ['text', 'html']
    }
  }
});