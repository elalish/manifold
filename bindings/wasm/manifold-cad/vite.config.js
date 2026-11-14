// vite.config.js
import {defineConfig} from 'vite';
import {coverageConfigDefaults} from 'vitest/config';

export default defineConfig({
  test: {
    testTimeout: 15000,
    exclude: ['**/node_modules/**', 'lib/*.test.js', 'test/fixtures/**'],
    coverage: {
      include: ['lib/**'],
      exclude: [
        'lib/*.js', 'lib/*.test.js', 'lib/node-http-import-hook.mjs',
        'lib/*.d.ts', 'examples/**', ...coverageConfigDefaults.exclude
      ],
      reporters: ['text', 'html']
    }
  }
});