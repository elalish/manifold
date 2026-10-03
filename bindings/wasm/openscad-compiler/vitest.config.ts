import {defineConfig} from 'vitest/config';

export default defineConfig({
  test: {
    exclude: ['**/*-single.test.ts', '**/node_modules/**'],
  },
});
