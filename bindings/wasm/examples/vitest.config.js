import {defineConfig} from 'vitest/config';

export default defineConfig({
  test: {},
  plugins: [
    {
      name: 'keep-import-meta-url',
      enforce: 'pre',
      transform(code, id, _options) {
        // prevent NormalizeURLPlugin from replacing import.meta.url with
        // self.location
        // https://github.com/vitest-dev/vitest/blob/d8304bb4fbe16285d014f63aa71ef9969865c691/packages/vitest/src/node/plugins/normalizeURL.ts#L11
        // since it breaks `new URL(..., import.meta.url)` used by emscripten
        // EXPORT_ES6 output
        // https://github.com/emscripten-core/emscripten/blob/228af1a7de1672b582e1448d4573c20c5d2a5b5a/src/shell.js#L242
        if (id.endsWith('/dist/lib.js')) {
          return code.replace(
              /\bimport\.meta\.url\b/g,
              `String(import.meta.url)`,
          );
        }
      },
    },
  ],
});