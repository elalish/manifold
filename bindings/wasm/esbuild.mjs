import * as esbuild from 'esbuild';

await esbuild.build({
  entryPoints: ['lib/worker.ts'],
  bundle: true,
  outfile: 'dist/worker.bundled.js',
  format: 'esm',
  target: 'esnext',
  sourcemap: 'inline',
  sourcesContent: false,
  external: [
    'node:path', 'node:url', 'node:fs', 'node:module', 'module',
  ]
});
