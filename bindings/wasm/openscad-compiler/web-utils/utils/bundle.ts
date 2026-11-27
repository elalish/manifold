import * as esbuild from 'esbuild-wasm';
import path from 'path-browserify';

import {vfs} from '../../vfs/vfs.js';

let initialized: Promise<void>|undefined;

async function initializeBrowser(wasmUrl?: string): Promise<void> {
  if (!initialized) {
    const url = wasmUrl ?
        new URL(wasmUrl, location.href).href :
        new URL('../../node_modules/esbuild-wasm/esbuild.wasm', import.meta.url)
            .href;
    initialized = esbuild.initialize({wasmURL: url}).catch(error => {
      initialized = undefined;
      throw error;
    });
  }
  await initialized;
}

// bundle the VFS module graph into one ES module
export async function bundleVfsEntry(
    entryPath: string, runtimeUrl: string, wasmUrl?: string): Promise<string> {
  if (typeof process === 'undefined') await initializeBrowser(wasmUrl);

  const result = await esbuild.build({
    entryPoints: [entryPath],
    bundle: true,
    format: 'esm',
    platform: 'browser',
    target: 'esnext',
    write: false,
    plugins: [{
      name: 'openscad-vfs',
      setup(build) {
        build.onResolve({filter: /.*/}, args => {
          if (!args.path.startsWith('.') && !args.path.startsWith('/')) {
            return {path: args.path, external: true};
          }
          const baseDir = args.importer ? path.dirname(args.importer) : '/';
          const absPath = path.resolve(baseDir, args.path);
          if (absPath === '/runtime/runtime.js') {
            return {path: runtimeUrl, external: true};
          }
          const sourcePath = absPath.replace(/\.js$/i, '.ts');
          if (!vfs.isFileSync(sourcePath)) {
            return {errors: [{text: `VFS module not found: ${absPath}`}]};
          }
          return {path: sourcePath, namespace: 'openscad-vfs'};
        });
        build.onLoad(
            {filter: /.*/, namespace: 'openscad-vfs'},
            args => ({
              contents: vfs.readFileSync(args.path, 'utf8') as string,
              loader: 'ts',
            }));
      },
    }],
  });

  const code = result.outputFiles[0]?.text;
  if (code === undefined) throw new Error('No bundled web output was produced');
  return code;
}
