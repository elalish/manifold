import {Plugin} from 'esbuild';
import textReplace from 'esbuild-plugin-text-replace';
import * as esbuild from 'esbuild-wasm';

import {initialize} from './worker.ts';

/**
 * These content delivery networks provide NPM modules as ES modules,
 * whether they were published that way or not.
 */
export const cdnUrlHelpers: {[key: string]: (specifier: string) => string} = {
  'esm.sh': (specifier) => `https://esm.sh/${specifier}`,
  'jsDelivr': (specifier) => `https://cdn.jsdelivr.net/npm/${specifier}/+esm`,
  'skypack': (specifier) => `https://cdn.skypack.dev/${specifier}`
};

let esbuild_initialized: boolean = false;

let esbuildManifoldPlugin = (context: any):esbuild.Plugin => ({
  name: 'http',
  setup(build) {
    const skipResolve = {};
    build.onResolve({filter: /.*/}, async (args) => {
      if (args.kind === 'entry-point') return null;
      if (args.namespace === 'http-url') return null;
      if (args.path.match(/^https?:\/\//)) return null;

      if (['manifold-3d/manifoldCAD', '../../lib/manifoldCAD.ts'].includes(args.path)) {
        return {
          namespace: 'manifold-replace',
          path: args.path
        };
      }

      // skipResolve used to avoid loops.
      // https://github.com/evanw/esbuild/issues/2198#issuecomment-1104566397
      if (args.pluginData === skipResolve) return null;

      const result = await build.resolve(args.path, {
        resolveDir: args.resolveDir,
        kind: 'import-statement',
        pluginData: skipResolve
      });

      // We found a local file.  Use that.
      if (result.errors.length === 0) {
        return result
      }

      // Built in resolver failed.  Let's try to get it from a CDN.
      const response = {
        path: cdnUrlHelpers['jsDelivr'](args.path),
        namespace: 'http-url',
      };
      return response;
    });

    build.onResolve({ filter: /^https?:\/\// }, args => ({
      path: args.path,
      namespace: 'http-url',
    }));

    build.onResolve({ filter: /.*/, namespace: 'http-url' }, args => {
      if (cdnUrlHelpers['jsDelivr']('manifold-3d/manifoldCAD').endsWith(args.path)) {
        const response =  { path: 'manifold-3d/manifoldCAD', namespace: 'manifold-replace' };
        return response;
      }

      return {
        path: new URL(args.path, args.importer).toString(),
        namespace: 'http-url'
      }
    });

    // ManifoldCAD libraries can include `manifold-3d/manifoldCAD` to get
    // access to the evaluator context.  Outside of the evaluator, this
    // involvesimporting a module.  Inside the evaluator, imports are in
    // the global namespace.  The imported module will not be able to see the
    // actual evaluator context.  Here, we swap it out for an object we will
    // inject at evaluation time.
    build.onLoad({filter: /.*/, namespace: 'manifold-replace'}, ():esbuild.OnLoadResult => {
      return {
        contents: `export const { ${Object.keys(context)} } = _manifold_context`,
      }
    })

    build.onLoad({ filter: /.*/, namespace: 'http-url' }, async (args): Promise<esbuild.OnLoadResult> => {
      console.log(`Fetching ${args.path}.`)
      const response = await fetch(args.path);
      // Fixme handle missing files.
      // Fixme better logging.
      return { contents: await response.text() };
    });
  },
});

const getBaseConfig = async(): Promise<esbuild.BuildOptions> => {
  const evaluator = await initialize();
  const context = await evaluator.getFullContext();

  if (!esbuild_initialized) {
    try {
      // FIXME package this better.
      await esbuild.initialize({ wasmURL: "https://unpkg.com/esbuild-wasm/esbuild.wasm"});
    } catch (e) {
      // FIXME catch specific error.
      await esbuild.initialize({});
    }
    esbuild_initialized = true;
  }

  return {
    // Create a bundle in memory.
    bundle: true,
    write: false,
    treeShaking: false,
    platform: 'node',
    format: 'cjs',

    plugins: [
      esbuildManifoldPlugin(context),
      textReplace({
        include: /./,
        pattern: [
          [/^const\s+result\s/g, 'export const result '],
          [/^let\s+result\s/g, 'export let result '],
          [/^var\s+result\s/g, 'export var result '],
        ]
      }) as unknown as Plugin,
    ],
    // Some CDN imports will check import.meta.env.  This is only present when
    // generating an ESM bundle.  In other cases, it generates log noise, so
    // let's drop it down a log level.
    logOverride: {'empty-import-meta': 'info'}
  };
};

export const bundleFile = async(entrypoint: string): Promise<string> => {
  let transpiled: string|null = null;
  const built = await esbuild.build({
    ...(await getBaseConfig()),
    entryPoints: [entrypoint],
  });
  transpiled = built.outputFiles![0].text;
  // FIXME throw errror.s

  return transpiled;
};

export const bundleCode = async(code: string): Promise<string> => {
  let transpiled: string|null = null;
  const built = await esbuild.build(
      {...(await getBaseConfig()), stdin: {contents: code}});
  transpiled = built.outputFiles![0].text;
  return transpiled;
};