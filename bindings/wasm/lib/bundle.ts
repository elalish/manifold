import {Plugin} from 'esbuild';
import type {BuildFailure} from 'esbuild';
import textReplace from 'esbuild-plugin-text-replace';
import * as esbuild from 'esbuild-wasm';

import {isNode} from './util.ts';

let esbuildWasmUrl: string|null = null;

export const setWasmUrl =
    (url: string) => {
      esbuildWasmUrl = url;
    }

export class BundlerError extends Error {
  location?: esbuild.Location;
  error: esbuild.Message;

  constructor(failure: BuildFailure, options?: ErrorOptions) {
    super(undefined, options);
    this.cause = failure;
    this.error = failure.errors[0];
  }

  get name(): string {
    return 'BundlerError';
  }

  get message(): string {
    return this.error.text;
  }

  get stack(): string|undefined {
    const {file, line, column} = this.error.location!;
    return `${this.toString()}\n    at ${file}:${line}:${column}`;
  }
};

/**
 * These content delivery networks provide NPM modules as ES modules,
 * whether they were published that way or not.
 */
export const cdnUrlHelpers: {[key: string]: (specifier: string) => string} = {
  'esm.sh': (specifier) => `https://esm.sh/${specifier}`,
  'jsDelivr': (specifier) => `https://cdn.jsdelivr.net/npm/${specifier}/+esm`,
  'skypack': (specifier) => `https://cdn.skypack.dev/${specifier}`
};

/**
 * This is a plugin for esbuild that has three functions:
 *
 *   * It resolves NPM packages to urls served by various CDNs.
 *   * It fetches imports from http/https urls.
 *   * It provides evaluation context to npm packages.
 *
 * ManifoldCAD libraries can include `manifold-3d/manifoldCAD` to get access to
 * the evaluator context.  Outside of the evaluator, this involves importing a
 * module.  Inside the evaluator, imports are in the global namespace.  The
 * imported module will not be able to see the actual evaluator context.  Here,
 * we will swap it out for an object we will inject at evaluation time.
 *
 * @returns
 */
export const esbuildManifoldPlugin = (jsCDN: string =
                                          'jsDelivr'): esbuild.Plugin => ({
  name: 'esbuild-manifold-plugin',
  async setup(build) {
    let manifoldCADExportPath: string|null = null;
    const manifoldCADExportSpecifier = 'manifold-3d/manifoldCAD'
    const ManifoldCADExportMatch = /^manifold-3d\/manifoldCAD(.ts|.js)?$/
    const manifoldCADExportNames = [
      'setMinCircularAngle', 'setMinCircularEdgeLength', 'setCircularSegments',
      'getCircularSegments', 'resetToCircularDefaults', 'Mesh', 'Manifold',
      'CrossSection', 'triangulate', 'show', 'only', 'setMaterial',
      'setMorphStart', 'setMorphEnd'
    ];

    if (isNode()) {
      // We only need to check against the local manifoldCAD context on disk if
      // we happen to be running in node.
      (async () => {
        const {dirname, resolve} = await import('node:path');
        const {fileURLToPath} = await import('node:url');
        const __dirname = dirname(fileURLToPath(import.meta.url));
        manifoldCADExportPath = resolve(__dirname, './manifoldCAD.ts');
      })();
    }

    const skipResolve = {};
    // Try to resolve local files.  If we can't, blindly resolve them to a CDN.
    build.onResolve({filter: /.*/}, async (args) => {
      // skipResolve used to avoid loops.
      // https://github.com/evanw/esbuild/issues/2198#issuecomment-1104566397
      if (args.pluginData === skipResolve) return null;

      // Skip a few cases handled elsewhere.
      if (args.kind === 'entry-point') return null;
      if (args.namespace === 'http-url') return null;
      if (args.path.match(/^https?:\/\//)) return null;

      // Is this a manifoldCAD context import?
      if (args.path.match(ManifoldCADExportMatch)) {
        return {namespace: 'manifold-replace', path: args.path};
      }

      // Try esbuilds' resolver first.
      const result = await build.resolve(args.path, {
        resolveDir: args.resolveDir,
        kind: 'import-statement',
        pluginData: skipResolve
      });

      // We found a local file!
      if (result.errors.length === 0) {
        if (manifoldCADExportPath && manifoldCADExportPath === result.path) {
          // It resolved to our local manifoldCAD context.
          return {namespace: 'manifold-replace', path: args.path};
        } else {
          return result;
        }
      }

      // Built in resolver failed.  Let's try to get it from a CDN.
      return {
        // Fixme.  Cases for no CDN, or unknown CDN (i.e. prefix).
        path: cdnUrlHelpers[jsCDN](args.path),
        namespace: 'http-url',
      };
    });

    // Resolve absolute urls.
    build.onResolve({filter: /^https?:\/\//}, args => {
      return {path: args.path, namespace: 'http-url'};
    });

    // Resolve relative http urls into absolute urls.
    build.onResolve({filter: /.*/, namespace: 'http-url'}, args => {
      const path = new URL(args.path, args.importer).toString();

      // Is this a manifoldCAD context import from a remote package?
      // e.g.: `/npm/manifold-3d/manifoldCAD/+esm`
      if (path === cdnUrlHelpers[jsCDN](manifoldCADExportSpecifier)) {
        const response = {path, namespace: 'manifold-replace'};
        return response;
      }

      return {path, namespace: 'http-url'};
    });

    // Inject context.
    build.onLoad(
        {filter: /.*/, namespace: 'manifold-replace'},
        (): esbuild.OnLoadResult => ({
          contents:
              `export const {${manifoldCADExportNames}} = _manifold_context`,
        }));

    // Fetch urls.
    // Fixme.  Should be possible to disable this.
    build.onLoad(
        {filter: /.*/, namespace: 'http-url'},
        async(args): Promise<esbuild.OnLoadResult> => {
          const response = await fetch(args.path);
          if (response.ok) {
            console.log(`Fetching ${args.path}.`);
            return {contents: await response.text()};
          } else {
            return {errors: [{text: await response.text()}]};
          }
        });
  },
});

let esbuild_initialized: boolean = false;
const getEsbuildConfig = async(): Promise<esbuild.BuildOptions> => {
  if (!esbuild_initialized) {
    const options: esbuild.InitializeOptions = {worker: false};
    if (!isNode() && esbuildWasmUrl) {
      options.wasmURL = esbuildWasmUrl;
    }
    await esbuild.initialize(options);
    esbuild_initialized = true;
  }

  return {
    // Create a bundle in memory.
    bundle: true,
    write: false,
    treeShaking: false,  // FIXME ensure that 'result' and dependencies are not
                         // shaken out.
    platform: 'node',
    sourcemap: 'inline',
    sourcesContent: false,  // We have the source handy already.
    format: 'cjs',

    plugins: [
      esbuildManifoldPlugin(),
      textReplace({
        // FIXME how necessary is this?
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
  try {
    const built = await esbuild.build({
      ...(await getEsbuildConfig()),
      entryPoints: [entrypoint],
    });
    return built.outputFiles![0].text;
  } catch (error) {
    if ((error as any).errors?.length) {
      throw new BundlerError(error as BuildFailure);
    } else {
      throw error;
    }
  }
};

export const bundleCode =
    async(code: string, filename?: string): Promise<string> => {
  try {
    const built = await esbuild.build({
      ...(await getEsbuildConfig()),
      stdin: {contents: code, sourcefile: filename}
    });
    return built.outputFiles![0].text;
  } catch (error) {
    if ((error as any).errors?.length) {
      throw new BundlerError(error as BuildFailure);
    } else {
      throw error;
    }
  }
};
