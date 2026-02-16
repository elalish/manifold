// Copyright 2025 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * The bundler resolves files and modules at load time, bundles them, and allows
 * the manifoldCAD runtime to provide objects (like manifold itself) and
 * properties (such as `import.meta.url`).
 * @packageDocumentation
 * @group ManifoldCAD
 * @category Core
 */

import resolve from '@jridgewell/resolve-uri';
import * as esbuild from 'esbuild-wasm';
import MagicString from 'magic-string';

import {BundlerError} from './error.ts';
import {isNode} from './util.ts';

let esbuildWasmUrl: string|null = null;
let esbuildHasOwnWorker: boolean = false;

export const setWasmUrl = (url: string) => {
  esbuildWasmUrl = url;
};

export const setHasOwnWorker = (x: boolean) => {
  esbuildHasOwnWorker = x;
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

const cdnUrl = (specifier: string, jsCDN?: string) => {
  if (!jsCDN) return specifier;
  const helper = cdnUrlHelpers[jsCDN];
  return helper ? helper(specifier) : `${jsCDN}${specifier}`;
};

export interface BundlerOptions {
  jsCDN?: string;
  fetchRemotePackages?: boolean;
  filename?: string;
  files?: Record<string, string>;
  resolveDir?: string;
}

// Swallow informational logs in testing framework
function log(...args: any[]) {
  if (typeof self !== 'undefined' && self.console) {
    self.console.log(...args);
  }
}

const insertMetaData =
    (text: string, sourceUrl?: string) => {
      if (sourceUrl) {
        const st = new MagicString(text);
        st.prepend(`const _import_meta_url=_manifold_runtime_url ?? '${
            sourceUrl}';\n`);

        const map = st.generateMap({hires: true});
        return `${st.toString()}\n//# sourceMappingURL=${map.toUrl()}`;
      };
      return text;
    }

/**
 * This is a plugin for esbuild that has three functions:
 *   * It resolves NPM packages to urls served by various CDNs.
 *   * It fetches imports from http/https urls.
 *   * It provides evaluation context to npm packages.
 *
 * ManifoldCAD libraries can include `manifold-3d/manifoldCAD` to get access to
 * the evaluator context.  Outside of the evaluator, this involves importing a
 * module.  Inside the evaluator, imports are in the global namespace.  The
 * imported module will not be able to see the actual evaluator context.  Here,
 * we will swap it out for an object we will inject at evaluation time.
 */
export const esbuildManifoldPlugin = (options: BundlerOptions = {}):
                                         esbuild.Plugin => ({
  name: 'esbuild-manifold-plugin',
  async setup(build) {
    let manifoldCADExportPath: string|null = null;
    const manifoldCADExportSpecifier = 'manifold-3d/manifoldCAD'
    const ManifoldCADExportMatch = /^manifold-3d\/manifoldCAD(.ts|.js)?$/
    const manifoldCADExportNames = [
      // Manifold classes.
      'Mesh', 'Manifold', 'CrossSection',
      // Manifold methods.
      'triangulate',

      // Scene builder exports.
      'show', 'only', 'setMaterial',
      // GLTFNode and utilities.
      'GLTFMaterial', 'GLTFNode', 'getGLTFNodes', 'VisualizationGLTFNode',
      'CrossSectionGLTFNode',
      // Import
      'importModel', 'importManifold',
      // Getters for global properties
      'getCircularSegments', 'getMinCircularAngle', 'getMinCircularEdgeLength',
      'getAnimationDuration', 'getAnimationFPS', 'getAnimationMode',
      // Setters for global properties.
      // These will only be defined for top level scripts
      'setMinCircularAngle', 'setMinCircularEdgeLength', 'setCircularSegments',
      'resetToCircularDefaults', 'setMorphStart', 'setMorphEnd',
      'setAnimationDuration', 'setAnimationFPS', 'setAnimationMode',
      'resetGLTFNodes',

      // ManifoldCAD specific exports.
      'isManifoldCAD'
    ];

    if (isNode()) {
      // We only need to check against the local manifoldCAD context on disk if
      // we happen to be running in node.  Truthfully, this is really only
      // necessary in development.  End users will almost always import
      // `manifold-3d/manifoldCAD` instead of `some/path/to/manifoldCAD`.
      (async () => {
        const {resolve, dirname} = await import('node:path');
        const {fileURLToPath} = await import('node:url');
        const dir = ('string' == typeof __dirname && __dirname) ||
            ('string' == typeof import.meta?.dirname && import.meta.dirname) ||
            dirname(fileURLToPath(import.meta.url));
        manifoldCADExportPath = resolve(dir, './manifoldCAD.ts');
      })();
    }

    let entrypoint: string|null = null;
    // Try to resolve local files.  If we can't, blindly resolve them to a CDN.
    build.onResolve({filter: /.*/}, async (args) => {
      // Avoid loops.
      if (args.pluginData !== undefined) return null;

      // Skip a few cases handled elsewhere.
      if (args.namespace === 'http-url') return null;
      if (args.path.match(/^https?:\/\//)) return null;

      if (!entrypoint && args.kind === 'entry-point') {
        entrypoint = resolve(args.path, args.resolveDir + '/');
      }

      // Is this a manifoldCAD context import?
      // FIXME resolve path here too.
      const pluginData = {
        toplevel: args.importer === entrypoint ||
            args.importer === options.filename || args.importer === '<stdin>',
      };
      if (args.path.match(ManifoldCADExportMatch)) {
        return {namespace: 'manifold-cad-globals', path: args.path, pluginData};
      }

      // Is this a virtual file?
      // FIXME Resolve paths!
      if (options.files && Object.keys(options.files).includes(args.path)) {
        return {namespace: 'virtual-file', path: args.path};
      }

      // Try esbuilds' resolver first.
      const result = await build.resolve(args.path, {
        resolveDir: args.resolveDir,
        kind: 'import-statement',
        pluginData: {resolveDir: options.resolveDir}
      });

      // We found a local file!
      if (result.errors.length === 0) {
        if (manifoldCADExportPath && manifoldCADExportPath === result.path) {
          // It resolved to our local manifoldCAD context.
          return {
            namespace: 'manifold-cad-globals',
            path: args.path,
            pluginData
          };
        }
        result.pluginData = {
          resolveDir: args.resolveDir.endsWith('/') ? args.resolveDir :
                                                      `${args.resolveDir}/`
        };
        return result;
      }

      // Built in resolver failed.  Are we fetching remote packages?
      if (options.fetchRemotePackages !== false && options.jsCDN) {
        return {
          path: cdnUrl(args.path, options.jsCDN),
          namespace: 'http-url',
        };
      }

      // Okay fine.  I give up.
      return null;
    });

    // Instead of loading manifoldCAD.ts, insert an instantiated copy.
    // The global variables enabling this are set by the worker.
    build.onLoad(
        {filter: /.*/, namespace: 'manifold-cad-globals'},
        (args): esbuild.OnLoadResult => {
          // This is a string replace.
          const globals = args.pluginData?.toplevel ?
              '_manifold_cad_top_level' :
              '_manifold_cad_library';
          return {
            // Type hinting isn't necessary.  Only esbuild will see the swap,
            // and it doesn't do type validation.
            contents: `export const {${manifoldCADExportNames}} = ${globals};`,
          };
        });

    // Unless disabled, handle HTTP/HTTPs urls.
    if (options.fetchRemotePackages !== false) {
      // Resolve absolute urls.
      build.onResolve({filter: /^https?:\/\//}, args => {
        return {path: args.path, namespace: 'http-url'};
      });

      // Resolve relative http urls into absolute urls.
      build.onResolve({filter: /.*/, namespace: 'http-url'}, args => {
        const path = new URL(args.path, args.importer).toString();

        // Is this a manifoldCAD context import from a remote package?
        // e.g.: `/npm/manifold-3d/manifoldCAD/+esm`
        if (path === cdnUrl(manifoldCADExportSpecifier, options.jsCDN)) {
          const response = {
            path,
            namespace: 'manifold-cad-globals',
            pluginData: {toplevel: false}
          };
          return response;
        }

        return {path, namespace: 'http-url'};
      });

      // Fetch urls.
      build.onLoad({filter: /.*/, namespace: 'http-url'}, async (args) => {
        const response = await fetch(args.path);
        if (response.ok) {
          log(`Fetching ${args.path}.`);
          return {contents: await response.text()};
        } else {
          return {errors: [{text: await response.text()}]};
        }
      });
    }

    // Virtual files.
    build.onLoad(
        {filter: /.*/, namespace: 'virtual-file'},
        (args): esbuild.OnLoadResult => {
          const text = (options.files!)[args.path];

          const contents = insertMetaData(text, `file://${args.path}`);
          const loader = (args.path.match(/\.js$/)) ? 'js' : 'ts';
          return {contents, loader};
        });

    // Finally, local files.
    build.onLoad(
        {filter: /.(ts|js)$/}, async(args): Promise<esbuild.OnLoadResult> => {
          const fs = await import('node:fs/promises');
          const text = await fs.readFile(args.path, 'utf8');

          const contents = insertMetaData(text, `file://${args.path}`);
          const loader = (args.path.match(/\.js$/)) ? 'js' : 'ts';
          return {contents, loader};
        });
  },
});

let esbuild_initialized: boolean = false;
const getEsbuildConfig =
    async(options: BundlerOptions = {}): Promise<esbuild.BuildOptions> => {
  if (!esbuild_initialized) {
    const esbuildOptions: esbuild.InitializeOptions = {};
    if (typeof esbuildWasmUrl === 'string' && esbuildWasmUrl) {
      esbuildOptions.wasmURL = esbuildWasmUrl;
      esbuildOptions.worker = esbuildHasOwnWorker === true;
    }
    await esbuild.initialize(esbuildOptions);
    esbuild_initialized = true;
  }

  return {
    // Create a bundle in memory.
    bundle: true,
    write: false,
    platform: 'neutral',
    treeShaking: false,
    sourcemap: 'inline',
    sourcesContent: false,  // We have the source handy already.
    format: 'cjs',
    logLevel: 'silent',
    plugins: [
      esbuildManifoldPlugin(options),
    ],
    // Some CDN imports will check import.meta.env.  This is only present when
    // generating an ESM bundle.  In other cases, it generates log noise, so
    // let's drop it down a log level.
    logOverride: {'empty-import-meta': 'info'},

    // Define some paths so we can find resources by relative URL.
    define: {
      'import.meta.url': '_import_meta_url',
    }
  };
};

export const bundleFile = async(
    entrypoint: string, options: BundlerOptions = {}): Promise<string> => {
  try {
    const built = await esbuild.build({
      ...(await getEsbuildConfig({...options, filename: entrypoint})),
      entryPoints: [entrypoint],
    });
    return built.outputFiles![0].text;
  } catch (error) {
    if ((error as any).errors?.length) {
      throw new BundlerError(error as esbuild.BuildFailure);
    } else {
      throw error;
    }
  }
};

export const bundleCode =
    async(code: string, options: BundlerOptions = {}): Promise<string> => {
  try {
    let resolveDir: string|undefined;
    if (isNode() && options.filename) {
      const {dirname} = await import('node:path');
      resolveDir = options.resolveDir ?? dirname(options.filename);
    }
    const built = await esbuild.build({
      ...(await getEsbuildConfig(options)),
      stdin: {
        contents: insertMetaData(code, `file://${options.filename}`),
        sourcefile: options.filename,
        resolveDir,
        loader: 'ts',
      }
    });
    return built.outputFiles![0].text;
  } catch (error) {
    if ((error as any).errors?.length) {
      throw new BundlerError(error as esbuild.BuildFailure);
    } else {
      throw error;
    }
  }
};
