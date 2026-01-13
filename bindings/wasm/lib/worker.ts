// Copyright 2022-2025 The Manifold Authors.
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
 * The worker is where everything comes together.
 * It handles worker communication, execution of a model and
 * exporting the final scene as a GLTF-Transform Document or URL encoded
 * Blob.
 *
 * This is an isomorphic module that can be used directly as a JavaScript or
 * TypeScript module.  It can be imported as a web worker, and defines
 * a set of interfaces for communication in that case.
 *
 * @packageDocumentation
 * @group manifoldCAD Runtime
 * @category Core
 */

import {Document} from '@gltf-transform/core';

import * as animation from './animation.ts';
import {bundleCode, setHasOwnWorker, setWasmUrl as setEsbuildWasmUrl} from './bundler.ts';
import {RuntimeError} from './error.ts';
import * as exportModel from './export-model.ts';
import * as garbageCollector from './garbage-collector.ts';
import * as gltfNode from './gltf-node.ts'
import * as levelOfDetail from './level-of-detail.ts';
import * as scenebuilder from './scene-builder.ts';
import {getSourceMappedStackTrace, isWebWorker} from './util.ts';
import {getManifoldModule, setWasmUrl as setManifoldWasmUrl} from './wasm.ts';

const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;

export type MessageType =
    'initialize'|'evaluate'|'export'|'ready'|'done'|'error'|'log'|'blob';

export interface Message {
  type: MessageType;
}

export namespace MessageToWorker {
  /**
   * Initialize the web worker.
   * If this message is not sent, the worker will
   * be initialized when first evaluating a model.
   *
   * The worker will respond with `MessageFromWorker.Ready`
   * or `MessageFromWorker.Error`.
   */
  export interface Initialize extends Message {
    type: 'initialize';
    manifoldWasmUrl?: string;
    esbuildWasmUrl?: string;
    esbuildHasOwnWorker?: boolean;
  }

  /**
   * Evaluate a ManifoldCAD script.
   *
   * The worker will respond with `MessageFromWorker.Done`
   * or `MessageFromWorker.Error`.
   *
   * `filename` doesn't have much meaning at the moment, but it
   * is useful for ensuring effective error messages later.
   *
   */
  export interface Evaluate extends Message {
    type: 'evaluate';
    code: string;
    filename?: string;
    doNotBundle?: boolean;
    jsCDN?: string;
    fetchRemotePackages?: boolean;
    files?: Record<string, string>;
    baseUrl?: string;
  }

  /**
   * Export an evaluated model as a URL encoded Blob.
   *
   * The worker will respond with `MessageFromWorker.Blob`
   * or `MessageFromWorker.Error`.
   */
  export interface Export extends Message {
    type: 'export';
    extension: string;
  }
}

export namespace MessageFromWorker {
  /**
   * The worker is instantiated and ready.
   */
  export interface Ready extends Message {
    type: 'ready';
  }

  /**
   * The worker has successfully evaluated a model.
   */
  export interface Done extends Message {
    type: 'done';
  }

  /**
   * The worker has encountered an error.
   *
   * After an error, the state of the worker is undefined.
   * Discard it and re-instantiate.
   *
   * The stack trace format is platform dependant.  Errors
   * should be formatted into something vaguely node-ish.
   */
  export interface Error extends Message {
    type: 'error';
    message: string;
    name?: string;
    stack?: string;
  }

  /**
   * The worker has emitted a log message.
   *
   * This is not an error condition.  Log messages may
   * be sent at any time.
   */
  export interface Log extends Message {
    type: 'log';
    message: string;
  }

  /**
   * The worker has successfully exported a model
   * as a URL encoded Blob.
   */
  export interface Blob extends Message {
    type: 'blob';
    blobURL: string;
    extension: string;
  }
}

// Swallow informational logs in testing framework
function log(...args: any[]) {
  if (typeof self !== 'undefined' && self.console) {
    self.console.log(...args);
  }
}

/**
 * Clean up any state from the last run.
 *
 * This includes any outstanding Manifold, Mesh or CrossSection objects,
 * even if referenced elsewhere.
 */
export function cleanup(): void {
  garbageCollector.cleanup();
  scenebuilder.cleanup();
  levelOfDetail.cleanup();
  gltfNode.cleanup();
}

/**
 * If `bundle` is set to false, the worker will not bundle code,
 * and assume that it has already been bundled.  An undefined
 * value will be treated by default as true.
 */
export interface evaluateOptions {
  filename?: string;
  doNotBundle?: boolean;
  jsCDN?: string;
  fetchRemotePackages?: boolean;
  baseUrl?: string;
}

/**
 * Transform a model from code to a GLTF document.
 *
 * @param code A string containing the code to evaluate.
 * @returns A gltf-transform Document.
 */
export async function evaluate(
    code: string, options: evaluateOptions = {}): Promise<Document> {
  cleanup();
  const t0 = performance.now();

  const {doNotBundle, ...bundleOpt} = options;
  const bundled =
      doNotBundle === true ? code : await bundleCode(code, bundleOpt);

  const t1 = performance.now();
  if (doNotBundle !== true) {
    log(`Bundling code took ${((t1 - t0) / 1000).toFixed(2)} seconds`);
  }

  // Customize `manifold-3d/manifoldCAD`.
  // Let models know that they are running inside this worker.
  const manifoldCAD = await import('./manifoldCAD.ts');
  const manifoldImport = {
    ...manifoldCAD,
    isManifoldCAD: () => true,
  }

  // If a top level script imports manifoldCAD, track GLTF nodes.
  // Libraries are expected to manage this on their own; if a
  // library somehow doesn't export a GLTF node, it must not
  // show up here.
  const toplevelImport = {
    ...manifoldImport,
    GLTFNode: gltfNode.GLTFNodeTracked,
    getGLTFNodes: gltfNode.getGLTFNodes,
    resetGLTFNodes: gltfNode.resetGLTFNodes,
  };

  // Set up global variables exposed to the model without an import.
  const globals = {
    // These accessors are only available to top level scripts.
    // See ../lib/manifoldCADGlobals.d.ts
    setCircularSegments: levelOfDetail.setCircularSegments,
    setMinCircularAngle: levelOfDetail.setMinCircularAngle,
    setMinCircularEdgeLength: levelOfDetail.setMinCircularEdgeLength,
    resetToCircularDefaults: levelOfDetail.resetToCircularDefaults,
    setAnimationDuration: animation.setAnimationDuration,
    setAnimationFPS: animation.setAnimationFPS,
    setAnimationMode: animation.setAnimationMode,

    // The bundler will swap these objects in when needed.
    _manifold_cad_top_level: toplevelImport,
    _manifold_cad_library: manifoldImport,

    // Bundled code may be referencing files by relative paths.
    // Set runtime value of import.meta.url
    _manifold_runtime_url: options.baseUrl ?? null,

    // While this project is built using ES modules, and we assume models and
    // libraries are ES modules, code executed via `new Function()` or `eval` is
    // treated as commonJS.
    // CommonJS expects 'exports' to exist:
    exports: {},
    // This is where we expect results after running the script.
    module: {exports: {default: null}},
  };

  let result: any = null;
  try {
    const evalFn = new AsyncFunction(...Object.keys(globals), bundled);
    await evalFn(...Object.values(globals));

    result = globals.module?.exports?.default;
    // If the default export is a function, execute it.  This way libraries can
    // preview their results when run as a top level script, without incurring
    // that overhead when imported into another model.
    if (typeof result === 'function') {
      result = await result();
    }
  } catch (error: any) {
    // "According to step 12 of
    // https://tc39.es/ecma262/#sec-createdynamicfunction, the Function
    // constructor always prefixes the source with additional 2 lines."
    // https://github.com/nodejs/node/issues/43047#issuecomment-1564068099
    const stacktrace = getSourceMappedStackTrace(bundled, error, -2);
    let newError: RuntimeError|null = null;

    const missing =
        Object.keys(toplevelImport).find((x: string) => error.message.match(x));
    if (error.name === 'ReferenceError' && missing) {
      newError = new RuntimeError(
          error,
          error.message + '.  Import it by adding \`' +
              `import {${missing}} from 'manifold-3d/manifoldCAD';` +
              '\` to the top of your model.');
    } else if (
        error.name === 'ReferenceError' && error.message.match(/glMatrix/)) {
      newError = new RuntimeError(
          error,
          'ManifoldCAD no longer includes gl-matrix directly.  ' +
              'Import it by adding `import * as glMatrix from \'gl-matrix\';` ' +
              'to the top of your model.');
    } else {
      newError = new RuntimeError(error);
    }
    newError.manifoldStack = stacktrace;
    throw newError;
  }

  // If we don't actually have a model, complain.
  if (!result || (Array.isArray(result) && !result.length)) {
    if (gltfNode.getGLTFNodes().length) {
      throw new Error(
          'GLTF Nodes were created, but not exported.  ' +
          'Add `const nodes = getGLTFNodes();` and `export default nodes;` ' +
          'to the end of your model.');
    }
    throw new Error(
        'No output as no model was exported.  Add a default export ' +
        '(e.g.: `export default result;`) to the bottom of your model.  ' +
        'The default export must be a `Manifold` or `GLTFNode` object, ' +
        'an array of `Manifold` or `GLTFNode` objects, ' +
        'or a function that returns any of the above.');
  }

  // Create a gltf-transform document.
  const nodes = await gltfNode.anyToGLTFNodeList(result);
  const doc = await scenebuilder.GLTFNodesToGLTFDoc(nodes);
  const t2 = performance.now();
  log(`Manifold took ${((t2 - t1) / 1000).toFixed(2)} seconds`);

  return doc;
}

/**
 * Convert an in-memory GLTF document to a URL encoded blob.
 *
 * @param doc The GLTF document.
 * @param extension The target file extension.
 * @returns A URL encoded blob.
 */
export const exportBlobURL =
    async(doc: Document, extension: string): Promise<string> => {
  const t0 = performance.now();

  const blob = await exportModel.toBlob(doc, {extension});
  const blobURL = URL.createObjectURL(blob);

  const t1 = performance.now();
  log(`Exporting ${extension.toUpperCase()} took ${
      (Math.round((t1 - t0) / 10) / 100).toLocaleString()} seconds`);
  return blobURL;
};

/**
 * Set up message handlers and logging when run as a web worker.
 */
const initializeWebWorker = (): void => {
  const interceptConsole = () => {
    console.debug('Intercepting console.log() in manifoldCAD worker.');
    if (self.console) {
      const oldLog = self.console.log;
      self.console.log = function(...args) {
        let message = '';
        for (const arg of args) {
          if (arg == null) {
            message += 'undefined';
          } else if (typeof arg == 'object') {
            message += JSON.stringify(arg, null, 4);
          } else {
            message += arg.toString();
          }
        }
        self.postMessage({type: 'log', message} as MessageFromWorker.Log);
        oldLog(...args);
      };
    };
  };

  const sendError = (error: Error) => {
    // Log the error / stack trace to the console.
    console.error(error);
    if (error.cause) console.error('Caused by:', error.cause);
    if ((error as RuntimeError).manifoldStack)
      console.error('manifoldStack:', (error as RuntimeError).manifoldStack);

    self.postMessage({
      type: 'error',
      name: error.name,
      message: error.message,
      stack: (error as any).manifoldStack ?? error.stack
    } as MessageFromWorker.Error);
  };

  const handleInitialize = async (message: MessageToWorker.Initialize) => {
    try {
      console.debug('Initializing ManifoldCAD worker.');
      if (message.manifoldWasmUrl) setManifoldWasmUrl(message.manifoldWasmUrl);
      if (message.esbuildWasmUrl) setEsbuildWasmUrl(message.esbuildWasmUrl);
      setHasOwnWorker(message.esbuildHasOwnWorker === true);

      await getManifoldModule();
      interceptConsole();

      self.postMessage({type: 'ready'} as MessageFromWorker.Ready);
      console.debug('Successfully initialized ManifoldCAD worker!');
    } catch (error) {
      sendError(error as Error);
    }
  };

  let gltfdoc: Document|null = null;

  const handleEvaluate = async (message: MessageToWorker.Evaluate) => {
    try {
      const {code, ...options} = message;
      gltfdoc = await evaluate(message.code, options as evaluateOptions);
      self.postMessage({type: 'done'} as MessageFromWorker.Done);
    } catch (error) {
      sendError(error as Error);
    }
  };

  const handleExport = async (message: MessageToWorker.Export) => {
    try {
      self.postMessage({
        type: 'blob',
        extension: message.extension,
        blobURL: await exportBlobURL(gltfdoc!, message.extension)
      } as MessageFromWorker.Blob);
    } catch (error) {
      sendError(error as Error);
    }
  };

  self.onmessage = async (e) => {
    const message = e.data as Message;
    if (message.type === 'initialize') {
      handleInitialize(message as MessageToWorker.Initialize);
    } else if (message.type === 'evaluate') {
      handleEvaluate(message as MessageToWorker.Evaluate);
    } else if (message.type === 'export') {
      handleExport(message as MessageToWorker.Export)
    }
  }
};
if (isWebWorker()) initializeWebWorker();
