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
 * The worker is where the scene builder and evaluator come together.
 * It handles instantiation of an evaluator, execution of a model and
 * exporting the final scene as a GLTF-Transform Document or URL encoded
 * Blob.
 *
 * This is a polymorphic module that can be used directly as a JavaScript or
 * TypeScript module.  It can be imported as a web worker, and defines
 * a set of interfaces for communication in that case.
 *
 * @module worker
 */

import {Document} from '@gltf-transform/core';
import * as glMatrix from 'gl-matrix';

import {Evaluator} from './evaluate.ts';
import {Export3MF} from './export-3mf.ts';
import {ExportGLTF} from './export-gltf.ts';
import * as scenebuilder from './scene-builder.ts';
import {GlobalDefaults} from './scene-builder.ts';
import {isWebWorker, transformStaticImportsToDynamic} from './util.ts';
import {getManifoldModule, setWasmUrl} from './wasm.ts';

let evaluator: Evaluator|null = null;
let exporters: Array<any>;

type MessageType =
    'initialize'|'evaluate'|'export'|'ready'|'done'|'error'|'log'|'blob'

export interface Message {
  type: MessageType;
}

export namespace MessageToWorker {
  /**
   * Initialize the web worker.
   * If this message is not sent, the worker will
   * be instantiated when first evaluating a model.
   *
   * The worker will respond with `MessageFromWorker.Ready`
   * or `MessageFromWorker.Error`.
   */
  export interface Initialize extends Message {
    type: 'initialize';
    manifoldWasmUrl?: string;
    remotePackagePrefix?: string;
  }

  /**
   * Evaluate a ManifoldCAD script.
   *
   * The worker will respond with `MessageFromWorker.Done`
   * or `MessageFromWorker.Error`.
   */
  export interface Evaluate extends Message {
    type: 'evaluate';
    code: string;
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
   */
  export interface Error extends Message {
    type: 'error';
    message: string;
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
 * Set up the evaluator, as well as any objects the worker may require.
 *
 * This is where the scene builder meets the evaluator.
 * @returns The evaluator.
 */
export function initialize(): Evaluator {
  if (evaluator)
    return evaluator;
  else
    evaluator = new Evaluator();

  // Exporters.
  // The end user can download either.
  // GLB (Binary GLTF) is used to send the model from this worker
  // to the viewer.
  exporters = [new Export3MF(), new ExportGLTF()]

  // Faster on modern browsers than Float32Array
  glMatrix.glMatrix.setMatrixArrayType(Array);
  evaluator.addContext({glMatrix});

  // These are methods that generate Manifold
  // or CrossSection objects.  Tell the evaluator to intercept
  // the calls, and add any created objects to the clean up list.
  evaluator.addContextMethodWithCleanup('show', scenebuilder.show)
  evaluator.addContextMethodWithCleanup('only', scenebuilder.only)
  evaluator.addContextMethodWithCleanup('setMaterial', scenebuilder.setMaterial)

  // Add additional context.  These need no garbage collection.
  evaluator.addContext({
    GLTFNode: scenebuilder.GLTFNode,
    setMorphStart: scenebuilder.setMorphStart,
    setMorphEnd: scenebuilder.setMorphEnd
  });

  return evaluator;
}

/**
 * Clean up any state stored in the evaluator or scene builder.
 *
 * This includes any outstanding Manifold, Mesh or CrossSection objects,
 * even if referenced elsewhere.
 */
export function cleanup(): void {
  evaluator?.cleanup();
  scenebuilder?.cleanup();
}

/**
 * Transform a model from code to a GLTF document.
 *
 * @param code A string containing the code to evaluate.
 * @returns A gltf-transform Document.
 */
export async function evaluate(code: string): Promise<Document> {
  if (!evaluator) initialize();

  // Global defaults can be populated by the script.  It's set per
  // evaluation, while the rest of evaluator context doesn't change from
  // run to run.
  // This can be used to set parameters elsewhere in ManifoldCAD.  For
  // example, the GLTF exporter will look for animation type and
  // framerate.
  const globalDefaults = {} as GlobalDefaults;
  evaluator!.context.globalDefaults = globalDefaults;

  const t0 = performance.now();
  const manifold = await evaluator!.evaluate(code);
  const t1 = performance.now();

  log(`Manifold took ${
      (Math.round((t1 - t0) / 10) / 100).toLocaleString()} seconds`);

  // If we don't actually have a model, complain.
  if (!manifold && !scenebuilder.hasGLTFNodes()) {
    throw new Error(
        'No output because "result" is undefined and no GLTF nodes were created.');
  }

  // Create a gltf-transform document.
  const doc = scenebuilder.hasGLTFNodes() ?
      scenebuilder.GLTFNodesToGLTFDoc(
          scenebuilder.getGLTFNodes(), globalDefaults) :
      scenebuilder.manifoldToGLTFDoc(manifold, globalDefaults);

  const t2 = performance.now();
  log(`Creating GLTF Document took ${
      (Math.round((t2 - t1) / 10) / 100).toLocaleString()} seconds`);

  return doc;
}

/**
 * Convert an in-memory GLTF document to a URL encoded blob.
 *
 * @param doc The GLTF document.
 * @param extension The target file extension.
 * @returns A URL encoded blob.
 */
export const exportBlobURL = async(doc: Document, extension: string):
    Promise<string> => {
      const t0 = performance.now();

      const blob = await exporters.find(ex => ex.extensions.includes(extension))
                       .asBlob(doc)
      const blobURL = URL.createObjectURL(blob);

      const t1 = performance.now();
      log(`Exporting ${extension.toUpperCase()} took ${
          (Math.round((t1 - t0) / 10) / 100).toLocaleString()} seconds`);
      return blobURL;
    }

/**
 * Set up message handlers and logging when run as a web worker.
 */
const initializeWebWorker = (): void => {
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

  let remotePackagePrefix = 'https://esm.run/';

  const handleInitialize = async (message: MessageToWorker.Initialize) => {
    try {
      if (message.manifoldWasmUrl) setWasmUrl(message.manifoldWasmUrl);
      if (message.remotePackagePrefix) {
        remotePackagePrefix = message.remotePackagePrefix;
      }

      initialize();
      await getManifoldModule();
      self.postMessage({type: 'ready'} as MessageFromWorker.Ready);

    } catch (error: any) {
      console.error('Error initializing web worker', error);
      console.log(error.toString());

      self.postMessage(
          {type: 'error', message: error.toString()} as
          MessageFromWorker.Error);
    }
  };

  let gltfdoc: Document|null = null;

  const handleEvaluate = async (message: MessageToWorker.Evaluate) => {
    try {
      const code =
          transformStaticImportsToDynamic(message.code, remotePackagePrefix);
      gltfdoc = await evaluate(code);
      self.postMessage({type: 'done'} as MessageFromWorker.Done);
    } catch (error: any) {
      console.error('Worker caught error', error);
      self.postMessage(
          {type: 'error', message: error.toString()} as
          MessageFromWorker.Error);
    } finally {
      cleanup();
    }
  };

  const handleExport = async (message: MessageToWorker.Export) => {
    const blobURL = await exportBlobURL(gltfdoc!, message.extension);
    self.postMessage(
        {type: 'blob', extension: message.extension, blobURL} as
        MessageFromWorker.Blob);
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
