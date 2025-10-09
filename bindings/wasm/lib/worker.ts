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

import {Document} from '@gltf-transform/core';
import * as glMatrix from 'gl-matrix';

import {Evaluator} from './evaluate';
import {Export3MF} from './export-3mf';
import {ExportGLTF} from './export-gltf';
import * as scenebuilder from './scene-builder';
import {GlobalDefaults} from './scene-builder';
import {getManifoldModule, setWasmUrl} from './wasm';

let evaluator: Evaluator|null = null;
let exporters: Array<any>;

type MessageType =
    'initialize'|'evaluate'|'export'|'ready'|'done'|'error'|'log'|'blob'

export interface Message {
  type: MessageType;
}

export type MessageToWorker = Message;
export namespace MessageToWorker {
  export interface Initialize extends Message {
    type: 'initialize';
    wasmUrl?: string;
  }

  export interface Evaluate extends Message {
    type: 'evaluate';
    code: string;
    filename?: string;
  }

  export interface Export extends Message {
    type: 'export';
    extension: string;
  }
}

export type MessageFromWorker = Message;
export namespace MessageFromWorker {
  export interface Ready extends Message {
    type: 'ready';
  }

  export interface Done extends Message {
    type: 'done';
  }

  export interface Error extends Message {
    type: 'error';
    message: string;
  }

  export interface Log extends Message {
    type: 'log';
    message: string;
  }

  export interface Blob extends Message {
    type: 'blob';
    blobURL: string;
    extension: string;
    filename?: string;
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
export function initialize() {
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
export function cleanup() {
  evaluator?.cleanup();
  scenebuilder?.cleanup();
}

/**
 * Transform a model from code to a GLTF document.
 *
 * @param code A string containing the code to evaluate.
 * @returns A gltf-transform Document.
 */
export async function evaluate(code: string) {
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
export const exportBlobURL =
    async (doc: Document, extension: string) => {
  const t0 = performance.now();

  const blob =
      await exporters.find(ex => ex.extensions.includes(extension)).asBlob(doc)
  const blobURL = URL.createObjectURL(blob);

  const t1 = performance.now();
  log(`Exporting ${extension.toUpperCase()} took ${
      (Math.round((t1 - t0) / 10) / 100).toLocaleString()} seconds`);
  return blobURL;
}

/**
 * This module is polymorphic.  If it's run as a web worker,
 * set up message handlers and logging.
 */
const initializeWebWorker =
    function() {
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

  const handleInitialize = async (message: MessageToWorker.Initialize) => {
    try {
      if (message.wasmUrl) setWasmUrl(message.wasmUrl);
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
      gltfdoc = await evaluate(message.code);
      self.postMessage({type: 'done'} as MessageFromWorker.Done);

    } catch (error: any) {
      console.error('Worker caught error', error);
      console.log(error.toString());

      self.postMessage(
          {type: 'error', message: error.toString()} as
          MessageFromWorker.Error);
    } finally {
      cleanup();
    }
  };

  const handleExport = async (message: MessageToWorker.Export) => {
    const blobURL = await exportBlobURL(gltfdoc!, message.extension);

    self.postMessage({
      type: 'blob',
      extension: message.extension,
      filename: `manifold${message.extension}`,
      blobURL
    } as MessageFromWorker.Blob);
  };

  self.onmessage = async function(e) {
    const message = e.data as Message;

    if (message.type === 'initialize') {
      handleInitialize(message as MessageToWorker.Initialize);
    } else if (message.type === 'evaluate') {
      handleEvaluate(message as MessageToWorker.Evaluate);
    } else if (message.type === 'export') {
      handleExport(message as MessageToWorker.Export)
    }
  };
}

const isWebWorker = () =>
    typeof self !== 'undefined' && typeof self.document == 'undefined';
if (isWebWorker()) initializeWebWorker();
