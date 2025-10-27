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

import type {ManifoldToplevel} from '../manifold.d.ts';
import Module from '../manifold.js';

import {isNode} from './util.ts';

// Instantiate Manifold WASM
let manifoldwasm: ManifoldToplevel|null = null;
let wasmUrl: string|null = null;

/**
 * Tell us how to find `manifold.wasm`.
 * This is important when using a bundler like WebPack.
 *
 * @param url Path to manifold.wasm
 */
export function setWasmUrl(url: string) {
  wasmUrl = url;
}

/**
 * Instantiate a new Manifold WASM instance.
 *
 * @returns The newly created instance.
 */
export async function instantiateManifold(): Promise<ManifoldToplevel> {
  let module: ManifoldToplevel|null = null;
  if (!isNode()) {
    if (typeof wasmUrl !== 'string' || !wasmUrl) {
      throw new Error('No URL given for \'manifold.wasm\'.');
    }
    module = await Module({locateFile : () => wasmUrl!});
  } else {
    module = await Module();
  }
  module.setup();
  return module;
}

/**
 * Instantiate or get a global Manifold WASM instance.
 *
 * @returns A manifold instance.
 */
export async function getManifoldModule(): Promise<ManifoldToplevel> {
  if (!manifoldwasm) manifoldwasm = await instantiateManifold();
  return manifoldwasm;
}
