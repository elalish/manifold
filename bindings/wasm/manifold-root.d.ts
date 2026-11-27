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
 * The core WASM bindings with no frills.
 *
 * @packageDocumentation
 * @primaryExport
 * @group none
 * @see {@link "Using Manifold" | Using Manifold}
 * @see {@link "Manifold Examples" | Manifold Examples}
 * @module manifold
 */

export type * from './manifold-global-types';
export type {MeshOptions, Mesh64Options, MIN_RELATIVE_PRECISION, triangulate, segmentsFor, forDivisions, version} from './manifold-encapsulated-types';
export {CrossSection, Manifold, ManifoldError, Mesh, Mesh64} from './manifold-encapsulated-types';

import type {MIN_RELATIVE_PRECISION, triangulate, segmentsFor, forDivisions, version} from './manifold-encapsulated-types';
import {CrossSection, Manifold, ManifoldError, Mesh, Mesh64} from './manifold-encapsulated-types';

import type {ExecutionContext} from './manifold-global-types';

export interface ManifoldToplevel {
  CrossSection: typeof CrossSection;
  Manifold: typeof Manifold;
  ManifoldError: typeof ManifoldError;
  ExecutionContext: new() => ExecutionContext;
  Mesh: typeof Mesh;
  Mesh64: typeof Mesh64;
  MIN_RELATIVE_PRECISION: typeof MIN_RELATIVE_PRECISION;
  triangulate: typeof triangulate;
  segmentsFor: typeof segmentsFor;
  forDivisions: typeof forDivisions;
  version: typeof version;
  setup: () => void;
}

export default function Module(config?: {locateFile: () => string}):
    Promise<ManifoldToplevel>;
