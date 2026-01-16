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
 */

export type * from './manifold-global-types';
export type {MeshOptions, triangulate, setMinCircularAngle, setMinCircularEdgeLength, setCircularSegments, getCircularSegments, resetToCircularDefaults} from './manifold-encapsulated-types';
export {CrossSection, Manifold, Mesh} from './manifold-encapsulated-types';

import type {triangulate, setMinCircularAngle, setMinCircularEdgeLength, setCircularSegments, getCircularSegments, resetToCircularDefaults} from './manifold-encapsulated-types';
import {CrossSection, Manifold, Mesh} from './manifold-encapsulated-types';

export interface ManifoldToplevel {
  CrossSection: typeof CrossSection;
  Manifold: typeof Manifold;
  Mesh: typeof Mesh;
  triangulate: typeof triangulate;
  setMinCircularAngle: typeof setMinCircularAngle;
  setMinCircularEdgeLength: typeof setMinCircularEdgeLength;
  setCircularSegments: typeof setCircularSegments;
  getCircularSegments: typeof getCircularSegments;
  resetToCircularDefaults: typeof resetToCircularDefaults;
  setup: () => void;
}

export default function Module(config?: {locateFile: () => string}):
    Promise<ManifoldToplevel>;
