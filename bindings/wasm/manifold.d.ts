// Copyright 2022 The Manifold Authors.
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

import * as T from './manifold-encapsulated-types';
export * from './manifold-global-types';

export type CrossSection = T.CrossSection;
export type Manifold = T.Manifold;
export type Mesh = T.Mesh;

export interface ManifoldToplevel {
  CrossSection: typeof T.CrossSection;
  Manifold: typeof T.Manifold;
  Mesh: typeof T.Mesh;
  triangulate: typeof T.triangulate;
  setMinCircularAngle: typeof T.setMinCircularAngle;
  setMinCircularEdgeLength: typeof T.setMinCircularEdgeLength;
  setCircularSegments: typeof T.setCircularSegments;
  getCircularSegments: typeof T.getCircularSegments;
  setup: () => void;
}

export default function Module(config?: {locateFile: () => string}):
    Promise<ManifoldToplevel>;
