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

export type Manifold = T.Manifold;
export type Mesh = T.Mesh;

export interface ManifoldStatic {
  cube: typeof T.cube;
  cylinder: typeof T.cylinder;
  sphere: typeof T.sphere;
  smooth: typeof T.smooth;
  tetrahedron: typeof T.tetrahedron;
  extrude: typeof T.extrude;
  revolve: typeof T.revolve;
  union: typeof T.union;
  difference: typeof T.difference;
  intersection: typeof T.intersection;
  compose: typeof T.compose;
  levelSet: typeof T.levelSet;
  setMinCircularAngle: typeof T.setMinCircularAngle;
  setMinCircularEdgeLength: typeof T.setMinCircularEdgeLength;
  setCircularSegments: typeof T.setCircularSegments;
  getCircularSegments: typeof T.getCircularSegments;
  reserveIDs: typeof T.reserveIDs;
  Manifold: typeof T.Manifold;
  Mesh: typeof T.Mesh;
  setup: () => void;
}

export default function Module(config: {locateFile: () => string}): Promise<ManifoldStatic>;
