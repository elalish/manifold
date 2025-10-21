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
 * These are the objects and functions that are available to manifoldCAD
 * scripts.
 *
 * This is an isomorphic module.  When imported within manifoldCAD, the bundler
 * will swap it out for an identical module built from the evaluator context.
 * When imported as an ES module, it will implicitly instantiate a worker with
 * an evaluator, and export that context.  This allows modules to behave
 * identically when running on manifoldCAD.org, through the CLI, or through
 * nodejs.
 *
 * @module manifoldCAD
 */

import type * as encapsulatedTypes from '../manifold-encapsulated-types.d.ts';
import type * as manifoldCADTypes from '../types/manifoldCAD.d.ts';

import {initialize} from './worker.ts';

const evaluator = initialize();
const context = await evaluator.getFullContext();

export const Manifold: encapsulatedTypes.Manifold = context.Manifold;
export const CrossSection: encapsulatedTypes.CrossSection =
    context.CrossSection;
export const Mesh: encapsulatedTypes.Mesh = context.Mesh;

export const triangulate: typeof encapsulatedTypes.triangulate =
    context.triangulate;
export const setMinCircularAngle: typeof encapsulatedTypes.setMinCircularAngle =
    context.setMinCircularAngle;
export const setMinCircularEdgeLength:
    typeof encapsulatedTypes.setMinCircularEdgeLength =
    context.setMinCircularEdgeLength;
export const setCircularSegments: typeof encapsulatedTypes.setCircularSegments =
    context.setCircularSegments;
export const getCircularSegments: typeof encapsulatedTypes.getCircularSegments =
    context.getCircularSegments;
export const resetToCircularDefaults:
    typeof encapsulatedTypes.resetToCircularDefaults =
    context.resetToCircularDefaults;

export const show: typeof manifoldCADTypes.show = context.show;
export const only: typeof manifoldCADTypes.only = context.only;
export const setMaterial: typeof manifoldCADTypes.setMaterial =
    context.setMaterial;
export const setMorphStart: typeof manifoldCADTypes.setMorphStart =
    context.setMorphStart;
export const setMorphEnd: typeof manifoldCADTypes.setMorphEnd =
    context.setMorphEnd;
export const GLTFNode: manifoldCADTypes.GLTFNode = context.GLTFNode;

export const isManifoldCAD: typeof manifoldCADTypes.isManifoldCAD = () => false;