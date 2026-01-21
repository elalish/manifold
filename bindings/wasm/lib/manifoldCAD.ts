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
 * All of the classes, functions and properties of this module are implemented
 * elsewhere and re-exported here.
 *
 * This is an isomorphic module.  When imported within manifoldCAD, the bundler
 * will swap it out for an identical module running in the worker context.
 * When imported as an ES module, it will implicitly instantiate a manifold wasm
 * module, and export it along with everything else listed here.
 * This allows models to behave identically when running on manifoldCAD.org,
 * through the CLI, or through nodejs.
 *
 * There is separate user facing documentation for this module;
 * `types/manifoldCAD.d.ts` which is rolled up into `dist/manifoldCAD.d.ts`.
 *
 * @see {@link https://manifoldcad.org/docs/jsuser/index.html | ManifoldCAD User Guide}
 * @see {@link https://manifoldcad.org/docs/jsapi/documents/Contributing.html#writing-for-the-user-guide | Writing for the User Guide}
 *
 * @packageDocumentation
 * @group ManifoldCAD
 * @category none
 * @module manifoldCAD
 */

import type {ManifoldToplevel} from '../manifold.d.ts';

import * as debug from './debug.ts';
import {garbageCollectFunction, garbageCollectManifold} from './garbage-collector.ts';
import * as material from './material.ts';
import {getManifoldModule} from './wasm.ts';

export {getAnimationDuration, getAnimationFPS, getAnimationMode, setMorphEnd, setMorphStart} from './animation.ts';
export type {GLTFAttribute, GLTFMaterial} from './gltf-node.ts';
export {GLTFNode, VisualizationGLTFNode} from './gltf-node.ts';
export {importManifold, importModel} from './import-model.ts';
export {getCircularSegments, getMinCircularAngle, getMinCircularEdgeLength} from './level-of-detail.ts';

const manifoldWasm: ManifoldToplevel =
    garbageCollectManifold(await getManifoldModule());

const {Mesh, Manifold, CrossSection, triangulate} = manifoldWasm;

// These methods are not intrinsic to manifold itself, but provided by the scene
// builder.
const show = garbageCollectFunction(debug.show);
const only = garbageCollectFunction(debug.only);
const setMaterial = garbageCollectFunction(material.setMaterial);
const getGLTFNodes = () => [];
const resetGLTFNodes = () => {};

// False whenever this module is imported directly.  The bundler will replace
// this with a function that returns true.
const isManifoldCAD = () => false;

export {
  Mesh,
  Manifold,
  CrossSection,
  triangulate,
  show,
  only,
  setMaterial,
  getGLTFNodes,
  resetGLTFNodes,
  isManifoldCAD
};
