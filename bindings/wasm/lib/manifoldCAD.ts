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
 * models.
 *
 * This is an isomorphic module.  When imported within manifoldCAD, the bundler
 * will swap it out for an identical module running in the worker context.
 * When imported as an ES module, it will implicitly instantiate a manifold wasm
 * module, and export it along with relevant scene-builder properties.
 * This allows models to behave identically when running on manifoldCAD.org,
 * through the CLI, or through nodejs.
 *
 * @module manifoldCAD
 */

import type {ManifoldToplevel} from '../manifold.d.ts';

import {garbageCollectFunction, garbageCollectManifold} from './garbage-collector.ts';
import * as scenebuilder from './scene-builder.ts';
import {getManifoldModule} from './wasm.ts';

const manifoldWasm: ManifoldToplevel =
    garbageCollectManifold(await getManifoldModule());

const {
  Mesh,
  Manifold,
  CrossSection,
  setMinCircularAngle,
  setMinCircularEdgeLength,
  setCircularSegments,
  getCircularSegments,
  resetToCircularDefaults,
  triangulate
} = manifoldWasm;

// These methods are not intrinsic to manifold itself, but provided by the scene
// builder.
const {setMorphStart, setMorphEnd} = scenebuilder;
const show = garbageCollectFunction(scenebuilder.show);
const only = garbageCollectFunction(scenebuilder.only);
const setMaterial = garbageCollectFunction(scenebuilder.setMaterial);
const {GLTFNode} = scenebuilder;
const getGLTFNodes = () => [];
const resetGLTFNodes = () => {};

// False whenever this module is imported directly.  The bundler will replace
// this with a function that returns true.
const isManifoldCAD = () => false;

export {
  Mesh,
  Manifold,
  CrossSection,
  setMinCircularAngle,
  setMinCircularEdgeLength,
  setCircularSegments,
  getCircularSegments,
  resetToCircularDefaults,
  triangulate,
  show,
  only,
  setMaterial,
  setMorphStart,
  setMorphEnd,
  GLTFNode,
  getGLTFNodes,
  resetGLTFNodes,
  isManifoldCAD
};
