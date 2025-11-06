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

import * as debug from './debug.ts';
import {garbageCollectFunction, garbageCollectManifold} from './garbage-collector.ts';
import * as material from './material.ts';
import {getManifoldModule} from './wasm.ts';

export {getAnimationDuration, getAnimationFPS, getAnimationMode, setMorphEnd, setMorphStart} from './animation.ts';
export {GLTFAttribute, GLTFMaterial, GLTFNode} from './gltf-node.ts';
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
