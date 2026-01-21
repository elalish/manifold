// Copyright 2023-2025 The Manifold Authors.
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

/* This file is used in two ways.
 * Run it through typedoc (`npm run docs:jsuser`), and it generates
 * the ManifoldCAD User Guide.
 * On the other hand API Extractor (`npm run postbuild`) rolls up
 * all of the imported references to generate `dist/manifoldCAD.d.ts`.
 *
 * Getting TypeDoc to play nice is hard!
 * Groups added here are not honoured on:
 *   * Classes.
 *   * Interfaces.
 *   * Types that already have TypeDoc comments.
 *   * Arrow functions.
 *
 * I've taken the approach of:
 *   * Switching exports away from arrow functions, and setting groups here.
 *   * Setting groups on Classes and Interfaces in their upstream files.
 *     This might require some editing on the Developer Guide side for
 *     consistency.
 *   * Marking a few types as `@internal` -- typical users won't interact
 *     Mesh objects or SealedUInt32Arrays.
 *   * Good user documentation outweighs good developer documentation.
 *     We can just read the code anyhow.
 *
 * It's also possible to just `export declare`, and _not_ re-export
 * the original comment.  That may be required if user and developer
 * documentation diverge significantly.
 */

/**
 * These are the objects and functions that are available within manifoldCAD
 * itself.
 *
 * @packageDocumentation
 * @module manifoldCAD
 * @see {@link https://www.manifoldcad.org | ManifoldCAD.org}
 * @see {@link https://manifoldcad.org/docs/jsapi/modules/manifoldCAD.html | Manifold WASM Developer Guide: Module manifoldCAD}
 * @sortStrategy kind
 */

/** @group Basics */
/** @group Animation */
export type {AnimationMode} from '../lib/animation';
/** @group Animation */
export {getAnimationDuration, getAnimationFPS, getAnimationMode, setMorphEnd, setMorphStart} from '../lib/animation';
/** @group Material */
export {only, show} from '../lib/debug';
/** @group Material */
export type {GLTFMaterial} from '../lib/gltf-node';
/** @group Scene Graph */
export {BaseGLTFNode, getGLTFNodes, GLTFAttribute, GLTFNode, resetGLTFNodes, VisualizationGLTFNode} from '../lib/gltf-node';
/** @group Input & Output */
export {importManifold, importModel} from '../lib/import-model';
/** @group Level of Detail */
export {getCircularSegments, getMinCircularAngle, getMinCircularEdgeLength} from '../lib/level-of-detail'
/** @group Material */
export {setMaterial} from '../lib/material';
export {CrossSection, Manifold, triangulate} from '../manifold';

/**
 * Is this module running in manifoldCAD.org or the ManifoldCAD CLI?
 *
 * @returns boolean
 * @group Information
 */
export declare function isManifoldCAD(): boolean

/* Type Aliases */
export type { FillRule, JoinType
} from '../manifold';
export type {Box, Rect} from '../manifold';
export type {Polygons, SimplePolygon} from '../manifold';
export type {Mat3, Mat4, Vec2, Vec3} from '../manifold';
export type {Smoothness} from '../manifold';
export type {ErrorStatus} from '../manifold';

/* See the Developer Guide for more detail on these: */
/** @internal */
export type {SealedFloat32Array, SealedUint32Array} from '../manifold';
/** @inernal */
export type {ImportOptions} from '../lib/import-model';
/** @internal */
export type {MeshOptions} from '../manifold';
/** @internal */
export {Mesh} from '../manifold';