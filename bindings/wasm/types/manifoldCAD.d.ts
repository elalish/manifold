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

/**
 * These are the objects and functions that are available within manifoldCAD
 * itself.
 *
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
 * It can be imported as `manifold-3d/manifoldCAD`.
 *
 * @packageDocumentation
 * @module manifoldCAD
 * @showGroups true
 * @primaryExport
 * @see {@link "Using manifoldCAD" | Using manifoldCAD}
 *
 * @groupDescription Global State
 * These objects and functions are specific to top-level scripts
 * running within manifoldCAD.
 *
 * They are only accessible as global objects by a top level script evaluated by
 * the worker.  Libraries will not have access to them.
 *
 * These functions will not be present at all when a model is imported as an ES
 * module. They can be imported through the {@link lib/scene-builder! | scene
 * builder} or directly from {@link lib/animation! | animation} and {@link
 * lib/level-of-detail! | level-of-detail} modules.
 */

export {AnimationMode, getAnimationDuration, getAnimationFPS, getAnimationMode, setMorphEnd, setMorphStart} from '../lib/animation.d.ts';
export {only, show} from '../lib/debug.ts';
export {BaseGLTFNode, getGLTFNodes, GLTFAttribute, GLTFMaterial, GLTFNode, resetGLTFNodes, VisualizationGLTFNode} from '../lib/gltf-node.d.ts';
export {importManifold, importModel, ImportOptions} from '../lib/import-model.ts';
export {getCircularSegments, getMinCircularAngle, getMinCircularEdgeLength} from '../lib/level-of-detail.d.ts'
export {setMaterial} from '../lib/material.d.ts';
export {Box, CrossSection, ErrorStatus, FillRule, JoinType, Manifold, Mat3, Mat4, Mesh, MeshOptions, Polygons, Rect, SealedFloat32Array, SealedUint32Array, SimplePolygon, Smoothness, triangulate, Vec2, Vec3} from '../manifold.d.ts';

/**
 * Is this module running in manifoldCAD evaluator?
 *
 * @returns boolean
 */
export declare function isManifoldCAD(): boolean
