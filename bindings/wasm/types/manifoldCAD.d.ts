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
 * These are the objects and functions that are available to all manifoldCAD
 * models.
 *
 * This is an isomorphic module.  When imported within manifoldCAD, the bundler
 * will swap it out for an identical module running in the worker context.
 * When imported as an ES module, it will implicitly instantiate a manifold wasm
 * module, and export it along with relevant scene-builder properties.
 * This allows models to behave identically when running on manifoldCAD.org,
 * through the CLI, or through nodejs.
 *
 * It can be imported as `manifold-3d/manifoldCAD`.
 *
 * @packageDocumentation
 * @module manifold-3d/manifoldCAD
 */

export {AnimationMode, getAnimationDuration, getAnimationFPS, getAnimationMode, setMorphEnd, setMorphStart} from '../lib/animation.d.ts';
export {only, show} from '../lib/debug.ts';
export {getGLTFNodes, GLTFAttribute, GLTFMaterial, GLTFNode, resetGLTFNodes} from '../lib/gltf-node.d.ts';
export {getCircularSegments, getMinCircularAngle, getMinCircularEdgeLength} from '../lib/level-of-detail.d.ts'
export {setMaterial} from '../lib/material.d.ts';
export {CrossSection, Manifold, Mesh, MeshOptions, triangulate} from '../manifold-encapsulated-types.d.ts';
export {Box, ErrorStatus, FillRule, JoinType, Mat3, Mat4, Polygons, Rect, SealedFloat32Array, SealedUint32Array, SimplePolygon, Smoothness, Vec2, Vec3} from '../manifold-global-types.d.ts';

/**
 * Is this module running in manifoldCAD evaluator?
 *
 * @returns boolean
 */
export declare function isManifoldCAD(): boolean
