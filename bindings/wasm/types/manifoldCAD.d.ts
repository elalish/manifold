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
 * @packageDocumentation
 * @module manifoldCAD
 * @see {@link "Using ManifoldCAD" | Using ManifoldCAD}
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
