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

import {CrossSection, Manifold, Mat4, Vec2, Vec3} from '../manifold';

declare class GLTFNode {
  manifold?: Manifold;
  translation?: Vec3|((t: number) => Vec3);
  rotation?: Vec3|((t: number) => Vec3);
  scale?: Vec3|((t: number) => Vec3);
  material?: GLTFMaterial;
  name?: string;
  constructor(parent?: GLTFNode);
  clone(parent?: GLTFNode): GLTFNode;
}

type Attribute = 'POSITION'|'NORMAL'|'TANGENT'|'TEXCOORD_0'|'TEXCOORD_1'|
    'COLOR_0'|'JOINTS_0'|'WEIGHTS_0'|'SKIP_1'|'SKIP_2'|'SKIP_3'|'SKIP_4';

declare class GLTFMaterial {
  attributes?: Attribute[];
  roughness?: number;
  metallic?: number;
  baseColorFactor?: [number, number, number];
  alpha?: number;
  unlit?: boolean;
  name?: string;
}

declare const globalDefaults: {
  roughness: number,
  metallic: number,
  baseColorFactor: [number, number, number],
  alpha: number,
  unlit: boolean,
  animationLength: number,
  animationMode: 'loop'|'ping-pong';
}

/**
 * Returns a shallow copy of the input manifold with the given material
 * properties applied. They will be carried along through operations.
 *
 * @param manifold The input object.
 * @param material A set of material properties to apply to this manifold.
 */
declare function setMaterial(manifold: Manifold, material: GLTFMaterial):
    Manifold;

/**
 * Apply a morphing animation to the input manifold. Specify the start
 * function which will be applied to the vertex positions of the first frame and
 * linearly interpolated across the length of the overall animation. This
 * animation will only be shown if this manifold is used directly on a GLTFNode.
 *
 * @param manifold The object to add morphing animation to.
 * @param func A warping function to apply to the first animation frame.
 */
declare function setMorphStart(
    manifold: Manifold, func: (v: Vec3) => void): void;

/**
 * Apply a morphing animation to the input manifold. Specify the end
 * function which will be applied to the vertex positions of the last frame and
 * linearly interpolated across the length of the overall animation. This
 * animation will only be shown if this manifold is used directly on a GLTFNode.
 *
 * @param manifold The object to add morphing animation to.
 * @param func A warping function to apply to the last animation frame.
 */
declare function setMorphEnd(manifold: Manifold, func: (v: Vec3) => void): void;

/**
 * Wrap any shape object with this method to display it and any copies in
 * transparent red. This is particularly useful for debugging subtract() as it
 * will allow you find the object even if it doesn't currently intersect the
 * result.
 *
 * @param shape The object to show - returned for chaining.
 */
declare function show(shape: CrossSection|Manifold): Manifold;

/**
 * Wrap any shape object with this method to display it and any copies as the
 * result, while ghosting out the final result in transparent gray. Helpful for
 * debugging as it allows you to see objects that may be hidden in the interior
 * of the result. Multiple objects marked only() will all be shown.
 *
 * @param shape The object to show - returned for chaining.
 */
declare function only(shape: CrossSection|Manifold): Manifold;
