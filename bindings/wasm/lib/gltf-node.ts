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

import type {Document as GLTFDocument, Node as GLTFTransformNode} from '@gltf-transform/core';

import type {Manifold} from '../manifold-encapsulated-types.d.ts';
import type {Vec3} from '../manifold-global-types.d.ts';

export type GLTFAttribute =
    'POSITION'|'NORMAL'|'TANGENT'|'TEXCOORD_0'|'TEXCOORD_1'|'COLOR_0'|
    'JOINTS_0'|'WEIGHTS_0'|'SKIP_1'|'SKIP_2'|'SKIP_3'|'SKIP_4';

export class GLTFMaterial {
  attributes?: GLTFAttribute[];
  roughness?: number;
  metallic?: number;
  baseColorFactor?: [number, number, number];
  alpha?: number;
  unlit?: boolean;
  name?: string;
}

export abstract class BaseGLTFNode {
  private _parent?: BaseGLTFNode;
  name?: string;

  translation?: Vec3|((t: number) => Vec3);
  rotation?: Vec3|((t: number) => Vec3);
  scale?: Vec3|((t: number) => Vec3);

  constructor(parent?: BaseGLTFNode) {
    this._parent = parent;
  }

  get parent() {
    return this._parent;
  }
}

export class GLTFNode extends BaseGLTFNode {
  manifold?: Manifold;
  material?: GLTFMaterial;

  clone(parent?: GLTFNode) {
    const copy = new GLTFNode(parent ?? this.parent);
    Object.assign(copy, this);
    return copy;
  }
}

export class NonManifoldGLTFNode extends BaseGLTFNode {
  gltfTransformNode?: GLTFTransformNode;
  gltfDocument?: GLTFDocument;
}

const nodes = new Array<GLTFNode>();

/**
 *
 * @internal
 */
export class GLTFNodeTracked extends GLTFNode {
  constructor(parent?: GLTFNode) {
    super(parent);
    nodes.push(this);
  }
}

/**
 * Get a list of GLTF nodes that have been created in this model.
 *
 * This function only works in scripts directly evaluated by the manifoldCAD
 * website or CLI. When called in an imported library it will always return an
 * empty array, and nodes created in libraries will not be included in the
 * result. This is intentional; libraries must not create geometry as a side
 * effect.
 *
 * @returns An array of GLTFNodes.
 */
export const getGLTFNodes = () => {
  return nodes;
};

/**
 * Clear the list of cached GLTF nodes.
 *
 * This function only works in scripts directly evaluated by the manifoldCAD
 * website or CLI.  When called in an imported library it will have no
 * effect.
 */
export const resetGLTFNodes = () => {
  nodes.length = 0;
};

export const cleanup = () => {
  resetGLTFNodes();
};

/**
 * Map various types to a flat array of GLTFNodes
 *
 * @param any An object or array of models.
 * @returns An array of GLTFNodes.
 */
export async function anyToGLTFNodeList(
    any: Manifold|BaseGLTFNode|
    Array<Manifold|BaseGLTFNode>): Promise<Array<BaseGLTFNode>> {
  if (Array.isArray(any)) {
    return await any.map(anyToGLTFNodeList)
        .reduce(
            async (acc, cur) => ([...(await acc), ...(await cur)]),
            new Promise(resolve => resolve([])))
  } else if (any instanceof BaseGLTFNode) {
    const node = any as BaseGLTFNode;
    if (!node.parent) return [node];
    return [await anyToGLTFNodeList(node.parent), node].flat();
  } else if (any.constructor.name === 'Manifold') {
    const node = new GLTFNode();
    node.manifold = any as Manifold;
    return [node];
  }

  throw new Error('Cannot convert model to GLTFNode!');
}
