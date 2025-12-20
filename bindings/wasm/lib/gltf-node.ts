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
 * Nodes enable ManifoldCAD users to build models that are beyond a single
 * Manifold object. This can range from assembling models to animation to even
 * including non-manifold objects from other sources.
 *
 * ManifoldCAD uses the `GLTFNode` class to manage objects that will be exported
 * into a gltf-transform document.
 *
 * It's important to note that gltf-transform *also* has a `Node` class,
 * representing objects that are *already* part of a gltf-transform document.
 * That class can mostly be seen in import and export related code.
 *
 * This leads to the confusing case where, `NonManifoldGLTFNode` is both.  It is
 * a manifoldCAD node that contains an imported gltf-transform `Node`.  It
 * specifies how that gltf-transform node will be scaled, rotated and translated
 * when it is eventually copied into a new gltf-transform document at export
 * time.
 *
 * @packageDocumentation
 */

import type * as GLTFTransform from '@gltf-transform/core';

import type {Manifold} from '../manifold-encapsulated-types.d.ts';
import type {Vec3} from '../manifold-global-types.d.ts';

const nodes = new Array<BaseGLTFNode>();

export type GLTFAttribute =
    'POSITION'|'NORMAL'|'TANGENT'|'TEXCOORD_0'|'TEXCOORD_1'|'COLOR_0'|
    'JOINTS_0'|'WEIGHTS_0'|'SKIP_1'|'SKIP_2'|'SKIP_3'|'SKIP_4';

export interface GLTFMaterial {
  attributes?: GLTFAttribute[];
  roughness?: number;
  metallic?: number;
  baseColorFactor?: [number, number, number];
  alpha?: number;
  unlit?: boolean;
  name?: string;
  sourceMaterial?: GLTFTransform.Material;
  sourceRunID?: number;
}

/**
 * The abstract class from which other classes inherit.  Common methods and
 * properties live here.
 */
export abstract class BaseGLTFNode {
  _parent?: BaseGLTFNode;
  name?: string;

  // Internally, gltf-transform stores transformations as separate translation,
  // rotation and scale vectors.  It can convert those vectors to and from a
  // transformation matrix as needed.
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

/**
 * Position a manifold model for later export.
 */
export class GLTFNode extends BaseGLTFNode {
  manifold?: Manifold;
  material?: GLTFMaterial;

  clone(newParent?: BaseGLTFNode) {
    const copy = new GLTFNode(newParent ?? this.parent);
    Object.assign(copy, this);
    return copy;
  }
}

/**
 * Track created GLTFNodes for top level scripts.
 *
 */
export class GLTFNodeTracked extends GLTFNode {
  constructor(parent?: BaseGLTFNode) {
    super(parent);
    nodes.push(this);
  }

  clone(newParent?: BaseGLTFNode) {
    const copy = new GLTFNodeTracked(newParent ?? this.parent);
    Object.assign(copy, this);
    return copy;
  }
}

/**
 * Include an imported model for visualization purposes.
 *
 * These nodes contain models that will be exported into the final GLTF
 * document.  They have not been converted into Manifold objects and cannot be
 * modified. They can only be transformed (rotation, scale, translation) or
 * displayed.
 *
 * This is useful for viewing ManifoldCAD models in the context of a larger
 * assembly.
 *
 * GLTF objects meeting the `manifold-gltf` extension will still be manifold
 * when exported.
 */
export class VisualizationGLTFNode extends BaseGLTFNode {
  node?: GLTFTransform.Node;
  document: GLTFTransform.Document;
  uri?: string;

  constructor(
      document: GLTFTransform.Document, node?: GLTFTransform.Node,
      parent?: BaseGLTFNode) {
    super(parent);
    this.document = document;
    this.node = node;
  }

  clone(newParent?: BaseGLTFNode) {
    const copy = new VisualizationGLTFNode(
        this.document, this.node, newParent ?? this.parent);
    Object.assign(copy, this);
    return copy;
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

/**
 * @internal
 */
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
            new Promise(resolve => resolve([])));
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
