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
 * @group ManifoldCAD
 * @category Core
 */

import type * as GLTFTransform from '@gltf-transform/core';

import type {CrossSection, Manifold, Vec3} from '../manifold.d.ts';

const nodes = new Array<BaseGLTFNode>();


/**
 * @inline
 * @internal
 */
export type GLTFAttribute =
    'POSITION'|'NORMAL'|'TANGENT'|'TEXCOORD_0'|'TEXCOORD_1'|'COLOR_0'|
    'JOINTS_0'|'WEIGHTS_0'|'SKIP_1'|'SKIP_2'|'SKIP_3'|'SKIP_4';

/**
 * Define a material using the glTF metallic-roughness physically-based
 * rendering model. Materials can be applied to a model through `setMaterial()`,
 * or set as a {@link GLTFNode.material | GLTFNode property}.
 *
 * @see {@link https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials | glTF 2.0 Specification: Materials}
 * @see {@link https://physicallybased.info/ | Physically Based - The PBR values database}
 * @sortStrategy source-order
 * @group Material
 */
export interface GLTFMaterial {
  /**
   * Every vertex in a glTF Mesh has a set of attributes.
   * `POSITION` cannot be specified -- ManifoldCAD will set it internally.
   *
   * This array specifies how vertex properties are arranged in memory.
   * For example, a value of `['TEXCOORD_0', 'NORMAL', 'SKIP_2', 'COLOR_0']`
   * would implicitly use property channels 0-2 for position, followed by
   * channels 3-4 for texture, 5-7 for surface normal, ignore 8-9, and 10-12 for
   * color.
   *
   * Some properties such as `TEXCOORD_0` or `COLOR_0` may be set when importing
   * a model that has a texture or material.
   *
   * When vertex property `COLOR_0` is specified, it will be multiplied
   * against {@link baseColorFactor}.
   *
   * @see {@link https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes-overview | glTF 2.0 Specification: Meshes Overview}
   * @see {@link https://manifoldcad.org/#Tetrahedron%20Puzzle | ManifoldCAD Example: Tetrahedron Puzzle}
   */
  attributes?: GLTFAttribute[];

  /**
   * Roughness of the material.
   * Ranges from 0 (smooth, specular) to 1.0 (rough, diffuse).
   *
   * @default 0.2
   */
  roughness?: number;

  /**
   * Metallic property of the material.
   * Ranges from 0 (dielectric, e.g.: yellow plastic) to 1.0 (conductor, e.g:
   * gold). Generally speaking materials are either one or the other and
   * intermediate values are just for blending.
   *
   * @default 1.0 // Metallic
   */
  metallic?: number;

  /**
   * Base colour of the material.
   * RGB values, ranging from 0 to 1.0.
   *
   * If the {@link attributes | attribute} `COLOR_0` is specifed, it will be
   * multiplied against `baseColorFactor`. In this case, use an appropriate
   * value like `[1.0, 1.0, 1.0]`.
   * @default [1.0, 1.0, 0.0] // Yellow
   */
  baseColorFactor?: [number, number, number];

  /**
   * Transparency of the material.
   * Ranges from 0 (fully transparent) to 1.0 (fully opaque).
   * @default 1.0 // Opaque
   */
  alpha?: number;

  /**
   * Render model as unlit or shadeless, as opposed to physically based
   * rendering.
   *
   * @see {@link https://github.com/KhronosGroup/gltf/tree/main/extensions/2.0/Khronos/KHR_materials_unlit | KHR_materials_unlit}
   * @see {@link https://gltf-transform.dev/modules/extensions/classes/KHRMaterialsUnlit | glTF Transform: KHRMaterialsUnlit}
   * @default false // Lit and shadowed.
   */
  unlit?: boolean;

  /**
   * Material name.  Will be passed through when exported.
   */
  name?: string;

  /**
   * If set, this material is a copy of another material on an in-memory glTF
   * model. This is used by `importManifold` and `importModel` to pass original
   * materials and textures through manifold.
   * @internal
   */
  sourceMaterial?: GLTFTransform.Material;

  /**
   * If set, this material is a copy of another material on an in-memory glTF
   * model. This is used by `importManifold` and `importModel` to pass original
   * materials and textures through manifold.
   * @internal
   */
  sourceRunID?: number;
}

/**
 * The abstract class from which other classes inherit.  Common methods and
 * properties live here.
 * @group Scene Graph
 */
export abstract class BaseGLTFNode {
  /** @internal */
  _parent?: BaseGLTFNode;
  name?: string;

  // Internally, gltf-transform stores transformations as separate translation,
  // rotation and scale vectors.  It can convert those vectors to and from a
  // transformation matrix as needed.
  translation?: Vec3|((t: number) => Vec3);

  /**
   * From the reference frame of the model being rotated, rotations are applied
   * in *z-y'-x"* order. That is yaw first, then pitch and finally roll.
   *
   * From the global reference frame, a model will be rotated in *x-y-z* order.
   * That is about the global X axis, then global Y axis, and finally global Z.
   *
   * This matches the behaviour of `Manifold.rotate()`.
   */
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
 * @group Scene Graph
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
 *
 * @group Scene Graph
 */
export class VisualizationGLTFNode extends BaseGLTFNode {
  /** @internal */
  node?: GLTFTransform.Node;
  /** @internal */
  document: GLTFTransform.Document;
  /** @internal */
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
 * Display a CrossSection in 3D space.
 *
 * A CrossSection object is two dimensional.  Attaching it as a node
 * allows it to be included in the final exported file, complete with
 * transformations.
 *
 * > [!NOTICE]
 * >
 * > CrossSections are not -- and can never be -- manifold.  That means
 * > some exporters (like `.3mf`) will just skip over them entirely.
 *
 * @group Scene Graph
 */
export class CrossSectionGLTFNode extends BaseGLTFNode {
  /** @internal */
  _crossSection?: CrossSection;
  material?: GLTFMaterial;

  constructor(cs?: CrossSection, parent?: BaseGLTFNode) {
    super(parent);
    this._crossSection = cs;
  }

  clone(newParent?: BaseGLTFNode) {
    const copy =
        new CrossSectionGLTFNode(this._crossSection, newParent ?? this.parent);
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
export function getGLTFNodes() {
  return nodes;
};

/**
 * Clear the list of cached GLTF nodes.
 *
 * This function only works in scripts directly evaluated by the manifoldCAD
 * website or CLI.  When called in an imported library it will have no
 * effect.
 */
export function resetGLTFNodes() {
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
    any: Manifold|BaseGLTFNode|CrossSection|
    Array<Manifold|BaseGLTFNode|CrossSection>): Promise<Array<BaseGLTFNode>> {
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
  } else if (any.constructor.name === 'CrossSection') {
    const node = new CrossSectionGLTFNode(any as CrossSection);
    return [node];
  }

  throw new Error('Cannot convert model to GLTFNode!');
}
