// Copyright 2022-2025 The Manifold Authors.
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

// NOTE: This file is undergoing active refactoring, as of August 2025.
// Interfaces and semantics may change.  Beware of wild geese.

import {Document, Material, Node} from '@gltf-transform/core';

import {GLTFMaterial} from '../examples/public/editor';
import {Manifold} from '../manifold-encapsulated-types';
import {Vec3} from '../manifold-global-types';

import {addAnimationToDoc, addMotion, cleanup as cleanupAni, cleanupAnimation, getMorph, morphEnd, morphStart, setMorph} from './animation.ts';
import {cleanup as cleanupDebug, getDebugGLTFMesh, getMaterialByID} from './debug.ts'
import {Properties, writeMesh} from './gltf-io.ts';
import {cleanup as cleanupMaterial, getBackupMaterial, getCachedMaterial} from './material.ts';
import {euler2quat} from './math.ts';

export {setMorphEnd, setMorphStart} from './animation.ts';
export {only, show} from './debug.ts';
export {setMaterial} from './material.ts';

export interface GlobalDefaults {
  roughness: number;
  metallic: number;
  baseColorFactor: [number, number, number];
  alpha: number;
  unlit: boolean;
  animationLength: number;
  animationMode: 'loop'|'ping-pong';
}

const GLOBAL_DEFAULTS = {
  roughness: 0.2,
  metallic: 1,
  baseColorFactor: [1, 1, 0] as [number, number, number],
  alpha: 1,
  unlit: false,
  animationLength: 1,
  animationMode: 'loop'
};

export const globalDefaults = {...GLOBAL_DEFAULTS};

const nodes = new Array<GLTFNode>();

export function cleanup() {
  cleanupAni();
  cleanupDebug();
  cleanupMaterial();
  nodes.length = 0;
}

export class GLTFNode {
  private _parent?: GLTFNode;
  manifold?: Manifold;
  translation?: Vec3|((t: number) => Vec3);
  rotation?: Vec3|((t: number) => Vec3);
  scale?: Vec3|((t: number) => Vec3);
  material?: GLTFMaterial;
  name?: string;

  constructor(parent?: GLTFNode) {
    this._parent = parent;
    nodes.push(this);
  }
  clone(parent?: GLTFNode) {
    const copy = {...this};
    copy._parent = parent;
    nodes.push(copy);
    return copy;
  }
  get parent() {
    return this._parent;
  }
}

// Swallow informational logs in testing framework
function log(...args: any[]) {
  if (typeof self !== 'undefined' && self.console) {
    self.console.log(...args);
  }
}

function createGLTFnode(doc: Document, node: GLTFNode): Node {
  const out = doc.createNode(node.name);

  // Animation Motion
  const pos = addMotion(doc, 'translation', node, out);
  if (pos != null) {
    out.setTranslation(pos);
  }

  const rot = addMotion(doc, 'rotation', node, out);
  if (rot != null) {
    out.setRotation(euler2quat(rot));
  }

  const scale = addMotion(doc, 'scale', node, out);
  if (scale != null) {
    out.setScale(scale);
  }

  return out;
}

function addMesh(
    doc: Document, node: Node, manifold: Manifold,
    backupMaterial: GLTFMaterial = {}) {
  const numTri = manifold.numTri();
  if (numTri == 0) {
    log('Empty manifold, skipping.');
    return;
  }

  log(`Triangles: ${numTri.toLocaleString()}`);
  const box = manifold.boundingBox();
  const size = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    size[i] = Math.round((box.max[i] - box.min[i]) * 10) / 10;
  }
  log(`Bounding Box: X = ${size[0].toLocaleString()} mm, Y = ${
      size[1].toLocaleString()} mm, Z = ${size[2].toLocaleString()} mm`);
  const volume = Math.round(manifold.volume() / 10);
  log(`Genus: ${manifold.genus().toLocaleString()}, Volume: ${
      (volume / 100).toLocaleString()} cm^3`);

  // From Z-up to Y-up (glTF)
  const manifoldMesh = manifold.getMesh();

  // Material
  const id2properties = new Map<number, Properties>();
  for (const id of manifoldMesh.runOriginalID!) {
    const material = getMaterialByID(id) || backupMaterial;
    id2properties.set(id, {
      material: getCachedMaterial(doc, material),
      attributes: ['POSITION', ...material.attributes ?? []]
    });
  }

  // Animation Morph
  const morph = getMorph(manifold);
  const inputPositions = morphStart(manifoldMesh, morph);

  // Core
  const mesh = writeMesh(doc, manifoldMesh, id2properties);
  node.setMesh(mesh);

  // Animation Morph
  morphEnd(doc, manifoldMesh, mesh, inputPositions, morph);


  // If we're using a debug mode (`show` or `only`), check
  // to see if this mesh requires special handling.
  const debugNodes =
      getDebugGLTFMesh(doc, manifoldMesh, id2properties, backupMaterial)
  for (const debugNode of debugNodes) {
    node.addChild(debugNode)
  }
}

function cloneNode(toNode: Node, fromNode: Node) {
  toNode.setMesh(fromNode.getMesh());
  fromNode.listChildren().forEach((child) => {
    const clone = child.clone();
    toNode.addChild(clone);
  });
}

function cloneNodeNewMaterial(
    doc: Document, toNode: Node, fromNode: Node, backupMaterial: Material,
    oldBackupMaterial: Material) {
  cloneNode(toNode, fromNode);
  const oldMesh = fromNode.getMesh()!;
  const newMesh = doc.createMesh();
  toNode.setMesh(newMesh);
  oldMesh.listPrimitives().forEach((primitive) => {
    const newPrimitive = primitive.clone();
    if (primitive.getMaterial() === oldBackupMaterial) {
      newPrimitive.setMaterial(backupMaterial);
    }
    newMesh.addPrimitive(newPrimitive);
  });
  // Track cloned meshes for easier export, later.
  newMesh.setExtras({clonedFrom: oldMesh});
}

function createNodeFromCache(
    doc: Document, nodeDef: GLTFNode,
    manifold2node: Map<Manifold, Map<GLTFMaterial, Node>>): Node {
  const node = createGLTFnode(doc, nodeDef);
  const {manifold} = nodeDef;
  if (manifold) {
    // Animation Morph
    setMorph(doc, node, manifold);
    const backupMaterial = getBackupMaterial(nodeDef);
    const cachedNodes = manifold2node.get(manifold);

    if (cachedNodes == null) {
      // Cache miss.
      addMesh(doc, node, manifold, backupMaterial);
      const cache = new Map<GLTFMaterial, Node>();
      cache.set(backupMaterial, node);
      manifold2node.set(manifold, cache);

    } else {
      // Cache hit...
      const cachedNode = cachedNodes.get(backupMaterial);
      if (cachedNode == null) {
        // ...but not for this material.
        const [oldBackupMaterial, oldNode] =
            cachedNodes.entries().next().value!;
        cloneNodeNewMaterial(
            doc, node, oldNode, getCachedMaterial(doc, backupMaterial),
            getCachedMaterial(doc, oldBackupMaterial));
        cachedNodes.set(backupMaterial, node);

      } else {
        // ...for this exact material.
        cloneNode(node, cachedNode);
      }
    }
  }
  return node;
}

function parseOptions(defaults: GlobalDefaults) {
  Object.assign(globalDefaults, GLOBAL_DEFAULTS);
  Object.assign(globalDefaults, defaults);
}

function createWrapper(doc: Document) {
  const halfRoot2 = Math.sqrt(2) / 2;
  const mm2m = 1 / 1000;
  const wrapper = doc.createNode('wrapper')
                      .setRotation([-halfRoot2, 0, 0, halfRoot2])
                      .setScale([mm2m, mm2m, mm2m]);
  doc.createScene().addChild(wrapper);
  return wrapper
}

export function manifoldToGLTFDoc(
    manifold: Manifold, defaults: GlobalDefaults) {
  const node = new GLTFNode();
  node.manifold = manifold;
  return GLTFNodesToGLTFDoc([node], defaults)
}

export function GLTFNodesToGLTFDoc(
    nodes: Array<GLTFNode>, defaults: GlobalDefaults) {
  parseOptions(defaults)

  if (nodes.length == 0) {
    throw new TypeError('nodes[] must contain at least one GLTFNode.')
  }

  const doc = new Document();
  const root = createWrapper(doc);

  addAnimationToDoc(doc);

  const node2gltf = new Map<GLTFNode, Node>();
  const manifold2node = new Map<Manifold, Map<GLTFMaterial, Node>>();
  let leafNodes = 0;

  // First, create a node in the GLTF document for each ManifoldCAD node.
  for (const nodeDef of nodes) {
    node2gltf.set(nodeDef, createNodeFromCache(doc, nodeDef, manifold2node));
    if (nodeDef.manifold) {
      ++leafNodes;
    }
  }

  // Step through each node and set its parent.
  // Nodes without parents are added directly to the root.
  for (const nodeDef of nodes) {
    const gltfNode = node2gltf.get(nodeDef)!;
    const {parent} = nodeDef;
    if (parent) {
      node2gltf.get(parent)!.addChild(gltfNode);
    } else {
      root.addChild(gltfNode);
    }
  }

  log('Total glTF nodes: ', nodes.length,
      ', Total mesh references: ', leafNodes);

  cleanupAnimation();
  return doc;
}

export function hasGLTFNodes() {
  return nodes.length > 0;
}

export function getGLTFNodes() {
  return nodes;
}
