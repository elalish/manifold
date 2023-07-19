// Copyright 2022 The Manifold Authors.
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

import {Document, Material, Node, WebIO} from '@gltf-transform/core';
import {KHRMaterialsUnlit, KHRONOS_EXTENSIONS} from '@gltf-transform/extensions';
import * as glMatrix from 'gl-matrix';

import Module from './built/manifold';
//@ts-ignore
import {setupIO, writeMesh} from './gltf-io';
import type {GLTFMaterial, Quat} from './public/editor';
import type {CrossSection, Manifold, ManifoldToplevel, Mesh, Vec3} from './public/manifold';

interface WorkerStatic extends ManifoldToplevel {
  GLTFNode: typeof GLTFNode;
  show(manifold: Manifold): Manifold;
  only(manifold: Manifold): Manifold;
  setMaterial(manifold: Manifold, material: GLTFMaterial): void;
  cleanup(): void;
}

const module = await Module() as unknown as WorkerStatic;
module.setup();

// Faster on modern browsers than Float32Array
glMatrix.glMatrix.setMatrixArrayType(Array);

const io = setupIO(new WebIO());
io.registerExtensions(KHRONOS_EXTENSIONS);

// manifold static methods (that return a new manifold)
const manifoldStaticFunctions = [
  'cube', 'cylinder', 'sphere', 'tetrahedron', 'extrude', 'revolve', 'compose',
  'union', 'difference', 'intersection', 'levelSet', 'smooth', 'ofMesh', 'hull'
];
// manifold member functions (that return a new manifold)
const manifoldMemberFunctions = [
  'add', 'subtract', 'intersect', 'decompose', 'warp', 'transform', 'translate',
  'rotate', 'scale', 'mirror', 'refine', 'setProperties', 'asOriginal',
  'trimByPlane', 'split', 'splitByPlane', 'hull'
];
// CrossSection static methods (that return a new cross-section)
const crossSectionStaticFunctions = [
  'square', 'circle', 'union', 'difference', 'intersection', 'compose',
  'ofPolygons', 'hull'
];
// CrossSection member functions (that return a new cross-section)
const crossSectionMemberFunctions = [
  'add', 'subtract', 'intersect', 'rectClip', 'decompose', 'transform',
  'translate', 'rotate', 'scale', 'mirror', 'simplify', 'offset', 'hull'
];
// top level functions that construct a new manifold/mesh
const toplevelConstructors = ['show', 'only', 'setMaterial'];
const toplevel = [
  'setMinCircularAngle', 'setMinCircularEdgeLength', 'setCircularSegments',
  'getCircularSegments', 'Mesh', 'GLTFNode', 'Manifold', 'CrossSection'
];
const exposedFunctions = toplevelConstructors.concat(toplevel);

// Setup memory management, such that users don't have to care about
// calling `delete` manually.
// Note that this only fixes memory leak across different runs: the memory
// will only be freed when the compilation finishes.

const memoryRegistry = new Array<Manifold|CrossSection>();

function addMembers(
    className: string, methodNames: Array<string>, areStatic: boolean) {
  //@ts-ignore
  const cls = module[className];
  const obj = areStatic ? cls : cls.prototype;
  for (const name of methodNames) {
    if (name != 'cylinder') {
      const originalFn = obj[name];
      obj[name] = function(...args: any) {
        //@ts-ignore
        const result = originalFn(...args);
        memoryRegistry.push(result);
        return result;
      };
    }
  }
}

addMembers('Manifold', manifoldMemberFunctions, false);
addMembers('Manifold', manifoldStaticFunctions, true);
addMembers('CrossSection', crossSectionMemberFunctions, false);
addMembers('CrossSection', crossSectionStaticFunctions, true);

for (const name of toplevelConstructors) {
  //@ts-ignore
  const originalFn = module[name];
  //@ts-ignore
  module[name] = function(...args: any) {
    const result = originalFn(...args);
    memoryRegistry.push(result);
    return result;
  };
}

module.cleanup = function() {
  for (const obj of memoryRegistry) {
    // decompose result is an array of manifolds
    if (obj instanceof Array)
      for (const elem of obj) elem.delete();
    else
      obj.delete();
  }
  memoryRegistry.length = 0;
};

// Debug setup to show source meshes
let ghost = false;
const shown = new Map<number, Mesh>();
const singles = new Map<number, Mesh>();

const SHOW = {
  baseColorFactor: [1, 0, 0],
  alpha: 0.25,
  roughness: 1,
  metallic: 0
} as GLTFMaterial;

const GHOST = {
  baseColorFactor: [0.5, 0.5, 0.5],
  alpha: 0.25,
  roughness: 1,
  metallic: 0
} as GLTFMaterial;

const nodes = new Array<GLTFNode>();
const id2material = new Map<number, GLTFMaterial>();
const materialCache = new Map<GLTFMaterial, Material>();

function cleanup() {
  ghost = false;
  shown.clear();
  singles.clear();
  nodes.length = 0;
  id2material.clear();
  materialCache.clear();
}

class GLTFNode {
  private _parent?: GLTFNode;
  manifold?: Manifold;
  translation?: Vec3;
  rotation?: Vec3;
  scale?: Vec3;
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

module.GLTFNode = GLTFNode;

module.setMaterial = (manifold: Manifold, material: GLTFMaterial): Manifold => {
  const out = manifold.asOriginal();
  id2material.set(out.originalID(), material);
  return out;
};

function debug(manifold: Manifold, map: Map<number, Mesh>) {
  let result = manifold.asOriginal();
  map.set(result.originalID(), result.getMesh());
  return result;
};

module.show = (manifold) => {
  return debug(manifold, shown);
};

module.only = (manifold) => {
  ghost = true;
  return debug(manifold, singles);
};

// Setup complete
self.postMessage(null);

if (self.console) {
  const oldLog = self.console.log;
  self.console.log = function(...args) {
    let message = '';
    for (const arg of args) {
      if (arg == null) {
        message += 'undefined';
      } else if (typeof arg == 'object') {
        message += JSON.stringify(arg, null, 4);
      } else {
        message += arg.toString();
      }
    }
    self.postMessage({log: message});
    oldLog(...args);
  };
}

// Swallow informational logs in testing framework
function log(...args: any[]) {
  if (self.console) {
    self.console.log(...args);
  }
}

self.onmessage = async (e) => {
  const content = e.data +
      '\nreturn exportGLB(typeof result === "undefined" ? undefined : result);\n';
  try {
    const f = new Function(
        'exportGLB', 'glMatrix', 'module', ...exposedFunctions, content);
    await f(
        exportGLB, glMatrix, module,  //@ts-ignore
        ...exposedFunctions.map(name => module[name]));
  } catch (error: any) {
    console.log(error.toString());
    self.postMessage({objectURL: null});
  } finally {
    module.cleanup();
    cleanup();
  }
};

function createGLTFnode(doc: Document, node: GLTFNode) {
  const out = doc.createNode(node.name);
  if (node.translation) {
    out.setTranslation(node.translation);
  }
  if (node.rotation) {
    const {quat} = glMatrix;
    const deg2rad = Math.PI / 180;
    const q = quat.create() as Quat;
    quat.rotateX(q, q, deg2rad * node.rotation[0]);
    quat.rotateY(q, q, deg2rad * node.rotation[1]);
    quat.rotateZ(q, q, deg2rad * node.rotation[2]);
    out.setRotation(q);
  }
  if (node.scale) {
    out.setScale(node.scale);
  }
  return out;
}

function getBackupMaterial(node?: GLTFNode): GLTFMaterial {
  if (node == null) {
    return {};
  }
  if (node.material == null) {
    node.material = getBackupMaterial(node.parent);
  }
  return node.material;
}

function makeDefaultedMaterial(doc: Document, {
  roughness = 0.2,
  metallic = 1,
  baseColorFactor = [1, 1, 0],
  alpha = 1,
  unlit = false,
  name = ''
}: GLTFMaterial = {}): Material {
  const material = doc.createMaterial(name);

  if (unlit) {
    const unlit = doc.createExtension(KHRMaterialsUnlit).createUnlit();
    material.setExtension('KHR_materials_unlit', unlit);
  }

  if (alpha < 1) {
    material.setAlphaMode(Material.AlphaMode.BLEND).setDoubleSided(true);
  }

  return material.setRoughnessFactor(roughness)
      .setMetallicFactor(metallic)
      .setBaseColorFactor([...baseColorFactor, alpha]);
}

function getCachedMaterial(doc: Document, matDef: GLTFMaterial): Material {
  if (!materialCache.has(matDef)) {
    materialCache.set(matDef, makeDefaultedMaterial(doc, matDef));
  }
  return materialCache.get(matDef)!;
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
  const volume = Math.round(manifold.getProperties().volume / 10);
  log(`Genus: ${manifold.genus().toLocaleString()}, Volume: ${
      (volume / 100).toLocaleString()} cm^3`);

  // From Z-up to Y-up (glTF)
  const manifoldMesh = manifold.getMesh();

  const materials = new Array<GLTFMaterial>();
  const attributes = new Array<Array<string>>();
  for (const id of manifoldMesh.runOriginalID!) {
    const material = id2material.get(id) || backupMaterial;
    materials.push(material);
    attributes.push(['POSITION', ...material.attributes ?? []]);
  }

  const gltfMaterials = materials.map((matDef) => {
    return getCachedMaterial(doc, ghost ? GHOST : matDef);
  });

  node.setMesh(writeMesh(doc, manifoldMesh, attributes, gltfMaterials));

  for (const [run, id] of manifoldMesh.runOriginalID!.entries()) {
    let inMesh = shown.get(id);
    let single = false;
    if (inMesh == null) {
      single = true;
      inMesh = singles.get(id);
    }
    if (inMesh == null) {
      continue;
    }
    const mat = single ? materials[run] : SHOW;
    const debugNode =
        doc.createNode('debug')
            .setMesh(writeMesh(
                doc, inMesh, attributes, [getCachedMaterial(doc, mat)]))
            .setMatrix(manifoldMesh.transform(run));
    node.addChild(debugNode);
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
  const mesh = doc.createMesh();
  toNode.setMesh(mesh);
  fromNode.getMesh()!.listPrimitives().forEach((primitive) => {
    const newPrimitive = primitive.clone();
    if (primitive.getMaterial() === oldBackupMaterial) {
      newPrimitive.setMaterial(backupMaterial);
    }
    mesh.addPrimitive(newPrimitive);
  });
}

function createNodeFromCache(
    doc: Document, nodeDef: GLTFNode,
    manifold2node: Map<Manifold, Map<GLTFMaterial, Node>>): Node {
  const node = createGLTFnode(doc, nodeDef);
  if (nodeDef.manifold != null) {
    const backupMaterial = getBackupMaterial(nodeDef);
    const cachedNodes = manifold2node.get(nodeDef.manifold);
    if (cachedNodes == null) {
      addMesh(doc, node, nodeDef.manifold, backupMaterial);
      const cache = new Map<GLTFMaterial, Node>();
      cache.set(backupMaterial, node);
      manifold2node.set(nodeDef.manifold, cache);
    } else {
      const cachedNode = cachedNodes.get(backupMaterial);
      if (cachedNode == null) {
        const [oldBackupMaterial, oldNode] = cachedNodes.entries().next().value;
        cloneNodeNewMaterial(
            doc, node, oldNode, getCachedMaterial(doc, backupMaterial),
            getCachedMaterial(doc, oldBackupMaterial));
        cachedNodes.set(backupMaterial, node);
      } else {
        cloneNode(node, cachedNode);
      }
    }
  }
  return node;
}

async function exportGLB(manifold?: Manifold) {
  const doc = new Document();
  const halfRoot2 = Math.sqrt(2) / 2;
  const mm2m = 1 / 1000;
  const wrapper = doc.createNode('wrapper')
                      .setRotation([-halfRoot2, 0, 0, halfRoot2])
                      .setScale([mm2m, mm2m, mm2m]);
  doc.createScene().addChild(wrapper);

  if (nodes.length > 0) {
    const node2gltf = new Map<GLTFNode, Node>();
    const manifold2node = new Map<Manifold, Map<GLTFMaterial, Node>>();
    let leafNodes = 0;

    for (const nodeDef of nodes) {
      node2gltf.set(nodeDef, createNodeFromCache(doc, nodeDef, manifold2node));
      if (nodeDef.manifold) {
        ++leafNodes;
      }
    }

    for (const nodeDef of nodes) {
      const gltfNode = node2gltf.get(nodeDef)!;
      if (nodeDef.parent == null) {
        wrapper.addChild(gltfNode);
      } else {
        node2gltf.get(nodeDef.parent)!.addChild(gltfNode);
      }
    }

    log('Total glTF nodes: ', nodes.length,
        ', Total mesh references: ', leafNodes);
  } else {
    if (manifold == null) {
      log('No output because "result" is undefined and no "GLTFNode"s were created.');
      return;
    }
    const node = doc.createNode();
    addMesh(doc, node, manifold);
    wrapper.addChild(node);
  }

  const glb = await io.writeBinary(doc);

  const blob = new Blob([glb], {type: 'application/octet-stream'});
  self.postMessage({objectURL: URL.createObjectURL(blob)});
}
