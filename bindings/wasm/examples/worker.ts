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

import Module from './built/manifold.js';
//@ts-ignore
import {setupIO, writeMesh} from './gltf-io.js';
import {GLTFMaterial, Quat} from './public/editor.js';
import {Manifold, ManifoldStatic, Mesh, Vec3} from './public/manifold.js';

interface WorkerStatic extends ManifoldStatic {
  GLTFNode: typeof GLTFNode;
  show(manifold: Manifold): Manifold;
  only(manifold: Manifold): Manifold;
  setMaterial(manifold: Manifold, material: GLTFMaterial): void;
  cleanup(): void;
}

const module = await Module() as WorkerStatic;
module.setup();

// Faster on modern browsers than Float32Array
glMatrix.glMatrix.setMatrixArrayType(Array);

const io = setupIO(new WebIO());
io.registerExtensions(KHRONOS_EXTENSIONS);

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

function debug(manifold: Manifold, map: Map<number, Mesh>) {
  const result = manifold.asOriginal();
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

// manifold member functions that returns a new manifold
const memberFunctions = [
  'add', 'subtract', 'intersect', 'trimByPlane', 'refine', 'warp',
  'setProperties', 'transform', 'translate', 'rotate', 'scale', 'mirror',
  'asOriginal', 'decompose'
];
// top level functions that constructs a new manifold
const constructors = [
  'cube', 'cylinder', 'sphere', 'tetrahedron', 'extrude', 'revolve', 'union',
  'difference', 'intersection', 'compose', 'levelSet', 'smooth', 'show', 'only',
  'setMaterial'
];
const utils = [
  'setMinCircularAngle', 'setMinCircularEdgeLength', 'setCircularSegments',
  'getCircularSegments', 'Mesh', 'GLTFNode'
];
const exposedFunctions = constructors.concat(utils);

// Setup memory management, such that users don't have to care about
// calling `delete` manually.
// Note that this only fixes memory leak across different runs: the memory
// will only be freed when the compilation finishes.

const manifoldRegistry = new Array<Manifold>();
for (const name of memberFunctions) {
  //@ts-ignore
  const originalFn = module.Manifold.prototype[name];
  //@ts-ignore
  module.Manifold.prototype['_' + name] = originalFn;
  //@ts-ignore
  module.Manifold.prototype[name] = function(...args: any) {
    //@ts-ignore
    const result = this['_' + name](...args);
    manifoldRegistry.push(result);
    return result;
  };
}

for (const name of constructors) {
  //@ts-ignore
  const originalFn = module[name];
  //@ts-ignore
  module[name] = function(...args: any) {
    const result = originalFn(...args);
    manifoldRegistry.push(result);
    return result;
  };
}

module.cleanup = function() {
  for (const obj of manifoldRegistry) {
    // decompose result is an array of manifolds
    if (obj instanceof Array)
      for (const elem of obj) elem.delete();
    else
      obj.delete();
  }
  manifoldRegistry.length = 0;
};

// Setup complete
postMessage(null);

const oldLog = console.log;
console.log = function(...args) {
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
  postMessage({log: message});
  oldLog(...args);
};

onmessage = async (e) => {
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
    postMessage({objectURL: null});
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
}: GLTFMaterial = {}) {
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
    console.log('Empty manifold, skipping.');
    return;
  }

  console.log(`Triangles: ${numTri.toLocaleString()}`);
  const box = manifold.boundingBox();
  const size = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    size[i] = Math.round((box.max[i] - box.min[i]) * 10) / 10;
  }
  console.log(`Bounding Box: X = ${size[0].toLocaleString()} mm, Y = ${
      size[1].toLocaleString()} mm, Z = ${size[2].toLocaleString()} mm`);
  const volume = Math.round(manifold.getProperties().volume / 10);
  console.log(`Genus: ${manifold.genus().toLocaleString()}, Volume: ${
      (volume / 100).toLocaleString()} cm^3`);

  // From Z-up to Y-up (glTF)
  const manifoldMesh = manifold.getMesh();

  const materials = new Array<GLTFMaterial>();
  const attributes = ['POSITION', ...backupMaterial.attributes ?? []];
  for (const id of manifoldMesh.runOriginalID!) {
    const material = id2material.get(id) || backupMaterial;
    materials.push(material);
    if (material.attributes != null &&
        material.attributes.length > attributes.length) {
      attributes.splice(1, Infinity, ...material.attributes);
    }
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

    console.log(
        'Total glTF nodes: ', nodes.length,
        ', Total mesh references: ', leafNodes);
  } else {
    if (manifold == null) {
      console.log(
          'No output because "result" is undefined and no "GLTFNode"s were created.');
      return;
    }
    const node = doc.createNode();
    addMesh(doc, node, manifold);
    wrapper.addChild(node);
  }

  const glb = await io.writeBinary(doc);

  const blob = new Blob([glb], {type: 'application/octet-stream'});
  postMessage({objectURL: URL.createObjectURL(blob)});
}