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

import * as glMatrix from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.3/+esm';
import {Accessor, Document, Material, WebIO} from 'https://cdn.skypack.dev/pin/@gltf-transform/core@v3.0.0-SfbIFhNPTRdr1UE2VSan/mode=imports,min/optimized/@gltf-transform/core.js';

import Module from '../manifold.js';

const wasm = await Module();
wasm.setup();

// Faster on modern browsers than Float32Array
glMatrix.glMatrix.setMatrixArrayType(Array);

// Scene setup
const io = new WebIO();
const doc = new Document();

const buffer = doc.createBuffer();
const position =
    doc.createAccessor().setBuffer(buffer).setType(Accessor.Type.VEC3);
const indices =
    doc.createAccessor().setBuffer(buffer).setType(Accessor.Type.SCALAR);
const material = doc.createMaterial()
                     .setBaseColorFactor([1, 1, 0, 1])
                     .setMetallicFactor(1)
                     .setRoughnessFactor(0.2);
const primitive = doc.createPrimitive()
                      .setMaterial(material)
                      .setIndices(indices)
                      .setAttribute('POSITION', position);
const mesh = doc.createMesh().addPrimitive(primitive);
const node = doc.createNode('result').setMesh(mesh);
const scene = doc.createScene().addChild(node);

// Debug setup to show source meshes
const shown = new Map();
const debugMaterial = doc.createMaterial()
                          .setBaseColorFactor([1, 0, 0, 0.25])
                          .setAlphaMode(Material.AlphaMode.BLEND)
                          .setDoubleSided(true)
                          .setMetallicFactor(0);

wasm.show = (manifold) => {
  const position =
      doc.createAccessor().setBuffer(buffer).setType(Accessor.Type.VEC3);
  const indices =
      doc.createAccessor().setBuffer(buffer).setType(Accessor.Type.SCALAR);

  const primitive = doc.createPrimitive()
                        .setMaterial(debugMaterial)
                        .setIndices(indices)
                        .setAttribute('POSITION', position);
  const mesh = doc.createMesh('debug').addPrimitive(primitive);

  const result = manifold.asOriginal();
  outputMesh(indices, position, result.getMesh());

  shown.set(result.originalID(), mesh);
  return result;
};

// manifold member functions that returns a new manifold
const memberFunctions = [
  'add', 'subtract', 'intersect', 'refine', 'transform', 'translate', 'rotate',
  'scale', 'asOriginal', 'decompose'
];
// top level functions that constructs a new manifold
const constructors = [
  'cube', 'cylinder', 'sphere', 'tetrahedron', 'extrude', 'revolve', 'union',
  'difference', 'intersection', 'compose', 'levelSet', 'smooth', 'show'
];
const utils = [
  'setMinCircularAngle', 'setMinCircularEdgeLength', 'setCircularSegments',
  'getCircularSegments', 'Mesh'
];
const exposedFunctions = constructors.concat(utils);

// Setup memory management, such that users don't have to care about
// calling `delete` manually.
// Note that this only fixes memory leak across different runs: the memory
// will only be freed when the compilation finishes.

const manifoldRegistry = [];
for (const name of memberFunctions) {
  const originalFn = wasm.Manifold.prototype[name];
  wasm.Manifold.prototype['_' + name] = originalFn;
  wasm.Manifold.prototype[name] = function(...args) {
    const result = this['_' + name](...args);
    manifoldRegistry.push(result);
    return result;
  };
}

for (const name of constructors) {
  const originalFn = wasm[name];
  wasm[name] = function(...args) {
    const result = originalFn(...args);
    manifoldRegistry.push(result);
    return result;
  };
}

wasm.cleanup = function() {
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
    if (typeof arg == 'object') {
      message += JSON.stringify(arg, null, 4);
    } else {
      message += arg.toString();
    }
  }
  postMessage({log: message});
  oldLog(...args);
};

onmessage = async (e) => {
  const content = e.data + '\nreturn exportGLB(result);\n';
  try {
    const f =
        new Function('exportGLB', 'glMatrix', ...exposedFunctions, content);
    await f(exportGLB, glMatrix, ...exposedFunctions.map(name => wasm[name]));
  } catch (error) {
    console.log(error.toString());
    postMessage({objectURL: null});
  } finally {
    wasm.cleanup();
  }
};

function outputMesh(indices, position, mesh) {
  indices.setArray(mesh.triVerts);

  const numVert = mesh.numVert;
  const numProp = mesh.numProp;
  const posArray = new Float32Array(3 * numVert);
  for (let i = 0; i < numVert; ++i) {
    posArray[3 * i] = mesh.vertProperties[numProp * i];
    posArray[3 * i + 1] = mesh.vertProperties[numProp * i + 1];
    posArray[3 * i + 2] = mesh.vertProperties[numProp * i + 2];
  }
  position.setArray(posArray);
}

async function exportGLB(manifold) {
  console.log(`Triangles: ${manifold.numTri().toLocaleString()}`);
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
  const mesh = manifold.rotate([-90, 0, 0]).getMesh();

  outputMesh(indices, position, mesh);

  for (const [run, id] of mesh.runOriginalID.entries()) {
    const outMesh = shown.get(id);
    if (outMesh == null) {
      continue;
    }
    const node =
        doc.createNode('debug').setMesh(outMesh).setMatrix(mesh.transform(run));
    scene.addChild(node);
  }

  const glb = await io.writeBinary(doc);

  const blob = new Blob([glb], {type: 'application/octet-stream'});
  postMessage({objectURL: URL.createObjectURL(blob)});

  // Clean up debug nodes
  for (const [id, mesh] of shown) {
    const primitive = mesh.listPrimitives()[0];
    primitive.getAttribute('POSITION').dispose();
    primitive.getIndices().dispose();
    primitive.dispose();
    mesh.dispose();
  }

  scene.traverse((node) => {
    if (node.getName() == 'debug') {
      node.dispose();
    }
  });
  shown.clear();
}