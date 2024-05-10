// Copyright 2024 The Manifold Authors.
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

import {Document, WebIO} from '@gltf-transform/core';
import {clearNodeTransform, flatten, prune} from '@gltf-transform/functions';

import Module, {Manifold, Mesh} from './built/manifold';
import {disposeMesh, Properties, readMesh, setupIO, writeMesh} from './gltf-io';

// Set up gltf-transform
const io = setupIO(new WebIO());
const doc = new Document();


// Set up Manifold WASM library
const wasm = await Module();
wasm.setup();
const {Manifold, Mesh} = wasm;

// Map of OriginalID to glTF material and attributes
const id2properties = new Map<number, Properties>();

// Wrapper for gltf-io readMesh() that processes a whole glTF and stores the
// properties in the above map. This function is simplified and intended only
// for reading single-object glTFs, as it simply unions any extra meshes
// together rather than returning a scene hierarchy.
async function readGLB(url: string) {
  const manifolds = Array<Manifold>();
  const docIn = await io.read(url);
  await docIn.transform(flatten());
  const nodes = docIn.getRoot().listNodes();
  const ids = Array<number>();
  for (const node of nodes) {
    clearNodeTransform(node);
    const gltfMesh = node.getMesh();
    if (gltfMesh == null) continue;
    const tmp = readMesh(gltfMesh);
    if (tmp == null) continue;

    const numID = tmp.runProperties.length;
    const firstID = Manifold.reserveIDs(numID);
    tmp.mesh.runOriginalID = new Uint32Array(numID);
    for (let i = 0; i < numID; ++i) {
      tmp.mesh.runOriginalID[i] = firstID + i;
      ids.push(firstID + i);
      id2properties.set(firstID + i, tmp.runProperties[i]);
    }
    manifolds.push(new Manifold(new Mesh(tmp.mesh)));
  }
  // pull in materials, TODO: replace with transfer() when available
  const startIdx = doc.getRoot().listMaterials().length;
  doc.merge(docIn);
  doc.getRoot().listScenes().forEach((s) => s.dispose());
  doc.getRoot().listBuffers().forEach((s) => s.dispose());
  doc.getRoot().listAccessors().forEach((s) => s.dispose());
  for (const [i, id] of ids.entries()) {
    const material = doc.getRoot().listMaterials()[startIdx + i];
    id2properties.get(id)!.material = material;
  }

  return Manifold.union(manifolds);
}

// Read static input glTFs
const space = await readGLB('/models/space.glb');
const moon = await readGLB('/models/moon.glb');

const node = doc.createNode();
doc.createScene().addChild(node);

// Set up UI for operations
type BooleanOp = 'union'|'difference'|'intersection';

function csg(operation: BooleanOp) {
  push2MV(Manifold[operation](space, moon));
}

csg('difference');
const selectElement = document.querySelector('select')!;
selectElement.onchange = function() {
  csg(selectElement.value as BooleanOp);
};

// The resulting glTF
let objectURL = '';

// Set up download UI
const downloadButton = document.querySelector('#download') as HTMLButtonElement;
downloadButton.onclick = function() {
  const link = document.createElement('a');
  link.download = 'manifold.glb';
  link.href = objectURL;
  link.click();
};

// <model-viewer> element for rendering resulting glTF.
const mv = document.querySelector('model-viewer');

// Use gltf-io and gltf-transform to convert the resulting Manifold to a glTF
// and display it with <model-viewer>.
async function push2MV(manifold: Manifold) {
  disposeMesh(node.getMesh()!);
  const mesh = writeMesh(doc, manifold.getMesh(), id2properties);
  node.setMesh(mesh);
  await doc.transform(prune());

  const glb = await io.writeBinary(doc);

  const blob = new Blob([glb], {type: 'application/octet-stream'});
  URL.revokeObjectURL(objectURL);
  objectURL = URL.createObjectURL(blob);
  (mv as any).src = objectURL;
}