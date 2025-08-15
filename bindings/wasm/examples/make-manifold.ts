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
import {KHRONOS_EXTENSIONS} from '@gltf-transform/extensions';
import {prune} from '@gltf-transform/functions';
import {SimpleDropzone} from 'simple-dropzone';

import {disposeMesh, Properties, readMesh, setupIO, writeMesh} from '../lib/gltf-io';

import Module from './built/manifold.js';

// Set up gltf-transform
const io = setupIO(new WebIO());
io.registerExtensions(KHRONOS_EXTENSIONS);

// Set up Manifold WASM library
const wasm = await Module();
wasm.setup();
const {Manifold, Mesh} = wasm;

// UX elements
const mv = document.querySelector('model-viewer');
const inputEl = document.querySelector('#input') as HTMLInputElement;
const downloadButton = document.querySelector('#download') as HTMLButtonElement;
const checkbox = document.querySelector('#viewFinal') as HTMLInputElement;
const dropCtrl = new SimpleDropzone(mv, inputEl) as any;

// glTF objects in memory
let inputGLBurl = '';
let outputGLBurl = '';
// Status of manifoldness of all meshes
let allManifold = true;
let anyManifold = false;

// The processing is run when a glTF is drag-and-dropped onto this element.
dropCtrl.on('drop', async ({files}: {files: Map<string, File>}) => {
  for (const [_path, file] of files) {
    const filename = file.name.toLowerCase();
    if (filename.match(/\.(gltf|glb)$/)) {
      URL.revokeObjectURL(inputGLBurl);
      inputGLBurl = URL.createObjectURL(file);
      await writeGLB(await readGLB(inputGLBurl));
      updateUI();
      break;
    }
  }
});

// UI functions

function updateUI() {
  if (allManifold) {
    checkbox.checked = true;
    checkbox.disabled = true;
  } else if (anyManifold) {
    checkbox.checked = false;
    checkbox.disabled = false;
  } else {
    checkbox.checked = false;
    checkbox.disabled = true;
  }
  onClick();
}

checkbox.onclick = onClick;

function onClick() {
  (mv as any).src = checkbox.checked ? outputGLBurl : inputGLBurl;
  downloadButton.disabled = !checkbox.checked;
};

downloadButton.onclick = () => {
  const link = document.createElement('a');
  link.download = 'manifold.glb';
  link.href = outputGLBurl;
  link.click();
};

// Write output glTF using gltf-transform, which contains only the meshes that
// are manifold, and using the EXT_mesh_manifold extension.
async function writeGLB(doc: Document): Promise<void> {
  URL.revokeObjectURL(outputGLBurl);
  if (!anyManifold) {
    return;
  }
  const glb = await io.writeBinary(doc);

  const blob = new Blob(
      [glb as Uint8Array<ArrayBuffer>], {type: 'application/octet-stream'});
  outputGLBurl = URL.createObjectURL(blob);
}

// Read the glTF ObjectURL and return a gltf-transform document with all the
// non-manifold meshes stripped out.
async function readGLB(url: string): Promise<Document> {
  allManifold = false;
  anyManifold = false;
  updateUI();
  allManifold = true;
  const docIn = await io.read(url);
  const nodes = docIn.getRoot().listNodes();
  for (const node of nodes) {
    const mesh = node.getMesh();
    if (!mesh) continue;

    const tmp = readMesh(mesh);
    if (!tmp) continue;

    const id2properties = new Map<number, Properties>();
    const numID = tmp.runProperties.length;
    const firstID = Manifold.reserveIDs(numID);
    tmp.mesh.runOriginalID = new Uint32Array(numID);
    for (let i = 0; i < numID; ++i) {
      tmp.mesh.runOriginalID[i] = firstID + i;
      id2properties.set(firstID + i, tmp.runProperties[i]);
    }
    const manifoldMesh = new Mesh(tmp.mesh);
    disposeMesh(mesh);
    // Make the mesh manifold if it's close.
    manifoldMesh.merge();

    try {
      // Test manifoldness - will throw if not.
      const manifold = new Manifold(manifoldMesh);
      // Replace the mesh with a manifold version
      node.setMesh(writeMesh(docIn, manifold.getMesh(), id2properties));
      manifold.delete();
      anyManifold = true;
    } catch (e) {
      console.log(mesh.getName(), e);
      allManifold = false;
    }
  }

  // Prune the leftovers after non-manifold mesh removal.
  await docIn.transform(prune());

  return docIn;
}
