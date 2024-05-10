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

import Module from './built/manifold.js';
import {disposeMesh, readMesh, setupIO, writeMesh} from './gltf-io';

const io = setupIO(new WebIO());
io.registerExtensions(KHRONOS_EXTENSIONS);

const wasm = await Module();
wasm.setup();
const {Manifold, Mesh} = wasm;

const mv = document.querySelector('model-viewer');
const inputEl = document.querySelector('#input') as HTMLInputElement;
const downloadButton = document.querySelector('#download') as HTMLButtonElement;
const checkbox = document.querySelector('#viewFinal') as HTMLInputElement;
const dropCtrl = new SimpleDropzone(mv, inputEl) as any;

let inputGLBurl = '';
let outputGLBurl = '';
let allManifold = true;
let anyManifold = false;

dropCtrl.on('drop', async ({files}) => {
  for (const [path, file] of files) {
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

async function writeGLB(doc: Document) {
  URL.revokeObjectURL(outputGLBurl);
  if (!anyManifold) {
    return;
  }
  const glb = await io.writeBinary(doc);

  const blob = new Blob([glb], {type: 'application/octet-stream'});
  outputGLBurl = URL.createObjectURL(blob);
}

async function readGLB(url: string) {
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

    const id2properties = new Map();
    const numID = tmp.runProperties.length;
    const firstID = Manifold.reserveIDs(numID);
    tmp.mesh.runOriginalID = new Uint32Array(numID);
    for (let i = 0; i < numID; ++i) {
      tmp.mesh.runOriginalID[i] = firstID + i;
      id2properties.set(firstID + i, tmp.runProperties[i]);
    }
    const manifoldMesh = new Mesh(tmp.mesh);
    disposeMesh(mesh);

    manifoldMesh.merge();

    try {
      // Test manifoldness - will throw if not.
      const manifold = new Manifold(manifoldMesh);
      node.setMesh(writeMesh(docIn, manifold.getMesh(), id2properties));
      manifold.delete();
      anyManifold = true;
    } catch (e) {
      console.log(mesh.getName(), e);
      allManifold = false;
    }
  }

  await docIn.transform(prune());

  return docIn;
}