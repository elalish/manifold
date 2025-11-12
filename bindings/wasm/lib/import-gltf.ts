// Copyright 2024-25 The Manifold Authors.
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

// import type {Manifold} from '../manifold.d.ts';

// import { getManifoldModule } from './wasm.ts';
import {Document, WebIO} from '@gltf-transform/core';
import {clearNodeTransform} from '@gltf-transform/functions';

import {setupIO} from './gltf-io.ts';
import {NonManifoldGLTFNode} from './gltf-node.ts';

// const manifoldModule = await getManifoldModule();
// const {Manifold, Mesh} = manifoldModule;

// Set up gltf-transform

// Map of OriginalID to glTF material and attributes
// const id2properties = new Map<number, Properties>();

const importedDocuments = new Set<Document>()

export const extensions: Array<string> = ['glb', 'gltf'];

export const cleanup = () => {
  importedDocuments.clear()
};

export const fetchAsGLTFNode = async (url: string) => {
  const io = setupIO(new WebIO());

  const doc = await io.read(url);
  importedDocuments.add(doc);
  const docNodes = doc.getRoot().listNodes() ?? [];
  return docNodes!.map(docNode => {
    const node = new NonManifoldGLTFNode();
    node.gltfTransformNode = clearNodeTransform(docNode);
    node.gltfDocument = doc;
    return node;
  });
};
/*
export const fetchAsManifold = async (url: string) => {
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
  //const startIdx = doc.getRoot().listMaterials().length;
  //mergeDocuments(doc, docIn);
  //doc.getRoot().listScenes().forEach((s) => s.dispose());
  //doc.getRoot().listBuffers().forEach((s) => s.dispose());
  //doc.getRoot().listAccessors().forEach((s) => s.dispose());
  //for (const [i, id] of ids.entries()) {
  //  const material = doc.getRoot().listMaterials()[startIdx + i];
  //  id2properties.get(id)!.material = material;
  //}

  return Manifold.union(manifolds);
}
  */