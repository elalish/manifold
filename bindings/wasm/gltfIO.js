// Copyright 2023 The Manifold Authors.
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

import {Accessor, Document, WebIO} from '@gltf-transform/core';

import {EXTManifold} from './manifold-gltf.js';

const io = new WebIO().registerExtensions([EXTManifold]);
const doc = new Document();
const manifoldExtension = doc.createExtension(EXTManifold);
doc.createBuffer();
const node = doc.createNode();
doc.createScene().addChild(node);

export const attributeDefs = {
  'POSITION': {type: Accessor.Type.VEC3, components: 3},
  'SKIP': {type: null, components: 1},
  'NORMAL': {type: Accessor.Type.VEC3, components: 3},
  'TANGENT': {type: Accessor.Type.VEC4, components: 4},
  'TEXCOORD_0': {type: Accessor.Type.VEC2, components: 2},
  'TEXCOORD_1': {type: Accessor.Type.VEC2, components: 2},
  'COLOR_0': {type: Accessor.Type.VEC3, components: 3},
  'JOINTS_0': {type: Accessor.Type.VEC4, components: 4},
  'WEIGHTS_0': {type: Accessor.Type.VEC4, components: 4},
};

export function getGLTFDoc() {
  return doc;
}

export function getRootNode() {
  return node;
}

export async function writeGLB() {
  return io.writeBinary(doc);
}

export function toGLTFMesh(manifoldMesh, attributeArray, materialArray) {
  if (doc.getRoot().listBuffers().length !== 1)
    throw new Error('Document must have a single buffer.');
  const buffer = doc.getRoot().listBuffers()[0];

  const indices = doc.createAccessor('indices')
                      .setBuffer(buffer)
                      .setType(Accessor.Type.SCALAR)
                      .setArray(manifoldMesh.triVerts);

  const mesh = doc.createMesh();
  const numPrimitive = manifoldMesh.runIndex.length - 1;
  if (materialArray.length != numPrimitive)
    throw new Error('materialArray does not match number of triangle runs.');

  for (let run = 0; run < numPrimitive; ++run) {
    const primitive = doc.createPrimitive().setIndices(indices).setMaterial(
        materialArray[run]);
    mesh.addPrimitive(primitive);
  }

  if (attributeArray.length < 1 || attributeArray[0] !== 'POSITION')
    throw new Error('First attribute must be "POSITION".');

  const numVert = manifoldMesh.numVert;
  const numProp = manifoldMesh.numProp;
  let offset = 0;
  for (const attribute of attributeArray) {
    if (attribute === 'SKIP') {
      ++offset;
      continue;
    }
    const def = attributeDefs[attribute];
    if (def == null)
      throw new Error(attribute + ' is not a recognized attribute.');

    const n = def.components;
    if (offset + n > numProp) throw new Error('Too many attribute channels.');

    const array = new Float32Array(n * numVert);
    for (let v = 0; v < numVert; ++v) {
      for (let i = 0; i < n; ++i) {
        array[n * v + i] =
            manifoldMesh.vertProperties[numProp * v + offset + i];
      }
    }

    const accessor =
        doc.createAccessor().setBuffer(buffer).setType(def.type).setArray(
            array);
    for (const primitive of mesh.listPrimitives()) {
      primitive.setAttribute(attribute, accessor);
    }
    offset += n;
  }

  const manifoldPrimitive = manifoldExtension.createManifoldPrimitive();
  mesh.setExtension('EXT_manifold', manifoldPrimitive);

  const primitive = doc.createPrimitive().setIndices(indices).setAttribute(
      'POSITION', mesh.listPrimitives()[0].getAttribute('POSITION'));
  manifoldPrimitive.setPrimitive(primitive, manifoldMesh.runIndex);

  const idx = [...manifoldMesh.mergeTriVert.keys()];
  idx.sort(
      (a, b) => manifoldMesh.mergeTriVert[a] - manifoldMesh.mergeTriVert[b]);
  const mergeTriVert =
      new Uint32Array(idx.map(i => manifoldMesh.mergeTriVert[i]));
  const mergeToVert =
      new Uint32Array(idx.map(i => manifoldMesh.mergeToVert[i]));
  const indicesAccessor = doc.createAccessor()
                              .setBuffer(buffer)
                              .setType(Accessor.Type.SCALAR)
                              .setArray(mergeTriVert);
  const valuesAccessor = doc.createAccessor()
                             .setBuffer(buffer)
                             .setType(Accessor.Type.SCALAR)
                             .setArray(mergeToVert);
  manifoldPrimitive.setMerge(indicesAccessor, valuesAccessor);

  return mesh;
}

export function disposeMesh(mesh) {
  if (!mesh) return;
  const primitives = mesh.listPrimitives();
  const manifoldPrimitive = mesh.getExtension('EXT_manifold');
  if (manifoldPrimitive) {
    manifoldPrimitive.getMergeIndices()?.dispose();
    manifoldPrimitive.getMergeValues()?.dispose();
    primitives.push(manifoldPrimitive.getPrimitive());
  }

  for (const primitive of primitives) {
    primitive.getIndices()?.dispose();
    for (const accessor of primitive.listAttributes()) {
      accessor.dispose();
    }
  }

  mesh.dispose();
}

export async function loadTexture(texture, uri) {
  const response = await fetch(uri);
  const blob = await response.blob();
  texture.setMimeType(blob.type);
  texture.setImage(new Uint8Array(await blob.arrayBuffer()));
}
