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

import {Accessor} from '@gltf-transform/core';

import {EXTManifold} from './manifold-gltf.js';

export const attributeDefs = {
  'POSITION': {type: Accessor.Type.VEC3, components: 3},
  'NORMAL': {type: Accessor.Type.VEC3, components: 3},
  'TANGENT': {type: Accessor.Type.VEC4, components: 4},
  'TEXCOORD_0': {type: Accessor.Type.VEC2, components: 2},
  'TEXCOORD_1': {type: Accessor.Type.VEC2, components: 2},
  'COLOR_0': {type: Accessor.Type.VEC3, components: 3},
  'JOINTS_0': {type: Accessor.Type.VEC4, components: 4},
  'WEIGHTS_0': {type: Accessor.Type.VEC4, components: 4},
  'SKIP_1': {type: null, components: 1},
  'SKIP_2': {type: null, components: 2},
  'SKIP_3': {type: null, components: 3},
  'SKIP_4': {type: null, components: 4},
};

export function setupIO(io) {
  return io.registerExtensions([EXTManifold]);
}

function readPrimitive(primitive, numProp, attributes) {
  const position = primitive.getAttribute('POSITION');
  const numVert = position.getCount();
  const vertProperties = [];
  let offset = 0;
  for (const attribute of attributes) {
    const size = attributeDefs[attribute].components;
    if (attribute === 'SKIP') {
      offset += size;
      continue;
    }
    const accessor = primitive.getAttribute(attribute);
    if (accessor) {
      const array = accessor.getArray();
      for (let i = 0; i < numVert; ++i) {
        for (let j = 0; j < size; ++j) {
          vertProperties[numProp * i + offset + j] = array[i * size + j];
        }
      }
    } else {
      for (let i = 0; i < numVert; ++i) {
        for (let j = 0; j < size; ++j) {
          vertProperties[numProp * i + offset + j] = 0;
        }
      }
    }
    offset += size;
  }
  return vertProperties;
}

export function readMesh(mesh, attributes, materials) {
  const primitives = mesh.listPrimitives();
  if (primitives.length === 0) {
    return {};
  }

  if (attributes.length === 0) {
    const attributeSet = new Set();
    for (const primitive of primitives) {
      const semantics = primitive.listSemantics();
      for (const semantic of semantics) {
        attributeSet.add(semantic);
      }
    }
    for (const semantic in attributeDefs) {
      if (attributeSet.has(semantic)) {
        attributes.push(semantic);
        attributeSet.delete(semantic);
      }
    }
    for (const semantic of attributeSet.keys()) {
      attributes.push(semantic);
    }
  }

  if (attributes.length < 1 || attributes[0] !== 'POSITION')
    throw new Error('First attribute must be "POSITION".');

  const numProp = attributes.map((def) => attributeDefs[def].components)
                      .reduce((a, b) => a + b);

  const manifoldPrimitive = mesh.getExtension('EXT_manifold');

  let vertPropArray = [];
  let triVertArray = [];
  const runIndexArray = [0];
  const mergeFromVert = [];
  const mergeToVert = [];
  if (manifoldPrimitive != null) {
    vertPropArray = readPrimitive(primitives[0], numProp, attributes);
    for (const primitive of primitives) {
      triVertArray = [...triVertArray, ...primitive.getIndices().getArray()];
      runIndexArray.push(triVertArray.length);
      materials.push(primitive.getMaterial());
    }
    const mergeTriVert = manifoldPrimitive.getMergeIndices()?.getArray() ?? [];
    const mergeTo = manifoldPrimitive.getMergeValues()?.getArray() ?? [];
    const vert2merge = new Map();
    for (const [i, idx] of mergeTriVert.entries()) {
      vert2merge.set(triVertArray[idx], mergeTo[i]);
    }
    for (const [from, to] of vert2merge.entries()) {
      mergeFromVert.push(from);
      mergeToVert.push(to);
    }
  } else {
    for (const primitive of primitives) {
      const numVert = vertPropArray.length / numProp;
      vertPropArray =
          [...vertPropArray, ...readPrimitive(primitive, numProp, attributes)];
      triVertArray = [
        ...triVertArray,
        ...primitive.getIndices().getArray().map((i) => i + numVert)
      ];
      runIndexArray.push(triVertArray.length);
      materials.push(primitive.getMaterial());
    }
  }
  const vertProperties = new Float32Array(vertPropArray);
  const triVerts = new Uint32Array(triVertArray);
  const runIndex = new Uint32Array(runIndexArray);

  return {
    numProp,
    triVerts,
    vertProperties,
    runIndex,
    mergeFromVert,
    mergeToVert
  };
}

export function writeMesh(doc, manifoldMesh, attributes, materials) {
  if (doc.getRoot().listBuffers().length === 0) {
    doc.createBuffer();
  }
  const buffer = doc.getRoot().listBuffers()[0];
  const manifoldExtension = doc.createExtension(EXTManifold);

  const attributeUnion = [];
  for (const matAttributes of attributes) {
    matAttributes.forEach((attribute, i) => {
      if (i >= attributeUnion.length) {
        attributeUnion.push(attribute);
      } else {
        const size = attributeDefs[attribute].components;
        const unionSize = attributeDefs[attributeUnion[i]].components;
        if (size != unionSize) {
          throw new Error(
              'Attribute sizes do not correspond: ', attribute, ' and ',
              attributeUnion[i]);
        }
        if (attributeDefs[attributeUnion[i]].type == null) {
          attributeUnion[i] = attribute;
        }
      }
    });
  }
  if (attributeUnion.length < 1 || attributeUnion[0] !== 'POSITION')
    throw new Error('First attribute must be "POSITION".');

  const mesh = doc.createMesh();
  const numPrimitive = manifoldMesh.runIndex.length - 1;
  for (let run = 0; run < numPrimitive; ++run) {
    const indices = doc.createAccessor('indices')
                        .setBuffer(buffer)
                        .setType(Accessor.Type.SCALAR)
                        .setArray(new Uint32Array());
    const primitive = doc.createPrimitive().setIndices(indices);
    const material = materials[run];
    if (material) {
      primitive.setMaterial(material);
    }
    mesh.addPrimitive(primitive);
  }

  const numVert = manifoldMesh.numVert;
  const numProp = manifoldMesh.numProp;
  let offset = 0;
  attributeUnion.forEach((attribute, aIdx) => {
    if (attribute === 'SKIP') {
      ++offset;
      return;
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

    mesh.listPrimitives().forEach((primitive, pIdx) => {
      if (attributes[pIdx].length > aIdx &&
          attributeDefs[attributes[pIdx][aIdx]].type != null) {
        primitive.setAttribute(attribute, accessor);
      }
    });
    offset += n;
  });

  const manifoldPrimitive = manifoldExtension.createManifoldPrimitive();
  mesh.setExtension('EXT_manifold', manifoldPrimitive);

  const indices = doc.createAccessor('indices')
                      .setBuffer(buffer)
                      .setType(Accessor.Type.SCALAR)
                      .setArray(manifoldMesh.triVerts);
  manifoldPrimitive.setIndices(indices);
  manifoldPrimitive.setRunIndex(manifoldMesh.runIndex);

  const vert2merge = [...Array(manifoldMesh.numVert).keys()];
  for (const [i, from] of manifoldMesh.mergeFromVert.entries()) {
    vert2merge[from] = manifoldMesh.mergeToVert[i];
  }
  const ind = [];
  const val = [];
  for (const [i, vert] of manifoldMesh.triVerts.entries()) {
    const newVert = vert2merge[vert];
    if (vert !== newVert) {
      ind.push(i);
      val.push(newVert);
    }
  }
  const indicesAccessor = doc.createAccessor()
                              .setBuffer(buffer)
                              .setType(Accessor.Type.SCALAR)
                              .setArray(new Uint32Array(ind));
  const valuesAccessor = doc.createAccessor()
                             .setBuffer(buffer)
                             .setType(Accessor.Type.SCALAR)
                             .setArray(new Uint32Array(val));
  manifoldPrimitive.setMerge(indicesAccessor, valuesAccessor);

  return mesh;
}

export function disposeMesh(mesh) {
  if (!mesh) return;
  const primitives = mesh.listPrimitives();
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
