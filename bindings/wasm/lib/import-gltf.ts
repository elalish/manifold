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

import type * as GLTFTransform from '@gltf-transform/core'
import {WebIO} from '@gltf-transform/core';
import {clearNodeTransform} from '@gltf-transform/functions';

import {Vec3} from '../manifold-global-types';

import {Properties, readMesh, setupIO} from './gltf-io.ts';
import {NonManifoldGLTFNode} from './gltf-node.ts';
import {setMaterialByID} from './material.ts';
import {getManifoldModuleSync} from './wasm.ts';

const documents = new Set<GLTFTransform.Document>();
const id2node = new Map<number, GLTFTransform.Node>();
const id2properties = new Map<number, Properties>();
const id2document = new Map<number, GLTFTransform.Document>();

export const extensions: Array<string> = ['glb', 'gltf'];

export const cleanup = () => {
  documents.clear();
  id2document.clear();
  id2node.clear();
  id2properties.clear();
};

export const fetchAsGLTFNodeList = async (url: string) => {
  const io = setupIO(new WebIO());
  const doc = await io.read(url);
  documents.add(doc);
  const docNodes = doc.getRoot().listNodes() ?? [];
  return docNodes!.map(docNode => {
    const node = new NonManifoldGLTFNode();
    node.node = clearNodeTransform(docNode);
    node.document = doc;
    node.name = docNode.getName();
    return node;
  });
};

/**
 * Import a model for display.
 *
 * @group Modelling Functions
 * @param url
 * @returns
 */
export const importModel = async(url: string): Promise<NonManifoldGLTFNode> => {
  const [firstNode] = await fetchAsGLTFNodeList(url);
  return firstNode;
};

/**
 * @internal
 */
export const getPropertiesByID = (runID: number) => id2properties.get(runID);

/**
 * @internal
 */
export const getDocumentByID = (runID: number) => id2document.get(runID);

/**
 * Convert a single gltf-transform node to a Manifold object.
 *
 * If a node has no mesh, the mesh has no geometry, or the mesh is not manifold,
 * the result will be `null`.  Other errors will be re-thrown for the caller to
 * handle.
 *
 * Each primitive in a gltf-transform mesh may have its own material and
 * attributes.  Those primitives become runs once translated into Manifold.
 * Each run may have a different material attached.  ManifoldCAD can manage this
 * case, although there is not a user-facing way to quickly assign materials to
 * parts of a model.  This function will index the original materials
 * (properties) to be copied into an exported GLTF document.  It will also set
 * ManifoldCAD materials (a subset of GLTF materials) as a fallback.
 *
 * @param document The gltf-transform document containing the node.
 * @param node The node to convert.
 * @returns A Manifold object if possible, `null` if not.
 */
export function gltfMeshToManifold(
    document: GLTFTransform.Document, node: GLTFTransform.Node) {
  const {Manifold, Mesh} = getManifoldModuleSync()!;

  const gltfmesh = node.getMesh();
  if (!gltfmesh) return null;
  const {mesh, runProperties} = readMesh(gltfmesh)!;

  // Get a a reserved ID from manifold for each run.
  const numID = runProperties.length;
  const firstID = Manifold.reserveIDs(numID);
  mesh.runOriginalID = new Uint32Array(numID);

  // Iterate through each primitive.
  for (let primitiveID = 0; primitiveID < numID; ++primitiveID) {
    // Set the manifold runID.  This will be parsed by `new Mesh()`.
    const runID = firstID + primitiveID;
    mesh.runOriginalID[primitiveID] = runID;
    const properties = runProperties[primitiveID]

    // Save these for later lookup.
    id2document.set(runID, document);
    id2node.set(runID, node);
    id2properties.set(runID, properties);

    // Import what we can as a manifoldCAD material.
    // This is really a fallback.  Ideally, we will copy the material
    // from the source document into the destination.
    const {attributes, material} = properties;
    setMaterialByID(runID, {
      // 'POSITION' is always present as an attribute; we don't need to
      // specify it.
      attributes: attributes.filter(x => x !== 'POSITION'),
      alpha: material.getAlpha(),
      baseColorFactor: material.getBaseColorFactor() as any as Vec3,
      metallic: material.getMetallicFactor(),
      roughness: material.getRoughnessFactor(),
      name: material.getName(),
    });
  }

  const manifoldMesh = new Mesh(mesh);
  try {
    const manifold = new Manifold(manifoldMesh);
    if (manifold && !manifold.isEmpty()) {
      return manifold;
    }
  } catch (e) {
    if ((e as any)?.name === 'ManifoldError' ||
        (e as any)?.code === 'NotManifold') {
      console.log(`Skipping non-manifold import`);
    } else {
      throw e;
    }
  }

  return null;
};
