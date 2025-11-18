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

/**
 * Import models into manifoldCAD.
 *
 * ManifoldCAD uses [gltf-transform](https://gltf-transform.dev/) internally to
 * represent scenes. Importers must convert their models to in-memory
 * gltf-transform Documents.
 *
 * @packageDocumentation
 */

import type * as GLTFTransform from '@gltf-transform/core';
import {clearNodeTransform} from '@gltf-transform/functions';

import type {Manifold} from '../manifold-encapsulated-types';
import type {Vec3} from '../manifold-global-types';

import {Properties, readMesh} from './gltf-io.ts';
import {NonManifoldGLTFNode} from './gltf-node.ts';
import * as importGLTF from './import-gltf.ts';
import {setMaterialByID} from './material.ts';
import {getManifoldModuleSync} from './wasm.ts';

export interface Format {
  extension: string;
  mimetype: string;
}

export interface Importer {
  supportedFormats: Array<Format>;
  fetchModel: (uri: string) => Promise<GLTFTransform.Document>;
}

const importers: Array<Importer> = [importGLTF];

const id2node = new Map<number, GLTFTransform.Node>();
const id2properties = new Map<number, Properties>();
const id2document = new Map<number, GLTFTransform.Document>();

export const cleanup = () => {
  id2document.clear();
  id2node.clear();
  id2properties.clear();
};

/**
 * @internal
 */
export const getPropertiesByID = (runID: number) => id2properties.get(runID);

/**
 * @internal
 */
export const getDocumentByID = (runID: number) => id2document.get(runID);

export function getImporterByExtension(extension: string) {
  const hasExtension =
      (format: Format) => [format.extension, `.${format.extension}`].includes(
          extension);
  const importer = importers.find(im => im.supportedFormats.find(hasExtension));

  if (!importer) {
    const extensionList =
        importers
            .map(importer => importer.supportedFormats.map(f => f.extension))
            .reduce((acc, cur) => ([...acc, ...cur]))
            .map(ext => `\`.${ext}\``)
            .reduceRight(
                (prev, cur, index) => cur + (index ? ', or ' : ', ') + prev);
    throw new Error(
        `Cannot import \`${extension}\`.  ` +
        `Format must be one of ${extensionList}`);
  }
  return importer;
}

/**
 *
 * @group Modelling Functions
 * @param uri
 * @returns
 */
export async function importModel(uri: string): Promise<NonManifoldGLTFNode> {
  const importer = importers[0];
  const sourceDoc = await importer.fetchModel(uri);

  const [sourceNode] = sourceDoc.getRoot().listNodes();
  if (!sourceNode) {
    throw new Error(`Model imported from \`${uri}\` contains no nodes.`);
  }
  clearNodeTransform(sourceNode);

  const targetNode = new NonManifoldGLTFNode(sourceDoc, sourceNode);
  targetNode.name = sourceNode.getName();
  return targetNode;
}

/**
 *
 * @group Modelling Functions
 * @param uri
 * @returns
 */
export async function importManifold(uri: string): Promise<Manifold> {
  const sourceNode = await importModel(uri);
  try {
    return sourceNode.makeManifold();
  } catch (e) {
    const newError = new Error(
        `Model imported from \`${uri}\` contains no manifold geometry.`);
    newError.cause = e;
    throw newError;
  }
}

/**
 * Convert a gltf-transform Node and its children into a Manifold object.
 *
 * The original imported model may consist of an entire tree of nodes, each of
 * which may or may not be manifold.  This method will convert each child node,
 * and then union the results together.  If a child node has no mesh, the mesh
 * has no geometry, or the mesh is not manifold, that child node will be
 * silently excluded.
 *
 * Other errors will be re-thrown for the caller to handle.
 *
 * @returns A valid Manifold object.
 */
export function gltfNodeToManifold(
    document: GLTFTransform.Document, node: GLTFTransform.Node): Manifold {
  const converted: Array<Manifold> = [];
  node.traverse(child => {
    const manifold = gltfMeshToManifold(document, child);
    if (manifold) converted.push(manifold);
  });
  if (!converted.length) {
    throw new Error(`Model contains no manifold geometry.`);
  }

  return getManifoldModuleSync()!.Manifold.union(converted);
};

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
      // 'POSITION' is always present; we don't need to specify it.
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
