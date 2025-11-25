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
import {clearNodeParent} from '@gltf-transform/functions';

import type {Manifold, Mesh} from '../manifold-encapsulated-types.d.ts';
import type {Vec3} from '../manifold-global-types.d.ts';

import {Properties, readMesh} from './gltf-io.ts';
import {VisualizationGLTFNode} from './gltf-node.ts';
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

const id2mesh = new Map<number, GLTFTransform.Mesh>();
const mesh2node = new Map<GLTFTransform.Mesh, GLTFTransform.Node>();
const node2doc = new Map<GLTFTransform.Node, GLTFTransform.Document>();
const doc2uri = new Map<GLTFTransform.Document, string>();
const id2properties = new Map<number, Properties>();

export const cleanup = () => {
  id2mesh.clear();
  mesh2node.clear();
  node2doc.clear();
  id2properties.clear();
};

/**
 * @internal
 */
export const getPropertiesByID = (runID: number) => id2properties.get(runID);

/**
 * @internal
 */
export const getDocumentByID = (runID: number) => {
  const mesh = id2mesh.get(runID);
  if (!mesh) return null;
  const node = mesh2node.get(mesh);
  if (!node) return null;
  return node2doc.get(node);
};

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
 * Import a model, for display only.
 *
 * @group Modelling Functions
 * @param uri
 * @returns
 */
export async function importModel(uri: string): Promise<VisualizationGLTFNode> {
  const importer = importers[0];
  const sourceDoc = await importer.fetchModel(uri);
  doc2uri.set(sourceDoc, uri);

  const [sourceNode] = sourceDoc.getRoot().listNodes();
  if (!sourceNode) {
    throw new Error(`Model imported from \`${uri}\` contains no nodes.`);
  }

  // glTF has a defined scale of 1:1 metre.
  // manifoldCAD has a defined scale of 1:1 mm.
  const scale = sourceNode.getScale();
  sourceNode.setScale([scale[0]*1000, scale[1]*1000, scale[2]*1000]);

  // Apply any transforms from ancester nodes, leaving this node in the overal scene coordinate space.
  clearNodeParent(sourceNode);

  // Wrap it for visualization.
  const targetNode = new VisualizationGLTFNode(sourceDoc, sourceNode);
  targetNode.name = sourceNode.getName();
  targetNode.uri = uri;

  return targetNode;
}

/**
 * Import a model, and convert it to a Manifold object for manipulation.
 *
 * The original imported model may consist of an entire tree of nodes, each of
 * which may or may not be manifold.  This method will convert each child node,
 * and then union the results together.  If a child node has no mesh, the mesh
 * has no geometry, or the mesh is not manifold, that child node will be
 * silently excluded.
 *
 * @group Modelling Functions
 * @param uri
 * @returns
 */
export async function importManifold(uri: string): Promise<Manifold> {
  const sourceNode = await importModel(uri);
  try {
    return gltfNodeToManifold(sourceNode.document, sourceNode.node);
  } catch (e) {
    const newError = new Error(
        `Model imported from \`${uri}\` contains no manifold geometry.`);
    newError.cause = e;
    throw newError;
  }
}

/**
 * Convert a gltf-transform Node and its descendant into a Manifold object.
 *
 * The original imported model may consist of an entire tree of nodes, each of
 * which may or may not be manifold.  This method will convert each child node,
 * and then union the results together.  If a child node has no mesh, the mesh
 * has no geometry, or the mesh is not manifold, that child node will be
 * silently excluded.
 *
 * Other errors will be re-thrown for the caller to handle.
 *
 * @internal
 * @returns A valid Manifold object.
 */
function gltfNodeToManifold(
    document: GLTFTransform.Document, node?: GLTFTransform.Node): Manifold {
  const meshes = gltfNodeToMeshes(document, node);
  if (!meshes.length) {
    throw new Error(`Model contains no meshes!`);
  }
  return meshesToManifold(meshes);
};

/**
 * Extract meshes from a gltf-transform node (and its descendants), or from all
 * nodes in a document, and convert them to Mesh objects.  Meshless nodes will
 * be silently skipped.
 *
 * @internal
 * @param document
 * @param node Optionally, traverse the descendants of this node.
 * @returns
 */
function gltfNodeToMeshes(
    document: GLTFTransform.Document, node?: GLTFTransform.Node): Array<Mesh> {
  const getDescendants = (root: GLTFTransform.Node) => {
    const descendants: Array<GLTFTransform.Node> = [];
    root.traverse(descendant => descendants.push(descendant));
    return descendants;
  };
  const descendants: Array<GLTFTransform.Node> =
      node ? getDescendants(node) : document.getRoot().listNodes();

  return descendants
      .map(descendant => {
        const gltfmesh = descendant.getMesh();
        if (!gltfmesh) return null;

        node2doc.set(descendant, document);
        mesh2node.set(gltfmesh, descendant);

        return gltfMeshToMesh(gltfmesh);
      })
      .filter(mesh => !!mesh);
}

/**
 * Convert a Mesh into a Manifold.  Returns null if the result is not manifold
 * or is empty.  All other exceptions will be re-thrown.
 *
 * @internal
 * @param mesh
 */
const tryToMakeManifold = (mesh: Mesh) => {
  const {Manifold} = getManifoldModuleSync()!;
  try {
    const manifold = new Manifold(mesh);
    if (manifold && !manifold.isEmpty()) {
      return manifold;
    }
  } catch (e) {
    if ((e as any)?.name === 'ManifoldError' ||
        (e as any)?.code === 'NotManifold') {
    } else {
      throw e;
    }
  }
  return null;
};

/**
 * Given a list of Mesh objects, attempt to convert them individually into
 * Manifold objects, and return the union.  Non-manifold Meshes will be silently
 * skipped.
 *
 * @internal
 * @param meshes
 * @returns
 */
function meshesToManifold(meshes: Array<Mesh>): Manifold {
  const {Manifold} = getManifoldModuleSync()!;

  const manifolds = meshes.map(mesh => tryToMakeManifold(mesh))
                        .filter(manifold => !!manifold)
                        .filter(manifold => !manifold.isEmpty());

  if (!manifolds?.length) {
    throw new Error(`Model contains no manifold geometry.`);
  }

  return Manifold.union(manifolds);
}

/**
 * Convert a single gltf-transform Mesh to a Mesh object.
 *
 * Each primitive in a gltf-transform mesh may have its own material and
 * attributes.  Those primitives become runs once translated into Manifold.
 * Each run may have a different material attached.  ManifoldCAD can manage this
 * case, although there is not a user-facing way to quickly assign materials to
 * parts of a model.  This function will index the original materials
 * (properties) to be copied into an exported GLTF document.  It will also set
 * ManifoldCAD materials (a subset of GLTF materials) as a fallback.
 *
 * @internal
 * @param gltfmesh The gltf-transform mesh.
 * @returns A Mesh object if possible, `null` if not.
 */
function gltfMeshToMesh(gltfmesh: GLTFTransform.Mesh): Mesh {
  const {Manifold, Mesh} = getManifoldModuleSync()!;

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
    id2mesh.set(runID, gltfmesh)
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

  return new Mesh(mesh);
}