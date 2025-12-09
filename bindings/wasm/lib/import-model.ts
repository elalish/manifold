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
 * The high level functions `importModel()` and `importManifold()` will import
 * models as display-only and full manifold objects respectively.  These
 * functions are available in manifoldCAD.
 *
 * @packageDocumentation
 */

import type * as GLTFTransform from '@gltf-transform/core';

import type {Manifold, Mesh} from '../manifold-encapsulated-types.d.ts';
import type {Vec3} from '../manifold-global-types.d.ts';

import {UnsupportedFormatError} from './error.ts';
import * as gltfIO from './gltf-io.ts';
import {VisualizationGLTFNode} from './gltf-node.ts';
import {setMaterialByID} from './material.ts';
import {findExtension, findMimeType, isNode} from './util.ts';
import {getManifoldModuleSync} from './wasm.ts';

/**
 * @inline
 */
interface Format {
  extension: string;
  mimetype: string;
}

export interface Importer {
  importFormats: Array<Format>;
  fromArrayBuffer:
      (buffer: Uint8Array<ArrayBufferLike>) => Promise<GLTFTransform.Document>;
}

/**
 * @inline
 */
export interface ImportOptions {
  /**
   * Use `mimetype` to determine the format of the imported model, rather than
   * inferring it.
   */
  mimetype?: string;
  /**
   * Tolerance for mesh simplification.  Any edge smaller than tolerance may be
   * collapsed.
   */
  tolerance?: number;
}

const importers: Array<Importer> = [];
register(gltfIO);

const id2mesh = new Map<number, GLTFTransform.Mesh>();
const mesh2node = new Map<GLTFTransform.Mesh, GLTFTransform.Node>();
const mesh2mesh = new Map<Mesh, GLTFTransform.Mesh>();
const node2doc = new Map<GLTFTransform.Node, GLTFTransform.Document>();

/**
 * @internal
 */
export const cleanup = () => {
  id2mesh.clear();
  mesh2node.clear();
  mesh2mesh.clear();
  node2doc.clear();
};

/**
 * @internal
 */
export const getDocumentByID = (runID: number): GLTFTransform.Document|null => {
  const mesh = id2mesh.get(runID);
  if (!mesh) return null;
  const node = mesh2node.get(mesh);
  if (!node) return null;
  return node2doc.get(node) ?? null;
};

function getFormat(identifier: string): Format {
  const formats = importers.flatMap(im => im.importFormats);
  const format = (findMimeType(identifier, formats) ??
                  findExtension(identifier, formats)) as Format;
  if (!format) throw new UnsupportedFormatError(identifier, formats);
  return format;
}

function getImporter(identifier: Format|string) {
  const format =
      typeof identifier === 'string' ? getFormat(identifier) : identifier;
  return importers.find(im => im.importFormats.includes(format))!;
}

/**
 * Returns true if a given extension or mimetype can be imported.
 *
 * @param filetype
 * @param throwOnFailure If true, throw an `UnsupportedFormatException` rather
 *     than return false.
 * @group Management Functions
 */
export function supports(filetype: string, throwOnFailure: false): boolean {
  if (throwOnFailure) return !!getFormat(filetype);

  try {
    return !!getFormat(filetype);
  } catch (e) {
    return false;
  }
}

/**
 * Register an importer.
 *
 * Supported formats will be inferred.
 * @group Management Functions
 */
export function register(importer: Importer) {
  importers.push(importer);
}

/**
 * Import a model, for display only.
 *
 * @group Modelling Functions
 * @returns
 */
export async function importModel(
    source: string|Blob|URL,
    options: ImportOptions = {}): Promise<VisualizationGLTFNode> {
  const sourceDoc = await readModel(source, options);

  const sourceNodes = sourceDoc.getRoot().listNodes();
  if (!sourceNodes.length) {
    throw new Error(`Model imported from \`${source}\` contains no nodes.`);
  }

  // Find top level nodes and correct their scale.
  for (const sourceNode of sourceNodes) {
    if (sourceNode.getParentNode()) continue;
    // glTF has a defined scale of 1:1 metre.
    // manifoldCAD has a defined scale of 1:1 mm.
    const scale = sourceNode.getScale();
    sourceNode.setScale([scale[0] * 1000, scale[1] * 1000, scale[2] * 1000]);
  }

  const targetNode = new VisualizationGLTFNode(sourceDoc);
  if (sourceNodes.length == 1) {
    const [sourceNode] = sourceNodes;
    targetNode.node = sourceNode;
    targetNode.name = sourceNode.getName();
  }
  if (typeof source === 'string') targetNode.uri = source;

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
 */
export async function importManifold(
    source: string|Blob|URL, options: ImportOptions = {}): Promise<Manifold> {
  const {document, node} = await importModel(source, options);
  try {
    return gltfDocToManifold(document, node, options.tolerance);
  } catch (e) {
    const newError = new Error(
        `Model imported from \`${source}\` contains no manifold geometry.`);
    newError.cause = e;
    throw newError;
  }
}

/**
 * Resolve and read a model, be it a file, a URL or a Blob.
 *
 * @group Low Level Functions
 **/
export async function readModel(
    source: string|Blob|URL,
    options: ImportOptions = {}): Promise<GLTFTransform.Document> {
  if (source instanceof Blob) {
    // Binary blob.
    return await fromBlob(source, options);
  }

  let path: string|null = null;
  if (source instanceof URL) {
    path = source.href;
  } else if ('string' === typeof source) {
    path = source;
  }

  if (path) {
    if (path.startsWith('data:') || path.startsWith('blob:')) {
      // Fetch can probably handle this.
      return await fetchModel(path, options);

    } else if (/^https?:\/\//.test(path)) {
      // Absolute URL.
      return await fetchModel(path, options);

    } else if (path.startsWith('file:')) {
      // File URL.
      return readFile(path, options);

    } else {
      // Relative URL.
      if (isNode()) {
        // In node, assume it's relative to the current working directory.
        // That may not be the same as relative to the source file.
        return await readFile(path, options);

      } else {
        // In the browser, it's relative to the current URL.
        return await fetchModel(path, options);
      }
    }
  }

  throw new Error(`Could not import model \`${source}\`.`);
}

/**
 * Fetch a model over HTTP/HTTPS.
 *
 * @group Low Level Functions
 **/
export async function fetchModel(
    uri: string, options: ImportOptions = {}): Promise<GLTFTransform.Document> {
  const importer = getImporter(options.mimetype ?? uri);
  const response = await fetch(uri);
  const blob = await response.blob();
  return await importer.fromArrayBuffer(await blob.bytes());
}

/**
 * Read a model from a Blob.
 *
 * @group Low Level Functions
 **/
export async function fromBlob(
    blob: Blob, options: ImportOptions = {}): Promise<GLTFTransform.Document> {
  const importer = getImporter(options.mimetype ?? blob.type);
  return await importer.fromArrayBuffer(await blob.bytes());
}

/**
 * Read a model from an ArrayBuffer.
 *
 * @group Low Level Functions
 **/
export async function fromArrayBuffer(
    buffer: Uint8Array<ArrayBufferLike>,
    identifier: string): Promise<GLTFTransform.Document> {
  const importer = getImporter(identifier);
  return await importer.fromArrayBuffer(buffer);
}

/**
 * Read a model from disk.
 * @group Low Level Functions
 **/
export async function readFile(filename: string, options: ImportOptions = {}) {
  if (!isNode()) {
    throw new Error('Must have a filesystem to read files.');
  }
  const importer = getImporter(options.mimetype ?? filename);
  const fs = await import('node:fs/promises');
  const {fileURLToPath} = await import('node:url');

  const path =
      filename.startsWith('file:') ? fileURLToPath(filename) : filename;
  const buffer = await fs.readFile(path);
  return await importer.fromArrayBuffer(buffer);
}

/**
 * Convert a gltf-transform Node and its descendants into a Manifold object.
 *
 * The original imported model may consist of an entire tree of nodes, each of
 * which may or may not be manifold.  This method will convert each child node,
 * and then union the results together.  If a child node has no mesh, the mesh
 * has no geometry, or the mesh is not manifold, that child node will be
 * silently excluded.
 *
 * Other errors will be re-thrown for the caller to handle.
 *
 * @group Low Level Functions
 */
export function gltfDocToManifold(
    document: GLTFTransform.Document, node?: GLTFTransform.Node,
    tolerance?: number): Manifold {
  const meshes = gltfNodeToMeshes(document, node);
  if (!meshes.length) {
    throw new Error(`Model contains no meshes!`);
  }
  return meshesToManifold(meshes, tolerance);
};

/**
 * Extract meshes from a gltf-transform node (and its descendants), or from all
 * nodes in a document, and convert them to Mesh objects.  Meshless nodes will
 * be silently skipped.
 *
 * @internal
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
 * @internal
 */
const tryToMakeManifold = (mesh: Mesh): Manifold|null => {
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
 */
function meshesToManifold(meshes: Array<Mesh>, tolerance?: number): Manifold {
  const {Manifold} = getManifoldModuleSync()!;

  const manifolds = [];
  for (const mesh of meshes) {
    let manifold = tryToMakeManifold(mesh);
    if (!manifold) {
      // That didn't work.  Do we need to merge primitives?
      mesh.merge();
      manifold = tryToMakeManifold(mesh);
    }
    if (!manifold && tolerance) {
      // That didn't work either.
      // Can we adjust the model within tolerance?
      mesh.tolerance = tolerance;
      mesh.merge();
      manifold = tryToMakeManifold(mesh);
    }
    if (!manifold) continue;

    // We have a manifold object, but it is in the local coordinate system of
    // the original glTF-transform node.  Find that node, and transform it back
    // if possible.
    const sourceMesh = mesh2mesh.get(mesh);
    const sourceNode = sourceMesh ? mesh2node.get(sourceMesh) : null;
    if (sourceNode) {
      manifolds.push(manifold.transform(sourceNode.getWorldMatrix()));
    } else {
      manifolds.push(manifold);
    }
  }

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
 */
function gltfMeshToMesh(gltfmesh: GLTFTransform.Mesh): Mesh {
  const {Manifold, Mesh} = getManifoldModuleSync()!;

  const {mesh, runProperties} = gltfIO.readMesh(gltfmesh)!;

  // Get a a reserved ID from manifold for each run.
  const numID = runProperties.length;
  const firstID = Manifold.reserveIDs(numID);
  mesh.runOriginalID = new Uint32Array(numID);

  // Iterate through each primitive.
  for (let primitiveID = 0; primitiveID < numID; ++primitiveID) {
    // Set the manifold runID.  This will be parsed by `new Mesh()`.
    const runID = firstID + primitiveID;
    mesh.runOriginalID[primitiveID] = runID;
    const {attributes, material} = runProperties[primitiveID]

    // Save these for later lookup.
    id2mesh.set(runID, gltfmesh)

    // Import what we can as a manifoldCAD material.
    // We'll leave the original material attached for export.
    setMaterialByID(runID, {
      // 'POSITION' is always present; we don't need to specify it.
      attributes: attributes.filter(x => x !== 'POSITION'),
      alpha: material.getAlpha(),
      baseColorFactor: material.getBaseColorFactor() as any as Vec3,
      metallic: material.getMetallicFactor(),
      roughness: material.getRoughnessFactor(),
      name: material.getName(),

      // Make sure we can find this source material later.
      sourceMaterial: material,
      sourceRunID: runID,
    });
  }

  const manifoldMesh = new Mesh(mesh);
  mesh2mesh.set(manifoldMesh, gltfmesh);
  return manifoldMesh;
}