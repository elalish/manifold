// Copyright 2023-2025 The Manifold Authors.
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

import * as GLTFTransform from '@gltf-transform/core';
import {fileForContentTypes, FileForRelThumbnail, to3dmodel} from '@jscadui/3mf-export';
import {strToU8, Zippable, zipSync} from 'fflate';

import type {Mat4} from '../manifold-global-types.d.ts';

import {ManifoldPrimitive} from './manifold-gltf';

const supportedFormat = {
  extension: '3mf',
  mimetype: 'model/3mf'
};

export const supportedFormats = [supportedFormat];

interface Mesh3MF {
  id: string;
  vertices: Float32Array;
  indices: Uint32Array;
  name?: string;
}

interface Child3MF {
  objectID: string;
  transform?: Mat4|Array<string>;
}

interface Component3MF {
  id: string;
  children: Array<Child3MF>;
  name?: string;
  transform?: Mat4|Array<string>;
}

interface Header {
  unit?: 'micron'|'millimeter'|'centimeter'|'inch'|'foot'|'meter';
  title?: string;
  author?: string;
  description?: string;
  application?: string;
  creationDate?: string;
  license?: string;
  modificationDate?: string;
}

interface To3MF {
  meshes: Array<Mesh3MF>;
  components: Array<Component3MF>;
  items: Array<Child3MF>;
  precision: number;
  header: Header;
}

const defaultHeader: Header = {
  unit: 'millimeter',
  title: 'ManifoldCAD.org model',
  description: 'ManifoldCAD.org model',
  application: 'ManifoldCAD.org',
}

/**
 * Convert a GLTF-Transform document to a 3MF model.
 *
 * @param doc The GLTF document to convert.
 * @returns A blob containing the converted model.
 */
export async function asBlob(doc: GLTFTransform.Document, header: Header = {}) {
  const to3mf = {
    meshes: [],
    components: [],
    items: [],
    precision: 7,
    header: {...defaultHeader, ...header}
  } as To3MF;

  // GLTF references by array index.
  // 3MF references by ID.
  let nextGlobalID = 1;
  const object2globalID =
      new Map<GLTFTransform.Node|GLTFTransform.Mesh, string>();
  const getObjectID = (obj: GLTFTransform.Node|GLTFTransform.Mesh) =>
      `${object2globalID.get(obj)}`;
  const getMeshID = (mesh: GLTFTransform.Mesh) => {
    // If a mesh has been cloned with a different material, find
    // the original mesh.  This isn't a general GLTF feature; this is set
    // by the ManifoldCAD GLTF exporter.
    const {clonedFrom} = mesh.getExtras();
    if (clonedFrom) {
      return object2globalID.get(clonedFrom as GLTFTransform.Mesh)
    }
    return object2globalID.get(mesh);
  };
  const setObjectID =
      (obj: GLTFTransform.Node|GLTFTransform.Mesh) => {
        const objectID = `${nextGlobalID++}`;
        object2globalID.set(obj, objectID);
        return objectID
      }

  // Get meshes in place first.
  for (const mesh of doc.getRoot().listMeshes()) {
    const manifoldPrimitive =
        mesh.getExtension('EXT_mesh_manifold') as ManifoldPrimitive;
    if (manifoldPrimitive) {
      // This mesh has a list of triangle vertices already.
      const indices = manifoldPrimitive.getIndices();
      const positionAccessor =
          mesh.listPrimitives()[0].getAttribute('POSITION')!;

      const objectID = setObjectID(mesh)
      to3mf.meshes.push({
        vertices: positionAccessor.getArray()! as Float32Array,
        indices: indices.getArray()! as Uint32Array,
        id: objectID
      });
    }

    const {clonedFrom} = mesh.getExtras();
    if (!manifoldPrimitive && clonedFrom) {
      // GLTF Mesh, instance of another mesh.
      // getMeshID will find this when adding it to components.
      continue;
    }
    if (!manifoldPrimitive && !clonedFrom) {
      // GLTF Mesh, no manifold primitive,
      // not an instance of another mesh.
      // We should handle this case, but for now we do not.
      console.log('skipping non-ManifoldCAD mesh')
    }
  }

  // Some 3MF parsers (like PrusaSlicer) expect child nodes
  // to be defined before their parents.
  const nodes = doc.getRoot().listNodes().reverse()
  for (const node of nodes) {
    const meshID = node.getMesh() && getMeshID(node.getMesh()!);
    to3mf.components.push({
      id: setObjectID(node),
      name: node.getName(),
      children: meshID ? [{objectID: meshID}] : [],
      transform: node.getMatrix().map(n => n.toFixed(to3mf.precision))
    });
  }

  // Now we can work out our node hierarchy.
  for (const node of doc.getRoot().listNodes()) {
    const objectID = getObjectID(node)
    if (!objectID) {
      console.log(`Could not find object ID for ${node.getName()}`)
      continue;
    }
    const child = {
      objectID,
      // Most 3MF parsers will not accept a number in scientific notation.
      // Transforms are serialized to a string, containing 12 numbers
      // separated by spaces.  If we force a number to a string here,
      // 3mf-export passes it through.
      transform: node.getMatrix().map(n => n.toFixed(to3mf?.precision ?? 7))
    };
    const parent = node.getParentNode();

    if (parent) {
      // This is a child node, add it to its parent.
      const parentID = getObjectID(parent);
      const parent3mf = to3mf.components.find((comp) => comp.id == parentID)!;
      parent3mf.children.push(child);
    } else {
      // This is a root node.
      // Add it to the build list.
      to3mf.items.push({objectID})
    }
  }

  const fileForRelThumbnail = new FileForRelThumbnail();
  fileForRelThumbnail.add3dModel('3D/3dmodel.model')

  const model = to3dmodel(to3mf as any);
  const files: Zippable = {};
  files['3D/3dmodel.model'] = strToU8(model);
  files[fileForContentTypes.name] = strToU8(fileForContentTypes.content);
  files[fileForRelThumbnail.name] = strToU8(fileForRelThumbnail.content);
  const zipFile = zipSync(files);
  return new Blob(
      [zipFile as Uint8Array<ArrayBuffer>], {type: supportedFormat.mimetype});
}
