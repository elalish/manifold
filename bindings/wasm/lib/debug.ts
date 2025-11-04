// Copyright 2022-2025 The Manifold Authors.
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

import {Document} from '@gltf-transform/core';

import {Manifold, Mesh} from '../manifold-encapsulated-types';

import {Properties, writeMesh} from './gltf-io.ts';
import {GLTFMaterial} from './gltf-node.ts';
import {getCachedMaterial, getMaterialByID as getOriginalMaterialByID} from './material.ts';

// Debug setup to show source meshes
let ghost = false;
const shown = new Map<number, Mesh>();
const singles = new Map<number, Mesh>();

const SHOW = {
  baseColorFactor: [1, 0, 0],
  alpha: 0.25,
  roughness: 1,
  metallic: 0
} as GLTFMaterial;

const GHOST = {
  baseColorFactor: [0.5, 0.5, 0.5],
  alpha: 0.25,
  roughness: 1,
  metallic: 0
} as GLTFMaterial;

/**
 * @internal
 */
export function cleanup() {
  ghost = false;
  shown.clear();
  singles.clear();
}

const getDebugMeshByID = (id: number):
    Mesh|undefined => {
      return shown.has(id) ? shown.get(id) : singles.get(id);
    }

/**
 * Material for a debug visualization.
 *
 * There are two conditions:
 *
 *   * This is a mesh that has been flagged with `show`.
 *     It will be highlighted with the SHOW material.
 *   * This is a mesh that has been flagged with `only`.
 *     Any other mesh will have the GHOST material, while
 *     this one gets it's natural material.
 *
 * @internal
 * @param id The `originalID` of the mesh.
 */
const getDebugMaterialByID = (id: number):
    GLTFMaterial|undefined => {
      const show = shown.has(id);
      const inMesh = show ? shown.get(id) : singles.get(id);
      if (show && inMesh) {
        return SHOW;
      }

      return getOriginalMaterialByID(id);
    }

/**
 * Override materials when debugging.
 *
 * When a mesh is flagged with `only`, we set the `ghost` global.
 * Everything gets rendered in the GHOST material, while the flagged
 * mesh is added as a debug node.
 *
 * @internal
 * @param id The `originalID` of the mesh.
 */
export const getMaterialByID = (id: number): GLTFMaterial|undefined =>
    ghost ? GHOST : getOriginalMaterialByID(id);

const debug = (manifold: Manifold, map: Map<number, Mesh>) => {
  let result = manifold.asOriginal();
  map.set(result.originalID(), result.getMesh());
  return result;
};

/**
 * Wrap any shape object with this method to display it and any copies in
 * transparent red. This is particularly useful for debugging `subtract()` as it
 * will allow you find the object even if it doesn't currently intersect the
 * result.
 *
 * @group Modelling Functions
 * @param manifold The object to show - returned for chaining.
 */
export const show = (manifold: Manifold) => {
  return debug(manifold, shown);
};

/**
 * Wrap any shape object with this method to display it and any copies as the
 * result, while ghosting out the final result in transparent gray. Helpful for
 * debugging as it allows you to see objects that may be hidden in the interior
 * of the result. Multiple objects marked `only()` will all be shown.
 *
 * @group Modelling Functions
 * @param manifold The object to show - returned for chaining.
 */
export const only = (manifold: Manifold) => {
  ghost = true;
  return debug(manifold, singles);
};

/**
 *
 * @internal
 */
export const getDebugGLTFMesh =
    (doc: Document, manifoldMesh: Mesh, id2properties: Map<number, Properties>,
     backupMaterial: GLTFMaterial = {}) => {
      const debugNodes = [];

      for (const [run, id] of manifoldMesh.runOriginalID!.entries()) {
        const debugMesh = getDebugMeshByID(id);
        if (!debugMesh) {
          continue;
        }

        // Here, we'll get back either a debug material (like SHOW),
        // or the original mesh material.
        const material = getDebugMaterialByID(id) || backupMaterial;
        id2properties.get(id)!.material = getCachedMaterial(doc, material);

        const debugNode = doc.createNode('debug')
                              .setMesh(writeMesh(doc, debugMesh, id2properties))
                              .setMatrix(manifoldMesh.transform(run));
        debugNodes.push(debugNode);
      }
      return debugNodes;
    };
