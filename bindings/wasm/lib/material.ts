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

/**
 * Assign glTF materials to manifoldCAD models.
 *
 * @packageDocumentation
 * @group manifoldCAD Runtime
 * @category Modelling
 */

import * as GLTFTransform from '@gltf-transform/core';
import {KHRMaterialsUnlit} from '@gltf-transform/extensions';
import {copyToDocument} from '@gltf-transform/functions';

import type {Manifold} from '../manifold.d.ts';

import {GLTFMaterial, GLTFNode} from './gltf-node.ts';
import {getDocumentByID} from './import-model.ts';

const defaultMaterial = {
  roughness: 0.2,
  metallic: 1,
  baseColorFactor: [1, 1, 0] as [number, number, number],
  alpha: 1,
  unlit: false,
};

const id2material = new Map<number, GLTFMaterial>();
const materialCache = new Map<GLTFMaterial, GLTFTransform.Material>();

export function cleanup() {
  id2material.clear();
  materialCache.clear();
}

/**
 * Return a shallow copy of the input manifold with the given material
 * properties applied. They will be carried along through operations.
 *
 * @group Modelling Functions
 * @param manifold The input object.
 * @param material A set of material properties to apply to this manifold.
 */
export const setMaterial =
    (manifold: Manifold, material: GLTFMaterial): Manifold => {
      const out = manifold.asOriginal();
      id2material.set(out.originalID(), material);
      return out;
    };

/**
 * @internal
 */
export const getMaterialByID = (id: number): GLTFMaterial|undefined => {
  return id2material.get(id);
};

/**
 * @internal
 */
export const setMaterialByID = (id: number, material: GLTFMaterial) => {
  id2material.set(id, material);
};

/**
 * @internal
 */
export function getBackupMaterial(node?: GLTFNode): GLTFMaterial {
  if (node == null) {
    return {};
  }
  if (node.material == null) {
    node.material = getBackupMaterial((node.parent as GLTFNode));
  }
  return node.material;
}

function copyImportedMaterial(
    doc: GLTFTransform.Document,
    matIn: GLTFMaterial = {}): GLTFTransform.Material|null {
  if (!matIn.sourceMaterial || !matIn.sourceRunID) {
    return null;
  }
  // This is an imported material, attached to an existing document.
  // We need to copy it into place.
  const sourceDoc = getDocumentByID(matIn.sourceRunID!)!;
  const sourceMaterial = matIn.sourceMaterial!;
  const map = copyToDocument(doc, sourceDoc, [sourceMaterial]);
  return map.get(sourceMaterial) as GLTFTransform.Material;
}

function makeDefaultMaterial(
    doc: GLTFTransform.Document,
    matIn: GLTFMaterial = {}): GLTFTransform.Material {
  const {roughness, metallic, baseColorFactor, alpha, unlit} = {
    ...defaultMaterial,
    ...matIn
  };
  const material =
      copyImportedMaterial(doc, matIn) ?? doc.createMaterial(matIn.name ?? '');

  if (unlit) {
    const unlit = doc.createExtension(KHRMaterialsUnlit).createUnlit();
    material.setExtension('KHR_materials_unlit', unlit);
  }

  if (alpha < 1) {
    material.setAlphaMode(GLTFTransform.Material.AlphaMode.BLEND)
        .setDoubleSided(true);
  }

  return material.setRoughnessFactor(roughness)
      .setMetallicFactor(metallic)
      .setBaseColorFactor([...baseColorFactor, alpha]);
}

/**
 *
 * @internal
 */
export function getCachedMaterial(
    doc: GLTFTransform.Document, matDef: GLTFMaterial): GLTFTransform.Material {
  if (!materialCache.has(matDef)) {
    materialCache.set(matDef, makeDefaultMaterial(doc, matDef));
  }
  return materialCache.get(matDef)!;
}