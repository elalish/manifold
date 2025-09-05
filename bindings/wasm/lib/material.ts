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

import {Document, Material} from '@gltf-transform/core';
import {KHRMaterialsUnlit} from '@gltf-transform/extensions';

import {GLTFMaterial} from '../examples/public/editor';
import {Manifold} from '../manifold-encapsulated-types';

import {globalDefaults, GLTFNode} from './scene-builder.ts'

const id2material = new Map<number, GLTFMaterial>();
const materialCache = new Map<GLTFMaterial, Material>();

export function cleanup() {
  id2material.clear();
  materialCache.clear();
}

export const setMaterial =
    (manifold: Manifold, material: GLTFMaterial): Manifold => {
      const out = manifold.asOriginal();
      id2material.set(out.originalID(), material);
      return out;
    };

export const getMaterialByID = (id: number):
    GLTFMaterial|undefined => {
      return id2material.get(id);
    }

export function getBackupMaterial(node?: GLTFNode):
    GLTFMaterial {
      if (node == null) {
        return {};
      }
      if (node.material == null) {
        node.material = getBackupMaterial(node.parent);
      }
      return node.material;
    }

function makeDefaultMaterial(doc: Document, matIn: GLTFMaterial = {}):
    Material {
      const defaults = {...globalDefaults};
      Object.assign(defaults, matIn);
      const {roughness, metallic, baseColorFactor, alpha, unlit} = defaults;

      const material = doc.createMaterial(matIn.name ?? '');

      if (unlit) {
        const unlit = doc.createExtension(KHRMaterialsUnlit).createUnlit();
        material.setExtension('KHR_materials_unlit', unlit);
      }

      if (alpha < 1) {
        material.setAlphaMode(Material.AlphaMode.BLEND).setDoubleSided(true);
      }

      return material.setRoughnessFactor(roughness)
          .setMetallicFactor(metallic)
          .setBaseColorFactor([...baseColorFactor, alpha]);
    }

export function getCachedMaterial(doc: Document, matDef: GLTFMaterial):
    Material {
      if (!materialCache.has(matDef)) {
        materialCache.set(matDef, makeDefaultMaterial(doc, matDef));
      }
      return materialCache.get(matDef)!;
    }