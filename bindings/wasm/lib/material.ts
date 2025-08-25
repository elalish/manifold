
import {Document, Material} from '@gltf-transform/core';
import {KHRMaterialsUnlit} from '@gltf-transform/extensions';

import {Manifold} from '../examples/built/manifold';
import {GLTFMaterial} from '../examples/public/editor';

import {globalDefaults, GLTFNode} from './export'

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