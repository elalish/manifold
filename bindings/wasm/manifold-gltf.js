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

import {Extension, ExtensionProperty, PropertyType} from '@gltf-transform/core';

const NAME = 'EXT_manifold';

export class ManifoldPrimitive extends ExtensionProperty {
  static EXTENSION_NAME = NAME;

  init() {
    this.EXTENSION_NAME = NAME;
    this.propertyType = 'ManifoldPrimitive';
    this.parentTypes = [PropertyType.MESH];

    this.runIndex = [];
  }

  getDefaults() {
    return Object.assign(
        super.getDefaults(),
        {manifoldPrimitive: null, mergeIndices: null, mergeValues: null});
  }

  getPrimitive() {
    return this.getRef('manifoldPrimitive');
  }

  setPrimitive(primitive, runIndex) {
    const mesh = this.listParents()[0];
    if (mesh.listPrimitives().length !== runIndex.length - 1)
      throw new Error(
          'You must attach all material primitives to the mesh before setting the manifold primitive.');
    this.runIndex = runIndex;
    mesh.addPrimitive(primitive);
    return this.setRef('manifoldPrimitive', primitive);
  }

  getMergeIndices() {
    return this.getRef('mergeIndices');
  }

  getMergeValues() {
    return this.getRef('mergeValues');
  }

  setMerge(indicesAccessor, valuesAccessor) {
    if (indicesAccessor.getCount() !== valuesAccessor.getCount())
      throw new Error('merge vectors must be the same length.');
    this.setRef('mergeIndices', indicesAccessor);
    return this.setRef('mergeValues', valuesAccessor);
  }
}

export class EXTManifold extends Extension {
  extensionName = NAME;
  static EXTENSION_NAME = NAME;

  createManifoldPrimitive() {
    return new ManifoldPrimitive(this.document.getGraph());
  }

  read(context) {
    const {json} = context.jsonDoc;
    const meshDefs = json.meshes || [];
    meshDefs.forEach((meshDef, meshIndex) => {
      if (!meshDef.extensions || !meshDef.extensions[NAME]) return;

      const mesh = context.meshes[meshIndex];
      const manifoldPrimitive = this.createManifoldPrimitive();
      mesh.setExtension(NAME, manifoldPrimitive);

      const manifoldDef = meshDef.extensions[NAME];

      if (manifoldDef.manifoldPrimitive) {
        for (const primitive of mesh.listPrimitives()) {
          const indices = context.accessors[primitive.indices];
          manifoldPrimitive.runIndex.push(indices.byteOffset / 4);
        }
      }

      if (manifoldDef.mergeIndices && manifoldDef.mergeValues) {
        manifoldPrimitive.setMerge(
            context.accessors[manifoldDef.mergeIndices],
            context.accessors[manifoldDef.mergeValues]);
      }
    });

    return this;
  }

  write(context) {
    const {json} = context.jsonDoc;

    this.document.getRoot().listMeshes().forEach((mesh) => {
      const manifoldPrimitive = mesh.getExtension(NAME);
      if (!manifoldPrimitive) return;

      const meshIndex = context.meshIndexMap.get(mesh);
      const meshDef = json.meshes[meshIndex];

      const mergeIndicesIndex =
          context.accessorIndexMap.get(manifoldPrimitive.getMergeIndices());
      const mergeValuesIndex =
          context.accessorIndexMap.get(manifoldPrimitive.getMergeValues());
      const mergeIndices = json.accessors[mergeIndicesIndex];
      const mergeValues = json.accessors[mergeValuesIndex];

      const primitive = meshDef.primitives.pop();
      const indices = json.accessors[primitive.indices];
      indices.sparse = {
        count: mergeIndices.count,
        indices: {
          bufferView: mergeIndices.bufferView,
          byteOffset: mergeIndices.byteOffset,
          componentType: mergeIndices.componentType
        },
        values: {
          bufferView: mergeValues.bufferView,
          byteOffset: mergeValues.byteOffset,
        }
      };

      const {runIndex} = manifoldPrimitive;
      const numPrimitive = runIndex.length - 1;
      for (let i = 0; i < numPrimitive; ++i) {
        meshDef.primitives[i].indices = json.accessors.length;
        json.accessors.push({
          type: 'SCALAR',
          componentType: indices.componentType,
          count: runIndex[i + 1] - runIndex[i],
          bufferView: indices.bufferView,
          byteOffset: 4 * runIndex[i]
        });
      }

      meshDef.extensions = meshDef.extensions || {};
      meshDef.extensions[NAME] = {
        manifoldPrimitive: primitive,
        mergeIndices: mergeIndicesIndex,
        mergeValues: mergeValuesIndex
      };

      // Test the manifold primitive by replacing the material primitives
      // meshDef.primitives = [primitive];
    });

    return this;
  }
}