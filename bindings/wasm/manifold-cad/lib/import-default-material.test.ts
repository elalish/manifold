import {Accessor, Document} from '@gltf-transform/core';
import {afterEach, beforeAll, expect, test} from 'vitest';

import {cleanup, gltfDocToManifold} from './import-model.ts';
import {EXTManifold} from './manifold-gltf.ts';
import {cleanup as cleanupMaterials, getMaterialByID} from './material.ts';
import {getManifoldModule, getManifoldModuleSync} from './wasm.ts';

beforeAll(async () => {
  await getManifoldModule();
});
afterEach(() => {
  cleanup();
  cleanupMaterials();
});

test.each([false, true])(
    'imports the glTF default material (extension %s)', extension => {
      const doc = new Document();
      const buffer = doc.createBuffer();
      const {Manifold} = getManifoldModuleSync()!;
      const cube = Manifold.cube();
      const source = cube.getMesh();
      cube.delete();
      const positions = doc.createAccessor()
                            .setBuffer(buffer)
                            .setType(Accessor.Type.VEC3)
                            .setArray(source.vertProperties);
      const indices = doc.createAccessor()
                          .setBuffer(buffer)
                          .setType(Accessor.Type.SCALAR)
                          .setArray(source.triVerts);
      const primitive = doc.createPrimitive()
                            .setAttribute('POSITION', positions)
                            .setIndices(indices);
      const mesh = doc.createMesh().addPrimitive(primitive);
      if (extension) {
        mesh.setExtension(
            'EXT_mesh_manifold',
            doc.createExtension(EXTManifold)
                .createManifoldPrimitive()
                .setIndices(indices)
                .setRunIndex([0, source.triVerts.length]));
      }
      doc.createScene().addChild(doc.createNode().setMesh(mesh));
      const result = gltfDocToManifold(doc);
      try {
        expect(result.volume()).toBeCloseTo(1);
        for (const id of result.getMesh().runOriginalID!) {
          const material = getMaterialByID(id)!;
          expect(material.alpha).toBe(1);
          expect(material.baseColorFactor).toEqual([1, 1, 1]);
          expect(material.metallic).toBe(1);
          expect(material.roughness).toBe(1);
          expect(material.sourceMaterial).toBeUndefined();
        }
        expect(primitive.getMaterial()).toBeNull();
        expect(doc.getRoot().listMaterials()).toHaveLength(0);
      } finally {
        result.delete();
      }
    });
