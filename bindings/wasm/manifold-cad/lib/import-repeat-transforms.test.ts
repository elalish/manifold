import {Accessor, Document} from '@gltf-transform/core';
import {afterEach, beforeAll, expect, test} from 'vitest';

import {cleanup, gltfDocToManifold} from './import-model.ts';
import {getManifoldModule, getManifoldModuleSync} from './wasm.ts';

beforeAll(async () => {
  await getManifoldModule();
});
afterEach(cleanup);

test(
    'uses the currently requested instance after earlier imports of a shared mesh',
    () => {
      const {Manifold} = getManifoldModuleSync()!;
      const cube = Manifold.cube();
      const source = cube.getMesh();
      cube.delete();
      const doc = new Document(), buffer = doc.createBuffer();
      const primitive = doc.createPrimitive()
                            .setMaterial(doc.createMaterial())
                            .setAttribute(
                                'POSITION',
                                doc.createAccessor()
                                    .setBuffer(buffer)
                                    .setType(Accessor.Type.VEC3)
                                    .setArray(source.vertProperties))
                            .setIndices(doc.createAccessor()
                                            .setBuffer(buffer)
                                            .setType(Accessor.Type.SCALAR)
                                            .setArray(source.triVerts));
      const mesh = doc.createMesh().addPrimitive(primitive);
      const first = doc.createNode().setMesh(mesh).setTranslation([2, 0, 0]);
      const second = doc.createNode().setMesh(mesh).setTranslation([10, 0, 0]);
      doc.createScene().addChild(first).addChild(second);
      for (const node of [first, second, first, second]) {
        const result = gltfDocToManifold(doc, node);
        try {
          expect(result.boundingBox().min[0])
              .toBeCloseTo(node.getTranslation()[0]);
          expect(result.boundingBox().max[0])
              .toBeCloseTo(node.getTranslation()[0] + 1);
          expect(result.volume()).toBeCloseTo(1);
        } finally {
          result.delete();
        }
      }
      const combined = gltfDocToManifold(doc);
      try {
        expect(combined.volume()).toBeCloseTo(2);
        expect(combined.boundingBox().min[0]).toBeCloseTo(2);
        expect(combined.boundingBox().max[0]).toBeCloseTo(11);
      } finally {
        combined.delete();
      }
    });
