import {Accessor, Document} from '@gltf-transform/core';
import {expect, test} from 'vitest';

import {readMesh} from './gltf-io.ts';

test.each([
  [Uint8Array, 257],
  [Uint16Array, 65537],
  [Uint32Array, 257],
] as const)('preserves widened offsets for %s indices', (IndexArray, count) => {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const mesh = doc.createMesh();
  for (const size of [count, 3]) {
    const positions = doc.createAccessor()
                          .setBuffer(buffer)
                          .setType(Accessor.Type.VEC3)
                          .setArray(new Float32Array(size * 3));
    const indices = doc.createAccessor()
                        .setBuffer(buffer)
                        .setType(Accessor.Type.SCALAR)
                        .setArray(new IndexArray([0, 1, 2]));
    mesh.addPrimitive(doc.createPrimitive()
                          .setAttribute('POSITION', positions)
                          .setIndices(indices));
  }
  const result = readMesh(mesh)!;
  expect(Array.from(result.mesh.triVerts)).toEqual([
    0, 1, 2, count, count + 1, count + 2
  ]);
  expect(Array.from(result.mesh.runIndex!)).toEqual([0, 3, 6]);
  expect(result.mesh.vertProperties.length).toBe((count + 3) * 3);
  expect(Array.from(mesh.listPrimitives()[1].getIndices()!.getArray()!))
      .toEqual([0, 1, 2]);
});
