import {Accessor, Document} from '@gltf-transform/core';
import {expect, test} from 'vitest';

import {readMesh} from './gltf-io.ts';
import {EXTManifold} from './manifold-gltf.ts';

const cases = [
  {
    attribute: 'COLOR_0',
    array: new Uint8Array([0, 128, 255]),
    normalized: true,
    expected: [0, 128 / 255, 1]
  },
  {
    attribute: 'NORMAL',
    array: new Int16Array([-32768, 0, 32767]),
    normalized: true,
    expected: [-1, 0, 1]
  },
  {
    attribute: 'COLOR_0',
    array: new Uint16Array([0, 32768, 65535]),
    normalized: true,
    expected: [0, 32768 / 65535, 1]
  },
  {
    attribute: 'NORMAL',
    array: new Float32Array([0, 0.5, 1]),
    normalized: false,
    expected: [0, 0.5, 1]
  },
];

for (const extension of [false, true]) {
  test.each(cases)(
      'decodes $array/$normalized, extension ' + extension,
      ({attribute, array, normalized, expected}) => {
        const doc = new Document();
        const buffer = doc.createBuffer();
        const positions =
            doc.createAccessor()
                .setBuffer(buffer)
                .setType(Accessor.Type.VEC3)
                .setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
        const normals = doc.createAccessor()
                            .setBuffer(buffer)
                            .setType(Accessor.Type.VEC3)
                            .setArray(new (array.constructor as any)(
                                [...array, ...array, ...array]))
                            .setNormalized(normalized);
        const indices = doc.createAccessor()
                            .setBuffer(buffer)
                            .setType(Accessor.Type.SCALAR)
                            .setArray(new Uint32Array([0, 1, 2]));
        const mesh = doc.createMesh().addPrimitive(
            doc.createPrimitive()
                .setAttribute('POSITION', positions)
                .setAttribute(attribute, normals)
                .setIndices(indices));
        if (extension) {
          mesh.setExtension(
              'EXT_mesh_manifold',
              doc.createExtension(EXTManifold)
                  .createManifoldPrimitive()
                  .setIndices(indices)
                  .setRunIndex([0, 3]));
        }
        const result = readMesh(mesh)!;
        for (let vertex = 0; vertex < 3; ++vertex) {
          for (let axis = 0; axis < 3; ++axis) {
            expect(result.mesh.vertProperties[vertex * 6 + 3 + axis])
                .toBeCloseTo(expected[axis], 6);
          }
        }
        expect(Array.from(result.mesh.triVerts)).toEqual([0, 1, 2]);
        expect(Array.from(normals.getArray()!.slice(0, 3)))
            .toEqual(Array.from(array));
      });
}
