import {Accessor, Document, Primitive} from '@gltf-transform/core';
import {expect, test} from 'vitest';

import {readMesh} from './gltf-io.ts';

test.each([
  [Primitive.Mode.TRIANGLES, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
] as const)(
    'expands mode %i with and without indices', (mode, order, expected) => {
      for (const indexed of [false, true]) {
        const doc = new Document(), buffer = doc.createBuffer();
        const positions = doc.createAccessor()
                              .setBuffer(buffer)
                              .setType(Accessor.Type.VEC3)
                              .setArray(new Float32Array(order.length * 3));
        const primitive = doc.createPrimitive().setMode(mode).setAttribute(
            'POSITION', positions);
        if (indexed) {
          primitive.setIndices(doc.createAccessor()
                                   .setBuffer(buffer)
                                   .setType(Accessor.Type.SCALAR)
                                   .setArray(new Uint16Array(order)));
        }
        const mesh = doc.createMesh().addPrimitive(primitive);
        const result = readMesh(mesh)!;
        expect(Array.from(result.mesh.triVerts)).toEqual(expected);
        expect(Array.from(result.mesh.runIndex!)).toEqual([0, expected.length]);
        expect(result.mesh.vertProperties.length).toBe(order.length * 3);
        expect(primitive.getMode()).toBe(mode);
        expect(primitive.getIndices() !== null).toBe(indexed);
      }
    });

test.each([
  [Primitive.Mode.TRIANGLE_STRIP, false],
  [Primitive.Mode.TRIANGLE_STRIP, true],
  [Primitive.Mode.TRIANGLE_FAN, false],
  [Primitive.Mode.TRIANGLE_FAN, true],
] as const)('rejects unsupported mode %i (indexed: %s)', (mode, indexed) => {
  const doc = new Document(), buffer = doc.createBuffer();
  const primitive = doc.createPrimitive().setMode(mode).setAttribute(
      'POSITION',
      doc.createAccessor()
          .setBuffer(buffer)
          .setType(Accessor.Type.VEC3)
          .setArray(new Float32Array(18)));
  if (indexed) {
    primitive.setIndices(doc.createAccessor()
                             .setBuffer(buffer)
                             .setType(Accessor.Type.SCALAR)
                             .setArray(new Uint16Array([0, 1, 2, 3, 4, 5])));
  }
  expect(() => readMesh(doc.createMesh().addPrimitive(primitive)))
      .toThrow('Unsupported glTF primitive mode');
});

test.each([
  Primitive.Mode.POINTS, Primitive.Mode.LINES, Primitive.Mode.LINE_LOOP,
  Primitive.Mode.LINE_STRIP
])('does not interpret mode %i as triangles', mode => {
  const doc = new Document(), buffer = doc.createBuffer();
  const primitive =
      doc.createPrimitive()
          .setMode(mode)
          .setAttribute(
              'POSITION',
              doc.createAccessor()
                  .setBuffer(buffer)
                  .setType(Accessor.Type.VEC3)
                  .setArray(new Float32Array(18)))
          .setIndices(doc.createAccessor()
                          .setBuffer(buffer)
                          .setType(Accessor.Type.SCALAR)
                          .setArray(new Uint16Array([0, 1, 2, 3, 4, 5])));
  const result = readMesh(doc.createMesh().addPrimitive(primitive))!;
  expect(result.mesh.triVerts.length).toBe(0);
  expect(result.runProperties).toHaveLength(0);
});
