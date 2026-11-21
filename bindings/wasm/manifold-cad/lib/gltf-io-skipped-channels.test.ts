import {Document} from '@gltf-transform/core';
import type {Mesh} from 'manifold-3d';
import {expect, test} from 'vitest';

import type {Attribute} from './gltf-io.ts';
import {writeMesh} from './gltf-io.ts';

test.each([1, 2, 3, 4])('advances by all SKIP_%i property channels', width => {
  const doc = new Document();
  const values = [
    0, 0, 0, ...new Array(width).fill(99), 0, 0, 1,
    1, 0, 0, ...new Array(width).fill(99), 0, 0, 1,
    0, 1, 0, ...new Array(width).fill(99), 0, 0, 1,
  ];
  const mesh = {
    numVert: 3,
    numProp: 6 + width,
    vertProperties: new Float32Array(values),
    triVerts: new Uint32Array([0, 1, 2]),
    runIndex: new Uint32Array([0, 3]),
    runOriginalID: new Uint32Array([7]),
  } as Mesh;
  const attributes: Attribute[] =
      ['POSITION', `SKIP_${width}` as Attribute, 'NORMAL'];
  const output = writeMesh(
      doc, mesh, new Map([[
        7, {
          material: doc.createMaterial(),
          attributes,
        }
      ]]),
      false);
  const primitive = output.listPrimitives()[0];
  expect(primitive.listSemantics()).toEqual(['POSITION', 'NORMAL']);
  expect(Array.from(primitive.getAttribute('NORMAL')!.getArray()!)).toEqual([
    0, 0, 1, 0, 0, 1, 0, 0, 1
  ]);
  expect(Array.from(primitive.getAttribute('POSITION')!.getArray()!)).toEqual([
    0, 0, 0, 1, 0, 0, 0, 1, 0
  ]);
});
