import {Document} from '@gltf-transform/core';
import type {Mesh} from 'manifold-3d';
import {expect, test} from 'vitest';

import {writeMesh} from './gltf-io.ts';

test.each([false, true])(
    'exports default POSITION without material mappings (extension %s)',
    extension => {
      const doc = new Document();
      const positions = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1];
      const mesh = {
        numVert: 4,
        numProp: 3,
        vertProperties: new Float32Array(positions),
        triVerts: new Uint32Array([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3]),
        runIndex: new Uint32Array([0, 6, 12]),
        runOriginalID: new Uint32Array([7, 8]),
      } as Mesh;
      const output = writeMesh(doc, mesh, new Map(), extension);
      expect(output.listPrimitives()).toHaveLength(2);
      for (const primitive of output.listPrimitives()) {
        expect(primitive.getAttribute('POSITION')).not.toBeNull();
        expect(Array.from(primitive.getAttribute('POSITION')!.getArray()!))
            .toEqual(positions);
        expect(primitive.getMaterial()).toBeNull();
      }
    });

test('does not allocate attributes for a mesh without runs', () => {
  const doc = new Document();
  const mesh = {
    numVert: 0,
    numProp: 3,
    vertProperties: new Float32Array(),
    triVerts: new Uint32Array(),
    runIndex: new Uint32Array([0]),
    runOriginalID: new Uint32Array(),
  } as Mesh;
  const output = writeMesh(doc, mesh, new Map(), false);
  expect(output.listPrimitives()).toHaveLength(0);
  expect(doc.getRoot().listAccessors()).toHaveLength(0);
});
