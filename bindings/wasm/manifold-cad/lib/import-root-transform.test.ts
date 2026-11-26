import {Document} from '@gltf-transform/core';
import {expect, test} from 'vitest';

import {fromArrayBuffer, register} from './import-model.ts';
import {euler2quat} from './math.ts';

let source: Document;
const mimetype = 'application/x-manifold-root-transform-test';
register({
  importFormats: [{extension: 'root-transform-test', mimetype}],
  fromArrayBuffer: async () => source,
});

test.each([1, 2])(
    'converts world coordinates with %i scene roots', async count => {
      source = new Document();
      const root = source.createNode('transformed')
                       .setTranslation([1, 2, 3])
                       .setRotation(euler2quat([20, 40, 70]))
                       .setScale([2, 3, 4]);
      const child = source.createNode('child').setTranslation([5, 6, 7]);
      root.addChild(child);
      const scene = source.createScene().addChild(root);
      if (count === 2) scene.addChild(source.createNode('other'));
      const before = child.getWorldMatrix();
      const expected = [...before];
      for (let column = 0; column < 4; ++column) {
        expected[column * 4] = before[column * 4] * 1000;
        expected[column * 4 + 1] = -before[column * 4 + 2] * 1000;
        expected[column * 4 + 2] = before[column * 4 + 1] * 1000;
      }
      await fromArrayBuffer(new ArrayBuffer(0), mimetype);
      const actual = child.getWorldMatrix();
      for (let i = 0; i < 16; ++i)
        expect(actual[i]).toBeCloseTo(expected[i], 7);
      expect(child.getTranslation()).toEqual([5, 6, 7]);
    });
