import {afterEach, beforeAll, describe, expect, suite, test} from 'vitest';

import type {Vec3} from '../manifold.d.ts';
import {equalsVec3Array, meshToVec3Array} from '../test/util.ts';

import {toArrayBuffer} from './export-model.ts';
import * as importer from './import-model.ts';
import * as wasm from './wasm.ts';
import * as worker from './worker.ts';

beforeAll(async () => await wasm.getManifoldModule());
afterEach(async () => worker.cleanup());

suite('Ensure manifold and glTF rotations match', () => {
  // Here's an asymmetric shape, and scripts that rotate it in 0-3 axes.
  const pts: Array<Vec3> =
      [[-1, 0, 0], [4, 0, 0], [0, -2, 0], [0, 3, 0], [0, 0, 5]];
  const getManifoldScript = (rotX: number, rotY: number, rotZ: number) =>
      `import {Manifold} from \'manifold-3d/manifoldCAD\';\n` +
      `export default Manifold\n` +
      `  .hull(${JSON.stringify(pts)})\n` +
      `  .rotate(${rotX}, ${rotY}, ${rotZ});\n`;
  const getGltfScript = (rotX: number, rotY: number, rotZ: number) =>
      `import {Manifold, GLTFNode} from \'manifold-3d/manifoldCAD\';\n` +
      `const node = new GLTFNode();` +
      `node.manifold = Manifold\n` +
      `  .hull(${JSON.stringify(pts)});\n` +
      `node.rotation = [${rotX}, ${rotY}, ${rotZ}];\n` +
      `export default node;`;

  const getPoints = async (script: string) => {
    // Evaluate, export it and re-import the model.
    // This ensures that glTF transformations are applied.
    const options = {mimetype: 'model/gltf-binary'};
    const doc = await worker.evaluate(script);
    const buffer = await toArrayBuffer(doc, options);
    const model = await importer.importManifold(buffer, options);
    return meshToVec3Array(model.getMesh());
  };

  // Essentially, just waggle the model around a little bit and make sure that
  // we get the same result from each script.
  const angles = [-30, 0, 30];
  describe.for(angles)('Rotate X by %i', (rotX) => {
    describe.for(angles)('Rotate Y by %i', (rotY) => {
      describe.for(angles)('Rotate Z by %i', (rotZ) => {
        test('Manifold and glTF rotations match', async () => {
          const manifoldScript = getManifoldScript(rotX, rotY, rotZ);
          const gltfScript = getGltfScript(rotX, rotY, rotZ);
          const manifoldPts = await getPoints(manifoldScript);
          const gltfPts = await getPoints(gltfScript);

          expect(equalsVec3Array(manifoldPts, gltfPts)).toBeTruthy();
        });
      });
    });
  });
});