import {afterEach, beforeAll, describe, expect, suite, test} from 'vitest';

import type {Vec3} from '../manifold.d.ts';

import {toArrayBuffer} from './export-model.ts';
import * as importer from './import-model.ts';
import * as wasm from './wasm.ts';
import * as worker from './worker.ts';

beforeAll(async () => await wasm.getManifoldModule());
afterEach(async () => worker.cleanup());

// Some quick vector math to check our results.
const vdiff = (a: Vec3, b: Vec3): number =>
    Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
const vCloseTo = (a: Vec3, b: Vec3, margin = 1): boolean =>
    vdiff(a, b) <= margin;
const vContains = (haystack: Array<Vec3>, needle: Vec3): boolean =>
    !!haystack.find(x => vCloseTo(needle, x));

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
    const mesh = model.getMesh();

    const pts: Array<Vec3> = [];
    for (let i = 0; i < mesh.numVert; i++) {
      pts.push(mesh.position(i) as unknown as Vec3);
    }
    return pts;
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

          // Do we have the same number of vertices?
          expect(manifoldPts.length).toEqual(gltfPts.length);
          // Are they the same vertices?
          const unmatched = gltfPts.filter(pt => !vContains(manifoldPts, pt));
          expect(unmatched).toHaveLength(0);
        });
      });
    });
  });
});