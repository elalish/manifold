// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {afterEach, beforeAll, describe, expect, suite, test} from 'vitest';

import type {Vec3} from '../manifold.d.ts';
import {equalsVec3Array, meshToVec3Array} from '../test/util.ts';

import {toArrayBuffer} from './export-model.ts';
import {BaseGLTFNode, CrossSectionGLTFNode, GLTFNode, GLTFNodeTracked, VisualizationGLTFNode} from './gltf-node.ts';
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

suite('GLTFNode', () => {
  test('clone()', () => {
    const parent = new GLTFNode();
    const node = new GLTFNode(parent);
    const clone = node.clone();

    expect(clone).toBeInstanceOf(BaseGLTFNode);
    expect(clone).toBeInstanceOf(GLTFNode);
    expect(clone.parent).toEqual(parent);
  });

  test('clone(newParent)', () => {
    const parent = new GLTFNode();
    const node = new GLTFNode(parent);
    const cloneParent = new GLTFNode();
    const clone = node.clone(cloneParent);

    expect(clone).toBeInstanceOf(BaseGLTFNode);
    expect(clone).toBeInstanceOf(GLTFNode);
    expect(clone.parent).toEqual(cloneParent);
  });
});

suite('GLTFNodeTracked', () => {
  test('clone()', () => {
    const parent = new GLTFNode();
    const node = new GLTFNodeTracked(parent);
    const clone = node.clone();

    expect(clone).toBeInstanceOf(BaseGLTFNode);
    expect(clone).toBeInstanceOf(GLTFNode);
    expect(clone).toBeInstanceOf(GLTFNodeTracked);
    expect(clone.parent).toEqual(parent);
  });

  test('clone(newParent)', () => {
    const parent = new GLTFNode();
    const node = new GLTFNodeTracked(parent);
    const cloneParent = new GLTFNode();
    const clone = node.clone(cloneParent);

    expect(clone).toBeInstanceOf(BaseGLTFNode);
    expect(clone).toBeInstanceOf(GLTFNode);
    expect(clone).toBeInstanceOf(GLTFNodeTracked);
    expect(clone.parent).toEqual(cloneParent);
  });
});

suite('VisualizationGLTFNode', () => {
  test('clone()', async () => {
    const url = new URL('../test/fixtures/models/box.glb', import.meta.url);
    const doc = await importer.readModel(url);
    const parent = new GLTFNode();
    const node = new VisualizationGLTFNode(parent);
    node.document = doc;
    const clone = node.clone();

    expect(clone).toBeInstanceOf(BaseGLTFNode);
    expect(clone).toBeInstanceOf(VisualizationGLTFNode);
    expect(clone.document).toEqual(doc);
    expect(clone.parent).toEqual(parent);
  });

  test('clone(newParent)', async () => {
    const url = new URL('../test/fixtures/models/box.glb', import.meta.url);
    const doc = await importer.readModel(url);
    const parent = new GLTFNode();
    const node = new VisualizationGLTFNode(parent);
    node.document = doc;
    const cloneParent = new GLTFNode();
    const clone = node.clone(cloneParent);

    expect(clone).toBeInstanceOf(BaseGLTFNode);
    expect(clone).toBeInstanceOf(VisualizationGLTFNode);
    expect(clone.document).toEqual(doc);
    expect(clone.parent).toEqual(cloneParent);
  });
});

suite('CrossSectionGTLFNode', () => {
  test('clone()', async () => {
    const parent = new GLTFNode();
    const node = new CrossSectionGLTFNode(parent)
    const clone = node.clone();

    expect(clone).toBeInstanceOf(BaseGLTFNode);
    expect(clone).toBeInstanceOf(CrossSectionGLTFNode);
    expect(clone.parent).toEqual(parent);
  });

  test('clone(newParent)', async () => {
    const parent = new GLTFNode();
    const node = new CrossSectionGLTFNode(parent)
    const cloneParent = new GLTFNode();
    const clone = node.clone(cloneParent);

    expect(clone).toBeInstanceOf(BaseGLTFNode);
    expect(clone).toBeInstanceOf(CrossSectionGLTFNode);
    expect(clone.parent).toEqual(cloneParent);
  });
});

suite('anyToGLTFNodeList()', () => {
  test('Rejects generic objects', async () => {
    const script = `export default {};`;

    const fn = async () => await worker.evaluate(script);
    await expect(fn()).rejects.toThrow();
  });

  test('Rejects empty arrays', async () => {
    const script = `export default [];`;

    const fn = async () => await worker.evaluate(script);
    await expect(fn()).rejects.toThrow();
  });

  test('Accepts CrossSection objects', async () => {
    const script = `import {CrossSection} from 'manifold-3d/manifoldCAD';\n` +
        `export default CrossSection.circle(1.0);`;

    const fn = async () => await worker.evaluate(script);
    await expect(fn()).resolves.toBeDefined();
  });

  test('Accepts Manifold objects', async () => {
    const script = `import {Manifold} from 'manifold-3d/manifoldCAD';\n` +
        `export default Manifold.cube(1.0);`;

    const fn = async () => await worker.evaluate(script);
    await expect(fn()).resolves.toBeDefined();
  });

  test('Accepts empty Manifold objects', async () => {
    const script = `import {Manifold} from 'manifold-3d/manifoldCAD';\n` +
        `export default Manifold.hull([]);`;

    const fn = async () => await worker.evaluate(script);
    await expect(fn()).resolves.toBeDefined();
  });

  test('Throws when .crossSection is empty', async () => {
    const script =
        `import {CrossSectionGLTFNode} from 'manifold-3d/manifoldCAD';\n` +
        `const node = new CrossSectionGLTFNode()` +
        `export default node;`;

    const fn = async () => await worker.evaluate(script);
    await expect(fn()).rejects.toThrow();
  });
});