// Copyright 2025 The Manifold Authors.
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

import * as validator from 'gltf-validator';
import * as fs from 'node:fs/promises';
import {afterEach, beforeAll, expect, suite, test} from 'vitest';

import type {Vec3} from '../manifold.d.ts';
import {equalsVec3Array, meshToVec3Array} from '../test/util.ts';

import {toArrayBuffer} from './export-model.ts';
import * as importer from './import-model.ts';
import * as wasm from './wasm.ts';
import * as worker from './worker.ts';

beforeAll(async () => await wasm.getManifoldModule());
afterEach(async () => worker.cleanup());

suite('supports()', () => {
  test('returns true for a known good mimetype', () => {
    expect(importer.supports('model/gltf-binary')).to.be.true;
  });

  test('returns true for 3mf mimetype', () => {
    expect(importer.supports('model/3mf')).to.be.true;
  });

  test('returns true for a known good mimetype', () => {
    expect(importer.supports('.glb')).to.be.true;
  });

  test('returns true for 3mf extension', () => {
    expect(importer.supports('.3mf')).to.be.true;
  });

  test('returns false for a known bad mimetype', () => {
    expect(importer.supports('model/not-a-real-3d-format')).to.be.false;
  });

  test('returns false for a known bad extension', () => {
    expect(importer.supports('.not-a-real-3d-format')).to.be.false;
  });

  test('Throws on demand', () => {
    expect(() => importer.supports('model/not-a-real-3d-format', true))
        .to.throw();
  });
});

suite('importManifold()', () => {
  test('imports a model with EXT_mesh_manifold', async () => {
    const model = await importer.importManifold(new URL(
        '../test/fixtures/models/boxExtMeshManifold.glb', import.meta.url));
    expect(model.volume()).to.be.closeTo(100 * 100 * 100, 1);
  });

  test('imports a model without EXT_mesh_manifold', async () => {
    const model = await importer.importManifold(
        new URL('../test/fixtures/models/box.glb', import.meta.url));
    expect(model.volume()).to.be.closeTo(100 * 100 * 100, 1);
  });

  test('throws when model is not manifold', async () => {
    const fn = async () => await importer.importManifold(
        new URL('../test/fixtures/models/boxNotManifold.glb', import.meta.url));
    await expect(fn).rejects.toThrowError();
  });

  test.skip(
      'succeeds when tolerance permits a non-manifold model', async () => {
        const model = await importer.importManifold(
            new URL(
                '../test/fixtures/models/boxNotManifold.glb', import.meta.url),
            {tolerance: 0.01});
        expect(model.volume()).to.be.closeTo(100 * 100 * 100, 1);
      });

  test('throws when tolerance is insufficient', async () => {
    const fn = async () => await importer.importManifold(
        new URL('../test/fixtures/models/boxNotManifold.glb', import.meta.url),
        {tolerance: 0.001});
    await expect(fn).rejects.toThrowError();
  });

  test('import a model by file URL', async () => {
    const model = await importer.importManifold(
        new URL('../test/fixtures/models/box.glb', import.meta.url));
    expect(model.volume()).to.be.closeTo(100 * 100 * 100, 1);
  });

  test('import a model by absolute path', async () => {
    const model = await importer.importManifold(
        import.meta.dirname + '/../test/fixtures/models/box.glb');
    expect(model.volume()).to.be.closeTo(100 * 100 * 100, 1);
  });

  test('import a model from a Blob (with mimetype)', async () => {
    const buffer = await fs.readFile(
        import.meta.dirname + '/../test/fixtures/models/box.glb');
    const blob =
        new Blob([buffer.buffer as ArrayBuffer], {type: 'model/gltf-binary'});

    const model = await importer.importManifold(blob);
    expect(model.volume()).to.be.closeTo(100 * 100 * 100, 1);
  });

  test('import a model from a Blob (specifying mimetype)', async () => {
    const buffer = await fs.readFile(
        import.meta.dirname + '/../test/fixtures/models/box.glb');
    const blob = new Blob([buffer.buffer as ArrayBuffer]);

    const model =
        await importer.importManifold(blob, {mimetype: 'model/gltf-binary'});
    expect(model.volume()).to.be.closeTo(100 * 100 * 100, 1);
  });

  test('import a model from an ArrayBuffer', async () => {
    const buffer = await fs.readFile(
        import.meta.dirname + '/../test/fixtures/models/box.glb');

    const model = await importer.importManifold(
        buffer.buffer as ArrayBuffer, {mimetype: 'model/gltf-binary'});
    expect(model.volume()).to.be.closeTo(100 * 100 * 100, 1);
  });

  test('imports 3mf through the main importer path', async () => {
    const script = `import {Manifold} from \'manifold-3d/manifoldCAD\';\n` +
        `export default Manifold.cube([100,100,100]);\n`;
    const doc = await worker.evaluate(script);
    const buffer = await toArrayBuffer(doc, {mimetype: 'model/3mf'});
    const model =
        await importer.importManifold(buffer, {mimetype: 'model/3mf'});
    expect(model.volume()).to.be.closeTo(100 * 100 * 100, 1);
  });

  test('Orients a model correctly', async () => {
    // The other tests implicitly cover conversion from glTF's default
    // scale of 1:1m.  This test covers conversion between up being +Y in
    // glTF and +Z in manifold.

    // Here's an asymmetric shape.
    const before: Array<Vec3> =
        [[-1, 0, 0], [4, 0, 0], [0, -2, 0], [0, 3, 0], [0, 0, 5]];
    const script = `import {Manifold} from \'manifold-3d/manifoldCAD\';\n` +
        `export default Manifold.hull(${JSON.stringify(before)});\n`;

    // Evaluate, export it and re-import the model.
    const options = {mimetype: 'model/gltf-binary'};
    const doc = await worker.evaluate(script);
    const buffer = await toArrayBuffer(doc, options);
    const model = await importer.importManifold(buffer, options);
    const after = meshToVec3Array(model.getMesh());

    expect(equalsVec3Array(before, after)).toBeTruthy();
  });
});

suite('importModel()', () => {
  test('uses source filename when imported node has no name', async () => {
    const node = await importer.importModel(
        new URL('../test/fixtures/models/box.glb', import.meta.url));
    expect(node.name).toBe('box.glb');
  });

  test('prefers source node name when present', async () => {
    const node = await importer.importModel(
        new URL('../test/fixtures/models/boxNotManifold.glb', import.meta.url));
    expect(node.name).toBe('obj1');
  });

  test('does not infer EXT_mesh_manifold', async () => {
    const node = await importer.importModel(
        new URL('../test/fixtures/models/boxNotManifold.glb', import.meta.url));
    for (const mesh of node.document!.getRoot().listMeshes()) {
      expect(mesh.getExtension('EXT_mesh_manifold')).toBeNull();
    };
  });

  test('deduplicates textures when importing nodes', async () => {
    const url = new URL('../test/fixtures/models/space.glb', import.meta.url);

    // Export and re-import the model.
    const script = `import {importModel} from \'manifold-3d/manifoldCAD\';\n` +
        `export default async () => {\n` +
        `  node1 = await importModel("${url}");\n` +
        `  node1.translation = [-60*1000,0,0];` +
        `  node2 = await importModel("${url}");\n` +
        `  node2.translation = [+60*1000,0,0];` +
        `  return [node1, node2];\n` +
        `};`;
    const options = {mimetype: 'model/gltf-binary'};
    const doc = await worker.evaluate(script);
    const buffer = await toArrayBuffer(doc, options);
    const model = await importer.importModel(buffer, options);
    const root = model.document!.getRoot();

    expect(root.listTextures().length).to.equal(1);
  });

  test('deduplicates textures when cloning nodes', async () => {
    const url = new URL('../test/fixtures/models/space.glb', import.meta.url);

    // Export and re-import the model.
    const script = `import {importModel} from \'manifold-3d/manifoldCAD\';\n` +
        `export default async () => {\n` +
        `  node1 = await importModel("${url}");\n` +
        `  node1.translation = [-60*1000,0,0];` +
        `  node2 = node1.clone();\n` +
        `  node2.translation = [+60*1000,0,0];` +
        `  return [node1, node2];\n` +
        `};`;
    const options = {mimetype: 'model/gltf-binary'};
    const doc = await worker.evaluate(script);
    const buffer = await toArrayBuffer(doc, options);
    const model = await importer.importModel(buffer, options);
    const root = model.document!.getRoot();

    expect(root.listTextures().length).to.equal(1);
  });

  suite('when .glb file has EXT_mesh_manifold', async () => {
    // Additional tests as this code path goes through manifold-gltf.

    // Ground truth.
    const url = new URL('../test/fixtures/models/space.glb', import.meta.url);
    const beforeModel = await importer.importModel(url);

    // Export and re-import the model.
    const script = `import {importModel} from \'manifold-3d/manifoldCAD\';\n` +
        `export default async () => await importModel("${url}");\n`;
    const options = {mimetype: 'model/gltf-binary'};
    const doc = await worker.evaluate(script);
    const buffer = await toArrayBuffer(doc, options);
    const afterModel = await importer.importModel(buffer, options);

    // These are used by more than one test.
    const [beforeMesh] = beforeModel.document?.getRoot().listMeshes() ?? [];
    const [beforePrim] = beforeMesh?.listPrimitives() ?? [];
    const [afterMesh] = afterModel.document?.getRoot().listMeshes() ?? [];
    const [afterPrim] = afterMesh?.listPrimitives() ?? [];

    test('no new validation errors are introduced', async () => {
      const beforeBuffer = await fs.readFile(
          import.meta.dirname + '/../test/fixtures/models/space.glb');
      const afterBuffer = new Uint8Array(buffer)
      const before = (await validator.validateBytes(beforeBuffer)).issues;
      const after = (await validator.validateBytes(afterBuffer)).issues;

      expect(after.numErrors).toBeLessThanOrEqual(before.numErrors);
      expect(after.numWarnings).toBeLessThanOrEqual(before.numWarnings);
      expect(after.numInfos).toBeLessThanOrEqual(before.numInfos);
    });

    test('imported models retain EXT_mesh_manifold', () => {
      expect(beforeMesh.getExtension('EXT_mesh_manifold')).not.toBeNull();
    });

    test('re-exported models have the same number of meshes', () => {
      expect(beforeModel.document!.getRoot().listMeshes().length).toEqual(1);
      expect(afterModel.document!.getRoot().listMeshes().length).toEqual(1);
    });

    test('re-exported meshes retain EXT_mesh_manifold', () => {
      expect(beforeMesh.getExtension('EXT_mesh_manifold')).not.toBeNull();
      expect(afterMesh.getExtension('EXT_mesh_manifold')).not.toBeNull();
    });

    test('re-exported meshes have the same number of primitives', () => {
      expect(beforeMesh.listPrimitives().length).toEqual(1);
      expect(afterMesh.listPrimitives().length).toEqual(1);
    });

    test('re-exported primitives retain indices', () => {
      const beforeVertices = [...beforePrim.getIndices()?.getArray() ?? []];
      const afterVertices = [...beforePrim.getIndices()?.getArray() ?? []];

      expect(afterVertices).to.deep.equal(beforeVertices);
    });

    test('re-exported primitives retain attributes', () => {
      const beforeAtt = beforePrim.listAttributes().map(att => att.getName());
      const afterAtt = afterPrim.listAttributes().map(att => att.getName());

      expect(afterAtt).to.deep.equal(beforeAtt);
    });

    test('re-exported attributes retain properties', () => {
      for (const beforeAtt of beforePrim.listAttributes()) {
        const afterAtt = afterPrim.getAttribute(beforeAtt.getName());
        const beforeProperties = [...beforeAtt?.getArray() ?? []];
        const afterProperties = [...afterAtt?.getArray() ?? []];

        expect(afterProperties).to.deep.equal(beforeProperties);
      }
    });
  });
});
