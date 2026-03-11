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

import * as GLTFTransform from '@gltf-transform/core';
import {unzipSync} from 'fflate';
import {afterEach, beforeAll, expect, suite, test} from 'vitest';

import {expectMeshesMatch} from '../test/util.ts';

import {exportFormats as exportFormats3MF, toArrayBuffer as toArrayBuffer3MF} from './export-3mf.ts';
import * as exportModel from './export-model.ts';
import {importManifold as importManifold3MF} from './import-model.ts';
import * as wasm from './wasm.ts';
import * as worker from './worker.ts';

suite('exportFormats', () => {
  test('contains a single 3mf format entry', () => {
    expect(exportFormats3MF).toHaveLength(1);
    expect(exportFormats3MF[0].extension).toBe('3mf');
    expect(exportFormats3MF[0].mimetype).toBe('model/3mf');
  });
});

suite('export-model integration', () => {
  test('supports 3mf extension and mimetype, rejects unknown', () => {
    expect(exportModel.supports('3mf', false)).toBe(true);
    expect(exportModel.supports('model/3mf', false)).toBe(true);
    expect(exportModel.supports('xyz', false)).toBe(false);
  });
});

suite('toArrayBuffer with empty document', () => {
  test(
      'output is a valid ZIP containing 3D/3dmodel.model and content types',
      async () => {
        const doc = new GLTFTransform.Document();
        const result = await toArrayBuffer3MF(doc);
        expect(result).toBeInstanceOf(ArrayBuffer);
        const bytes = new Uint8Array(result);
        // ZIP magic bytes
        expect(bytes[0]).toBe(0x50);  // 'P'
        expect(bytes[1]).toBe(0x4b);  // 'K'
        const files = unzipSync(bytes);
        expect(files['3D/3dmodel.model']).toBeDefined();
        expect(files['[Content_Types].xml']).toBeDefined();
      });
});

suite('toArrayBuffer header options', () => {
  test(
      'default header contains ManifoldCAD.org application and millimeter unit',
      async () => {
        const doc = new GLTFTransform.Document();
        const result = await toArrayBuffer3MF(doc);
        const files = unzipSync(new Uint8Array(result));
        const modelXml = new TextDecoder().decode(files['3D/3dmodel.model']);
        expect(modelXml).toContain('ManifoldCAD.org');
        expect(modelXml).toContain('millimeter');
      });

  test('custom title and unit appear in model XML', async () => {
    const doc = new GLTFTransform.Document();
    const result = await toArrayBuffer3MF(
        doc, {header: {title: 'MyTestModel', unit: 'inch'}});
    const files = unzipSync(new Uint8Array(result));
    const modelXml = new TextDecoder().decode(files['3D/3dmodel.model']);
    expect(modelXml).toContain('MyTestModel');
    expect(modelXml).toContain('inch');
  });
});

suite('toArrayBuffer with manifold models', () => {
  beforeAll(async () => await wasm.getManifoldModule());
  afterEach(async () => worker.cleanup());

  test(
      'cube: exports to valid 3MF and round-trips to correct geometry',
      async () => {
        const script = `import {Manifold} from 'manifold-3d/manifoldCAD';\n` +
            `export default Manifold.cube([100,100,100]);`;
        const doc = await worker.evaluate(script);
        const {Manifold} = wasm.getManifoldModuleSync()!;
        const sourceMesh = Manifold.cube([100, 100, 100]).getMesh();
        const result = await toArrayBuffer3MF(doc);
        expect(result).toBeInstanceOf(ArrayBuffer);
        expect(unzipSync(new Uint8Array(result))['3D/3dmodel.model'])
            .toBeDefined();

        const model = await importManifold3MF(result, {mimetype: 'model/3mf'});
        const roundTripMesh = model.getMesh();
        expectMeshesMatch(sourceMesh, roundTripMesh);

        expect(model.volume()).toBeCloseTo(100 * 100 * 100, 1);
        expect(model.genus()).toBe(0);
      });

  test(
      'sphere: exports to valid 3MF and round-trips to correct geometry',
      async () => {
        const script = `import {Manifold} from 'manifold-3d/manifoldCAD';\n` +
            `export default Manifold.sphere(10, 64);`;
        const doc = await worker.evaluate(script);
        const {Manifold} = wasm.getManifoldModuleSync()!;
        const sourceMesh = Manifold.sphere(10, 64).getMesh();
        const result = await toArrayBuffer3MF(doc);

        const model = await importManifold3MF(result, {mimetype: 'model/3mf'});
        const roundTripMesh = model.getMesh();
        expectMeshesMatch(sourceMesh, roundTripMesh);
      });

  test(
      'GLTFNode scene: dispatches correctly via extension and mimetype',
      async () => {
        const script =
            `import {Manifold, GLTFNode} from 'manifold-3d/manifoldCAD';\n` +
            `const node = new GLTFNode();\n` +
            `node.manifold = Manifold.cube([1,1,1]);\n` +
            `export default node;`;
        const doc = await worker.evaluate(script);

        const result = await toArrayBuffer3MF(doc);
        expect(result).toBeInstanceOf(ArrayBuffer);
        expect(unzipSync(new Uint8Array(result))['3D/3dmodel.model'])
            .toBeDefined();

        // Verify dispatch through export-model works for both identifiers.
        const byExt = await exportModel.toArrayBuffer(doc, {extension: '3mf'});
        const byMime =
            await exportModel.toArrayBuffer(doc, {mimetype: 'model/3mf'});
        expect(new Uint8Array(byExt)[0]).toBe(0x50);
        expect(new Uint8Array(byMime)[0]).toBe(0x50);
      });

  test(
      'shared-manifold instances round-trip with component transforms',
      async () => {
        const script =
            `import {Manifold, GLTFNode} from 'manifold-3d/manifoldCAD';\n` +
            `const shared = Manifold.cube([2,2,2]);\n` +
            `const parent = new GLTFNode();\n` +
            `parent.manifold = shared;\n` +
            `parent.translation = [7,0,0];\n` +
            `const child = new GLTFNode(parent);\n` +
            `child.manifold = shared;\n` +
            `child.translation = [5,0,0];\n` +
            `export default [parent, child];`;
        const doc = await worker.evaluate(script);
        const result = await toArrayBuffer3MF(doc);
        const files = unzipSync(new Uint8Array(result));
        const modelXml = new TextDecoder().decode(files['3D/3dmodel.model']);
        expect(modelXml).toContain('<vertices>');
        expect(modelXml).toContain('<triangles>');
        expect(modelXml).toContain('<component');

        const model = await importManifold3MF(result, {mimetype: 'model/3mf'});
        expect(model.volume()).toBeCloseTo(16, 1);
      });
});
