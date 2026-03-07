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

import {exportFormats, toArrayBuffer} from './export-3mf.ts';
import * as exportModel from './export-model.ts';
import * as wasm from './wasm.ts';
import * as worker from './worker.ts';

suite('exportFormats', () => {
  test('contains a single 3mf format entry', () => {
    expect(exportFormats).toHaveLength(1);
    expect(exportFormats[0].extension).toBe('3mf');
    expect(exportFormats[0].mimetype).toBe('model/3mf');
  });
});

suite('export-model integration', () => {
  test('supports 3mf extension', () => {
    expect(exportModel.supports('3mf', false)).toBe(true);
  });

  test('supports model/3mf mimetype', () => {
    expect(exportModel.supports('model/3mf', false)).toBe(true);
  });

  test('does not support unknown format', () => {
    expect(exportModel.supports('xyz', false)).toBe(false);
  });
});

suite('toArrayBuffer with empty document', () => {
  test('returns an ArrayBuffer', async () => {
    const doc = new GLTFTransform.Document();
    const result = await toArrayBuffer(doc);
    expect(result).toBeInstanceOf(ArrayBuffer);
    expect(result.byteLength).toBeGreaterThan(0);
  });

  test('output starts with ZIP magic bytes', async () => {
    const doc = new GLTFTransform.Document();
    const result = await toArrayBuffer(doc);
    const bytes = new Uint8Array(result);
    expect(bytes[0]).toBe(0x50);  // 'P'
    expect(bytes[1]).toBe(0x4b);  // 'K'
  });

  test('ZIP contains 3D/3dmodel.model', async () => {
    const doc = new GLTFTransform.Document();
    const result = await toArrayBuffer(doc);
    const files = unzipSync(new Uint8Array(result));
    expect(files['3D/3dmodel.model']).toBeDefined();
  });

  test('ZIP contains [Content_Types].xml', async () => {
    const doc = new GLTFTransform.Document();
    const result = await toArrayBuffer(doc);
    const files = unzipSync(new Uint8Array(result));
    expect(files['[Content_Types].xml']).toBeDefined();
  });
});

suite('toArrayBuffer header options', () => {
  test('default header contains ManifoldCAD.org application', async () => {
    const doc = new GLTFTransform.Document();
    const result = await toArrayBuffer(doc);
    const files = unzipSync(new Uint8Array(result));
    const modelXml = new TextDecoder().decode(files['3D/3dmodel.model']);
    expect(modelXml).toContain('ManifoldCAD.org');
  });

  test('custom title appears in model XML', async () => {
    const doc = new GLTFTransform.Document();
    const result = await toArrayBuffer(doc, {header: {title: 'MyTestModel'}});
    const files = unzipSync(new Uint8Array(result));
    const modelXml = new TextDecoder().decode(files['3D/3dmodel.model']);
    expect(modelXml).toContain('MyTestModel');
  });

  test('custom unit appears in model XML', async () => {
    const doc = new GLTFTransform.Document();
    const result = await toArrayBuffer(doc, {header: {unit: 'inch'}});
    const files = unzipSync(new Uint8Array(result));
    const modelXml = new TextDecoder().decode(files['3D/3dmodel.model']);
    expect(modelXml).toContain('inch');
  });

  test('default unit is millimeter', async () => {
    const doc = new GLTFTransform.Document();
    const result = await toArrayBuffer(doc);
    const files = unzipSync(new Uint8Array(result));
    const modelXml = new TextDecoder().decode(files['3D/3dmodel.model']);
    expect(modelXml).toContain('millimeter');
  });
});

suite('toArrayBuffer with manifold models', () => {
  beforeAll(async () => await wasm.getManifoldModule());
  afterEach(async () => worker.cleanup());

  test('exports a cube', async () => {
    const script = `import {Manifold} from 'manifold-3d/manifoldCAD';\n` +
        `export default Manifold.cube([1,1,1]);`;
    const doc = await worker.evaluate(script);
    const result = await toArrayBuffer(doc);
    expect(result).toBeInstanceOf(ArrayBuffer);
    expect(result.byteLength).toBeGreaterThan(0);
    const files = unzipSync(new Uint8Array(result));
    expect(files['3D/3dmodel.model']).toBeDefined();
  });

  test('cube model XML contains vertices and triangles', async () => {
    const script = `import {Manifold} from 'manifold-3d/manifoldCAD';\n` +
        `export default Manifold.cube([1,1,1]);`;
    const doc = await worker.evaluate(script);
    const result = await toArrayBuffer(doc);
    const files = unzipSync(new Uint8Array(result));
    const modelXml = new TextDecoder().decode(files['3D/3dmodel.model']);
    expect(modelXml).toContain('<vertices>');
    expect(modelXml).toContain('<triangles>');
  });

  test('exports a sphere', async () => {
    const script = `import {Manifold} from 'manifold-3d/manifoldCAD';\n` +
        `export default Manifold.sphere(10, 64);`;
    const doc = await worker.evaluate(script);
    const result = await toArrayBuffer(doc);
    const files = unzipSync(new Uint8Array(result));
    const modelXml = new TextDecoder().decode(files['3D/3dmodel.model']);
    expect(modelXml).toContain('<vertices>');
    expect(modelXml).toContain('<triangles>');
  });

  test('exports a GLTFNode scene', async () => {
    const script =
        `import {Manifold, GLTFNode} from 'manifold-3d/manifoldCAD';\n` +
        `const node = new GLTFNode();\n` +
        `node.manifold = Manifold.cube([1,1,1]);\n` +
        `export default node;`;
    const doc = await worker.evaluate(script);
    const result = await toArrayBuffer(doc);
    expect(result).toBeInstanceOf(ArrayBuffer);
    const files = unzipSync(new Uint8Array(result));
    expect(files['3D/3dmodel.model']).toBeDefined();
  });

  test('exports a parent-child node hierarchy', async () => {
    const script =
        `import {Manifold, GLTFNode} from 'manifold-3d/manifoldCAD';\n` +
        `const parent = new GLTFNode();\n` +
        `parent.manifold = Manifold.cube([2,2,2]);\n` +
        `const child = new GLTFNode(parent);\n` +
        `child.manifold = Manifold.cube([1,1,1]);\n` +
        `export default parent;`;
    const doc = await worker.evaluate(script);
    const result = await toArrayBuffer(doc);
    expect(result).toBeInstanceOf(ArrayBuffer);
    const files = unzipSync(new Uint8Array(result));
    const modelXml = new TextDecoder().decode(files['3D/3dmodel.model']);
    // Both parent and child meshes should appear in the output
    expect(modelXml).toContain('<vertices>');
    expect(modelXml).toContain('<triangles>');
    // A component referencing a child should be present
    expect(modelXml).toContain('<component');
  });

  test('dispatches through export-model.toArrayBuffer with extension', async () => {
    const script = `import {Manifold} from 'manifold-3d/manifoldCAD';\n` +
        `export default Manifold.cube([1,1,1]);`;
    const doc = await worker.evaluate(script);
    const result = await exportModel.toArrayBuffer(doc, {extension: '3mf'});
    expect(result).toBeInstanceOf(ArrayBuffer);
    const bytes = new Uint8Array(result);
    expect(bytes[0]).toBe(0x50);  // ZIP magic 'P'
    expect(bytes[1]).toBe(0x4b);  // ZIP magic 'K'
  });

  test('dispatches through export-model.toArrayBuffer with mimetype', async () => {
    const script = `import {Manifold} from 'manifold-3d/manifoldCAD';\n` +
        `export default Manifold.cube([1,1,1]);`;
    const doc = await worker.evaluate(script);
    const result =
        await exportModel.toArrayBuffer(doc, {mimetype: 'model/3mf'});
    expect(result).toBeInstanceOf(ArrayBuffer);
    const files = unzipSync(new Uint8Array(result));
    expect(files['3D/3dmodel.model']).toBeDefined();
  });
});
