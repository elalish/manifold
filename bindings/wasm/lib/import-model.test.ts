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

import * as fs from 'node:fs/promises';
import {afterEach, beforeAll, expect, suite, test} from 'vitest';

import * as importer from './import-model.ts';
import * as wasm from './wasm.ts';
import * as worker from './worker.ts';

beforeAll(async () => await wasm.getManifoldModule());
afterEach(async () => worker.cleanup());

suite('supports()', () => {
  test('returns true for a known good mimetype', () => {
    expect(importer.supports('model/gltf-binary')).to.be.true;
  });

  test('returns true for a known good mimetype', () => {
    expect(importer.supports('.glb')).to.be.true;
  });

  test('returns false for a known bad mimetype', () => {
    expect(importer.supports('model/not-a-real-3d-format')).to.be.false;
  });

  test('returns false for a known bad mimetype', () => {
    expect(importer.supports('.not-a-real-3d-format')).to.be.false;
  });

  test('Throws on demand', () => {
    expect(() => importer.supports('model/not-a-real-3d-format', true))
        .to.throw();
  });
});

suite('importModel()', () => {
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

  test('succeeds when tolerance permits a non-manifold model', async () => {
    const model = await importer.importManifold(
        new URL('../test/fixtures/models/boxNotManifold.glb', import.meta.url),
        {tolerance: 0.005});
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
});
