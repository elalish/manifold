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

import {Document} from '@gltf-transform/core';
import {getSceneVertexCount, VertexCountMethod} from '@gltf-transform/functions';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {afterEach, beforeEach, expect, suite, test, vi} from 'vitest';

import {bundleFile} from '../lib/bundler.ts';
import * as scenebuilder from '../lib/scene-builder';
import * as worker from '../lib/worker.ts';

const countVertices = (doc: Document) => {
  const scene = doc.getRoot().listScenes()[0];
  if (!scene) return -1;
  return getSceneVertexCount(scene, VertexCountMethod.UPLOAD_NAIVE);
};

beforeEach(() => worker.cleanup());

suite('From a string, the worker will', () => {
  test('Build a model', async () => {
    const filename = resolve(import.meta.dirname, './fixtures/unitSphere.mjs');
    const code = await readFile(filename, 'utf-8');
    const result = await worker.evaluate(code, {filename});
    expect(countVertices(result)).toBeGreaterThan(0);
  });

  test('Convert a Manifold object into a GLTF Document', async () => {
    const filename = resolve(import.meta.dirname, './fixtures/unitSphere.mjs');
    const code = await readFile(filename, 'utf-8');
    const result = await worker.evaluate(code, {filename});
    expect(result).toBeInstanceOf(Document);
  });

  test('Collect GLTFNodes into a GLTF Document', async () => {
    const filename =
        resolve(import.meta.dirname, './fixtures/unitSphereGLTF.mjs');
    const code = await readFile(filename, 'utf-8');
    const result = await worker.evaluate(code, {filename});
    expect(result).toBeInstanceOf(Document);
  });

  test('Build a model with local imports', async () => {
    const filename =
        resolve(import.meta.dirname, './fixtures/importUnitSphere.mjs');
    const code = await readFile(filename, 'utf-8');
    const result = await worker.evaluate(code, {filename});
    expect(countVertices(result)).toBeGreaterThan(0);
  });

  test('Fail to find local imports without a filename', async () => {
    const filename =
        resolve(import.meta.dirname, './fixtures/importUnitSphere.mjs');
    const code = await readFile(filename, 'utf-8');
    const ev = async () => await worker.evaluate(code);
    await expect(ev()).rejects.toThrowError();
  });

  test('Bundle npm imports from a CDN', async () => {
    const filename = resolve(import.meta.dirname, './examples/voronoi.mjs');
    const code = await readFile(filename, 'utf-8');
    const result = await worker.evaluate(code, {filename, jsCDN: 'jsDelivr'});
    expect(countVertices(result)).toBeGreaterThan(0);
  });

  test('Bundle npm imports from a CDN without a filename', async () => {
    const filename = resolve(import.meta.dirname, './examples/voronoi.mjs');
    const code = await readFile(filename, 'utf-8');
    const result = await worker.evaluate(code, {jsCDN: 'jsDelivr'});
    expect(countVertices(result)).toBeGreaterThan(0);
  });

  test('Build even if missing a top-level manifoldCAD import', async () => {
    const filename =
        resolve(import.meta.dirname, './fixtures/importUnitSphere.mjs');
    const code = await readFile(filename, 'utf-8');
    const result = await worker.evaluate(code, {filename});
    expect(countVertices(result)).toBeGreaterThan(0);
  });

  test('make isManifoldCAD() return true', async () => {
    const filename =
        resolve(import.meta.dirname, './fixtures/isManifoldCAD.mjs');
    const code = await readFile(filename, 'utf-8');
    const result = await worker.evaluate(code, {filename});
    expect(countVertices(result)).toBe(8);
  });
});

suite('From the filesystem, the worker will', () => {
  test('Build a model', async () => {
    const filename = resolve(import.meta.dirname, './fixtures/unitSphere.mjs');
    const bundle = await bundleFile(filename);
    const result = await worker.evaluate(bundle, {doNotBundle: true});
    expect(countVertices(result)).toBeGreaterThan(0);
  });

  test('Convert a Manifold object into a GLTF Document', async () => {
    const filename = resolve(import.meta.dirname, './fixtures/unitSphere.mjs');
    const bundle = await bundleFile(filename);
    const result = await worker.evaluate(bundle, {doNotBundle: true});
    expect(result).toBeInstanceOf(Document);
  });

  test('Collect GLTFNodes into a GLTF Document', async () => {
    const filename =
        resolve(import.meta.dirname, './fixtures/unitSphereGLTF.mjs');
    const bundle = await bundleFile(filename);
    const result = await worker.evaluate(bundle, {doNotBundle: true});
    expect(result).toBeInstanceOf(Document);
  });

  test('Build a model with local imports', async () => {
    const filename =
        resolve(import.meta.dirname, './fixtures/importUnitSphere.mjs');
    const bundle = await bundleFile(filename);
    const result = await worker.evaluate(bundle, {doNotBundle: true});
    expect(countVertices(result)).toBeGreaterThan(0);
  });

  test('Build even if missing a top-level manifoldCAD import', async () => {
    const filename =
        resolve(import.meta.dirname, './fixtures/importUnitSphere.mjs');
    const bundle = await bundleFile(filename);
    const ev = async () => await worker.evaluate(bundle, {doNotBundle: true});
    await expect(ev()).resolves.toBeInstanceOf(Document);
  });

  test('Bundle npm imports from a CDN', async () => {
    const filename = resolve(import.meta.dirname, './examples/voronoi.mjs');
    const bundle = await bundleFile(filename, {jsCDN: 'jsDelivr'});
    const result = await worker.evaluate(bundle, {doNotBundle: true});
    expect(countVertices(result)).toBeGreaterThan(0);
  });

  test('make isManifoldCAD() return true', async () => {
    const filename =
        resolve(import.meta.dirname, './fixtures/isManifoldCAD.mjs');
    const bundle = await bundleFile(filename);
    const result = await worker.evaluate(bundle, {doNotBundle: true});
    expect(countVertices(result)).toBe(8);
  });
});

suite('Without a worker, an import will', () => {
  test('Build a model', async () => {
    const {default: result} = await import('./fixtures/unitSphere.mjs');
    expect(result.volume()).toBeCloseTo(2.9428, 0.1);
  });

  test('Not convert Manifold objects', async () => {
    const {default: result} = await import('./fixtures/unitSphere.mjs');
    expect(result).not.toBeInstanceOf(Document);
    expect(result.constructor.name).toBe('Manifold');
  });

  test('Not convert GLTFNode objects', async () => {
    const {default: result} = await import('./fixtures/unitSphereGLTF.mjs');

    expect(result.constructor.name).toBe('GLTFNode');
  });

  test('Build a model with local imports', async () => {
    const {default: result} = await import('./fixtures/importUnitSphere.mjs');
    expect(result.volume()).toBeCloseTo(2.9428, 0.1);
  });

  test('Throw if missing a top-level manifoldCAD import', async () => {
    const ev = async () => {
      const {default: result} =
          await import('./fixtures/unitSphereNoManifoldImport.mjs');
      return result;
    };
    await expect(ev()).rejects.toThrowError();
  });

  test('Throw if imports are not present', async () => {
    const ev = async () => {
      const {default: result} = await import('./examples/voronoi.mjs');
      return result;
    };
    await expect(ev()).rejects.toThrowError();
  });

  test('make isManifoldCAD() return false', async () => {
    const {isManifoldCADReturns, default: result} =
        await import('./fixtures/isManifoldCAD.mjs');
    expect(isManifoldCADReturns).toBeFalsy();
    expect(isManifoldCADReturns).toBeTypeOf('boolean');
    expect(result.numVert()).toBeGreaterThan(8);
  })
});
