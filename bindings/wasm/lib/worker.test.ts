// Copyright 2022-2025 The Manifold Authors.
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

import {strict as assert} from 'assert';
import {glob} from 'glob';
import * as fs from 'node:fs/promises';
import {resolve} from 'path';
import {afterEach, expect, suite, test} from 'vitest';

// @ts-ignore
import {Mesh} from '../manifold';

import {gltfDocToManifold, importModel} from './import-model.ts';
import {cleanup, evaluate, exportBlobURL} from './worker.ts';

async function resolveExample(name: string) {
  const [filepath] =
      await glob(name.toLowerCase().replaceAll(' ', '-') + '.{ts,mjs,js}', {
        cwd: resolve(import.meta.dirname, '../test/examples/'),
        withFileTypes: true
      });
  if (!filepath) {
    throw new Error(`Could not find example '${name}'.`);
  }
  return filepath.fullpath()
}

async function runExample(name: string) {
  const filename = await resolveExample(name);
  const code = await fs.readFile(filename, 'utf-8');
  const doc = await evaluate(code, {jsCDN: 'jsDelivr', filename});
  const glbURL = await exportBlobURL(doc, 'glb')
  cleanup();
  assert.ok(glbURL);

  // These tests are agains the first glTF node containing meshes in a given
  // model.
  const {document} = await importModel(glbURL, {mimetype: 'model/gltf-binary'});
  const node = document.getRoot().listNodes().find(node => !!node.getMesh());
  const manifold = await gltfDocToManifold(document, node);
  URL.revokeObjectURL(glbURL);

  if (manifold) {
    const volume = manifold.volume();
    const surfaceArea = manifold.surfaceArea();
    const genus = manifold.genus();
    cleanup();
    return {volume, surfaceArea, genus};
  }
  assert.ok(false);
}

// allow vitest to report progress after each test
// before going into heavy computation which blocks main thread
afterEach(async () => {await new Promise(resolve => setTimeout(resolve, 500))})

suite('Examples', () => {
  test('Intro', async () => {
    const result = await runExample('Intro');
    expect(result?.genus).to.equal(5, 'Genus');
    expect(result?.volume).to.be.closeTo(203164, 1, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(62046, 1, 'Surface Area');
  });

  test('Auger', async () => {
    const result = await runExample('Auger');
    expect(result?.genus).to.equal(0, 'Genus');
    expect(result?.volume).to.be.closeTo(16835, 1, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(10519, 1, 'Surface Area');
  });

  test('Tetrahedron Puzzle', async () => {
    const result = await runExample('Tetrahedron Puzzle');
    expect(result?.genus).to.equal(0, 'Genus');
    expect(result?.volume).to.be.closeTo(7223, 1, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(3235, 1, 'Surface Area');
  });

  test('Rounded Frame', async () => {
    const result = await runExample('Rounded Frame');
    expect(result?.genus).to.equal(5, 'Genus');
    expect(result?.volume).to.be.closeTo(270807, 10, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(74599, 1, 'Surface Area');
  });

  test('Heart', async () => {
    const result = await runExample('Heart');
    expect(result?.genus).to.equal(0, 'Genus');
    expect(result?.volume).to.be.closeTo(282742, 10, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(22186, 1, 'Surface Area');
  });

  test('Scallop', async () => {
    const result = await runExample('Scallop');
    expect(result?.genus).to.equal(0, 'Genus');
    expect(result?.volume).to.be.closeTo(39900, 100, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(7930, 10, 'Surface Area');
  });

  test('Torus Knot', async () => {
    const result = await runExample('Torus Knot');
    expect(result?.genus).to.equal(1, 'Genus');
    expect(result?.volume).to.be.closeTo(20786, 1, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(11177, 1, 'Surface Area');
  });

  test('Menger Sponge', async () => {
    const result = await runExample('Menger Sponge');
    expect(result?.genus).to.equal(729, 'Genus');
    expect(result?.volume).to.be.closeTo(203222, 10, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(130475, 10, 'Surface Area');
  });

  test('Stretchy Bracelet', async () => {
    const result = await runExample('Stretchy Bracelet');
    expect(result?.genus).to.equal(1, 'Genus');
    expect(result?.volume).to.be.closeTo(3992, 1, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(22267, 1, 'Surface Area');
  });

  test('Gyroid Module', async () => {
    const result = await runExample('Gyroid Module');
    expect(result?.genus).to.equal(15, 'Genus');
    expect(result?.volume).to.be.closeTo(4175, 1, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(5645, 1, 'Surface Area');
  });

  test('Involute Gear Library', async () => {
    const result = await runExample('Involute Gear Library');
    expect(result?.genus).to.equal(0, 'Genus');
    expect(result?.volume).to.be.closeTo(2185, 1, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(1667, 1, 'Surface Area');
  });

  test('Gear Bearing', async () => {
    const result = await runExample('Gear Bearing');
    expect(result?.genus).to.equal(1, 'Genus');
    expect(result?.volume).to.be.closeTo(9074, 1, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(7009, 1, 'Surface Area');
  });

  test('Voronoi', async () => {
    const result = await runExample('Voronoi');
    // This model is non-deterministic.
    // These values must be very conservative.
    expect(result?.genus).to.be.lessThan(-25, 'Genus');
    expect(result?.volume).to.be.greaterThan(5000, 'Volume');
    expect(result?.surfaceArea).to.be.greaterThan(10000, 'Surface Area');
  });

  test('Import Manifold', async () => {
    const result = await runExample('Import Manifold');
    expect(result?.genus).to.equal(3, 'Genus');
    // There are a 1e9 cubic millimeters in a cubic metre.
    // They add up fast.
    expect(result?.volume).to.be.closeTo(2.10e15, 1e13, 'Volume');
    expect(result?.surfaceArea).to.be.closeTo(1.67e11, 1e9, 'Surface Area');
  });
});
