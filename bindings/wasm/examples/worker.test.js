// Copyright 2022 The Manifold Authors.
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

import '@vitest/web-worker';

import {WebIO} from '@gltf-transform/core';
import {expect, suite, test} from 'vitest';

import Module from './built/manifold.js';
import {readMesh, setupIO} from './gltf-io';
import {examples} from './public/examples.js';
import ManifoldWorker from './worker?worker';

const io = setupIO(new WebIO());

const wasm = await Module();
wasm.setup();

function initialized(worker) {
  return new Promise((resolve) => {
    worker.onmessage = function(e) {
      if (e.data == null) {
        resolve();
      } else {
        reject();
      }
    };
  });
}

let glbURL = null;

async function runExample(name) {
  const worker = new ManifoldWorker();
  await initialized(worker);
  return new Promise((resolve, reject) => {
    worker.onmessageerror = function({data}) {
      reject(data);
    };

    worker.onerror = function(e) {
      reject(e);
    };

    worker.onmessage = async function(e) {
      try {
        URL.revokeObjectURL(glbURL);
        glbURL = e.data.glbURL;
        if (glbURL == null) {
          reject('no glbURL)');
        }
        const docIn = await io.read(glbURL);
        const nodes = docIn.getRoot().listNodes();
        for (const node of nodes) {
          const docMesh = node.getMesh();
          if (!docMesh) {
            continue;
          }
          const {mesh} = readMesh(docMesh);
          const manifold = wasm.Manifold(mesh);
          const prop = manifold.getProperties();
          const genus = manifold.genus();
          manifold.delete();
          // Return properties of first mesh encountered.
          resolve({...prop, genus});
        }
      } catch (e) {
        reject(e);
      }
    };

    worker.postMessage(examples.functionBodies.get(name));
  });
}

suite('Examples', () => {
  test('Intro', async () => {
    const result = await runExample('Intro');
    expect(result.genus).to.equal(5, 'Genus');
    expect(result.volume).to.be.closeTo(203164, 1, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(62046, 1, 'Surface Area');
  });

  test('Tetrahedron Puzzle', async () => {
    const result = await runExample('Tetrahedron Puzzle');
    expect(result.genus).to.equal(0, 'Genus');
    expect(result.volume).to.be.closeTo(7240, 1, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(3235, 1, 'Surface Area');
  });

  test('Rounded Frame', async () => {
    const result = await runExample('Rounded Frame');
    expect(result.genus).to.equal(5, 'Genus');
    expect(result.volume).to.be.closeTo(353706, 10, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(68454, 1, 'Surface Area');
  });

  test('Heart', async () => {
    const result = await runExample('Heart');
    expect(result.genus).to.equal(0, 'Genus');
    expect(result.volume).to.be.closeTo(3.342, 0.001, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(11.51, 0.01, 'Surface Area');
  });

  test('Scallop', async () => {
    const result = await runExample('Scallop');
    expect(result.genus).to.equal(0, 'Genus');
    expect(result.volume).to.be.closeTo(41400, 100, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(7770, 10, 'Surface Area');
  });

  test('Torus Knot', async () => {
    const result = await runExample('Torus Knot');
    expect(result.genus).to.equal(1, 'Genus');
    expect(result.volume).to.be.closeTo(20786, 1, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(11177, 1, 'Surface Area');
  });

  test('Menger Sponge', async () => {
    const result = await runExample('Menger Sponge');
    expect(result.genus).to.equal(729, 'Genus');
    expect(result.volume).to.be.closeTo(203222, 10, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(130475, 10, 'Surface Area');
  });

  test('Stretchy Bracelet', async () => {
    const result = await runExample('Stretchy Bracelet');
    expect(result.genus).to.equal(1, 'Genus');
    expect(result.volume).to.be.closeTo(3992, 1, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(22267, 1, 'Surface Area');
  });

  test('Gyroid Module', async () => {
    const result = await runExample('Gyroid Module');
    expect(result.genus).to.equal(15, 'Genus');
    expect(result.volume).to.be.closeTo(4167, 1, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(5642, 1, 'Surface Area');
  });
});
