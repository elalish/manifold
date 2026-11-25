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

import {Document, WebIO} from '@gltf-transform/core';
import type {Manifold, ManifoldToplevel, Mesh, Vec3} from 'manifold-3d';
import {afterEach, beforeAll, expect, test, vi} from 'vitest';

import {morphEnd, morphStart, setMorphEnd, setMorphStart} from './animation.ts';
import {fromArrayBuffer, type Properties, toArrayBuffer, writeMesh} from './gltf-io.ts';
import {cleanup as cleanupNodes} from './gltf-node.ts';
import {setMaterial} from './material.ts';
import {cleanup, manifoldToGLTFDoc} from './scene-builder.ts';
import {getManifoldModule} from './wasm.ts';

let wasm: ManifoldToplevel;
const nativeObjects: Manifold[] = [];
beforeAll(async () => {
  wasm = await getManifoldModule();
});
afterEach(() => {
  cleanup();
  cleanupNodes();
  for (const object of nativeObjects.splice(0)) object.delete();
});

const positions = [0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4];

function createMesh(numProp = 3, runs = 1): Mesh {
  const properties: number[] = [];
  for (let i = 0; i < positions.length / 3; ++i) {
    properties.push(...positions.slice(3 * i, 3 * i + 3));
    properties.push(...[0, 0, 1, 0.25, 0.75].slice(0, numProp - 3));
  }
  return new wasm.Mesh({
    numProp,
    vertProperties: new Float32Array(properties),
    triVerts: new Uint32Array([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3]),
    runIndex: new Uint32Array(
        runs === 1     ? [0, 12] :
            runs === 2 ? [0, 6, 12] :
                         [0, 3, 6, 12]),
    runOriginalID: new Uint32Array(Array.from({length: runs}, (_, i) => i + 1))
  });
}

function start(v: Vec3) {
  v[0] += 1;
  v[1] -= 2;
  v[2] = 2 * v[2] + 3;
}

function end(v: Vec3) {
  v[0] *= 2;
  v[1] += 4;
  v[2] -= 1;
}

function transformed(input: number[], transform?: (v: Vec3) => void) {
  const result: number[] = [];
  for (let i = 0; i < input.length; i += 3) {
    const vertex = input.slice(i, i + 3) as Vec3;
    transform?.(vertex);
    result.push(...vertex);
  }
  return result;
}

test.each(
    [3, 6, 8])('morph start preserves a %i-channel vertex record', numProp => {
  const mesh = createMesh(numProp);
  const before = Array.from(mesh.vertProperties);
  const callback = vi.fn(start);
  const original = morphStart(mesh, {start: callback});

  expect(original).toEqual(positions);
  expect(callback).toHaveBeenCalledTimes(mesh.numVert);
  const expected = transformed(positions, start);
  for (let i = 0; i < mesh.numVert; ++i) {
    expect(Array.from(mesh.vertProperties.slice(i * numProp, i * numProp + 3)))
        .toEqual(expected.slice(3 * i, 3 * i + 3));
    expect(Array.from(
               mesh.vertProperties.slice(i * numProp + 3, (i + 1) * numProp)))
        .toEqual(before.slice(i * numProp + 3, (i + 1) * numProp));
  }
  expect(mesh.vertProperties.every(Number.isFinite)).toBe(true);
});

for (const runs of [1, 2, 3]) {
  test.each([
    'start', 'end', 'both'
  ])(`exports %s morph geometry across ${runs} triangle runs`, async mode => {
    const mesh = createMesh(3, runs);
    const startCallback = vi.fn(start);
    const endCallback = vi.fn(end);
    const morph = {
      start: mode === 'end' ? undefined : startCallback,
      end: mode === 'start' ? undefined : endCallback
    };
    const original = morphStart(mesh, morph);
    const doc = new Document();
    const materials = new Map<number, Properties>();
    for (const id of mesh.runOriginalID) {
      materials.set(
          id, {material: doc.createMaterial(), attributes: ['POSITION']});
    }
    const output = writeMesh(doc, mesh, materials, false);
    morphEnd(doc, mesh, output, original, morph);
    doc.createScene().addChild(doc.createNode().setMesh(output));

    expect(startCallback).toHaveBeenCalledTimes(morph.start ? mesh.numVert : 0);
    expect(endCallback).toHaveBeenCalledTimes(morph.end ? mesh.numVert : 0);
    const startPositions = transformed(positions, morph.start && start);
    const endPositions = transformed(positions, morph.end && end);
    const deltas = endPositions.map((v, i) => v - startPositions[i]);
    const primitives = output.listPrimitives();
    expect(primitives).toHaveLength(runs);
    const target = primitives[0].listTargets()[0].getAttribute('POSITION');
    for (const primitive of primitives) {
      expect(primitive.listTargets()[0].getAttribute('POSITION') === target)
          .toBe(true);
    }

    const io = new WebIO();
    const roundtrip = await io.readBinary(await io.writeBinary(doc));
    expect(roundtrip.getRoot().listMeshes()[0].getWeights()).toEqual([0]);
    for (const primitive of roundtrip.getRoot()
             .listMeshes()[0]
             .listPrimitives()) {
      expect(Array.from(primitive.getAttribute('POSITION')!.getArray()!))
          .toEqual(startPositions);
      const values =
          primitive.listTargets()[0].getAttribute('POSITION')!.getArray()!;
      expect(Array.from(values)).toEqual(deltas);
      expect(Array.from(values).every(Number.isFinite)).toBe(true);
    }
  });
}

test('does not add a morph target without a morph', () => {
  const mesh = createMesh();
  const before = mesh.vertProperties.slice();
  const doc = new Document();
  const output = writeMesh(
      doc, mesh,
      new Map<number, Properties>(
          [[1, {material: doc.createMaterial(), attributes: ['POSITION']}]]),
      false);
  const original = morphStart(mesh);
  morphEnd(doc, mesh, output, original);
  expect(original).toEqual([]);
  expect(mesh.vertProperties).toEqual(before);
  expect(output.getWeights()).toEqual([]);
  expect(output.listPrimitives()[0].listTargets()).toEqual([]);
});

test(
    'exports a finite two-material colored Manifold morph to GLB', async () => {
      const keep = (object: Manifold) => {
        nativeObjects.push(object);
        return object;
      };
      const cube = keep(wasm.Manifold.cube());
      const colored = keep(cube.setProperties(3, (properties, position) => {
        for (let i = 0; i < 3; ++i) properties[i] = position[i];
      }));
      const first = keep(setMaterial(colored, {attributes: ['COLOR_0']}));
      const translated = keep(colored.translate([2, 0, 0]));
      const second = keep(setMaterial(translated, {attributes: ['COLOR_0']}));
      const model = keep(first.add(second));
      const input = model.getMesh();
      setMorphStart(model, v => {
        v[2] += 2;
      });
      setMorphEnd(model, v => {
        v[0] += 4;
      });

      const doc = await manifoldToGLTFDoc(model);
      const roundtrip = await fromArrayBuffer(await toArrayBuffer(doc));
      expect(roundtrip.getRoot().listAnimations()).toHaveLength(1);
      const mesh = roundtrip.getRoot().listMeshes()[0];
      expect(mesh.listPrimitives()).toHaveLength(2);
      for (const primitive of mesh.listPrimitives()) {
        const start = primitive.getAttribute('POSITION')!.getArray()!;
        const target =
            primitive.listTargets()[0].getAttribute('POSITION')!.getArray()!;
        expect(start).toHaveLength(input.numVert * 3);
        expect(Array.from(start).every(Number.isFinite)).toBe(true);
        expect(Array.from(target).every(Number.isFinite)).toBe(true);
        for (let i = 0; i < input.numVert; ++i) {
          expect(Array.from(start.slice(3 * i, 3 * i + 3))).toEqual([
            input.vertProperties[i * input.numProp],
            input.vertProperties[i * input.numProp + 1],
            input.vertProperties[i * input.numProp + 2] + 2
          ]);
          expect(Array.from(target.slice(3 * i, 3 * i + 3))).toEqual([
            4, 0, -2
          ]);
        }
        expect(primitive.getAttribute('COLOR_0')!.getCount())
            .toBe(input.numVert);
      }
      expect(model.volume()).toBe(2);
      expect(model.getMesh().vertProperties).toEqual(input.vertProperties);
    });
