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

import {Document, type Node} from '@gltf-transform/core';
import {strToU8, zipSync} from 'fflate';
import {beforeAll, expect, test} from 'vitest';

import {fromArrayBuffer} from './import-3mf.ts';
import {cleanup, importManifold} from './import-model.ts';
import {euler2quat} from './math.ts';
import {getManifoldModule} from './wasm.ts';

const vertices = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1],
  [0, 1, 1]
];
const triangles = [
  [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6],
  [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]
];
function cubeObject(scale = 1): string {
  return `<object id="1" name="cube"><mesh><vertices>${
      vertices
          .map(
              ([x, y, z]) => `<vertex x="${x * scale}" y="${y * scale}" z="${
                  z * scale}"/>`)
          .join('')}</vertices><triangles>${
      triangles
          .map(([v1, v2, v3]) => `<triangle v1="${v1}" v2="${v2}" v3="${v3}"/>`)
          .join('')}</triangles></mesh></object>`;
}
const cube = cubeObject();
const shear = '1 0 0 1 1 0 0 0 1 0 0 0';
const reflectedShear = '-1 0 0 -1 1 0 0 0 1 3 0 0';
const translatedRotation = '0 1 0 -1 0 0 0 0 1 5 0 0';

function archive(
    resources: string, build: string, unit = 'millimeter'): ArrayBuffer {
  const xml = `<model unit="${
      unit}" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>${
      resources}</resources><build>${build}</build></model>`;
  return zipSync({'3D/3dmodel.model': strToU8(xml)}).buffer as ArrayBuffer;
}

function meshNodes(doc: Document): Node[] {
  return doc.getRoot().listNodes().filter(node => node.getMesh());
}

function worldPositions(node: Node): number[][] {
  const positions =
      node.getMesh()!.listPrimitives()[0].getAttribute('POSITION')!;
  const matrix = node.getWorldMatrix();
  return Array.from({length: positions.getCount()}, (_, i) => {
    const [x, y, z] = positions.getElement(i, [0, 0, 0]);
    return [0, 1, 2].map(
        row => matrix[row] * x + matrix[row + 4] * y + matrix[row + 8] * z +
            matrix[row + 12]);
  });
}

function sortedPoints(points: number[][]): number[][] {
  return points
      .map(point => point.map(value => {
        const rounded = Math.round(value * 1e10) / 1e10;
        return rounded === 0 ? 0 : rounded;
      }))
      .sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
}

function expectGeometry(node: Node, points: number[][], meters = 1e-3) {
  // Model coordinates use +Z up; the imported document uses +Y up and meters.
  const expected =
      points.map(([x, y, z]) => [x * meters, z * meters, -y * meters]);
  expect(sortedPoints(worldPositions(node))).toEqual(sortedPoints(expected));
}

function signedVolume(node: Node): number {
  const points = worldPositions(node);
  const indices = node.getMesh()!.listPrimitives()[0].getIndices()!.getArray()!;
  let volume = 0;
  for (let i = 0; i < indices.length; i += 3) {
    const [a, b, c] =
        [points[indices[i]], points[indices[i + 1]], points[indices[i + 2]]];
    volume += a[0] * (b[1] * c[2] - b[2] * c[1]) +
        a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  }
  return volume / 6;
}

test.each([{unit: 'millimeter', meters: 1e-3}, {unit: 'inch', meters: 0.0254}])(
    'preserves a sheared build item in $unit units', async ({unit, meters}) => {
      const doc = await fromArrayBuffer(
          archive(cube, `<item objectid="1" transform="${shear}"/>`, unit));
      const [node] = meshNodes(doc);
      expectGeometry(node, vertices.map(([x, y, z]) => [x + y, y, z]), meters);
      expect(signedVolume(node) / meters ** 3).toBeCloseTo(1, 10);
    });

test('keeps reflected shear winding outward', async () => {
  const doc = await fromArrayBuffer(
      archive(cube, `<item objectid="1" transform="${reflectedShear}"/>`));
  const [node] = meshNodes(doc);
  expectGeometry(node, vertices.map(([x, y, z]) => [3 - x - y, y, z]));
  expect(signedVolume(node) / 1e-9).toBeCloseTo(1, 10);
});

test(
    'composes a sheared parent with child rotation and translation',
    async () => {
      const resources = cube +
          `<object id="2"><components><component objectid="1" transform="${
                            translatedRotation}"/></components></object>`;
      const doc = await fromArrayBuffer(archive(
          resources, `<item objectid="2" transform="1 0 0 1 1 0 0 0 1 0 7 0"/>`,
          'inch'));
      const [node] = meshNodes(doc);
      expectGeometry(
          node, vertices.map(([x, y, z]) => [x - y + 5, x + 7, z]), 0.0254);
      expect(signedVolume(node) / 0.0254 ** 3).toBeCloseTo(1, 10);
    });

test(
    'bakes a sheared component without applying its retained TRS parent twice',
    async () => {
      const resources = cube +
          `<object id="2"><components><component objectid="1" transform="${
                            shear}"/></components></object>`;
      const doc = await fromArrayBuffer(archive(
          resources, `<item objectid="2" transform="${translatedRotation}"/>`));
      const [node] = meshNodes(doc);
      expectGeometry(node, vertices.map(([x, y, z]) => [5 - y, x + y, z]));
    });

test.each([false, true])(
    'isolates the sheared instance (shear first: %s)', async shearFirst => {
      const items =
          [`<item objectid="1"/>`, `<item objectid="1" transform="${shear}"/>`];
      if (shearFirst) items.reverse();
      const doc = await fromArrayBuffer(archive(cube, items.join('')));
      const nodes = meshNodes(doc);
      const plain = nodes[shearFirst ? 1 : 0];
      const sheared = nodes[shearFirst ? 0 : 1];
      expectGeometry(plain, vertices);
      expectGeometry(sheared, vertices.map(([x, y, z]) => [x + y, y, z]));
      const original = plain.getMesh()!;
      const transformed = sheared.getMesh()!;
      expect(transformed).not.toBe(original);
      expect(original.listPrimitives()[0].getAttribute('POSITION')!.getArray())
          .toEqual(new Float32Array(vertices.flat()));
      expect(transformed.listPrimitives()[0].getMaterial())
          .toBe(original.listPrimitives()[0].getMaterial());
    });

test('preserves shared geometry for ordinary TRS instances', async () => {
  const transforms = [
    '1 0 0 0 1 0 0 0 1 0 0 0', '0 2 0 -3 0 0 0 0 4 3 4 5',
    '-2 0 0 0 3 0 0 0 4 3 4 5'
  ];
  const doc = await fromArrayBuffer(archive(
      cube,
      transforms
          .map(transform => `<item objectid="1" transform="${transform}"/>`)
          .join('')));
  const nodes = meshNodes(doc);
  expect(nodes).toHaveLength(3);
  expect(nodes[1].getMesh()).toBe(nodes[0].getMesh());
  expect(nodes[2].getMesh()).toBe(nodes[0].getMesh());
  expect(doc.getRoot().listMeshes()).toHaveLength(1);
  expect(
      nodes[0].getMesh()!.listPrimitives()[0].getAttribute(
                                                 'POSITION')!.getArray())
      .toEqual(new Float32Array(vertices.flat()));
  expect(nodes[0].getMesh()!.listPrimitives()[0].getIndices()!.getArray())
      .toEqual(new Uint32Array(triangles.flat()));
  expectGeometry(nodes[0], vertices);
  expectGeometry(
      nodes[1], vertices.map(([x, y, z]) => [3 - 3 * y, 4 + 2 * x, 5 + 4 * z]));
  expectGeometry(
      nodes[2], vertices.map(([x, y, z]) => [3 - 2 * x, 4 + 3 * y, 5 + 4 * z]));
});

test('keeps six-decimal rotation instances on shared geometry', async () => {
  const rotations: [number, number, number][] =
      [[17, 29, 43], [13, 57, -31], [-47, 21, 63]];
  const transforms = rotations.map(angles => {
    const doc = new Document();
    return doc.createNode()
        .setRotation(euler2quat(angles))
        .getMatrix()
        .map(value => Number(value.toFixed(6)));
  });
  const build = '<item objectid="1"/>' +
      transforms
          .map(matrix => {
            const transform =
                matrix.filter((_, index) => index % 4 !== 3).join(' ');
            return `<item objectid="1" transform="${transform}"/>`;
          })
          .join('');
  const doc = await fromArrayBuffer(archive(cube, build));
  const nodes = meshNodes(doc);
  expect(nodes).toHaveLength(4);
  expect(doc.getRoot().listMeshes()).toHaveLength(1);
  for (const [index, matrix] of transforms.entries()) {
    const node = nodes[index + 1];
    expect(node.getMesh()).toBe(nodes[0].getMesh());
    for (const [vertexIndex, [x, y, z]] of vertices.entries()) {
      const [a, b, c] = [0, 1, 2].map(
          row => matrix[row] * x + matrix[row + 4] * y + matrix[row + 8] * z);
      const expected = [a * 1e-3, c * 1e-3, -b * 1e-3];
      for (let axis = 0; axis < 3; ++axis) {
        expect(
            Math.abs(worldPositions(node)[vertexIndex][axis] - expected[axis]))
            .toBeLessThan(3e-9);
      }
    }
  }
});

test('still bakes shear larger than the print tolerance', async () => {
  const doc = await fromArrayBuffer(archive(
      cube,
      '<item objectid="1"/><item objectid="1" transform="1 0 0 0.0001 1 0 0 0 1 0 0 0"/>'));
  const [plain, sheared] = meshNodes(doc);
  expect(sheared.getMesh()).not.toBe(plain.getMesh());
  expectGeometry(sheared, vertices.map(([x, y, z]) => [x + 0.0001 * y, y, z]));
});

test(
    'preserves a small affine scale applied to large source coordinates',
    async () => {
      const scale = '0.000000000000001';
      const scaledCube = cubeObject(1e15);
      const plain = `${scale} 0 0 0 ${scale} 0 0 0 ${scale} 0 0 0`;
      const sheared = `${scale} 0 0 ${scale} ${scale} 0 0 0 ${scale} 0 0 0`;
      const doc = await fromArrayBuffer(archive(
          scaledCube,
          `<item objectid="1" transform="${
              plain}"/><item objectid="1" transform="${sheared}"/>`));
      const [plainNode, shearNode] = meshNodes(doc);
      expectGeometry(plainNode, vertices);
      expectGeometry(shearNode, vertices.map(([x, y, z]) => [x + y, y, z]));
      expect(signedVolume(shearNode) / 1e-9).toBeCloseTo(1, 6);
      expect(plainNode.getMesh()).toBe(doc.getRoot().listMeshes()[0]);
      expect(
          plainNode.getMesh()!.listPrimitives()[0].getAttribute(
                                                      'POSITION')!.getArray())
          .toEqual(
              new Float32Array(vertices.flat().map(value => value * 1e15)));
    });

beforeAll(async () => await getManifoldModule());

test.each([shear, reflectedShear])(
    'imports affine geometry as a valid unit-volume Manifold (%s)',
    async transform => {
      const manifold = await importManifold(
          archive(cube, `<item objectid="1" transform="${transform}"/>`),
          {mimetype: 'model/3mf'});
      try {
        expect(manifold.status()).toBe('NoError');
        expect(manifold.volume()).toBeCloseTo(1, 6);
        expect([...manifold.getMesh().vertProperties].every(Number.isFinite))
            .toBe(true);
      } finally {
        manifold.delete();
        cleanup();
      }
    });
