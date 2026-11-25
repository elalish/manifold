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

import {Accessor, Document} from '@gltf-transform/core';
import type {Mesh as ManifoldMesh} from 'manifold-3d';
import {expect, test} from 'vitest';

import {readMesh, writeMesh} from './gltf-io.ts';
import type {Attribute} from './gltf-io.ts';
import {EXTManifold} from './manifold-gltf.ts';

test('readMesh places normals in the standard slot and marks their run', () => {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const mesh = doc.createMesh();
  const primitive = doc.createPrimitive();
  const position = doc.createAccessor('positions')
                       .setBuffer(buffer)
                       .setType(Accessor.Type.VEC3)
                       .setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
  const color = doc.createAccessor('colors')
                    .setBuffer(buffer)
                    .setType(Accessor.Type.VEC3)
                    .setArray(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
  const normal = doc.createAccessor('normals')
                     .setBuffer(buffer)
                     .setType(Accessor.Type.VEC3)
                     .setArray(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]));
  const indices = doc.createAccessor('indices')
                      .setBuffer(buffer)
                      .setType(Accessor.Type.SCALAR)
                      .setArray(new Uint16Array([0, 1, 2]));
  primitive.setAttribute('POSITION', position);
  primitive.setAttribute('COLOR_0', color);
  primitive.setAttribute('NORMAL', normal);
  primitive.setIndices(indices);
  mesh.addPrimitive(primitive);

  const result = readMesh(mesh, ['POSITION', 'COLOR_0', 'NORMAL'])!;

  expect(result.mesh.numProp).toBe(9);
  expect(Array.from(result.mesh.runFlags!)).toEqual([0x02]);
  expect(result.runProperties[0].attributes).toEqual([
    'POSITION', 'NORMAL', 'COLOR_0'
  ]);
  expect(Array.from(result.mesh.vertProperties!.slice(3, 6))).toEqual([
    0, 0, 1
  ]);
});

test('writeMesh reads NORMAL after POSITION despite material order', () => {
  const doc = new Document();
  const material = doc.createMaterial();
  const color = [0.1, 0.2, 0.3];
  const manifoldMesh = {
    numProp: 9,
    numVert: 3,
    vertProperties: new Float32Array([
      0, 0, 0,        0, 0, 2, ...color, 1, 0, 0,       0,
      0, 2, ...color, 0, 1, 0, 0,        0, 2, ...color
    ]),
    triVerts: new Uint32Array([0, 1, 2]),
    runIndex: new Uint32Array([0, 3]),
    runOriginalID: new Uint32Array([12])
  } as unknown as ManifoldMesh;
  const properties = new Map([[
    12, {material, attributes: ['POSITION', 'COLOR_0', 'NORMAL'] as Attribute[]}
  ]]);

  const primitive =
      writeMesh(doc, manifoldMesh, properties, false).listPrimitives()[0];
  expect(Array.from(primitive.getAttribute('NORMAL')!.getArray()!)).toEqual([
    0, 0, 1, 0, 0, 1, 0, 0, 1
  ]);
  expect(Array.from(primitive.getAttribute('COLOR_0')!.getArray()!))
      .toEqual(Array.from(new Float32Array([...color, ...color, ...color])));
});

function mixedPrimitiveMesh(withExtension: boolean) {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const mesh = doc.createMesh();
  const position = doc.createAccessor('positions')
                       .setBuffer(buffer)
                       .setType(Accessor.Type.VEC3)
                       .setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
  const color = doc.createAccessor('colors')
                    .setBuffer(buffer)
                    .setType(Accessor.Type.VEC3)
                    .setArray(new Float32Array(
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]));
  const normal = doc.createAccessor('normals')
                     .setBuffer(buffer)
                     .setType(Accessor.Type.VEC3)
                     .setArray(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]));
  const indices = doc.createAccessor('indices')
                      .setBuffer(buffer)
                      .setType(Accessor.Type.SCALAR)
                      .setArray(new Uint16Array([0, 1, 2]));

  const withNormals = doc.createPrimitive();
  withNormals.setAttribute('POSITION', position);
  withNormals.setAttribute('NORMAL', normal);
  withNormals.setAttribute('COLOR_0', color);
  withNormals.setIndices(indices);
  mesh.addPrimitive(withNormals);

  const withoutNormals = doc.createPrimitive();
  withoutNormals.setAttribute('POSITION', position);
  withoutNormals.setAttribute('COLOR_0', color);
  withoutNormals.setIndices(indices);
  mesh.addPrimitive(withoutNormals);

  if (withExtension) {
    const extension = doc.createExtension(EXTManifold);
    mesh.setExtension('EXT_mesh_manifold', extension.createManifoldPrimitive());
  }
  return mesh;
}

test.each([false, true])(
    'readMesh zero-fills mixed normal layouts (%s extension)',
    (withExtension) => {
      const requested: Attribute[] = ['POSITION', 'COLOR_0', 'TANGENT'];
      const result = readMesh(mixedPrimitiveMesh(withExtension), requested)!;

      expect(requested).toEqual(['POSITION', 'COLOR_0', 'TANGENT']);
      expect(result.mesh.numProp).toBe(13);
      expect(Array.from(result.mesh.runFlags!)).toEqual([0x02, 0x00]);
      expect(result.runProperties.map((properties) => properties.attributes))
          .toEqual([
            ['POSITION', 'NORMAL', 'COLOR_0', 'SKIP_4'],
            ['POSITION', 'SKIP_3', 'COLOR_0', 'SKIP_4']
          ]);
      expect(Array.from(result.mesh.vertProperties!).every(Number.isFinite))
          .toBe(true);
    });

test('writeMesh preserves mixed normal runs and normalizes normals', () => {
  const doc = new Document();
  const material = doc.createMaterial();
  const firstColor = [0.1, 0.2, 0.3];
  const secondColor = [0.4, 0.5, 0.6];
  const vertProperties = new Float32Array([
    0, 0, 0, 0, 0, 2, ...firstColor,  1, 0, 0, 0, 0, 2, ...firstColor,
    0, 1, 0, 0, 0, 2, ...firstColor,  2, 0, 0, 0, 0, 0, ...secondColor,
    3, 0, 0, 0, 0, 0, ...secondColor, 2, 1, 0, 0, 0, 0, ...secondColor
  ]);
  const manifoldMesh = {
    numProp: 9,
    numVert: 6,
    vertProperties,
    triVerts: new Uint32Array([0, 1, 2, 3, 4, 5]),
    runIndex: new Uint32Array([0, 3, 6]),
    runOriginalID: new Uint32Array([17, 18])
  } as unknown as ManifoldMesh;
  const properties = new Map([
    [
      17,
      {material, attributes: ['POSITION', 'NORMAL', 'COLOR_0'] as Attribute[]}
    ],
    [
      18,
      {material, attributes: ['POSITION', 'SKIP_3', 'COLOR_0'] as Attribute[]}
    ]
  ]);

  const output = writeMesh(doc, manifoldMesh, properties, false);
  const [withNormals, withoutNormals] = output.listPrimitives();
  expect(
      Array.from(withNormals.getAttribute('NORMAL')!.getArray()!).slice(0, 9))
      .toEqual([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  expect(withoutNormals.getAttribute('NORMAL')).toBeNull();
  expect(
      Array.from(withoutNormals.getAttribute('COLOR_0')!.getArray()!).slice(9))
      .toEqual(Array.from(
          new Float32Array([...secondColor, ...secondColor, ...secondColor])));

  const roundTrip = readMesh(output)!;
  expect(Array.from(roundTrip.mesh.vertProperties!).every(Number.isFinite))
      .toBe(true);
  expect(Array.from(roundTrip.mesh.runFlags!)).toEqual([0x02, 0x00]);
  expect(roundTrip.runProperties.map((run) => run.attributes)).toEqual([
    ['POSITION', 'NORMAL', 'COLOR_0'], ['POSITION', 'SKIP_3', 'COLOR_0']
  ]);
});

test.each([1, 2, 3, 4])(
    'write/read roundtrip preserves COLOR_0 and TEXCOORD_0 after SKIP_%i',
    (skipComponents) => {
      const doc = new Document();
      const material = doc.createMaterial();
      const skip = `SKIP_${skipComponents}` as Attribute;
      const colors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];
      const texcoords = [[0, 0], [1, 0], [0, 1]];
      const values: number[] = [];
      for (let vert = 0; vert < 3; ++vert) {
        values.push(vert === 1 ? 1 : 0, vert === 2 ? 1 : 0, 0);
        values.push(...new Array<number>(skipComponents).fill(7 + vert));
        values.push(...colors[vert], ...texcoords[vert]);
      }
      const manifoldMesh = {
        numProp: 8 + skipComponents,
        numVert: 3,
        vertProperties: new Float32Array(values),
        triVerts: new Uint32Array([0, 1, 2]),
        runIndex: new Uint32Array([0, 3]),
        runOriginalID: new Uint32Array([29])
      } as unknown as ManifoldMesh;
      const properties = new Map([[
        29, {
          material,
          attributes: ['POSITION', skip, 'COLOR_0', 'TEXCOORD_0'] as Attribute[]
        }
      ]]);

      const output = writeMesh(doc, manifoldMesh, properties, false);
      const primitive = output.listPrimitives()[0];
      expect(Array.from(primitive.getAttribute('COLOR_0')!.getArray()!))
          .toEqual(Array.from(new Float32Array(colors.flat())));
      expect(Array.from(primitive.getAttribute('TEXCOORD_0')!.getArray()!))
          .toEqual(Array.from(new Float32Array(texcoords.flat())));

      const roundTrip = readMesh(output)!;
      expect(Array.from(roundTrip.mesh.vertProperties!).every(Number.isFinite))
          .toBe(true);
      expect(roundTrip.mesh.numProp).toBe(8);
      for (let vert = 0; vert < 3; ++vert) {
        const offset = vert * roundTrip.mesh.numProp;
        expect(Array.from(roundTrip.mesh.vertProperties!.slice(
                   offset + 3, offset + 8)))
            .toEqual(Array.from(
                new Float32Array([...texcoords[vert], ...colors[vert]])));
      }
    });
