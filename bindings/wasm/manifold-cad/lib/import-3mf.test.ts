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

import {strToU8, zipSync} from 'fflate';
import {expect, test} from 'vitest';

import {fromArrayBuffer} from './import-3mf.ts';

const vertices = [
  '<vertex x="0" y="0" z="0"/>', '<vertex x="1" y="0" z="0"/>',
  '<vertex x="0" y="1" z="0"/>', '<vertex x="0" y="0" z="1"/>'
];
const triangles = [
  '<triangle v1="0" v2="2" v3="1"/>', '<triangle v1="0" v2="1" v3="3"/>',
  '<triangle v1="0" v2="3" v3="2"/>', '<triangle v1="1" v2="2" v3="3"/>'
];

function model(vertexRecords = vertices, triangleRecords = triangles) {
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources><object id="1"><mesh>
    <vertices>${vertexRecords.join('')}</vertices>
    <triangles>${triangleRecords.join('')}</triangles>
  </mesh></object></resources>
  <build><item objectid="1"/></build>
</model>`;
  const bytes = zipSync({'3D/3dmodel.model': strToU8(xml)});
  return bytes.buffer as ArrayBuffer;
}

function expectInvalidMesh(buffer: ArrayBuffer, message: string) {
  // Avoid dumping the entire document graph if malformed input is accepted.
  return expect(fromArrayBuffer(buffer).then(() => undefined))
      .rejects.toThrow(message);
}

test(
    'preserves the vertex order and triangle indices of a valid mesh',
    async () => {
      const document = await fromArrayBuffer(model());
      const primitive = document.getRoot().listMeshes()[0].listPrimitives()[0];
      expect(Array.from(primitive.getAttribute('POSITION')!.getArray()!))
          .toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
      expect(Array.from(primitive.getIndices()!.getArray()!)).toEqual([
        0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3
      ]);
    });

test.each(['x', 'y', 'z'])(
    'rejects a missing %s coordinate without shifting the vertex array',
    async missing => {
      const attributes = ['x', 'y', 'z']
                             .filter(axis => axis !== missing)
                             .map(axis => `${axis}="0"`)
                             .join(' ');
      await expectInvalidMesh(
          model([`<vertex ${attributes}/>`, ...vertices]),
          'Invalid 3MF mesh: vertex 0');
    });

test.each([
  'NaN', 'Infinity', '-Infinity', '', ' '
])('rejects the invalid coordinate %j instead of dropping or changing a vertex',
   async value => {
     await expectInvalidMesh(
         model([`<vertex x="${value}" y="0" z="0"/>`, ...vertices]),
         'Invalid 3MF mesh: vertex 0');
   });

test.each(['-1', '0.5', '4', '4294967297', 'NaN', 'Infinity', ''])(
    'rejects triangle index %j before converting it to Uint32', async value => {
      const invalid = `<triangle v1="0" v2="${value}" v3="2"/>`;
      await expectInvalidMesh(
          model(vertices, [invalid, ...triangles]),
          'Invalid 3MF mesh: triangle 0');
    });

test.each(['v1', 'v2', 'v3'])(
    'rejects a missing %s index instead of silently dropping a triangle',
    async missing => {
      const attributes = ['v1', 'v2', 'v3']
                             .filter(name => name !== missing)
                             .map((name, index) => `${name}="${index}"`)
                             .join(' ');
      await expectInvalidMesh(
          model(vertices, [`<triangle ${attributes}/>`, ...triangles]),
          'Invalid 3MF mesh: triangle 0');
    });
