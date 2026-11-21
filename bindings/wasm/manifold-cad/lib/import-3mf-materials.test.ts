import {strToU8, zipSync} from 'fflate';
import {expect, test} from 'vitest';

import {fromArrayBuffer} from './import-3mf.ts';

async function importMesh(objectProperties: string, triangleProperties = '') {
  const xml =
      `<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
    <basematerials id="10"><base name="Red" displaycolor="#FF0000"/><base name="Gray" displaycolor="#80808080"/></basematerials>
    <basematerials id="20"><base name="Blue" displaycolor="#0000FF"/></basematerials>
    <object id="1" ${objectProperties}><mesh>
      <vertices><vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/><vertex x="0" y="0" z="1"/></vertices>
      <triangles><triangle v1="0" v2="2" v3="1" ${triangleProperties}/>
        <triangle v1="0" v2="1" v3="3"/><triangle v1="1" v2="2" v3="3"/><triangle v1="2" v2="0" v3="3"/>
      </triangles></mesh></object>
  </resources><build><item objectid="1"/></build></model>`;
  const archive = zipSync({'3D/3dmodel.model': strToU8(xml)});
  const doc = await fromArrayBuffer(archive.buffer as ArrayBuffer);
  return doc.getRoot().listMeshes()[0];
}

test('imports object base material names, linear color and alpha', async () => {
  const mesh = await importMesh('pid="10" pindex="1"');
  expect(mesh.listPrimitives()).toHaveLength(1);
  const primitive = mesh.listPrimitives()[0];
  const material = primitive.getMaterial()!;
  expect(material.getName()).toBe('Gray');
  const color = material.getBaseColorFactor();
  for (let i = 0; i < 3; ++i) expect(color[i]).toBeCloseTo(0.2158605, 6);
  expect(color[3]).toBeCloseTo(128 / 255);
  expect(material.getAlphaMode()).toBe('BLEND');
  expect(primitive.getIndices()!.getCount()).toBe(12);
});

test(
    'groups triangle overrides without changing vertex positions or winding',
    async () => {
      const mesh = await importMesh('pid="10" pindex="1"', 'pid="20" p1="0"');
      const primitives = mesh.listPrimitives();
      expect(primitives).toHaveLength(2);
      const blue = primitives.find(p => p.getMaterial()!.getName() === 'Blue')!;
      const gray = primitives.find(p => p.getMaterial()!.getName() === 'Gray')!;
      expect(Array.from(blue.getIndices()!.getArray()!)).toEqual([0, 2, 1]);
      expect(Array.from(gray.getIndices()!.getArray()!)).toEqual([
        0, 1, 3, 1, 2, 3, 2, 0, 3
      ]);
      expect(blue.getAttribute('POSITION')).toBe(gray.getAttribute('POSITION'));
      expect(blue.getMaterial()!.getBaseColorFactor()).toEqual([0, 0, 1, 1]);
    });

test(
    'uses the first vertex material when a triangle requests a gradient',
    async () => {
      const mesh =
          await importMesh('pid="10" pindex="1"', 'p1="0" p2="1" p3="1"');
      const red =
          mesh.listPrimitives().find(p => p.getMaterial()!.getName() === 'Red');
      expect(red).toBeDefined();
      expect(Array.from(red!.getIndices()!.getArray()!)).toEqual([0, 2, 1]);
    });

test('keeps the existing fallback when no material is assigned', async () => {
  const mesh = await importMesh('');
  expect(mesh.listPrimitives()).toHaveLength(1);
  expect(mesh.listPrimitives()[0].getMaterial()!.getBaseColorFactor()).toEqual([
    1, 1, 1, 1
  ]);
});
