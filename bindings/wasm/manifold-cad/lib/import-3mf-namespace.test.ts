import {strToU8, zipSync} from 'fflate';
import {expect, test} from 'vitest';

import {fromArrayBuffer} from './import-3mf.ts';

test.each([
  '', 'm:', 'core:'
])('imports core elements qualified with %s', async prefix => {
  const ns = 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02';
  const namespace =
      prefix ? `xmlns:${prefix.slice(0, -1)}="${ns}"` : `xmlns="${ns}"`;
  const xml = `<${prefix}model ${namespace} unit="centimeter">
    <${prefix}resources><${prefix}object id="1" name="Part &amp; sample"><${
      prefix}mesh>
      <${prefix}vertices>
        <${prefix}vertex x="0" y="0" z="0"/>
        <${prefix}vertex x="1" y="0" z="0"/>
        <${prefix}vertex x="0" y="1" z="0"/>
      </${prefix}vertices>
      <${prefix}triangles><${prefix}triangle v1="0" v2="1" v3="2"/></${
      prefix}triangles>
    </${prefix}mesh></${prefix}object></${prefix}resources>
    <${prefix}build><${
      prefix}item objectid="1" transform="1 0 0 0 1 0 0 0 1 5 6 7"/></${
      prefix}build>
  </${prefix}model>`;
  const bytes = zipSync({'3D/3dmodel.model': strToU8(xml)});
  const doc = await fromArrayBuffer(bytes.buffer as ArrayBuffer);
  const root = doc.getRoot().listScenes()[0].listChildren()[0];
  expect(root.getScale()).toEqual([0.01, 0.01, 0.01]);
  const child = root.listChildren()[0];
  expect(child.getName()).toBe('Part & sample');
  expect(child.getTranslation()).toEqual([5, 6, 7]);
  const primitive = child.getMesh()!.listPrimitives()[0];
  expect(Array.from(primitive.getAttribute('POSITION')!.getArray()!)).toEqual([
    0, 0, 0, 1, 0, 0, 0, 1, 0
  ]);
  expect(Array.from(primitive.getIndices()!.getArray()!)).toEqual([0, 1, 2]);
});
