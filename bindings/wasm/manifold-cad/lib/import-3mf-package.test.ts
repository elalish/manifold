import {strToU8, zipSync} from 'fflate';
import {expect, test} from 'vitest';

import {fromArrayBuffer} from './import-3mf.ts';

const model =
    `<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources><object id="1" name="selected"><mesh>
  <vertices><vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/></vertices>
  <triangles><triangle v1="0" v2="1" v3="2"/></triangles>
  </mesh></object></resources><build><item objectid="1"/></build></model>`;
const relationship = (target: string, mode = 'Internal') => strToU8(
    `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="model" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" Target="${
        target}" TargetMode="${mode}"/>
    </Relationships>`);

test.each(
    ['3D/assembly.model', '/3D/assembly.model', '3D/parts/../assembly.model'])(
    'uses the package StartPart target %s', async target => {
      const archive = zipSync({
        '_rels/.rels': relationship(target),
        '3D/assembly.model': strToU8(model),
        '3D/3dmodel.model': strToU8(model.replace('selected', 'unused')),
      });
      const doc = await fromArrayBuffer(archive.buffer as ArrayBuffer);
      expect(doc.getRoot().listNodes().map(node => node.getName()))
          .toContain('selected');
      expect(doc.getRoot().listNodes().map(node => node.getName()))
          .not.toContain('unused');
    });

test(
    'reports a missing referenced model instead of using an unrelated default',
    async () => {
      const archive = zipSync({
        '_rels/.rels': relationship('/3D/missing.model'),
        '3D/3dmodel.model': strToU8(model),
      });
      const parse = async () => {
        await fromArrayBuffer(archive.buffer as ArrayBuffer);
      };
      await expect(parse()).rejects.toThrow('3D/missing.model');
    });

test('does not follow an external model relationship', async () => {
  const archive = zipSync({
    '_rels/.rels': relationship('https://example.test/model', 'External'),
    '3D/3dmodel.model': strToU8(model),
  });
  const parse = async () => {
    await fromArrayBuffer(archive.buffer as ArrayBuffer);
  };
  await expect(parse()).rejects.toThrow(/external/i);
});

test(
    'keeps compatibility with the conventional path when relationships are absent',
    async () => {
      const archive = zipSync({'3D/3dmodel.model': strToU8(model)});
      const doc = await fromArrayBuffer(archive.buffer as ArrayBuffer);
      expect(doc.getRoot().listMeshes()).toHaveLength(1);
    });
