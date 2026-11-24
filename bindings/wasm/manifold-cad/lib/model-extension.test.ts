import {expect, test} from 'vitest';

import {supports} from './import-model.ts';
import {findExtension} from './util.ts';

const formats = [{extension: 'glb'}, {extension: '3mf'}];

test.each([
  'glb',
  '.glb',
  'GLB',
  '.GLB',
  'part.GLB',
  'https://example.test/part.glb?signature=abc',
  'https://example.test/part.glb#preview',
  'https://example.test/part%2Eglb?download=1',
  'models/part.glb?v=2#preview',
  'C:\\models\\part.GLB',
])('recognizes the model path in %s', value => {
  expect(findExtension(value, formats)).toBe(formats[0]);
  expect(supports(value)).toBe(true);
});

test('does not use an extension appearing only in a query or host', () => {
  expect(
      findExtension('https://example.test/part.3mf?return=part.glb', formats))
      .toBe(formats[1]);
  expect(findExtension('https://host.glb/model', formats)).toBeFalsy();
  expect(findExtension('models.glb/unknown', formats)).toBeFalsy();
});

test('keeps format objects and unknown extensions unchanged', () => {
  expect(findExtension('part.unknown', formats)).toBeFalsy();
  expect(findExtension('part.%ZZ', formats)).toBeFalsy();
  expect(formats).toEqual([{extension: 'glb'}, {extension: '3mf'}]);
});


test.each([
  'shape#draft.glb',
  'shape?rev.glb',
  'models/shape#draft.glb',
  'models/shape?rev.glb',
  'C:\\models\\shape#draft.glb',
])('preserves a supported local filename suffix in %s', value => {
  expect(findExtension(value, formats)).toBe(formats[0]);
  expect(supports(value)).toBe(true);
});

test('distinguishes explicit URL paths from literal filename suffixes', () => {
  const formats = [{extension: 'obj'}, {extension: 'stl'}];
  expect(findExtension('shape#draft.stl', formats)).toBe(formats[1]);
  expect(findExtension('shape?rev.stl', formats)).toBe(formats[1]);
  expect(
      findExtension('https://example.com/shape.obj?download=mesh.stl', formats))
      .toBe(formats[0]);
  expect(findExtension('//example.com/shape.obj?download=mesh.stl', formats))
      .toBe(formats[0]);
  expect(findExtension('models/shape.obj?signature=abc#preview', formats))
      .toBe(formats[0]);
  // An unqualified relative string can be a legal filename containing '?'.
  // Preserve the old recognized full suffix when there is no URL authority.
  expect(findExtension('shape.obj?download=mesh.stl', formats))
      .toBe(formats[1]);
});
