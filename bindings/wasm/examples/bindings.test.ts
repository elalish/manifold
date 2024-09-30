import {beforeAll, expect, suite, test} from 'vitest';

import Module, {type ManifoldToplevel} from './built/manifold'

let manifoldModule: ManifoldToplevel;

beforeAll(async () => {
  manifoldModule = await Module();
  manifoldModule.setup();
});

suite('CrossSection Bindings', () => {
  test('ToPolygons return correct shape', () => {
    const polygons = manifoldModule.CrossSection.square().toPolygons();
    expect(polygons).toHaveLength(1);
    expect(polygons[0]).toHaveLength(4);
    expect(polygons[0]).toContainEqual([0, 0]);
    expect(polygons[0]).toContainEqual([0, 1]);
    expect(polygons[0]).toContainEqual([1, 0]);
    expect(polygons[0]).toContainEqual([1, 1]);
  });
});