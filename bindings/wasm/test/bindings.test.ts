import {beforeAll, expect, suite, test} from 'vitest';

import Module, {type ManifoldToplevel} from '../manifold'

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

  test('project creates a valid polygon', () => {
    const cs = manifoldModule.Manifold.sphere(1).project();
    expect(cs.numContour()).toEqual(1);
    expect(cs.area()).to.be.greaterThan(0);
  });

  test('simplify argument is defaulted', () => {
    const cs = manifoldModule.CrossSection.circle(1).simplify();
    expect(cs.numContour()).toEqual(1);
    expect(cs.area()).to.be.greaterThan(0);
  });
});

suite('Manifold Bindings', () => {
  test('Simplify supports default argument', () => {
    const manifold = manifoldModule.Manifold.sphere(1).simplify();
    expect(manifold.volume()).toBeGreaterThan(0);
  });

  test('refineToTolerance does not throw (issue #1545)', () => {
    // Reproduces the original failing geometry from issue #1545: a flat-faced
    // mesh with normals set via calculateNormals + smoothByNormals. On flat
    // faces, tangents are parallel to the edges so d == 0 exactly, making
    // edgeDivisions return 0 for *any* tolerance. This triggers longest == 0 in
    // the keepInterior block — the integer divide-by-zero that traps in WASM.
    const cube =
        manifoldModule.Manifold.cube([10, 10, 10]).calculateNormals(0, 30);
    const smooth =
        manifoldModule.Manifold.ofMesh(cube.getMesh(0)).smoothByNormals(0);
    // Before the fix, any tolerance value crashed: RuntimeError: divide by zero
    const refined = smooth.refineToTolerance(0.1);
    expect(refined.volume()).toBeGreaterThan(0);
  });
});