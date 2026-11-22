import {beforeAll, expect, suite, test} from 'vitest';

import Module, {type ManifoldToplevel, type Polygons} from '../manifold'

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

  test('evenOdd fills by parity, not sign', () => {
    const nested: Polygons = [
      [[-2, -2], [2, -2], [2, 2], [-2, 2]],
      [[-1, -1], [1, -1], [1, 1], [-1, 1]],
    ];
    // Positive fill leaves the nested pair solid; even-odd cuts the inner
    // square out.
    expect(manifoldModule.CrossSection.ofPolygons(nested).area())
        .toBeCloseTo(16);
    const evenOdd = manifoldModule.CrossSection.evenOdd(nested);
    expect(evenOdd.numContour()).toEqual(2);
    expect(evenOdd.area()).toBeCloseTo(12);
  });

  test(
      'nonZero fills both signs and even windings, and cancels opposites',
      () => {
        const clockwise: Polygons = [[[0, 0], [0, 2], [2, 2], [2, 0]]];
        const reversed: Polygons = [clockwise[0].slice().reverse()];
        const single = manifoldModule.CrossSection.nonZero(clockwise);
        const doubled =
            manifoldModule.CrossSection.nonZero([clockwise[0], clockwise[0]]);
        const cancelled =
            manifoldModule.CrossSection.nonZero([clockwise[0], reversed[0]]);
        expect(single.area()).toBeCloseTo(4);
        expect(doubled.area()).toBeCloseTo(4);
        expect(cancelled.isEmpty()).toBe(true);
        expect(
            manifoldModule.CrossSection.ofPolygons(single.toPolygons()).area())
            .toBeCloseTo(4);
      });

  test('simplify argument is defaulted', () => {
    const cs = manifoldModule.CrossSection.circle(1).simplify();
    expect(cs.numContour()).toEqual(1);
    expect(cs.area()).to.be.greaterThan(0);
  });

  test('extrude scales X and Y by a single scaleTop (issue #1780)', () => {
    const square = manifoldModule.CrossSection.square(10);
    expect(square.extrude(10, 0, 0, 1).volume()).toBeCloseTo(1000);
    expect(square.extrude(10, 0, 0, 0.5).volume())
        .toBeCloseTo(square.extrude(10, 0, 0, [0.5, 0.5]).volume());
    expect(manifoldModule.Manifold.extrude(square, 10, 0, 0, 1).volume())
        .toBeCloseTo(1000);
  });
});

suite('Circular quality', () => {
  test('factories take a count or a quality object', () => {
    const {CrossSection, Manifold, segmentsFor} = manifoldModule;
    expect(CrossSection.circle(5).numVert()).toEqual(32);
    expect(CrossSection.circle(5, 7).numVert()).toEqual(7);
    expect(CrossSection.circle(5, {minAngle: 30}).numVert()).toEqual(12);
    expect(CrossSection.circle(5, {minAngle: undefined}).numVert()).toEqual(32);
    expect(Manifold.cylinder(1, 5, -1, {minAngle: 30}).numVert()).toEqual(24);
    expect(segmentsFor(5, {minAngle: 30})).toEqual(12);
    expect(segmentsFor(5, 7)).toEqual(7);
    expect(segmentsFor(5)).toEqual(32);
  });

  test('forDivisions gives exact revolve divisions', () => {
    const {CrossSection, forDivisions} = manifoldModule;
    const ring = CrossSection.square([1, 1]).translate([4.5, 0]);
    // A partial revolve of this ring has 4 vertices per slice.
    // 30 per full circle over 90 degrees is 7.5, rounded up to 8.
    expect(ring.revolve(30, 90).numVert()).toEqual(4 * (8 + 1));
    for (const degrees of [0.7, 360 / 19, 90, 119]) {
      for (const n of [1, 5, 21]) {
        expect(ring.revolve(forDivisions(n, degrees), degrees).numVert())
            .toEqual(4 * (n + 1));
      }
    }
  });
});

suite('Manifold Bindings', () => {
  test('Simplify supports default argument', () => {
    const manifold = manifoldModule.Manifold.sphere(1).simplify();
    expect(manifold.volume()).toBeGreaterThan(0);
  });

  test('merge returns a new mesh', () => {
    const {Manifold, Mesh} = manifoldModule;
    // Normals split the vertices, so the mesh without its merge vectors is
    // open and merge() builds them.
    const cube = Manifold.cube().calculateNormals().getMesh();
    const open = new Mesh({
      numProp: cube.numProp,
      vertProperties: cube.vertProperties,
      triVerts: cube.triVerts
    });
    const merged = open.merge();
    expect(merged).not.toBe(open);
    expect(open.mergeFromVert).toBeUndefined();
    expect(merged.mergeFromVert.length).toBeGreaterThan(0);
    expect(new Manifold(merged).volume()).toBeCloseTo(1);
  });

  test('precision is set at ingest and by withPrecision', () => {
    const {Manifold} = manifoldModule;
    const mesh = Manifold.sphere(1, 64).getMesh();
    expect(Manifold.ofMesh(mesh).getPrecision()).toEqual(1e-12);
    const coarse = Manifold.ofMesh(mesh, 1e-3);
    expect(coarse.getPrecision()).toEqual(1e-3);
    expect(new Manifold(mesh, 1e-3).getPrecision()).toEqual(1e-3);
    const stamped = Manifold.cube().withPrecision(1e-2);
    expect(stamped.getPrecision()).toEqual(1e-2);
    expect(coarse.add(stamped).getPrecision()).toEqual(1e-3);
    expect(Manifold.hull([coarse, stamped]).getPrecision()).toEqual(1e-3);
  });

  test('merge closes gaps up to its precision', () => {
    const {Manifold, Mesh} = manifoldModule;
    const cube = Manifold.cube().calculateNormals(0).getMesh();
    const vertProperties = cube.vertProperties.slice();
    vertProperties[0] += 1e-4;
    const open = new Mesh(
        {numProp: cube.numProp, vertProperties, triVerts: cube.triVerts});
    expect(() => new Manifold(open.merge())).toThrow();
    expect(new Manifold(open.merge(0, 1e-3)).status()).toEqual('NoError');
  });

  test('hull of Manifolds is computed at their smallest precision', () => {
    const {Manifold} = manifoldModule;
    const coarse = Manifold.sphere(1, 256).withPrecision(1e-2);
    const hull = Manifold.hull([coarse]);
    expect(hull.getPrecision()).toEqual(1e-2);
    expect(hull.numTri()).toEqual(coarse.hull().numTri());
    const withPoint = Manifold.hull([coarse, [2, 0, 0]]);
    expect(withPoint.getPrecision()).toEqual(1e-2);
    expect(withPoint.boundingBox().max[0]).toBeCloseTo(2);
    const cube = Manifold.cube();
    expect(Manifold.hull(cube, [2, 0, 0]).boundingBox().max[0]).toEqual(2);
    expect(Manifold.hull(cube, [[3, 0, 0], [0, 3, 0], [0, 0, 3]]).volume())
        .toBeCloseTo(4.5);
  });

  test('rayCast returns hits through cube', () => {
    const cube = manifoldModule.Manifold.cube([1, 1, 1], true);
    const hits = cube.rayCast([0, 0, -5], [0, 0, 5]);
    expect(hits).toHaveLength(2);
    expect(hits[0].position[2]).toEqual(-0.5);
    expect(hits[1].position[2]).toEqual(0.5);
    expect(hits[0].crossing).toEqual(1);
    expect(hits[1].crossing).toEqual(-1);
  });

  test('rayCast reports originalID and faceID', () => {
    const cube = manifoldModule.Manifold.cube([1, 1, 1], true).asOriginal(-1);
    const hits = cube.rayCast([0, 0, -5], [0, 0, 5]);
    expect(hits).toHaveLength(2);
    expect(hits[0].originalID).toEqual(cube.originalID());
    expect(hits[0].originalID).toBeGreaterThan(0);
    expect(hits[0].faceID).toEqual(0);
  });

  test('rayCast returns empty on miss', () => {
    const cube = manifoldModule.Manifold.cube([1, 1, 1], true);
    const hits = cube.rayCast([10, 10, -5], [10, 10, 5]);
    expect(hits).toHaveLength(0);
  });

  test('ExecutionContext happy path returns NoError', () => {
    const ctx = new manifoldModule.ExecutionContext();
    expect(ctx.cancelled()).toBe(false);
    // Fresh ctx with no scheduled work reads as trivially complete (1.0).
    expect(ctx.progress()).toEqual(1);

    const cube = manifoldModule.Manifold.cube([1, 1, 1], true);
    const sphere = manifoldModule.Manifold.sphere(1.0, 8);
    const u = cube.add(sphere);
    const uCtx = u.withContext(ctx);
    expect(uCtx.status()).toEqual('NoError');

    cube.delete();
    sphere.delete();
    u.delete();
    uCtx.delete();
    ctx.delete();
  });

  test('ExecutionContext cancels pending evaluation', () => {
    const ctx = new manifoldModule.ExecutionContext();
    ctx.cancel();
    expect(ctx.cancelled()).toBe(true);

    const cube = manifoldModule.Manifold.cube([1, 1, 1], true);
    const sphere = manifoldModule.Manifold.sphere(1.0, 8);
    const u = cube.add(sphere);
    const uCtx = u.withContext(ctx);
    expect(uCtx.status()).toEqual('Cancelled');

    cube.delete();
    sphere.delete();
    u.delete();
    uCtx.delete();
    ctx.delete();
  });

  test('ExecutionContext static factories', () => {
    // levelSet via the context: no source Manifold to attach via withContext.
    const sdfCtx = new manifoldModule.ExecutionContext();
    const sphere = sdfCtx.levelSet(
        ([x, y, z]) => 1 - Math.sqrt(x * x + y * y + z * z),
        {min: [-1.1, -1.1, -1.1], max: [1.1, 1.1, 1.1]}, 0.2);
    expect(sphere.status()).toEqual('NoError');
    expect(sdfCtx.progress()).toEqual(1);

    // fromMesh via the context.
    const ingestCtx = new manifoldModule.ExecutionContext();
    const cube = manifoldModule.Manifold.cube([2, 2, 2], true);
    const ingested = ingestCtx.fromMesh(cube.getMesh());
    expect(ingested.status()).toEqual('NoError');
    expect(ingestCtx.progress()).toEqual(1);

    // A pre-cancelled ctx propagates through a factory; this proves the ctx is
    // wired (NoError + progress==1 also hold for an untouched ctx).
    const cancelCtx = new manifoldModule.ExecutionContext();
    cancelCtx.cancel();
    const cancelled = cancelCtx.fromMesh(cube.getMesh());
    expect(cancelled.status()).toEqual('Cancelled');

    sphere.delete();
    cube.delete();
    ingested.delete();
    cancelled.delete();
    sdfCtx.delete();
    ingestCtx.delete();
    cancelCtx.delete();
  });

  test('refineToTolerance does not throw (issue #1545)', () => {
    // Reproduces the original failing geometry from issue #1545: a flat-faced
    // mesh with normals set via calculateNormals + smoothByNormals. On flat
    // faces, tangents are parallel to the edges so d == 0 exactly, making
    // edgeDivisions return 0 for *any* tolerance. This triggers longest == 0 in
    // the keepInterior block — the integer divide-by-zero that traps in WASM.
    const cube =
        manifoldModule.Manifold.cube([10, 10, 10]).calculateNormals(30);
    const smooth =
        manifoldModule.Manifold.ofMesh(cube.getMesh()).smoothByNormals();
    // Before the fix, any tolerance value crashed: RuntimeError: divide by zero
    const refined = smooth.refineToTolerance(0.1);
    expect(refined.volume()).toBeGreaterThan(0);
  });
});
