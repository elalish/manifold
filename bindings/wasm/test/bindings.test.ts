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
    expect(new manifoldModule.CrossSection(nested).area()).toBeCloseTo(16);
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
        expect(new manifoldModule.CrossSection(single.toPolygons()).area())
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

  test('precision is the mesh default and set by withPrecision', () => {
    const {Manifold, MIN_RELATIVE_PRECISION} = manifoldModule;
    const mesh = Manifold.sphere(1, 64).getMesh();
    expect(new Manifold(mesh).getPrecision()).toEqual(2 ** -23);
    expect(new manifoldModule.ExecutionContext().fromMesh(mesh).getPrecision())
        .toEqual(2 ** -23);
    expect(Manifold.cube().getPrecision()).toEqual(2 ** -23);
    const square = manifoldModule.CrossSection.square();
    for (const shape
             of [Manifold.extrude(square, 1),
                 Manifold.revolve(square.translate([1, 0])),
                 Manifold.levelSet(
                     p => 1 - Math.hypot(...p),
   {min: [-2, -2, -2], max: [2, 2, 2]}, 0.5),
                 Manifold.hull([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])]) {
      expect(shape.getPrecision()).toEqual(2 ** -23);
    }
    // 0, negative values, NaN and infinity give the default.
    for (const unset of [0, -1, NaN, Infinity]) {
      expect(Manifold.cube()
                 .withPrecision(1e-3)
                 .withPrecision(unset)
                 .getPrecision())
          .toEqual(2 ** -23);
    }
    expect(MIN_RELATIVE_PRECISION).toEqual(1e-12);
    expect(Manifold.cube().withPrecision(MIN_RELATIVE_PRECISION).getPrecision())
        .toEqual(1e-12);
    expect(Manifold.cube().withPrecision(1e-15).getPrecision()).toEqual(1e-12);
    const coarse = new Manifold(mesh).withPrecision(1e-3);
    expect(coarse.getPrecision()).toEqual(1e-3);
    // The mesh does not store the precision.
    expect(new Manifold(coarse.getMesh()).getPrecision()).toEqual(2 ** -23);
    const stamped = Manifold.cube().withPrecision(1e-2);
    expect(stamped.getPrecision()).toEqual(1e-2);
    expect(coarse.add(stamped).getPrecision()).toEqual(1e-3);
    expect(Manifold.hull([coarse, stamped]).getPrecision()).toEqual(1e-3);
  });

  test('merge closes gaps up to its tolerance', () => {
    const {Manifold, Mesh} = manifoldModule;
    const cube = Manifold.cube().calculateNormals(0).getMesh();
    const vertProperties = cube.vertProperties.slice();
    vertProperties[0] += 1e-4;
    const open = new Mesh(
        {numProp: cube.numProp, vertProperties, triVerts: cube.triVerts});
    expect(() => new Manifold(open.merge())).toThrow();
    expect(new Manifold(open.merge(1e-3)).status()).toEqual('NoError');
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

  test('ManifoldError messages', () => {
    const ManifoldError = (manifoldModule as any).ManifoldError;
    const status = (manifoldModule as any).status;
    const codes =
        Object.keys(status).filter(k => typeof status[k]?.value === 'number');
    expect(codes).toContain('Cancelled');
    expect(codes.map(code => status[code].value)).toEqual([...codes.keys()]);
    const messages = codes.map(code => {
      const error = new ManifoldError(code);
      expect(error).toBeInstanceOf(Error);
      expect(error.name).toEqual('ManifoldError');
      expect(error.code).toEqual(code);
      return error.message;
    });
    const unknown = new ManifoldError('Bogus').message;
    expect(messages.every(m => m.length > 0 && m !== unknown)).toBe(true);
    expect(new Set(messages).size).toEqual(codes.length);
    expect(new ManifoldError('Cancelled').message).not.toEqual('Unknown error');

    const mesh = new manifoldModule.Mesh({
      numProp: 3,
      vertProperties: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      triVerts: new Uint32Array([0, 1, 2])
    });
    let thrown: any;
    try {
      new manifoldModule.Manifold(mesh);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(ManifoldError);
    expect(thrown.code).toEqual('NotManifold');
    expect(thrown.message).toEqual(new ManifoldError('NotManifold').message);
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
        new manifoldModule.Manifold(cube.getMesh()).smoothByNormals();
    // Before the fix, any tolerance value crashed: RuntimeError: divide by zero
    const refined = smooth.refineToTolerance(0.1);
    expect(refined.volume()).toBeGreaterThan(0);
  });
});

suite('Mesh64', () => {
  // Not representable in float32, so any float round trip changes it.
  const offset = 1234567.123456789;

  // A unit cube moved to [offset, offset + 1]^3, with doubles computed in JS.
  function doubleCube() {
    const cube = manifoldModule.Manifold.cube();
    const mesh = cube.getMesh();
    cube.delete();
    const vertProperties =
        Float64Array.from(mesh.vertProperties, x => x + offset);
    return new manifoldModule.Mesh64(
        {numProp: mesh.numProp, triVerts: mesh.triVerts, vertProperties});
  }

  // The distinct vertex positions, sorted.
  function sortedVerts(
      mesh: {numProp: number, vertProperties: ArrayLike<number>}) {
    const keys = new Set<string>();
    const props = Array.from(mesh.vertProperties);
    for (let i = 0; i < props.length; i += mesh.numProp) {
      keys.add(JSON.stringify(props.slice(i, i + 3)));
    }
    return [...keys]
        .map(key => JSON.parse(key) as number[])
        .sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  }

  test('getMesh() stays float, getMesh64() is exact', () => {
    const {Manifold, Mesh, Mesh64} = manifoldModule;
    const input = doubleCube();
    expect(input.numTri).toBe(12);
    expect(input.numVert).toBe(8);
    expect(input.position(0)).toBeInstanceOf(Float64Array);
    const manifold = new Manifold(input);

    const float = manifold.getMesh();
    expect(float).toBeInstanceOf(Mesh);
    expect(float.vertProperties).toBeInstanceOf(Float32Array);
    expect(float.vertProperties).toContain(Math.fround(offset));

    const double = manifold.getMesh64();
    expect(double).toBeInstanceOf(Mesh64);
    expect(double.vertProperties).toBeInstanceOf(Float64Array);
    expect(double.triVerts).toBeInstanceOf(Uint32Array);
    expect(double.mergeFromVert).toBeInstanceOf(Uint32Array);
    expect(double.mergeToVert).toBeInstanceOf(Uint32Array);
    expect(double.runFlags).toBeInstanceOf(Uint8Array);
    expect(double.runIndex).toBeInstanceOf(Uint32Array);
    expect(double.runOriginalID).toBeInstanceOf(Uint32Array);
    expect(double.faceID).toBeInstanceOf(Uint32Array);
    expect(double.runTransform).toBeInstanceOf(Float64Array);
    expect(typeof double.numProp).toBe('number');
    expect(double.position(0)).toBeInstanceOf(Float64Array);
    expect(sortedVerts(double)).toEqual(sortedVerts(input));
    manifold.delete();
  });

  test('Boolean of Mesh64 inputs keeps exact input coordinates', () => {
    const {Manifold} = manifoldModule;
    // A slab through the middle of the cube, wider in x and y.
    const slabMesh = doubleCube();
    for (let i = 0; i < slabMesh.vertProperties.length; i += 3) {
      const v = slabMesh.vertProperties;
      v[i] = v[i] == offset ? offset - 1 : offset + 2;
      v[i + 1] = v[i + 1] == offset ? offset - 1 : offset + 2;
      v[i + 2] = v[i + 2] == offset ? offset + 0.5 : offset + 0.75;
    }
    const cube = new Manifold(doubleCube());
    const slab = new Manifold(slabMesh);
    const result = cube.subtract(slab);
    expect(result.volume()).toBeCloseTo(0.75, 6);
    const zs = new Set<number>();
    for (const [x, y, z] of sortedVerts(result.getMesh64())) {
      // New verts lie on the cube's vertical edges and the slab's planes.
      expect([offset, offset + 1]).toContain(x);
      expect([offset, offset + 1]).toContain(y);
      zs.add(z);
    }
    expect([...zs].sort((a, b) => a - b)).toEqual([
      offset, offset + 0.5, offset + 0.75, offset + 1
    ]);
    cube.delete();
    slab.delete();
    result.delete();
  });

  test('faceID and runOriginalID survive a Mesh64 round trip', () => {
    const {Manifold} = manifoldModule;
    const input = doubleCube();
    const id = Manifold.reserveIDs(2);
    input.faceID = Uint32Array.from({length: input.numTri}, (_, i) => 100 + i);
    input.runOriginalID = new Uint32Array([id, id + 1]);
    input.runIndex = new Uint32Array([0, 18, input.triVerts.length]);
    const manifold = new Manifold(input);
    const output = manifold.getMesh64();
    expect(Array.from(output.runOriginalID)).toEqual([id, id + 1]);
    expect(Array.from(output.faceID).sort((a, b) => a - b))
        .toEqual(Array.from(input.faceID));

    // Feed the output back in: a lossless round trip.
    const again = new Manifold(output);
    const againMesh = again.getMesh64();
    expect(Array.from(againMesh.runOriginalID)).toEqual([id, id + 1]);
    expect(Array.from(againMesh.faceID).sort((a, b) => a - b))
        .toEqual(Array.from(input.faceID));
    expect(sortedVerts(againMesh)).toEqual(sortedVerts(input));
    manifold.delete();
    again.delete();
  });

  test('extras, tangents, run transforms and flags stay double', () => {
    const {Manifold} = manifoldModule;
    // Normals add extra properties and set runFlags bit 1, smoothByNormals adds
    // tangents, and moving an original records its run transform.
    const sphere = Manifold.sphere(1, 8).calculateNormals();
    const smooth = sphere.smoothByNormals();
    const original = smooth.asOriginal();
    const manifold = original.translate(offset, 0, 0);
    const float = manifold.getMesh();
    const double = manifold.getMesh64();
    expect(double.halfedgeTangent).toBeInstanceOf(Float64Array);
    expect(double.halfedgeTangent.length).toBeGreaterThan(0);
    expect(double.halfedgeTangent)
        .not.toEqual(Float64Array.from(float.halfedgeTangent));
    expect(double.tangent(0)).toBeInstanceOf(Float64Array);
    expect(double.runTransform[9]).toBe(offset);
    expect(float.runTransform[9]).toBe(Math.fround(offset));
    expect(double.transform(0)[12]).toBe(offset);
    expect(double.hasNormals(0)).toBe(true);
    expect(double.extras(0)).toBeInstanceOf(Float64Array);

    // Feed it back in: every field comes back unchanged.
    const roundTrip = new Manifold(double);
    const again = roundTrip.getMesh64();
    const rows = (values: Float64Array, stride: number) => {
      const out = [];
      for (let i = 0; i < values.length; i += stride) {
        out.push(Array.from(values.subarray(i, i + stride)));
      }
      return out.sort((a, b) => {
        for (let j = 0; j < stride; ++j) {
          if (a[j] != b[j]) return a[j] - b[j];
        }
        return 0;
      });
    };
    expect(again.numProp).toBe(double.numProp);
    expect(rows(again.vertProperties, again.numProp))
        .toEqual(rows(double.vertProperties, double.numProp));
    expect(rows(again.halfedgeTangent, 4))
        .toEqual(rows(double.halfedgeTangent, 4));
    expect(again.runTransform).toEqual(double.runTransform);
    expect(again.runFlags).toEqual(double.runFlags);
    for (const m of [sphere, smooth, original, manifold, roundTrip]) m.delete();
  });

  test('Mesh64.merge() stays double', () => {
    const {Manifold, Mesh64} = manifoldModule;
    const cube = new Manifold(doubleCube());
    const withNormals = cube.calculateNormals();
    const normalsMesh = withNormals.getMesh64();
    const open = new Mesh64({
      numProp: normalsMesh.numProp,
      vertProperties: normalsMesh.vertProperties,
      triVerts: normalsMesh.triVerts
    });
    const merged = open.merge();
    expect(merged).toBeInstanceOf(Mesh64);
    expect(merged.vertProperties).toBeInstanceOf(Float64Array);
    expect(merged.mergeFromVert).toBeInstanceOf(Uint32Array);
    expect(merged.mergeFromVert.length).toBeGreaterThan(0);
    expect(merged.vertProperties).toEqual(normalsMesh.vertProperties);
    const manifold = new Manifold(merged);
    expect(manifold.volume()).toBeCloseTo(1, 6);
    expect(sortedVerts(manifold.getMesh64()))
        .toEqual(sortedVerts(doubleCube()));
    for (const m of [cube, withNormals, manifold]) m.delete();
  });

  test('fromMesh also reads a Mesh64 as double', () => {
    const {ExecutionContext} = manifoldModule;
    const input = doubleCube();
    const ctx = new ExecutionContext();
    const manifold = ctx.fromMesh(input);
    expect(sortedVerts(manifold.getMesh64())).toEqual(sortedVerts(input));
    manifold.delete();
    ctx.delete();
  });

  test('a Manifold from a Mesh64 has the finest precision', () => {
    const {Manifold, Mesh, ExecutionContext, MIN_RELATIVE_PRECISION} =
        manifoldModule;
    const input = doubleCube();
    const fromFloat = new Manifold(new Mesh({
      numProp: input.numProp,
      triVerts: input.triVerts,
      vertProperties: Float32Array.from(input.vertProperties)
    }));
    expect(fromFloat.getPrecision()).toEqual(2 ** -23);
    const ctx = new ExecutionContext();
    const fromDouble = [new Manifold(input), ctx.fromMesh(input)];
    for (const manifold of fromDouble) {
      expect(manifold.getPrecision()).toEqual(MIN_RELATIVE_PRECISION);
    }
    const coarse = fromDouble[0].withPrecision(1e-3);
    expect(coarse.getPrecision()).toEqual(1e-3);
    // 0 gives float epsilon, even for a Manifold made from a Mesh64.
    const unset = fromDouble[0].withPrecision(0);
    expect(unset.getPrecision()).toEqual(2 ** -23);
    // The mesh does not store the precision.
    const again = new Manifold(coarse.getMesh64());
    expect(again.getPrecision()).toEqual(MIN_RELATIVE_PRECISION);
    for (const m of [fromFloat, ...fromDouble, coarse, unset, again]) {
      m.delete();
    }
    ctx.delete();
  });

  test('a Mesh64 from another module instance reads as double', async () => {
    const other = await Module();
    other.setup();
    const input = doubleCube();
    const ctx = new other.ExecutionContext();
    for (const manifold of [new other.Manifold(input), ctx.fromMesh(input)]) {
      expect(sortedVerts(manifold.getMesh64())).toEqual(sortedVerts(input));
      manifold.delete();
    }
    ctx.delete();
  });

  test('a plain object wrapped in Mesh64 reads as double', () => {
    const {Manifold, Mesh64, ExecutionContext} = manifoldModule;
    const input = doubleCube();
    const plain = {
      numProp: input.numProp,
      triVerts: input.triVerts,
      vertProperties: input.vertProperties
    };
    const ctx = new ExecutionContext();
    for (const manifold
             of [new Manifold(new Mesh64(plain)),
                 ctx.fromMesh(new Mesh64(plain))]) {
      expect(sortedVerts(manifold.getMesh64())).toEqual(sortedVerts(input));
      manifold.delete();
    }
    ctx.delete();
  });

  test('a bad Mesh64 throws', () => {
    const {Manifold} = manifoldModule;
    const input = doubleCube();
    input.triVerts = Uint32Array.from(input.triVerts, v => v == 0 ? 8 : v);
    expect(() => new Manifold(input)).toThrow('Vertex index out of bounds');
  });

  test('a Mesh reads as float, whatever its array type', () => {
    const {Manifold, Mesh} = manifoldModule;
    const input = doubleCube();
    const expected = sortedVerts(input).map(v => v.map(Math.fround));
    for (const vertProperties
             of [Array.from(input.vertProperties),
                 Float32Array.from(input.vertProperties),
                 Float64Array.from(input.vertProperties)]) {
      const mesh = new Mesh({
        numProp: input.numProp,
        triVerts: input.triVerts,
        vertProperties: vertProperties as Float32Array
      });
      const manifold = new Manifold(mesh);
      expect(sortedVerts(manifold.getMesh64())).toEqual(expected);
      manifold.delete();
    }
  });

  test('a Boolean can mix Mesh64 and Mesh inputs', () => {
    const {Manifold, Mesh} = manifoldModule;
    const double = new Manifold(doubleCube());
    const floatMesh = doubleCube();
    const floatManifold = new Manifold(new Mesh({
      numProp: floatMesh.numProp,
      triVerts: floatMesh.triVerts,
      vertProperties: Float32Array.from(floatMesh.vertProperties)
    }));
    const moved = floatManifold.translate(0, 0, 2);
    const union = double.add(moved);
    expect(union.volume()).toBeCloseTo(2, 6);
    const xs = new Set(sortedVerts(union.getMesh64()).map(v => v[0]));
    expect([...xs].sort((a, b) => a - b)).toEqual([
      Math.fround(offset), offset, Math.fround(offset + 1), offset + 1
    ].sort((a, b) => a - b));
    for (const m of [double, floatManifold, moved, union]) m.delete();
  });
});

suite('Bindings match C++', () => {
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

  test('the transform of a run with no runTransform is the identity', () => {
    const {Manifold, Mesh, Mesh64} = manifoldModule;
    const cube = Manifold.cube();
    const sphere = Manifold.sphere(1, 8);
    for (const m of [cube, sphere]) {
      for (const mesh of [m.getMesh(), m.getMesh64()]) {
        expect(mesh.numRun).toBeGreaterThan(0);
        for (let run = 0; run < mesh.numRun; ++run) {
          expect(mesh.transform(run)).toEqual(identity);
        }
      }
    }
    // As in C++, a run past the end of runTransform gets the identity, and so
    // does an invalid run.
    const row = Array.from({length: 12}, (_, i) => i + 2);
    for (const [MeshType, Float] of [
             [Mesh, Float32Array], [Mesh64, Float64Array]] as const) {
      for (const runTransform of [undefined, new Float(), Float.from(row)]) {
        const mesh = new MeshType({
          numProp: 3,
          triVerts: new Uint32Array(),
          vertProperties: new Float(),
          runOriginalID: new Uint32Array([1, 2]),
          runTransform
        } as any);
        const first = mesh.transform(0);
        if (runTransform?.length) {
          expect([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14].map(i => first[i]))
              .toEqual(row);
        } else {
          expect(first).toEqual(identity);
        }
        for (const run of [1, -1, NaN]) {
          expect(mesh.transform(run)).toEqual(identity);
        }
      }
    }
    cube.delete();
    sphere.delete();
  });

  test('each run transform maps its original onto that run', () => {
    const {Manifold} = manifoldModule;
    const cube = Manifold.cube();
    const moved = cube.rotate(0, 0, 90).translate(5, 2, 3);
    const union = cube.add(moved);
    const original = cube.getMesh64();
    for (const mesh of [union.getMesh(), union.getMesh64()]) {
      expect(mesh.numRun).toBe(2);
      for (let run = 0; run < mesh.numRun; ++run) {
        const m = mesh.transform(run);
        expect([m[3], m[7], m[11], m[15]]).toEqual([0, 0, 0, 1]);
        const mapped = [];
        for (let v = 0; v < original.numVert; ++v) {
          const p = [...original.position(v), 1];
          mapped.push([0, 1, 2].map(
              row => [0, 1, 2, 3].reduce(
                  (sum, col) => sum + m[4 * col + row] * p[col], 0)));
        }
        const begin = mesh.runIndex[run] / 3;
        const end = mesh.runIndex[run + 1] / 3;
        for (let tri = begin; tri < end; ++tri) {
          for (const v of mesh.verts(tri)) {
            const p = mesh.position(v);
            expect(
                mapped.some(q => q.every((x, i) => Math.abs(x - p[i]) < 1e-6)))
                .toBe(true);
          }
        }
      }
    }
    for (const m of [cube, moved, union]) m.delete();
  });

  test('a NaN transform argument gives NonFiniteVertex, as in C++', () => {
    const {Manifold} = manifoldModule;
    const cube = Manifold.cube();
    const args = [[NaN], [0, NaN], [0, 0, NaN], [1, 2, 3], [1], [1, 2]];
    for (const op of ['translate', 'rotate'] as const) {
      for (const a of args) {
        const spread = (cube[op] as any)(...a);
        const array = cube[op]([a[0], a[1] ?? 0, a[2] ?? 0]);
        const expected = a.some(isNaN) ? 'NonFiniteVertex' : 'NoError';
        expect(spread.status()).toBe(expected);
        expect(array.status()).toBe(expected);
        if (expected == 'NoError') {
          expect(spread.boundingBox()).toEqual(array.boundingBox());
        }
        spread.delete();
        array.delete();
      }
    }
    cube.delete();
  });

  test('CrossSection.translate passes NaN on to C++', () => {
    const {CrossSection} = manifoldModule;
    const square = CrossSection.square();
    for (const a of [[NaN, 5], [5, NaN], [1, 2], [1]]) {
      const spread = (square.translate as any)(...a);
      const array = square.translate([a[0], a[1] ?? 0]);
      expect(spread.bounds()).toEqual(array.bounds());
      spread.delete();
      array.delete();
    }
    square.delete();
  });

  test('empty polygons give empty results', () => {
    const {CrossSection, Manifold, triangulate} = manifoldModule;
    const sections = [
      new CrossSection([]), CrossSection.evenOdd([]), CrossSection.nonZero([]),
      CrossSection.hull([]), CrossSection.union([]),
      CrossSection.square().intersect([])
    ];
    for (const cs of sections) {
      expect(cs.isEmpty()).toBe(true);
      expect(cs.numContour()).toBe(0);
      cs.delete();
    }
    // Empty polygons extrude and revolve like any empty CrossSection, so the
    // status is whatever C++ gives for empty input.
    const square = CrossSection.square();
    const empty = square.subtract(square);
    const pairs = [
      [Manifold.extrude([], 1), empty.extrude(1)],
      [Manifold.revolve([]), empty.revolve()]
    ];
    for (const [fromPolygons, fromEmpty] of pairs) {
      expect(fromPolygons.status()).toBe(fromEmpty.status());
      expect(fromPolygons.isEmpty()).toBe(true);
      fromPolygons.delete();
      fromEmpty.delete();
    }
    square.delete();
    empty.delete();
    expect(triangulate([])).toEqual([]);
  });

  test('Polygons are read by shape, not by contour length', () => {
    const {CrossSection} = manifoldModule;
    const square: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
    const small: [number, number][][] =
        [[], [[5, 5]], [[5, 5], [6, 5]], [[5, 5], [6, 5], [5, 6]]];
    for (const first of small) {
      const both = new CrossSection([first, square]);
      const alone = new CrossSection([first]);
      const ref = new CrossSection(square);
      expect(both.area()).toBeCloseTo(ref.area() + alone.area(), 12);
      expect(new CrossSection(first).area()).toBe(alone.area());
      for (const cs of [both, alone, ref]) cs.delete();
    }
  });

  test('points may be {x, y} objects, including at the origin', () => {
    const {CrossSection, Manifold} = manifoldModule;
    const tri = [[0, 0], [1, 0], [0, 1]];
    const objects = tri.map(([x, y]) => ({x, y})) as any;
    for (const cs
             of [new CrossSection(objects), new CrossSection([objects]),
                 CrossSection.hull(objects), CrossSection.hull([objects])]) {
      expect(cs.area()).toBeCloseTo(0.5, 12);
      cs.delete();
    }
    const corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const hull = Manifold.hull(corners.map(([x, y, z]) => ({x, y, z})) as any);
    expect(hull.volume()).toBeCloseTo(1 / 6, 12);
    hull.delete();
  });

  test('version is three non-negative integers', () => {
    const v = manifoldModule.version();
    expect(Array.isArray(v)).toBe(true);
    expect(v).toHaveLength(3);
    for (const x of v) {
      expect(Number.isInteger(x)).toBe(true);
      expect(x).toBeGreaterThanOrEqual(0);
    }
  });
});
