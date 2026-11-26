import {afterEach, beforeAll, expect, suite, test, vi} from 'vitest';

import Module, {type ErrorStatus, type ExecutionContext, type ManifoldToplevel} from '../manifold';

let wasm: ManifoldToplevel;
const handles: any[] = [];

beforeAll(async () => {
  wasm = await Module();
  wasm.setup();
});

afterEach(() => {
  vi.restoreAllMocks();
  for (const handle of handles) {
    if (!handle.isDeleted()) handle.delete();
  }
  handles.length = 0;
});

// Tracks embind handles, including those inside arrays, for deletion after
// each test.
function keep<T>(value: T): T {
  for (const v of Array.isArray(value) ? value : [value]) {
    if (typeof v?.isDeleted === 'function') handles.push(v);
  }
  return value;
}

// A: returns Manifolds and checks them. B: returns something else and checks
// its Manifold inputs. C: never throws. D: like A or B, and also throws
// 'Cancelled' through a cancelled context.
type Category = 'A'|'B'|'C'|'D';

interface Entry {
  cat: Category;
  // Called with errored Manifold inputs for B and D, and for A where the
  // call has a Manifold input. A constructor fixture passes an invalid
  // parameter instead, with its code.
  call: () => unknown;
  code?: ErrorStatus;
  // D only: the same call through a cancelled context.
  cancel?: (ctx: ExecutionContext) => unknown;
}

// Evaluates to NonFiniteVertex, but does not throw when created.
const bad = () => keep(keep(wasm.Manifold.cube()).translate([NaN, 0, 0]));
const good = () => keep(wasm.Manifold.cube());
// A deferred Boolean, so evaluating it under a context sees a cancel.
const lazy = () =>
    keep(keep(wasm.Manifold.cube()).add(keep(wasm.Manifold.sphere(0.6, 8))));
const square = () => keep(wasm.CrossSection.square());
const triangle = () => new wasm.Mesh({
  numProp: 3,
  vertProperties: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
  triVerts: new Uint32Array([0, 1, 2])
});
const sdf = (p: number[]) => 1 - Math.hypot(...p);
const box = {
  min: [-2, -2, -2],
  max: [2, 2, 2]
} as const;

const CATEGORIES: Record<string, Entry> = {
  'Manifold.constructor': {
    cat: 'A',
    call: () => new wasm.Manifold(triangle()),
    code: 'NotManifold'
  },
  // Cannot fail.
  'Manifold.tetrahedron':
      {cat: 'A', call: () => wasm.Manifold.tetrahedron(), code: 'NoError'},
  'Manifold.cube': {
    cat: 'A',
    call: () => wasm.Manifold.cube([-1, 1, 1]),
    code: 'InvalidConstruction'
  },
  'Manifold.cylinder': {
    cat: 'A',
    call: () => wasm.Manifold.cylinder(NaN, 1),
    code: 'InvalidConstruction'
  },
  'Manifold.sphere': {
    cat: 'A',
    call: () => wasm.Manifold.sphere(-1),
    code: 'InvalidConstruction'
  },
  'Manifold.extrude': {
    cat: 'A',
    call: () => wasm.Manifold.extrude(square(), 0),
    code: 'InvalidConstruction'
  },
  'Manifold.revolve': {
    cat: 'A',
    call: () => wasm.Manifold.revolve(square(), 0, 0),
    code: 'InvalidConstruction'
  },
  'Manifold.levelSet': {
    cat: 'A',
    call: () => wasm.Manifold.levelSet(sdf, box, -1),
    code: 'InvalidConstruction'
  },
  'Manifold.hull': {cat: 'A', call: () => wasm.Manifold.hull([bad()])},
  'Manifold.reserveIDs': {cat: 'C', call: () => wasm.Manifold.reserveIDs(1)},
  'Manifold.union': {cat: 'C', call: () => wasm.Manifold.union(bad(), good())},
  'Manifold.difference':
      {cat: 'C', call: () => wasm.Manifold.difference([bad(), good()])},
  'Manifold.intersection':
      {cat: 'C', call: () => wasm.Manifold.intersection(bad(), good())},
  'Manifold.add': {cat: 'C', call: () => bad().add(good())},
  'Manifold.subtract': {cat: 'C', call: () => bad().subtract(good())},
  'Manifold.intersect': {cat: 'C', call: () => good().intersect(bad())},
  'Manifold.translate': {cat: 'C', call: () => bad().translate(1, 0, 0)},
  'Manifold.rotate': {cat: 'C', call: () => bad().rotate(90)},
  'Manifold.scale': {cat: 'C', call: () => bad().scale(2)},
  'Manifold.mirror': {cat: 'C', call: () => bad().mirror([0, 0, 0])},
  'Manifold.transform': {
    cat: 'C',
    call: () =>
        bad().transform([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
  },
  'Manifold.trimByPlane': {cat: 'C', call: () => bad().trimByPlane([0, 0, 1])},
  'Manifold.withContext': {
    cat: 'C',
    call: () => bad().withContext(keep(new wasm.ExecutionContext()))
  },
  'Manifold.status': {cat: 'C', call: () => bad().status()},
  'Manifold.delete': {cat: 'C', call: () => bad().delete()},
  'Manifold.warp': {cat: 'A', call: () => bad().warp(() => {})},
  'Manifold.warpBatch': {cat: 'A', call: () => bad().warpBatch(() => {})},
  'Manifold.setProperties': {
    cat: 'A',
    call: () => good().setProperties(-1, () => {}),
    code: 'InvalidConstruction'
  },
  'Manifold.calculateCurvature':
      {cat: 'A', call: () => bad().calculateCurvature(3, 4)},
  'Manifold.calculateNormals': {cat: 'A', call: () => bad().calculateNormals()},
  'Manifold.smoothByNormals': {cat: 'A', call: () => bad().smoothByNormals()},
  'Manifold.simplify': {cat: 'A', call: () => bad().simplify()},
  'Manifold.asOriginal': {cat: 'A', call: () => bad().asOriginal()},
  'Manifold.decompose': {cat: 'A', call: () => bad().decompose()},
  'Manifold.split': {cat: 'A', call: () => good().split(bad())},
  'Manifold.splitByPlane':
      {cat: 'A', call: () => bad().splitByPlane([0, 0, 1])},
  'Manifold.refine': {
    cat: 'D',
    call: () => bad().refine(2),
    cancel: ctx => lazy().withContext(ctx).refine(2)
  },
  'Manifold.refineToLength': {
    cat: 'D',
    call: () => good().refineToLength(0),
    code: 'InvalidConstruction',
    cancel: ctx => lazy().withContext(ctx).refineToLength(0.5)
  },
  'Manifold.refineToTolerance': {
    cat: 'D',
    call: () => good().refineToTolerance(0),
    code: 'InvalidConstruction',
    cancel: ctx => lazy().withContext(ctx).refineToTolerance(0.1)
  },
  'Manifold.hull#member': {
    cat: 'D',
    call: () => bad().hull(),
    cancel: ctx => lazy().withContext(ctx).hull()
  },
  'Manifold.minkowskiSum': {
    cat: 'D',
    call: () => good().minkowskiSum(bad()),
    cancel: ctx => lazy().withContext(ctx).minkowskiSum(good())
  },
  'Manifold.minkowskiDifference': {
    cat: 'D',
    call: () => bad().minkowskiDifference(good()),
    cancel: ctx => lazy().withContext(ctx).minkowskiDifference(good())
  },
  'Manifold.withPrecision': {
    cat: 'D',
    call: () => bad().withPrecision(1e-3),
    cancel: ctx => lazy().withContext(ctx).withPrecision(1e-3)
  },
  'Manifold.getPrecision': {
    cat: 'D',
    call: () => bad().getPrecision(),
    cancel: ctx => lazy().withContext(ctx).getPrecision()
  },
  'Manifold.getMesh': {cat: 'B', call: () => bad().getMesh()},
  'Manifold.getMesh64': {cat: 'B', call: () => bad().getMesh64()},
  'Manifold.isEmpty': {cat: 'B', call: () => bad().isEmpty()},
  'Manifold.numVert': {cat: 'B', call: () => bad().numVert()},
  'Manifold.numEdge': {cat: 'B', call: () => bad().numEdge()},
  'Manifold.numTri': {cat: 'B', call: () => bad().numTri()},
  'Manifold.numProp': {cat: 'B', call: () => bad().numProp()},
  'Manifold.numPropVert': {cat: 'B', call: () => bad().numPropVert()},
  'Manifold.boundingBox': {cat: 'B', call: () => bad().boundingBox()},
  'Manifold.genus': {cat: 'B', call: () => bad().genus()},
  'Manifold.volume': {cat: 'B', call: () => bad().volume()},
  'Manifold.surfaceArea': {cat: 'B', call: () => bad().surfaceArea()},
  'Manifold.originalID': {cat: 'B', call: () => bad().originalID()},
  'Manifold.minGap': {cat: 'B', call: () => good().minGap(bad(), 1)},
  'Manifold.rayCast':
      {cat: 'B', call: () => bad().rayCast([0, 0, -5], [0, 0, 5])},
  'Manifold.windingNumber':
      {cat: 'B', call: () => bad().windingNumber([[0, 0, 0]])},
  'Manifold.slice': {cat: 'B', call: () => bad().slice(0.5)},
  'Manifold.project': {cat: 'B', call: () => bad().project()},

  // CrossSection has no status, so only extrude and revolve can throw.
  'CrossSection.constructor':
      {cat: 'C', call: () => new wasm.CrossSection([[0, 0], [1, 0], [0, 1]])},
  'CrossSection.evenOdd': {
    cat: 'C',
    call: () => wasm.CrossSection.evenOdd([[0, 0], [1, 0], [0, 1]])
  },
  'CrossSection.nonZero': {
    cat: 'C',
    call: () => wasm.CrossSection.nonZero([[0, 0], [1, 0], [0, 1]])
  },
  'CrossSection.square': {cat: 'C', call: () => wasm.CrossSection.square(-1)},
  'CrossSection.circle': {cat: 'C', call: () => wasm.CrossSection.circle(-1)},
  'CrossSection.union':
      {cat: 'C', call: () => wasm.CrossSection.union(square(), square())},
  'CrossSection.difference': {
    cat: 'C',
    call: () => wasm.CrossSection.difference([square(), square()])
  },
  'CrossSection.intersection': {
    cat: 'C',
    call: () => wasm.CrossSection.intersection(square(), square())
  },
  'CrossSection.hull':
      {cat: 'C', call: () => wasm.CrossSection.hull([square()])},
  'CrossSection.extrude': {
    cat: 'A',
    call: () => square().extrude(NaN),
    code: 'InvalidConstruction'
  },
  'CrossSection.revolve': {
    cat: 'A',
    call: () => square().revolve(0, NaN),
    code: 'InvalidConstruction'
  },
  'CrossSection.area': {cat: 'C', call: () => square().area()},
  'CrossSection.numVert': {cat: 'C', call: () => square().numVert()},
  'CrossSection.numContour': {cat: 'C', call: () => square().numContour()},
  'CrossSection.isEmpty': {cat: 'C', call: () => square().isEmpty()},
  'CrossSection.bounds': {cat: 'C', call: () => square().bounds()},
  'CrossSection.translate': {cat: 'C', call: () => square().translate(1, 0)},
  'CrossSection.rotate': {cat: 'C', call: () => square().rotate(90)},
  'CrossSection.scale': {cat: 'C', call: () => square().scale(2)},
  'CrossSection.mirror': {cat: 'C', call: () => square().mirror([0, 0])},
  'CrossSection.transform':
      {cat: 'C', call: () => square().transform([1, 0, 0, 0, 1, 0, 0, 0, 1])},
  'CrossSection.warp': {cat: 'C', call: () => square().warp(v => {v[0] = NaN})},
  'CrossSection.warpBatch':
      {cat: 'C', call: () => square().warpBatch(v => {v[0] = NaN})},
  'CrossSection.simplify': {cat: 'C', call: () => square().simplify()},
  'CrossSection.offset': {cat: 'C', call: () => square().offset(NaN)},
  'CrossSection.add': {cat: 'C', call: () => square().add(square())},
  'CrossSection.subtract': {cat: 'C', call: () => square().subtract(square())},
  'CrossSection.intersect':
      {cat: 'C', call: () => square().intersect(square())},
  'CrossSection.hull#member': {cat: 'C', call: () => square().hull()},
  'CrossSection.decompose': {cat: 'C', call: () => square().decompose()},
  'CrossSection.toPolygons': {cat: 'C', call: () => square().toPolygons()},
  'CrossSection.delete': {cat: 'C', call: () => square().delete()},

  'ExecutionContext.constructor':
      {cat: 'C', call: () => new wasm.ExecutionContext()},
  'ExecutionContext.cancel':
      {cat: 'C', call: () => keep(new wasm.ExecutionContext()).cancel()},
  'ExecutionContext.cancelled':
      {cat: 'C', call: () => keep(new wasm.ExecutionContext()).cancelled()},
  'ExecutionContext.progress':
      {cat: 'C', call: () => keep(new wasm.ExecutionContext()).progress()},
  'ExecutionContext.fromMesh': {
    cat: 'D',
    call: () => keep(new wasm.ExecutionContext()).fromMesh(triangle()),
    code: 'NotManifold',
    cancel: ctx => ctx.fromMesh(good().getMesh())
  },
  'ExecutionContext.levelSet': {
    cat: 'D',
    call: () => keep(new wasm.ExecutionContext()).levelSet(sdf, box, -1),
    code: 'InvalidConstruction',
    cancel: ctx => ctx.levelSet(sdf, box, 0.5)
  },
  'ExecutionContext.delete':
      {cat: 'C', call: () => new wasm.ExecutionContext().delete()},

  'triangulate':
      {cat: 'C', call: () => wasm.triangulate([[0, 0], [1, 0], [0, 1]])},
  'segmentsFor': {cat: 'C', call: () => wasm.segmentsFor(1)},
  'forDivisions': {cat: 'C', call: () => wasm.forDivisions(4, 90)},
  'version': {cat: 'C', call: () => wasm.version()},
};

const own = (o: object) => Object.getOwnPropertyNames(o).filter(
    n => !n.startsWith('_') && n !== 'constructor');

// A member and a static of the same name are told apart by '#member'.
function publicNames(m: any): Set<string> {
  const names = new Set<string>();
  for (const cls of ['Manifold', 'CrossSection']) {
    const statics = Object.getOwnPropertyNames(m[cls]).filter(
        n => !['length', 'name', 'prototype'].includes(n));
    for (const n of statics) names.add(`${cls}.${n}`);
    // bindings.js puts its methods on the embind prototype, one level up from
    // the prototype of the public constructor.
    const proto = m[cls].prototype;
    for (const n of [...own(proto), ...own(Object.getPrototypeOf(proto))]) {
      names.add(statics.includes(n) ? `${cls}.${n}#member` : `${cls}.${n}`);
    }
    names.add(`${cls}.constructor`);
    names.add(`${cls}.delete`);
  }
  for (const n of own(m.ExecutionContext.prototype)) {
    names.add(`ExecutionContext.${n}`);
  }
  names.add('ExecutionContext.constructor');
  names.add('ExecutionContext.delete');
  // Classes, registered vectors and enums, the Emscripten runtime, and module
  // setup.
  const exempt = new Set([
    'Manifold', 'CrossSection', 'ExecutionContext', 'Mesh', 'Mesh64',
    'ManifoldError', 'status', 'jointype', 'addFunction', 'removeFunction',
    'setup', 'initTBB'
  ]);
  for (const n of Object.keys(m)) {
    if (typeof m[n] === 'function' && !n.startsWith('_') &&
        !n.startsWith('Vector') && !exempt.has(n)) {
      names.add(n);
    }
  }
  return names;
}

function thrown(call: () => unknown): any {
  try {
    keep(call());
  } catch (error) {
    return error;
  }
  return undefined;
}

suite('Error checks', () => {
  test('every public method is categorized', () => {
    expect([...publicNames(wasm)].sort())
        .toEqual(Object.keys(CATEGORIES).sort());
  });

  test.each(Object.entries(CATEGORIES))(
      '%s', (_, {cat, call, code, cancel}) => {
        const error = thrown(call);
        if (cat === 'C' || code === 'NoError') {
          expect(error).toBeUndefined();
        } else {
          expect(error).toBeInstanceOf(wasm.ManifoldError);
          expect(error.code).toEqual(code ?? 'NonFiniteVertex');
        }
        if (cat === 'D') {
          const ctx = keep(new wasm.ExecutionContext());
          ctx.cancel();
          const cancelled = thrown(() => cancel!(ctx));
          expect(cancelled).toBeInstanceOf(wasm.ManifoldError);
          expect(cancelled.code).toEqual('Cancelled');
        } else {
          expect(cancel).toBeUndefined();
        }
      });

  test('empty input gives an empty Manifold with no error', () => {
    const {CrossSection, Manifold} = wasm;
    const empty = keep(square().subtract(square()));
    expect(empty.isEmpty()).toBe(true);
    for (const result
             of [empty.extrude(1), empty.revolve(),
                 Manifold.hull([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                 Manifold.hull([keep(empty.extrude(1))])]) {
      keep(result);
      expect(result.status()).toEqual('NoError');
      expect(result.isEmpty()).toBe(true);
    }
  });

  test(
      'static hull checks every Manifold input, with or without points', () => {
        const {Manifold} = wasm;
        for (const args
                 of [[good(), bad()], [bad(), [2, 0, 0]], [[2, 0, 0], bad()]]) {
          expect(thrown(() => Manifold.hull(args)).code)
              .toEqual('NonFiniteVertex');
        }
      });

  test('a failed array result deletes every element', () => {
    const vector = (wasm as any).Vector_manifold.prototype;
    const vectorGet = vector.get;
    const elements: any[] = [];
    vi.spyOn(vector, 'get').mockImplementation(function(this: any, i: number) {
      return keep(elements[elements.push(vectorGet.call(this, i)) - 1]);
    });
    for (const [call, length] of [
             [() => good().split(bad()), 2], [() => bad().split(good()), 2],
             [() => bad().splitByPlane([0, 0, 1]), 2],
             [() => bad().decompose(), 1]] as const) {
      elements.length = 0;
      expect(thrown(call).code).toEqual('NonFiniteVertex');
      expect(elements).toHaveLength(length);
      for (const element of elements) expect(element.isDeleted()).toBe(true);
    }
  });

  test('checks leave the context progress alone', () => {
    const progressAfter = (call: (m: any) => unknown) => {
      const ctx = keep(new wasm.ExecutionContext());
      ctx.cancel();
      thrown(() => call(keep(lazy().withContext(ctx))));
      return ctx.progress();
    };
    expect(progressAfter(m => m.getPrecision()))
        .toEqual(progressAfter(m => m._GetPrecision()));

    // A cancelled evaluation leaves the progress below 1.
    const ctx = keep(new wasm.ExecutionContext());
    ctx.cancel();
    keep(lazy().withContext(ctx)).status();
    const empty = keep(good().subtract(keep(wasm.Manifold.cube(2))));
    expect(empty.isEmpty()).toBe(true);
    const progress = ctx.progress();
    expect(progress).toBeLessThan(1);
    keep(empty.withContext(ctx)).volume();
    expect(ctx.progress()).toEqual(progress);
  });

  test('minGap checks both inputs', () => {
    for (const call
             of [() => bad().minGap(good(), 1),
                () => good().minGap(bad(), 1)]) {
      expect(thrown(call).code).toEqual('NonFiniteVertex');
    }
  });

  test('extrude with center checks before centering', () => {
    const translate =
        vi.spyOn(Object.getPrototypeOf(wasm.Manifold.prototype), '_Translate');
    expect(thrown(() => square().extrude(0, 0, 0, 1, true)).code)
        .toEqual('InvalidConstruction');
    expect(translate).not.toHaveBeenCalled();
    keep(square().extrude(2, 0, 0, 1, true));
    expect(translate).toHaveBeenCalledTimes(1);
  });
});
