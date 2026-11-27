import {afterEach, beforeEach, expect, test, vi} from 'vitest';

import Module, {type ManifoldToplevel} from '../manifold';

let module: ManifoldToplevel;
const handles: any[] = [];

function keep<T>(handle: T): T {
  handles.push(handle);
  return handle;
}

beforeEach(async () => {
  module = await Module();
  module.setup();
});

afterEach(() => {
  vi.restoreAllMocks();
  for (const handle of handles) {
    if (!handle.isDeleted()) handle.delete();
  }
  handles.length = 0;
});

function expectThrownValue(call: () => unknown, expected: unknown) {
  let threw = false;
  let actual: unknown;
  try {
    call();
  } catch (error) {
    threw = true;
    actual = error;
  }
  expect(threw).toBe(true);
  expect(actual).toBe(expected);
}

interface MutationCase {
  name: string;
  nativeMethod: string;
  create: () => any;
  call: (input: any, callback: (values: any) => void) => any;
}

const mutations: MutationCase[] = [
  {
    name: 'CrossSection.warp',
    nativeMethod: '_Warp',
    create: () => module.CrossSection.square(),
    call: (input, callback) => input.warp(callback)
  },
  {
    name: 'CrossSection.warpBatch',
    nativeMethod: '_WarpBatch',
    create: () => module.CrossSection.square(),
    call: (input, callback) => input.warpBatch(callback)
  },
  {
    name: 'Manifold.warp',
    nativeMethod: '_Warp',
    create: () => module.Manifold.cube(),
    call: (input, callback) => input.warp(callback)
  },
  {
    name: 'Manifold.warpBatch',
    nativeMethod: '_WarpBatch',
    create: () => module.Manifold.cube(),
    call: (input, callback) => input.warpBatch(callback)
  },
  {
    name: 'Manifold.setProperties',
    nativeMethod: '_SetProperties',
    create: () => module.Manifold.cube(),
    call: (input, callback) => input.setProperties(1, callback)
  }
];

function observeNativeResults(input: any, method: string) {
  const results: Array<{handle: any, measure: number, status: string}> = [];
  const native = input[method];
  input[method] = (...args: any[]) => {
    const handle = keep(native.apply(input, args));
    results.push({
      handle,
      measure: handle.volume ? handle.volume() : handle.area(),
      status: handle.status ? handle.status() : 'NoError'
    });
    return handle;
  };
  return results;
}

test.each(mutations)(
    '$name releases native results and callback slots after partial writes',
    ({create, call, nativeMethod}) => {
      const input = keep(create());
      const before = input.getMesh ? input.getMesh() : input.toPolygons();
      const results = observeNativeResults(input, nativeMethod);
      const failure = new Error('mutation failed');
      let calls = 0;
      const callback = (values: any) => {
        calls++;
        values.fill(NaN);
        throw failure;
      };
      // Warm the free-list before measuring table growth.
      expectThrownValue(() => call(input, callback), failure);
      const grow = vi.spyOn(WebAssembly.Table.prototype, 'grow');
      for (let i = 0; i < 3; i++) {
        expectThrownValue(() => call(input, callback), failure);
      }
      expect(grow).not.toHaveBeenCalled();
      expect(calls).toBe(4);
      // The native operation must return normally, with valid geometry, so
      // its C++ locals can be destroyed before the original value is rethrown.
      expect(results).toHaveLength(4);
      for (const {handle, measure, status} of results) {
        expect(status).toBe('NoError');
        expect(measure).toBeCloseTo(1);
        expect(handle.isDeleted()).toBe(true);
      }
      expect(input.isDeleted()).toBe(false);
      expect(input.volume ? input.volume() : input.area()).toBeCloseTo(1);
      if (input.getMesh) {
        expect(input.getMesh().vertProperties).toEqual(before.vertProperties);
        expect(input.getMesh().triVerts).toEqual(before.triVerts);
      } else {
        expect(input.toPolygons()).toEqual(before);
      }
      const reused = keep(call(input, () => {}));
      expect(reused.volume ? reused.volume() : reused.area()).toBeCloseTo(1);
    });

const bounds = {
  min: [-1, -1, -1],
  max: [1, 1, 1]
};
const ball = (p: number[]) => 0.5 - Math.hypot(...p);

const levelSets = [
  {
    name: 'Manifold.levelSet',
    owner: () => module as any,
    call: (sdf: (p: number[]) => number) =>
        module.Manifold.levelSet(sdf, bounds, 0.5)
  },
  {
    name: 'ExecutionContext.levelSet',
    owner: () => keep(new module.ExecutionContext()),
    call: (sdf: (p: number[]) => number, ctx: any) =>
        ctx.levelSet(sdf, bounds, 0.5)
  }
];

test.each(levelSets)(
    '$name releases native results and callback slots after a throw',
    ({owner, call}) => {
      const target = owner();
      const results = observeNativeResults(target, '_LevelSet');
      const failure = new Error('sdf failed');
      let calls = 0;
      const sdf = (p: number[]) => {
        if (++calls === 10) throw failure;
        return ball(p);
      };
      // Warm the free-list before measuring table growth.
      expectThrownValue(() => call(sdf, target), failure);
      const grow = vi.spyOn(WebAssembly.Table.prototype, 'grow');
      for (let i = 0; i < 3; i++) {
        calls = 0;
        expectThrownValue(() => call(sdf, target), failure);
        // The SDF is not called again after it throws.
        expect(calls).toBe(10);
      }
      expect(grow).not.toHaveBeenCalled();
      // The native operation must return normally so its C++ locals are
      // destroyed before the original value is rethrown.
      expect(results).toHaveLength(4);
      for (const {handle} of results) expect(handle.isDeleted()).toBe(true);
      const reused = keep(call(ball, target));
      expect(reused.volume()).toBeGreaterThan(0);
    });

test.each([undefined, null, 0, 'callback failed'])(
    'preserves the thrown value %j after rolling back a batch mutation',
    failure => {
      const input = keep(module.Manifold.cube());
      const results = observeNativeResults(input, '_WarpBatch');
      expectThrownValue(
          () => input.warpBatch(vertices => {
            vertices.fill(Infinity);
            throw failure;
          }),
          failure);
      expect(results).toHaveLength(1);
      expect(results[0].status).toBe('NoError');
      expect(results[0].measure).toBeCloseTo(1);
      expect(results[0].handle.isDeleted()).toBe(true);
      expect(input.volume()).toBeCloseTo(1);
    });

test.each(['warp', 'warpBatch'] as const)(
    '%s deletes a result rejected by the finite-coordinate check', method => {
      const input = keep(module.Manifold.cube());
      const results = observeNativeResults(
          input, method === 'warp' ? '_Warp' : '_WarpBatch');
      expect(() => input[method](vertices => {
               vertices[0] = NaN;
             }))
          .toThrow();
      expect(results).toHaveLength(1);
      expect(results[0].status).toBe('NonFiniteVertex');
      expect(results[0].handle.isDeleted()).toBe(true);
      expect(input.volume()).toBeCloseTo(1);
    });

test.each([
  ['Mesh', Float32Array],
  ['Mesh64', Float64Array],
] as const)(
    'deletes a native Manifold rejected by the %s constructor',
    (meshType, Float) => {
      const prototype = Object.getPrototypeOf(module.Manifold.prototype);
      const status = prototype.status;
      const rejected: any[] = [];
      vi.spyOn(prototype, 'status').mockImplementation(function(this: any) {
        const result = status.call(this);
        if (result !== 'NoError') rejected.push(keep(this));
        return result;
      });
      const mesh = new module[meshType]({
        numProp: 3,
        vertProperties: new Float([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        triVerts: new Uint32Array([0, 1, 9])
      });
      expect(() => new module.Manifold(mesh)).toThrow();
      expect(rejected).toHaveLength(1);
      expect(rejected[0].isDeleted()).toBe(true);
    });

test('preserves successful scalar and batch geometry mutations', () => {
  const input = keep(module.Manifold.cube());
  const scalar = keep(input.warp(vertex => {
    vertex[0] += 2;
  }));
  const batch = keep(input.warpBatch(vertices => {
    for (let i = 0; i < vertices.length; i += 3) vertices[i] += 2;
  }));
  const expected = keep(input.translate([2, 0, 0]));
  for (const result of [scalar, batch]) {
    expect(result.getMesh().vertProperties)
        .toEqual(expected.getMesh().vertProperties);
    expect(result.getMesh().triVerts).toEqual(expected.getMesh().triVerts);
    expect(result.volume()).toBeCloseTo(1);
  }
  const square = keep(module.CrossSection.square());
  const warped = keep(square.warp(vertex => {
    vertex[0] += 2;
  }));
  const shifted = keep(square.translate([2, 0]));
  expect(keep(warped.subtract(shifted)).area()).toBe(0);
  expect(keep(shifted.subtract(warped)).area()).toBe(0);
  expect(square.bounds()).toEqual({min: [0, 0], max: [1, 1]});
});

test('CrossSection warpBatch matches warp with the same function', () => {
  const circle = keep(module.CrossSection.circle(1, 32));
  const input = keep(circle.translate([3, 0]));
  const before = input.toPolygons();
  const numVert = input.numVert();
  const bend = (x: number, y: number) => [x * Math.cos(y), y + x * x / 4];
  const scalar = keep(input.warp(vertex => {
    [vertex[0], vertex[1]] = bend(vertex[0], vertex[1]);
  }));
  const batch = keep(input.warpBatch((vertices, count) => {
    expect(count).toBe(numVert);
    expect(vertices.length).toBe(count * 2);
    for (let i = 0; i < vertices.length; i += 2) {
      [vertices[i], vertices[i + 1]] = bend(vertices[i], vertices[i + 1]);
    }
  }));
  expect(batch.toPolygons()).toEqual(scalar.toPolygons());
  expect(batch.area()).toBeGreaterThan(0);
  expect(input.toPolygons()).toEqual(before);
});

test('preserves successful property mutations', () => {
  const input = keep(module.Manifold.cube());
  const result = keep(input.setProperties(1, (properties, position) => {
    properties[0] = position[0] + 2 * position[1];
  }));
  const mesh = result.getMesh();
  expect(mesh.numProp).toBe(4);
  for (let i = 0; i < mesh.vertProperties.length; i += mesh.numProp) {
    const p = mesh.vertProperties;
    expect(p[i + 3]).toBe(p[i] + 2 * p[i + 1]);
  }
  expect(input.numProp()).toBe(0);
  expect(result.volume()).toBeCloseTo(input.volume());
});

test(
    'preserves batch writes across native allocations inside the callback',
    () => {
      const input = keep(module.Manifold.cube());
      const allocation = keep(new (module as any).Vector_vec3());
      const grow = vi.spyOn(WebAssembly.Memory.prototype, 'grow');
      const result = keep(input.warpBatch((vertices, count) => {
        vertices[0] += 2;
        // Use the native allocator so Emscripten updates its heap views too.
        allocation.resize(2 * 1024 * 1024, {x: 0, y: 0, z: 0});
        expect(grow).toHaveBeenCalled();
        expect(vertices.length).toBe(count * 3);
        for (let i = 1; i < count; i++) vertices[i * 3] += 2;
      }));
      const expected = keep(input.translate([2, 0, 0]));
      expect(result.getMesh().vertProperties)
          .toEqual(expected.getMesh().vertProperties);
      expect(result.getMesh().triVerts).toEqual(expected.getMesh().triVerts);
      expect(input.volume()).toBeCloseTo(1);
    });
