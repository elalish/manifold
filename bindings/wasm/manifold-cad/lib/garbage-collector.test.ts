// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Module, {type CrossSection, type Manifold, type ManifoldToplevel} from 'manifold-3d';
import {afterEach, beforeAll, expect, test} from 'vitest';

import {show} from './debug.ts';
import {cleanup, garbageCollectFunction, garbageCollectManifold} from './garbage-collector.ts';
import {setMaterial} from './material.ts';

type Geometry = Manifold|CrossSection;
type Handle = Geometry&{isDeleted(): boolean};

let module: ManifoldToplevel;
const handles: Handle[] = [];

function remember<T extends Geometry>(object: T): T {
  handles.push(object as T & Handle);
  return object;
}

function expectCollected() {
  cleanup();
  for (const handle of handles) expect(handle.isDeleted()).toBe(true);
  expect(() => cleanup()).not.toThrow();
}

beforeAll(async () => {
  module = await Module();
  module.setup();
  garbageCollectManifold(module);
});

afterEach(() => {
  cleanup();
  // Also release uncollected handles when a regression assertion fails.
  for (const handle of handles) {
    if (!handle.isDeleted()) handle.delete();
  }
  handles.length = 0;
});

test('only lists callable WASM methods for automatic cleanup', async () => {
  const native = await Module();
  native.setup();
  const verify = (target: object, label: string) => new Proxy(target, {
    get(target, key, receiver) {
      const value = Reflect.get(target, key, receiver);
      expect.soft(value, `${label}.${String(key)}`).toBeTypeOf('function');
      return value;
    }
  });
  const checked =
      Object.fromEntries((['Manifold', 'CrossSection'] as const).map(name => {
        const constructor = native[name];
        const statics = verify(constructor, name);
        const members = verify(constructor.prototype, `${name}.prototype`);
        // Keep the real owner in the prototype chain so interception still
        // installs each wrapper on the method's defining object.
        return [
          name, new Proxy(Object.create(constructor), {
            get(_target, key) {
              return key === 'prototype' ? members : Reflect.get(statics, key);
            }
          })
        ];
      }));
  garbageCollectManifold({...native, ...checked});
});

test(
    'collects chained Manifold member results and preserves their receiver',
    () => {
      const cube = remember(module.Manifold.cube());
      const translated = remember(cube.translate([2, 3, 4]));
      const scaled = remember(translated.scale(2));
      expect(scaled.boundingBox()).toEqual({min: [4, 6, 8], max: [6, 8, 10]});
      expectCollected();
    });

test('collects a Manifold built with its constructor', () => {
  const cube = remember(module.Manifold.cube());
  const manifold = remember(new module.Manifold(cube.getMesh()));
  expect(manifold).toBeInstanceOf(module.Manifold);
  expect(manifold.translate([1, 0, 0]).volume()).toBeCloseTo(1);
  expectCollected();
});

test('collects a CrossSection built with its constructor', () => {
  const triangle = remember(new module.CrossSection([[0, 0], [1, 0], [0, 1]]));
  expect(triangle).toBeInstanceOf(module.CrossSection);
  expect(triangle.translate([1, 0]).area()).toBeCloseTo(0.5);
  expectCollected();
});

test('collects constructors called without new', () => {
  const cube = remember(module.Manifold.cube());
  const manifold = remember((module.Manifold as any)(cube.getMesh()));
  const triangle =
      remember((module.CrossSection as any)([[0, 0], [1, 0], [0, 1]]));
  expect(manifold.volume()).toBeCloseTo(1);
  expect(triangle.area()).toBeCloseTo(0.5);
  expectCollected();
});

test('collects chained CrossSection member results', () => {
  const square = remember(module.CrossSection.square());
  const translated = remember(square.translate([2, 3]));
  const scaled = remember(translated.scale(2));
  expect(scaled.bounds()).toEqual({min: [4, 6], max: [6, 8]});
  expectCollected();
});

test('collects batch warp and Minkowski results', () => {
  const cube = remember(module.Manifold.cube(2));
  const tool = remember(module.Manifold.cube());
  const warped = remember(cube.warpBatch(vertices => {
    for (let i = 0; i < vertices.length; i += 3) vertices[i] += 1;
  }));
  const sum = remember(cube.minkowskiSum(tool));
  const difference = remember(cube.minkowskiDifference(tool));
  expect(warped.boundingBox().min[0]).toBe(1);
  expect(sum.volume()).toBeCloseTo(27);
  expect(difference.volume()).toBeCloseTo(1);
  expectCollected();
});

test('collects a Manifold carrying an execution context', () => {
  const context = new module.ExecutionContext();
  try {
    const cube = remember(module.Manifold.cube());
    const contextual = remember(cube.withContext(context));
    expect(contextual.volume()).toBeCloseTo(1);
    expectCollected();
  } finally {
    context.delete();
  }
});

test('collects CrossSection warp, extrusion and revolution results', () => {
  const square = remember(module.CrossSection.square());
  const warped = remember(square.warp(vertex => {
    vertex[0] += 2;
  }));
  const batch = remember(square.warpBatch(vertices => {
    for (let i = 0; i < vertices.length; i += 2) vertices[i] += 2;
  }));
  expect(batch.bounds().min[0]).toBe(2);
  const extruded = remember(warped.extrude(3));
  const revolved = remember(warped.revolve(8));
  expect(warped.bounds().min[0]).toBe(2);
  expect(extruded.volume()).toBeCloseTo(3);
  expect(revolved.volume()).toBeGreaterThan(0);
  expectCollected();
});

test('collects every handle returned by split and decompose', () => {
  const cube = remember(module.Manifold.cube(2));
  const parts = cube.splitByPlane([1, 0, 0], 1).map(remember);
  const components = parts[0].decompose().map(remember);
  const square = remember(module.CrossSection.square());
  const contours = square.decompose().map(remember);
  expect(parts).toHaveLength(2);
  expect(components).toHaveLength(1);
  expect(contours).toHaveLength(1);
  expectCollected();
});

test('collects slice and projection results', () => {
  const cube = remember(module.Manifold.cube(2));
  const slice = remember(cube.slice(1));
  const projection = remember(cube.project());
  expect(slice.area()).toBeCloseTo(4);
  expect(projection.area()).toBeCloseTo(4);
  expectCollected();
});

test('deletes results of nested factory and member calls only once', () => {
  const square = remember(module.CrossSection.square());
  const centered = remember(square.extrude(2, 0, 0, [1, 1], true));
  const staticExtrusion =
      remember(module.Manifold.extrude(square, 2, 0, 0, [1, 1], true));
  const ring = remember(square.translate([2, 0]));
  const staticRevolution = remember(module.Manifold.revolve(ring, 8));
  expect(centered.boundingBox().min[2]).toBe(-1);
  expect(staticExtrusion.volume()).toBeCloseTo(2);
  expect(staticRevolution.volume()).toBeGreaterThan(0);
  expectCollected();
});

test('deletes debug and material results only once', () => {
  const cube = remember(module.Manifold.cube());
  const shown = remember(garbageCollectFunction(show)(cube));
  const colored = remember(
      garbageCollectFunction(setMaterial)(shown, {baseColorFactor: [1, 0, 0]}));
  expect(colored.volume()).toBeCloseTo(1);
  expectCollected();
});

test('allows handles to be deleted explicitly before cleanup', () => {
  const cube = remember(module.Manifold.cube());
  const translated = remember(cube.translate([1, 0, 0]));
  const scaled = remember(cube.scale(2));
  translated.delete();
  cube.delete();
  expect(scaled.volume()).toBeCloseTo(8);
  expectCollected();
});

test('collects each handle once after repeated initialization', () => {
  garbageCollectManifold(module);
  const cube = remember(module.Manifold.cube());
  const translated = remember(cube.translate([1, 0, 0]));
  expect(translated.volume()).toBeCloseTo(1);
  expectCollected();
});
