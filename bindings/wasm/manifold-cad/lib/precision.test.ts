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

import {expect, test} from 'vitest';

import * as manifoldCAD from './manifoldCAD.ts';
import {Manifold} from './manifoldCAD.ts';
import {cleanup, evaluate} from './worker.ts';

// Float epsilon, the library default for shapes and float meshes.
const floatEpsilon = 2 ** -23;

test('Each Manifold carries its own precision', () => {
  const cube = Manifold.cube();
  expect(cube.getPrecision()).toEqual(floatEpsilon);
  const coarse = cube.withPrecision(1e-3);
  expect(coarse.getPrecision()).toEqual(1e-3);
  expect(cube.getPrecision()).toEqual(floatEpsilon);
  expect(coarse.add(Manifold.sphere(1)).getPrecision()).toEqual(floatEpsilon);
  expect(Manifold.ofMesh(cube.getMesh(), 1e-3).getPrecision()).toEqual(1e-3);
  expect(new Manifold(cube.getMesh(), 1e-3).getPrecision()).toEqual(1e-3);
  expect(Manifold.ofMesh(cube.getMesh()).getPrecision()).toEqual(floatEpsilon);
});

test('Finer precision is set per Manifold', () => {
  const cube = Manifold.cube();
  expect(cube.withPrecision(1e-12).getPrecision()).toEqual(1e-12);
  expect(Manifold.ofMesh(cube.getMesh(), 1e-12).getPrecision()).toEqual(1e-12);
  expect(new Manifold(cube.getMesh(), 1e-12).getPrecision()).toEqual(1e-12);
});

test('Scripts have no global precision setter or getter', async () => {
  expect(manifoldCAD).not.toHaveProperty('getRelativePrecision');
  for (const name of ['setRelativePrecision', 'getRelativePrecision']) {
    const code = `import {Manifold} from 'manifold-cad/manifoldCAD';
${name}(1e-3);
export default Manifold.sphere(1);`;
    await expect(evaluate(code)).rejects.toThrowError(name);
    cleanup();
  }
});

test('A script sets precision with withPrecision', async () => {
  const code = `import {Manifold} from 'manifold-cad/manifoldCAD';
const coarse = Manifold.cube().withPrecision(1e-3);
if (coarse.getPrecision() !== 1e-3) throw new Error('precision');
export default coarse;`;
  await expect(evaluate(code)).resolves.toBeTruthy();
  cleanup();
});
