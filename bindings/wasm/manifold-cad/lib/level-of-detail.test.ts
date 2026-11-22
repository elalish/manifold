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

import {expect, suite, test} from 'vitest';

import {cleanup, getRelativePrecision, setRelativePrecision} from './level-of-detail.ts';
import {CrossSection, Manifold} from './manifoldCAD.ts';

suite('Relative precision', () => {
  test('applies to Manifolds created after it is set', () => {
    const before = Manifold.cube();
    expect(before.getPrecision()).toEqual(1e-12);
    setRelativePrecision(1e-3);
    expect(getRelativePrecision()).toEqual(1e-3);
    expect(Manifold.cube().getPrecision()).toEqual(1e-3);
    expect(Manifold.sphere(1).getPrecision()).toEqual(1e-3);
    expect(CrossSection.square().extrude(1).getPrecision()).toEqual(1e-3);
    expect(Manifold.extrude([[0, 0], [1, 0], [0, 1]], 1).getPrecision())
        .toEqual(1e-3);
    expect(Manifold.revolve([[1, 0], [2, 0], [1, 1]]).getPrecision())
        .toEqual(1e-3);
    expect(Manifold.ofMesh(before.getMesh()).getPrecision()).toEqual(1e-3);
    const constructed = new Manifold(before.getMesh());
    expect(constructed.getPrecision()).toEqual(1e-3);
    expect(constructed instanceof Manifold).toBe(true);
    expect(before.getPrecision()).toEqual(1e-12);
    expect(Manifold.hull([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
               .getPrecision())
        .toEqual(1e-3);
    expect(Manifold.hull(before, [2, 0, 0]).getPrecision()).toEqual(1e-12);
    setRelativePrecision(Infinity);
    expect(getRelativePrecision()).toEqual(1e-12);
    cleanup();
    expect(getRelativePrecision()).toEqual(1e-12);
    expect(Manifold.cube().getPrecision()).toEqual(1e-12);
  });
});
