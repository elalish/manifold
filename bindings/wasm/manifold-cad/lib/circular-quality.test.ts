// Copyright 2025 The Manifold Authors.
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
import {CrossSection, forDivisions, Manifold, segmentsFor} from './manifoldCAD.ts';
import {cleanup, evaluate} from './worker.ts';

test('Each circular constructor takes a quality', () => {
  // 360 / 30 = 12 segments for a radius of 5, instead of the default 32.
  const q = {minAngle: 30};
  expect(segmentsFor(q, 5)).toBe(12);
  expect(segmentsFor({}, 5)).toBe(32);
  expect(CrossSection.circle(5, q).numVert()).toBe(12);
  expect(Manifold.cylinder(1, 5, -1, q).numVert()).toBe(24);
  expect(Manifold.sphere(5, q).numTri()).toBe(Manifold.sphere(5, 12).numTri());

  const ring = CrossSection.square([1, 1]).translate([4.5, 0]);
  expect(ring.revolve(q).numVert()).toBe(ring.revolve(12).numVert());
  expect(Manifold.revolve(ring, q).numVert()).toBe(ring.revolve(12).numVert());
  // A partial revolve of this ring has 4 vertices per slice.
  expect(ring.revolve(forDivisions(5, 90), 90).numVert()).toBe(4 * 6);

  const tiny = CrossSection.square([1e-3, 1e-3], true);
  expect(tiny.offset(5, 'Round', 2, q).numVert())
      .toBe(tiny.offset(5, 'Round', 2, 12).numVert());
});

test('The circular setters and getters are gone', async () => {
  for (const name
           of ['getCircularSegments', 'getMinCircularAngle',
               'getMinCircularEdgeLength']) {
    expect(manifoldCAD).not.toHaveProperty(name);
  }
  for (const name
           of ['setCircularSegments', 'setMinCircularAngle',
               'setMinCircularEdgeLength', 'resetToCircularDefaults']) {
    const code = `import {Manifold} from 'manifold-cad/manifoldCAD';
${name}(2);
export default Manifold.sphere(1);`;
    await expect(evaluate(code)).rejects.toThrowError(name);
    cleanup();
  }
});
