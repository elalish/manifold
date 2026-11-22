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

import {afterEach, expect, test} from 'vitest';

import * as levelOfDetail from './level-of-detail.ts';
import {CrossSection, Manifold} from './manifoldCAD.ts';

afterEach(() => levelOfDetail.cleanup());

test('Run quality reaches every circular constructor', () => {
  // 360 / 30 = 12 segments for a radius of 5, instead of the default 32.
  levelOfDetail.setMinCircularAngle(30);
  expect(levelOfDetail.getCircularSegments(5)).toBe(12);
  expect(CrossSection.circle(5).numVert()).toBe(12);
  expect(Manifold.cylinder(1, 5).numVert()).toBe(24);
  expect(Manifold.sphere(5).numVert()).toBe(Manifold.sphere(5, 12).numVert());

  const ring = CrossSection.square([1, 1]).translate([4.5, 0]);
  expect(ring.revolve().numVert()).toBe(ring.revolve(12).numVert());
  expect(Manifold.revolve(ring).numVert()).toBe(ring.revolve(12).numVert());

  const tiny = CrossSection.square([1e-3, 1e-3], true);
  expect(tiny.offset(5).numVert())
      .toBe(tiny.offset(5, 'Round', 2, 12).numVert());

  // Fields the script leaves out or undefined keep the run's values.
  expect(CrossSection.circle(5, {minAngle: undefined}).numVert()).toBe(12);
  expect(CrossSection.circle(5, {minEdgeLength: 1}).numVert()).toBe(12);

  // 2 * pi * 5 / 5 rounds to 8 segments, fewer than the angle allows.
  levelOfDetail.setMinCircularEdgeLength(5);
  expect(CrossSection.circle(5).numVert()).toBe(8);
});

test('Counts a constructor ignores keep the run segments', () => {
  levelOfDetail.setCircularSegments(12);
  for (const n of [0, 1, 2]) {
    expect(CrossSection.circle(5, n).numVert()).toBe(12);
    expect(Manifold.cylinder(1, 5, -1, n).numVert()).toBe(24);
  }
  expect(CrossSection.circle(5, 3).numVert()).toBe(3);

  const ring = CrossSection.square([1, 1]).translate([4.5, 0]);
  expect(ring.revolve(2).numVert()).toBe(ring.revolve(12).numVert());
  expect(Manifold.revolve(ring, 2).numVert()).toBe(ring.revolve(12).numVert());
  const tiny = CrossSection.square([1e-3, 1e-3], true);
  expect(tiny.offset(5, 'Round', 2, 2).numVert())
      .toBe(tiny.offset(5, 'Round', 2, 12).numVert());

  // Sphere takes any positive count, rounding it up to a multiple of four.
  expect(Manifold.sphere(5, 1).numTri()).toBe(8);
  expect(Manifold.sphere(5, 0).numTri()).toBe(Manifold.sphere(5, 12).numTri());
});

test('Run segments count per full circle, as in 3.5', () => {
  // Expected counts come from 3.5 with Quality::SetCircularSegments.
  const ring = CrossSection.square([1, 1]).translate([4.5, 0]);
  levelOfDetail.setCircularSegments(30);
  expect(ring.revolve(undefined, 90).numVert()).toBe(32);
  expect(Manifold.revolve(ring, undefined, 200).numVert()).toBe(68);
  expect(ring.revolve().numVert()).toBe(120);
  expect(ring.revolve(0, -90).numVert()).toBe(32);
  expect(ring.revolve(0, 720).numVert()).toBe(120);
  levelOfDetail.setCircularSegments(28);
  expect(Manifold.sphere(5).numTri()).toBe(392);
  levelOfDetail.setCircularSegments(13);
  expect(ring.revolve(0, 200).numVert()).toBe(32);
  // One division, as 3 * 200 / 360 rounds down to 1.
  levelOfDetail.setCircularSegments(3);
  expect(ring.revolve(0, 200).numVert()).toBe(8);
});

test('A sphere rounds the run segments up to a multiple of four', () => {
  // 3.5 rounded a set count down instead, giving 72 triangles here.
  levelOfDetail.setCircularSegments(13);
  expect(Manifold.sphere(5).numTri()).toBe(128);
});

test('Counts convert to an int first, as in 3.5', () => {
  const ring = CrossSection.square([1, 1]).translate([4.5, 0]);
  // 30.9 is stored as 30, so 200 degrees gets 16 divisions.
  levelOfDetail.setCircularSegments(30.9);
  expect(ring.revolve(0, 200).numVert()).toBe(68);
  // 2^32 + 1 converts to 1, which leaves the run's 30.
  expect(ring.revolve(2 ** 32 + 1, 90).numVert()).toBe(32);
  expect(CrossSection.circle(5, '12' as any).numVert()).toBe(12);
});

test('Cleanup restores the default quality', () => {
  levelOfDetail.setCircularSegments(12);
  levelOfDetail.setMinCircularAngle(30);
  levelOfDetail.setMinCircularEdgeLength(5);
  levelOfDetail.cleanup();
  expect(levelOfDetail.getCircularSegments(5)).toBe(32);
  expect(CrossSection.circle(5).numVert()).toBe(32);
});
