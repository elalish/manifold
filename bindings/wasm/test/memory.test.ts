import {beforeAll, expect, suite, test, vi} from 'vitest';

import Module, {type ManifoldToplevel} from '../manifold';

let manifoldModule: ManifoldToplevel;

beforeAll(async () => {
  manifoldModule = await Module();
  manifoldModule.setup();
});

suite('Called in a loop, the bindings', () => {
  test('do not grow the WASM memory', () => {
    const {CrossSection, Manifold, triangulate} = manifoldModule;
    const circle = CrossSection.circle(10, 100);
    const ring = circle.translate(20, 0);
    const polygons = circle.toPolygons();
    const ringPolygons = ring.toPolygons();
    const cube = Manifold.cube();
    const errored = cube.translate([NaN, 0, 0]);
    const throws = (call: () => unknown) => {
      try {
        call();
      } catch (e) {
        return;
      }
      throw new Error('expected a throw');
    };
    // Large enough that leaking its polygons on a throw grows the memory.
    const big = CrossSection.circle(10, 10000);
    const bigPolygons = big.toPolygons();
    const bigPoints = bigPolygons[0].map(([x, y]) => [x, y, 0]);
    const evaluate = () => {
      throws(() => big.extrude(0));
      throws(() => big.extrude(0, 0, 0, 1, true));
      throws(() => Manifold.extrude(bigPolygons, 0));
      throws(() => errored.decompose());
      throws(() => errored.split(cube));
      throws(() => Manifold.hull([bigPoints, errored]));
      circle.extrude(1).delete();
      circle.extrude(1, 10, 90, [0.5, 0.5], true).delete();
      ring.revolve(3).delete();
      circle.toPolygons();
      new CrossSection(polygons).delete();
      CrossSection.evenOdd(polygons).delete();
      CrossSection.nonZero(polygons).delete();
      circle.add(polygons).delete();
      circle.subtract(polygons).delete();
      circle.intersect(polygons).delete();
      CrossSection.union([circle, polygons]).delete();
      Manifold.extrude(polygons, 1).delete();
      Manifold.revolve(ringPolygons, 3).delete();
      triangulate(polygons);
    };
    evaluate();
    const grow = vi.spyOn(WebAssembly.Memory.prototype, 'grow');
    for (let i = 0; i < 100; i++) evaluate();
    expect(grow).not.toHaveBeenCalled();
  });
});
