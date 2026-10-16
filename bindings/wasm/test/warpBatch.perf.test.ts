import {beforeAll, expect, suite, test} from 'vitest';

import Module, {type ManifoldToplevel} from '../manifold';

let manifoldModule: ManifoldToplevel;

beforeAll(async () => {
  manifoldModule = await Module();
  manifoldModule.setup();
});

function nowMs(): number {
  return Number(process.hrtime.bigint()) / 1e6;
}

suite('warpBatch tests', () => {
  test('actually changes geometry', async () => {
    const m = manifoldModule.Manifold.cube(2, true);

    const before = m.boundingBox();

    const dx = 10;
    const out = m.warpBatch((verts: Float64Array, count: number) => {
      for (let i = 0; i < count; i++) {
        verts[i * 3 + 0] += dx;
      }
    });

    const after = out.boundingBox();

    // bbox x should shift by dx
    expect(after.min[0]).toBe(before.min[0] + dx);
    expect(after.max[0]).toBe(before.max[0] + dx);

    // y/z should remain unchanged
    expect(after.min[1]).toBe(before.min[1]);
    expect(after.max[1]).toBe(before.max[1]);
    expect(after.min[2]).toBe(before.min[2]);
    expect(after.max[2]).toBe(before.max[2]);

    // original manifold should be unchanged
    const origAfter = m.boundingBox();
    expect(origAfter.min[0]).toBe(before.min[0]);
    expect(origAfter.max[0]).toBe(before.max[0]);
  });
});
