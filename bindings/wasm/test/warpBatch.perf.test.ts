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

  test('warp vs warpBatch benchmark', async () => {
    const refineLevels = [0, 1, 2, 3, 4];
    const iters = 80;

    for (const refineLevel of refineLevels) {
      let m = manifoldModule.Manifold.sphere(10, 0);
      m = m.refine(refineLevel);

      m.warp((v: number[]) => {
         v[0] += v[2] * v[2];
       }).boundingBox();

      m.warpBatch((verts: Float64Array, count: number) => {
         for (let i = 0; i < count; i++) {
           const z = verts[i * 3 + 2];
           verts[i * 3 + 0] += z * z;
         }
       }).boundingBox();

      let warpTotal = 0;
      const warpT0 = nowMs();
      for (let k = 0; k < iters; k++) {
        const out = m.warp((v: number[]) => {
          const z = v[2];
          v[0] += z * z;
        });
      }
      const warpT1 = nowMs();
      warpTotal += (warpT1 - warpT0);
      const warpAvg = warpTotal / iters;

      let batchTotal = 0;
      const warpBatchT0 = nowMs();
      for (let k = 0; k < iters; k++) {
        const out = m.warpBatch((verts: Float64Array, count: number) => {
          for (let i = 0; i < count; i++) {
            const z = verts[i * 3 + 2];
            verts[i * 3 + 0] += z * z;
          }
        });
      }
      const warpBatchT1 = nowMs();
      batchTotal += (warpBatchT1 - warpBatchT0);
      const batchAvg = batchTotal / iters;

      console.log(
          `warp():      avg ${warpAvg.toFixed(2)} ms over ${iters} runs`);
      console.log(
          `warpBatch(): avg ${batchAvg.toFixed(2)} ms over ${iters} runs`);
      console.log(`speedup:     ${(warpAvg / batchAvg).toFixed(2)}x`);

      expect(warpAvg).toBeGreaterThan(0);
      expect(batchAvg).toBeGreaterThan(0);
    }
  });
});
