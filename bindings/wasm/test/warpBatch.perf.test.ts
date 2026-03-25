import {join} from 'node:path';
import {expect, test} from 'vitest';

import Module, {type ManifoldToplevel} from '../manifold';

function nowMs(): number {
  return Number(process.hrtime.bigint()) / 1e6;
}

test('warp vs warpBatch benchmark', async () => {
  const wasmPath = join(process.cwd(), 'manifold.wasm');

  const originalFetch = (globalThis as any).fetch;
  (globalThis as any).fetch = undefined;

  let wasm: ManifoldToplevel;
  try {
    wasm = await Module({locateFile: () => wasmPath});
  } finally {
    (globalThis as any).fetch = originalFetch;
  }

  wasm.setup();
  const {Manifold} = wasm;

  const refineLevels = [0, 1, 2, 3, 4];
  const iters = 80;

  for (const refineLevel of refineLevels) {
    let m = Manifold.sphere(10, 0);
    m = m.refine(refineLevel);

    const n = m.numVert();
    console.log(`mesh: sphere refine(${refineLevel}), numVert=${n}`);

    m.warp((v: number[]) => {
      v[0] += v[2] * v[2];
    });
    m.warpBatch((verts: Float64Array, count: number) => {
      for (let i = 0; i < count; i++) {
        const z = verts[i * 3 + 2];
        verts[i * 3 + 0] += z * z;
      }
    });

    let warpTotal = 0;
    for (let k = 0; k < iters; k++) {
      const t0 = nowMs();
      const out = m.warp((v: number[]) => {
        const z = v[2];
        v[0] += z * z;
      });
      const t1 = nowMs();
      out.volume();
      warpTotal += (t1 - t0);
    }
    const warpAvg = warpTotal / iters;

    let batchTotal = 0;
    for (let k = 0; k < iters; k++) {
      const t0 = nowMs();
      const out = m.warpBatch((verts: Float64Array, count: number) => {
        for (let i = 0; i < count; i++) {
          const z = verts[i * 3 + 2];
          verts[i * 3 + 0] += z * z;
        }
      });
      const t1 = nowMs();
      out.volume();
      batchTotal += (t1 - t0);
    }
    const batchAvg = batchTotal / iters;

    console.log(`warp():      avg ${warpAvg.toFixed(2)} ms over ${iters} runs`);
    console.log(
        `warpBatch(): avg ${batchAvg.toFixed(2)} ms over ${iters} runs`);
    console.log(`speedup:     ${(warpAvg / batchAvg).toFixed(2)}x`);

    expect(warpAvg).toBeGreaterThan(0);
    expect(batchAvg).toBeGreaterThan(0);
  }
});
