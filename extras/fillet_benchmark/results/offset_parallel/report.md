# Boolean2 offset parallelism

Two round offsets for fillet-style opening/closing, 32 segments per circle. Release with MANIFOLD_PAR=ON, 1/2/4/8 TBB threads, affinity to eight physical P cores (0,2,4,6,8,10,12,14). Five timed calls per large case after warmup; table values are medians.

Before: existing parallel cleanup, sequential contour generation. After: additionally parallel contour generation above 1,024 input vertices when multiple contours exist. Small inputs and builds with MANIFOLD_PAR=OFF retain the sequential generation path.

**The added contour-parallel production change was not retained:** it produced no meaningful end-to-end speedup in these measurements. The tested source and determinism test are preserved in `contour_parallel.patch`. The existing MANIFOLD_PAR backend is the recommended way to obtain the measured threading benefit.

| Case | Input contours | Before 1 thread, ms | Before 8 threads, ms | After 1 thread, ms | After 8 threads, ms | After 1→8 speedup | Added change speedup at 8 threads |
|---|---:|---:|---:|---:|---:|---:|---:|
| zebra/Zebra/0.500000 | 217 | 324.350 | 266.128 | 325.193 | 264.978 | 1.227× | 1.004× |
| zebra/Zebra/-0.500000 | 217 | 351.434 | 288.514 | 353.756 | 289.617 | 1.221× | 0.996× |
| zebra3/Zebra3/0.500000 | 218 | 332.415 | 271.290 | 329.246 | 270.087 | 1.219× | 1.004× |
| zebra3/Zebra3/-0.500000 | 218 | 357.620 | 292.201 | 357.021 | 295.125 | 1.210× | 0.990× |
| disjoint/4096/0.500000 | 4096 | 230.244 | 197.264 | 229.949 | 198.463 | 1.159× | 0.994× |
| disjoint/4096/-0.500000 | 4096 | 4327.896 | 4035.802 | 4313.403 | 4039.309 | 1.068× | 0.999× |

Speedup above 1 means faster. The before/after columns distinguish existing cleanup parallelism from the added contour loop. Separate runs remain subject to frequency and scheduling variation; small changes should not be interpreted as decisive speedups.

## Thread scaling after the change

| Case | 1 thread, ms | 2 threads, ms | 4 threads, ms | 8 threads, ms |
|---|---:|---:|---:|---:|
| zebra/Zebra/0.500000 | 325.193 | 288.868 | 274.270 | 264.978 |
| zebra/Zebra/-0.500000 | 353.756 | 313.285 | 294.380 | 289.617 |
| zebra3/Zebra3/0.500000 | 329.246 | 293.474 | 277.275 | 270.087 |
| zebra3/Zebra3/-0.500000 | 357.021 | 320.700 | 301.498 | 295.125 |
| disjoint/4096/0.500000 | 229.949 | 212.057 | 202.436 | 198.463 |
| disjoint/4096/-0.500000 | 4313.403 | 4161.659 | 4075.629 | 4039.309 |

## Phase profile, 8 threads

Profiling is a separate warmup invocation with phase clocks enabled, not the median timed call. Phase percentages use cleanup time, excluding contour generation and final polygon tracing.

| Case | Cleanup, ms | Sweep, ms | Sweep / cleanup |
|---|---:|---:|---:|
| zebra/Zebra/0.500000 | 262.050 | 240.735 | 91.9% |
| zebra/Zebra/-0.500000 | 283.158 | 261.343 | 92.3% |
| zebra3/Zebra3/0.500000 | 266.680 | 245.035 | 91.9% |
| zebra3/Zebra3/-0.500000 | 288.613 | 266.221 | 92.2% |
| disjoint/4096/0.500000 | 186.638 | 169.348 | 90.7% |
| disjoint/4096/-0.500000 | 4026.301 | 3951.251 | 98.1% |

Every warmup and timed result was compared coordinate-for-coordinate and contour-for-contour with the same case's one-thread result. Before/after output hashes also match for every case/thread pair. This confirms deterministic scheduling for the tested inputs; existing geometry differences from Clipper2 are not repaired.

The full verification-only run checked 3892 outputs across 973 cases and 1/2/4/8 threads. Its single profiled invocations are not performance samples.
