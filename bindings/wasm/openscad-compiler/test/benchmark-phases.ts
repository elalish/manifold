import fs from 'fs';
import {execFileSync} from 'node:child_process';
import {mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {pathToFileURL} from 'node:url';
import path from 'path';

import {compileConsumer} from '../core/orchestrate.js';
import {getOpenSCADLibraryPaths} from '../core/resolver.js';

const CWD = process.cwd();
const argv = process.argv.slice(2);
const RUNS = parseInt(argv.find(a => /^\d+$/.test(a)) ?? '5', 10);
const DIR_ARG = argv.find(a => a.startsWith('--dir='))?.slice(6);

const EXAMPLES_DIR = path.resolve(CWD, DIR_ARG ?? 'test/examples/bosl2');
const OUT_DIR = path.resolve(CWD, 'test/out/benchmark-phases');
const RESULTS_PREFIX = path.resolve(CWD, 'test/out/benchmark-phases-results');

const CHILD_FLAG = '__OSCAD_PHASE_CHILD';

// measure one file - one iteration
async function childMain() {
  const target = process.env[CHILD_FLAG]!;
  const runtimeUrl =
      pathToFileURL(path.join(CWD, 'runtime', 'runtime.js')).href;
  const moduleUrl = pathToFileURL(target).href;

  try {
    // runtime init (wasm boot, wasm.setup(), font/canvas wiring)
    const t0 = performance.now();
    const rt: any = await import(runtimeUrl);
    const t1 = performance.now();

    // Logical Part: The generated file's top-level body runs here. Module and
    // function evaluation, for-loops, list comprehensions, BOSL2 path and vnf
    // math, and the construction of the Manifold operation DAG
    const mod: any = await import(moduleUrl);
    const t2 = performance.now();

    // Geometry Part: Force the DAG to flush, result may be a Manifold(3D) or a
    // CrossSection (2D)
    const result = mod.result;

    if (result && typeof result.getMesh === 'function') {
      const mesh = result.getMesh();
    } else if (result && typeof result.toPolygons === 'function') {
      const polys = result.toPolygons();
    }
    const t3 = performance.now();

    process.stdout.write(JSON.stringify({
      ok: true,
      init: t1 - t0,
      logic: t2 - t1,
      geometry: t3 - t2,
    }));
  } catch (err) {
    process.stdout.write(
        JSON.stringify({ok: false, error: (err as Error).message}));
    process.exitCode = 1;
  }
}

interface Stat {
  mean: number;
  median: number;
  min: number;
  max: number;
}

function computeStat(vals: number[]): Stat {
  const sorted = [...vals].sort((a, b) => a - b);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 :
                                           sorted[mid];
  return {mean, median, min: sorted[0], max: sorted[sorted.length - 1]};
}

// Geometry is a difference of two independently-measured medians, so it can go
// slightly negative on files where the mesh work is near zero. Clamp for
// display.
function clampNonNegative(n: number): number {
  return n < 0 ? 0 : n;
}

interface PhaseResult {
  file: string;
  openscadLogic: Stat|null;       // t(csg)
  openscadFull: Stat|null;        // t(off)
  openscadGeometry: number|null;  // median(off) - median(csg)
  ourInit: Stat|null;
  ourLogic: Stat|null;
  ourGeometry: Stat|null;
  logicRatio: number|null;  // openscad logic/our logic
  geometryRatio: number|null;
  totalRatio: number|null;
  error?: string;
}

interface SkippedFile {
  file: string;
  reason: string;
}
function isSkipped(r: PhaseResult|SkippedFile): r is SkippedFile {
  return 'reason' in r;
}

function isOpenscadAvailable(): boolean {
  try {
    execFileSync('openscad', ['--version'], {stdio: 'ignore'});
    return true;
  } catch {
    return false;
  }
}

const HAS_TASKSET = (() => {
  try {
    execFileSync('taskset', ['--version'], {stdio: 'ignore'});
    return true;
  } catch {
    return false;
  }
})();

function runOpenscadOnce(absFile: string, ext: 'csg'|'off'): number {
  const dir = mkdtempSync(path.join(tmpdir(), 'scad-phase-'));
  try {
    const out = path.join(dir, `out.${ext}`);
    const libraryPaths =
        [path.dirname(absFile), CWD, ...getOpenSCADLibraryPaths()];
    const env = {
      ...process.env,
      OPENSCADPATH: [
        ...libraryPaths, process.env.OPENSCADPATH ?? ''
      ].filter(Boolean).join(path.delimiter),
    };

    const args =
        ['-o', out, `--export-format=${ext}`, '--backend=manifold', absFile];
    const t0 = performance.now();
    if (HAS_TASKSET)
      execFileSync(
          'taskset', ['-c', '0', 'openscad', ...args], {stdio: 'ignore', env});
    else
      execFileSync('openscad', args, {stdio: 'ignore', env});
    return performance.now() - t0;
  } finally {
    rmSync(dir, {recursive: true, force: true});
  }
}

interface ChildSample {
  init: number;
  logic: number;
  geometry: number;
}

// one fresh child process per iteration
function runOursOnce(tsFile: string): ChildSample {
  const raw = execFileSync(
      process.execPath,
      ['--import', 'tsx', path.resolve(CWD, 'test/benchmark-phases.ts')],
      {
        env: {...process.env, [CHILD_FLAG]: tsFile},
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'ignore'],
        cwd: CWD,
      },
  );

  // The generated module may echo() to stdout - take the last JSON object
  const start = raw.lastIndexOf('{');
  const parsed = JSON.parse(raw.slice(start));
  if (!parsed.ok) throw new Error(parsed.error);
  return parsed;
}

function findScadFiles(dir: string): string[] {
  let out: string[] = [];
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory())
      out = out.concat(findScadFiles(full));
    else if (entry.isFile() && full.endsWith('.scad'))
      out.push(full);
  }
  return out;
}

function benchmarkFile(
    absFile: string, openscadAvailable: boolean): PhaseResult|SkippedFile {
  const rel = path.relative(EXAMPLES_DIR, absFile).replace(/\\/g, '/');
  const outTs = path.join(OUT_DIR, rel.replace(/\.scad$/i, '.ts'));
  const errors: string[] = [];

  // Transpile if pre-transpiled output is missing (untimed)
  if (!fs.existsSync(outTs)) {
    try {
      fs.mkdirSync(path.dirname(outTs), {recursive: true});
      const libraryPaths =
          [path.dirname(absFile), CWD, ...getOpenSCADLibraryPaths()];
      const {code} = compileConsumer(absFile, outTs, libraryPaths, CWD);
      fs.writeFileSync(outTs, code);
    } catch (err) {
      return {
        file: rel,
        reason: `transpile: ${(err as Error).message.split('\n')[0]}`
      };
    }
  }

  // OpenSCAD: .csg (logic) then .off (logic + geometry)
  const csgTimes: number[] = [];
  const offTimes: number[] = [];
  if (openscadAvailable) {
    for (let i = 0; i < RUNS; i++) {
      try {
        csgTimes.push(runOpenscadOnce(absFile, 'csg'));
      } catch (err) {
        return {
          file: rel,
          reason: `openscad csg: ${(err as Error).message.split('\n')[0]}`
        };
      }
    }
    for (let i = 0; i < RUNS; i++) {
      try {
        offTimes.push(runOpenscadOnce(absFile, 'off'));
      } catch (err) {
        return {
          file: rel,
          reason: `openscad off: ${(err as Error).message.split('\n')[0]}`
        };
      }
    }
  }

  // Ours: init / logic / geometry, fresh process per iteration
  const initTimes: number[] = [];
  const logicTimes: number[] = [];
  const geomTimes: number[] = [];

  for (let i = 0; i < RUNS; i++) {
    try {
      const s = runOursOnce(outTs);
      initTimes.push(s.init);
      logicTimes.push(s.logic);
      geomTimes.push(s.geometry);
    } catch (err) {
      errors.push(`run: ${(err as Error).message.split('\n')[0]}`);
      break;
    }
  }

  const complete = logicTimes.length === RUNS;
  const ourInit = complete ? computeStat(initTimes) : null;
  const ourLogic = complete ? computeStat(logicTimes) : null;
  const ourGeometry = complete ? computeStat(geomTimes) : null;

  const openscadLogic = csgTimes.length === RUNS ? computeStat(csgTimes) : null;
  const openscadFull = offTimes.length === RUNS ? computeStat(offTimes) : null;
  const openscadGeometry = openscadLogic && openscadFull ?
      clampNonNegative(openscadFull.median - openscadLogic.median) :
      null;

  const logicRatio = openscadLogic && ourLogic && ourLogic.median > 0 ?
      openscadLogic.median / ourLogic.median :
      null;
  const geometryRatio =
      openscadGeometry !== null && ourGeometry && ourGeometry.median > 0 ?
      openscadGeometry / ourGeometry.median :
      null;
  const totalRatio = openscadFull && ourLogic && ourGeometry ?
      openscadFull.median / (ourLogic.median + ourGeometry.median) :
      null;

  return {
    file: rel,
    openscadLogic,
    openscadFull,
    openscadGeometry,
    ourInit,
    ourLogic,
    ourGeometry,
    logicRatio,
    geometryRatio,
    totalRatio,
    error: errors.length ? errors.join(' | ') : undefined,
  };
}

function writeJson(
    results: PhaseResult[], skipped: SkippedFile[],
    openscadAvailable: boolean) {
  fs.mkdirSync(path.dirname(RESULTS_PREFIX), {recursive: true});
  fs.writeFileSync(
      `${RESULTS_PREFIX}.json`,
      JSON.stringify(
          {
            generatedAt: new Date().toISOString(),
            runsPerFile: RUNS,
            examplesDir: path.relative(CWD, EXAMPLES_DIR).replace(/\\/g, '/'),
            openscadAvailable,
            coresPinned: HAS_TASKSET,
            note:
                'Transpilation excluded by design. init is measured and excluded from logic.',
            benchmarkedCount: results.length,
            skippedCount: skipped.length,
            results,
            skipped,
          },
          null, 2));
}

function writeCsv(results: PhaseResult[]) {
  const n = (v: number|null|undefined) =>
      (v === null || v === undefined ? '' : v.toFixed(2));
  const header = [
    'file',
    'openscad_logic_median',
    'openscad_geometry_median',
    'openscad_total_median',
    'our_init_median',
    'our_logic_median',
    'our_geometry_median',
    'our_total_median',
    'logic_ratio',
    'geometry_ratio',
    'total_ratio',
    'error',
  ].join(',');
  const lines = results.map(r => {
    const ourTotal = r.ourLogic && r.ourGeometry ?
        r.ourLogic.median + r.ourGeometry.median :
        null;
    return [
      r.file,
      n(r.openscadLogic?.median),
      n(r.openscadGeometry),
      n(r.openscadFull?.median),
      n(r.ourInit?.median),
      n(r.ourLogic?.median),
      n(r.ourGeometry?.median),
      n(ourTotal),
      n(r.logicRatio),
      n(r.geometryRatio),
      n(r.totalRatio),
      r.error ? `"${r.error.replace(/"/g, '""').slice(0, 300)}"` : '',
    ].join(',');
  });
  fs.writeFileSync(
      `${RESULTS_PREFIX}.csv`, [header, ...lines].join('\n') + '\n');
}

function writeMarkdown(
    results: PhaseResult[], skipped: SkippedFile[],
    openscadAvailable: boolean) {
  const f = (v: number|null|undefined) =>
      (v === null || v === undefined ? 'ERR' : v.toFixed(1));
  const x = (v: number|null) => (v === null ? 'n/a' : v.toFixed(2) + 'x');

  const lines = [
    '# Phase-Separated Benchmark: Logic vs Geometry',
    '',
    `Generated: ${new Date().toISOString()}`,
    '',
    `Runs per file: ${RUNS} · OpenSCAD available: ${
        openscadAvailable ?
            'yes' :
            'no'} · Cores pinned: ${HAS_TASKSET ? 'yes' : 'no'}`,
    `Examples: \`${
        path.relative(CWD, EXAMPLES_DIR)
            .replace(/\\/g, '/')}\` · Benchmarked: ${
        results.length} · Skipped: ${skipped.length}`,
    '',
    '## Method',
    '',
    'OpenSCAD logic = `-o out.csg` (parse + evaluate the language, dump CSG tree, no meshing).',
    'OpenSCAD geometry = median(`-o out.off`) − median(`-o out.csg`).',
    '',
    'Our logic = importing the generated module, which builds Manifold\'s lazy operation DAG.',
    'Our geometry = forcing that DAG to flush (`getMesh()` / `toPolygons()`).',
    'Runtime init (wasm boot + `wasm.setup()` + fonts) is measured separately and excluded.',
    '',
    'Transpilation is excluded: it is a one-time build step with no OpenSCAD analogue.',
    '',
    'All timings are medians in milliseconds. Ratio > 1x means we are faster.',
    '',
    '| File | OS logic | OS geom | Our logic | Our geom | Logic ratio | Geom ratio | Total ratio ',
    '|---|---:|---:|---:|---:|---:|---:|---:|',
    ...results.map(
        r => `| ${r.file} | ${f(r.openscadLogic?.median)} | ${
                 f(r.openscadGeometry)} | ` +
            `${f(r.ourLogic?.median)} | ${f(r.ourGeometry?.median)} | ` +
            `${x(r.logicRatio)} | ${x(r.geometryRatio)} | ${
                 x(r.totalRatio)} | `),
    '',
    '',
    `Runtime init overhead (excluded above): median ` +
        `${
            f(results.length ?
                  computeStat(
                      results.flatMap(r => r.ourInit ? [r.ourInit.median] : []))
                      .median :
                  null)} ms per process.`,
  ];

  if (skipped.length) {
    lines.push(
        '', '## Skipped', '', '| File | Reason |', '|---|---|',
        ...skipped.map(
            s => `| ${s.file} | ${
                s.reason.replace(/\|/g, '/').slice(0, 150)} |`));
  }

  fs.writeFileSync(`${RESULTS_PREFIX}.md`, lines.join('\n') + '\n');
}

function printTable(results: PhaseResult[]) {
  const f = (v: number|null|undefined) =>
      (v === null || v === undefined ? 'ERR' : v.toFixed(1));
  const x = (v: number|null) => (v === null ? 'n/a' : v.toFixed(2) + 'x');
  const w = Math.max(4, ...results.map(r => r.file.length));
  const header =
      `${'file'.padEnd(w)}  ${'os-logic'.padStart(9)}  ${
                                                    'os-geom'.padStart(9)}  ` +
      `${'our-logic'.padStart(9)}  ${
                                'our-geom'.padStart(
                                    9)}  ${'logic'.padStart(7)}  ${
                                                              'geom'.padStart(
                                                                  7)}`;
  console.log(`\n--- Median timings (ms) over ${RUNS} runs ---`);
  console.log(header);
  console.log('-'.repeat(header.length));
  for (const r of results) {
    console.log(
        `${r.file.padEnd(w)}  ${f(r.openscadLogic?.median).padStart(9)}  ` +
        `${f(r.openscadGeometry).padStart(9)}  ${
            f(r.ourLogic?.median).padStart(9)}  ` +
        `${f(r.ourGeometry?.median).padStart(9)}  ${
            x(r.logicRatio).padStart(7)}  ${x(r.geometryRatio).padStart(7)}`);
  }
}


// entry point for the parent process. Spawns children to measure phases
function parentMain() {
  const openscadAvailable = isOpenscadAvailable();

  if (openscadAvailable) {
    console.log('OpenSCAD available');
  } else {
    console.log('OpenSCAD unavailable');
    return;
  }

  if (!HAS_TASKSET)
    console.log(
        'taskset unavailable - OpenSCAD will use all cores; ratios will favour it.');

  const files = findScadFiles(EXAMPLES_DIR).sort();
  console.log(`Found ${files.length} .scad file(s) under ${EXAMPLES_DIR}`);
  console.log(`Running ${RUNS} iterations per file per phase...\n`);

  const results: PhaseResult[] = [];
  const skipped: SkippedFile[] = [];

  files.forEach((absFile, i) => {
    const rel = path.relative(EXAMPLES_DIR, absFile).replace(/\\/g, '/');
    console.log(`[${i + 1}/${files.length}] ${rel} ... `);
    const r = benchmarkFile(absFile, openscadAvailable);
    if (isSkipped(r)) {
      skipped.push(r);
      console.log(`SKIPPED (${r.reason.slice(0, 90)})`);
    } else {
      results.push(r);
      console.log(
          r.error ? `FAILED (${r.error.slice(0, 90)})` :
                    `ok  logic ${r.logicRatio?.toFixed(2) ?? '?'}x`);
    }
  });

  writeJson(results, skipped, openscadAvailable);
  writeCsv(results);
  writeMarkdown(results, skipped, openscadAvailable);
  printTable(results);

  const ratios =
      results.flatMap(r => (r.logicRatio !== null ? [r.logicRatio] : []));
  if (ratios.length) {
    console.log(`Logic-phase speedup across ${ratios.length} files: median ${
        computeStat(ratios).median.toFixed(2)}x`);
  }
  console.log(`Wrote ${RESULTS_PREFIX}.{json,csv,md}`);
}

if (process.env[CHILD_FLAG]) {
  await childMain();
} else {
  parentMain();
}