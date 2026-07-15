import fs from "fs";
import path from "path";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { compileConsumer } from "../core/orchestrate.js";
import { getOpenSCADLibraryPaths } from "../core/resolver.js";

const RUNS = parseInt(process.argv[2] ?? "5", 10);
const CWD = process.cwd();
const EXAMPLES_DIR = path.resolve(CWD, "test/examples");
const OUT_DIR = path.resolve(CWD, "test/out/benchmark");
const RESULTS_PREFIX = path.resolve(CWD, "test/out/benchmark-results");

interface Stat {
  mean: number;
  median: number;
  min: number;
  max: number;
}

interface FileResult {
  file: string;
  ourCompile: Stat | null;
  ourRun: Stat | null;
  ourTotal: Stat | null;
  openscad: Stat | null;
  speedRatio: number | null;
  error?: string;
}

interface SkippedFile {
  file: string;
  reason: string;
}

function computeStat(vals: number[]): Stat {
  const sorted = [...vals].sort((a, b) => a - b);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  return { mean, median, min: sorted[0], max: sorted[sorted.length - 1] };
}

function findScadFiles(dir: string): string[] {
  let out: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) out = out.concat(findScadFiles(fullPath));
    else if (entry.isFile() && fullPath.endsWith(".scad")) out.push(fullPath);
  }
  return out;
}

function isOpenscadAvailable(): boolean {
  try {
    execFileSync("openscad", ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function runOpenscadOnce(absFile: string): number {
  const dir = mkdtempSync(path.join(tmpdir(), "scad-bench-"));
  try {
    const out = path.join(dir, "out.off");
    const openscadArgs = ["-o", out, "--backend=manifold", absFile];
    const t0 = performance.now();
    execFileSync("taskset", ["-c", "0", "openscad", ...openscadArgs], { stdio: "ignore" });
    return performance.now() - t0;
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// Returns a FileResult, or a SkippedFile if native OpenSCAD couldn't compile the file at all
function benchmarkFile(absFile: string, openscadAvailable: boolean): FileResult | SkippedFile {
  const rel = path.relative(EXAMPLES_DIR, absFile).replace(/\\/g, "/");
  const outTs = path.join(OUT_DIR, rel.replace(/\.scad$/i, ".ts"));
  fs.mkdirSync(path.dirname(outTs), { recursive: true });
  const libraryPaths = [path.dirname(absFile), CWD, ...getOpenSCADLibraryPaths()];

  const openscadTimes: number[] = [];

  // Native OpenSCAD
  if (openscadAvailable) {
    for (let i = 0; i < RUNS; i++) {
      try {
        openscadTimes.push(runOpenscadOnce(absFile));
      } catch (err) {
        return { file: rel, reason: `openscad: ${(err as Error).message.split("\n")[0]}` };
      }
    }
  }

  // Compile
  const compileTimes: number[] = [];
  const errors: string[] = [];
  for (let i = 0; i < RUNS; i++) {
    try {
      const t0 = performance.now();
      const { code } = compileConsumer(absFile, outTs, libraryPaths, CWD);
      compileTimes.push(performance.now() - t0);
      fs.writeFileSync(outTs, code);
    } catch (err) {
      errors.push(`compile: ${(err as Error).message}`);
      break;
    }
  }

  // Run the compiled output
  const runTimes: number[] = [];
  if (compileTimes.length === RUNS) {
    for (let i = 0; i < RUNS; i++) {
      try {
        const t0 = performance.now();
        execFileSync(process.execPath, ["--import", "tsx", outTs], { stdio: "ignore" });
        runTimes.push(performance.now() - t0);
      } catch (err) {
        errors.push(`run: ${(err as Error).message}`);
        break;
      }
    }
  }

  const ourCompile = compileTimes.length === RUNS ? computeStat(compileTimes) : null;
  const ourRun = runTimes.length === RUNS ? computeStat(runTimes) : null;
  const ourTotal = ourCompile && ourRun ? computeStat(compileTimes.map((c, i) => c + runTimes[i])) : null;
  const openscad = openscadTimes.length === RUNS ? computeStat(openscadTimes) : null;
  const speedRatio = openscad && ourTotal ? openscad.median / ourTotal.median : null;

  return { file: rel, ourCompile, ourRun, ourTotal, openscad, speedRatio, error: errors.length ? errors.join(" | ") : undefined };
}

function isSkipped(r: FileResult | SkippedFile): r is SkippedFile {
  return "reason" in r;
}

function writeJson(results: FileResult[], skipped: SkippedFile[], openscadAvailable: boolean) {
  fs.mkdirSync(path.dirname(RESULTS_PREFIX), { recursive: true });
  fs.writeFileSync(`${RESULTS_PREFIX}.json`, JSON.stringify({
    generatedAt: new Date().toISOString(),
    runsPerFile: RUNS,
    openscadAvailable,
    fileCount: results.length + skipped.length,
    benchmarkedCount: results.length,
    skippedCount: skipped.length,
    results,
    skipped,
  }, null, 2));
}

function writeCsv(results: FileResult[]) {
  const cols = (s: Stat | null) => s ? [s.mean, s.median, s.min, s.max].map(n => n.toFixed(2)) : ["", "", "", ""];
  const header = [
    "file",
    "openscad_mean", "openscad_median", "openscad_min", "openscad_max",
    "compile_mean", "compile_median", "compile_min", "compile_max",
    "run_mean", "run_median", "run_min", "run_max",
    "total_mean", "total_median", "total_min", "total_max",
    "speed_ratio", "error",
  ].join(",");
  const lines = results.map(r => [
    r.file,
    ...cols(r.openscad),
    ...cols(r.ourCompile),
    ...cols(r.ourRun),
    ...cols(r.ourTotal),
    r.speedRatio !== null ? r.speedRatio.toFixed(2) : "",
    r.error ? `"${r.error.replace(/"/g, '""').slice(0, 300)}"` : "",
  ].join(","));
  fs.writeFileSync(`${RESULTS_PREFIX}.csv`, [header, ...lines].join("\n") + "\n");
}

function writeMarkdown(results: FileResult[], skipped: SkippedFile[], openscadAvailable: boolean) {
  const fmt = (s: Stat | null) => s ? s.median.toFixed(1) : "ERR";
  const succeeded = results.filter(r => r.ourTotal).length;

  const lines = [
    "# Compiler Benchmark Results",
    "",
    `Generated: ${new Date().toISOString()}`,
    "",
    `Runs per file: ${RUNS} · OpenSCAD available: ${openscadAvailable ? "yes" : "no"}`,
    `Benchmarked: ${results.length} files (${succeeded} fully succeeded) · Skipped (openscad compile error): ${skipped.length}`,
    "",
    "All timings are medians in milliseconds. \"Ratio\" = openscad median / our-total median (>1x means we're faster).",
    "",
    "| File | OpenSCAD | Our compile | Our run | Our total | Ratio | Notes |",
    "|---|---:|---:|---:|---:|---:|---|",
    ...results.map(r => {
      const ratio = r.speedRatio !== null ? r.speedRatio.toFixed(2) + "x" : "n/a";
      const notes = r.error ? r.error.replace(/\|/g, "/").slice(0, 100) : "";
      return `| ${r.file} | ${openscadAvailable ? fmt(r.openscad) : "n/a"} | ${fmt(r.ourCompile)} | ${fmt(r.ourRun)} | ${fmt(r.ourTotal)} | ${ratio} | ${notes} |`;
    }),
  ];

  if (skipped.length > 0) {
    lines.push(
      "",
      "## Skipped files (native OpenSCAD failed to compile)",
      "",
      "| File | Reason |",
      "|---|---|",
      ...skipped.map(s => `| ${s.file} | ${s.reason.replace(/\|/g, "/").slice(0, 150)} |`),
    );
  }

  fs.writeFileSync(`${RESULTS_PREFIX}.md`, lines.join("\n") + "\n");
}

function printTable(results: FileResult[], openscadAvailable: boolean) {
  const fmt = (s: Stat | null) => s ? s.median.toFixed(1) : "ERR";
  const fileW = Math.max(4, ...results.map(r => r.file.length));
  const header = `${"file".padEnd(fileW)}  ${"openscad".padStart(9)}  ${"compile".padStart(8)}  ${"run".padStart(9)}  ${"total".padStart(9)}  ratio`;
  console.log("--- Median timings (ms) over " + RUNS + " runs ---");
  console.log(header);
  console.log("-".repeat(header.length));
  for (const r of results) {
    const ratio = r.speedRatio !== null ? r.speedRatio.toFixed(2) + "x" : "n/a";
    console.log(`${r.file.padEnd(fileW)}  ${(openscadAvailable ? fmt(r.openscad) : "n/a").padStart(9)}  ${fmt(r.ourCompile).padStart(8)}  ${fmt(r.ourRun).padStart(9)}  ${fmt(r.ourTotal).padStart(9)}  ${ratio}`);
  }
}

function main() {
  const openscadAvailable = isOpenscadAvailable();
  console.log(`OpenSCAD on PATH: ${openscadAvailable ? "yes" : "no"}`);

  const files = findScadFiles(EXAMPLES_DIR).sort();
  console.log(`Found ${files.length} .scad file(s) under ${EXAMPLES_DIR}`);
  console.log(`Running ${RUNS} iterations per file per phase...`);

  const results: FileResult[] = [];
  const skipped: SkippedFile[] = [];

  files.splice(0, 5).forEach((absFile, i) => {
    const rel = path.relative(EXAMPLES_DIR, absFile).replace(/\\/g, "/");
    process.stdout.write(`[${i + 1}/${files.length}] ${rel} ... `);
    const r = benchmarkFile(absFile, openscadAvailable);
    if (isSkipped(r)) {
      skipped.push(r);
      console.log(`SKIPPED (${r.reason.slice(0, 100)})`);
    } else {
      results.push(r);
      console.log(r.error ? `FAILED (${r.error.slice(0, 100)})` : "ok");
    }
  });

  const succeeded = results.filter(r => r.ourTotal).length;
  console.log(`${results.length} file(s) benchmarked (${succeeded} fully succeeded), ${skipped.length} skipped due to native OpenSCAD compile errors.`);

  writeJson(results, skipped, openscadAvailable);
  writeCsv(results);
  writeMarkdown(results, skipped, openscadAvailable);
  printTable(results, openscadAvailable);

  console.log(`Wrote ${RESULTS_PREFIX}.json, ${RESULTS_PREFIX}.csv, ${RESULTS_PREFIX}.md`);
}

main();