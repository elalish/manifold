import fs from 'fs';
import {strict as assert} from 'assert';
import {afterEach, expect, suite, test} from 'vitest';
import path from 'path';
import { fork } from "child_process";

async function getOpenScadProperties(filename: string) {
  const firstLine = fs.readFileSync(filename).toString().split('\n')[0];
  if (!firstLine) {
    console.log(`File is empty: ${filename}`);
    return {volume: undefined, surfaceArea: undefined}
  }

  const volumeStr = firstLine.match(/Volume: ([\d\.]+)/)?.[1];
  const surfaceAreaStr = firstLine.match(/SurfaceArea: ([\d\.]+)/)?.[1];

  const volume = volumeStr ? parseFloat(volumeStr) : undefined;
  const surfaceArea = surfaceAreaStr ? parseFloat(surfaceAreaStr) : undefined;

  return {volume, surfaceArea};
}

async function getCompiledManifoldProperties(
  filename: string
): Promise<{ volume: number; surfaceArea: number }> {
  return new Promise((resolve, reject) => {
    const execArgv = (() => {
      try {
        require.resolve('tsx');
        return ['--import', 'tsx/esm'];
      } catch {
        return [];
      }
    })();

    const workerPath = path.resolve('./test-worker.ts');

    const worker = fork(workerPath, [filename], {
      execArgv,
      stdio: ['inherit', 'pipe', 'pipe', 'ipc'],
      timeout: 15000,
    });

    let stdout = '';
    let stderr = '';

    worker.stdout?.on('data', (chunk: Buffer) => { stdout += chunk.toString(); });
    worker.stderr?.on('data', (chunk: Buffer) => { stderr += chunk.toString(); });

    worker.on('exit', (code, signal) => {
      if (code === 0) {
        try {
          const jsonMatch = stdout.match(/(\{.*\})/s);
          const jsonStr = jsonMatch?.[1];
          if (!jsonStr) throw new Error('No JSON found in output');
          resolve(JSON.parse(jsonStr));
        } catch {
          reject(new Error(`Bad JSON from worker.\nstdout: ${stdout}\nstderr: ${stderr}`));
        }
      } else {
        reject(new Error(
          `Worker failed (code=${code}, signal=${signal})\n` +
          `file: ${filename}\n` +
          `stderr: ${stderr}\n` +
          `stdout: ${stdout}`
        ));
      }
    });

    worker.on('error', (err) => {
      reject(new Error(`Worker failed: ${err.message}\nworkerPath: ${workerPath}`));
    });
  });
}


function expectApproximatelyEqual(
  actual: number,
  expected: number,
  relativeTolerance: number
) {
  assert(actual !== undefined);
  assert(expected !== undefined);

  const diff = Math.abs(actual - expected);
  const largest = Math.max(Math.abs(actual), Math.abs(expected), 1);

  const relativeError = diff / largest;

  expect(relativeError).toBeLessThan(relativeTolerance);
}

function getAllFiles(dir: string): string[] {
    let results: string[] = [];

    const items = fs.readdirSync(dir, {
      withFileTypes: true
    });

    for (const item of items) {
      const fullPath = path.join(dir, item.name);

      if (item.isDirectory()) {
        results = results.concat(getAllFiles(fullPath));
      } else {
        results.push(fullPath);
      }
    }

    return results;
}

suite('Compiled Examples', async () => {
  const openscadFiles = getAllFiles("./examples");

  for (const file of openscadFiles) {
    if (file.endsWith(".scad")) {
      const {volume, surfaceArea} = await getOpenScadProperties(file);

      if (volume == undefined || surfaceArea == undefined) continue;

      test(`Test for ${file}`, async () => {
        const compiledFile = file.replace(".scad", ".ts").replace("examples", "out");
        const { volume: compiledVolume, surfaceArea: compiledSurfaceArea } = await getCompiledManifoldProperties(compiledFile);

        const tolerance = 0.001;
        expectApproximatelyEqual(volume, compiledVolume, tolerance);
        expectApproximatelyEqual(surfaceArea, compiledSurfaceArea, tolerance);
      });
    }
  }
});