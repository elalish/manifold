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
          if (stderr) console.log("Worker stderr:", stderr);
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

suite('Single Compiled Example', async () => {
  const fileName = process.env.TEST_FILE;

  if (!fileName) {
    throw new Error("TEST_FILE not provided");
  }

  test(`Test for ${fileName}`, async () => {
    const {volume, surfaceArea} = await getOpenScadProperties(fileName);

    if (volume == undefined || surfaceArea == undefined) return;

    const compiledFile = fileName.replace(".scad", ".ts").replace("examples", "out");
    const { volume: compiledVolume, surfaceArea: compiledSurfaceArea } = await getCompiledManifoldProperties(compiledFile);

    const tolerance = 0.001;

    console.log(`expected volume: ${volume}, recieved volume: ${compiledVolume}`);
    expectApproximatelyEqual(volume, compiledVolume, tolerance);
    console.log(`expected surfaceArea: ${surfaceArea}, recieved surfaceArea: ${compiledSurfaceArea}`);
    expectApproximatelyEqual(surfaceArea, compiledSurfaceArea, tolerance);
  });
});
