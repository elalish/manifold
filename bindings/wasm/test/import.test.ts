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

import {readFile} from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const __dirname = dirname(fileURLToPath(import.meta.url));

import {expect, suite, test} from 'vitest';
import * as worker from '../lib/worker.ts';
import { dropStaticImport, transformStaticImportsToDynamic } from '../lib/util.ts';

suite('Build model with the evaluator', () => {
  test.skip('Import a model', async () => {
    const filepath = resolve(__dirname, "./fixtures/unitSphere.mjs");
    let code = await readFile(filepath, 'utf-8');

    code = dropStaticImport(code, '../../lib/manifoldCAD.ts')
    code = transformStaticImportsToDynamic(code);
    console.log(code)

    const result = await worker.evaluate(code);
    expect(result.volume()).toBeCloseTo(2.9428, 4);
  })

  test.skip('Import a model with imports', async () => {
    const filepath = resolve(__dirname, "./fixtures/importUnitSphere.mjs");
    let code = await readFile(filepath, 'utf-8');

    code = transformStaticImportsToDynamic(code);

    const result = await worker.evaluate(code);
    expect(result.volume()).toBeCloseTo(2.9428, 4);
  })
});

suite('Build model without the evaluator', () => {

  test('Import a model', async () => {
    const { unitSphere } = await import("./fixtures/unitSphere.mjs");
    const result = unitSphere();
    expect(result.volume()).toBeCloseTo(2.9428, 4);
  })

  test('Import a model with imports', async () => {
    const { result } = await import("./fixtures/importUnitSphere.mjs");
    expect(result.volume()).toBeCloseTo(2.9428, 4);
  })
});

