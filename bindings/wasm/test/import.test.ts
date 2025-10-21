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

import {resolve} from 'node:path';
import {beforeEach, expect, suite, test} from 'vitest';

import {bundleFile} from '../lib/bundle.ts';
import * as worker from '../lib/worker.ts';

let evaluator = worker.getEvaluator() || worker.initialize();
beforeEach(() => worker.cleanup());

suite('Build model with the evaluator', () => {
  test('Import a model that declares \'result\'', async () => {
    const filepath =
        resolve(import.meta.dirname, './fixtures/unitSphereResult.mjs');
    let code = await bundleFile(filepath);

    const result = await evaluator.evaluate(code);
    expect(result.volume()).toBeCloseTo(2.9428, 4);
  });

  test('Import a model that exports `result`', async () => {
    const filepath =
        resolve(import.meta.dirname, './fixtures/unitSphereExportResult.mjs');
    let code = await bundleFile(filepath);

    const result = await evaluator.evaluate(code);
    expect(result.volume()).toBeCloseTo(2.9428, 4);
  });

  test('Import a model with imports', async () => {
    const filepath =
        resolve(import.meta.dirname, './fixtures/importUnitSphere.mjs');
    let code = await bundleFile(filepath);

    const result = await evaluator.evaluate(code);
    expect(result.volume()).toBeCloseTo(2.9428, 4);
  })
});

suite('Build model without the evaluator', () => {
  test('Importing a model with no exports does nothing.', async () => {
    // @ts-ignore
    const {result} = await import('./fixtures/unitSphereResult.mjs');
    expect(result).toBeUndefined();
  });

  test('Import a model that exports `result`', async () => {
    const {result} = await import('./fixtures/unitSphereExportResult.mjs');
    expect(result.volume()).toBeCloseTo(2.9428, 4);
  });

  test('Import a model with imports', async () => {
    const {result} = await import('./fixtures/importUnitSphere.mjs');
    expect(result.volume()).toBeCloseTo(2.9428, 4);
  });
});
