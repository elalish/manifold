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

import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const __dirname = dirname(fileURLToPath(import.meta.url));

import {beforeEach, expect, suite, test} from 'vitest';
import * as worker from '../lib/worker.ts';
import {bundleFile} from '../lib/bundle.ts';

let evaluator = worker.getEvaluator() || worker.initialize();
beforeEach(() => worker.cleanup());

suite('Build model with the evaluator', () => {
  test('Import a model that generates an error', async () => {
    const filepath =
        resolve(__dirname, './fixtures/generateReferenceError.mjs');
    let code = await bundleFile(filepath);

    const ev = async () => await evaluator.evaluate(code);
    await expect(ev()).rejects.toThrowError()
  });
});

suite('Build model without the evaluator', () => {
  test('Import a model that generates an error', async () => {
    const imp = async () =>
        await import('./fixtures/generateReferenceError.mjs');
    await expect(imp()).rejects.toThrowError()
  });

  test('Import a model that generates an error', async () => {
    const imp = async () =>
        await import('./fixtures/generateReferenceError.mjs');
    try {
      await imp()
    } catch (e: any) {
      console.log(e.stack)
    }
  });
});
