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
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const __dirname = dirname(fileURLToPath(import.meta.url));

import {beforeEach, expect, suite, test} from 'vitest';
import * as worker from '../lib/worker.ts';
import {bundleFile, bundleCode} from '../lib/bundle.ts';

let evaluator = worker.getEvaluator() || worker.initialize();
beforeEach(() => worker.cleanup());

suite('Import modules', () => {
  test('when bundled from a file', async () => {
    const entrypoint = resolve(__dirname, './fixtures/importFromCDN.mjs');
    const bundle = await bundleFile(entrypoint);

    const result = await evaluator.evaluate(bundle);
    expect(result.volume()).toBeGreaterThan(0);
  });

  test('when bundled from a string', async () => {
    const filepath = resolve(__dirname, './fixtures/importFromCDN.mjs');
    const code = await readFile(filepath, 'utf-8');

    const bundle = await bundleCode(code);
    const result = await evaluator.evaluate(bundle);
    expect(result.volume()).toBeGreaterThan(0);
  });
});
