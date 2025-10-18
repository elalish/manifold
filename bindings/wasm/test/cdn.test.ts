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
import {bundleFile, bundleCode} from '../lib/bundle.ts';


suite('Import modules...', () => {
  test('...from jsDelivr', async () => {
    const entrypoint = resolve(__dirname, "./fixtures/importFromCDN.mjs");
    const bundle = await bundleFile(entrypoint);

    const evaluator = worker.initialize();
    const result = await evaluator.evaluate(bundle);
    expect(result.volume()).toBeGreaterThan(0);
  })

    test('...from a string', async () => {
    const filepath = resolve(__dirname, "./fixtures/importFromCDN.mjs");
      let code = await readFile(filepath, 'utf-8');
        code = await bundleCode(code);
  

    const evaluator = worker.initialize();
    const result = await evaluator.evaluate(code);
    expect(result.volume()).toBeGreaterThan(0);
  })
});
