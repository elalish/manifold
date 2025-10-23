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

beforeEach(() => worker.cleanup());

suite('Build a model with the worker', () => {
  test('Exceptions should throw', async () => {
    const filepath =
        resolve(import.meta.dirname, './fixtures/generateReferenceError.mjs');
    let code = await bundleFile(filepath);

    const ev = async () => await worker.evaluate(code);
    await expect(ev()).rejects.toThrowError()
  });

  test.skip(
      'Exceptions should be wrapped',
      async () => {

      })
});

suite('Build a model without the worker', () => {
  test('Exceptions should throw', async () => {
    const imp = async () =>
        await import('./fixtures/generateReferenceError.mjs');
    await expect(imp()).rejects.toThrowError()
  });

  test.skip(
      'Exceptions should be native',
      async () => {

      })
});
