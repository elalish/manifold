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
import * as stackTraceParser from 'stacktrace-parser';
import {beforeEach, expect, suite, test} from 'vitest';

import {bundleFile} from '../lib/bundler.ts';
import {BundlerError, RuntimeError} from '../lib/error.ts';
import {parseStackTrace} from '../lib/util.ts';
import * as worker from '../lib/worker.ts';

beforeEach(() => worker.cleanup());

suite('Parses stacktraces from', () => {
  test('V8', () => {
    const stack = `ReferenceError: fail is not defined
    at dostuff (eval at evaluate (http://localhost:5173/@fs/Users/default/manifold/bindings/wasm/lib/worker.js?worker_file&type=module:96:24), <anonymous>:28:3)
    at eval (eval at evaluate (http://localhost:5173/@fs/Users/default/manifold/bindings/wasm/lib/worker.js?worker_file&type=module:96:24), <anonymous>:30:26)
    at evaluate (http://localhost:5173/@fs/Users/default/manifold/bindings/wasm/lib/worker.js?worker_file&type=module:97:15)
    at async handleEvaluate (http://localhost:5173/@fs/Users/default/manifold/bindings/wasm/lib/worker.js?worker_file&type=module:232:23)`;

    expect(parseStackTrace(stack)).deep.equals([
      {line: 28, column: 3, methodName: 'dostuff'},
      {line: 30, column: 26, methodName: null}
    ]);
  });

  test('Spidermonkey', () => {
    const stack =
        `dostuff@http://localhost:5173/@fs/Users/default/manifold/bindings/wasm/lib/worker.js?worker_file&type=module line 96 > AsyncFunction:28:3
anonymous@http://localhost:5173/@fs/Users/default/manifold/bindings/wasm/lib/worker.js?worker_file&type=module line 96 > AsyncFunction:30:26
evaluate@http://localhost:5173/@fs/Users/default/manifold/bindings/wasm/lib/worker.js?worker_file&type=module:97:15
async*handleEvaluate@http://localhost:5173/@fs/Users/default/manifold/bindings/wasm/lib/worker.js?worker_file&type=module:232:29
initializeWebWorker/self.onmessage@http://localhost:5173/@fs/Users/default/manifold/bindings/wasm/lib/worker.js?worker_file&type=module:257:27
EventHandlerNonNull*initializeWebWorker@http://localhost:5173/@fs/Users/default/manifold/bindings/wasm/lib/worker.js?worker_file&type=module:251:5
@http://localhost:5173/@fs/Users/default/manifold/bindings/wasm/lib/worker.js?worker_file&type=module:265:5`;

    expect(parseStackTrace(stack)).deep.equals([
      {line: 28, column: 3, methodName: 'dostuff'},
      {line: 30, column: 26, methodName: null}
    ]);
  });
  test('JavaScriptCore', () => {
    const stack = `dostuff@
@
anonymous@
@http://localhost:5173/@fs/Users/tony.thompson/dev/manifold/bindings/wasm/lib/worker.js:97:21`

    expect(parseStackTrace(stack).length).toBe(0);
  });
})

suite('Build a model with the worker', () => {
  test('Exceptions should throw', async () => {
    const filepath =
        resolve(import.meta.dirname, './fixtures/generateReferenceError.mjs');
    let code = await bundleFile(filepath);
    const ev = async () => await worker.evaluate(code);
    await expect(ev()).rejects.toThrowError()
  });

  test('Runtime exceptions should be wrapped', async () => {
    const filepath =
        resolve(import.meta.dirname, './fixtures/generateReferenceError.mjs');
    let code = await bundleFile(filepath);
    const ev = async () => await worker.evaluate(code);
    await expect(ev()).rejects.toThrowError(RuntimeError);
  });

  test('Stack trace (manifoldStack) should be accurate', async () => {
    const filepath =
        resolve(import.meta.dirname, './fixtures/generateReferenceError.mjs');
    let code = await bundleFile(filepath);
    let error: RuntimeError|null = null;
    try {
      await worker.evaluate(code);
    } catch (e) {
      error = e as RuntimeError;
    }
    const [frame, ...rest] = stackTraceParser.parse(error!.manifoldStack!);
    expect(frame.lineNumber).toBe(5);
    expect(frame.column).toBe(7);
    expect(frame.methodName).toBe('dostuff');
    expect(frame.file).toMatch(/generateReferenceError\.mjs$/);
    expect(rest.length).toEqual(1);
  });

  test('Stack traces should descend through imports', async () => {
    const filepath =
        resolve(import.meta.dirname, './fixtures/importReferenceError.mjs');
    let code = await bundleFile(filepath);
    let error: RuntimeError|null = null;
    try {
      await worker.evaluate(code);
    } catch (e) {
      error = e as RuntimeError;
    }
    expect(error!.message).toMatch(/fail/);
    const [frame, ...rest] = stackTraceParser.parse(error!.manifoldStack!);
    expect(frame.lineNumber).toBe(5);
    expect(frame.column).toBe(7);
    expect(frame.methodName).toBe('dostuff');
    expect(frame.file).toMatch(/exportReferenceError\.mjs$/);
    expect(rest.length).toEqual(2);
  });

  test('Bundler exceptions should be wrapped', async () => {
    const filepath =
        resolve(import.meta.dirname, './fixtures/bundlerError.mjs');
    const ev = async () => await bundleFile(filepath);
    await expect(ev()).rejects.toThrowError(BundlerError);
  });

  test('Bundler exceptions should have accurate stack traces.', async () => {
    const filepath =
        resolve(import.meta.dirname, './fixtures/bundlerError.mjs');
    let error: RuntimeError|null = null;
    try {
      await bundleFile(filepath);
    } catch (e) {
      error = e as RuntimeError;
    }
    const [frame, ...rest] = stackTraceParser.parse(error!.manifoldStack!);
    expect(frame.lineNumber).toBe(1);
    expect(frame.column).toBe(21);
    expect(frame.file).toMatch(/bundlerError\.mjs$/);
    expect(rest.length).toEqual(0);
  });
});

suite('Build a model without the worker', () => {
  test('Exceptions should throw', async () => {
    const imp = async () =>
        await import('./fixtures/generateReferenceError.mjs');
    await expect(imp()).rejects.toThrowError()
  });

  test('Exceptions should be native', async () => {
    const imp = async () =>
        await import('./fixtures/generateReferenceError.mjs');
    await expect(imp()).rejects.not.toThrowError(RuntimeError);
  });

  test('Stack trace (stack) should be accurate', async () => {
    let error: Error|null = null;
    try {
      await import('./fixtures/generateReferenceError.mjs');
    } catch (e) {
      error = e as Error;
    }
    const [frame, ...rest] = stackTraceParser.parse(error!.stack!);
    expect(frame.lineNumber).toBe(5);
    expect(frame.column).toBe(7);
    expect(frame.methodName).toBe('dostuff');
    expect(frame.file).toMatch(/generateReferenceError\.mjs$/);
    expect(rest.length).toBeGreaterThan(1);
  });
});
