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

import {Document} from '@gltf-transform/core';
import {getSceneVertexCount, VertexCountMethod} from '@gltf-transform/functions';
import {resolve} from 'node:path';
import {beforeEach, expect, suite, test} from 'vitest';

import {bundleFile} from '../lib/bundler.ts';
import * as worker from '../lib/worker.ts';

const countVertices = (doc: Document) => {
  const scene = doc.getRoot().listScenes()[0];
  if (!scene) return -1;
  return getSceneVertexCount(scene, VertexCountMethod.UPLOAD_NAIVE);
};

beforeEach(() => worker.cleanup());

suite('Import remote modules from', () => {
  test.skip('esm.sh', async () => {
    const entrypoint = resolve(import.meta.dirname, './examples/voronoi.mjs');
    const bundle = await bundleFile(entrypoint, {jsCDN: 'esm.sh'});
    const result = await worker.evaluate(bundle, {doNotBundle: true});
    expect(countVertices(result)).toBeGreaterThan(0);
  });

  test('jsDelivr', async () => {
    const entrypoint = resolve(import.meta.dirname, './examples/voronoi.mjs');
    const bundle = await bundleFile(entrypoint, {jsCDN: 'jsDelivr'});
    const result = await worker.evaluate(bundle, {doNotBundle: true});
    expect(countVertices(result)).toBeGreaterThan(0);
  });

  test.skip('skypack', async () => {
    const entrypoint = resolve(import.meta.dirname, './examples/voronoi.mjs');
    const bundle = await bundleFile(entrypoint, {jsCDN: 'skypack'});
    const result = await worker.evaluate(bundle, {doNotBundle: true});
    expect(countVertices(result)).toBeGreaterThan(0);
  });
});
