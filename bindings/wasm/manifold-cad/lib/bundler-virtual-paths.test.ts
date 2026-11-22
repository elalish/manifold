// Copyright 2026 The Manifold Authors.
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

import {expect, test} from 'vitest';

import {bundleCode, bundleFile} from './bundler.ts';

function execute(bundle: string, topLevel: Record<string, unknown> = {}) {
  const module = {exports: {} as Record<string, unknown>};
  new Function(
      'module', 'exports', '_manifold_runtime_url', '_manifold_cad_top_level',
      '_manifold_cad_library',
      bundle)(module, module.exports, null, topLevel, {});
  return module.exports;
}

test.each(['js', 'ts'])(
    'resolves a root-relative virtual %s import', async extension => {
      const bundle = await bundleCode(
          `import {value} from './part.${extension}'; export default value;`, {
            files: {[`part.${extension}`]: 'export const value = 42;'},
            fetchRemotePackages: false
          });
      expect(execute(bundle).default).toBe(42);
    });

test(
    'resolves an import relative to an absolute virtual source filename',
    async () => {
      const bundle = await bundleCode(
          'import {value} from \'./parts/part.ts\'; export default value;', {
            filename: '/virtual/main.ts',
            files: {'/virtual/parts/part.ts': 'export const value = 42;'},
            fetchRemotePackages: false
          });
      expect(execute(bundle).default).toBe(42);
    });

test('resolves sibling and parent imports from a virtual module', async () => {
  const bundle = await bundleCode(
      'import {value} from \'parts/assembly.ts\'; export default value;', {
        files: {
          'parts/assembly.ts': `import {width} from './dimensions.ts';
            import {extra} from '../common/extra.ts';
            export const value = width + extra;`,
          'parts/dimensions.ts': 'export const width = 30;',
          'common/extra.ts': 'export const extra = 12;'
        },
        fetchRemotePackages: false
      });
  expect(execute(bundle).default).toBe(42);
});

test(
    'keeps same-named modules in different virtual directories distinct',
    async () => {
      const bundle = await bundleCode(
          `
    import {value as left} from 'left/model.ts';
    import {value as right} from 'right/model.ts';
    export default [left, right];`,
          {
            files: {
              'left/model.ts': 'export {value} from \'./dimensions.ts\';',
              'right/model.ts': 'export {value} from \'./dimensions.ts\';',
              'left/dimensions.ts': 'export const value = 12;',
              'right/dimensions.ts': 'export const value = 30;'
            },
            fetchRemotePackages: false
          });
      expect(execute(bundle).default).toEqual([12, 30]);
    });

test(
    'prefers an importer-relative file over a global exact-key fallback',
    async () => {
      const bundle = await bundleCode(
          'import {value} from \'parts/model.ts\'; export default value;', {
            files: {
              'parts/model.ts': 'export {value} from \'./dimensions.ts\';',
              './dimensions.ts': 'export const value = 999;',
              'parts/dimensions.ts': 'export const value = 42;'
            },
            fetchRemotePackages: false
          });
      expect(execute(bundle).default).toBe(42);
    });

test('preserves exact bare virtual-module aliases', async () => {
  const bundle = await bundleCode(
      'import {value} from \'parts/model.ts\'; export default value;', {
        files: {
          'parts/model.ts': 'export {value} from \'dimensions.ts\';',
          'dimensions.ts': 'export const value = 42;',
          'parts/dimensions.ts': 'export const value = 999;'
        },
        fetchRemotePackages: false
      });
  expect(execute(bundle).default).toBe(42);
});

test('normalizes dot segments and reuses the same virtual module', async () => {
  const bundle = await bundleCode(
      `
    import {state as first} from './parts/../parts/state.ts';
    import {state as second} from 'parts/state.ts';
    first.value += 1;
    export default second.value;`,
      {
        files: {'parts/state.ts': 'export const state = {value: 41};'},
        fetchRemotePackages: false
      });
  expect(execute(bundle).default).toBe(42);
});

test.each(['helpers.ts', './helpers.ts'])(
    'retains leading-dot exact keys while resolving %s', async helper => {
      const bundle = await bundleCode(
          'import {value} from \'./parts/model.ts\'; export default value;', {
            files: {
              './parts/model.ts': 'export {value} from \'../helpers.ts\';',
              [helper]: 'export const value = 42;'
            },
            fetchRemotePackages: false
          });
      expect(execute(bundle).default).toBe(42);
    });

test(
    'bundles a relative virtual entry point with its top-level CAD context',
    async () => {
      const settings: number[] = [];
      const bundle = await bundleFile('./project/main.ts', {
        files: {
          'project/main.ts': `import {value} from './part.ts';
        import {setCircularSegments} from 'manifold-cad/manifoldCAD';
        setCircularSegments(value);
        export default value;`,
          'project/part.ts': 'export const value = 42;'
        },
        fetchRemotePackages: false
      });
      expect(execute(bundle, {
               setCircularSegments: (value: number) => settings.push(value)
             }).default)
          .toBe(42);
      expect(settings).toEqual([42]);
    });

test(
    'resolves a cycle without creating duplicate virtual modules', async () => {
      const bundle = await bundleCode(
          'import {value} from \'lib/a.ts\'; export default value();', {
            files: {
              'lib/a.ts': `import {getB} from './b.ts';
            export const a = 20;
            export function value() { return a + getB(); }`,
              'lib/b.ts': `import {a} from './a.ts';
            export function getB() { return a + 2; }`
            },
            fetchRemotePackages: false
          });
      expect(execute(bundle).default).toBe(42);
    });

test(
    'does not turn a missing relative import into a bare-module alias',
    async () => {
      await expect(bundleCode('import \'parts/model.ts\';', {
        files: {
          'parts/model.ts': 'import \'./missing.ts\';',
          'missing.ts': 'export const unrelated = 42;'
        },
        fetchRemotePackages: false
      })).rejects.toThrow('Could not resolve "./missing.ts"');
    });
