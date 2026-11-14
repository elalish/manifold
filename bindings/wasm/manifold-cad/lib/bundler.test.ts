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

import {expect, suite, test} from 'vitest';

import {bundleCode} from './bundler.ts';
import * as worker from './worker.ts';

suite('Bundler', () => {
  test('Sets all manifoldCAD methods and properties', async () => {
    // Generate a list of known manifoldCAD keys outside of the worker.
    // Throw if any of them are undefined inside the worker.
    // This is mapped in `manifoldCADExportNames`, in `bundler.ts`.
    // Metaprogramming like this works in a test, but the bundler doesn't have
    // (and shouldn't need) an instantiated copy of manifoldCAD.
    const manifoldCAD = await import('./manifoldCAD.ts');
    const keys = Object
                     .keys(manifoldCAD)
                     // @ts-ignore
                     .filter(key => 'undefined' !== typeof manifoldCAD[key]);

    const script = `import * as manifoldCAD from 'manifold-3d/manifoldCAD';` +
        `const check = () => {` +
        `  const keys=${JSON.stringify(keys)};` +
        `  for (const key of keys) {` +
        `    if ('undefined' === typeof manifoldCAD[key]) {` +
        `      throw new Error(\`manifoldCAD.\${key} is not defined inside manifoldCAD.\`);` +
        `    }` +
        `  }` +
        `  return manifoldCAD.Manifold.sphere(1);` +
        `};` +
        `export default check;`;
    await expect(worker.evaluate(script)).to.resolves.not.toBeNull();
  });

  test('Accepts manifold-cad/manifoldCAD as the import name', async () => {
    const script = `import {Manifold} from 'manifold-cad/manifoldCAD';` +
        `export default Manifold.sphere(1);`;
    await expect(worker.evaluate(script)).to.resolves.not.toBeNull();
  });

  test('Accepts manifold-cad/manifoldCAD in a library', async () => {
    const files = {
      'lib.js': `import {Manifold} from 'manifold-cad/manifoldCAD';` +
          `export const ball = () => Manifold.sphere(1);`
    };
    const script = `import {ball} from 'lib.js';` +
        `export default ball();`;
    await expect(worker.evaluate(script, {files})).to.resolves.not.toBeNull();
  });

  test('Bundles files whose path has backslashes (issue #1915)', async () => {
    // `\u`, `\x` and `\2` used to end up as escape sequences in the
    // generated `import.meta.url` string, which made the bundle invalid.
    const path = 'C:\\units\\xparts\\2026\\model.js';
    const files = {[path]: 'export const url = import.meta.url;'};
    const code = `import {url} from ${JSON.stringify(path)}; export {url};`;
    await expect(bundleCode(code, {files})).resolves.toBeTypeOf('string');
  });
});
