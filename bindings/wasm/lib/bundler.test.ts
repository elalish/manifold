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
});
