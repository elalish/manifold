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

import {transformStaticImportsToDynamic} from './util.ts';

suite('transformStaticImportsToDynamic', () => {
  test('import foo from "bar"', () => {
    const input = 'import foo from "bar";';
    const result = transformStaticImportsToDynamic(input);
    expect(result).to.equal('const {default:foo} = await import(\'bar\');');
  });

  test('import { foo } from "bar"', () => {
    const input = 'import { foo } from "bar";';
    const result = transformStaticImportsToDynamic(input);
    expect(result).to.equal('const { foo } = await import(\'bar\');');
  });

  test('import { foo, bar } from "baz"', () => {
    const input = 'import { foo, bar } from "baz";';
    const result = transformStaticImportsToDynamic(input);
    expect(result).to.equal('const { foo, bar } = await import(\'baz\');');
  });

  test('import { foo /*, bar */ } from "baz"', () => {
    const input = 'import { foo /*, bar */ } from "baz";';
    const result = transformStaticImportsToDynamic(input);
    expect(result).to.equal('const { foo } = await import(\'baz\');');
  });

  test('import { foo as bar } from "baz"', () => {
    const input = 'import { foo as bar } from "baz";';
    const result = transformStaticImportsToDynamic(input);
    expect(result).to.equal('const { foo: bar } = await import(\'baz\');');
  });

  test('import { foo, bar as baz } from "qux"', () => {
    const input = 'import { foo, bar as baz } from "qux";';
    const result = transformStaticImportsToDynamic(input);
    expect(result).to.equal('const { foo, bar: baz } = await import(\'qux\');');
  });

  test('import * as foo from "bar"', () => {
    const input = 'import * as foo from "bar";';
    const result = transformStaticImportsToDynamic(input);
    expect(result).to.equal('const foo = await import(\'bar\');');
  });

  /*
    test('Remote package prefix', () => {
      const input = 'import * as foo from "bar";';
      const result = transformStaticImportsToDynamic(input, 'https://esm.run/');
      expect(result).to.equal(
          'const foo = await import(\'https://esm.run/bar\');');
    });
    */

  test('Two imports', () => {
    const input = 'import * as foo from "foobar";\n' +
        'import baz from "quz";'
    const result = transformStaticImportsToDynamic(input);
    console.log(result);
    expect(result).to.equal(
        'const foo = await import(\'foobar\');\n' +
        'const {default:baz} = await import(\'quz\');');
  });
});