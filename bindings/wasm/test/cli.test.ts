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

import {exec as execSync} from 'child_process';
import {glob} from 'glob';
import * as fs from 'node:fs/promises';
import * as path from 'path';
import {promisify} from 'util';
import {beforeAll, expect, suite, test} from 'vitest';

const exec = promisify(execSync);

const resultPath = path.resolve(import.meta.dirname, './results/cli/')

beforeAll(async () => {
  await fs.mkdir(resultPath, {recursive: true});
});

const execCLI =
    async (infile: string, outfile: string = `${resultPath}/${infile}.glb`) => {
  const cmd = [
    '../bin/manifold-cad', `"./fixtures/${infile}"`, `"${outfile}"`
  ].join(' ');
  return await exec(cmd, {cwd: import.meta.dirname});
};

suite('When executed, the CLI will', () => {
  test('Successfully build a known-good model', async () => {
    await expect(execCLI('unitSphere.mjs')).resolves.toBeDefined();
  });

  test('Fail to build a known-bad model', async () => {
    await expect(execCLI('unitSphereNoManifoldImport.mjs')).rejects.toThrow();
  });
});