import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {expect, suite, test} from 'vitest';

const execFileAsync = promisify(execFile);

suite('With code generation from strings disallowed, the module', () => {
  test('loads and runs', async () => {
    const probe = execFileAsync(
        process.execPath,
        ['--disallow-code-generation-from-strings', './fixtures/cspProbe.mjs'],
        {cwd: import.meta.dirname});
    await expect(probe).resolves.toBeDefined();
  });
});
