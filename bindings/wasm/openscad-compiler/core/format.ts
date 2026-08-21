import {execFileSync} from 'child_process';
import fs from 'fs';
import {createRequire} from 'module';
import path from 'path';


export function formatWritten(file: string): void {
  try {
    const here = typeof __dirname !== 'undefined' ?
        __dirname :
        path.dirname(
            new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/i, '$1'));
    const projectRoot = path.resolve(here, '..');

    const require = createRequire(path.join(projectRoot, 'package.json'));
    const bin = path.join(
        path.dirname(require.resolve('prettier')), 'bin/prettier.cjs');
    if (!fs.existsSync(bin)) return;

    execFileSync(
        process.execPath,
        [bin, '--ignore-path=.prettierignore-absent', '--write', file],
        {stdio: 'ignore'});
  } catch {
    // unindented code without prettier
  }
}
