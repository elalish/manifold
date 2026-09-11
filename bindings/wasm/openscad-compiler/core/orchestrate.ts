import path from 'path';

import {compile} from './compiler.js';
import type {ResolvedExternalLib, ResolvedProgramWithLibraries} from './types.js';

export async function compileConsumer(
    entryFile: string, outputFile: string, cwd: string,
    externalLibraries: ResolvedExternalLib[],
    resolved: ResolvedProgramWithLibraries):
    Promise<{code: string; resolvedFiles: string[]}> {
  const entryAbs = path.resolve(entryFile);

  const outDir = path.dirname(path.resolve(outputFile));

  let relPath = path.relative(outDir, cwd);
  if (relPath === '') relPath = '.';
  let rp = relPath.replace(/\\/g, '/');
  if (!rp.startsWith('.') && !rp.startsWith('/')) rp = './' + rp;
  const runtimeJSPath = rp + '/runtime/runtime.js';

  const ast = {
    kind: 'program' as const,
    statements: resolved.statements,
    filename: entryAbs
  };
  const code =
      await compile(ast, {runtimePath: runtimeJSPath, externalLibraries});

  return {
    code,
    resolvedFiles: resolved.resolvedFiles,
  };
}
