import path from 'path';

import type {Argument} from './ast.js';
import {compileExpr, findArg} from './expr.js';
import {globalFileResolver, RT} from './state.js';

// compile surface
export async function compileSurface(
    args: Argument[], sourceFile: string): Promise<string> {
  const file = findArg(args, 'file', 0);
  const center = findArg(args, 'center', 1);
  const invert = findArg(args, 'invert', 2);

  if (!file?.value || file.value.kind !== 'string') {
    console.warn(`Warning: surface(): no file argument given, ignoring.`);
    return 'Manifold.union([])';
  }
  const filenameStr = file.value.value;

  // surface() resolves files relative to the calling .scad; otherwise, the
  // host's base directory is used.
  const filePath =
      await globalFileResolver?.getSurfaceFilePath(filenameStr, sourceFile) ??
      '';

  const isImage = path.extname(filePath).toLowerCase() === '.png';

  const centerStr = center ? compileExpr(center.value) : 'false';
  // invert only reaches the image path; a text matrix has no pixels to flip
  const opts = isImage ?
      `center: ${centerStr}, invert: ${
          invert ? compileExpr(invert.value) : 'false'}, kind: "image"` :
      `center: ${centerStr}, kind: "text"`;

  // JSON.stringify escapes the separators, so a Windows path does not
  // turn its backslashes into escape sequences in the emitted string literal
  return `${RT.surface}(${JSON.stringify(filePath)}, { ${opts}, fn: ${
      RT.ctx}.$fn, fa: ${RT.ctx}.$fa, fs: ${RT.ctx}.$fs })`;
}
