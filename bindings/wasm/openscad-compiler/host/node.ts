import fs from 'fs';
import path from 'path';

import type {FileResolver, ScadFileHit} from '../core/types.js';


// Directories searched for include <...>/use <...>: the file's folder, working
// directory, then OPENSCADPATH
function searchRoots(entryDir: string): string[] {
  return [
    entryDir,
    process.cwd(),
    ...(process.env.OPENSCADPATH?.split(path.delimiter) ?? []),
  ];
}

// Resolves include/use paths
function findScadFile(includePath: string, fromDir: string, entryDir: string):
    ScadFileHit|undefined {
  const relative = path.resolve(fromDir, includePath);
  if (fs.existsSync(relative)) return {path: relative};

  const firstSegment = includePath.replace(/\\/g, '/').split('/')[0] || '';
  for (const root of searchRoots(entryDir)) {
    const candidate = path.resolve(root, includePath);
    if (!fs.existsSync(candidate)) continue;
    if (firstSegment && firstSegment !== '.' && firstSegment !== '..') {
      return {
        path: candidate,
        libraryName: firstSegment,
        libraryRoot: path.resolve(root, firstSegment),
      };
    }
    return {path: candidate};
  }

  return undefined;
}

export const nodeFileResolver: FileResolver = {
  readText(filePath: string): Promise<string|null> {
    if (!fs.existsSync(filePath)) return Promise.resolve(null);
    try {
      return Promise.resolve(fs.readFileSync(filePath, 'utf8'));
    } catch (err) {
      return Promise.reject(err);
    }
  },
  exists(filePath: string): Promise<boolean> {
    try {
      return Promise.resolve(fs.existsSync(filePath));
    } catch (err) {
      return Promise.reject(err);
    }
  },
  findScadFile(includePath: string, fromDir: string, entryDir: string) {
    return Promise.resolve(findScadFile(includePath, fromDir, entryDir));
  },
  getSurfaceFilePath(filenameStr: string, sourceFile: string): Promise<string> {
    const base = process.cwd();
    const basePath =
        sourceFile ? path.dirname(path.resolve(base, sourceFile)) : base;
    return Promise.resolve(path.resolve(basePath, filenameStr));
  }
}
