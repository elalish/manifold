import path from 'path-browserify';

import type {FileResolver, PathResolver, ScadFileHit} from '../core/types.js';
import {fetchAndSaveLibrary} from '../vfs/github.js';
import {vfs} from '../vfs/vfs.js';

// Directories searched for include <...>/use <...>: the file's folder, working
// directory, then OPENSCADPATH
function searchRoots(entryDir: string): string[] {
  return [
    entryDir,
    '/',
    '/openscad_libs',
  ];
}

function fileHit(filePath: string): ScadFileHit {
  const libraryRel = path.relative('/openscad_libs', filePath);
  if (libraryRel && libraryRel !== '..' && !libraryRel.startsWith('../') &&
      !path.isAbsolute(libraryRel)) {
    const libraryName = libraryRel.split('/')[0]!;
    return {
      path: filePath,
      libraryName,
      libraryRoot: path.join('/openscad_libs', libraryName),
    };
  }
  return {path: filePath};
}

// Resolves include/use paths
async function findScadFile(
    includePath: string, fromDir: string,
    entryDir: string): Promise<ScadFileHit|undefined> {
  const normalized = includePath.replace(/\\/g, '/');
  for (const root of [fromDir, ...searchRoots(entryDir)]) {
    const candidate = path.resolve(root, normalized);
    if (vfs.isFileSync(candidate)) return fileHit(candidate);
  }

  const segments = normalized.split('/');
  const libraryName = segments[0];
  if (!path.isAbsolute(normalized) && segments.length > 1 && libraryName &&
      !segments.some(part => part === '.' || part === '..')) {
    const libRoot = path.join('/openscad_libs', libraryName);
    if (!vfs.existsSync(libRoot)) await fetchAndSaveLibrary(libraryName);
    const candidate = path.resolve('/openscad_libs', normalized);
    if (vfs.isFileSync(candidate)) return fileHit(candidate);
    throw new Error(`Library file not found: ${includePath}`);
  }

  return undefined;
}

export const webFileResolver: FileResolver = {
  readText(filePath: string): Promise<string|null> {
    if (!vfs.isFileSync(filePath)) return Promise.resolve(null);
    try {
      return Promise.resolve(vfs.readFileSync(filePath, 'utf8') as string);
    } catch (err) {
      return Promise.reject(err);
    }
  },
  exists(filePath: string): Promise<boolean> {
    try {
      return Promise.resolve(vfs.isFileSync(filePath));
    } catch (err) {
      return Promise.reject(err);
    }
  },
  findScadFile(includePath: string, fromDir: string, entryDir: string) {
    return findScadFile(includePath, fromDir, entryDir);
  },
  getSurfaceFilePath(filenameStr: string, sourceFile: string): Promise<string> {
    const base = '/';
    const basePath =
        sourceFile ? path.dirname(path.resolve(base, sourceFile)) : base;
    return Promise.resolve(path.resolve(basePath, filenameStr));
  }
}

// in web mode all compiler paths are rooted at the VFS root
const vfsPath: typeof path = {
  ...path,
  resolve(...paths: string[]) {
    return path.resolve('/', ...paths);
  },
  relative(from: string, to: string) {
    return path.relative(path.resolve('/', from), path.resolve('/', to));
  },
};

export const webPathResolver: PathResolver = {
  path: vfsPath
};
