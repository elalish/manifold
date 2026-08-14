import fs from 'fs';
import os from 'os';
import path from 'path';

import type {Program, Statement} from './ast.js';
import {Lexer} from './lexer.js';
import {Parser} from './parser.js';

// FONTPATH as set in the user's shell/OS environment
export function getFontPath(): string|undefined {
  const fp = process.env['FONTPATH'];
  return fp && fp.trim() !== '' ? fp.trim() : undefined;
}

// Resolves and recursively parses OpenSCAD include/use directives
export interface ResolvedProgram {
  // All statements from included files + the main file, in order
  statements: Statement[];
  // Paths of all files that were resolved (for debugging/caching)
  resolvedFiles: string[];
}

export function getOpenSCADLibraryPaths(): string[] {
  const paths: string[] = [];

  if (process.env.OPENSCADPATH) {
    paths.push(...process.env.OPENSCADPATH.split(path.delimiter));
  }

  // User library paths by OS
  const home = os.homedir();
  if (process.platform === 'win32') {
    paths.push(path.join(home, 'Documents', 'OpenSCAD', 'libraries'));
    paths.push(
        path.join(home, 'OneDrive', 'Documents', 'OpenSCAD', 'libraries'));
  } else if (process.platform === 'darwin') {
    paths.push(path.join(home, 'Documents', 'OpenSCAD', 'libraries'));
  } else {
    // Linux and others
    paths.push(path.join(home, '.local', 'share', 'OpenSCAD', 'libraries'));
  }

  // Filter to keep only those that actually exist
  return paths.filter(p => fs.existsSync(p));
}

// Keep only what a `use` imports. OpenSCAD compiles the used file separately and takes just its declarations, so its top-level actions never run
function importableDecls(stmts: Statement[]): Statement[] {
  return stmts.filter(
      s => s.kind === 'moduleDecl' || s.kind === 'functionDecl' ||
          (s.kind === 'variableDecl' && !s.name.startsWith('$')));
}

// Wrap a used file's declarations as one scope, so the compiler can emit them
// into a JS scope of their own and keep the file's top-level variables private
// without renaming anything. A file with no top-level variables has nothing to
// keep private, so it stays flat.
function usedFileScope(ownScope: Statement[]): Statement[] {
  const hasPrivateVars = ownScope.some(s => s.kind === 'variableDecl');
  if (!hasPrivateVars) return ownScope;
  const first = ownScope[0];
  return [{
    kind: 'scope',
    statements: ownScope,
    ...(first?.filename !== undefined ? {filename: first.filename} : {}),
    ...(first?.loc !== undefined ? {loc: first.loc} : {}),
  }];
}

function parseFile(code: string, absPath: string): Program {
  try {
    return new Parser(new Lexer(code, absPath)).parseProgram();
  } catch (err) {
    throw new Error(`failed to parse ${absPath}: ${(err as Error).message}`);
  }
}

// Parses an import with OpenSCAD rules: `include` errors propagate, while `use` errors discard the imported file but allow the current file to keep compiling.
function importAtBoundary(
    kind: 'include'|'use', load: () => Statement[]): Statement[] {
  if (kind === 'include') return load();
  try {
    return load();
  } catch (err) {
    console.warn(`Warning: ignoring use'd file: ${(err as Error).message}`);
    return [];
  }
}

export function resolveProgram(
    entryFile: string,
    libraryPaths: string[] = [],
    ): ResolvedProgram {
  const resolvedFiles: string[] = [];
  const visited = new Set<string>();
  const entryAbsPath = path.resolve(entryFile);

  const statements =
      resolveFile(entryAbsPath, 'include', visited, resolvedFiles, libraryPaths);

  return {statements, resolvedFiles};
}

function resolveFile(
    filePath: string,
    mode: 'include'|'use',
    visited: Set<string>,
    resolvedFiles: string[],
    libraryPaths: string[],
    ): Statement[] {
  const absPath = path.resolve(filePath);

  // Prevent circular includes
  if (visited.has(absPath)) return [];
  visited.add(absPath);

  if (!fs.existsSync(absPath)) {
    console.warn(`Warning: could not find file: ${filePath}`);
    return [];
  }

  resolvedFiles.push(absPath);

  const code = fs.readFileSync(absPath, 'utf8');
  const program = parseFile(code, absPath);
  const result: Statement[] = [];
  // Declarations forming THIS file's own scope (its own plus those of files it `include`s), collected in `use` mode so they can be wrapped as one scope.
  const ownScope: Statement[] = [];
  const fileDir = path.dirname(absPath);

  for (const stmt of program.statements) {
    if (stmt.kind === 'include' || stmt.kind === 'use') {
      const resolvedPath = resolveIncludePath(stmt.path, fileDir, libraryPaths);
      if (resolvedPath) {
        const imported = importAtBoundary(
            stmt.kind,
            () => resolveFile(
                resolvedPath, stmt.kind, visited, resolvedFiles, libraryPaths));
        if (mode === 'use' && stmt.kind === 'include') {
          ownScope.push(...importableDecls(imported));
        } else {
          result.push(...imported);
        }
      } else {
        console.warn(`Warning: could not resolve ${stmt.kind} <${
            stmt.path}> from ${filePath}`);
      }
      continue;
    }

    if (mode === 'use') {
      ownScope.push(...importableDecls([stmt]));
    } else {
      // include imports everything
      result.push(stmt);
    }
  }

  if (mode === 'use') result.push(...usedFileScope(ownScope));

  return result;
}

function resolveIncludePath(
    includePath: string,
    currentDir: string,
    libraryPaths: string[],
    ): string|undefined {
  return classifyIncludePath(includePath, currentDir, libraryPaths)?.resolved;
}

interface IncludeClassification {
  resolved: string;
  libraryName?: string;
  libraryRoot?: string;
}

function classifyIncludePath(
    includePath: string, currentDir: string,
    libraryPaths: string[]): IncludeClassification|undefined {
  // Relative to the current file always takes precedence and is never external.
  const relative = path.resolve(currentDir, includePath);
  if (fs.existsSync(relative)) return {resolved: relative};

  const firstSegment = includePath.replace(/\\/g, '/').split('/')[0] || '';
  for (const libPath of libraryPaths) {
    const candidate = path.resolve(libPath, includePath);
    if (fs.existsSync(candidate)) {
      if (firstSegment && firstSegment !== '.' && firstSegment !== '..') {
        return {
          resolved: candidate,
          libraryName: firstSegment,
          libraryRoot: path.resolve(libPath, firstSegment),
        };
      }
      return {resolved: candidate};
    }
  }

  return undefined;
}

export interface ExternalLibraryRef {
  name: string;
  root: string;
  entries: {file: string; mode: 'include' | 'use'}[];
}

export interface ResolvedProgramWithLibraries extends ResolvedProgram {
  externalLibraries: Map<string, ExternalLibraryRef>;
}

export function resolveProgramWithLibraries(
    entryFile: string,
    libraryPaths: string[] = []): ResolvedProgramWithLibraries {
  const resolvedFiles: string[] = [];
  const visited = new Set<string>();
  const externalLibraries = new Map<string, ExternalLibraryRef>();
  const entryAbsPath = path.resolve(entryFile);

  const statements = resolveConsumerFile(
      entryAbsPath, 'include', visited, resolvedFiles, libraryPaths,
      externalLibraries);

  return {statements, resolvedFiles, externalLibraries};
}

function recordExternalLibrary(
    externalLibraries: Map<string, ExternalLibraryRef>,
    cls: IncludeClassification, mode: 'include'|'use'): void {
  const name = cls.libraryName!;
  let ref = externalLibraries.get(name);
  if (!ref) {
    ref = {name, root: cls.libraryRoot!, entries: []};
    externalLibraries.set(name, ref);
  }
  if (!ref.entries.some(e => e.file === cls.resolved && e.mode === mode)) {
    ref.entries.push({file: cls.resolved, mode});
  }
}

function resolveConsumerFile(
    filePath: string, mode: 'include'|'use', visited: Set<string>,
    resolvedFiles: string[], libraryPaths: string[],
    externalLibraries: Map<string, ExternalLibraryRef>): Statement[] {
  const absPath = path.resolve(filePath);
  if (visited.has(absPath)) return [];
  visited.add(absPath);

  if (!fs.existsSync(absPath)) {
    console.warn(`Warning: could not find file: ${filePath}`);
    return [];
  }
  resolvedFiles.push(absPath);

  const code = fs.readFileSync(absPath, 'utf8');
  const program = parseFile(code, absPath);

  const result: Statement[] = [];
  const ownScope: Statement[] = [];
  const fileDir = path.dirname(absPath);

  for (const stmt of program.statements) {
    if (stmt.kind === 'include' || stmt.kind === 'use') {
      const cls = classifyIncludePath(stmt.path, fileDir, libraryPaths);
      if (!cls) {
        console.warn(`Warning: could not resolve ${stmt.kind} <${
            stmt.path}> from ${filePath}`);
        continue;
      }
      if (cls.libraryName) {
        // External library: not to be inlined
        recordExternalLibrary(externalLibraries, cls, stmt.kind);
      } else {
        // Local file: inline
        const sub = importAtBoundary(
            stmt.kind,
            () => resolveConsumerFile(
                cls.resolved, stmt.kind, visited, resolvedFiles, libraryPaths,
                externalLibraries));
        if (mode === 'use' && stmt.kind === 'include') {
          ownScope.push(...importableDecls(sub));
        } else {
          result.push(...sub);
        }
      }
      continue;
    }

    if (mode === 'use') {
      ownScope.push(...importableDecls([stmt]));
    } else {
      result.push(stmt);
    }
  }

  if (mode === 'use') result.push(...usedFileScope(ownScope));

  return result;
}

export interface LibraryEdge {
  rel: string;
  mode: 'include'|'use';
}

export interface LibraryClosure {
  name: string;
  root: string;
  files: Map<string, Program>;
  deps: Map<string, string[]>;
  edges: Map<string, LibraryEdge[]>;
  entryRels: string[];
}

export function resolveLibraryClosure(
    name: string, libraryRoot: string, entryFiles: string[],
    libraryPaths: string[]): LibraryClosure {
  const root = path.resolve(libraryRoot);
  const files = new Map<string, Program>();
  const deps = new Map<string, string[]>();
  const edges = new Map<string, LibraryEdge[]>();
  const visited = new Set<string>();

  const relOf = (abs: string) => path.relative(root, abs).replace(/\\/g, '/');
  const underRoot = (abs: string) => {
    const rel = path.relative(root, abs);
    return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
  };

  const walk = (absPath: string) => {
    const abs = path.resolve(absPath);
    const rel = relOf(abs);
    if (visited.has(abs)) return;
    visited.add(abs);
    if (!fs.existsSync(abs)) {
      console.warn(`Warning: library file not found: ${abs}`);
      return;
    }
    const code = fs.readFileSync(abs, 'utf8');
    const program = parseFile(code, abs);
    files.set(rel, program);

    const fileDeps: string[] = [];
    const fileEdges: LibraryEdge[] = [];
    const fileDir = path.dirname(abs);
    for (const stmt of program.statements) {
      if (stmt.kind === 'include' || stmt.kind === 'use') {
        const cls = classifyIncludePath(stmt.path, fileDir, libraryPaths);
        if (cls && underRoot(cls.resolved)) {
          const depRel = relOf(path.resolve(cls.resolved));
          if (!fileDeps.includes(depRel)) fileDeps.push(depRel);
          if (!fileEdges.some(e => e.rel === depRel && e.mode === stmt.kind))
            fileEdges.push({rel: depRel, mode: stmt.kind});
          walk(cls.resolved);
        }
      }
    }
    deps.set(rel, fileDeps);
    edges.set(rel, fileEdges);
  };

  const entryRels: string[] = [];
  for (const entry of entryFiles) {
    entryRels.push(relOf(path.resolve(entry)));
    walk(entry);
  }
  return {name, root, files, deps, edges, entryRels};
}
