import path from 'path';

import type {Program, Statement} from './ast.js';
import {Lexer} from './lexer.js';
import {Parser} from './parser.js';
import {globalFileResolver} from './state.js';
import type {ExternalLibraryRef, LibraryClosure, LibraryEdge, ResolvedProgram, ResolvedProgramWithLibraries, ScadFileHit} from './types.js';

// FONTPATH as set in the user's shell/OS environment
export async function getFontPath(): Promise<string|undefined> {
  const fp = await globalFileResolver?.fontPath();
  return fp && fp.trim() !== '' ? fp.trim() : undefined;
}

// Keep only what a `use` imports. OpenSCAD compiles the used file separately
// and takes just its declarations, so its top-level actions never run
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

// Parses an import with OpenSCAD rules: `include` errors propagate, while `use`
// errors discard the imported file but allow the current file to keep
// compiling.
async function importAtBoundary(
    kind: 'include'|'use',
    load: () => Promise<Statement[]>): Promise<Statement[]> {
  if (kind === 'include') return await load();
  try {
    return await load();
  } catch (err) {
    console.warn(`Warning: ignoring use'd file: ${(err as Error).message}`);
    return [];
  }
}

export async function resolveProgram(entryFile: string):
    Promise<ResolvedProgram> {
  const resolvedFiles: string[] = [];
  const visited = new Set<string>();
  const entryAbsPath = path.resolve(entryFile);

  const statements = await resolveFile(
      entryAbsPath, 'include', visited, resolvedFiles,
      path.dirname(entryAbsPath));

  return {statements, resolvedFiles};
}

async function resolveFile(
    filePath: string,
    mode: 'include'|'use',
    visited: Set<string>,
    resolvedFiles: string[],
    entryDir: string,
    ): Promise<Statement[]> {
  const absPath = path.resolve(filePath);

  // Prevent circular includes
  if (visited.has(absPath)) return [];
  visited.add(absPath);

  if (!await globalFileResolver?.exists(absPath)) {
    console.warn(`Warning: could not find file: ${filePath}`);
    return [];
  }

  resolvedFiles.push(absPath);

  const code = await globalFileResolver?.readText(absPath)!;
  const program = parseFile(code as string, absPath);
  const result: Statement[] = [];
  // Declarations forming THIS file's own scope (its own plus those of files it
  // `include`s), collected in `use` mode so they can be wrapped as one scope.
  const ownScope: Statement[] = [];
  const fileDir = path.dirname(absPath);

  for (const stmt of program.statements) {
    if (stmt.kind === 'include' || stmt.kind === 'use') {
      const hit =
          await globalFileResolver?.findScadFile(stmt.path, fileDir, entryDir);
      if (hit) {
        const imported = await importAtBoundary(
            stmt.kind,
            () => resolveFile(
                hit.path, stmt.kind, visited, resolvedFiles, entryDir));
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

export async function resolveProgramWithLibraries(entryFile: string):
    Promise<ResolvedProgramWithLibraries> {
  const resolvedFiles: string[] = [];
  const visited = new Set<string>();
  const externalLibraries = new Map<string, ExternalLibraryRef>();
  const entryAbsPath = path.resolve(entryFile);

  const statements = await resolveConsumerFile(
      entryAbsPath, 'include', visited, resolvedFiles,
      path.dirname(entryAbsPath), externalLibraries);

  return {statements, resolvedFiles, externalLibraries};
}

function recordExternalLibrary(
    externalLibraries: Map<string, ExternalLibraryRef>, hit: ScadFileHit,
    mode: 'include'|'use'): void {
  const name = hit.libraryName!;
  let ref = externalLibraries.get(name);
  if (!ref) {
    ref = {name, root: hit.libraryRoot!, entries: []};
    externalLibraries.set(name, ref);
  }
  if (!ref.entries.some(e => e.file === hit.path && e.mode === mode)) {
    ref.entries.push({file: hit.path, mode});
  }
}

async function resolveConsumerFile(
    filePath: string, mode: 'include'|'use', visited: Set<string>,
    resolvedFiles: string[], entryDir: string,
    externalLibraries: Map<string, ExternalLibraryRef>): Promise<Statement[]> {
  const absPath = path.resolve(filePath);
  if (visited.has(absPath)) return [];
  visited.add(absPath);

  if (!await globalFileResolver?.exists(absPath)) {
    console.warn(`Warning: could not find file: ${filePath}`);
    return [];
  }
  resolvedFiles.push(absPath);

  const code = await globalFileResolver?.readText(absPath)!;
  const program = parseFile(code as string, absPath);

  const result: Statement[] = [];
  const ownScope: Statement[] = [];
  const fileDir = path.dirname(absPath);

  for (const stmt of program.statements) {
    if (stmt.kind === 'include' || stmt.kind === 'use') {
      const hit =
          await globalFileResolver?.findScadFile(stmt.path, fileDir, entryDir);
      if (!hit) {
        console.warn(`Warning: could not resolve ${stmt.kind} <${
            stmt.path}> from ${filePath}`);
        continue;
      }
      if (hit.libraryName) {
        // External library: not to be inlined
        recordExternalLibrary(externalLibraries, hit, stmt.kind);
      } else {
        // Local file: inline
        const sub = await importAtBoundary(
            stmt.kind,
            () => resolveConsumerFile(
                hit.path, stmt.kind, visited, resolvedFiles, entryDir,
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

export async function resolveLibraryClosure(
    name: string, libraryRoot: string, entryFiles: string[],
    entryDir: string): Promise<LibraryClosure> {
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

  const walk = async (absPath: string) => {
    const abs = path.resolve(absPath);
    const rel = relOf(abs);
    if (visited.has(abs)) return;
    visited.add(abs);
    if (!await globalFileResolver?.exists(abs)) {
      console.warn(`Warning: library file not found: ${abs}`);
      return;
    }
    const code = await globalFileResolver?.readText(abs)!;
    const program = parseFile(code as string, abs);
    files.set(rel, program);

    const fileDeps: string[] = [];
    const fileEdges: LibraryEdge[] = [];
    const fileDir = path.dirname(abs);
    for (const stmt of program.statements) {
      if (stmt.kind === 'include' || stmt.kind === 'use') {
        const hit = await globalFileResolver?.findScadFile(
            stmt.path, fileDir, entryDir);
        if (hit && underRoot(hit.path)) {
          const depRel = relOf(path.resolve(hit.path));
          if (!fileDeps.includes(depRel)) fileDeps.push(depRel);
          if (!fileEdges.some(e => e.rel === depRel && e.mode === stmt.kind))
            fileEdges.push({rel: depRel, mode: stmt.kind});
          await walk(hit.path);
        }
      }
    }
    deps.set(rel, fileDeps);
    edges.set(rel, fileEdges);
  };

  const entryRels: string[] = [];
  for (const entry of entryFiles) {
    entryRels.push(relOf(path.resolve(entry)));
    await walk(entry);
  }
  return {name, root, files, deps, edges, entryRels};
}
