import fs from 'fs';
import os from 'os';
import path from 'path';

import type {Expr, ListCompGenerator, Program, Statement} from './ast.js';
import {Lexer} from './lexer.js';
import {Parser} from './parser.js';

let dotEnvLoaded = false;
export function loadDotEnv(): void {
  if (dotEnvLoaded) return;
  dotEnvLoaded = true;
  const envFile = path.resolve(process.cwd(), '.env');
  if (!fs.existsSync(envFile)) return;
  const content = fs.readFileSync(envFile, 'utf8');
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx < 0) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    const val = trimmed.slice(eqIdx + 1).trim();
    if (key && process.env[key] === undefined) {
      process.env[key] = val;
    }
  }
}

export function getFontPath(): string|undefined {
  loadDotEnv();
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
  loadDotEnv();
  const paths: string[] = [];

  // OPENSCADPATH environment variable
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

let usedFileScopeCounter = 0;

function privatizeUsedScope(scopeStmts: Statement[]): void {
  const varNames = new Set<string>();
  for (const s of scopeStmts) {
    if (s.kind === 'variableDecl') varNames.add(s.name);
  }
  if (varNames.size === 0) return;
  const suffix = `$u${usedFileScopeCounter++}`;
  for (const s of scopeStmts) renameStmt(s, varNames, suffix, new Set());
}

function renamed(
    name: string, vars: Set<string>, suffix: string,
    bound: Set<string>): string {
  return vars.has(name) && !bound.has(name) ? name + suffix : name;
}

function scopeWith(bound: Set<string>, stmts: Statement[]): Set<string> {
  const inner = new Set(bound);
  for (const s of stmts) {
    if (s.kind === 'variableDecl' || s.kind === 'moduleDecl' ||
        s.kind === 'functionDecl') {
      inner.add(s.name);
    }
  }
  return inner;
}

function renameStmt(
    stmt: Statement, vars: Set<string>, suffix: string,
    bound: Set<string>): void {
  switch (stmt.kind) {
    case 'variableDecl':
      stmt.name = renamed(stmt.name, vars, suffix, bound);
      renameExpr(stmt.value, vars, suffix, bound);
      break;
    case 'functionDecl': {
      const inner = new Set(bound);
      for (const p of stmt.params) {
        if (p.defaultValue) renameExpr(p.defaultValue, vars, suffix, bound);
        inner.add(p.name);
      }
      renameExpr(stmt.body, vars, suffix, inner);
      break;
    }
    case 'moduleDecl': {
      const inner = new Set(bound);
      for (const p of stmt.params) {
        if (p.defaultValue) renameExpr(p.defaultValue, vars, suffix, bound);
        inner.add(p.name);
      }
      renameStmt(stmt.body, vars, suffix, inner);
      break;
    }
    case 'moduleCall':
      for (const a of stmt.args) renameExpr(a.value, vars, suffix, bound);
      if (stmt.child) renameStmt(stmt.child, vars, suffix, bound);
      break;
    case 'block': {
      const inner = scopeWith(bound, stmt.statements);
      for (const s of stmt.statements) renameStmt(s, vars, suffix, inner);
      break;
    }
    case 'for': {
      const inner = new Set(bound);
      for (const v of stmt.variables) {
        renameExpr(v.range, vars, suffix, bound);
        inner.add(v.name);
      }
      renameStmt(stmt.body, vars, suffix, inner);
      break;
    }
    case 'if':
      renameExpr(stmt.condition, vars, suffix, bound);
      renameStmt(stmt.thenBody, vars, suffix, bound);
      if (stmt.elseBody) renameStmt(stmt.elseBody, vars, suffix, bound);
      break;
  }
}

function renameExpr(
    expr: Expr|undefined, vars: Set<string>, suffix: string,
    bound: Set<string>): void {
  if (!expr) return;
  switch (expr.kind) {
    case 'identifier':
      expr.name = renamed(expr.name, vars, suffix, bound);
      break;
    case 'number':
    case 'string':
    case 'boolean':
    case 'undef':
      break;
    case 'vector':
      for (const e of expr.elements) renameExpr(e, vars, suffix, bound);
      break;
    case 'range':
      renameExpr(expr.start, vars, suffix, bound);
      renameExpr(expr.end, vars, suffix, bound);
      renameExpr(expr.step, vars, suffix, bound);
      break;
    case 'binary':
      renameExpr(expr.left, vars, suffix, bound);
      renameExpr(expr.right, vars, suffix, bound);
      break;
    case 'unary':
      renameExpr(expr.operand, vars, suffix, bound);
      break;
    case 'group':
    case 'each':
      renameExpr(expr.expr, vars, suffix, bound);
      break;
    case 'ternary':
      renameExpr(expr.condition, vars, suffix, bound);
      renameExpr(expr.ifTrue, vars, suffix, bound);
      renameExpr(expr.ifFalse, vars, suffix, bound);
      break;
    case 'call':
      // expr.name is a function name (exported by `use`), never a variable.
      for (const a of expr.args) renameExpr(a.value, vars, suffix, bound);
      break;
    case 'index':
      renameExpr(expr.object, vars, suffix, bound);
      renameExpr(expr.index, vars, suffix, bound);
      break;
    case 'member':
      renameExpr(expr.object, vars, suffix, bound);
      break;
    case 'echo':
    case 'assert':
      for (const a of expr.args) renameExpr(a.value, vars, suffix, bound);
      renameExpr(expr.expr, vars, suffix, bound);
      break;
    case 'let': {
      const inner = new Set(bound);
      for (const a of expr.assignments) {
        renameExpr(a.value, vars, suffix, inner);
        inner.add(a.name);
      }
      renameExpr(expr.body, vars, suffix, inner);
      break;
    }
    case 'lambda': {
      const inner = new Set(bound);
      for (const p of expr.params) {
        if (p.defaultValue) renameExpr(p.defaultValue, vars, suffix, bound);
        inner.add(p.name);
      }
      renameExpr(expr.body, vars, suffix, inner);
      break;
    }
    case 'dynCall':
      renameExpr(expr.callee, vars, suffix, bound);
      for (const a of expr.args) renameExpr(a.value, vars, suffix, bound);
      break;
    case 'listComp':
      renameGenerator(expr.generator, vars, suffix, bound);
      break;
  }
}

function renameGenerator(
    gen: ListCompGenerator, vars: Set<string>, suffix: string,
    bound: Set<string>): void {
  switch (gen.kind) {
    case 'lcFor': {
      const inner = new Set(bound);
      for (const v of gen.variables) {
        renameExpr(v.range, vars, suffix, bound);
        inner.add(v.name);
      }
      renameGenerator(gen.body, vars, suffix, inner);
      break;
    }
    case 'lcCFor': {
      const inner = new Set(bound);
      for (const a of gen.inits) {
        renameExpr(a.value, vars, suffix, inner);
        inner.add(a.name);
      }
      renameExpr(gen.condition, vars, suffix, inner);
      for (const a of gen.updates) renameExpr(a.value, vars, suffix, inner);
      renameGenerator(gen.body, vars, suffix, inner);
      break;
    }
    case 'lcIf':
      renameExpr(gen.condition, vars, suffix, bound);
      renameGenerator(gen.ifTrue, vars, suffix, bound);
      if (gen.ifFalse) renameGenerator(gen.ifFalse, vars, suffix, bound);
      break;
    case 'lcLet': {
      const inner = new Set(bound);
      for (const a of gen.assignments) {
        renameExpr(a.value, vars, suffix, inner);
        inner.add(a.name);
      }
      renameGenerator(gen.body, vars, suffix, inner);
      break;
    }
    case 'lcExpr':
      renameExpr(gen.expr, vars, suffix, bound);
      break;
  }
}

export function resolveProgram(
    entryFile: string,
    libraryPaths: string[] = [],
    ): ResolvedProgram {
  const resolvedFiles: string[] = [];
  const visited = new Set<string>();
  const entryAbsPath = path.resolve(entryFile);

  const statements = resolveFile(
      entryAbsPath, 'include', visited, resolvedFiles, libraryPaths,
      entryAbsPath);

  return {statements, resolvedFiles};
}

function resolveFile(
    filePath: string,
    mode: 'include'|'use',
    visited: Set<string>,
    resolvedFiles: string[],
    libraryPaths: string[],
    entryAbsPath: string,
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
  const lexer = new Lexer(code, absPath);
  const parser = new Parser(lexer);

  let program: Program;
  try {
    program = parser.parseProgram();
  } catch (err) {
    const msg = `failed to parse ${absPath}: ${(err as Error).message}`;
    if (absPath === entryAbsPath) {
      throw new Error(msg);
    }
    console.warn(`Warning: ${msg}`);
    return [];
  }

  for (const warning of parser.warnings) {
    console.warn(
        `Warning: dropped unparseable statement in ${absPath}: ${warning}`);
  }

  const result: Statement[] = [];
  // Statements forming THIS file's own scope (its own declarations plus the
  // contents of files it `include`s). Used only in `use` mode to privatize the
  // file's top-level variables; a nested `use` is its own already-isolated
  // scope.
  const ownScope: Statement[] = [];
  const fileDir = path.dirname(absPath);

  for (const stmt of program.statements) {
    if (stmt.kind === 'include' || stmt.kind === 'use') {
      const resolvedPath = resolveIncludePath(stmt.path, fileDir, libraryPaths);
      if (resolvedPath) {
        const imported = resolveFile(
            resolvedPath, stmt.kind, visited, resolvedFiles, libraryPaths,
            entryAbsPath);
        result.push(...imported);
        if (mode === 'use' && stmt.kind === 'include')
          ownScope.push(...imported);
      } else {
        console.warn(`Warning: could not resolve ${stmt.kind} <${
            stmt.path}> from ${filePath}`);
      }
      continue;
    }

    if (mode === 'use') {
      // `use` imports module and function declarations. Top-level variables are
      // kept (so those functions/modules can close over them) but privatized
      // below so they don't leak into the consumer's scope.
      if (stmt.kind === 'moduleDecl' || stmt.kind === 'functionDecl' ||
          stmt.kind === 'variableDecl') {
        if (stmt.kind === 'variableDecl' && stmt.name.startsWith('$')) continue;
        result.push(stmt);
        ownScope.push(stmt);
      }
    } else {
      // include imports everything
      result.push(stmt);
    }
  }

  if (mode === 'use') privatizeUsedScope(ownScope);

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
      entryAbsPath, externalLibraries);

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
    resolvedFiles: string[], libraryPaths: string[], entryAbsPath: string,
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
  const parser = new Parser(new Lexer(code, absPath));
  let program: Program;
  try {
    program = parser.parseProgram();
  } catch (err) {
    const msg = `failed to parse ${absPath}: ${(err as Error).message}`;
    if (absPath === entryAbsPath) throw new Error(msg);
    console.warn(`Warning: ${msg}`);
    return [];
  }
  for (const warning of parser.warnings) {
    console.warn(
        `Warning: dropped unparseable statement in ${absPath}: ${warning}`);
  }

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
        const sub = resolveConsumerFile(
            cls.resolved, stmt.kind, visited, resolvedFiles, libraryPaths,
            entryAbsPath, externalLibraries);
        result.push(...sub);
        if (mode === 'use' && stmt.kind === 'include') ownScope.push(...sub);
      }
      continue;
    }

    if (mode === 'use') {
      if (stmt.kind === 'moduleDecl' || stmt.kind === 'functionDecl' ||
          stmt.kind === 'variableDecl') {
        if (stmt.kind === 'variableDecl' && stmt.name.startsWith('$')) continue;
        result.push(stmt);
        ownScope.push(stmt);
      }
    } else {
      result.push(stmt);
    }
  }

  if (mode === 'use') privatizeUsedScope(ownScope);

  return result;
}

export interface LibraryClosure {
  name: string;
  root: string;
  files: Map<string, Program>;
  deps: Map<string, string[]>;
  entryRels: string[];
}

export function resolveLibraryClosure(
    name: string, libraryRoot: string, entryFiles: string[],
    libraryPaths: string[]): LibraryClosure {
  const root = path.resolve(libraryRoot);
  const files = new Map<string, Program>();
  const deps = new Map<string, string[]>();
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
    const parser = new Parser(new Lexer(code, abs));
    let program: Program;
    try {
      program = parser.parseProgram();
    } catch (err) {
      console.warn(`Warning: failed to parse library file ${abs}: ${
          (err as Error).message}`);
      return;
    }
    for (const warning of parser.warnings) {
      console.warn(
          `Warning: dropped unparseable statement in ${abs}: ${warning}`);
    }
    files.set(rel, program);

    const fileDeps: string[] = [];
    const fileDir = path.dirname(abs);
    for (const stmt of program.statements) {
      if (stmt.kind === 'include' || stmt.kind === 'use') {
        const cls = classifyIncludePath(stmt.path, fileDir, libraryPaths);
        if (cls && underRoot(cls.resolved)) {
          const depRel = relOf(path.resolve(cls.resolved));
          if (!fileDeps.includes(depRel)) fileDeps.push(depRel);
          walk(cls.resolved);
        }
      }
    }
    deps.set(rel, fileDeps);
  };

  const entryRels: string[] = [];
  for (const entry of entryFiles) {
    entryRels.push(relOf(path.resolve(entry)));
    walk(entry);
  }
  return {name, root, files, deps, entryRels};
}
