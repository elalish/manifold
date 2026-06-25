import fs from "fs";
import path from "path";
import os from "os";
import { Lexer } from "./lexer.js";
import { Parser } from "./parser.js";
import type { Program, Statement } from "./ast.js";

let dotEnvLoaded = false;
export function loadDotEnv(): void {
  if (dotEnvLoaded) return;
  dotEnvLoaded = true;
  const envFile = path.resolve(process.cwd(), ".env");
  if (!fs.existsSync(envFile)) return;
  const content = fs.readFileSync(envFile, "utf8");
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx < 0) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    const val = trimmed.slice(eqIdx + 1).trim();
    if (key && process.env[key] === undefined) {
      process.env[key] = val;
    }
  }
}

export function getFontPath(): string | undefined {
  loadDotEnv();
  const fp = process.env["FONTPATH"];
  return fp && fp.trim() !== "" ? fp.trim() : undefined;
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
  if (process.platform === "win32") {
    paths.push(path.join(home, "Documents", "OpenSCAD", "libraries"));
    paths.push(path.join(home, "OneDrive", "Documents", "OpenSCAD", "libraries"));
  } else if (process.platform === "darwin") {
    paths.push(path.join(home, "Documents", "OpenSCAD", "libraries"));
  } else {
    // Linux and others
    paths.push(path.join(home, ".local", "share", "OpenSCAD", "libraries"));
  }

  // Filter to keep only those that actually exist
  return paths.filter(p => fs.existsSync(p));
}

export function resolveProgram(
  entryFile: string,
  libraryPaths: string[] = [],
): ResolvedProgram {
  const resolvedFiles: string[] = [];
  const visited = new Set<string>();
  const entryAbsPath = path.resolve(entryFile);

  const statements = resolveFile(entryAbsPath, "include", visited, resolvedFiles, libraryPaths, entryAbsPath);

  return { statements, resolvedFiles };
}

function resolveFile(
  filePath: string,
  mode: "include" | "use",
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

  const code = fs.readFileSync(absPath, "utf8");
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
    console.warn(`Warning: dropped unparseable statement in ${absPath}: ${warning}`);
  }

  const result: Statement[] = [];
  const fileDir = path.dirname(absPath);

  for (const stmt of program.statements) {
    if (stmt.kind === "include" || stmt.kind === "use") {
      const resolvedPath = resolveIncludePath(stmt.path, fileDir, libraryPaths);
      if (resolvedPath) {
        const imported = resolveFile(resolvedPath, stmt.kind, visited, resolvedFiles, libraryPaths, entryAbsPath);
        result.push(...imported);
      } else {
        console.warn(`Warning: could not resolve ${stmt.kind} <${stmt.path}> from ${filePath}`);
      }
      continue;
    }

    if (mode === "use") {
      // use only imports module and function declarations
      if (stmt.kind === "moduleDecl" || stmt.kind === "functionDecl") {
        result.push(stmt);
      }
    } else {
      // include imports everything
      result.push(stmt);
    }
  }

  return result;
}

function resolveIncludePath(
  includePath: string,
  currentDir: string,
  libraryPaths: string[],
): string | undefined {
  return classifyIncludePath(includePath, currentDir, libraryPaths)?.resolved;
}

interface IncludeClassification {
  resolved: string;
  libraryName?: string;
  libraryRoot?: string;
}

function classifyIncludePath(includePath: string, currentDir: string, libraryPaths: string[]): IncludeClassification | undefined {
  // Relative to the current file always takes precedence and is never external.
  const relative = path.resolve(currentDir, includePath);
  if (fs.existsSync(relative)) return { resolved: relative };

  const firstSegment = includePath.replace(/\\/g, "/").split("/")[0] || "";
  for (const libPath of libraryPaths) {
    const candidate = path.resolve(libPath, includePath);
    if (fs.existsSync(candidate)) {
      if (firstSegment && firstSegment !== "." && firstSegment !== "..") {
        return {
          resolved: candidate,
          libraryName: firstSegment,
          libraryRoot: path.resolve(libPath, firstSegment),
        };
      }
      return { resolved: candidate };
    }
  }

  return undefined;
}

export interface ExternalLibraryRef {
  name: string;
  root: string;
  entries: { file: string; mode: "include" | "use" }[];
}

export interface ResolvedProgramWithLibraries extends ResolvedProgram {
  externalLibraries: Map<string, ExternalLibraryRef>;
}

export function resolveProgramWithLibraries(entryFile: string, libraryPaths: string[] = []): ResolvedProgramWithLibraries {
  const resolvedFiles: string[] = [];
  const visited = new Set<string>();
  const externalLibraries = new Map<string, ExternalLibraryRef>();
  const entryAbsPath = path.resolve(entryFile);

  const statements = resolveConsumerFile(entryAbsPath, "include", visited, resolvedFiles, libraryPaths, entryAbsPath, externalLibraries);

  return { statements, resolvedFiles, externalLibraries };
}

function recordExternalLibrary(externalLibraries: Map<string, ExternalLibraryRef>, cls: IncludeClassification, mode: "include" | "use"): void {
  const name = cls.libraryName!;
  let ref = externalLibraries.get(name);
  if (!ref) {
    ref = { name, root: cls.libraryRoot!, entries: [] };
    externalLibraries.set(name, ref);
  }
  if (!ref.entries.some(e => e.file === cls.resolved && e.mode === mode)) {
    ref.entries.push({ file: cls.resolved, mode });
  }
}

function resolveConsumerFile(filePath: string, mode: "include" | "use", visited: Set<string>, resolvedFiles: string[], libraryPaths: string[], entryAbsPath: string, externalLibraries: Map<string, ExternalLibraryRef>): Statement[] {
  const absPath = path.resolve(filePath);
  if (visited.has(absPath)) return [];
  visited.add(absPath);

  if (!fs.existsSync(absPath)) {
    console.warn(`Warning: could not find file: ${filePath}`);
    return [];
  }
  resolvedFiles.push(absPath);

  const code = fs.readFileSync(absPath, "utf8");
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
    console.warn(`Warning: dropped unparseable statement in ${absPath}: ${warning}`);
  }

  const result: Statement[] = [];
  const fileDir = path.dirname(absPath);

  for (const stmt of program.statements) {
    if (stmt.kind === "include" || stmt.kind === "use") {
      const cls = classifyIncludePath(stmt.path, fileDir, libraryPaths);
      if (!cls) {
        console.warn(`Warning: could not resolve ${stmt.kind} <${stmt.path}> from ${filePath}`);
        continue;
      }
      if (cls.libraryName) {
        // External library: not to be inlined
        recordExternalLibrary(externalLibraries, cls, stmt.kind);
      } else {
        // Local file: inline
        result.push(...resolveConsumerFile(cls.resolved, stmt.kind, visited, resolvedFiles, libraryPaths, entryAbsPath, externalLibraries));
      }
      continue;
    }

    if (mode === "use") {
      if (stmt.kind === "moduleDecl" || stmt.kind === "functionDecl") result.push(stmt);
    } else {
      result.push(stmt);
    }
  }

  return result;
}

export interface LibraryClosure {
  name: string;
  root: string;
  files: Map<string, Program>;
  deps: Map<string, string[]>;
  entryRels: string[];
}

export function resolveLibraryClosure(name: string, libraryRoot: string, entryFiles: string[], libraryPaths: string[]): LibraryClosure {
  const root = path.resolve(libraryRoot);
  const files = new Map<string, Program>();
  const deps = new Map<string, string[]>();
  const visited = new Set<string>();

  const relOf = (abs: string) => path.relative(root, abs).replace(/\\/g, "/");
  const underRoot = (abs: string) => {
    const rel = path.relative(root, abs);
    return rel !== "" && !rel.startsWith("..") && !path.isAbsolute(rel);
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
    const code = fs.readFileSync(abs, "utf8");
    const parser = new Parser(new Lexer(code, abs));
    let program: Program;
    try {
      program = parser.parseProgram();
    } catch (err) {
      console.warn(`Warning: failed to parse library file ${abs}: ${(err as Error).message}`);
      return;
    }
    for (const warning of parser.warnings) {
      console.warn(`Warning: dropped unparseable statement in ${abs}: ${warning}`);
    }
    files.set(rel, program);

    const fileDeps: string[] = [];
    const fileDir = path.dirname(abs);
    for (const stmt of program.statements) {
      if (stmt.kind === "include" || stmt.kind === "use") {
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
  return { name, root, files, deps, entryRels };
}
