import fs from "fs";
import path from "path";
import os from "os";
import { Lexer } from "./lexer.js";
import { Parser } from "./parser.js";
import type { Program, Statement } from "./ast.js";

// Resolves and recursively parses OpenSCAD include/use directives
export interface ResolvedProgram {
  // All statements from included files + the main file, in order
  statements: Statement[];
  // Paths of all files that were resolved (for debugging/caching)
  resolvedFiles: string[];
}

export function getOpenSCADLibraryPaths(): string[] {
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

// Resolve an include/use path against (The directory of the current file, Each library search path)
function resolveIncludePath(
  includePath: string,
  currentDir: string,
  libraryPaths: string[],
): string | undefined {
  // Try relative to current file first
  const relative = path.resolve(currentDir, includePath);
  if (fs.existsSync(relative)) return relative;

  // Try each library path
  for (const libPath of libraryPaths) {
    const candidate = path.resolve(libPath, includePath);
    if (fs.existsSync(candidate)) return candidate;
  }

  return undefined;
}
