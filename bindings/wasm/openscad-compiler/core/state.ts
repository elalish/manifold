import type {Expr, FunctionDeclStmt, Parameter} from './ast.js';
import {BUILTIN_FUNCTIONS, BUILTIN_MODULES, BUILTIN_VAR_CONSTANTS, RUNTIME_SYMBOLS,} from './builtins.js';
import type {BindOptions, BindResult, FileResolver, ModuleDeclStmtType, Scope,} from './types.js';

// Signatures
export interface Signature {
  params: string[];
  defaults: (Expr|undefined)[];
  noArg?: boolean[];
}

// Module/function declarations of the program being compiled
export interface LocalDecl {
  count: number;
  params: Parameter[];
}

// Generated `surface()` data file with decoded pixels or raw matrix data
export interface SurfaceAsset {
  stem: string;
  exportName: string;
  kind: 'image'|'text';
}

export let globalFileResolver: FileResolver|undefined;

export const signatures = new Map<string, Signature>();
export const localDecls = new Map<string, LocalDecl>();
export const noArgDemotions = new Map<string, boolean[]>();
export const moduleDeclRegistry = new Map<string, ModuleDeclStmtType>();

// Track unique fonts encountered during compilation for base64 generation.
export const encounteredFonts = new Set<string>();
// Generated surface data modules to import, keyed by source filename
export const encounteredSurfaceData = new Map<string, SurfaceAsset>();

// Loaded from library manifests so their calls aren't treated as unknown
// modules
export const externalModuleNames: Set<string> = new Set();
export const externalFunctionNames: Set<string> = new Set();
export const externalVariableNames: Set<string> = new Set();

// Track special variables that need module-level declarations for dynamic
// scoping
export const dynamicScopeVars: Set<string> = new Set();
// Compile-time divergence detection for non-tail recursion
export const userFunctionDefs = new Map<string, FunctionDeclStmt>();
// Names introduced by the emitter
export const unitTakenNames = new Set<string>();
export const tempNames = new Map<string, string>();

// Local name for each runtime export
export let RT: Record<string, string> =
    Object.fromEntries(RUNTIME_SYMBOLS.map(s => [s, s]));

export let currentRuntimePath: string = './runtime/runtime.js';
// Name of the file currently being compiled
export let currentMainFilename: string = '';
// Source file the statement being emitted came from
export let currentSourceFilename: string = '';
// Result of the lexical binding pass for the unit being compiled

export let bindResult: BindResult|undefined;
// Scope that top-level names resolve in for the unit currently being emitted
export let currentScope: Scope|undefined;

// Keyword for top-level variables. Use `let`, but `var` in libraries to avoid
// TDZ issues with circular imports and match OpenSCAD's `undef` behavior
export let globalVarDeclKeyword = 'let';

export let parentModulesReadInFunction = false;

let tailTempCounter = 0;

export const currentBindOptions: BindOptions = {
  builtinFunctions: BUILTIN_FUNCTIONS,
  builtinModules: BUILTIN_MODULES,
  builtinConstants: BUILTIN_VAR_CONSTANTS,
  externalFunctions: externalFunctionNames,
  externalModules: externalModuleNames,
  externalVariables: externalVariableNames,
};


export function setGlobalFileResolver(fileResolver: FileResolver): void {
  globalFileResolver = fileResolver;
}

export function setModuleDecls(decls: Map<string, ModuleDeclStmtType>): void {
  moduleDeclRegistry.clear();
  for (const [key, decl] of decls) moduleDeclRegistry.set(key, decl);
}

export function setRT(locals: Record<string, string>): void {
  RT = locals;
}

export function setCurrentRuntimePath(p: string): void {
  currentRuntimePath = p;
}

export function setMainFilename(name: string): void {
  currentMainFilename = name;
}

export function setCurrentSourceFilename(name: string): void {
  currentSourceFilename = name;
}

export function setBindResult(b: BindResult|undefined): void {
  bindResult = b;
}

export function setCurrentScope(s: Scope|undefined): void {
  currentScope = s;
}

export function setGlobalVarDeclKeyword(kw: string): void {
  globalVarDeclKeyword = kw;
}

export function setParentModulesReadInFunction(v: boolean): void {
  parentModulesReadInFunction = v;
}

export function nextTailTemp(): number {
  return tailTempCounter++;
}

export function resetTailTemps(): void {
  tailTempCounter = 0;
}
