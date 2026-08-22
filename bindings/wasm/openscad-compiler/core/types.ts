import type {Argument, Expr, ForVariable, FunctionCallExpr, LetAssignment, Parameter, Program, Statement,} from './ast.js';
import type {TokenType} from './lexer.js';


export interface FileResolver {
  readText(filePath: string): Promise<string|null>;
  readBinary(filePath: string): Promise<Uint8Array|null>;
  writeText(filePath: string, content: string): Promise<void>;
  removeDir(path: string): Promise<void>;
  makeDir(path: string): Promise<void>;
  readDir(path: string): Promise<string[]>;
  exists(filePath: string): Promise<boolean>;
  libraryPaths(): Promise<string[]>;
  fontPath(): Promise<string|undefined>;
}

// Lexing

// Position in the source code
export interface SourceLocation {
  line: number;
  column: number;
  offset: number;
}

// A range in the source code (start and end locations)
export interface SourceRange {
  start: SourceLocation;
  end: SourceLocation;
}

export interface Token {
  type: TokenType;
  value?: string;
  range: SourceRange;
}


// Binding

export type Namespace = 'var'|'fn'|'mod';

export type BindingKind =
    // Provided by the runtime
    'builtin'|
    // Exported by a separately compiled library and imported
    'external'|
    // Top-level declaration of the program being compiled
    'global'|
    // Top-level variable of a `use`d file, private to that file's scope
    'filePrivate'|
    // Module/function/lambda parameter
    'param'|
    // Variable declared inside a module body or a block
    'local'|
    // `let` binding
    'let'|
    // for loop variable
    'loop'|
    // Dynamically scoped `$` variable
    'special';

export interface Binding {
  id: number;
  name: string;
  ns: Namespace;
  kind: BindingKind;
  scope: Scope;
  // All declarations of this binding; declarative scopes are last-wins, so one
  // name stays one binding
  decls: (Statement|Parameter|LetAssignment|ForVariable)[];
  // For library closures, the file that declares this binding, relative to the
  // library root
  file?: string;
  jsName: string;  // Assigned by `assignPrettyNames` in naming.ts
  reads: number;
}

export type ScopeKind = 'root'|'external'|'global'|'file'|'module'|'function'|
    'lambda'|'let'|'for'|'block'|'comprehension';

export interface Scope {
  id: number;
  kind: ScopeKind;
  parent: Scope|null;
  children: Scope[];
  bindings: Record<Namespace, Map<string, Binding>>;
}

// a call site resolves to a CallRef
export interface CallRef {
  fn: Binding|null;
  value: Binding|null;
  mod: Binding|null;
}

export interface BindOptions {
  builtinFunctions: Iterable<string>;
  builtinModules: Iterable<string>;
  builtinConstants: Iterable<string>;
  externalFunctions?: Iterable<string>;
  externalModules?: Iterable<string>;
  externalVariables?: Iterable<string>;
}

// One file of a library closure, with the directives that pull other files in
export interface LibraryFileInput {
  rel: string;
  program: Program;
  edges: {rel: string; mode: 'include' | 'use'}[];
}

export interface BindResult {
  root: Scope;
  global: Scope;
  bindings: Binding[];
  scopes: Scope[];
}

export interface LibraryBindResult extends BindResult {
  // Resolution scope for each closure file, keyed by its path from the library
  // root
  fileScopes: Map<string, Scope>;
}


// Naming

export interface PrettyNameOptions {
  // Runtime symbols and other fixed names no binding may take
  reserved: Iterable<string>;
  // `<ns>:<name>` -> the symbol a separately compiled library exported it as
  externalSymbols?: Map<string, string>;
  builtinSymbols?: Map<string, string>;
}


// Include/use resolution

// Resolves and recursively parses OpenSCAD include/use directives
export interface ResolvedProgram {
  // All statements from included files + the main file, in order
  statements: Statement[];
  // Paths of all files that were resolved (for debugging/caching)
  resolvedFiles: string[];
}

export interface ExternalLibraryRef {
  name: string;
  root: string;
  entries: {file: string; mode: 'include' | 'use'}[];
}

export interface ResolvedProgramWithLibraries extends ResolvedProgram {
  externalLibraries: Map<string, ExternalLibraryRef>;
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


// Compilation

// The two declaration statements the emitter singles out often enough to name
export type ModuleDeclStmtType = Extract<Statement, {kind: 'moduleDecl'}>;
export type FunctionDeclStmtType = Extract<Statement, {kind: 'functionDecl'}>;

export interface LibraryManifest {
  manifestVersion?: number;
  library: string;
  compiledAt: string;
  runtimeVersion: string;
  files: Record < string, {
    out: string;
    modules: string[];
    functions: string[];
    variables: string[]
  }
  > ;
  exports: {
    modules:
        Record<string, string>;  // module name -> owning source file (relPath)
    functions: Record<string, string>;  // function name -> owning source file
    variables: Record<string, string>;  // variable name -> owning source file
  };
  ambiguous: Record<string, string[]>;
  // Emitted JS name for each export, since consumers can't infer library-chosen
  // names
  symbols?: {
    modules: Record<string, string>; functions: Record<string, string>;
    variables: Record<string, string>;
  };
  signatures: Record<string, string[]>;
  signatureNoArg?: Record<string, boolean[]>;
}

export interface ResolvedExternalLib {
  name: string;
  manifest: LibraryManifest;
  importSpecifierFor: (sourceRelPath: string) => string;
  sideEffectSpecifiers?: string[];
}

export interface CompileOptions {
  runtimePath?: string;
  externalLibraries?: ResolvedExternalLib[];
}

export interface CompiledLibraryFile {
  sourceRel: string;
  outRel: string;
  code: string;
}

export interface CompiledLibrary {
  manifest: LibraryManifest;
  files: CompiledLibraryFile[];
}


// The single whole-program scan
export interface ProgramReferences {
  modules: Set<string>;    // module-call statements
  functions: Set<string>;  // function-call expressions
  variables: Set<string>;  // bare identifier reads (non-$)
}

export interface ProgramScan {
  refs: ProgramReferences;
  // Unresolved reads become `undefined` instead of throwing; resolved ones use
  // their bindings
  unresolved: Set<string>;  // unresolved reads, escaped
  moduleArgNames:
      Set<string>;  // non-$ named args of user module calls, escaped
  parentModulesInFunction: boolean;
  topLevelChildren: boolean;
  functionDefs: Map<string, FunctionDeclStmtType>;
  divergenceCandidates: FunctionCallExpr[];
  font: FontScan;
}

// Record text-reachable modules and candidate literals; resolve the transitive
// closure afterward
export interface FontScan {
  // moduleName -> modules called directly in its body, excluding nested
  // declarations
  edges: Map<string, Set<string>>;
  // Literals whose relevance is already settled: font/style/family variables
  literals: Set<string>;
  // Sites kept only if the named module turns out to reach text()
  paramDefaults: {module: string; exprs: Expr[]}[];
  calls: {name: string; args: Argument[]}[];
  scopedVars: {module: string; value: Expr}[];
}

// Top-level modules used by the font fixpoint; nested declarations are ignored
// Splices `scope` like the old pass and only processes top-level statements
export interface FontTargets {
  names: Set<string>;
  decls: Set<ModuleDeclStmtType>;
}

export interface ScanOptions {
  // Open `NO_ARG` slots for demotion as calls are seen; library calls skip this
  // analysis
  noArgSlots?: Map<string, boolean[]>;
  // Collect candidate call sites for the divergence check. Consumer only
  divergence?: boolean;
  // Top-level modules from `fontCandidateNames`; their presence enables font
  // recording
  fontCandidates?: FontTargets;
}
