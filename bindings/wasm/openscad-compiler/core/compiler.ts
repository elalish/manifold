import {createCanvas, Image} from 'canvas';
import fs from 'fs';
import path from 'path';

import {forEachChild} from './ast.js';
import type {Argument, ASTNode, BlockStmt, Comment, Expr, ForStmt, ForVariable, IfStmt, KindedNode, ListCompGenerator, ModuleCallStmt, Program, ScopeStmt, Statement,} from './ast.js';
import type {IRBooleanNode, IRChildrenNode, IRForNode, IRIfNode, IRModuleCallNode, IRNode, IRPrimitiveNode, IRSequenceNode, IRTransformNode,} from './ir.js';
import {assignPrettyNames, bindLibrary, bindProgram, escapeName, isLexicalVar, lookup, shadowsOuterVar,} from './binder.js';
import type {BindOptions, BindResult, Binding, CallRef, Namespace, Scope,} from './binder.js';
import {getFontPath} from './resolver.js';
import type {LibraryClosure} from './resolver.js';

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
  // Emitted JS symbol for each export. Needed because a library picks its own names, and a consumer importing them cannot re-derive what it chose
  symbols?: {
    modules: Record<string, string>;
    functions: Record<string, string>;
    variables: Record<string, string>;
  };
  signatures: Record<string, string[]>;
  signatureNoArg?: Record<string, boolean[]>;
}

export const MANIFEST_VERSION = 2;

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

const BUILTIN_CONSTANTS_CODE =
    `let PI: number = __rt.PI;\n` +
    `let INF: number = __rt.INF;\n` +
    `let NAN: number = __rt.NAN;\n` +
    `let undef: undefined = __rt.undef;\n` +
    `let _EPSILON: number = __rt._EPSILON;\n` +
    `let __NO_ARG: symbol = Symbol.for("__OPENSCAD_NO_ARG__");\n`;


const BUILTIN_MODULES = new Set([
  'cube',
  'sphere',
  'cylinder',
  'circle',
  'square',
  'polygon',
  'polyhedron',
  'text',
  'surface',
  'import',
  'translate',
  'rotate',
  'scale',
  'mirror',
  'multmatrix',
  'resize',
  'offset',
  'color',
  'render',
  'projection',
  'group',
  'union',
  'difference',
  'intersection',
  'hull',
  'minkowski',
  'linear_extrude',
  'rotate_extrude',
  'echo',
  'assert',
  'let',
  'children',
  'intersection_for',
  'parent_module',
]);

interface ProgramReferences {
  modules: Set<string>;    // module-call statements
  functions: Set<string>;  // function-call expressions
  variables: Set<string>;  // bare identifier reads (non-$)
}

function collectProgramReferences(stmts: Statement[]): ProgramReferences {
  const modules = new Set<string>();
  const functions = new Set<string>();
  const variables = new Set<string>();

  const visit = (node: KindedNode): void => {
    switch (node.kind) {
      case 'identifier':
        if (node.name !== '$children' && !node.name.startsWith('$'))
          variables.add(node.name);
        break;
      case 'call':
        functions.add(node.name);
        break;
      case 'moduleCall':
        if (node.name !== 'children') modules.add(node.name);
        break;
    }
    forEachChild(node, visit);
  };

  stmts.forEach(visit);
  return {modules, functions, variables};
}

function locTag(node: ASTNode): string {
  if (!node.loc) return '';
  const s = node.loc.start;
  return ` @${s.line}:${s.column}`;
}

function commentLines(comment: Comment, indent = ''): string[] {
  return comment.value.split(/\r?\n/).map(line => `${indent}${line}`);
}

function leadingCommentLines(node: ASTNode|undefined, indent = ''): string[] {
  return (node?.leadingComments ?? [])
      .flatMap(comment => commentLines(comment, indent));
}

function trailingCommentText(node: ASTNode|undefined): string {
  const comments = node?.trailingComments ?? [];
  if (comments.length === 0) return '';
  return ` ${
      comments.map(comment => comment.value.replace(/\r?\n/g, ' ')).join(' ')}`;
}

function returnExpr(expr: string, indent = ''): string {
  const trimmed = expr.trim();
  if (trimmed.startsWith('//') || trimmed.startsWith('/*')) {
    return `(\n${expr}\n${indent})`;
  }
  return expr;
}

function pushCommentedLine(
    lines: string[], node: ASTNode, line: string, indent = ''): void {
  lines.push(...leadingCommentLines(node, indent));
  lines.push(`${line}${trailingCommentText(node)}`);
}


// Signatures
interface Signature {
  params: string[];
  defaults: (Expr|undefined)[];
  // Per-param: declaration applies its default via the __NO_ARG sentinel
  // prologue, so call sites fill unprovided slots with __NO_ARG (explicit undef
  // then reaches the body as undef instead of re-triggering the default)
  noArg?: boolean[];
}
const signatures = new Map<string, Signature>();

// Signature-table key. Namespaced by the OpenSCAD name
function sigKey(name: string, ns: Namespace): string {
  return `${ns}:${name}`;
}

// Manifest signature keys are written in this same namespaced form
function manifestSigKey(sym: string): string {
  if (sym.startsWith('fn:') || sym.startsWith('mod:')) return sym;
  if (sym.endsWith('$mod')) return sigKey(sym.slice(0, -4), 'mod');
  if (sym.endsWith('_fn')) return sigKey(sym.slice(0, -3), 'fn');
  return sigKey(sym, 'fn');
}

// Params whose default is applied via the __NO_ARG prologue: $-params keep
// dynamic-scope handling and self-referencing defaults (x = x) must evaluate in
// the parameter scope where the name still resolves to the outer binding
function paramUsesNoArg(p: Param): boolean {
  return !!p.defaultValue && !p.name.startsWith('$') &&
      !nodeReferencesIdentifier(p.defaultValue, p.name);
}

type ModuleDeclStmtType = Extract<Statement, {kind: 'moduleDecl'}>;

interface IRLowerContext {
  modules: Map<string, ModuleDeclStmtType>;
  children: IRNode[];
  callStack: string[];
}

let moduleDeclRegistry = new Map<string, ModuleDeclStmtType>();

let tailTempCounter = 0;

// Track unique fonts encountered during compilation for base64 generation.
let encounteredFonts = new Set<string>();

// OpenSCAD's default typeface, used when a text() call names none
const DEFAULT_FONT_SPEC = 'Liberation Sans:style=Regular';

// Track unique surface data encountered during compilation for base64
// generation.
let encounteredSurfaceData = new Map < string, {
  stem: string;
  exportName: string;
  kind: 'image'|'text'
}
>();

let currentRuntimePath: string = './runtime/runtime.js';

// Name of the file currently being compiled
let currentMainFilename: string = '';

// Source file the statement being emitted came from
let currentSourceFilename: string = '';

// Aborts a C-style for loop once its counter exceeds this many iterations,
// guarding against infinite loops
const MAX_FOR_ITERATIONS = 1000000;

function fontSpecToFilename(fontSpec: string): string {
  const cleaned = fontSpec.replace(/"/g, '').trim();
  const parts = cleaned.split(':');
  const family = (parts[0] || 'Liberation Sans').trim().replace(/\s+/g, '');

  let style = 'Regular';
  for (let i = 1; i < parts.length; i++) {
    const part = parts[i]!.trim();
    const match = part.match(/^style\s*=\s*(.+)$/i);
    if (match) {
      style = match[1]!.trim().replace(/\s+/g, '');
      break;
    }
  }

  return `${family}-${style}`;
}

function generateFontBase64(fontSpec: string, compilerDir: string): string|
    undefined {
  const fontDir = getFontPath();
  if (!fontDir) {
    console.warn(`Warning: FONTPATH environment variable not set — cannot load font "${
        fontSpec}". Text will render as empty cross-section.`);
    return undefined;
  }

  let filename = fontSpec;
  let ttfPath = path.join(fontDir, `${filename}.ttf`);
  let otfPath = path.join(fontDir, `${filename}.otf`);

  if (!fs.existsSync(ttfPath) && !fs.existsSync(otfPath)) {
    filename = fontSpecToFilename(fontSpec);
    ttfPath = path.join(fontDir, `${filename}.ttf`);
    otfPath = path.join(fontDir, `${filename}.otf`);
  }

  let fontFilePath: string|undefined;
  let mimeType: string;

  if (fs.existsSync(ttfPath)) {
    fontFilePath = ttfPath;
    mimeType = 'font/ttf';
  } else if (fs.existsSync(otfPath)) {
    fontFilePath = otfPath;
    mimeType = 'font/otf';
  } else {
    console.warn(`Warning: Font file not found at "${ttfPath}" or "${
        otfPath}" — text using "${
        fontSpec}" will render as empty cross-section.`);
    return undefined;
  }

  const fontBytes = fs.readFileSync(fontFilePath);
  const base64 = fontBytes.toString('base64');

  const fontsDir = path.join(compilerDir, 'runtime', 'fonts');
  fs.mkdirSync(fontsDir, {recursive: true});

  const outFile = path.join(fontsDir, `${filename}_base64.ts`);
  const content =
      `// Auto-generated by openscad-to-manifold compiler — do not edit.\nexport const fontBase64 = "data:${
          mimeType};base64,${base64}";\n`;
  fs.writeFileSync(outFile, content, 'utf8');
  console.log(`Generated font base64: ${outFile} (${
      (fontBytes.length / 1024).toFixed(1)} KB)`);

  return filename;
}

const MAX_IR_INLINE_DEPTH = 2;
const MAX_IR_INLINE_COMPLEXITY = 120;

const BUILTIN_SIGNATURES: Record<string, string[]> = {
  'cube$mod': ['size', 'center'],
  'cylinder$mod':
      ['h', 'r', 'r1', 'r2', 'd', 'd1', 'd2', 'center', '$fn', '$fa', '$fs'],
  'sphere$mod': ['r', 'd', '$fn', '$fa', '$fs'],
  'square$mod': ['size', 'center'],
  'circle$mod': ['r', 'd', '$fn', '$fa', '$fs'],
  'polygon$mod': ['points', 'paths', 'convexity'],
  'polyhedron$mod': ['points', 'faces', 'convexity'],
  'linear_extrude$mod': [
    'height', 'v', 'scale', 'center', 'twist', 'slices', 'segments',
    'convexity', 'h', '$fn', '$fa', '$fs', '$fe'
  ],
  'rotate_extrude$mod': ['angle', 'convexity', '$fn'],
  'text$mod': [
    'text', 'size', 'font', 'halign', 'valign', 'spacing', 'direction',
    'language', 'script', '$fn'
  ],
  'surface$mod': ['file', 'center', 'invert', 'convexity'],
  'import$mod': ['file', 'convexity', 'layer'],
  'projection$mod': ['cut'],
  'translate$mod': ['v'],
  'rotate$mod': ['a', 'v'],
  'scale$mod': ['v'],
  'resize$mod': ['newsize', 'auto'],
  'mirror$mod': ['v'],
  'multmatrix$mod': ['m'],
  'color$mod': ['c', 'alpha'],
  'offset$mod': ['r', 'delta', 'chamfer'],
};

function collectSignatures(stmts: Statement[]) {
  for (const stmt of stmts) {
    if (stmt.kind === 'functionDecl' || stmt.kind === 'moduleDecl') {
      const name = sigKey(
          stmt.name, stmt.kind === 'functionDecl' ? 'fn' : 'mod');
      signatures.set(name, {
        params: stmt.params.map(p => p.name),
        defaults: stmt.params.map(p => p.defaultValue),
        noArg: stmt.params.map(paramUsesNoArg),
      });
      if (stmt.kind === 'moduleDecl' && stmt.body.kind === 'block') {
        collectSignatures(stmt.body.statements);
      }
    } else if (stmt.kind === 'block' || stmt.kind === 'scope') {
      collectSignatures(stmt.statements);
    } else if (stmt.kind === 'if') {
      if (stmt.thenBody.kind === 'block')
        collectSignatures(stmt.thenBody.statements);
      if (stmt.elseBody && stmt.elseBody.kind === 'block')
        collectSignatures(stmt.elseBody.statements);
    }
  }
}

function collectModuleDeclarations(
    stmts: Statement[],
    into: Map<string, ModuleDeclStmtType> =
        new Map<string, ModuleDeclStmtType>(),
    ): Map<string, ModuleDeclStmtType> {
  for (const stmt of stmts) {
    if (stmt.kind === 'moduleDecl') {
      into.set(stmt.name, stmt);
      if (stmt.body.kind === 'block') {
        collectModuleDeclarations(stmt.body.statements, into);
      }
      continue;
    }
    if (stmt.kind === 'block' || stmt.kind === 'scope') {
      collectModuleDeclarations(stmt.statements, into);
      continue;
    }
    if (stmt.kind === 'if') {
      if (stmt.thenBody.kind === 'block')
        collectModuleDeclarations(stmt.thenBody.statements, into);
      if (stmt.elseBody && stmt.elseBody.kind === 'block')
        collectModuleDeclarations(stmt.elseBody.statements, into);
    }
  }
  return into;
}

function baseIRContext(modules = moduleDeclRegistry): IRLowerContext {
  return {modules, children: [], callStack: []};
}

// Node count of a subtree, used only to keep IR inlining to small bodies
function estimateNodeComplexity(node: KindedNode): number {
  let total = 1;
  forEachChild(node, child => {
    total += estimateNodeComplexity(child);
  });
  return total;
}


function nodeUsesModuleScope(node: KindedNode): boolean {
  if (node.kind === 'identifier') return node.name === '$children';
  if (node.kind === 'moduleCall' && node.name === 'children') return true;
  let found = false;
  forEachChild(node, child => {
    found = found || nodeUsesModuleScope(child);
  });
  return found;
}

function shouldInlineModuleToIR(
    decl: ModuleDeclStmtType, ctx: IRLowerContext): boolean {
  if (ctx.callStack.length >= MAX_IR_INLINE_DEPTH) return false;
  if (nodeUsesModuleScope(decl.body)) return false;
  return estimateNodeComplexity(decl.body) <= MAX_IR_INLINE_COMPLEXITY;
}

function compileArgList(key: string, args: Argument[]): string {
  const sig = signatures.get(key);
  if (!sig) {
    return args
        .map(
            a => a.name ? `/* ${a.name} = */ ${compileExpr(a.value)}` :
                          compileExpr(a.value))
        .join(', ');
  }

  // Missing arguments use a sentinel to apply defaults, while an explicit
  // `undef` is kept as is and doesn't reapply the default
  const fillFor = (i: number) => (sig.noArg?.[i] ? '__NO_ARG' : 'undefined');
  const compiledArgs: string[] =
      new Array(sig.params.length).fill('').map((_, i) => fillFor(i));
  const namedClaimed: boolean[] = new Array(sig.params.length).fill(false);
  const extraArgs: string[] = [];

  let posCursor = 0;
  for (const a of args) {
    if (a.name) {
      if (a.name.startsWith('$')) continue;
      const idx = sig.params.indexOf(a.name);
      if (idx >= 0) {
        compiledArgs[idx] = `/* ${a.name} = */ ${compileExpr(a.value)}`;
        namedClaimed[idx] = true;
      }
    } else {
      while (posCursor < sig.params.length && namedClaimed[posCursor]) {
        posCursor++;
      }
      if (posCursor < sig.params.length) {
        compiledArgs[posCursor] = compileExpr(a.value);
        posCursor++;
      } else {
        extraArgs.push(compileExpr(a.value));
      }
    }
  }

  // Trailing fills can be dropped since the prologue's arguments.length check
  // treats absent the same as __NO_ARG
  while (compiledArgs.length > 0 &&
         compiledArgs[compiledArgs.length - 1] ===
             fillFor(compiledArgs.length - 1) &&
         extraArgs.length === 0) {
    compiledArgs.pop();
  }

  return compiledArgs.concat(extraArgs).join(', ');
}

const BUILTIN_FUNCTIONS = new Set([
  'is_undef',    'is_bool',     'is_num',        'is_string', 'is_list',
  'is_function', 'sin',         'cos',           'tan',       'asin',
  'acos',        'atan',        'atan2',         'abs',       'sign',
  'floor',       'ceil',        'round',         'sqrt',      'exp',
  'ln',          'log',         'pow',           'min',       'max',
  'norm',        'cross',       'len',           'str',       'chr',
  'ord',         'concat',      'search',        'lookup',    'rands',
  'version',     'version_num', 'parent_module',
]);

// Experimental or Enable only builtins
const EXPERIMENTAL_BUILTIN_FUNCTIONS = new Set([
  'is_object',
]);


const RUNTIME_SYMBOLS: string[] = [
  '__cube',
  '__square',
  '__sphere',
  '__cylinder',
  '__circle',
  '__radius',
  '__polygon',
  '__polyhedron',
  'Manifold',
  'CrossSection',
  'wasm',
  'is_undef_fn',
  'is_bool_fn',
  'is_num_fn',
  'is_string_fn',
  'is_list_fn',
  'is_function_fn',
  '__unknown_fn',
  'sin_fn',
  'cos_fn',
  'tan_fn',
  'asin_fn',
  'acos_fn',
  'atan_fn',
  'atan2_fn',
  'abs_fn',
  'sign_fn',
  'floor_fn',
  'ceil_fn',
  'round_fn',
  'sqrt_fn',
  'exp_fn',
  'ln_fn',
  'log_fn',
  'pow_fn',
  'min_fn',
  'max_fn',
  'norm_fn',
  'cross_fn',
  'len_fn',
  'str_fn',
  'chr_fn',
  'ord_fn',
  'concat_fn',
  'search_fn',
  'lookup_fn',
  'rands_fn',
  'parent_module_fn',
  'openscad_assert_fn',
  '__truthy',
  '__eq',
  '__lt',
  '__gt',
  '__le',
  '__ge',
  '__add',
  '__sub',
  '__mul',
  '__div',
  '__mod',
  '__band',
  '__bor',
  '__shl',
  '__shr',
  '__bnot',
  '__neg',
  '__pos',
  '__index',
  'version_fn',
  'version_num_fn',
  '__ctx',
  '__withSpecials',
  '__children_stack',
  '__with_children',
  '__pick_children',
  '__is_finite_matrix4',
  '__to_manifold_mat4',
  '__safe_transform',
  '__identity4',
  '__safe_offset2d',
  '__safe_project3d',
  '__apply_color',
  '__each',
  '__flat_map_iter',
  '__range',
  '__rangeCount',
  '__union2d3d',
  '__difference2d3d',
  '__intersection2d3d',
  '__hull2d3d',
  '__minkowski2d3d',
  '__rootMod',
  '__applyRoot',
  '__extrude',
  '__revolve',
  '__rotate',
  '__translate',
  '__scale',
  '__mirror',
  '__resize',
  '__text',
  '__parse_color_for_scope',
  '__surface',
  '__echo',
  '__oecho',
  '__fnlit',
  '__font_registry',
  '__tc',
  '__call',
];

/* Emit target for a (possibly special) variable name. Special $-prefixed vars
 are stored in the shared runtime context object (`__ctx`); every other name is
 a plain lexical binding. Used wherever the emitter reads/writes a variable
 that might be a special variable (dynamic-scope save/assign/restore sites).
*/
function svTarget(name: string): string {
  return name.startsWith('$') ? `__ctx.${name}` : name;
}

function buildRuntimeImport(runtimePath: string): string {
  return (
      `import * as __rt from "${runtimePath}";\n` +
      `const { ${RUNTIME_SYMBOLS.join(', ')} } = __rt;\n`);
}

/* Names of modules/functions/variables that are defined in a separately
  compiled external library and imported (not inlined). Populated from the
  libraries' manifests at the start of compile(); consulted so calls to them
  are recognized rather than reported as unknown modules.
*/
let externalModuleNames: Set<string> = new Set();
let externalFunctionNames: Set<string> = new Set();
let externalVariableNames: Set<string> = new Set();

// Constants the runtime pre-declares as ordinary bindings
const BUILTIN_VAR_CONSTANTS = ['PI', 'INF', 'NAN', 'undef', '_EPSILON'];

// Result of the lexical binding pass for the unit being compiled
let bindResult: BindResult|undefined;

// Names the emitter or the runtime already owns, which no binding may take
function reservedNames(): string[] {
  return [
    ...RUNTIME_SYMBOLS,
    ...BUILTIN_VAR_CONSTANTS,
    '__NO_ARG',
    // Emitted into every module body, and into the module's own output
    'children',
    '$children',
    'result',
    'background',
  ];
}

function currentBindOptions(): BindOptions {
  return {
    builtinFunctions: BUILTIN_FUNCTIONS,
    builtinModules: BUILTIN_MODULES,
    builtinConstants: BUILTIN_VAR_CONSTANTS,
    externalFunctions: externalFunctionNames,
    externalModules: externalModuleNames,
    externalVariables: externalVariableNames,
  };
}

/* Keyword for top-level (file-scope) variable declarations. `let` for normal
  single-file output; `var` for library files, where cross-file circular ES
  imports would otherwise hit the temporal dead zone on `let` exports. `var`
  hoists to `undefined`, matching OpenSCAD's declarative `undef`-by-default.
*/
let globalVarDeclKeyword = 'let';

// Track $ variables that need module-level declarations for dynamic scoping
let dynamicScopeVars: Set<string> = new Set();


function collectStringLiterals(node: KindedNode, literals: Set<string>): void {
  if (node.kind === 'string') literals.add(node.value);
  forEachChild(node, child => collectStringLiterals(child, literals));
}

// Collect every identifier referenced as a value, and every name bound anywhere
// in the program (variable/param/let/for/comprehension binding). A name that is
// referenced but never bound is an undefined variable: OpenSCAD evaluates such
// a reference to `undef` rather than erroring, so we declare these as
// `undefined` to avoid a ReferenceError in the compiled code.
interface IdentifierUsage {
  referenced: Set<string>;
  bound: Set<string>;
}

function collectIdentifierUsage(program: Program): IdentifierUsage {
  const referenced = new Set<string>();
  const bound = new Set<string>();

  // Only the names a node binds are per-kind here; reaching the children is
  // the visitor's job
  const visit = (node: KindedNode): void => {
    switch (node.kind) {
      case 'identifier':
        if (node.name !== '$children') referenced.add(bindJsName(node));
        break;
      case 'variableDecl':
        bound.add(bindJsName(node));
        break;
      case 'lambda':
      case 'functionDecl':
      case 'moduleDecl':
        node.params.forEach(p => bound.add(bindJsName(p)));
        break;
      case 'let':
      case 'lcLet':
        node.assignments.forEach(a => bound.add(bindJsName(a)));
        break;
      case 'for':
      case 'lcFor':
        node.variables.forEach(v => bound.add(bindJsName(v)));
        break;
      case 'lcCFor':
        node.inits.forEach(a => bound.add(bindJsName(a)));
        node.updates.forEach(a => bound.add(bindJsName(a)));
        break;
    }
    forEachChild(node, visit);
  };

  program.statements.forEach(visit);
  return {referenced, bound};
}

function resolveCallArgs(
    moduleCallName: string, callArgs: Argument[]): Map<string, Expr> {
  const resolved = new Map<string, Expr>();
  const sig = signatures.get(sigKey(moduleCallName, 'mod'));
  if (!sig) return resolved;

  // Initialize with default values
  for (let i = 0; i < sig.params.length; i++) {
    const paramName = sig.params[i]!;
    const defaultVal = sig.defaults[i];
    if (defaultVal) {
      resolved.set(paramName, defaultVal);
    }
  }

  // Map positional arguments
  let pos = 0;
  while (pos < callArgs.length && !callArgs[pos]!.name) {
    if (pos < sig.params.length) {
      resolved.set(sig.params[pos]!, callArgs[pos]!.value);
    }
    pos++;
  }

  // Map named arguments
  for (let i = pos; i < callArgs.length; i++) {
    const a = callArgs[i]!;
    if (a.name) {
      resolved.set(a.name, a.value);
    }
  }

  return resolved;
}

function collectFontRelatedLiterals(program: Program): Set<string> {
  const modulesCallingText = new Set<string>(['text']);
  const flatten = (stmts: Statement[]): Statement[] => stmts.flatMap(
      s => s.kind === 'scope' ? flatten(s.statements) : [s]);
  const topLevel = flatten(program.statements);

  let changed = true;
  while (changed) {
    changed = false;
    for (const stmt of topLevel) {
      if (stmt.kind === 'moduleDecl' && !modulesCallingText.has(stmt.name)) {
        let callsText = false;

        const checkCalls = (s: Statement) => {
          if (s.kind === 'moduleCall') {
            if (modulesCallingText.has(s.name)) {
              callsText = true;
            }
            if (s.child) checkCalls(s.child);
          } else if (s.kind === 'block' || s.kind === 'scope') {
            for (const sub of s.statements) checkCalls(sub);
          } else if (s.kind === 'for') {
            checkCalls(s.body);
          } else if (s.kind === 'if') {
            checkCalls(s.thenBody);
            if (s.elseBody) checkCalls(s.elseBody);
          }
        };

        checkCalls(stmt.body);
        if (callsText) {
          modulesCallingText.add(stmt.name);
          changed = true;
        }
      }
    }
  }

  const literals = new Set<string>();

  // Helper to collect all string literals from an expression
  const collectExpr = (expr: Expr) => {
    if (!expr) return;
    collectStringLiterals(expr, literals);
  };

  // Traverse AST to collect literals from related definitions and invocations
  const traverse = (s: Statement, insideTextModule: boolean) => {
    if (!s) return;

    if (s.kind === 'moduleDecl') {
      const isTextMod = modulesCallingText.has(s.name);
      if (isTextMod) {
        for (const p of s.params) {
          if (p.defaultValue) collectExpr(p.defaultValue);
        }
      }
      traverse(s.body, isTextMod);
    } else if (s.kind === 'moduleCall') {
      if (modulesCallingText.has(s.name)) {
        // Resolve arguments including default parameter values if omitted
        const argMap = resolveCallArgs(s.name, s.args);
        for (const [paramName, expr] of argMap.entries()) {
          collectExpr(expr);
        }
      }
      if (s.child) traverse(s.child, insideTextModule);
    } else if (s.kind === 'variableDecl') {
      const nameLower = s.name.toLowerCase();
      const isRelated = insideTextModule || nameLower.includes('font') ||
          nameLower.includes('style') || nameLower.includes('family');
      if (isRelated) {
        collectExpr(s.value);
      }
    } else if (s.kind === 'block' || s.kind === 'scope') {
      for (const sub of s.statements) traverse(sub, insideTextModule);
    } else if (s.kind === 'for') {
      traverse(s.body, insideTextModule);
    } else if (s.kind === 'if') {
      traverse(s.thenBody, insideTextModule);
      if (s.elseBody) traverse(s.elseBody, insideTextModule);
    }
  };

  for (const stmt of program.statements) {
    traverse(stmt, false);
  }

  return literals;
}

// Main entry
export function compile(program: Program, options?: CompileOptions): string {
  currentRuntimePath = options?.runtimePath ?? './runtime/runtime.js';
  currentMainFilename = program.filename ?? '';
  currentSourceFilename = currentMainFilename;
  dynamicScopeVars = new Set();
  encounteredFonts = new Set();
  encounteredSurfaceData.clear();
  externalModuleNames = new Set();
  externalFunctionNames = new Set();
  externalVariableNames = new Set();
  tailTempCounter = 0;
  signatures.clear();
  for (const [k, v] of Object.entries(BUILTIN_SIGNATURES)) {
    signatures.set(
        sigKey(k.replace(/\$mod$/, ''), 'mod'),
        {params: v, defaults: new Array(v.length).fill(undefined)});
  }
  // Register external-library export signatures and names BEFORE collecting
  // local ones, so a local declaration of the same name overrides the import
  const externalLibraries = options?.externalLibraries ?? [];
  for (const lib of externalLibraries) {
    for (const [sym, params] of Object.entries(lib.manifest.signatures)) {
      const noArg = lib.manifest.signatureNoArg?.[sym];
      signatures.set(manifestSigKey(sym), {
        params,
        defaults: new Array(params.length).fill(undefined),
        ...(noArg ? {noArg} : {}),
      });
    }
    for (const name of Object.keys(lib.manifest.exports.modules))
      externalModuleNames.add(name);
    for (const name of Object.keys(lib.manifest.exports.functions))
      externalFunctionNames.add(name);
    for (const name of Object.keys(lib.manifest.exports.variables))
      externalVariableNames.add(name);
  }
  collectSignatures(program.statements);
  moduleDeclRegistry = collectModuleDeclarations(program.statements);

  bindResult = bindProgram(program, currentBindOptions());
  const externalSymbols = new Map<string, string>();
  for (const lib of externalLibraries) {
    const syms = lib.manifest.symbols;
    if (!syms) continue;
    for (const [n, sym] of Object.entries(syms.modules))
      externalSymbols.set(`mod:${n}`, sym);
    for (const [n, sym] of Object.entries(syms.functions))
      externalSymbols.set(`fn:${n}`, sym);
    for (const [n, sym] of Object.entries(syms.variables))
      externalSymbols.set(`var:${n}`, sym);
  }
  assignPrettyNames(
      bindResult, {reserved: reservedNames(), externalSymbols});
  currentScope = bindResult.global;

  // Reject top-level constant-argument calls to non-tail recursive functions
  // that provably never terminate
  userFunctionDefs = new Map();
  collectFunctionDefs(program.statements, userFunctionDefs);
  detectDivergentCalls(program.statements);


  /* `text()` compiled straight from the builtin registers its face while
     emitting. A call routed through a library's own `text` module never
     reaches that path, so register the default face up front whenever the
     program mentions text at all. */
  const programRefs = collectProgramReferences(program.statements);
  if (programRefs.modules.has('text') || programRefs.functions.has('text'))
    encounteredFonts.add(DEFAULT_FONT_SPEC);

  // Gather all font-related string literals from the program
  const fontLiterals = collectFontRelatedLiterals(program);

  // Scan FONTPATH and match fonts
  const fontDir = getFontPath();
  if (fontDir && fs.existsSync(fontDir)) {
    try {
      const files = fs.readdirSync(fontDir);
      const cleanedLiterals =
          Array.from(fontLiterals)
              .map(lit => lit.toLowerCase().replace(/[^a-z0-9]/g, ''));

      for (const file of files) {
        const ext = path.extname(file).toLowerCase();
        if (ext === '.ttf' || ext === '.otf') {
          const basename = path.basename(file, ext);
          const dashIdx = basename.indexOf('-');
          const family = dashIdx >= 0 ? basename.slice(0, dashIdx) : basename;
          const style = dashIdx >= 0 ? basename.slice(dashIdx + 1) : 'Regular';

          const cleanedFamily = family.toLowerCase().replace(/[^a-z0-9]/g, '');
          const cleanedStyle = style.toLowerCase().replace(/[^a-z0-9]/g, '');

          // Check if family matches any of the cleaned literals
          const familyMatched = cleanedLiterals.some(
              lit => lit.includes(cleanedFamily) ||
                  (lit.length >= 4 && cleanedFamily.includes(lit)));
          if (familyMatched) {
            const styleMatched =
                cleanedStyle === 'regular' || cleanedLiterals.some(lit => {
                  if (cleanedStyle === 'bolditalic') {
                    return (lit.includes('bold') && lit.includes('italic')) ||
                        lit.includes('bolditalic');
                  }
                  return lit.includes(cleanedStyle);
                });
            if (styleMatched) {
              encounteredFonts.add(basename);
            }
          }
        }
      }
    } catch (e) {
      console.warn('Warning: failed to read font directory for matching:', e);
    }
  }

  // Build declarations, deduplicating by output name (last wins, matching
  // OpenSCAD semantics)
  const declMap = new Map < string, {
    stmt: Statement;
    code: string
  }
  >();
  const declOrder: string[] = [];
  const geometryLines: string[] = [];

  const scopeUnits: string[] = [];

  let lastGeoFilename = '';
  const processStmt = (stmt: Statement) => {
    if (stmt.kind === 'empty') return;
    if (stmt.filename) currentSourceFilename = stmt.filename;
    // `{}` doesn't create a new scope in OpenSCAD: merge its assignments into
    // the enclosing scope (last assignment wins) and inline its actions
    if (stmt.kind === 'block') {
      for (const s of stmt.statements) processStmt(s);
      return;
    }
    // A `use`d file is a scope: its variables stay inside
    // and only its modules and functions are published, as forwarders so they
    // dedupe against the consumer's own declarations of the same name
    if (stmt.kind === 'scope') {
      const unit = `__scope${scopeUnits.length}`;
      const {code, exports} = compileUsedFileScope(stmt, unit);
      scopeUnits.push(code);
      for (const ex of exports) {
        const key = `fn:${ex}`;
        if (!declMap.has(key)) declOrder.push(key);
        declMap.set(key, {
          stmt,
          code: `function ${ex}(...__args: any[]): any { return ${unit}.${
              ex}(...__args); }`,
        });
      }
      return;
    }
    if (stmt.kind === 'variableDecl' || stmt.kind === 'moduleDecl' ||
        stmt.kind === 'functionDecl') {
      // Compute the output name to detect duplicates
      let key: string;
      if (stmt.kind === 'variableDecl')
        key = `var:${declJsName(stmt, 'var')}`;
      else if (stmt.kind === 'moduleDecl')
        key = `fn:${declJsName(stmt, 'mod')}`;
      else
        key = `fn:${declJsName(stmt, 'fn')}`;

      if (!declMap.has(key)) declOrder.push(key);
      declMap.set(
          key, {stmt, code: compileDeclaration(stmt, {assignmentOnly: true})});
    } else if (stmt.kind === 'use' || stmt.kind === 'include') {
      const key = `comment:${stmt.path}`;
      if (!declMap.has(key)) declOrder.push(key);
      declMap.set(key, {stmt, code: `// ${stmt.kind} <${stmt.path}>`});
    } else {
      const geo = compileGeometry(stmt);
      if (geo) {
        const filename = stmt.filename;
        if (filename && filename !== lastGeoFilename) {
          const relativePath =
              path.relative(process.cwd(), filename).replace(/\\/g, '/');
          geometryLines.push(`\n// ${relativePath}`);
          lastGeoFilename = filename;
        }

        if (hasBackgroundModifier(stmt)) {
          pushCommentedLine(
              geometryLines, stmt, `__background_items.push(${geo});`);
        } else if (
            stmt.kind === 'moduleCall' && !stmt.modifier &&
            isModuleCallBackgroundOnly(stmt, moduleDeclRegistry)) {
          pushCommentedLine(
              geometryLines, stmt, `__background_items.push(${geo});`);
        } else {
          pushCommentedLine(
              geometryLines, stmt, `__result_items.push(${geo});`);
        }
      }
    }
  };
  for (const stmt of program.statements) processStmt(stmt);

  /*
    Variables get hoisted to undef first, then assigned in the order they first
    appear in the source. Not in dependency order, so a forward reference reads
    undef just like OpenSCAD, without tripping JS's temporal-dead-zone.
    Functions and modules skip this entirely since they're plain hoisted
    declarations.
  */
  const hoistNames: string[] = [];
  const seenHoist = new Set<string>();
  for (const k of declOrder) {
    const e = declMap.get(k)!;
    if (e.stmt.kind !== 'variableDecl') continue;
    const nm = e.stmt.name;
    if (nm.startsWith('$') || PRE_DECLARED_VARS.has(nm)) continue;
    const en = escapeName(nm);
    if (seenHoist.has(en)) continue;
    seenHoist.add(en);
    hoistNames.push(en);
  }

  const declarations: string[] = [];
  if (hoistNames.length) {
    declarations.push(
        hoistNames.map(n => `${globalVarDeclKeyword} ${n}: any = undef;`)
            .join('\n'));
  }
  declarations.push(...scopeUnits);
  let lastFilename = '';
  for (const k of declOrder) {
    const entry = declMap.get(k)!;
    const filename = entry.stmt.filename;
    if (filename && filename !== lastFilename) {
      const relativePath =
          path.relative(process.cwd(), filename).replace(/\\/g, '/');
      declarations.push(`\n// ${relativePath}`);
      lastFilename = filename;
    }
    declarations.push(entry.code);
  }

  const currentFileDir = typeof __dirname !== 'undefined' ?
      __dirname :
      path.dirname(
          new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/i, '$1'));
  const compilerDir = path.resolve(currentFileDir, '..');
  const fontImports: string[] = [];
  const resolvedFonts =
      new Map<string, string>();  // fontFamily → sanitized name (if resolved)

  for (const fontFamily of encounteredFonts) {
    const sanitized = generateFontBase64(fontFamily, compilerDir);
    if (sanitized) {
      resolvedFonts.set(fontFamily, sanitized);
    }
  }

  const RUNTIME_IMPORT =
      buildRuntimeImport(options?.runtimePath ?? './runtime/runtime.js');

  let output = RUNTIME_IMPORT;

  // Inject imports for names referenced from separately compiled external
  // libraries (resolved per kind against each library's manifest exports).
  if (externalLibraries.length > 0) {
    const refs = collectProgramReferences(program.statements);
    const importsBySpec = new Map<string, Set<string>>();
    const addImp = (spec: string, sym: string) => {
      let set = importsBySpec.get(spec);
      if (!set) {
        set = new Set();
        importsBySpec.set(spec, set);
      }
      set.add(sym);
    };
    for (const lib of externalLibraries) {
      for (const m of refs.modules) {
        const file = lib.manifest.exports.modules[m];
        if (file) addImp(lib.importSpecifierFor(file), globalJsName(m, 'mod'));
      }
      for (const f of refs.functions) {
        const file = lib.manifest.exports.functions[f];
        if (file) addImp(lib.importSpecifierFor(file), globalJsName(f, 'fn'));
      }
      for (const v of refs.variables) {
        const file = lib.manifest.exports.variables[v];
        if (file) addImp(lib.importSpecifierFor(file), globalJsName(v, 'var'));
      }
    }
    // Side-effect imports first so library top-level statements (e.g. setting
    // __ctx.$slop) run before the consumer body, matching include semantics
    const seenSideEffect = new Set<string>();
    for (const lib of externalLibraries) {
      for (const spec of lib.sideEffectSpecifiers ?? []) {
        if (seenSideEffect.has(spec)) continue;
        seenSideEffect.add(spec);
        output += `import "${spec}";\n`;
      }
    }
    for (const [spec, syms] of importsBySpec) {
      output += `import { ${[...syms].join(', ')} } from "${spec}";\n`;
    }
  }

  // Add font base64 imports for each resolved font.
  const seenImports = new Set<string>();
  for (const [fontFamily, sanitized] of resolvedFonts) {
    if (seenImports.has(sanitized)) continue;
    seenImports.add(sanitized);
    const runtimeDir = options?.runtimePath ?
        path.dirname(options.runtimePath).replace(/\\/g, '/') :
        './runtime';
    const importPath = `${runtimeDir}/fonts/${sanitized}_base64.js`;
    const varName = `__font_${sanitized.replace(/-/g, '_')}`;
    output += `import { fontBase64 as ${varName} } from "${importPath}";\n`;
  }

  // Add data imports for each resolved surface file (decoded pixels or matrix)
  for (const [filename, info] of encounteredSurfaceData) {
    const runtimeDir = options?.runtimePath ?
        path.dirname(options.runtimePath).replace(/\\/g, '/') :
        './runtime';
    const importPath = `${runtimeDir}/surface_data/${info.stem}_data.js`;
    output += `import { ${info.exportName} } from "${importPath}";\n`;
  }

  // One shared table lives in the runtime, so a `text()` routed through a
  // library's own module reads the same faces the consumer embedded
  output += `Object.assign(__font_registry, {\n`;
  const seenSanitized = new Set<string>();
  for (const [fontFamily, sanitized] of resolvedFonts) {
    if (seenSanitized.has(sanitized)) continue;
    seenSanitized.add(sanitized);
    const varName = `__font_${sanitized.replace(/-/g, '_')}`;
    output += `  ${JSON.stringify(sanitized)}: ${varName},\n`;
  }
  output += `});\n\n`;

  output += BUILTIN_CONSTANTS_CODE + '\n';

  if (declarations.length) {
    output += declarations.join('\n') + '\n\n';
  }

  const topLevelVarKeys =
      new Set(declOrder.filter(k => k.startsWith('var:')).map(k => k.slice(4)));
  const alreadyDeclaredAtTop = new Set<string>([
    ...PRE_DECLARED_VARS,
    '$color',
    '$idx',
    ...topLevelVarKeys,
  ]);

  for (const v of dynamicScopeVars) {
    if (!v.startsWith('$') && !alreadyDeclaredAtTop.has(v)) {
      output += `let ${v}: any;\n`;
    }
  }

  const {referenced} = collectIdentifierUsage(program);
  const moduleLevelDeclared = new Set<string>([
    ...alreadyDeclaredAtTop,
    ...dynamicScopeVars,
    ...BUILTIN_FUNCTIONS,
    // Variables imported from external libraries
    ...[...externalVariableNames].map(escapeName),
    // Reserved module-level names emitted by the compiler/runtime
    'result',
    'background',
    'Manifold',
    'CrossSection',
    'wasm',
    '__NO_ARG',
  ]);

  const undefinedNames =
      [...referenced]
          .filter(
              n => !n.startsWith('__') && !n.startsWith('$') &&
                  !moduleLevelDeclared.has(n))
          .sort();
  for (const name of undefinedNames) {
    output += `let ${name}: any = undefined;\n`;
  }

  // children() used outside a module's scope - warns and yields nothing. The
  // children stack is empty at top level, so this resolves to empty geometry.
  output +=
      `function children(i?: any): any { const __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 }; return __c.fn ? __c.fn(i) : Manifold.union([]); }\n`;

  if (geometryLines.length === 0) {
    output += `export const result = Manifold.union([]);\n`;
  } else {
    output += `const __result_items: any[] = [];\n`;
    output += `const __background_items: any[] = [];\n`;
    output += `${geometryLines.join('\n')}\n`;
    output +=
        `export const result = __union2d3d(__applyRoot(__result_items));\n`;
    output +=
        `export const background = __union2d3d(__applyRoot(__background_items, true));\n`;
  }
  output +=
      `export const __viewport = { vpr: __ctx.$vpr, vpt: __ctx.$vpt, vpd: __ctx.$vpd, vpf: __ctx.$vpf };\n`;

  return output;
}


// Separate library compilation
export interface CompiledLibraryFile {
  sourceRel: string;
  outRel: string;
  code: string;
}
export interface CompiledLibrary {
  manifest: LibraryManifest;
  files: CompiledLibraryFile[];
}

type LibDeclKind = 'module'|'function'|'variable';
function declKindAndName(stmt: Statement): {kind: LibDeclKind; name: string}|
    undefined {
  if (stmt.kind === 'moduleDecl') return {kind: 'module', name: stmt.name};
  if (stmt.kind === 'functionDecl') return {kind: 'function', name: stmt.name};
  if (stmt.kind === 'variableDecl' && !(stmt.name.startsWith('$')))
    return {kind: 'variable', name: stmt.name};
  return undefined;
}

const LIB_BUILTIN_CONSTS = new Set(['PI', 'INF', 'NAN', 'undef', '_EPSILON']);

export function compileLibrary(
    closure: LibraryClosure,
    opts: {runtimeVersion: string; runtimePathFor: (outRel: string) => string},
    ): CompiledLibrary {
  const sourceRels = [...closure.files.keys()].sort();
  const outRelOf = (sourceRel: string) => sourceRel.replace(/\.scad$/i, '.ts');
  // Library files use `var` for top-level vars to survive circular ESM imports.
  globalVarDeclKeyword = 'var';

  signatures.clear();
  for (const [k, v] of Object.entries(BUILTIN_SIGNATURES)) {
    signatures.set(
        sigKey(k.replace(/\$mod$/, ''), 'mod'),
        {params: v, defaults: new Array(v.length).fill(undefined)});
  }
  externalModuleNames = new Set();
  externalFunctionNames = new Set();
  externalVariableNames = new Set();
  const allStatements: Statement[] = [];
  for (const rel of sourceRels)
    allStatements.push(...closure.files.get(rel)!.statements);
  collectSignatures(allStatements);
  moduleDeclRegistry = collectModuleDeclarations(allStatements);

  // Each file of the closure resolves in a scope of its own, following the
  // library's own include/use graph
  const libBind = bindLibrary(
      sourceRels.map(rel => ({
                      rel,
                      program: closure.files.get(rel)!,
                      edges: closure.edges.get(rel) ?? [],
                    })),
      closure.entryRels, currentBindOptions());
  bindResult = libBind;
  assignPrettyNames(libBind, {reserved: reservedNames()});


  // Build the per-kind export map (name -> owning source file), last-wins with
  // collisions recorded, plus the manifest signatures and per-file decl lists
  const exportsByKind = {
    module: new Map<string, string>(),
    function: new Map<string, string>(),
    variable: new Map<string, string>(),
  };
  const ambiguous: Record<string, string[]> = {};
  const manifestSignatures: Record<string, string[]> = {};
  const manifestSymbols = {
    modules: {} as Record<string, string>,
    functions: {} as Record<string, string>,
    variables: {} as Record<string, string>,
  };
  const manifestSignatureNoArg: Record<string, boolean[]> = {};
  const perFileDecls = new Map < string, {
    modules: string[];
    functions: string[];
    variables: string[]
  }
  >();

  for (const rel of sourceRels) {
    const program = closure.files.get(rel)!;
    const lists = {
      modules: [] as string[],
      functions: [] as string[],
      variables: [] as string[]
    };
    for (const stmt of program.statements) {
      const dk = declKindAndName(stmt);
      if (!dk) continue;
      const map = exportsByKind[dk.kind];
      const prior = map.get(dk.name);
      if (prior !== undefined && prior !== rel) {
        const key = `${dk.kind}:${dk.name}`;
        if (!ambiguous[key]) ambiguous[key] = [prior];
        ambiguous[key].push(rel);
      }
      map.set(dk.name, rel);  // last-wins
      const ns: Namespace =
          dk.kind === 'module' ? 'mod' : dk.kind === 'function' ? 'fn' : 'var';
      manifestSymbols[dk.kind === 'module' ? 'modules' :
                          dk.kind === 'function' ? 'functions' :
                                                   'variables'][dk.name] =
          declJsName(stmt as {name: string; binding?: Binding}, ns);
      if (dk.kind === 'module') {
        lists.modules.push(dk.name);
        manifestSignatures[sigKey(dk.name, 'mod')] =
            (stmt as any).params.map((p: any) => p.name);
        manifestSignatureNoArg[sigKey(dk.name, 'mod')] =
            (stmt as any).params.map(paramUsesNoArg);
      } else if (dk.kind === 'function') {
        lists.functions.push(dk.name);
        manifestSignatures[sigKey(dk.name, 'fn')] =
            (stmt as any).params.map((p: any) => p.name);
        manifestSignatureNoArg[sigKey(dk.name, 'fn')] =
            (stmt as any).params.map(paramUsesNoArg);
      } else {
        lists.variables.push(dk.name);
      }
    }
    perFileDecls.set(rel, lists);
  }

  // Emit each file
  const files: CompiledLibraryFile[] = [];
  for (const rel of sourceRels) {
    const program = closure.files.get(rel)!;
    const outRel = outRelOf(rel);
    files.push({
      sourceRel: rel,
      outRel,
      code: emitLibraryFile(rel, outRel, program, {
        deps: closure.deps.get(rel) ?? [],
        outRelOf,
        runtimePath: opts.runtimePathFor(outRel),
        scope: libBind.fileScopes.get(rel),
      }),
    });
  }

  const manifestFiles: LibraryManifest['files'] = {};
  for (const rel of sourceRels) {
    const lists = perFileDecls.get(rel)!;
    manifestFiles[rel] = {out: outRelOf(rel), ...lists};
  }

  const manifest: LibraryManifest = {
    manifestVersion: MANIFEST_VERSION,
    library: closure.name,
    compiledAt: new Date().toISOString(),
    runtimeVersion: opts.runtimeVersion,
    files: manifestFiles,
    exports: {
      modules: Object.fromEntries(exportsByKind.module),
      functions: Object.fromEntries(exportsByKind.function),
      variables: Object.fromEntries(exportsByKind.variable),
    },
    ambiguous,
    symbols: manifestSymbols,
    signatures: manifestSignatures,
    signatureNoArg: manifestSignatureNoArg,
  };

  globalVarDeclKeyword = 'let';
  return {manifest, files};
}

function emitLibraryFile(
    sourceRel: string,
    outRel: string,
    program: Program,
    ctx: {
      deps: string[];
      outRelOf: (sourceRel: string) => string;
      runtimePath: string;
      scope?: Scope | undefined;
    },
    ): string {
  // Reset per-file emitter state
  currentScope = ctx.scope;
  tailTempCounter = 0;
  dynamicScopeVars = new Set();
  encounteredFonts = new Set();
  encounteredSurfaceData.clear();
  currentRuntimePath = ctx.runtimePath;
  currentMainFilename = program.filename ?? '';
  currentSourceFilename = currentMainFilename;

  // Top-level declarations, deduped last-wins (matching compile())
  const declMap = new Map < string, {
    stmt: Statement;
    code: string
  }
  >();
  const declOrder: string[] = [];
  const ownNames = {
    module: new Set<string>(),
    function: new Set<string>(),
    variable: new Set<string>()
  };
  const exportedSymbols: string[] = [];

  for (const stmt of program.statements) {
    if (stmt.kind === 'variableDecl' || stmt.kind === 'moduleDecl' ||
        stmt.kind === 'functionDecl') {
      let key: string;
      if (stmt.kind === 'variableDecl')
        key = `var:${declJsName(stmt, 'var')}`;
      else if (stmt.kind === 'moduleDecl')
        key = `fn:${declJsName(stmt, 'mod')}`;
      else
        key = `fn:${declJsName(stmt, 'fn')}`;
      if (!declMap.has(key)) declOrder.push(key);
      declMap.set(key, {stmt, code: compileDeclaration(stmt)});
      const dk = declKindAndName(stmt);
      if (dk) ownNames[dk.kind].add(dk.name);
    }
  }

  const declarations: string[] = [];
  for (const key of declOrder) {
    const entry = declMap.get(key)!;
    declarations.push(entry.code);
    const dk = declKindAndName(entry.stmt);
    if (dk?.kind === 'module')
      exportedSymbols.push(declJsName(entry.stmt as any, 'mod'));
    else if (dk?.kind === 'function')
      exportedSymbols.push(declJsName(entry.stmt as any, 'fn'));
    else if (dk?.kind === 'variable')
      exportedSymbols.push(declJsName(entry.stmt as any, 'var'));
  }

  // Resolve cross-file references to imports
  const refs = collectProgramReferences(program.statements);
  const importsBySpec = new Map<string, Set<string>>();
  const importedVarNames = new Set<string>();
  const addImp = (ownerRel: string, sym: string) => {
    const spec = relImportSpecifier(outRel, ctx.outRelOf(ownerRel));
    let set = importsBySpec.get(spec);
    if (!set) {
      set = new Set();
      importsBySpec.set(spec, set);
    }
    set.add(sym);
  };

  // A reference reaches another file only when it resolves, in this file's scope, to a binding that another file declares. A name resolving to a builtin or to nothing, is never imported
  const importIfForeign = (name: string, ns: Namespace): Binding|null => {
    const b = ctx.scope ? lookup(ctx.scope, name, ns) : null;
    if (!b || b.kind === 'builtin' || b.kind === 'external') return null;
    if (!b.file || b.file === sourceRel) return null;
    addImp(b.file, b.jsName);
    return b;
  };

  for (const m of refs.modules) {
    if (ownNames.module.has(m)) continue;
    if (importIfForeign(m, 'mod')) continue;
    if (!BUILTIN_MODULES.has(m) && !lookup(ctx.scope ?? null, m, 'mod')) {
      console.warn(`Warning: library ${sourceRel}: unresolved module '${
          m}' (emitting no-op call)`);
    }
  }
  for (const f of refs.functions) {
    if (ownNames.function.has(f)) continue;
    importIfForeign(f, 'fn');
  }
  for (const v of refs.variables) {
    if (LIB_BUILTIN_CONSTS.has(v) || ownNames.variable.has(v)) continue;
    const b = importIfForeign(v, 'var');
    if (b) importedVarNames.add(b.jsName);
  }

  // Side-effect imports for under-root deps, to preserve include-time execution
  // order
  let sideEffectBlock = '';
  for (const dep of ctx.deps) {
    if (dep === sourceRel) continue;
    sideEffectBlock +=
        `import "${relImportSpecifier(outRel, ctx.outRelOf(dep))}";\n`;
  }

  let importBlock = '';
  for (const [spec, syms] of importsBySpec) {
    importBlock += `import { ${[...syms].join(', ')} } from "${spec}";\n`;
  }

  // Undefined fallbacks: referenced identifiers that aren't local, imported, a
  // builtin const, or a $-special. OpenSCAD resolves unknown reads to undef
  const {referenced} = collectIdentifierUsage(program);
  // Names this file already declares, under the names it actually emits them with everything else that is referenced needs an `undefined` fallback
  const declaredHere = (names: Set<string>, ns: Namespace) =>
      [...names].map(n => globalJsName(n, ns));
  const localDeclared = new Set<string>([
    ...declaredHere(ownNames.variable, 'var'),
    ...declaredHere(ownNames.module, 'mod'),
    ...declaredHere(ownNames.function, 'fn'),
    ...exportedSymbols,
    ...[...importsBySpec.values()].flatMap(set => [...set]),
    ...LIB_BUILTIN_CONSTS,
    ...RUNTIME_SYMBOLS,
    ...importedVarNames,
    'Manifold',
    'CrossSection',
    'wasm',
    '__NO_ARG',
  ]);
  const undefinedNames =
      [...referenced]
          .filter(
              n => !n.startsWith('__') && !n.startsWith('$') &&
                  !localDeclared.has(n))
          .sort();

  let out = buildRuntimeImport(ctx.runtimePath);
  out += sideEffectBlock;
  out += importBlock;
  out += BUILTIN_CONSTANTS_CODE;
  for (const name of undefinedNames) out += `let ${name}: any = undefined;\n`;
  out += '\n';
  if (declarations.length) out += declarations.join('\n') + '\n';
  if (exportedSymbols.length)
    out += `\nexport { ${exportedSymbols.join(', ')} };\n`;
  return out;
}

// Relative ES import specifier from one output file to another
function relImportSpecifier(fromOutRel: string, toOutRel: string): string {
  let rel =
      path.relative(path.dirname(fromOutRel), toOutRel).replace(/\\/g, '/');
  rel = rel.replace(/\.ts$/i, '.js');
  if (!rel.startsWith('.')) rel = './' + rel;
  return rel;
}

// build geometry IR trees from top level statements
export function buildProgramIR(program: Program): IRNode[] {
  const modules = collectModuleDeclarations(program.statements);
  const ctx = baseIRContext(modules);
  const out: IRNode[] = [];
  for (const stmt of program.statements) {
    const ir = lowerGeometryToIR(stmt, ctx);
    if (ir && ir.kind !== 'empty') out.push(ir);
  }
  return out;
}

// Declarations
const PRE_DECLARED_VARS = new Set([
  '$fn', '$fa', '$fs', '$vpr', '$vpt', '$vpd', '$vpf', '$parent_modules', '$t',
  '$preview', '$color', '$idx', 'PI', 'INF', 'NAN', 'undef', '_EPSILON'
]);

// Emit a `use`d file's scope as a JS scope of its own. The file's top-level variables become plain bindings inside an IIFE, so they keep their OpenSCAD names and are private by construction: the consumer has no way to reach them and no way to shadow them. The modules and functions declared alongside them close over those bindings and are returned, for the caller to publish under their global names.
function compileUsedFileScope(scope: ScopeStmt, unitName: string): {code: string; exports: string[]} {
  // `$` variables never reach here (they stay dynamically scoped) and the
  // pre-declared names refer to the program-wide bindings, so neither becomes
  // private to the file.
  const privateNames: string[] = [];
  const seenPrivate = new Set<string>();
  for (const s of scope.statements) {
    if (s.kind !== 'variableDecl') continue;
    if (s.name.startsWith('$') || PRE_DECLARED_VARS.has(s.name)) continue;
    const n = bindJsName(s);
    if (seenPrivate.has(n)) continue;
    seenPrivate.add(n);
    privateNames.push(n);
  }

  return (() => {
    // Deduplicate by output name the same way the top level does: last
    // declaration wins, at the position where the name first appeared.
    const declCode = new Map<string, string>();
    const declOrder: string[] = [];
    const exports: string[] = [];
    const savedSourceFilename = currentSourceFilename;

    for (const s of scope.statements) {
      if (s.filename) currentSourceFilename = s.filename;
      let key: string;
      if (s.kind === 'variableDecl')
        key = `var:${declJsName(s, 'var')}`;
      else if (s.kind === 'moduleDecl')
        key = declJsName(s, 'mod');
      else if (s.kind === 'functionDecl')
        key = declJsName(s, 'fn');
      else
        continue;
      if (!declCode.has(key)) {
        declOrder.push(key);
        if (s.kind !== 'variableDecl') exports.push(key);
      }
      declCode.set(key, compileDeclaration(s, {assignmentOnly: true}));
    }

    currentSourceFilename = savedSourceFilename;

    const lines = [`const ${unitName} = (() => {`];
    if (privateNames.length) {
      lines.push(`  let ${
          privateNames.map(n => `${n}: any = undef`).join(', ')};`);
    }
    for (const k of declOrder) {
      lines.push('  ' + declCode.get(k)!.split('\n').join('\n  '));
    }
    lines.push(`  return {${exports.join(', ')}};`);
    lines.push('})();');
    return {code: lines.join('\n'), exports};
  })();
}

// Same, for the mangling scheme's own fallback shape.
function legacyJsName(name: string, ns: Namespace): string {
  const base = escapeName(name);
  return ns === 'fn' ? `${base}_fn` : ns === 'mod' ? `${base}$mod` : base;
}

// The JS name of a bound parameter, let binding or loop variable. Comes from the binder
function bindJsName(node: {name: string; binding?: Binding|null|undefined}):
    string {
  return node.binding ? node.binding.jsName : escapeName(node.name);
}

// Scope that top-level names resolve in for the unit currently being emitted: the program's global scope, or the current file's scope inside a library.
let currentScope: Scope|undefined;

function globalJsName(name: string, ns: Namespace): string {
  const b = currentScope ? lookup(currentScope, name, ns) : null;
  return b ? b.jsName : legacyJsName(name, ns);
}

function declJsName(
    stmt: {name: string; binding?: Binding|undefined},
    ns: Namespace): string {
  if (stmt.binding) return stmt.binding.jsName;
  const base = escapeName(stmt.name);
  return ns === 'fn' ? `${base}_fn` : ns === 'mod' ? `${base}$mod` : base;
}

function compileDeclaration(
    stmt: Statement, opts?: {assignmentOnly?: boolean}): string {
  const withLeading = (code: string) => {
    const leading = leadingCommentLines(stmt);
    const suffix = trailingCommentText(stmt);
    return `${leading.length ? `${leading.join('\n')}\n` : ''}${code}${suffix}`;
  };

  switch (stmt.kind) {
    case 'variableDecl': {
      const name = declJsName(stmt, 'var');
      if (stmt.name.startsWith('$') && stmt.name !== '$children') {
        return withLeading(
            `${svTarget(stmt.name)} = ${compileExpr(stmt.value)};`);
      }
      if (PRE_DECLARED_VARS.has(stmt.name)) {
        return withLeading(`${name} = ${compileExpr(stmt.value)};`);
      }
      // When the variable is hoisted to `undef` up front, emit a bare
      // assignment so it lands at its first-occurrence position without a
      // redeclaration
      if (opts?.assignmentOnly) {
        return withLeading(`${name} = ${compileExpr(stmt.value)};`);
      }
      return withLeading(
          `${globalVarDeclKeyword} ${name}: any = ${compileExpr(stmt.value)};`);
    }

    case 'moduleDecl': {
      const dedup = deduplicateParams(stmt.params);
      const isDyn = (n: string) => n.startsWith('$') && n !== '$children';
      const renamedParams: string[] = [];
      const params =
          dedup
              .map(p => {
                const base = bindJsName(p);
                const selfRef = !!p.defaultValue &&
                    nodeReferencesIdentifier(p.defaultValue, p.name);
                let pname: string;
                if (isDyn(p.name)) {
                  pname = `${base}__arg`;
                } else if (selfRef) {
                  pname = `${base}__arg`;
                  renamedParams.push(base);
                } else {
                  pname = base;
                }
                if (paramUsesNoArg(p)) return `${pname}: any`;
                return p.defaultValue ?
                    `${pname}: any = ${compileExpr(p.defaultValue)}` :
                    `${pname}: any`;
              })
              .join(', ');
      const defaultsPrologue = emitNoArgDefaults(dedup, '  ');
      if (!dedup.some(p => p.name === stmt.name) &&
          moduleAlwaysRecurses(stmt.body, stmt.name)) {
        const base = currentMainFilename ? path.basename(currentMainFilename) :
                                           '<unknown>';
        const line = stmt.loc?.start.line ?? 0;
        throw new Error(`Recursion detected calling module '${
            stmt.name}' in file ${base}, line ${line}`);
      }
      const localParams = dedup.map(bindJsName);
      const dollarParams = dedup.filter(p => isDyn(p.name)).map(bindJsName);
      const body = compileModuleBody(
          stmt.body, stmt.name, localParams, dollarParams, renamedParams);
      return withLeading(`function ${declJsName(stmt, 'mod')}(${
          params}): any {\n${defaultsPrologue}${body}\n}`);
    }

    case 'functionDecl': {
      const dedup = deduplicateParams(stmt.params);
      const renamedParams: string[] = [];
      const params =
          dedup
              .map(p => {
                const base = bindJsName(p);
                const selfRef = !!p.defaultValue &&
                    nodeReferencesIdentifier(p.defaultValue, p.name);
                let pname = base;
                if (selfRef) {
                  pname = `${base}__arg`;
                  renamedParams.push(base);
                }
                if (paramUsesNoArg(p)) return `${pname}: any`;
                return p.defaultValue ?
                    `${pname}: any = ${compileExpr(p.defaultValue)}` :
                    `${pname}: any`;
              })
              .join(', ');
      const localParams = dedup.map(bindJsName);
      const rebinds =
          renamedParams.map(n => `  let ${n}: any = ${n}__arg;\n`).join('');
      const defaultsPrologue = emitNoArgDefaults(dedup, '  ');
      // Tail-recursive functions are lowered into an iterative loop so deep
      // recursion doesn't overflow
      if (!dedup.some(p => p.name === stmt.name) &&
          hasSelfTailCall(stmt.body, stmt.name)) {
        if (tailAlwaysRecurses(stmt.body, stmt.name)) {
          const base = currentMainFilename ?
              path.basename(currentMainFilename) :
              '<unknown>';
          const line = stmt.loc?.start.line ?? 0;
          throw new Error(`Recursion detected calling function '${
              stmt.name}' in file ${base}, line ${line}`);
        }
        const loopBody = (emitTailBody(stmt.body, stmt.name, dedup, '    '));
        return withLeading(`function ${declJsName(stmt, 'fn')}(${
            params}): any {\n${rebinds}${defaultsPrologue}  while (true) {\n${
            loopBody}\n  }\n}`);
      }
      const bodyExpr =
          (compileExpr(stmt.body));
      return withLeading(
          `function ${declJsName(stmt, 'fn')}(${params}): any {\n${rebinds}${
              defaultsPrologue}  return ${bodyExpr};\n}`);
    }

    default:
      return `/* unsupported declaration: ${(stmt as Statement).kind}${
          locTag(stmt)} */`;
  }
}

// Apply OpenSCAD defaults: missing or sentinel-filled args use the default,
// while an explicit `undef` stays `undef`
function emitNoArgDefaults(params: Param[], indent: string): string {
  let out = '';
  params.forEach((p, i) => {
    if (!paramUsesNoArg(p)) return;
    const pname = bindJsName(p);
    out += `${indent}if (${pname} === __NO_ARG || arguments.length <= ${i}) ${
        pname} = ${compileExpr(p.defaultValue!)};\n`;
  });
  return out;
}

// True when an expression references the identifier name anywhere within it
function nodeReferencesIdentifier(
    node: KindedNode|undefined, name: string): boolean {
  if (!node) return false;
  if (node.kind === 'identifier' && node.name === name) return true;
  let found = false;
  forEachChild(node, child => {
    found = found || nodeReferencesIdentifier(child, name);
  });
  return found;
}

// Compile-time divergence detection for non-tail recursion
let userFunctionDefs: Map<string, import('./ast.js').FunctionDeclStmt> =
    new Map();

// Depth at which constant evaluation gives up and declares divergence. Kept
// below the JS engine's own stack limit so the cap fires before evalConstExpr
// overflows
const CONST_EVAL_DEPTH_CAP = 4000;
const CONST_UNKNOWN = Symbol('const-unknown');
const CONST_DIVERGE = Symbol('const-diverge');

function collectFunctionDefs(
    stmts: Statement[],
    into: Map<string, import('./ast.js').FunctionDeclStmt>): void {
  for (const s of stmts) {
    switch (s.kind) {
      case 'functionDecl':
        into.set(s.name, s);
        break;
      case 'block':
      case 'scope':
        collectFunctionDefs(s.statements, into);
        break;
      case 'if':
        if (s.thenBody.kind === 'block')
          collectFunctionDefs(s.thenBody.statements, into);
        if (s.elseBody?.kind === 'block')
          collectFunctionDefs(s.elseBody.statements, into);
        break;
      case 'for':
        if (s.body.kind === 'block')
          collectFunctionDefs(s.body.statements, into);
        break;
      case 'moduleDecl':
        if (s.body.kind === 'block')
          collectFunctionDefs(s.body.statements, into);
        break;
    }
  }
}

// True when `node` contains a call to `name` anywhere (over-approximate; used
// only to decide whether a function is worth attempting to evaluate)
function containsCallTo(node: KindedNode|undefined, name: string): boolean {
  if (!node) return false;
  if (node.kind === 'call' && node.name === name) return true;
  let found = false;
  forEachChild(node, child => {
    found = found || containsCallTo(child, name);
  });
  return found;
}

// Evaluate a purely-constant expression. Returns the value, CONST_UNKNOWN when
// it depends on anything not statically known (vectors, builtins, $vars, etc),
// or throws CONST_DIVERGE when recursion exceeds the cap
function evalConstExpr(expr: Expr, env: Map<string, any>, depth: number): any {
  if (depth > CONST_EVAL_DEPTH_CAP) throw CONST_DIVERGE;
  switch (expr.kind) {
    case 'number':
      return expr.value;
    case 'boolean':
      return expr.value;
    case 'string':
      return expr.value;
    case 'undef':
      return undefined;
    case 'identifier':
      return env.has(expr.name) ? env.get(expr.name) : CONST_UNKNOWN;
    case 'group':
      return evalConstExpr(expr.expr, env, depth);
    case 'unary': {
      const v = evalConstExpr(expr.operand, env, depth);
      if (v === CONST_UNKNOWN) return CONST_UNKNOWN;
      if (expr.op === '-') return typeof v === 'number' ? -v : CONST_UNKNOWN;
      if (expr.op === '+') return typeof v === 'number' ? +v : CONST_UNKNOWN;
      if (expr.op === '!') return typeof v === 'boolean' ? !v : CONST_UNKNOWN;
      return CONST_UNKNOWN;
    }
    case 'binary': {
      const l = evalConstExpr(expr.left, env, depth);
      if (l === CONST_UNKNOWN) return CONST_UNKNOWN;
      const r = evalConstExpr(expr.right, env, depth);
      if (r === CONST_UNKNOWN) return CONST_UNKNOWN;
      const nums = typeof l === 'number' && typeof r === 'number';
      switch (expr.op) {
        case '+':
          return nums ? l + r : CONST_UNKNOWN;
        case '-':
          return nums ? l - r : CONST_UNKNOWN;
        case '*':
          return nums ? l * r : CONST_UNKNOWN;
        case '/':
          return nums ? l / r : CONST_UNKNOWN;
        case '%':
          return nums ? l % r : CONST_UNKNOWN;
        case '<':
          return nums ? l < r : CONST_UNKNOWN;
        case '>':
          return nums ? l > r : CONST_UNKNOWN;
        case '<=':
          return nums ? l <= r : CONST_UNKNOWN;
        case '>=':
          return nums ? l >= r : CONST_UNKNOWN;
        case '==':
          return l === r;
        case '!=':
          return l !== r;
        case '&&':
          return (typeof l === 'boolean' && typeof r === 'boolean') ?
              (l && r) :
              CONST_UNKNOWN;
        case '||':
          return (typeof l === 'boolean' && typeof r === 'boolean') ?
              (l || r) :
              CONST_UNKNOWN;
        default:
          return CONST_UNKNOWN;
      }
    }
    case 'ternary': {
      const c = evalConstExpr(expr.condition, env, depth);
      if (typeof c !== 'boolean') return CONST_UNKNOWN;
      return evalConstExpr(c ? expr.ifTrue : expr.ifFalse, env, depth);
    }
    case 'let': {
      const newEnv = new Map(env);
      for (const a of expr.assignments) {
        if (a.name.startsWith('$')) return CONST_UNKNOWN;
        const v = evalConstExpr(a.value, newEnv, depth);
        if (v === CONST_UNKNOWN) return CONST_UNKNOWN;
        newEnv.set(a.name, v);
      }
      return evalConstExpr(expr.body, newEnv, depth);
    }
    case 'call': {
      const fn = userFunctionDefs.get(expr.name);
      if (!fn)
        return CONST_UNKNOWN;  // builtin / unknown — not statically evaluable
      const dedup = deduplicateParams(fn.params);
      const provided = resolveArgsToParams(expr.args, dedup);
      const callEnv = new Map<string, any>();
      for (let i = 0; i < dedup.length; i++) {
        let v: any;
        if (provided[i] !== undefined)
          v = evalConstExpr(provided[i]!, env, depth);
        else if (dedup[i]!.defaultValue)
          v = evalConstExpr(dedup[i]!.defaultValue!, callEnv, depth);
        else
          v = undefined;
        if (v === CONST_UNKNOWN) return CONST_UNKNOWN;
        callEnv.set(dedup[i]!.name, v);
      }
      return evalConstExpr(fn.body, callEnv, depth + 1);
    }
    default:
      return CONST_UNKNOWN;  // vector, range, index, member, lambda, each,
                             // echo, assert, list-comp
  }
}

// Walk top-level expressions for constant-argument calls to non-tail recursive
// functions and throw a compile error if any provably diverges. Does not
// descend into function/module bodies (those run only when called).
function detectDivergentCalls(stmts: Statement[]): void {
  const visitExpr = (expr: Expr|undefined): void => {
    if (!expr || typeof expr !== 'object') return;
    if (expr.kind === 'call') {
      const fn = userFunctionDefs.get(expr.name);
      // Only non-tail self-recursive functions can run away on the stack; tail
      // ones are compiled to loops and are intentionally not evaluated here
      if (fn && containsCallTo(fn.body, fn.name) &&
          !hasSelfTailCall(fn.body, fn.name)) {
        try {
          evalConstExpr(expr, new Map(), 0);
        } catch (e) {
          if (e === CONST_DIVERGE || e instanceof RangeError) {
            const base = currentMainFilename ?
                path.basename(currentMainFilename) :
                '<unknown>';
            const line = fn.loc?.start.line ?? expr.loc?.start.line ?? 0;
            throw new Error(`Recursion detected calling function '${
                fn.name}' in file ${base}, line ${line}`);
          }
          throw e;
        }
      }
    }
    forEachChild(expr, child => visitExpr(child as Expr));
  };
  
  const visitStmt = (s: Statement): void => {
    switch (s.kind) {
      case 'variableDecl':
        visitExpr(s.value);
        break;
      case 'moduleCall':
        s.args.forEach(a => visitExpr(a.value));
        if (s.child) visitStmt(s.child);
        break;
      case 'block':
      case 'scope':
        s.statements.forEach(visitStmt);
        break;
      case 'if':
        visitExpr(s.condition);
        visitStmt(s.thenBody);
        if (s.elseBody) visitStmt(s.elseBody);
        break;
      case 'for':
        s.variables.forEach(v => visitExpr(v.range));
        visitStmt(s.body);
        break;
        // functionDecl / moduleDecl bodies are not evaluated until called, so
        // skip
    }
  };
  stmts.forEach(visitStmt);
}

// Deduplicate parameters: keep last occurrence of each name (OpenSCAD allows
// duplicates)
function deduplicateParams(params: import('./ast.js').Parameter[]):
    import('./ast.js').Parameter[] {
  const seen = new Map<string, number>();
  for (let i = 0; i < params.length; i++) {
    seen.set(params[i]!.name, i);
  }
  return params.filter((p, i) => seen.get(p.name) === i);
}

// Tail-recursion elimination

// True when `expr` can reach a tail call to `funcName` along the positions the
// loop lowering actually handles (so detection stays in sync with emitTailBody)
function hasSelfTailCall(expr: Expr, funcName: string): boolean {
  switch (expr.kind) {
    case 'group':
      return hasSelfTailCall(expr.expr, funcName);
    case 'ternary':
      return hasSelfTailCall(expr.ifTrue, funcName) ||
          hasSelfTailCall(expr.ifFalse, funcName);
    case 'assert':
    case 'echo':
      return hasSelfTailCall(expr.expr, funcName);
    case 'let':
      if (expr.assignments.some(a => a.name.startsWith('$'))) return false;
      if (expr.assignments.some(a => a.name === funcName)) return false;
      return hasSelfTailCall(expr.body, funcName);
    case 'call':
      return expr.name === funcName;
    default:
      return false;
  }
}

/* True when EVERY tail-position leaf of `expr` is a call back to `funcName`,
  i.e. the function has no terminating branch and recurses unconditionally. This
  is statically-detectable non-termination (e.g. `function f() = f();`), which
  we reject at compile time rather than emitting a loop that would spin forever.
*/
function tailAlwaysRecurses(expr: Expr, funcName: string): boolean {
  switch (expr.kind) {
    case 'group':
      return tailAlwaysRecurses(expr.expr, funcName);
    case 'ternary':
      return tailAlwaysRecurses(expr.ifTrue, funcName) &&
          tailAlwaysRecurses(expr.ifFalse, funcName);
    case 'assert':
    case 'echo':
      return tailAlwaysRecurses(expr.expr, funcName);
    case 'let':
      if (expr.assignments.some(a => a.name.startsWith('$'))) return false;
      if (expr.assignments.some(a => a.name === funcName)) return false;
      return tailAlwaysRecurses(expr.body, funcName);
    case 'call':
      return expr.name === funcName;
    default:
      return false;
  }
}

/* True when instantiating `stmt` unconditionally calls module `moduleName`,
  i.e. the module recurses on every path with no base case (e.g. `module m()
  m();`). Conservative: only direct self-calls count, and any conditional/loop
  makes a branch escapable, so legitimate recursive modules (which always have a
  base case behind an `if`) are never flagged. OpenSCAD aborts such a module at
  runtime with a recursion error; we detect the static case at compile time.
*/
function moduleAlwaysRecurses(stmt: Statement, moduleName: string): boolean {
  switch (stmt.kind) {
    case 'moduleCall':
      return stmt.name === moduleName;
    case 'block':
      return stmt.statements.some(s => moduleAlwaysRecurses(s, moduleName));
    case 'if':
      return !!stmt.elseBody &&
          moduleAlwaysRecurses(stmt.thenBody, moduleName) &&
          moduleAlwaysRecurses(stmt.elseBody, moduleName);
    default:
      return false;
  }
}

type Param = import('./ast.js').Parameter;

// Map a call's arguments onto the parameter list, returning the supplied value
// expression for each parameter index (or undefined when omitted). Mirrors the
// positional/named resolution in compileArgList.
function resolveArgsToParams(
    args: Argument[], params: Param[]): (Expr|undefined)[] {
  const result: (Expr|undefined)[] = new Array(params.length).fill(undefined);
  const claimed: boolean[] = new Array(params.length).fill(false);
  let pos = 0;
  for (const a of args) {
    if (a.name) {
      if (a.name.startsWith('$')) continue;
      const idx = params.findIndex(p => p.name === a.name);
      if (idx >= 0) {
        result[idx] = a.value;
        claimed[idx] = true;
      }
    } else {
      while (pos < params.length && claimed[pos]) pos++;
      if (pos < params.length) {
        result[pos] = a.value;
        claimed[pos] = true;
        pos++;
      }
    }
  }
  return result;
}

/* Emit the loop continuation for a self tail call: stage the supplied arguments
 into temporaries (so they read the old parameter values), then re-bind every
 parameter - omitted ones reset to their defaults, matching a fresh call.
*/
function emitSelfTailCall(
    call: Extract<Expr, {kind: 'call'}>, params: Param[],
    indent: string): string {
  if (call.args.some(a => a.name && a.name.startsWith('$'))) {
    return `${indent}return ${compileCallExpr(call)};`;
  }
  const provided = resolveArgsToParams(call.args, params);
  const stage: string[] = [];
  const assign: string[] = [];
  for (let i = 0; i < params.length; i++) {
    const pname = bindJsName(params[i]!);
    if (provided[i] !== undefined) {
      const tmp = `__tc${tailTempCounter++}`;
      stage.push(`${indent}const ${tmp}: any = ${compileExpr(provided[i]!)};`);
      assign.push(`${indent}${pname} = ${tmp};`);
    } else if (params[i]!.defaultValue) {
      assign.push(
          `${indent}${pname} = ${compileExpr(params[i]!.defaultValue!)};`);
    } else {
      assign.push(`${indent}${pname} = undefined;`);
    }
  }
  return [...stage, ...assign, `${indent}continue;`].join('\n');
}

// Lower an expression in tail position into statements that either `return` a
// value or re-bind parameters and `continue` the enclosing `while (true)` loop.
function emitTailBody(
    expr: Expr, funcName: string, params: Param[], indent: string): string {
  switch (expr.kind) {
    case 'group':
      return emitTailBody(expr.expr, funcName, params, indent);
    case 'ternary': {
      const cond = compileExpr(expr.condition);
      const t = emitTailBody(expr.ifTrue, funcName, params, indent + '  ');
      const f = emitTailBody(expr.ifFalse, funcName, params, indent + '  ');
      return `${indent}if (__truthy(${cond})) {\n${t}\n${indent}} else {\n${
          f}\n${indent}}`;
    }
    case 'assert': {
      const condition = expr.args[0] ? compileExpr(expr.args[0].value) : 'true';
      const message =
          expr.args[1] ? compileExpr(expr.args[1].value) : '"Assertion failed"';
      return `${indent}openscad_assert_fn(${condition}, ${message});\n${
          emitTailBody(expr.expr, funcName, params, indent)}`;
    }
    case 'echo': {
      const eArgs =
          expr.args
              .map(
                  a => a.name ? `(${JSON.stringify(a.name + ' = ')} + __oecho(${
                                    compileExpr(a.value)}))` :
                                `__oecho(${compileExpr(a.value)})`)
              .join(', ');
      return `${indent}__echo(${eArgs});\n${
          emitTailBody(expr.expr, funcName, params, indent)}`;
    }
    case 'let': {
      // A $-special `let`, or one that shadows the function's own name, isn't
      // lowered into the loop - compile it through the normal (scope-aware)
      // path
      if (expr.assignments.some(
              a => a.name.startsWith('$') || a.name === funcName)) {
        return `${indent}return ${compileExpr(expr)};`;
      }
      const localAssignNames = expr.assignments.map(bindJsName);
      return (() => {
        const savedNames = expr.assignments.map(a => a.binding?.jsName);
        const lines: string[] = [];
        // Sequential let: each value sees the bindings established before it
        for (const a of expr.assignments) {
          const tmp = `__tl${tailTempCounter++}`;
          // Lambda values see their own binding (all their lookups are deferred
          // to call time), enabling let-bound recursive lambdas
          if (a.value.kind === 'lambda' && a.binding) a.binding.jsName = tmp;
          lines.push(`${indent}const ${tmp}: any = ${compileExpr(a.value)};`);
          if (a.binding) a.binding.jsName = tmp;
        }
        const body = emitTailBody(expr.body, funcName, params, indent);
        expr.assignments.forEach((a, i) => {
          const saved = savedNames[i];
          if (a.binding && saved !== undefined) a.binding.jsName = saved;
        });
        return lines.join('\n') + '\n' + body;
      })();
    }
    case 'call':
      if (expr.name === funcName) return emitSelfTailCall(expr, params, indent);
      return `${indent}return ${compileExpr(expr)};`;
    default:
      return `${indent}return ${compileExpr(expr)};`;
  }
}

// Module body compilation
function compileModuleBody(
    body: Statement, moduleName?: string, localParamNames: string[] = [],
    dollarParamNames: string[] = [], renamedParamNames: string[] = []): string {
  const stmts = body.kind === 'block' ? body.statements : [body];

  const lines: string[] = [];

  // Capture children from the stack at the start of every module body
  lines.push(
      '  let __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };');
  lines.push('  let $children: any = __c.count;');
  lines.push(
      '  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }');
  lines.push('  let __save_$parent_modules: any = __ctx.$parent_modules;');
  lines.push('  __ctx.$parent_modules = __children_stack.length;');
  lines.push('  const __items: any[] = [];');

  const decls: string[] = [];
  const geos: string[] = [];
  const dollarSaves: string[] = [];
  const dollarRestores: string[] = [];
  const dollarParamSets: string[] = [];

  for (const dp of dollarParamNames) {
    dollarSaves.push(`  let __save_${dp}: any = ${svTarget(dp)};`);
    dollarParamSets.push(`  ${svTarget(dp)} = ${dp}__arg;`);
    dollarRestores.push(`  ${svTarget(dp)} = __save_${dp};`);
  }

  const shadowLocals = new Set<string>();
  for (const s of stmts) {
    if (s.kind !== 'variableDecl' || !s.binding) continue;
    if (s.binding.kind === 'local' && shadowsOuterVar(s.binding))
      shadowLocals.add(bindJsName(s));
  }
  // A body local that shadows an outer variable is emitted under a distinct
  // name, which its binding then carries. Restored afterwards because a body
  // may be emitted more than once.
  const renamedLocals: {binding: Binding; saved: string}[] = [];

  const declaredInBody = new Set<string>(localParamNames);
  const savedDollars = new Set<string>(dollarParamNames);

  {
    for (const s of stmts) {
      if (s.kind === 'empty') continue;
      if (s.kind === 'variableDecl') {
        const name = bindJsName(s);
        const valueExpr = compileExpr(s.value);
        const commentsBefore = leadingCommentLines(s, '  ');
        const commentAfter = trailingCommentText(s);
        if (s.name.startsWith('$') && s.name !== '$children') {
          // Dynamic scoping: save/assign/restore for $ variables (in __ctx)
          if (!savedDollars.has(name)) {
            savedDollars.add(name);
            dollarSaves.push(`  let __save_${name}: any = ${svTarget(name)};`);
            dollarRestores.push(`  ${svTarget(name)} = __save_${name};`);
          }
          decls.push(...commentsBefore);
          decls.push(`  ${svTarget(name)} = ${valueExpr};${commentAfter}`);
        } else {
          const emitName = shadowLocals.has(name) ? `${name}__sl` : name;
          decls.push(...commentsBefore);
          if (declaredInBody.has(emitName)) {
            decls.push(`  ${emitName} = ${valueExpr};${commentAfter}`);
          } else {
            declaredInBody.add(emitName);
            decls.push(`  let ${emitName}: any = ${valueExpr};${commentAfter}`);
          }
          if (shadowLocals.has(name) && s.binding) {
            renamedLocals.push({binding: s.binding, saved: s.binding.jsName});
            s.binding.jsName = emitName;
          }
        }
      } else if (s.kind === 'functionDecl' || s.kind === 'moduleDecl') {
        // Indent the nested declaration
        const decl = compileDeclaration(s);
        decls.push('  ' + decl.split('\n').join('\n  '));
      } else {
        const geo = compileGeometry(s);
        if (!geo) continue;
        if (hasBackgroundModifier(s)) {
          pushCommentedLine(
              geos, s, `  __background_items.push(${geo});`, '  ');
        } else {
          pushCommentedLine(geos, s, `  __items.push(${geo});`, '  ');
        }
      }
    }
  };

  for (const r of renamedLocals) r.binding.jsName = r.saved;

  if (dollarRestores.length > 0) {
    lines.splice(4, 0, ...dollarSaves);
  }

  lines.push(...dollarParamSets);
  // Rebind self-referential parameters (renamed to `<name>__arg` in the
  // signature) to their OpenSCAD names so the body can use them normally.
  lines.push(...renamedParamNames.map(n => `  let ${n}: any = ${n}__arg;`));
  lines.push(...decls);
  lines.push(...geos);

  lines.push(`  try {`);
  lines.push(`    return __union2d3d(__items);`);
  lines.push(`  } finally {`);
  if (dollarRestores.length > 0) {
    lines.push(...dollarRestores.map(r => `  ${r}`));
  }
  lines.push(
      `    __ctx.$parent_modules = __save_$parent_modules;`);  // ← restore
  lines.push(`  }`);

  return lines.join('\n');
}

function hasBackgroundModifier(stmt: Statement): boolean {
  const m = (stmt as {modifier?: string}).modifier;
  return typeof m === 'string' && m.includes('%');
}

function isStatementBackgroundOnly(
    stmt: Statement, modules: Map<string, ModuleDeclStmtType>,
    visited: Set<string>): boolean {
  if (hasBackgroundModifier(stmt)) return true;
  switch (stmt.kind) {
    case 'empty':
    case 'variableDecl':
    case 'moduleDecl':
    case 'functionDecl':
    case 'use':
    case 'include':
      return true;
    case 'block':
      return stmt.statements.every(
          s => isStatementBackgroundOnly(s, modules, visited));
    case 'for':
      return isStatementBackgroundOnly(stmt.body, modules, visited);
    case 'if':
      return isStatementBackgroundOnly(stmt.thenBody, modules, visited) &&
          (!stmt.elseBody ||
           isStatementBackgroundOnly(stmt.elseBody, modules, visited));
    case 'moduleCall':
      if (modules.has(stmt.name)) {
        if (visited.has(stmt.name)) {
          return true;
        }
        visited.add(stmt.name);
        const decl = modules.get(stmt.name)!;
        const res = isStatementBackgroundOnly(decl.body, modules, visited);
        visited.delete(stmt.name);
        return res;
      }
      return false;
    default:
      return false;
  }
}

function isModuleCallBackgroundOnly(
    stmt: ModuleCallStmt, modules: Map<string, ModuleDeclStmtType>): boolean {
  return isStatementBackgroundOnly(stmt, modules, new Set<string>());
}

// Geometry compilation
function compileGeometry(stmt: Statement): string {
  return compileGeometryLegacy(stmt);
}

const IR_PRIMITIVES = new Set([
  'cube',
  'sphere',
  'cylinder',
  'circle',
  'square',
  'polygon',
  'polyhedron',
  'text',
  'surface',
]);

const IR_TRANSFORMS = new Set([
  'translate',
  'rotate',
  'scale',
  'mirror',
  'multmatrix',
  'resize',
  'offset',
  'color',
  'render',
  'projection',
]);

const IR_BOOLEANS =
    new Set(['union', 'difference', 'intersection', 'hull', 'minkowski']);




function wrapWithLetBindings(
    node: IRNode, bindings: Argument[], loc?: ASTNode['loc']): IRNode {
  if (bindings.length === 0) return node;
  return {
    kind: 'moduleCall',
    name: 'let',
    args: bindings,
    children: [node],
    loc,
  } as IRModuleCallNode;
}

function resolveChildrenReference(
    args: Argument[],
    boundChildren: IRNode[],
    loc?: ASTNode['loc'],
    ): IRNode {
  if (boundChildren.length === 0) return {kind: 'empty', loc};

  const iArg = findArg(args, 'i', 0);
  if (!iArg) {
    if (boundChildren.length === 1) return boundChildren[0]!;
    return {kind: 'sequence', items: boundChildren, loc} as IRSequenceNode;
  }

  if (iArg.value.kind === 'number') {
    const idx = Math.trunc(iArg.value.value);
    if (idx < 0 || idx >= boundChildren.length) return {kind: 'empty', loc};
    return boundChildren[idx]!;
  }

  let out: IRNode = {kind: 'empty', loc};
  for (let idx = boundChildren.length - 1; idx >= 0; idx--) {
    out = {
      kind: 'if',
      condition: {
        kind: 'binary',
        op: '==',
        left: iArg.value,
        right: {kind: 'number', value: idx},
      },
      thenNode: boundChildren[idx]!,
      elseNode: out,
      loc,
    } as IRIfNode;
  }
  return out;
}

function buildModuleParamBindings(
    params: import('./ast.js').Parameter[],
    callArgs: Argument[],
    ): Argument[] {
  const deduped = deduplicateParams(params);
  const bound = new Map<string, Expr>();

  for (const p of deduped) {
    bound.set(p.name, p.defaultValue ?? {kind: 'undef'});
  }

  let pos = 0;
  while (pos < callArgs.length && !callArgs[pos]!.name) {
    if (pos < deduped.length) {
      bound.set(deduped[pos]!.name, callArgs[pos]!.value);
    }
    pos++;
  }

  for (let i = pos; i < callArgs.length; i++) {
    const a = callArgs[i]!;
    if (a.name && bound.has(a.name)) {
      bound.set(a.name, a.value);
    }
  }

  return deduped.map((p) => ({name: p.name, value: bound.get(p.name)!}));
}

function tryExpandUserModuleCallToIR(
    stmt: ModuleCallStmt,
    decl: ModuleDeclStmtType,
    loweredChildren: IRNode[],
    ctx: IRLowerContext,
    ): IRNode|undefined {
  const paramBindings = buildModuleParamBindings(decl.params, stmt.args);
  const innerCtx: IRLowerContext = {
    modules: ctx.modules,
    children: loweredChildren,
    callStack: [...ctx.callStack, decl.name],
  };

  const loweredBody = lowerGeometryToIR(decl.body, innerCtx);
  if (!loweredBody) return undefined;
  return wrapWithLetBindings(loweredBody, paramBindings, stmt.loc);
}

function lowerGeometryToIR(stmt: Statement, ctx: IRLowerContext): IRNode|
    undefined {
  switch (stmt.kind) {
    case 'moduleCall':
      return lowerModuleCallToIR(stmt, ctx);
    case 'block': {
      const items: IRNode[] = [];
      const letBindings: Argument[] = [];
      let activeModules = new Map(ctx.modules);

      for (const s of stmt.statements) {
        if (s.kind === 'empty' || s.kind === 'use' || s.kind === 'include')
          continue;

        if (s.kind === 'variableDecl') {
          letBindings.push({name: s.name, value: s.value});
          continue;
        }

        if (s.kind === 'moduleDecl') {
          activeModules.set(s.name, s);
          continue;
        }

        if (s.kind === 'functionDecl') {
          return undefined;
        }

        const lowered = lowerGeometryToIR(s, {...ctx, modules: activeModules});
        if (!lowered) return undefined;
        if (lowered.kind !== 'empty')
          items.push(wrapWithLetBindings(lowered, letBindings, s.loc));
      }
      return {kind: 'sequence', items, loc: stmt.loc} as IRSequenceNode;
    }
    case 'for': {
      const body = lowerGeometryToIR(stmt.body, ctx);
      if (!body) return undefined;
      return {kind: 'for', variables: stmt.variables, body, loc: stmt.loc} as
          IRForNode;
    }
    case 'if': {
      const thenNode = lowerGeometryToIR(stmt.thenBody, ctx);
      if (!thenNode) return undefined;
      const elseNode =
          stmt.elseBody ? lowerGeometryToIR(stmt.elseBody, ctx) : undefined;
      if (stmt.elseBody && !elseNode) return undefined;
      return {
        kind: 'if',
        condition: stmt.condition,
        thenNode,
        elseNode,
        loc: stmt.loc
      } as IRIfNode;
    }
    case 'empty':
      return {kind: 'empty', loc: stmt.loc};
    case 'variableDecl':
    case 'moduleDecl':
    case 'functionDecl':
    case 'use':
    case 'include':
      return {kind: 'empty', loc: stmt.loc};
    default:
      return undefined;
  }
}

function lowerModuleCallToIR(stmt: ModuleCallStmt, ctx: IRLowerContext): IRNode|
    undefined {
  const name = stmt.name;
  const children = lowerModuleChildrenToIR(stmt.child, ctx);
  if (stmt.child && !children) return undefined;
  const loweredChildren = children ?? [];

  if (!ctx.callStack.includes(name)) {
    const decl = ctx.modules.get(name);
    if (decl && shouldInlineModuleToIR(decl, ctx)) {
      const expanded =
          tryExpandUserModuleCallToIR(stmt, decl, loweredChildren, ctx);
      if (expanded) return expanded;
    }
  }

  if (name === 'children') {
    return resolveChildrenReference(stmt.args, ctx.children, stmt.loc);
  }

  // A library's attachable override is used only when children exist
  if (stmt.ref?.mod && stmt.ref.mod.kind !== 'builtin') {
    return {
      kind: 'moduleCall',
      name,
      args: stmt.args,
      children: loweredChildren,
      loc: stmt.loc,
      ref: stmt.ref,
    } as IRModuleCallNode;
  }

  if (IR_PRIMITIVES.has(name)) {
    return {
      kind: 'primitive',
      primitive: name as IRPrimitiveNode['primitive'],
      args: stmt.args,
      loc: stmt.loc,
    };
  }

  if (IR_TRANSFORMS.has(name)) {
    const child = loweredChildren.length === 1 ?
        loweredChildren[0]! :
        ({kind: 'sequence', items: loweredChildren, loc: stmt.loc} as
         IRSequenceNode);
    return {
      kind: 'transform',
      transform: name as IRTransformNode['transform'],
      args: stmt.args,
      child,
      loc: stmt.loc,
    };
  }

  if (IR_BOOLEANS.has(name)) {
    return {
      kind: 'boolean',
      op: name as IRBooleanNode['op'],
      children: loweredChildren,
      loc: stmt.loc,
    };
  }

  return {
    kind: 'moduleCall',
    name,
    args: stmt.args,
    children: loweredChildren,
    loc: stmt.loc,
    ref: stmt.ref,
  } as IRModuleCallNode;
}

function lowerModuleChildrenToIR(
    child: Statement|undefined, ctx: IRLowerContext): IRNode[]|undefined {
  if (!child || child.kind === 'empty') return [];
  if (child.kind === 'block') {
    const items: IRNode[] = [];
    const letBindings: Argument[] = [];
    let activeModules = new Map(ctx.modules);

    for (const s of child.statements) {
      if (s.kind === 'empty' || s.kind === 'use' || s.kind === 'include')
        continue;

      if (s.kind === 'variableDecl') {
        letBindings.push({name: s.name, value: s.value});
        continue;
      }

      if (s.kind === 'moduleDecl') {
        activeModules.set(s.name, s);
        continue;
      }

      if (s.kind === 'functionDecl') {
        return undefined;
      }

      const lowered = lowerGeometryToIR(s, {...ctx, modules: activeModules});
      if (!lowered) return undefined;
      if (lowered.kind !== 'empty') {
        items.push(wrapWithLetBindings(lowered, letBindings, s.loc));
      }
    }
    return items;
  }
  const lowered = lowerGeometryToIR(child, ctx);
  if (!lowered) return undefined;
  return lowered.kind === 'empty' ? [] : [lowered];
}

function compileIRNode(node: IRNode): string {
  switch (node.kind) {
    case 'empty':
      return 'Manifold.union([])';

    case 'primitive':
      return compileIRPrimitive(node);

    case 'transform':
      return compileIRTransform(node);

    case 'boolean':
      return compileIRBoolean(node);

    case 'moduleCall':
      return compileIRModuleCall(node);

    case 'children':
      return node.indexExpr ? `children(${compileExpr(node.indexExpr)})` :
                              'children()';

    case 'sequence': {
      const items = node.items.map(compileIRNode)
                        .filter(x => x && x !== 'Manifold.union([])');
      if (items.length === 0) return 'Manifold.union([])';
      if (items.length === 1) return items[0]!;
      return `__union2d3d([\n  ${items.join(',\n  ')}\n])`;
    }

    case 'if': {
      const cond = compileExpr(node.condition);
      const thenNode = compileIRNode(node.thenNode);
      const elseNode =
          node.elseNode ? compileIRNode(node.elseNode) : 'Manifold.union([])';
      return `(__truthy(${cond}) ? ${thenNode} : ${elseNode})`;
    }

    case 'for':
      return buildNestedFor(node.variables, 0, compileIRNode(node.body));

    case 'astFallback':
      return compileGeometryLegacy(node.statement);

    default:
      return `/* unsupported ir node */`;
  }
}

function compileIRPrimitive(node: IRPrimitiveNode): string {
  switch (node.primitive) {
    case 'cube':
      return compileCube(node.args);
    case 'sphere':
      return compileSphere(node.args);
    case 'cylinder':
      return compileCylinder(node.args);
    case 'circle':
      return compileCircle(node.args);
    case 'square':
      return compileSquare(node.args);
    case 'polygon':
      return compilePolygon(node.args);
    case 'polyhedron':
      return compilePolyhedron(node.args);
    case 'text':
      return compileText(node.args);
    case 'surface':
      return compileSurface(node.args);
    default:
      return '/* unsupported primitive */';
  }
}

function compileIRTransform(node: IRTransformNode): string {
  const child = compileIRNode(node.child);
  switch (node.transform) {
    case 'translate':
      return `__translate(${child}, ${
          node.args[0] ? compileExpr(node.args[0].value) : '[0, 0, 0]'})`;
    case 'rotate': {
      const aArg = findArg(node.args, 'a', 0);
      const vArg = findArg(node.args, 'v', 1);
      const a = aArg ? compileExpr(aArg.value) : 'undefined';
      const v = vArg ? compileExpr(vArg.value) : 'undefined';
      return `__rotate(${child}, ${a}, ${v})`;
    }
    case 'scale':
      return `__scale(${child}, ${
          node.args[0] ? compileExpr(node.args[0].value) : '[1, 1, 1]'})`;
    case 'mirror':
      return `__mirror(${child}, ${
          node.args[0] ? compileExpr(node.args[0].value) : '[1, 0, 0]'})`;
    case 'multmatrix':
      return node.args[0] ?
          `__safe_transform(${child}, ${compileExpr(node.args[0].value)})` :
          child;
    case 'resize': {
      const newsize = findArg(node.args, 'newsize', 0);
      const auto = findArg(node.args, 'auto', 1);
      const ns = newsize ? compileExpr(newsize.value) : 'undefined';
      const au = auto ? compileExpr(auto.value) : 'undefined';
      return `__resize(${child}, ${ns}, ${au})`;
    }
    case 'offset': {
      const r = findArg(node.args, 'r', 0);
      const delta = findArg(node.args, 'delta');
      const amount = r ?? delta;
      const amt = amount ? compileExpr(amount.value) : '0';
      return `__safe_offset2d(${child}, ${
          amt}, "Round", 2, __ctx.$fn, __ctx.$fa, __ctx.$fs)`;
    }
    case 'color': {
      const c = findArg(node.args, 'c', 0);
      const alpha = findArg(node.args, 'alpha', 1);
      const cExpr = c ? compileExpr(c.value) : 'undefined';
      const aExpr = alpha ? compileExpr(alpha.value) : 'undefined';
      return `(() => { let __save_$color: any = __ctx.$color; __ctx.$color = __parse_color_for_scope(${
          cExpr}, ${aExpr}); try { return __apply_color(${child}, ${cExpr}, ${
          aExpr}); } finally { __ctx.$color = __save_$color; } })()`;
    }
    case 'render':
      return `/* render(${
          node.args.map(a => compileExpr(a.value)).join(', ')}) */ ${child}`;
    case 'projection': {
      const cut = findArg(node.args, 'cut', 0);
      const cutStr = cut ? compileExpr(cut.value) : 'false';
      return `__safe_project3d(${child}, ${cutStr})`;
    }
    default:
      return child;
  }
}

function compileIRBoolean(node: IRBooleanNode): string {
  const children = node.children.map(compileIRNode)
                       .filter(x => x && x !== 'Manifold.union([])');

  if (children.length === 0) return 'Manifold.union([])';
  if (children.length === 1) return children[0]!;

  switch (node.op) {
    case 'union':
      return `__union2d3d([\n  ${children.join(',\n  ')}\n])`;
    case 'difference': {
      const [first, ...rest] = children;
      return `__difference2d3d(${first}, [\n  ${rest.join(',\n  ')}\n])`;
    }
    case 'intersection':
      return `__intersection2d3d([\n  ${children.join(',\n  ')}\n])`;
    case 'hull':
      return `__hull2d3d([\n  ${children.join(',\n  ')}\n])`;
    case 'minkowski':
      return `__minkowski2d3d([\n  ${children.join(',\n  ')}\n])`;
    default:
      return `Manifold.union([\n  ${children.join(',\n  ')}\n])`;
  }
}

function buildWithChildrenCall(
    callExpr: string, children: string[], moduleName: string): string {
  if (children.length === 0) {
    return `__with_children(() => Manifold.union([]), 0, () => ${callExpr}, ${
        JSON.stringify(moduleName)})`;
  }

  const childrenCode = children.map(child => `() => (${child})`).join(',\n  ');
  const hasAwait =
      childrenCode.includes('await ') || callExpr.includes('await ');

  if (hasAwait) {
    return `await (() => { ` +
        `const __childFns = [\n  ${
               children.map(child => `async () => (${child})`)
                   .join(',\n  ')}\n]; ` +
        `return __with_children(async (i) => ` +
        `__union2d3d(await Promise.all(__pick_children(__childFns, i).map(fn => fn())))` +
        `, __childFns.length, async () => await ${callExpr}, ${
               JSON.stringify(moduleName)}); ` +
        `})()`;
  }

  return `(() => { ` +
      `const __childFns = [\n  ${
             children.map(child => `() => (${child})`).join(',\n  ')}\n]; ` +
      `return __with_children((i) => ` +
      `__union2d3d(__pick_children(__childFns, i).map(fn => fn()))` +
      `, __childFns.length, () => ${callExpr}, ${
             JSON.stringify(moduleName)}); ` +
      `})()`;
}

function compileIRModuleCall(node: IRModuleCallNode): string {
  switch (node.name) {
    case 'linear_extrude': {
      const child = node.children.length === 0 ?
          'Manifold.union([])' :
          node.children.length === 1 ?
          compileIRNode(node.children[0]!) :
          `__union2d3d([\n  ${
              node.children.map(compileIRNode).join(',\n  ')}\n])`;
      const height = findArg(node.args, 'height', 0) ?? findArg(node.args, 'h');
      const hStr = height ? compileExpr(height.value) : 'undefined';
      const twist = findArg(node.args, 'twist');
      const slices = findArg(node.args, 'slices');
      const scale = findArg(node.args, 'scale');
      const center = findArg(node.args, 'center');
      const vArg = findArg(node.args, 'v');
      const segments = findArg(node.args, 'segments');

      const fn = findArg(node.args, '$fn');
      const fa = findArg(node.args, '$fa');
      const fs = findArg(node.args, '$fs');
      const fe = findArg(node.args, '$fe');

      const opts: string[] = [];

      if (twist) opts.push(`twist: ${compileExpr(twist.value)}`);
      if (scale) opts.push(`scale: ${compileExpr(scale.value)}`);
      if (center) opts.push(`center: ${compileExpr(center.value)}`);
      if (vArg) opts.push(`v: ${compileExpr(vArg.value)}`);
      if (segments) opts.push(`segments: ${compileExpr(segments.value)}`);

      opts.push(`fn: ${fn ? compileExpr(fn.value) : '__ctx.$fn'}`);

      opts.push(`fa: ${fa ? compileExpr(fa.value) : '__ctx.$fa'}`);

      opts.push(`fs: ${fs ? compileExpr(fs.value) : '__ctx.$fs'}`);

      opts.push(`fe: ${fe ? compileExpr(fe.value) : '__ctx.$fe'}`);

      if (slices) {
        opts.push(`slices: ${compileExpr(slices.value)}`);
      }
      return opts.length ?
          `__extrude(${child}, ${hStr}, { ${opts.join(', ')} })` :
          `__extrude(${child}, ${hStr})`;
    }

    case 'rotate_extrude': {
      const child = node.children.length === 0 ?
          'Manifold.union([])' :
          node.children.length === 1 ?
          compileIRNode(node.children[0]!) :
          `__union2d3d([\n  ${
              node.children.map(compileIRNode).join(',\n  ')}\n])`;
      const angle = findArg(node.args, 'angle', 0) ?? findArg(node.args, 'a');
      const aStr = angle ? compileExpr(angle.value) : '360';
      const fn = findArg(node.args, '$fn');
      const fa = findArg(node.args, '$fa');
      const fs = findArg(node.args, '$fs');
      const fnStr = fn ? compileExpr(fn.value) : '__ctx.$fn';
      const faStr = fa ? compileExpr(fa.value) : '__ctx.$fa';
      const fsStr = fs ? compileExpr(fs.value) : '__ctx.$fs';
      return `__revolve(${child}, ${fnStr}, ${faStr}, ${fsStr}, ${aStr})`;
    }

    case 'echo': {
      const args =
          node.args
              .map(
                  a => a.name ? `(${JSON.stringify(a.name + ' = ')} + __oecho(${
                                    compileExpr(a.value)}))` :
                                `__oecho(${compileExpr(a.value)})`)
              .join(', ');
      const child = node.children.length === 0 ?
          'Manifold.union([])' :
          node.children.length === 1 ?
          compileIRNode(node.children[0]!) :
          `__union2d3d([\n  ${
              node.children.map(compileIRNode).join(',\n  ')}\n])`;
      return `(__echo(${args}), ${child})`;
    }

    case 'assert': {
      const condition = node.args[0] ? compileExpr(node.args[0].value) : 'true';
      const message =
          node.args[1] ? compileExpr(node.args[1].value) : '"Assertion failed"';
      const child = node.children.length === 0 ?
          'Manifold.union([])' :
          node.children.length === 1 ?
          compileIRNode(node.children[0]!) :
          `__union2d3d([\n  ${
              node.children.map(compileIRNode).join(',\n  ')}\n])`;
      return `(openscad_assert_fn(${condition}, ${message}), ${child})`;
    }

    case 'let': {
      let child = node.children.length === 0 ?
          'Manifold.union([])' :
          node.children.length === 1 ?
          compileIRNode(node.children[0]!) :
          `__union2d3d([\n  ${
              node.children.map(compileIRNode).join(',\n  ')}\n])`;
      for (let i = node.args.length - 1; i >= 0; i--) {
        const a = node.args[i]!;
        const name = a.name ? escapeName(a.name) : '_';
        // Lambda values bind via default parameter so they can see their own
        // name (let-bound recursive lambdas)
        if (child.includes('await ')) {
          child = a.value.kind === 'lambda' ?
              `await (async (${name} = ${compileExpr(a.value)}) => (${
                  child}))()` :
              `await (async (${name}) => (${child}))(${compileExpr(a.value)})`;
        } else {
          child = a.value.kind === 'lambda' ?
              `((${name} = ${compileExpr(a.value)}) => (${child}))()` :
              `((${name}) => (${child}))(${compileExpr(a.value)})`;
        }
      }
      return child;
    }

    case 'intersection_for': {
      const variables: ForVariable[] = node.args.map(arg => ({
                                                       name: arg.name || '_',
                                                       range: arg.value,
                                                       loc: arg.loc,
                                                     }));
      const body = node.children.length === 0 ?
          'Manifold.union([])' :
          node.children.length === 1 ?
          compileIRNode(node.children[0]!) :
          `__union2d3d([\n  ${
              node.children.map(compileIRNode).join(',\n  ')}\n])`;
      return buildNestedIntersectionFor(variables, 0, body);
    }

    default: {
      if (!moduleDeclRegistry.has(node.name) &&
          !externalModuleNames.has(node.name)) {
        const line = node.loc?.start.line;
        const where = line ? ` at line ${line}` : '';
        console.warn(`Warning: Ignoring unknown module '${node.name}'${where}`);
        return 'Manifold.union([])';
      }

      const callName = node.ref?.mod?.jsName ?? `${escapeName(node.name)}$mod`;
      const argList = compileArgList(sigKey(node.name, 'mod'), node.args);
      const children = node.children.map(compileIRNode).filter(Boolean);
      return buildWithChildrenCall(
          `${callName}(${argList})`, children, node.name);
    }
  }
}

function compileGeometryLegacy(stmt: Statement): string {
  const modifier = (stmt as {modifier?: string}).modifier;
  if (typeof modifier === 'string' && modifier.includes('*'))
    return '';  // disable modifier: subtree is ignored
  const geo = compileGeometryDispatch(stmt);
  if (geo && typeof modifier === 'string' && modifier.includes('!')) {
    return `__rootMod(${geo})`;
  }
  return geo;
}

function compileGeometryDispatch(stmt: Statement): string {
  switch (stmt.kind) {
    case 'moduleCall':
      return compileModuleCall(stmt);
    case 'block':
      return compileBlockGeometry(stmt);
    case 'for':
      return compileForGeometry(stmt);
    case 'if':
      return compileIfGeometry(stmt);
    case 'empty':
      return '';
    case 'variableDecl':
    case 'moduleDecl':
    case 'functionDecl':
      // Standalone declarations in geometry context are handled by
      // compileBlockGeometry and compileModuleBody. If we get here,
      // it means a declaration appeared outside a block (unusual).
      return '';
    case 'use':
    case 'include':
      return '';
    default:
      return `/* unsupported: ${(stmt as Statement).kind}${locTag(stmt)} */`;
  }
}

// Module call dispatch
function compileModuleCall(stmt: ModuleCallStmt): string {
  const dollarArgs =
      stmt.args.filter(arg => arg.name && arg.name.startsWith('$'));
  const userSig =
      signatures.get(sigKey(stmt.name, 'mod'));
  const extraArgs = (moduleDeclRegistry.has(stmt.name) && userSig) ?
      stmt.args.filter(
          a => a.name && !a.name.startsWith('$') &&
              !userSig.params.includes(a.name)) :
      [];

  let result: string;
  // A user or library module of the same name shadows the builtin, exactly as
  // any other declaration shadows what it is declared over. A library's own
  // wrapper still reaches the builtin, because a `use`d file resolves in a
  // scope where the override was never declared.
  if (stmt.ref?.mod && stmt.ref.mod.kind !== 'builtin') {
    result = compileUserModuleCall(stmt);
  } else
    switch (stmt.name) {
      // Primitives
      case 'cube':
        result = compileCube(stmt.args);
        break;
      case 'sphere':
        result = compileSphere(stmt.args);
        break;
      case 'cylinder':
        result = compileCylinder(stmt.args);
        break;
      case 'circle':
        result = compileCircle(stmt.args);
        break;
      case 'square':
        result = compileSquare(stmt.args);
        break;
      case 'polygon':
        result = compilePolygon(stmt.args);
        break;
      case 'polyhedron':
        result = compilePolyhedron(stmt.args);
        break;
      case 'text':
        result = compileText(stmt.args);
        break;
      case 'surface':
        result = compileSurface(stmt.args);
        break;

      // Transforms
      case 'translate':
        result = compileTransform(stmt, 'translate');
        break;
      case 'rotate':
        result = compileTransform(stmt, 'rotate');
        break;
      case 'scale':
        result = compileTransform(stmt, 'scale');
        break;
      case 'mirror':
        result = compileMirror(stmt);
        break;
      case 'multmatrix':
        result = compileMultMatrix(stmt);
        break;
      case 'resize':
        result = compileResize(stmt);
        break;
      case 'offset':
        result = compileOffset(stmt);
        break;
      case 'color':
        result = compileColor(stmt);
        break;
      case 'render':
        result = compilePassthrough(stmt, 'render');
        break;
      case 'projection':
        result = compileProjection(stmt);
        break;

      // Boolean operations
      case 'group':
        result = compileBoolOp(stmt, 'union');
        break;  // group() == implicit union
      case 'union':
        result = compileBoolOp(stmt, 'union');
        break;
      case 'difference':
        result = compileDifference(stmt);
        break;
      case 'intersection':
        result = compileBoolOp(stmt, 'intersection');
        break;
      case 'hull':
        result = compileBoolOp(stmt, 'hull');
        break;
      case 'minkowski':
        result = compileMinkowski(stmt);
        break;

      // Extrusion
      case 'linear_extrude':
        result = compileLinearExtrude(stmt);
        break;
      case 'rotate_extrude':
        result = compileRotateExtrude(stmt);
        break;

      // Builtin statement modifiers
      case 'echo':
        result = compileEchoModule(stmt);
        break;
      case 'assert':
        result = compileAssertModule(stmt);
        break;
      case 'let':
        result = compileLetModule(stmt);
        break;
      case 'children':
        result = compileChildrenModule(stmt);
        break;
      case 'intersection_for':
        result = compileIntersectionFor(stmt);
        break;

      default:
        result = compileUserModuleCall(stmt);
        break;
    }

  const dynArgs = [...dollarArgs, ...extraArgs];
  if (dynArgs.length === 0) {
    return result;
  }

  const decls: string[] = [];
  const saves: string[] = [];
  const restores: string[] = [];
  const shadowNames: string[] = [];

  for (const arg of dynArgs) {
    const name = escapeName(arg.name!);
    // $-vars live in __ctx; non-$ extra args remain module-level bindings
    if (!name.startsWith('$')) {
      dynamicScopeVars.add(name);
      shadowNames.push(name);
    }
    const valStr = compileExpr(arg.value);
    saves.push(`let __save_${name}: any = ${svTarget(name)};`);
    decls.push(`${svTarget(name)} = ${valStr};`);
    restores.push(`${svTarget(name)} = __save_${name};`);
  }

  // Extra (non-$) args are local to the module, so restore their original
  // values while compiling the child block to preserve the caller's scope
  if (shadowNames.length > 0) {
    const params = shadowNames.map(n => `${n}: any`).join(', ');
    const vals = shadowNames.map(n => `__save_${n}`).join(', ');
    if (result.includes('await ')) {
      result = `await (async (${params}) => (${result}))(${vals})`;
    } else {
      result = `((${params}) => (${result}))(${vals})`;
    }
  }

  const hasAwait = result.includes('await ');
  if (hasAwait) {
    return `await (async () => { ${saves.join(' ')} ${
        decls.join(' ')} try { return await ${result}; } finally { ${
        restores.join(' ')} } })()`;
  } else {
    return `(() => { ${saves.join(' ')} ${decls.join(' ')} try { return ${
        result}; } finally { ${restores.join(' ')} } })()`;
  }
}

// Builtin module helpers
function compileEchoModule(stmt: ModuleCallStmt): string {
  const args =
      stmt.args
          .map(
              a => a.name ? `(${JSON.stringify(a.name + ' = ')} + __oecho(${
                                compileExpr(a.value)}))` :
                            `__oecho(${compileExpr(a.value)})`)
          .join(', ');
  if (stmt.child && stmt.child.kind !== 'empty') {
    const child = compileGeometry(stmt.child);
    return `(__echo(${args}), ${child || 'Manifold.union([])'})`;
  }
  return `(__echo(${args}), Manifold.union([]))`;
}

function compileAssertModule(stmt: ModuleCallStmt): string {
  const condition = stmt.args[0] ? compileExpr(stmt.args[0].value) : 'true';
  const message =
      stmt.args[1] ? compileExpr(stmt.args[1].value) : '"Assertion failed"';
  if (stmt.child && stmt.child.kind !== 'empty') {
    const child = compileGeometry(stmt.child);
    return `(openscad_assert_fn(${condition}, ${message}), ${
        child || 'Manifold.union([])'})`;
  }
  return `(openscad_assert_fn(${condition}, ${message}), Manifold.union([]))`;
}

function compileLetModule(stmt: ModuleCallStmt): string {
  let child = 'Manifold.union([])';
  if (stmt.child && stmt.child.kind !== 'empty') {
    child = compileGeometry(stmt.child) || child;
  }
  let result = child;
  for (let i = stmt.args.length - 1; i >= 0; i--) {
    const a = stmt.args[i]!;
    const name = a.name ? escapeName(a.name) : '_';
    // Lambda values bind via default parameter so they can see their own name
    // (let-bound recursive lambdas)
    if (result.includes('await ')) {
      result = a.value.kind === 'lambda' ?
          `await (async (${name}: any = ${compileExpr(a.value)}) => (${
              result}))()` :
          `await (async (${name}: any) => (${result}))(${
              compileExpr(a.value)})`;
    } else {
      result = a.value.kind === 'lambda' ?
          `((${name}: any = ${compileExpr(a.value)}) => (${result}))()` :
          `((${name}: any) => (${result}))(${compileExpr(a.value)})`;
    }
  }
  return result;
}

function compileChildrenModule(stmt: ModuleCallStmt): string {
  if (stmt.args.length > 0) {
    return `children(${compileExpr(stmt.args[0]!.value)})`;
  }
  return `children()`;
}

// Primitive compilation
function compileCube(args: Argument[]): string {
  const size = findArg(args, 'size', 0);
  const center = findArg(args, 'center', 1);

  const sizeStr = size ? compileExpr(size.value) : '1';
  const centerStr = center ? compileExpr(center.value) : 'false';

  // `size` can be a scalar or a runtime vector expression
  return `__cube(${sizeStr}, ${centerStr})`;
}

function compileSphere(args: Argument[]): string {
  const r = findArg(args, 'r', 0);
  const d = findArg(args, 'd');
  const fn = findArg(args, '$fn');
  const fa = findArg(args, '$fa');
  const fs = findArg(args, '$fs');

  // Resolve d vs r at runtime: a pass-through wrapper may forward both with one
  const argOr = (a: Argument|undefined) =>
      (a ? compileExpr(a.value) : 'undefined');
  const radiusStr =
      `__radius(undefined, undefined, ${argOr(d)}, ${argOr(r)}, 1)`;

  const fnStr = fn ? compileExpr(fn.value) : '__ctx.$fn';
  const faStr = fa ? compileExpr(fa.value) : '__ctx.$fa';
  const fsStr = fs ? compileExpr(fs.value) : '__ctx.$fs';
  return `__sphere(${radiusStr}, ${fnStr}, ${faStr}, ${fsStr})`;
}

function compileCylinder(args: Argument[]): string {
  const h = findArg(args, 'h', 0);
  const r = findArg(args, 'r');
  const r1 = findArg(args, 'r1', 1);
  const r2 = findArg(args, 'r2', 2);
  const d = findArg(args, 'd');
  const d1 = findArg(args, 'd1');
  const d2 = findArg(args, 'd2');
  const center = findArg(args, 'center', 3);
  const fn = findArg(args, '$fn');
  const fa = findArg(args, '$fa');
  const fs = findArg(args, '$fs');

  const hStr = h ? compileExpr(h.value) : '1';

  // Resolve each radius at runtime following OpenSCAD precedence
  const argOr = (a: Argument|undefined) =>
      (a ? compileExpr(a.value) : 'undefined');
  const rLow =
      `__radius(${argOr(d1)}, ${argOr(r1)}, ${argOr(d)}, ${argOr(r)}, 1)`;
  const rHigh =
      `__radius(${argOr(d2)}, ${argOr(r2)}, ${argOr(d)}, ${argOr(r)}, 1)`;

  const fnStr = fn ? compileExpr(fn.value) : '__ctx.$fn';
  const centerStr = center ? compileExpr(center.value) : 'false';
  const faStr = fa ? compileExpr(fa.value) : '__ctx.$fa';
  const fsStr = fs ? compileExpr(fs.value) : '__ctx.$fs';

  return `__cylinder(${hStr}, ${rLow}, ${rHigh}, ${fnStr}, ${centerStr}, ${
      faStr}, ${fsStr})`;
}

function compileCircle(args: Argument[]): string {
  const r = findArg(args, 'r', 0);
  const d = findArg(args, 'd');
  const fn = findArg(args, '$fn');
  const fa = findArg(args, '$fa');
  const fs = findArg(args, '$fs');

  // Resolve d-vs-r at runtime (see __radius).
  const argOr = (a: Argument|undefined) =>
      (a ? compileExpr(a.value) : 'undefined');
  const radiusStr =
      `__radius(undefined, undefined, ${argOr(d)}, ${argOr(r)}, 1)`;

  const fnStr = fn ? compileExpr(fn.value) : '__ctx.$fn';
  const faStr = fa ? compileExpr(fa.value) : '__ctx.$fa';
  const fsStr = fs ? compileExpr(fs.value) : '__ctx.$fs';
  return `__circle(${radiusStr}, ${fnStr}, ${faStr}, ${fsStr})`;
}

function compileSquare(args: Argument[]): string {
  const size = findArg(args, 'size', 0);
  const center = findArg(args, 'center', 1);

  const sizeStr = size ? compileExpr(size.value) : '1';
  const centerStr = center ? compileExpr(center.value) : 'false';

  // `size` can be a scalar or a runtime vector expression
  return `__square(${sizeStr}, ${centerStr})`;
}

function compilePolygon(args: Argument[]): string {
  const points = findArg(args, 'points', 0);
  const paths = findArg(args, 'paths', 1);
  if (!points) return `__polygon(/* missing points */[])`;
  const pointsStr = compileExpr(points.value);
  const pathsStr = paths ? compileExpr(paths.value) : 'undefined';
  return `__polygon(${pointsStr}, ${pathsStr})`;
}

function compileText(args: Argument[]): string {
  const txt = findArg(args, 'text', 0);
  const size = findArg(args, 'size', 1);
  const font = findArg(args, 'font');
  const halign = findArg(args, 'halign');
  const valign = findArg(args, 'valign');
  const spacing = findArg(args, 'spacing');
  const dir = findArg(args, 'direction');
  const fn = findArg(args, '$fn');

  const txtStr = txt ? compileExpr(txt.value) : `""`;
  const sizeStr = size ? compileExpr(size.value) : `10`;
  const fontStr =
      font ? compileExpr(font.value) : `"Liberation Sans:style=Regular"`;
  const halignStr = halign ? compileExpr(halign.value) : `"left"`;
  const valignStr = valign ? compileExpr(valign.value) : `"baseline"`;
  const spacingStr = spacing ? compileExpr(spacing.value) : `1`;
  const dirStr = dir ? compileExpr(dir.value) : `"ltr"`;
  const fnStr = fn ? compileExpr(fn.value) : `__ctx.$fn`;

  // Track font for base64 generation and resolve variable name.
  const rawFontSpec = font && font.value.kind === 'string' ?
      font.value.value :
      DEFAULT_FONT_SPEC;
  encounteredFonts.add(rawFontSpec);
  const filename = fontSpecToFilename(rawFontSpec);
  console.log('Font file name: ', filename);

  return `__text(${txtStr}, ${sizeStr}, ${fontStr}, ${halignStr}, ${
      valignStr}, ${spacingStr}, ${dirStr}, ${fnStr}, __font_registry)`;
}

function compilePolyhedron(args: Argument[]): string {
  const points = findArg(args, 'points', 0);
  const triangles = findArg(args, 'triangles', 1);
  let faces = findArg(args, 'faces', 2);
  if (triangles) faces = triangles;

  if (!points || !faces) return `/* polyhedron: missing points or faces */`;

  return `__polyhedron(${compileExpr(points.value)}, ${
      compileExpr(faces.value)})`;
}

// Transforms
function compileTransform(
    stmt: ModuleCallStmt,
    method: string,
    ): string {
  if (!stmt.child) return 'Manifold.union([])';

  const child = compileGeometry(stmt.child);
  if (method === 'rotate') {
    const a = findArg(stmt.args, 'a', 0);
    const v = findArg(stmt.args, 'v', 1);
    return `__rotate(${child}, ${a ? compileExpr(a.value) : 'undefined'}, ${
        v ? compileExpr(v.value) : 'undefined'})`;
  }
  const vec = stmt.args[0];
  const defaultVec = method === 'translate' ? '[0, 0, 0]' : '[1, 1, 1]';
  const vecStr = vec ? compileExpr(vec.value) : defaultVec;
  return `__${method}(${child}, ${vecStr})`;
}

function compileMirror(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'Manifold.union([])';
  const child = compileGeometry(stmt.child);
  const vec = stmt.args[0];
  const vecStr = vec ? compileExpr(vec.value) : '[1, 0, 0]';
  return `__mirror(${child}, ${vecStr})`;
}

function compileMultMatrix(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'Manifold.union([])';
  const child = compileGeometry(stmt.child);
  const mat = stmt.args[0];
  if (!mat) return `${child}`;
  return `__safe_transform(${child}, ${compileExpr(mat.value)})`;
}

function compileColor(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'Manifold.union([])';
  const child = compileGeometry(stmt.child);
  const c = findArg(stmt.args, 'c', 0);
  const alpha = findArg(stmt.args, 'alpha', 1);
  const cExpr = c ? compileExpr(c.value) : 'undefined';
  const aExpr = alpha ? compileExpr(alpha.value) : 'undefined';
  if (child.includes('await ')) {
    return `await (async () => { let __save_$color: any = __ctx.$color; __ctx.$color = __parse_color_for_scope(${
        cExpr}, ${aExpr}); try { return await __apply_color(${child}, ${
        cExpr}, ${aExpr}); } finally { __ctx.$color = __save_$color; } })()`;
  }
  return `(() => { let __save_$color: any = __ctx.$color; __ctx.$color = __parse_color_for_scope(${
      cExpr}, ${aExpr}); try { return __apply_color(${child}, ${cExpr}, ${
      aExpr}); } finally { __ctx.$color = __save_$color; } })()`;
}

function compileResize(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'Manifold.union([])';
  const child = compileGeometry(stmt.child);
  const newsize = findArg(stmt.args, 'newsize', 0);
  const auto = findArg(stmt.args, 'auto', 1);
  const ns = newsize ? compileExpr(newsize.value) : 'undefined';
  const au = auto ? compileExpr(auto.value) : 'undefined';
  return `__resize(${child}, ${ns}, ${au})`;
}

function compilePassthrough(stmt: ModuleCallStmt, tag: string): string {
  if (!stmt.child) return 'Manifold.union([])';
  return `/* ${tag}(${
      stmt.args.map(a => compileExpr(a.value)).join(', ')}) */ ${
      compileGeometry(stmt.child)}`;
}

function compileOffset(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'CrossSection.square(0)';
  const child = compileGeometry(stmt.child);
  const r = findArg(stmt.args, 'r', 0);
  const delta = findArg(stmt.args, 'delta');
  const amount = r ?? delta;
  const amt = amount ? compileExpr(amount.value) : '0';
  return `__safe_offset2d(${child}, ${
      amt}, "Round", 2, __ctx.$fn, __ctx.$fa, __ctx.$fs)`;
}

function compileProjection(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'CrossSection.square(0)';
  const child = compileGeometry(stmt.child);
  if (child === 'Manifold.union([])') return 'CrossSection.square(0)';
  const cut = findArg(stmt.args, 'cut', 0);
  const cutStr = cut ? compileExpr(cut.value) : 'false';
  return `__safe_project3d(${child}, ${cutStr})`;
}

// echo()/assert() with no child are pure side-effect statements, not geometry
function isSideEffectOnlyModule(s: Statement): boolean {
  return s.kind === 'moduleCall' &&
      (s.name === 'echo' || s.name === 'assert') && !s.child;
}

function collectChildrenWithDecls(
    stmt: ModuleCallStmt, sideEffectsAsChildren = false):
    {decls: string[]; geos: string[];
     dollars: {name: string; code: string}[]} {
  if (!stmt.child) return {decls: [], geos: [], dollars: []};
  if (stmt.child.kind === 'block') {
    return compileBlockStatementsWithDecls(
        stmt.child.statements, sideEffectsAsChildren);
  }
  if (hasBackgroundModifier(stmt.child))
    return {decls: [], geos: [], dollars: []};
  if (isSideEffectOnlyModule(stmt.child) && !sideEffectsAsChildren) {
    return {decls: [`${compileGeometry(stmt.child)};`], geos: [], dollars: []};
  }
  const g = compileGeometry(stmt.child);
  return {decls: [], geos: g ? [g] : [], dollars: []};
}

function compileBlockStatementsWithDecls(
    stmts: Statement[], sideEffectsAsChildren = false):
    {decls: string[]; geos: string[];
     dollars: {name: string; code: string}[]} {
  const varDecls = new Map < string, {
    code: string;
    order: number
  }
  >();
  const otherDecls: string[] = [];
  const geos: string[] = [];
  const dollars: {name: string; code: string}[] = [];
  let order = 0;

  const collect = (list: Statement[]) => {
    for (const s of list) {
      if (s.kind === 'empty') continue;
      if (hasBackgroundModifier(s)) continue;
      if (s.kind === 'variableDecl') {
        const name = bindJsName(s);
        if (s.name.startsWith('$') && s.name !== '$children') {
          const prior = dollars.findIndex(d => d.name === name);
          const entry = {name, code: compileExpr(s.value)};
          if (prior >= 0) dollars[prior] = entry;
          else dollars.push(entry);
          continue;
        }
        const code = `${leadingCommentLines(s).join('\n')}${
            s.leadingComments?.length ? '\n' : ''}let ${name}: any = ${
            compileExpr(s.value)};${trailingCommentText(s)}`;
        const existing = varDecls.get(name);
        varDecls.set(name, {code, order: existing ? existing.order : order++});
      } else if (s.kind === 'functionDecl' || s.kind === 'moduleDecl') {
        otherDecls.push(compileDeclaration(s));
      } else if (s.kind === 'block') {
        collect(s.statements);
      } else if (isSideEffectOnlyModule(s) && !sideEffectsAsChildren) {
        // Run echo()/assert() for their side effects, but keep them out of geos
        otherDecls.push(`${compileGeometry(s)};`);
      } else {
        const g = compileGeometry(s);
        if (g) {
          const leading = leadingCommentLines(s);
          geos.push(`${leading.length ? `${leading.join('\n')}\n` : ''}${g}`);
        }
      }
    }
  };
  collect(stmts);

  const orderedVars =
      [...varDecls.values()].sort((a, b) => a.order - b.order).map(v => v.code);
  const decls = [...orderedVars, ...otherDecls];

  return {decls, geos, dollars};
}

function wrapDollarScope(
    body: string, dollars: {name: string; code: string}[]): string {
  let out = body;
  for (let i = dollars.length - 1; i >= 0; i--) {
    const d = dollars[i]!;
    const t = svTarget(d.name);
    if (out.includes('await ')) {
      out = `await (async () => { let __save_${d.name}: any = ${t}; ${t} = ${
          d.code}; try { return await ${
          returnExpr(out, '  ')}; } finally { ${t} = __save_${d.name}; } })()`;
    } else {
      out = `(() => { let __save_${d.name}: any = ${t}; ${t} = ${
          d.code}; try { return ${returnExpr(out, '  ')}; } finally { ${
          t} = __save_${d.name}; } })()`;
    }
  }
  return out;
}

function emptyGeometryWithDecls(decls: string[]): string {
  if (decls.length === 0) return 'Manifold.union([])';
  const body = `${decls.join('\n  ')}\n  return Manifold.union([]);`;
  if (decls.some(d => d.includes('await '))) {
    return `await (async () => {\n  ${body}\n})()`;
  }
  return `(() => {\n  ${body}\n})()`;
}

function compileBoolOp(stmt: ModuleCallStmt, op: string): string {
  const {decls, geos} = collectChildrenWithDecls(stmt);
  if (geos.length === 0) return emptyGeometryWithDecls(decls);

  let result: string;
  if (geos.length === 1) {
    result = geos[0]!;
  } else if (op === 'union') {
    result = `__union2d3d([\n  ${geos.join(',\n  ')}\n])`;
  } else if (op === 'intersection') {
    result = `__intersection2d3d([\n  ${geos.join(',\n  ')}\n])`;
  } else if (op === 'hull') {
    result = `__hull2d3d([\n  ${geos.join(',\n  ')}\n])`;
  } else {
    result = `Manifold.${op}([\n  ${geos.join(',\n  ')}\n])`;
  }

  if (decls.length > 0) {
    if (result.includes('await ')) {
      return `await (async () => {\n  ${decls.join('\n  ')}\n  return await ${
          returnExpr(result, '  ')};\n})()`;
    }
    return `(() => {\n  ${decls.join('\n  ')}\n  return ${
        returnExpr(result, '  ')};\n})()`;
  }
  return result;
}

function compileDifference(stmt: ModuleCallStmt): string {
  const {decls, geos} = collectChildrenWithDecls(stmt);
  if (geos.length === 0) return emptyGeometryWithDecls(decls);

  let result: string;
  if (geos.length === 1) {
    result = geos[0]!;
  } else {
    const [first, ...rest] = geos;
    result = `__difference2d3d(${first}, [\n  ${rest.join(',\n  ')}\n])`;
  }

  if (decls.length > 0) {
    if (result.includes('await ')) {
      return `await (async () => {\n  ${decls.join('\n  ')}\n  return await ${
          returnExpr(result, '  ')};\n})()`;
    }
    return `(() => {\n  ${decls.join('\n  ')}\n  return ${
        returnExpr(result, '  ')};\n})()`;
  }
  return result;
}

function compileMinkowski(stmt: ModuleCallStmt): string {
  const {decls, geos} = collectChildrenWithDecls(stmt);
  if (geos.length === 0) return emptyGeometryWithDecls(decls);

  let result: string;
  if (geos.length === 1) {
    result = geos[0]!;
  } else {
    result = `__minkowski2d3d([\n  ${geos.join(',\n  ')}\n])`;
  }

  if (decls.length > 0) {
    if (result.includes('await ')) {
      return `await (async () => {\n  ${decls.join('\n  ')}\n  return await ${
          returnExpr(result, '  ')};\n})()`;
    }
    return `(() => {\n  ${decls.join('\n  ')}\n  return ${
        returnExpr(result, '  ')};\n})()`;
  }
  return result;
}

function compileLinearExtrude(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'Manifold.union([])';
  const child = compileGeometry(stmt.child);
  if (child === 'Manifold.union([])') return 'Manifold.union([])';
  const height = findArg(stmt.args, 'height', 0) ?? findArg(stmt.args, 'h');
  const hStr = height ? compileExpr(height.value) : 'undefined';

  const twist = findArg(stmt.args, 'twist');
  const slices = findArg(stmt.args, 'slices');
  const scale = findArg(stmt.args, 'scale');
  const center = findArg(stmt.args, 'center');
  const vArg = findArg(stmt.args, 'v');
  const segments = findArg(stmt.args, 'segments');
  const fn = findArg(stmt.args, '$fn');
  const fa = findArg(stmt.args, '$fa');
  const fs = findArg(stmt.args, '$fs');
  const fe = findArg(stmt.args, '$fe');

  const opts: string[] = [];

  if (twist) {
    opts.push(`twist: ${compileExpr(twist.value)}`);
  }

  if (scale) {
    opts.push(`scale: ${compileExpr(scale.value)}`);
  }

  if (center) {
    opts.push(`center: ${compileExpr(center.value)}`);
  }

  if (vArg) {
    opts.push(`v: ${compileExpr(vArg.value)}`);
  }

  if (segments) {
    opts.push(`segments: ${compileExpr(segments.value)}`);
  }

  opts.push(`fn: ${fn ? compileExpr(fn.value) : '__ctx.$fn'}`);

  opts.push(`fa: ${fa ? compileExpr(fa.value) : '__ctx.$fa'}`);

  opts.push(`fs: ${fs ? compileExpr(fs.value) : '__ctx.$fs'}`);

  opts.push(`fe: ${fe ? compileExpr(fe.value) : '__ctx.$fe'}`);

  if (slices) {
    opts.push(`slices: ${compileExpr(slices.value)}`);
  }

  if (opts.length) {
    return `__extrude(${child}, ${hStr}, { ${opts.join(', ')} })`;
  }
  return `__extrude(${child}, ${hStr})`;
}

function compileRotateExtrude(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'Manifold.union([])';
  const child = compileGeometry(stmt.child);
  if (child === 'Manifold.union([])') return 'Manifold.union([])';

  const angle = findArg(stmt.args, 'angle', 0) ?? findArg(stmt.args, 'a');
  const aStr = angle ? compileExpr(angle.value) : '360';
  const fn = findArg(stmt.args, '$fn');
  const fa = findArg(stmt.args, '$fa');
  const fs = findArg(stmt.args, '$fs');
  const fnStr = fn ? compileExpr(fn.value) : '__ctx.$fn';
  const faStr = fa ? compileExpr(fa.value) : '__ctx.$fa';
  const fsStr = fs ? compileExpr(fs.value) : '__ctx.$fs';
  return `__revolve(${child}, ${fnStr}, ${faStr}, ${fsStr}, ${aStr})`;
}

// Block geometry
function compileBlockGeometry(block: BlockStmt): string {
  const items: {kind: 'var'|'dollar'|'func'|'geo'; name?: string;
                                                   code: string}[] = [];

  {
    for (const s of block.statements) {
      if (s.kind === 'empty') continue;
      // A '%' (background) subtree is excluded from the enclosing union.
      if (hasBackgroundModifier(s)) continue;
      if (s.kind === 'variableDecl') {
        const name = bindJsName(s);
        const code = compileExpr(s.value);
        if (s.name.startsWith('$')) {
          items.push({kind: 'dollar', name, code});
        } else {
          items.push({kind: 'var', name, code});
        }
      } else if (s.kind === 'functionDecl' || s.kind === 'moduleDecl') {
        items.push({kind: 'func', code: compileDeclaration(s)});
      } else {
        const g = compileGeometry(s);
        if (g) {
          const leading = leadingCommentLines(s);
          items.push({
            kind: 'geo',
            code: `${leading.length ? `${leading.join('\n')}\n` : ''}${g}`
          });
        }
      }
    }
  };

  // Collect geometry expressions
  const geos = items.filter(i => i.kind === 'geo').map(i => i.code);
  const result = geos.length === 0 ? 'Manifold.union([])' :
      geos.length === 1            ? geos[0]! :
                                     `__union2d3d([\n  ${geos.join(',\n  ')}\n])`;

  // Collect declarations (var, dollar, func) in order
  const decls = items.filter(i => i.kind !== 'geo');
  if (decls.length === 0) return result;

  // Build inside-out so OpenSCAD let() semantics work -> vars capture outer
  // values, $vars use dynamic scoping, and functions wrap the remaining body.
  let body = result;

  for (let i = decls.length - 1; i >= 0; i--) {
    const d = decls[i]!;
    if (d.kind === 'var') {
      if (body.includes('await ')) {
        body = `await (async (${d.name}: any) => (${body}))(${d.code})`;
      } else {
        body = `((${d.name}: any) => (${body}))(${d.code})`;
      }
    } else if (d.kind === 'dollar') {
      const t = svTarget(d.name!);
      if (body.includes('await ')) {
        body = `await (async () => { let __save_${d.name}: any = ${t}; ${t} = ${
            d.code}; try { return await ${
            returnExpr(
                body, '  ')}; } finally { ${t} = __save_${d.name}; } })()`;
      } else {
        body = `(() => { let __save_${d.name}: any = ${t}; ${t} = ${
            d.code}; try { return ${returnExpr(body, '  ')}; } finally { ${
            t} = __save_${d.name}; } })()`;
      }
    } else {
      // Wrap remaining body in IIFE with the declaration
      if (body.includes('await ')) {
        body = `await (async () => {\n  ${d.code}\n  return await ${
            returnExpr(body, '  ')};\n})()`;
      } else {
        body =
            `(() => {\n  ${d.code}\n  return ${returnExpr(body, '  ')};\n})()`;
      }
    }
  }

  return body;
}

function compileIntersectionFor(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'Manifold.union([])';

  const variables: ForVariable[] = stmt.args.map(arg => ({
                                                   name: arg.name || '_',
                                                   range: arg.value,
                                                   loc: arg.loc,
                                                 }));

  const lines = [
    '(() => {',
    '  const __items: any[] = [];',
    ...buildNestedForStatements(variables, 0, stmt.child, 1),
    '  return __intersection2d3d(__items);',
    '})()',
  ];
  const code = lines.join('\n');
  if (code.includes('await ')) {
    lines[0] = 'async () => {';
    lines[lines.length - 1] = '})()';
    return 'await (' + lines.join('\n');
  }
  return code;
}

function buildNestedIntersectionFor(
    vars: ForVariable[], idx: number, body: string): string {
  if (vars.length === 0) return 'Manifold.union([])';
  if (idx >= vars.length) return body;

  const v = vars[idx]!;
  const inner = buildNestedIntersectionFor(vars, idx + 1, body);

  if (v.range.kind === 'range') {
    const start = compileExpr(v.range.start);
    const end = compileExpr(v.range.end);
    const step = v.range.step ? compileExpr(v.range.step) : '1';
    return `__intersection2d3d((() => {\n` +
        `  const __items = [];\n` +
        `  const __start: any = ${start}, __step: any = ${step}, __end: any = ${
               end};\n` +
        `  const __cnt: any = __rangeCount(__start, __step, __end);\n` +
        `  for (let __i = 0; __i < __cnt; __i++) {\n` +
        `    const ${
               escapeName(
                   v.name)}: any = __i === 0 ? __start : __start + __i * __step;\n` +
        `    __items.push(${inner});\n` +
        `  }\n` +
        `  return __items;\n` +
        `})())`;
  }

  // vector iteration
  const rangeExpr = compileExpr(v.range);
  return `__intersection2d3d(__flat_map_iter(${rangeExpr}, (${
      bindJsName(
          v)}: any, __i: any) => { let __save_$idx: any = __ctx.$idx; __ctx.$idx = __i; try { return [${
      inner}]; } finally { __ctx.$idx = __save_$idx; } }))`;
}

// For / If geometry
function compileForGeometry(stmt: ForStmt): string {
  if (stmt.variables.length === 0) return 'Manifold.union([])';
  const lines = [
    '(() => {',
    '  const __items = [];',
    ...buildNestedForStatements(stmt.variables, 0, stmt.body, 1),
    '  return __union2d3d(__items);',
    '})()',
  ];
  const code = lines.join('\n');
  if (code.includes('await ')) {
    lines[0] = 'async () => {';
    lines[lines.length - 1] = '})()';
    return 'await (' + lines.join('\n');
  }
  return code;
}

function buildNestedFor(
    vars: ForVariable[], idx: number, body: string): string {
  if (vars.length === 0) return 'Manifold.union([])';
  if (idx >= vars.length) return body;

  const v = vars[idx]!;
  const inner = buildNestedFor(vars, idx + 1, body);

  if (v.range.kind === 'range') {
    const start = compileExpr(v.range.start);
    const end = compileExpr(v.range.end);
    const step = v.range.step ? compileExpr(v.range.step) : '1';
    return `__union2d3d((() => {\n` +
        `  const __items = [];\n` +
        `  const __start: any = ${start}, __step: any = ${step}, __end: any = ${
               end};\n` +
        `  const __cnt: any = __rangeCount(__start, __step, __end);\n` +
        `  for (let __i = 0; __i < __cnt; __i++) {\n` +
        `    const ${
               escapeName(
                   v.name)}: any = __i === 0 ? __start : __start + __i * __step;\n` +
        `    __items.push(${inner});\n` +
        `  }\n` +
        `  return __items;\n` +
        `})())`;
  }

  // vector iteration
  const rangeExpr = compileExpr(v.range);
  return `__union2d3d(__flat_map_iter(${rangeExpr}, (${
      bindJsName(
          v)}: any, __i: any) => { let __save_$idx: any = __ctx.$idx; __ctx.$idx = __i; try { return [${
      inner}]; } finally { __ctx.$idx = __save_$idx; } }))`;
}

function buildNestedForStatements(
    vars: ForVariable[],
    idx: number,
    body: Statement,
    indentLevel: number,
    ): string[] {
  const indent = '  '.repeat(indentLevel);
  if (idx >= vars.length) {
    const lines: string[] = [];
    const geo = compileGeometry(body);
    if (geo)
      pushCommentedLine(lines, body, `${indent}__items.push(${geo});`, indent);
    return lines;
  }

  const v = vars[idx]!;
  const vName = bindJsName(v);
  if (v.range.kind === 'range') {
    const start = compileExpr(v.range.start);
    const end = compileExpr(v.range.end);
    const step = v.range.step ? compileExpr(v.range.step) : '1';
    const stepName = `__step_${idx}`;
    return [
      `${indent}{`,
      `${indent}  const __start_${idx}: any = ${start}, ${stepName}: any = ${
          step}, __end_${idx}: any = ${end};`,
      `${indent}  const __cnt_${idx}: any = __rangeCount(__start_${idx}, ${
          stepName}, __end_${idx});`,
      `${indent}  for (let __i_${idx} = 0; __i_${idx} < __cnt_${idx}; __i_${
          idx}++) {`,
      `${indent}    const ${vName}: any = __i_${idx} === 0 ? __start_${
          idx} : __start_${idx} + __i_${idx} * ${stepName};`,
      ...buildNestedForStatements(vars, idx + 1, body, indentLevel + 2),
      `${indent}  }`,
      `${indent}}`,
    ];
  }

  const iterName = `__iter_${idx}`;
  const idxName = `__idx_${idx}`;
  return [
    `${indent}{`,
    // __each mirrors doForEach: scalars iterate once, undef skips, strings
    // iterate by code point
    `${indent}  const ${iterName}: any = __each(${compileExpr(v.range)});`,
    `${indent}  for (let ${idxName} = 0; ${idxName} < ${iterName}.length; ${
        idxName}++) {`,
    `${indent}    const ${vName}: any = ${iterName}[${idxName}];`,
    `${indent}    let __save_$idx: any = __ctx.$idx; __ctx.$idx = ${idxName};`,
    `${indent}    try {`,
    ...buildNestedForStatements(vars, idx + 1, body, indentLevel + 3),
    `${indent}    } finally { __ctx.$idx = __save_$idx; }`,
    `${indent}  }`,
    `${indent}}`,
  ];
}

function compileIfGeometry(stmt: IfStmt): string {
  const cond = `__truthy(${compileExpr(stmt.condition)})`;
  const then = compileGeometry(stmt.thenBody);
  if (stmt.elseBody) {
    const els = compileGeometry(stmt.elseBody);
    const lines = [
      '(() => {',
      `  if (${cond}) {`,
    ];
    pushCommentedLine(
        lines, stmt.thenBody, `    return ${returnExpr(then, '    ')};`,
        '    ');
    lines.push('  }');
    lines.push('  else {');
    pushCommentedLine(
        lines, stmt.elseBody, `    return ${returnExpr(els, '    ')};`, '    ');
    lines.push('  }');
    lines.push('})()');
    return lines.join('\n');
  }
  const lines = [
    '(() => {',
    `  if (${cond}) {`,
  ];
  pushCommentedLine(
      lines, stmt.thenBody, `    return ${returnExpr(then, '    ')};`, '    ');
  lines.push('  }');
  lines.push('  return Manifold.union([]);');
  lines.push('})()');
  return lines.join('\n');
}

// User module call
function compileUserModuleCall(stmt: ModuleCallStmt): string {
  if (!moduleDeclRegistry.has(stmt.name) &&
      !externalModuleNames.has(stmt.name)) {
    const line = stmt.loc?.start.line;
    const where = line ? ` at line ${line}` : '';
    console.warn(`Warning: Ignoring unknown module '${stmt.name}'${where}`);
    return 'Manifold.union([])';
  }

  const name = stmt.ref?.mod?.jsName ?? `${escapeName(stmt.name)}$mod`;
  const argList = compileArgList(sigKey(stmt.name, 'mod'), stmt.args);
  const {decls, geos, dollars} = stmt.child && stmt.child.kind !== 'empty' ?
      collectChildrenWithDecls(stmt, true) :
      {decls: [], geos: [], dollars: []};
  const result = wrapDollarScope(
      buildWithChildrenCall(`${name}(${argList})`, geos, stmt.name), dollars);

  if (decls.length > 0) {
    if (result.includes('await ')) {
      return `await (async () => {\n  ${decls.join('\n  ')}\n  return await ${
          returnExpr(result, '  ')};\n})()`;
    }
    return `(() => {\n  ${decls.join('\n  ')}\n  return ${
        returnExpr(result, '  ')};\n})()`;
  }
  return result;
}

// Decodes an image to raw pixels at compile time so the runtime never has to
function decodeImagePixels(filePath: string):
    {width: number, height: number, rgb: string}|undefined {
  const img = new Image();
  // A Buffer src decodes in place; the typings only admit a string
  (img as {src: unknown}).src = fs.readFileSync(filePath);
  const {width, height} = img;
  if (!width || !height) return undefined;

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const {data} = ctx.getImageData(0, 0, width, height);

  // Drop the alpha channel: surface() only reads luminance
  const rgb = Buffer.allocUnsafe(width * height * 3);
  for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) {
    rgb[j] = data[i]!;
    rgb[j + 1] = data[i + 1]!;
    rgb[j + 2] = data[i + 2]!;
  }
  return {width, height, rgb: rgb.toString('base64')};
}

// compile surface
export function compileSurface(args: Argument[]): string {
  const file = findArg(args, 'file', 0);
  const center = findArg(args, 'center', 1);
  const invert = findArg(args, 'invert', 2);

  // OpenSCAD does not abort when a surface() file is missing or cannot be
  // opened - it prints a warning and the module yields no geometry
  if (!file?.value || file.value.kind !== 'string') {
    console.warn(`Warning: surface(): no file argument given, ignoring.`);
    return 'Manifold.union([])';
  }
  const filenameStr = file.value.value;

  // OpenSCAD resolves a surface() file relative to the .scad file that contains the call
  const sourceFile = currentSourceFilename || currentMainFilename;
  const basePath = sourceFile ? path.dirname(path.resolve(sourceFile)) :
                                process.cwd();
  const filePath = path.resolve(basePath, filenameStr);
  if (!fs.existsSync(filePath)) {
    console.warn(`Warning: surface("${filenameStr}"): can't open file "${
        filePath}", ignoring.`);
    return 'Manifold.union([])';
  }

  const ext = path.extname(filePath).toLowerCase();
  const isImage = ext === '.png';  // OpenSCAD treats only PNG as an image;
                                   // everything else is a text matrix

  const stem = path.basename(filenameStr, path.extname(filenameStr))
                   .replace(/[^a-zA-Z0-9_]/g, '_');

  const currentFileDir = typeof __dirname !== 'undefined' ?
      __dirname :
      path.dirname(
          new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/i, '$1'));
  const compilerDir = path.resolve(currentFileDir, '..');
  const surfaceDataDir = path.join(compilerDir, 'runtime', 'surface_data');
  fs.mkdirSync(surfaceDataDir, {recursive: true});

  const centerStr = center ? compileExpr(center.value) : 'false';
  const invertStr = invert ? compileExpr(invert.value) : 'false';

  if (isImage) {
    // Embed the decoded pixels
    const pixels = decodeImagePixels(filePath);
    if (!pixels) {
      console.warn(`Warning: surface("${filenameStr}"): can't decode image "${
          filePath}", ignoring.`);
      return 'Manifold.union([])';
    }
    const exportName = `__img_${stem}`;

    const tsContent = `// Auto-generated by OpenSCAD compiler — do not edit\n` +
        `// Source: ${filePath}\n` +
        `// ${pixels.width}x${pixels.height}, base64 of 3 bytes (RGB) per ` +
        `pixel, row-major from the top-left\n` +
        `export const ${exportName} = { width: ${pixels.width}, height: ${
            pixels.height}, rgb: "${pixels.rgb}" };\n`;

    fs.writeFileSync(
        path.join(surfaceDataDir, `${stem}_data.ts`), tsContent, 'utf8');
    encounteredSurfaceData.set(filenameStr, {stem, exportName, kind: 'image'});

    return `__surface(${exportName}, { center: ${centerStr}, invert: ${
        invertStr}, kind: "image", fn: __ctx.$fn, fa: __ctx.$fa, fs: __ctx.$fs })`;
  } else {
    // Text matrix (.dat / .txt) - embed raw content as a string literal
    const raw = fs.readFileSync(filePath, 'utf8');
    const exportName = `__surfacedata_${stem}`;

    const tsContent = `// Auto-generated by OpenSCAD compiler — do not edit\n` +
        `// Source: ${filePath}\n` +
        `export const ${exportName} = ${JSON.stringify(raw)};\n`;

    fs.writeFileSync(
        path.join(surfaceDataDir, `${stem}_data.ts`), tsContent, 'utf8');
    encounteredSurfaceData.set(filenameStr, {stem, exportName, kind: 'text'});

    return `__surface(${exportName}, { center: ${
        centerStr}, kind: "text", fn: __ctx.$fn, fa: __ctx.$fa, fs: __ctx.$fs })`;
  }
}

// Expression compilation
function compileExpr(expr: Expr): string {
  switch (expr.kind) {
    case 'number':
      return String(expr.value);
    case 'string':
      return JSON.stringify(expr.value);
    case 'boolean':
      return String(expr.value);
    case 'undef':
      return 'undefined';
    case 'identifier': {
      // $children stays as $children (the count variable set in module body)
      if (expr.name === '$children') return '$children';
      // All other special ($-prefixed) variables read through the shared
      // runtime context so a value set in one compiled file is visible in
      // another.
      if (expr.name.startsWith('$')) return `__ctx.${expr.name}`;
      const en = escapeName(expr.name);
      return expr.binding ? expr.binding.jsName : en;
    }
    case 'vector':
      return `[${expr.elements.map(compileExpr).join(', ')}]`;
    case 'range':
      if (expr.step) {
        return `__range(${compileExpr(expr.start)}, ${
            compileExpr(expr.step)}, ${compileExpr(expr.end)})`;
      }
      return `__range(${compileExpr(expr.start)}, 1, ${compileExpr(expr.end)})`;
    case 'binary':
      if (expr.op === '^') {
        return `Math.pow(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      }
      if (expr.op === '==') {
        return `__eq(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      }
      if (expr.op === '!=') {
        return `(!__eq(${compileExpr(expr.left)}, ${compileExpr(expr.right)}))`;
      }
      if (expr.op === '+')
        return `__add(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '-')
        return `__sub(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '*')
        return `__mul(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '/')
        return `__div(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '%')
        return `__mod(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '&')
        return `__band(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '|')
        return `__bor(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '<<')
        return `__shl(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '>>')
        return `__shr(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '<')
        return `__lt(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '>')
        return `__gt(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '<=')
        return `__le(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '>=')
        return `__ge(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === '&&' || expr.op === '||') {
        return `(__truthy(${compileExpr(expr.left)}) ${expr.op} __truthy(${
            compileExpr(expr.right)}))`;
      }
      return `(${compileExpr(expr.left)} ${expr.op} ${
          compileExpr(expr.right)})`;
    case 'unary':
      if (expr.op === '-') return `__neg(${compileExpr(expr.operand)})`;
      if (expr.op === '+') return `__pos(${compileExpr(expr.operand)})`;
      if (expr.op === '!') return `(!__truthy(${compileExpr(expr.operand)}))`;
      if (expr.op === '~') return `__bnot(${compileExpr(expr.operand)})`;
      return `(${expr.op}${compileExpr(expr.operand)})`;
    case 'ternary': {
      let ifTrue = compileExpr(expr.ifTrue);
      let ifFalse = compileExpr(expr.ifFalse);
      const trueSpread = ifTrue.startsWith('...');
      const falseSpread = ifFalse.startsWith('...');
      if (trueSpread || falseSpread) {
        if (trueSpread)
          ifTrue = ifTrue.slice(3);
        else
          ifTrue = `[${ifTrue}]`;
        if (falseSpread)
          ifFalse = ifFalse.slice(3);
        else
          ifFalse = `[${ifFalse}]`;
        return `...(__truthy(${compileExpr(expr.condition)}) ? ${ifTrue} : ${
            ifFalse})`;
      }
      return `(__truthy(${compileExpr(expr.condition)}) ? ${ifTrue} : ${
          ifFalse})`;
    }
    case 'call':
      return compileCallExpr(expr);
    case 'index':
      return `__index(${compileExpr(expr.object)}, ${compileExpr(expr.index)})`;
    case 'member': {
      const memberMap: Record<string, string> = {x: '0', y: '1', z: '2'};
      const idx = memberMap[expr.property];
      if (idx !== undefined) {
        return `${compileExpr(expr.object)}?.[${idx}]`;
      }
      return `${compileExpr(expr.object)}.${expr.property}`;
    }
    case 'group': {
      const inner = compileExpr(expr.expr);
      if (inner.startsWith('...')) return inner;
      return `(${inner})`;
    }
    case 'echo': {
      const eArgs =
          expr.args
              .map(
                  a => a.name ? `(${JSON.stringify(a.name + ' = ')} + __oecho(${
                                    compileExpr(a.value)}))` :
                                `__oecho(${compileExpr(a.value)})`)
              .join(', ');
      return `(__echo(${eArgs}), ${compileExpr(expr.expr)})`;
    }
    case 'assert': {
      const condition = expr.args[0] ? compileExpr(expr.args[0].value) : 'true';
      const message =
          expr.args[1] ? compileExpr(expr.args[1].value) : '"Assertion failed"';
      return `(openscad_assert_fn(${condition}, ${message}), ${
          compileExpr(expr.expr)})`;
    }
    case 'let': {
      const localAssignNames =
          expr.assignments.filter(a => !a.name.startsWith('$'))
              .map(bindJsName);
      return (() => {
        const bound: string[] = [];
        const vals = expr.assignments.map(a => {
          const selfRec = a.value.kind === 'lambda' && !a.name.startsWith('$');
          const suppress = selfRec ? [...bound, bindJsName(a)] : bound;
          const val =
              (compileExpr(a.value));
          if (!a.name.startsWith('$')) bound.push(bindJsName(a));
          return val;
        });
        let result = (compileExpr(expr.body));
        for (let i = expr.assignments.length - 1; i >= 0; i--) {
          const a = expr.assignments[i]!;
          const name = bindJsName(a);
          const val = vals[i]!;
          if (a.name.startsWith('$')) {
            const t = svTarget(name);
            result = `(() => { const __save_${name}: any = ${t}; ${t} = ${
                val}; try { return ${result}; } finally { ${t} = __save_${
                name}; } })()`;
          } else if (a.value.kind === 'lambda') {
            result = `((${name} = ${val}) => (${result}))()`;
          } else {
            result = `((${name}) => (${result}))(${val})`;
          }
        }
        return result;
      })();
    }
    case 'each': {
      // `each <generator>` must expand every yielded element, so wrap generator
      // results with `__each` to preserve the extra `each` expansion
      if (expr.expr.kind === 'listComp') {
        return `...(__flat_map_iter(${
            compileListComp(
                expr.expr.generator)}, (__ev: any) => __each(__ev)))`;
      }
      const inner = compileExpr(expr.expr);
      if (inner.startsWith('...')) return inner;
      return `...__each(${inner})`;
    }
    case 'lambda': {
      const localParams = expr.params.map(bindJsName);
      const params =
          expr.params
              .map(
                  p => p.defaultValue ? `${bindJsName(p)} = ${
                                            compileExpr(p.defaultValue)}` :
                                        bindJsName(p))
              .join(', ');
      // Compile the body in tail position so tail calls through function
      // values become `__tc` thunks the trampoline can drive iteratively
      const bodyExpr =
          (compileExprTail(expr.body));

      return `__fnlit((${params}) => ${bodyExpr}, ${
          JSON.stringify(oscadSource(expr))})`;
    }
    case 'listComp': {
      return `...(${compileListComp(expr.generator)})`;
    }
    case 'dynCall': {
      const staticType = staticNonFunctionType(expr.callee);
      if (staticType) {
        const line = expr.loc?.start.line;
        const where = line ? ` at line ${line}` : '';
        console.warn(`Warning: Can't call function on ${staticType}${where}`);
        return 'undef';
      }
      const callee = compileExpr(expr.callee);
      const args =
          expr.args
              .map(
                  a => a.name ? `/* ${a.name} = */ ${compileExpr(a.value)}` :
                                compileExpr(a.value))
              .join(', ');
      // Invoking a function value goes through the trampoline driver
      return `__call(${callee}${args ? `, ${args}` : ''})`;
    }
    default:
      return `/* unsupported expr: ${(expr as Expr).kind}${
          locTag(expr as ASTNode)} */`;
  }
}

// Reconstruct the OpenSCAD source form of an expression, mirroring OpenSCAD's
// own Expression serialization
function oscadSource(expr: Expr): string {
  switch (expr.kind) {
    case 'number':
      return String(expr.value);
    case 'string':
      return '"' + expr.value.replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
    case 'boolean':
      return expr.value ? 'true' : 'false';
    case 'undef':
      return 'undef';
    case 'identifier':
      return expr.name;
    case 'vector':
      return '[' + expr.elements.map(oscadSource).join(', ') + ']';
    case 'range':
      return expr.step ?
          `[${oscadSource(expr.start)} : ${oscadSource(expr.step)} : ${
              oscadSource(expr.end)}]` :
          `[${oscadSource(expr.start)} : ${oscadSource(expr.end)}]`;
    case 'binary':
      return `(${oscadSource(expr.left)} ${expr.op} ${
          oscadSource(expr.right)})`;
    case 'unary':
      return `${expr.op}${oscadSource(expr.operand)}`;
    case 'ternary':
      return `(${oscadSource(expr.condition)} ? ${oscadSource(expr.ifTrue)} : ${
          oscadSource(expr.ifFalse)})`;
    case 'call':
      return `${expr.name}(${expr.args.map(oscadArgSource).join(', ')})`;
    case 'index':
      return `${oscadSource(expr.object)}[${oscadSource(expr.index)}]`;
    case 'member':
      return `${oscadSource(expr.object)}.${expr.property}`;
    case 'group':
      return oscadSource(expr.expr);
    case 'lambda':
      return `function(${expr.params.map(oscadParamSource).join(', ')}) ${
          oscadSource(expr.body)}`;
    case 'let':
      return `let(${
          expr.assignments.map(a => `${a.name} = ${oscadSource(a.value)}`)
              .join(', ')}) ${oscadSource(expr.body)}`;
    case 'dynCall':
      return `${oscadSource(expr.callee)}(${
          expr.args.map(oscadArgSource).join(', ')})`;
    default:
      return '';
  }
}
function oscadArgSource(a: Argument): string {
  return a.name ? `${a.name} = ${oscadSource(a.value)}` : oscadSource(a.value);
}
function oscadParamSource(p: import('./ast.js').Parameter): string {
  return p.defaultValue ? `${p.name} = ${oscadSource(p.defaultValue)}` : p.name;
}

function staticNonFunctionType(e: Expr): string|undefined {
  switch (e.kind) {
    case 'number':
      return 'number';
    case 'string':
      return 'string';
    case 'boolean':
      return 'bool';
    case 'undef':
      return 'undefined';
    case 'vector':
      return 'vector';
    case 'range':
      return 'range';
    case 'group':
      return staticNonFunctionType(e.expr);
    default:
      return undefined;
  }
}

// Compile an expression that is in tail position of a function-literal body
function compileExprTail(expr: Expr): string {
  switch (expr.kind) {
    case 'group':
      return compileExprTail(expr.expr);
    case 'ternary': {
      return `(__truthy(${compileExpr(expr.condition)}) ? ${
          compileExprTail(expr.ifTrue)} : ${compileExprTail(expr.ifFalse)})`;
    }
    case 'echo': {
      const eArgs =
          expr.args
              .map(
                  a => a.name ? `(${JSON.stringify(a.name + ' = ')} + __oecho(${
                                    compileExpr(a.value)}))` :
                                `__oecho(${compileExpr(a.value)})`)
              .join(', ');
      return `(__echo(${eArgs}), ${compileExprTail(expr.expr)})`;
    }
    case 'assert': {
      const condition = expr.args[0] ? compileExpr(expr.args[0].value) : 'true';
      const message =
          expr.args[1] ? compileExpr(expr.args[1].value) : '"Assertion failed"';
      return `(openscad_assert_fn(${condition}, ${message}), ${
          compileExprTail(expr.expr)})`;
    }
    case 'let': {
      const localAssignNames =
          expr.assignments.filter(a => !a.name.startsWith('$'))
              .map(bindJsName);
      return (() => {
        const bound: string[] = [];
        const vals = expr.assignments.map(a => {
          const selfRec = a.value.kind === 'lambda' && !a.name.startsWith('$');
          const suppress = selfRec ? [...bound, bindJsName(a)] : bound;
          const val =
              (compileExpr(a.value));
          if (!a.name.startsWith('$')) bound.push(bindJsName(a));
          return val;
        });
        let result =
            (compileExprTail(expr.body));
        for (let i = expr.assignments.length - 1; i >= 0; i--) {
          const a = expr.assignments[i]!;
          const name = bindJsName(a);
          const val = vals[i]!;
          if (a.name.startsWith('$')) {
            const t = svTarget(name);
            result = `(() => { const __save_${name}: any = ${t}; ${t} = ${
                val}; try { return ${result}; } finally { ${t} = __save_${
                name}; } })()`;
          } else if (a.value.kind === 'lambda') {
            result = `((${name} = ${val}) => (${result}))()`;
          } else {
            result = `((${name}) => (${result}))(${val})`;
          }
        }
        return result;
      })();
    }
    case 'call':
      return compileCallExpr(expr, true);
    case 'dynCall': {
      const staticType = staticNonFunctionType(expr.callee);
      if (staticType) return 'undef';
      const callee = compileExpr(expr.callee);
      const args =
          expr.args
              .map(
                  a => a.name ? `/* ${a.name} = */ ${compileExpr(a.value)}` :
                                compileExpr(a.value))
              .join(', ');
      return `__tc(${callee}${args ? `, ${args}` : ''})`;
    }
    default:
      return compileExpr(expr);
  }
}

function compileCallExpr(
    expr: {
      kind: 'call'; name: string; args: Argument[]; loc?: ASTNode['loc'];
      ref?: CallRef | undefined;
    },
    tail = false): string {
  const escaped = escapeName(expr.name);
  const isKnownFunction =
      BUILTIN_FUNCTIONS.has(expr.name) || signatures.has(sigKey(expr.name, 'fn'));

  // A call to a name that is not a builtin, a user/external function, a local
  // binding, or a variable that might hold a function literal is an unknown
  // function.
  const isSpecialVarCallee =
      expr.name.startsWith('$') && expr.name !== '$children';
  const callsValue = isLexicalVar(expr.ref?.value);
  const isCallableValue = callsValue || isSpecialVarCallee;
  if (!isKnownFunction && !isCallableValue) {
    const line = expr.loc?.start.line;
    const where = line ? ` at line ${line}` : '';
    // Experimental builtins are OpenSCAD functions that are disabled
    if (EXPERIMENTAL_BUILTIN_FUNCTIONS.has(expr.name)) {
      console.warn(`Warning: Experimental builtin function '${
          expr.name}' is not enabled${where}`);
    }
    console.warn(`Warning: Ignoring unknown function '${expr.name}'${where}`);
    // Lower to a runtime no-op that evaluates to undef
    return `__unknown_fn(${JSON.stringify(expr.name)})`;
  }

  // OpenSCAD resolves variables and functions separately, so if both share a
  // name, dispatch at runtime: call the variable only if it's a function,
  // otherwise use the named function
  const shadowingValue = !isSpecialVarCallee && callsValue;
  const dualDispatch = isKnownFunction && shadowingValue;

  // When the callee isn't a known function definition but resolves to a value
  const fnName = expr.ref?.fn?.jsName ?? `${escaped}_fn`;
  const valueName = expr.ref?.value?.jsName ?? escaped;
  const name = isSpecialVarCallee          ? svTarget(escaped) :
      (shadowingValue && !isKnownFunction) ? valueName :
                                             fnName;

  // A call through a function VALUE goes through the tail-call trampoline:
  // `__call` drives it from non-tail position, while a tail call returns a
  // `__tc` thunk so self-recursion through function values runs iteratively
  const isValueCall = name !== fnName;

  const sig = signatures.get(sigKey(expr.name, 'fn'));
  const dollarArgs = expr.args.filter(
      a => a.name && a.name.startsWith('$') &&
          !(sig && sig.params.includes(a.name)));
  const positionalArgs = dollarArgs.length === 0 ?
      expr.args :
      expr.args.filter(a => !dollarArgs.includes(a));

  const argList = compileArgList(sigKey(expr.name, 'fn'), positionalArgs);
  const call = dualDispatch ?
      (() => {
        // A call through a value has no declared signature to match against.
        const valueArgList = compileArgList(sigKey(expr.name, 'var'), positionalArgs);
        return `(typeof ${valueName} === "function" ? ${
            tail ? '__tc' : '__call'}(${valueName}${
            valueArgList ? `, ${valueArgList}` : ''}) : ${name}(${argList}))`;
      })() :
      isValueCall ?
      `${tail ? '__tc' : '__call'}(${name}${argList ? `, ${argList}` : ''})` :
                                                      `${name}(${argList})`;
  if (dollarArgs.length === 0) {
    return call;
  }

  const saves: string[] = [];
  const decls: string[] = [];
  const restores: string[] = [];
  // Duplicate special-var named args resolve last-wins
  const lastValue = new Map<string, typeof dollarArgs[number]['value']>();
  for (const arg of dollarArgs) lastValue.set(arg.name!, arg.value);
  for (const [argName, value] of lastValue) {
    const dn = escapeName(argName);
    const t = svTarget(dn);
    saves.push(`let __save_${dn}: any = ${t};`);
    decls.push(`${t} = ${compileExpr(value)};`);
    restores.push(`${t} = __save_${dn};`);
  }
  return `(() => { ${saves.join(' ')} ${decls.join(' ')} try { return ${
      call}; } finally { ${restores.join(' ')} } })()`;
}

// List comprehension
function compileListComp(gen: ListCompGenerator): string {
  switch (gen.kind) {
    case 'lcFor': {
      // Each variable's range sees the variables bound before it; the body sees
      // all of them and every binding shadows any renamed outer local of the
      // same name
      const bound: string[] = [];
      const ranges = gen.variables.map(v => {
        const parts = (v.range.kind === 'range' ?
                [
                  compileExpr(v.range.start),
                  v.range.step ? compileExpr(v.range.step) : '1',
                  compileExpr(v.range.end)
                ] :
                [compileExpr(v.range)]);
        bound.push(bindJsName(v));
        return parts;
      });
      let result = (compileListComp(gen.body));
      for (let i = gen.variables.length - 1; i >= 0; i--) {
        const v = gen.variables[i]!;
        const vName = bindJsName(v);
        if (v.range.kind === 'range') {
          const [start, step, end] = ranges[i]!;
          result = `(() => { const __r = []; const __start: any = ${
              start}, __step: any = ${step}, __end: any = ${
              end}; const __cnt: any = __rangeCount(__start, __step, __end); for (let __i = 0; __i < __cnt; __i++) { const ${
              vName}: any = __i === 0 ? __start : __start + __i * __step; __r.push(...(${
              result})); } return __r; })()`;
        } else {
          result = `__flat_map_iter(${ranges[i]![0]}, (${vName}) => ${result})`;
        }
      }
      return result;
    }
    case 'lcIf': {
      const cond = compileExpr(gen.condition);
      let ifTrue = compileListComp(gen.ifTrue);
      let ifFalse = gen.ifFalse ? compileListComp(gen.ifFalse) : '[]';
      // Both branches are now guaranteed to evaluate to an array.
      return `(__truthy(${cond}) ? ${ifTrue} : ${ifFalse})`;
    }
    case 'lcLet': {
      // Sequential let: each value sees the bindings established before it, and
      // every binding shadows any renamed outer local of the same name
      const bound: string[] = [];
      const vals = gen.assignments.map(a => {
        const val = (compileExpr(a.value));
        bound.push(bindJsName(a));
        return val;
      });
      let result = (compileListComp(gen.body));
      for (let i = gen.assignments.length - 1; i >= 0; i--) {
        const a = gen.assignments[i]!;
        result = `((${bindJsName(a)}) => (${result}))(${vals[i]})`;
      }
      return result;
    }
    case 'lcExpr': {
      const expr = compileExpr(gen.expr);
      if (expr.startsWith('...')) {
        return `[${expr}]`;
      }
      return `[${expr}]`;
    }
    case 'lcCFor': {
      // Init values evaluate in the outer scope; the loop names shadow any
      // renamed outer locals for the condition, updates and body
      const inits =
          gen.inits.map(a => `${bindJsName(a)} = ${compileExpr(a.value)}`)
              .join(', ');
      const loopNames = gen.inits.map(bindJsName);
      const [cond, updates, inner] = ([compileExpr(gen.condition),
               gen.updates
                   .map(a => `${bindJsName(a)} = ${compileExpr(a.value)}`)
                   .join(', '),
               compileListComp(gen.body),
      ]);
      // Abort the loop once its counter exceeds the limit
      const base = currentMainFilename ? path.basename(currentMainFilename) :
                                         '<unknown>';
      const line = gen.loc?.start.line ?? 0;
      const errMsg =
          JSON.stringify(`ERROR: For loop counter exceeded limit in file ${
              base}, line ${line}`);
      return `(() => { const __r = []; let __fc = 0; for (let ${
          inits}; __truthy(${cond}); ${updates}) { if (__fc++ >= ${
          MAX_FOR_ITERATIONS}) throw new Error(${errMsg}); __r.push(...(${
          inner})); } return __r; })()`;
    }
  }
}

// Argument lookup
function findArg(
    args: Argument[],
    name: string,
    positionalIndex?: number,
    ): Argument|undefined {
  const named = args.find((a) => a.name === name);
  if (named) return named;
  if (positionalIndex !== undefined && positionalIndex < args.length) {
    const a = args[positionalIndex]!;
    if (!a.name) return a;
  }
  return undefined;
}
