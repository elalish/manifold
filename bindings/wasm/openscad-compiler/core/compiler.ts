import {createCanvas, Image} from 'canvas';
import fs from 'fs';
import path from 'path';

import {type FunctionDeclStmt, type Parameter, someNode, walk} from './ast.js';
import type {Argument, ASTNode, BinaryExpr, BlockStmt, Comment, Expr, ForStmt, ForVariable, FunctionCallExpr, IfStmt, KindedNode, ListCompGenerator, ModuleCallStmt, Program, ScopeStmt, Statement,} from './ast.js';
import {bindLibrary, bindProgram, isLexicalVar, lookup, shadowsOuterVar,} from './binder.js';
import {assignPrettyNames, escapeName} from './naming.js';
import {getFontPath} from './resolver.js';
import type {Binding, BindOptions, BindResult, CallRef, CompiledLibrary, CompiledLibraryFile, CompileOptions, FontScan, FontTargets, FunctionDeclStmtType, LibraryClosure, LibraryManifest, ModuleDeclStmtType, Namespace, ProgramScan, ScanOptions, Scope,} from './types.js';

export const MANIFEST_VERSION = 2;

function builtinConstantDecls(): [string, string][] {
  const noArg = T('NO_ARG');
  return [
    ['PI', `let PI: number = ${RUNTIME_NS}.PI;`],
    ['INF', `let INF: number = ${RUNTIME_NS}.INF;`],
    ['NAN', `let NAN: number = ${RUNTIME_NS}.NAN;`],
    ['undef', `let undef: undefined = ${RUNTIME_NS}.undef;`],
    ['_EPSILON', `let _EPSILON: number = ${RUNTIME_NS}._EPSILON;`],
    [noArg, `let ${noArg}: symbol = Symbol.for("OPENSCAD_NO_ARG");`],
  ];
}

const RUNTIME_NS = 'rt';
const GEOMETRY_TYPE = 'InstanceType<typeof Manifold | typeof CrossSection>';

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

function fontCandidateNames(stmts: Statement[]): FontTargets {
  const names = new Set<string>(['text']);
  const decls = new Set<ModuleDeclStmtType>();
  const visit = (list: Statement[]): void => {
    for (const s of list) {
      if (s.kind === 'scope') {
        visit(s.statements);
      } else if (s.kind === 'moduleDecl') {
        names.add(s.name);
        decls.add(s);
      }
    }
  };
  visit(stmts);
  return {names, decls};
}

function isFontRelatedName(name: string): boolean {
  const lower = name.toLowerCase();
  return lower.includes('font') || lower.includes('style') ||
      lower.includes('family');
}

function resolveFontLiterals(
    font: FontScan, candidates: Set<string>): Set<string> {
  const textModules = new Set<string>(['text']);
  let changed = true;
  while (changed) {
    changed = false;
    for (const name of candidates) {
      if (textModules.has(name)) continue;
      const called = font.edges.get(name);
      if (!called) continue;
      for (const callee of called) {
        if (!textModules.has(callee)) continue;
        textModules.add(name);
        changed = true;
        break;
      }
    }
  }

  const literals = font.literals;
  for (const {module, exprs} of font.paramDefaults)
    if (textModules.has(module))
      for (const e of exprs) collectStringLiterals(e, literals);
  for (const {name, args} of font.calls)
    if (textModules.has(name))
      for (const e of resolveCallArgs(name, args).values())
        collectStringLiterals(e, literals);
  for (const {module, value} of font.scopedVars)
    if (textModules.has(module)) collectStringLiterals(value, literals);
  return literals;
}

// Scan FONTPATH and return the basenames of the .ttf/.otf files whose family
// and style match one of the program's font literals
function fontsMatchingLiterals(fontLiterals: Set<string>): string[] {
  const fontDir = getFontPath();
  if (!fontDir || !fs.existsSync(fontDir)) return [];

  const matched: string[] = [];
  try {
    const files = fs.readdirSync(fontDir);
    const cleanedLiterals =
        Array.from(fontLiterals)
            .map(lit => lit.toLowerCase().replace(/[^a-z0-9]/g, ''));

    for (const file of files) {
      const ext = path.extname(file).toLowerCase();
      if (ext !== '.ttf' && ext !== '.otf') continue;

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
      if (!familyMatched) continue;

      const styleMatched =
          cleanedStyle === 'regular' || cleanedLiterals.some(lit => {
            if (cleanedStyle === 'bolditalic') {
              return (lit.includes('bold') && lit.includes('italic')) ||
                  lit.includes('bolditalic');
            }
            return lit.includes(cleanedStyle);
          });
      if (styleMatched) matched.push(basename);
    }
  } catch (e) {
    console.warn('Warning: failed to read font directory for matching:', e);
  }
  return matched;
}

// Generic whole-program scan
function scanProgram(
    stmts: Statement[], options: ScanOptions = {}): ProgramScan {
  const openSlots = options.noArgSlots;
  const scan: ProgramScan = {
    refs: {modules: new Set(), functions: new Set(), variables: new Set()},
    unresolved: new Set(),
    moduleArgNames: new Set(),
    parentModulesInFunction: false,
    topLevelChildren: false,
    functionDefs: new Map(),
    divergenceCandidates: [],
    font: {
      edges: new Map(),
      literals: new Set(),
      paramDefaults: [],
      calls: [],
      scopedVars: [],
    },
  };

  let moduleDeclDepth = 0;  // children(): file scope only
  let functionDepth = 0;    // $parent_modules: inside a function only
  let declBodyDepth = 0;    // divergence: bodies run only when called
  let callChildDepth = 0;   // functionDefs: match forEachDeclaration's reach
  const callChildren = new Set<KindedNode>();
  const moduleStack: ModuleDeclStmtType[] = [];
  const fontTargets = options.fontCandidates;
  const fontNames = fontTargets?.names;

  const enter = (node: KindedNode): void => {
    if (callChildren.has(node)) callChildDepth++;
    switch (node.kind) {
      case 'identifier':
        if (!node.binding && node.name !== '$children')
          scan.unresolved.add(bindJsName(node));
        if (node.name === '$parent_modules' && functionDepth > 0)
          scan.parentModulesInFunction = true;
        if (node.name !== '$children' && !node.name.startsWith('$'))
          scan.refs.variables.add(node.name);
        break;
      case 'call':
        scan.refs.functions.add(node.name);
        if (node.name === 'children' && moduleDeclDepth === 0)
          scan.topLevelChildren = true;
        if (options.divergence && declBodyDepth === 0)
          scan.divergenceCandidates.push(node);
        if (openSlots) demoteNoArgSlots(node, openSlots);
        break;
      case 'moduleCall':
        if (node.name === 'children') {
          if (moduleDeclDepth === 0) scan.topLevelChildren = true;
        } else {
          scan.refs.modules.add(node.name);
        }
        if (!BUILTIN_MODULES.has(node.name))
          for (const arg of node.args)
            if (arg.name && !arg.name.startsWith('$'))
              scan.moduleArgNames.add(escapeName(arg.name));
        if (openSlots) demoteNoArgSlots(node, openSlots);
        if (node.child) callChildren.add(node.child);
        if (fontTargets && fontNames) {
          const caller = moduleStack[0];
          if (moduleStack.length === 1 && fontTargets.decls.has(caller!)) {
            let called = scan.font.edges.get(caller!.name);
            if (!called) scan.font.edges.set(caller!.name, called = new Set());
            called.add(node.name);
          }
          if (fontNames.has(node.name))
            scan.font.calls.push({name: node.name, args: node.args});
        }
        break;
      case 'variableDecl':
        if (fontNames) {
          if (isFontRelatedName(node.name)) {
            collectStringLiterals(node.value, scan.font.literals);
          } else {
            const owner = moduleStack[moduleStack.length - 1];
            if (owner !== undefined && fontNames.has(owner.name))
              scan.font.scopedVars.push(
                  {module: owner.name, value: node.value});
          }
        }
        break;
      case 'functionDecl':
        if (callChildDepth === 0) scan.functionDefs.set(node.name, node);
        functionDepth++;
        declBodyDepth++;
        break;
      case 'moduleDecl':
        moduleDeclDepth++;
        declBodyDepth++;
        if (fontNames) {
          moduleStack.push(node);
          if (fontNames.has(node.name)) {
            const exprs = node.params.map(p => p.defaultValue)
                              .filter((e): e is Expr => !!e);
            if (exprs.length)
              scan.font.paramDefaults.push({module: node.name, exprs});
          }
        }
        break;
      case 'lambda':
        functionDepth++;
        break;
    }
  };

  const exit = (node: KindedNode): void => {
    if (callChildren.has(node)) callChildDepth--;
    switch (node.kind) {
      case 'functionDecl':
        functionDepth--;
        declBodyDepth--;
        break;
      case 'moduleDecl':
        moduleDeclDepth--;
        declBodyDepth--;
        if (fontNames) moduleStack.pop();
        break;
      case 'lambda':
        functionDepth--;
        break;
    }
  };

  for (const s of stmts) walk(s, enter, exit);
  return scan;
}

function locTag(node: ASTNode): string {
  if (!node.loc) return '';
  const s = node.loc.start;
  return ` @${s.line}:${s.column}`;
}

function leadingCommentLines(node: ASTNode|undefined, indent = ''): string[] {
  return (node?.leadingComments ?? [])
      .flatMap(
          comment =>
              comment.value.split(/\r?\n/).map(line => `${indent}${line}`));
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
  noArg?: boolean[];
}
const signatures = new Map<string, Signature>();


function paramUsesNoArg(p: Parameter): boolean {
  return !!p.defaultValue && !p.name.startsWith('$') &&
      !nodeReferencesIdentifier(p.defaultValue, p.name);
}

// Module/function declarations of the program being compiled
interface LocalDecl {
  count: number;
  params: Parameter[];
}
const localDecls = new Map<string, LocalDecl>();

const noArgDemotions = new Map<string, boolean[]>();

function slotUsesNoArg(key: string, p: Parameter, i: number): boolean {
  return paramUsesNoArg(p) && !noArgDemotions.get(key)?.[i];
}

// Slots that could still use the __NO_ARG prologue, keyed by signature
function openNoArgSlots(): Map<string, boolean[]> {
  const open = new Map<string, boolean[]>();
  for (const [key, decl] of localDecls) {
    const sig = signatures.get(key);
    if (decl.count !== 1 || sig?.params.length !== decl.params.length) continue;
    // Duplicate parameter names shift slots between the signature and the
    // deduplicated list the declaration is emitted from
    if (new Set(decl.params.map(p => p.name)).size !== decl.params.length)
      continue;
    const slots = decl.params.map(
        (p, i) => paramUsesNoArg(p) &&
            // A default reading a parameter declared later still sees that
            // parameter in the prologue, but would hit the temporal dead zone
            // as a JS default
            !decl.params.slice(i + 1).some(
                q => nodeReferencesIdentifier(p.defaultValue, q.name)));
    if (slots.some(Boolean)) open.set(key, slots);
  }
  return open;
}

// One call site's effect on the open slots of the signature it targets
function demoteNoArgSlots(
    node: FunctionCallExpr|ModuleCallStmt, open: Map<string, boolean[]>): void {
  const key = `${node.kind === 'call' ? 'fn' : 'mod'}:${node.name}`
  const slots = open.get(key);
  if (!slots) return;
  const supplied = resolveArgsToParams(node.args, localDecls.get(key)!.params);
  supplied.forEach((arg, i) => {
    if (arg && !isDefinitelyDefined(arg, new Set())) slots[i] = false;
  });
}

// Variable kinds whose every write is visible in the program being compiled
const ANALYZABLE_VAR_KINDS =
    new Set<string>(['global', 'filePrivate', 'local', 'let']);

// True when every declaration writing `b` assigns a value satisfying `pred`.
// `seen` breaks reference cycles (`a = b; b = a;`) conservatively
function everyDeclValue(
    b: Binding|null|undefined, seen: Set<Binding>,
    pred: (e: Expr, seen: Set<Binding>) => boolean): boolean {
  if (!b || seen.has(b) || !ANALYZABLE_VAR_KINDS.has(b.kind)) return false;
  if (b.decls.length === 0) return false;
  seen.add(b);
  return b.decls.every(d => {
    // Anything but a plain assignment (a parameter, a loop variable) has no
    // single value expression to inspect
    const value = (d as {value?: Expr}).value;
    return !!value && pred(value, seen);
  });
}

// Expressions that compile to a value which is never `undefined`. Conservative:
// anything not listed here is assumed to possibly be `undef`
function isDefinitelyDefined(e: Expr, seen: Set<Binding>): boolean {
  switch (e.kind) {
    case 'number':
    case 'string':
    case 'boolean':
    case 'vector':  // an array literal, whatever its elements hold
    case 'range':   // range always builds an array
    case 'lambda':
      return true;
    case 'group':
      return isDefinitelyDefined(e.expr, seen);
    case 'ternary':
      return isDefinitelyDefined(e.ifTrue, seen) &&
          isDefinitelyDefined(e.ifFalse, seen);
    case 'unary':
      return e.op === '!' || isDefinitelyNumber(e, seen);
    case 'binary':
      return e.op === '==' || e.op === '!=' || isDefinitelyNumber(e, seen);
    case 'identifier':
      return everyDeclValue(e.binding, seen, isDefinitelyDefined);
    default:
      return false;
  }
}

function isDefinitelyNumber(e: Expr, seen: Set<Binding>): boolean {
  switch (e.kind) {
    case 'number':
      return true;
    case 'group':
      return isDefinitelyNumber(e.expr, seen);
    case 'unary':
      return (e.op === '-' || e.op === '+') &&
          isDefinitelyNumber(e.operand, seen);
    case 'binary':
      return ['+', '-', '*', '/', '%', '^'].includes(e.op) &&
          isDefinitelyNumber(e.left, seen) && isDefinitelyNumber(e.right, seen);
    case 'identifier':
      return everyDeclValue(e.binding, seen, isDefinitelyNumber);
    default:
      return false;
  }
}

let moduleDeclRegistry = new Map<string, ModuleDeclStmtType>();

let tailTempCounter = 0;

// Track unique fonts encountered during compilation for base64 generation.
let encounteredFonts = new Set<string>();

// OpenSCAD's default typeface, used when a text() call names none
const DEFAULT_FONT_SPEC = 'Liberation Sans:style=Regular';

// Generated surface data modules this compilation has to import, keyed by
// the filename literal from the source
let encounteredSurfaceData = new Map<string, SurfaceAsset>();

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

// mapping of lowercased filename to the name as it actually sits on disk, per
// font directory
const fontDirListings = new Map<string, Map<string, string>>();

function fontDirListing(fontDir: string): Map<string, string> {
  let listing = fontDirListings.get(fontDir);
  if (listing) return listing;

  listing = new Map<string, string>();
  try {
    for (const file of fs.readdirSync(fontDir))
      listing.set(file.toLowerCase(), file);
  } catch (e) {
    console.warn(`Warning: failed to read font directory "${fontDir}":`, e);
  }
  fontDirListings.set(fontDir, listing);
  return listing;
}

// Resolves `basename` to a font file, ignoring case. Returns the name as it is
// on disk
function resolveFontFile(fontDir: string, basename: string):
    {filePath: string; basename: string; mimeType: string}|undefined {
  const listing = fontDirListing(fontDir);
  const candidates: ReadonlyArray<[string, string]> =
      [['.ttf', 'font/ttf'], ['.otf', 'font/otf']];

  for (const [ext, mimeType] of candidates) {
    const file = listing.get(`${basename}${ext}`.toLowerCase());
    if (file) {
      return {
        filePath: path.join(fontDir, file),
        basename: path.basename(file, path.extname(file)),
        mimeType,
      };
    }
  }
  return undefined;
}

function generateFontBase64(fontSpec: string, compilerDir: string): string|
    undefined {
  const fontDir = getFontPath();
  if (!fontDir) {
    console.warn(
        `Warning: FONTPATH environment variable not set — cannot load font "${
            fontSpec}". Text will render as empty cross-section.`);
    return undefined;
  }

  const canonical = fontSpecToFilename(fontSpec);
  const resolved =
      resolveFontFile(fontDir, fontSpec) ?? resolveFontFile(fontDir, canonical);

  if (!resolved) {
    console.warn(`Warning: No "${fontSpec}" or "${canonical}" .ttf/.otf in "${
        fontDir}" — text using "${
        fontSpec}" will render as empty cross-section.`);
    return undefined;
  }

  const {filePath: fontFilePath, basename: filename, mimeType} = resolved;
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

const BUILTIN_SIGNATURES: Record<string, string[]> = {
  'cube': ['size', 'center'],
  'cylinder':
      ['h', 'r', 'r1', 'r2', 'd', 'd1', 'd2', 'center', '$fn', '$fa', '$fs'],
  'sphere': ['r', 'd', '$fn', '$fa', '$fs'],
  'square': ['size', 'center'],
  'circle': ['r', 'd', '$fn', '$fa', '$fs'],
  'polygon': ['points', 'paths', 'convexity'],
  'polyhedron': ['points', 'faces', 'convexity'],
  'linear_extrude': [
    'height', 'v', 'scale', 'center', 'twist', 'slices', 'segments',
    'convexity', 'h', '$fn', '$fa', '$fs', '$fe'
  ],
  'rotate_extrude': ['angle', 'convexity', '$fn'],
  'text': [
    'text', 'size', 'font', 'halign', 'valign', 'spacing', 'direction',
    'language', 'script', '$fn'
  ],
  'surface': ['file', 'center', 'invert', 'convexity'],
  'import': ['file', 'convexity', 'layer'],
  'projection': ['cut'],
  'translate': ['v'],
  'rotate': ['a', 'v'],
  'scale': ['v'],
  'resize': ['newsize', 'auto'],
  'mirror': ['v'],
  'multmatrix': ['m'],
  'color': ['c', 'alpha'],
  'offset': ['r', 'delta', 'chamfer'],
};

// Visits every function/module declaration reachable through declarative
// statement positions
function forEachDeclaration(
    stmts: Statement[],
    visit: (decl: FunctionDeclStmtType|ModuleDeclStmtType) => void): void {
  for (const stmt of stmts) {
    if (stmt.kind === 'functionDecl' || stmt.kind === 'moduleDecl') {
      visit(stmt);
      if (stmt.kind === 'moduleDecl' && stmt.body.kind === 'block')
        forEachDeclaration(stmt.body.statements, visit);
    } else if (stmt.kind === 'block' || stmt.kind === 'scope') {
      forEachDeclaration(stmt.statements, visit);
    } else if (stmt.kind === 'if') {
      if (stmt.thenBody.kind === 'block')
        forEachDeclaration(stmt.thenBody.statements, visit);
      if (stmt.elseBody && stmt.elseBody.kind === 'block')
        forEachDeclaration(stmt.elseBody.statements, visit);
    }
  }
}

function recordSignature(decl: FunctionDeclStmtType|ModuleDeclStmtType): void {
  const name = `${decl.kind === 'functionDecl' ? 'fn' : 'mod'}:${decl.name}`
  signatures.set(name, {
    params: decl.params.map(p => p.name),
    defaults: decl.params.map(p => p.defaultValue),
    noArg: decl.params.map(paramUsesNoArg),
  });
  const seenDecl = localDecls.get(name);
  if (seenDecl) {
    seenDecl.count++;
    seenDecl.params = decl.params;
  } else {
    localDecls.set(name, {count: 1, params: decl.params});
  }
}

// Collects module declarations and records signatures
function collectDeclarations(stmts: Statement[]):
    Map<string, ModuleDeclStmtType> {
  const modules = new Map<string, ModuleDeclStmtType>();
  forEachDeclaration(stmts, decl => {
    recordSignature(decl);
    if (decl.kind === 'moduleDecl') modules.set(decl.name, decl);
  });
  return modules;
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
  // `undef` is kept as is and doesn't reapply the default. A demoted slot took
  // a JS default instead, which `undefined` triggers on its own
  const demoted = noArgDemotions.get(key);
  const fillFor = (i: number) =>
      (sig.noArg?.[i] && !demoted?.[i]) ? T('NO_ARG') : 'undefined';
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

  return compiledArgs.concat(extraArgs).join(', ');
}

function emitArgs(...args: string[]): string {
  let end = args.length;
  while (end > 0 && args[end - 1] === 'undefined') end--;
  return args.slice(0, end).join(', ');
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

// Builtins that always return a number when called with arity
const NUMERIC_BUILTINS = new Map<string, number>([
  ['sin', 1],
  ['cos', 1],
  ['tan', 1],
  ['asin', 1],
  ['acos', 1],
  ['atan', 1],
  ['atan2', 2],
  ['abs', 1],
  ['sign', 1],
  ['floor', 1],
  ['ceil', 1],
  ['round', 1],
  ['sqrt', 1],
  ['exp', 1],
  ['ln', 1],
  ['log', 1],
  ['pow', 2],
]);


const RUNTIME_SYMBOLS: string[] = [
  'Manifold',
  'CrossSection',
  'wasm',
  'cube',
  'square',
  'sphere',
  'cylinder',
  'circle',
  'radius',
  'rotate',
  'polygon',
  'polyhedron',
  'translate',
  'scale',
  'mirror',
  'resize',
  'is_undef',
  'is_bool',
  'is_num',
  'is_string',
  'is_list',
  'is_function',
  'unknown_fn',
  'sin',
  'cos',
  'tan',
  'asin',
  'acos',
  'atan',
  'atan2',
  'abs',
  'sign',
  'floor',
  'ceil',
  'round',
  'sqrt',
  'exp',
  'ln',
  'log',
  'pow',
  'min',
  'max',
  'norm',
  'cross',
  'len',
  'str',
  'chr',
  'ord',
  'concat',
  'search',
  'lookup',
  'rands',
  'assert',
  'truthy',
  'eq',
  'lt',
  'gt',
  'le',
  'ge',
  'add',
  'sub',
  'mul',
  'div',
  'mod',
  'band',
  'bor',
  'shl',
  'shr',
  'bnot',
  'neg',
  'pos',
  'index',
  'version',
  'version_num',
  'ctx',
  'withSpecials',
  'children_stack',
  'with_children',
  'pick_children',
  'parent_module',
  'is_finite_matrix4',
  'to_manifold_mat4',
  'transform',
  'identity4',
  'offset',
  'projection',
  'color',
  'each',
  'flat_map_iter',
  'range',
  'rangeCount',
  'is2D',
  'union',
  'difference',
  'intersection',
  'hull',
  'minkowski',
  'rootMod',
  'applyRoot',
  'linear_extrude',
  'rotate_extrude',
  'text',
  'parse_color_for_scope',
  'surface',
  'echo',
  'oecho',
  'fnlit',
  'font_registry',
  'tc',
  'call',
];

const RUNTIME_BUILTIN_MODULES = new Set([
  'cube',       'square',       'sphere',         'cylinder',       'circle',
  'polygon',    'polyhedron',   'text',           'surface',        'translate',
  'rotate',     'scale',        'mirror',         'resize',         'offset',
  'projection', 'color',        'linear_extrude', 'rotate_extrude', 'union',
  'difference', 'intersection', 'hull',           'minkowski',      'echo',
]);
const RUNTIME_BUILTIN_FUNCTIONS =
    new Set<string>([...BUILTIN_FUNCTIONS, 'assert']);

// Local name for each runtime export
let RT: Record<string, string> =
    Object.fromEntries(RUNTIME_SYMBOLS.map(s => [s, s]));

// Gives each runtime export a local name - uses its own name first, if it
// clashes with a user declaration, import it as `runtime_...` instead
function resolveRuntimeLocals(blocked: ReadonlySet<string>): void {
  const claimed = new Set<string>();
  const locals: Record<string, string> = {};
  for (const name of RUNTIME_SYMBOLS) {
    let local = name;
    for (let n = 1; blocked.has(local) || claimed.has(local); n++)
      local = n === 1 ? `runtime_${name}` : `runtime_${name}_${n}`;
    claimed.add(local);
    locals[name] = local;
  }
  RT = locals;
}

// Names introduced by the emitter
let unitTakenNames = new Set<string>();
const tempNames = new Map<string, string>();

function resetTempNames(taken: Iterable<string>): void {
  unitTakenNames = new Set(taken);
  tempNames.clear();
}

function T(base: string): string {
  const memo = tempNames.get(base);
  if (memo !== undefined) return memo;
  let name = base;
  for (let n = 1; unitTakenNames.has(name); n++) name = `${base}_${n}`;
  tempNames.set(base, name);
  // so two different bases cannot converge on one name either
  unitTakenNames.add(name);
  return name;
}

function namesInUse(bind: BindResult, unresolved: Set<string>): string[] {
  const used = [...reservedNames()];
  for (const b of bind.bindings) used.push(b.jsName);
  used.push(...unresolved);
  return used;
}

function builtinSymbolNames(): Map<string, string> {
  const byKey = new Map<string, string>();
  for (const name of RUNTIME_BUILTIN_FUNCTIONS)
    if (RT[name]) byKey.set(`fn:${name}`, RT[name]!);
  for (const name of RUNTIME_BUILTIN_MODULES)
    if (RT[name]) byKey.set(`mod:${name}`, RT[name]!);
  return byKey;
}

// Names emitted as declarations and unavailable to runtime imports. This
// includes declared bindings, unresolved reads, and non-$ named arguments used
// as slots
function namesBlockingRuntimeLocals(
    bind: BindResult, scan: ProgramScan,
    externalSymbols?: Map<string, string>): Set<string> {
  const blocked = new Set<string>();
  for (const b of bind.bindings) {
    // Runtime imports are module-scoped, so only module-scope declarations can
    // clash
    if (b.kind !== 'global' && b.kind !== 'filePrivate' &&
        b.kind !== 'external')
      continue;
    blocked.add(escapeName(b.name));
    if (b.kind === 'external') {
      blocked.add(legacyJsName(b.name, b.ns));
      const sym = externalSymbols?.get(`${b.ns}:${b.name}`);
      if (sym) blocked.add(sym);
    }
  }
  if (externalSymbols)
    for (const sym of externalSymbols.values()) blocked.add(sym);
  for (const name of scan.unresolved) blocked.add(name);
  for (const name of scan.moduleArgNames) blocked.add(name);
  return blocked;
}

// Emit the target for a variable. $-prefixed vars live in ctx; others are
// regular lexical bindings. Used for reads/writes that may target special vars
function svTarget(name: string): string {
  return name.startsWith('$') ? `${RT.ctx}.${name}` : name;
}

// Identifiers mentioned in code
function mentionedIdentifiers(body: string): Set<string> {
  return new Set(body.match(/(?<![\w$])(?<!(?<!\.)\.)[A-Za-z_$][\w$]*/g) ?? []);
}

// The OpenSCAD constant declarations this body needs, in their fixed order
function builtinConstantsFor(body: string): string {
  const mentioned = mentionedIdentifiers(body);
  return builtinConstantDecls()
      .filter(([name]) => mentioned.has(name))
      .map(([, decl]) => `${decl}\n`)
      .join('');
}

function buildRuntimeImport(runtimePath: string, body: string): string {
  const mentioned = mentionedIdentifiers(body);
  const entries: string[] = [];
  for (const exportName of RUNTIME_SYMBOLS) {
    const local = RT[exportName]!;
    if (mentioned.has(local))
      entries.push(
          local === exportName ? exportName : `${exportName}: ${local}`);
  }
  return (
      `import * as ${RUNTIME_NS} from "${runtimePath}";\n` +
      `const { ${entries.join(', ')} } = ${RUNTIME_NS};\n`);
}

// Names defined in separately compiled libraries and imported, not inlined.
// Loaded from library manifests at compile start so their calls are not treated
// as unknown modules
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
    ...RUNTIME_SYMBOLS.map(s => RT[s]!),
    ...BUILTIN_VAR_CONSTANTS,
    'Manifold',
    'CrossSection',
    'wasm',
    RUNTIME_NS,
    'viewport',
    'children',
    '$children',
    'result',
    'background',
  ];
}

const currentBindOptions: BindOptions = {
  builtinFunctions: BUILTIN_FUNCTIONS,
  builtinModules: BUILTIN_MODULES,
  builtinConstants: BUILTIN_VAR_CONSTANTS,
  externalFunctions: externalFunctionNames,
  externalModules: externalModuleNames,
  externalVariables: externalVariableNames,
}

// Keyword for top-level variable declarations. Use let normally, but var in
// libraries to avoid TDZ issues with circular imports. var hoists to
// undefined, matching OpenSCAD's default undef
let globalVarDeclKeyword = 'let';

// Track $ variables that need module-level declarations for dynamic scoping
let dynamicScopeVars: Set<string> = new Set();

let parentModulesReadInFunction = false;



function collectStringLiterals(node: KindedNode, literals: Set<string>): void {
  walk(node, n => {
    if (n.kind === 'string') literals.add(n.value);
  });
}


function resolveCallArgs(
    moduleCallName: string, callArgs: Argument[]): Map<string, Expr> {
  const resolved = new Map<string, Expr>();
  const sig = signatures.get(`mod:${moduleCallName}`);
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

function collectSlots(
    list: Statement[], slotOrder: string[], slotExpr: Map<string, Expr>): void {
  for (const s of list) {
    if (s.kind === 'block') {
      collectSlots(s.statements, slotOrder, slotExpr);
      continue;
    }
    if (s.kind !== 'variableDecl') continue;
    if (s.name.startsWith('$') || PRE_DECLARED_VARS.has(s.name)) continue;
    const n = declJsName(s, 'var');
    if (!slotExpr.has(n)) slotOrder.push(n);
    slotExpr.set(n, s.value);
  }
};

function processStmt(
    stmt: Statement, scopeUnits: string[],
    declMap:
        Map<string,
            {
              stmt: Statement;
              code: string
            }>,
    declOrder: string[], lastGeoFilename: string, geometryLines: string[],
    predeclared: Set<string>,
    moduleDeclRegistry: Map<string, ModuleDeclStmtType>,
    signatures: Map<string, Signature>): void {
  if (stmt.kind === 'empty') return;
  if (stmt.filename) currentSourceFilename = stmt.filename;
  // `{}` doesn't create a new scope - merge its assignments into
  // the enclosing scope
  if (stmt.kind === 'block') {
    for (const s of stmt.statements)
      processStmt(
          s, scopeUnits, declMap, declOrder, lastGeoFilename, geometryLines,
          predeclared, moduleDeclRegistry, signatures);
    return;
  }
  // A `use`d file is a scope: its variables stay inside
  // and only its modules and functions are published, as forwarders so they
  // dedupe against the consumer's own declarations of the same name
  if (stmt.kind === 'scope') {
    const unit = `${T(`scope${scopeUnits.length}`)}`;
    const {code, exports} = compileUsedFileScope(stmt, unit);
    scopeUnits.push(code);
    for (const ex of exports) {
      const key = `fn:${ex}`;
      if (!declMap.has(key)) declOrder.push(key);
      declMap.set(key, {
        stmt,
        code: `function ${ex}(...${T('args')}: any[]): any { return ${unit}.${
            ex}(...${T('args')}); }`,
      });
    }
    return;
  }
  if (isDecl(stmt)) {
    const key = declKey(stmt);
    if (!declMap.has(key)) declOrder.push(key);
    // A variable only needs the bare-assignment form when its declaration is
    // hoisted to undef ahead of the slots
    const hoisted = stmt.kind !== 'variableDecl' ||
        predeclared.has(declJsName(stmt, 'var'));
    declMap.set(
        key, {stmt, code: compileDeclaration(stmt, {assignmentOnly: hoisted})});
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
            geometryLines, stmt, `${T('background_items')}.push(${geo});`);
      } else if (
          stmt.kind === 'moduleCall' && !stmt.modifier &&
          isModuleCallBackgroundOnly(stmt, moduleDeclRegistry)) {
        pushCommentedLine(
            geometryLines, stmt, `${T('background_items')}.push(${geo});`);
      } else {
        pushCommentedLine(
            geometryLines, stmt, `${T('result_items')}.push(${geo});`);
      }
    }
  }
};

// Main entry
export function compile(program: Program, options?: CompileOptions): string {
  currentRuntimePath = options?.runtimePath ?? './runtime/runtime.js';
  currentMainFilename = program.filename ?? '';
  currentSourceFilename = currentMainFilename;
  dynamicScopeVars = new Set();
  encounteredFonts = new Set();
  encounteredSurfaceData.clear();
  externalModuleNames.clear();
  externalFunctionNames.clear();
  externalVariableNames.clear();
  tailTempCounter = 0;
  signatures.clear();
  localDecls.clear();
  noArgDemotions.clear();

  // Record the builtin signatures
  for (const [k, v] of Object.entries(BUILTIN_SIGNATURES)) {
    signatures.set(
        `mod:${k}`, {params: v, defaults: new Array(v.length).fill(undefined)});
  }

  // Record the external library signatures
  const externalLibraries = options?.externalLibraries ?? [];
  const externalSymbols = new Map<string, string>();
  for (const lib of externalLibraries) {
    for (const [sym, params] of Object.entries(lib.manifest.signatures)) {
      const noArg = lib.manifest.signatureNoArg?.[sym];
      signatures.set(sym, {
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

    // record external symbols
    const syms = lib.manifest.symbols;
    if (!syms) continue;
    for (const [n, sym] of Object.entries(syms.modules))
      externalSymbols.set(`mod:${n}`, sym);
    for (const [n, sym] of Object.entries(syms.functions))
      externalSymbols.set(`fn:${n}`, sym);
    for (const [n, sym] of Object.entries(syms.variables))
      externalSymbols.set(`var:${n}`, sym);
  }
  moduleDeclRegistry = collectDeclarations(program.statements);
  const fontCandidates = fontCandidateNames(program.statements);

  bindResult = bindProgram(program, currentBindOptions);

  const openSlots = openNoArgSlots();

  const scan = scanProgram(
      program.statements,
      {noArgSlots: openSlots, divergence: true, fontCandidates});
  for (const [key, slots] of openSlots) {
    if (slots.some(Boolean)) noArgDemotions.set(key, slots);
  }
  parentModulesReadInFunction = scan.parentModulesInFunction;

  resolveRuntimeLocals(
      namesBlockingRuntimeLocals(bindResult, scan, externalSymbols));
  assignPrettyNames(bindResult, {
    reserved: reservedNames(),
    externalSymbols,
    builtinSymbols: builtinSymbolNames(),
  });

  resetTempNames(namesInUse(bindResult, scan.unresolved));
  currentScope = bindResult.global;

  // Reject top-level constant-argument calls to non-tail recursive functions
  userFunctionDefs = scan.functionDefs;
  reportDivergentCalls(scan.divergenceCandidates);


  // Library `text` calls bypass face registration, so register the default face
  // upfront
  const programRefs = scan.refs;
  if (programRefs.modules.has('text') || programRefs.functions.has('text'))
    encounteredFonts.add(DEFAULT_FONT_SPEC);

  // Gather all font-related string literals from the program
  const fontLiterals = resolveFontLiterals(scan.font, fontCandidates.names);

  for (const name of fontsMatchingLiterals(fontLiterals)) {
    encounteredFonts.add(name);
  }

  const slotOrder: string[] = [];
  const slotExpr = new Map<string, Expr>();

  collectSlots(program.statements, slotOrder, slotExpr);
  const predeclared = namesNeedingPredeclaration(
      slotOrder.map(n => ({name: n, expr: slotExpr.get(n)!})));

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

  for (const stmt of program.statements) {
    processStmt(
        stmt, scopeUnits, declMap, declOrder, lastGeoFilename, geometryLines,
        predeclared, moduleDeclRegistry, signatures);
  }

  // Only variables readable by an earlier slot are hoisted to undef; the rest
  // share one typed declaration at their slot
  const hoistNames: string[] = [];
  const seenHoist = new Set<string>();
  for (const k of declOrder) {
    const e = declMap.get(k)!;
    if (e.stmt.kind !== 'variableDecl') continue;
    const nm = e.stmt.name;
    if (nm.startsWith('$') || PRE_DECLARED_VARS.has(nm)) continue;
    const en = escapeName(nm);
    if (seenHoist.has(en) || !predeclared.has(declJsName(e.stmt, 'var')))
      continue;
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

  let output = '';

  // Inject imports for names referenced from separately compiled external
  // libraries (resolved per kind against each library's manifest exports)
  if (externalLibraries.length > 0) {
    const refs = scan.refs;
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
    // ctx.$slop) run before the consumer body, matching include semantics
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
    const varName = `${T(`font_${sanitized.replace(/-/g, '_')}`)}`;
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
  if (resolvedFonts.size > 0) {
    output += `Object.assign(${RT.font_registry}, {\n`;
    const seenSanitized = new Set<string>();
    for (const [fontFamily, sanitized] of resolvedFonts) {
      if (seenSanitized.has(sanitized)) continue;
      seenSanitized.add(sanitized);
      const varName = `${T(`font_${sanitized.replace(/-/g, '_')}`)}`;
      output += `  ${JSON.stringify(sanitized)}: ${varName},\n`;
    }
    output += `});\n\n`;
  }

  const preamble = output;
  output = '';

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

  const referenced = scan.unresolved;
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
    T('NO_ARG'),
  ]);

  const undefinedNames =
      [...referenced]
          .filter(
              n => !n.startsWith('__') && !n.startsWith('$') &&
                  !moduleLevelDeclared.has(n))
          .sort();
  // Unbound OpenSCAD names read as undef, and a top-level initializer may
  // reference one before its later let declaration
  for (const name of undefinedNames) {
    output += `let ${name}: any = undefined;\n`;
  }

  if (declarations.length) {
    output += (output ? '\n' : '') + declarations.join('\n') + '\n\n';
  }

  // children() used outside a module's scope - warns and yields nothing. The
  // children stack is empty at top level, so this resolves to empty geometry
  if (scan.topLevelChildren) {
    output += `function children(i?: any): any { const ${T('c')}: any = ${
        RT.children_stack}.length > 0 ? ${RT.children_stack}[${
        RT.children_stack}.length - 1] : { fn: undefined, count: 0 }; return ${
        T('c')}.fn ? ${T('c')}.fn(i) : Manifold.union([]); }\n`;
  }

  if (geometryLines.length === 0) {
    output += `export const result = Manifold.union([]);\n`;
  } else {
    output += `const ${T('result_items')}: ${GEOMETRY_TYPE}[] = [];\n`;
    output += `const ${T('background_items')}: ${GEOMETRY_TYPE}[] = [];\n`;
    output += `${geometryLines.join('\n')}\n`;
    output += `export const result = ${RT.union}(${RT.applyRoot}(${
        T('result_items')}));\n`;
    output += `export const background = ${RT.union}(${RT.applyRoot}(${
        T('background_items')}, true));\n`;
  }
  output += `export const viewport = { vpr: ${RT.ctx}.$vpr, vpt: ${
      RT.ctx}.$vpt, vpd: ${RT.ctx}.$vpd, vpf: ${RT.ctx}.$vpf };\n`;

  const constants = builtinConstantsFor(preamble + output);

  const rest = output.replace(/^\n+/, '');
  const full = preamble + constants + (constants ? '\n' : '') + rest;
  return buildRuntimeImport(
             options?.runtimePath ?? './runtime/runtime.js', full) +
      full;
}


// Separate library compilation
type LibDeclKind = 'module'|'function'|'variable';

// The three declaration kinds and their library manifest mappings: `ns` is the
// declared namespace, while `kind` and `plural` are the singular/plural
// manifest keys
interface DeclInfo {
  kind: LibDeclKind;
  ns: Namespace;
  plural: 'modules'|'functions'|'variables';
}

const DECL_INFO = {
  variableDecl: {kind: 'variable', ns: 'var', plural: 'variables'},
  moduleDecl: {kind: 'module', ns: 'mod', plural: 'modules'},
  functionDecl: {kind: 'function', ns: 'fn', plural: 'functions'},
} as const satisfies Record<string, DeclInfo>;

type DeclStmt = Extract<Statement, {kind: keyof typeof DECL_INFO}>;

function isDecl(stmt: Statement): stmt is DeclStmt {
  return stmt.kind in DECL_INFO;
}

function declKey(stmt: DeclStmt): string {
  const {ns} = DECL_INFO[stmt.kind];
  return `${ns === 'var' ? 'var' : 'fn'}:${declJsName(stmt, ns)}`;
}

// The library-manifest view of a declaration, or undefined for a statement that
// isn't one. Special variables are dynamically scoped and never exported
function declKindAndName(stmt: Statement): DeclInfo&{name: string}|undefined {
  if (!(isDecl(stmt))) return undefined;
  if (stmt.kind === 'variableDecl' && stmt.name.startsWith('$'))
    return undefined;
  return {...DECL_INFO[stmt.kind], name: stmt.name};
}

const LIB_BUILTIN_CONSTS = new Set(BUILTIN_VAR_CONSTANTS);

export function compileLibrary(
    closure: LibraryClosure,
    opts: {runtimeVersion: string; runtimePathFor: (outRel: string) => string},
    ): CompiledLibrary {
  const sourceRels = [...closure.files.keys()].sort();
  const outRelOf = (sourceRel: string) => sourceRel.replace(/\.scad$/i, '.ts');

  globalVarDeclKeyword = 'var';

  signatures.clear();
  localDecls.clear();
  noArgDemotions.clear();

  for (const [k, v] of Object.entries(BUILTIN_SIGNATURES)) {
    signatures.set(
        `mod:${k}`, {params: v, defaults: new Array(v.length).fill(undefined)});
  }
  externalModuleNames.clear();
  externalFunctionNames.clear();
  externalVariableNames.clear();
  const allStatements: Statement[] = [];
  for (const rel of sourceRels)
    allStatements.push(...closure.files.get(rel)!.statements);
  moduleDeclRegistry = collectDeclarations(allStatements);

  // Each file of the closure resolves in a scope of its own, following the
  // library's own include/use graph
  const libBind = bindLibrary(
      sourceRels.map(rel => ({
                       rel,
                       program: closure.files.get(rel)!,
                       edges: closure.edges.get(rel) ?? [],
                     })),
      closure.entryRels, currentBindOptions);
  bindResult = libBind;

  const scan = scanProgram(allStatements);
  parentModulesReadInFunction = scan.parentModulesInFunction;

  resolveRuntimeLocals(namesBlockingRuntimeLocals(libBind, scan));
  assignPrettyNames(
      libBind,
      {reserved: reservedNames(), builtinSymbols: builtinSymbolNames()});
  resetTempNames(namesInUse(libBind, scan.unresolved));


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
      manifestSymbols[dk.plural][dk.name] = declJsName(stmt as DeclStmt, dk.ns);
      lists[dk.plural].push(dk.name);
      // Only modules and functions take parameters; `mod:`/`fn:` are exactly
      // the namespace tags, so the signature key needs no second mapping
      if (dk.ns !== 'var') {
        const sigKey = `${dk.ns}:${dk.name}`;
        manifestSignatures[sigKey] =
            (stmt as any).params.map((p: any) => p.name);
        manifestSignatureNoArg[sigKey] =
            (stmt as any).params.map(paramUsesNoArg);
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
      deps: string[]; outRelOf: (sourceRel: string) => string;
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

  // Top-level declarations
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
    if (!isDecl(stmt)) continue;
    const key = declKey(stmt);
    if (!declMap.has(key)) declOrder.push(key);
    declMap.set(key, {stmt, code: compileDeclaration(stmt)});
    const dk = declKindAndName(stmt);
    if (dk) ownNames[dk.kind].add(dk.name);
  }

  const declarations: string[] = [];
  for (const key of declOrder) {
    const entry = declMap.get(key)!;
    declarations.push(entry.code);
    const dk = declKindAndName(entry.stmt);
    if (dk) exportedSymbols.push(declJsName(entry.stmt as DeclStmt, dk.ns));
  }

  const fileScan = scanProgram(program.statements);

  // Resolve cross-file references to imports
  const refs = fileScan.refs;
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

  // A reference reaches another file only when it resolves, in this file's
  // scope, to a binding that another file declares. A name resolving to a
  // builtin or to nothing, is never imported
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
  // builtin const, or a special variable. OpenSCAD resolves unknown reads to
  // undef
  const referenced = fileScan.unresolved;
  // Names this file already declares, under the names it actually emits them
  // with everything else that is referenced needs an `undefined` fallback
  const declaredHere = (names: Set<string>, ns: Namespace) => [...names].map(
      n => globalJsName(n, ns));
  const localDeclared = new Set<string>([
    ...declaredHere(ownNames.variable, 'var'),
    ...declaredHere(ownNames.module, 'mod'),
    ...declaredHere(ownNames.function, 'fn'),
    ...exportedSymbols,
    ...[...importsBySpec.values()].flatMap(set => [...set]),
    ...LIB_BUILTIN_CONSTS,
    ...RUNTIME_SYMBOLS.map(s => RT[s]!),
    ...importedVarNames,
    'Manifold',
    'CrossSection',
    'wasm',
    T('NO_ARG'),
  ]);
  const undefinedNames =
      [...referenced]
          .filter(
              n => !n.startsWith('__') && !n.startsWith('$') &&
                  !localDeclared.has(n))
          .sort();

  let body = '';
  for (const name of undefinedNames) body += `let ${name}: any = undefined;\n`;
  body += '\n';
  if (declarations.length) body += declarations.join('\n') + '\n';
  if (exportedSymbols.length)
    body += `\nexport { ${exportedSymbols.join(', ')} };\n`;

  const out = sideEffectBlock + importBlock + builtinConstantsFor(body) + body;
  return buildRuntimeImport(ctx.runtimePath, out) + out;
}

// Relative ES import specifier from one output file to another
function relImportSpecifier(fromOutRel: string, toOutRel: string): string {
  let rel =
      path.relative(path.dirname(fromOutRel), toOutRel).replace(/\\/g, '/');
  rel = rel.replace(/\.ts$/i, '.js');
  if (!rel.startsWith('.')) rel = './' + rel;
  return rel;
}

// Declarations
const PRE_DECLARED_VARS = new Set([
  '$fn', '$fa', '$fs', '$vpr', '$vpt', '$vpd', '$vpf', '$parent_modules', '$t',
  '$preview', '$color', '$idx', ...BUILTIN_VAR_CONSTANTS
]);

// Emit a `use`d file in its own IIFE so its variables stay private while
// exported modules/functions close over them
function compileUsedFileScope(
    scope: ScopeStmt, unitName: string): {code: string; exports: string[]} {
  // Special variables stay dynamic; pre-declared names use program-wide
  // bindings
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
      lines.push(
          `  let ${privateNames.map(n => `${n}: any = undef`).join(', ')};`);
    }
    for (const k of declOrder) {
      lines.push('  ' + declCode.get(k)!.split('\n').join('\n  '));
    }
    lines.push(`  return {${exports.join(', ')}};`);
    lines.push('})();');
    return {code: lines.join('\n'), exports};
  })();
}

function legacyJsName(name: string, ns: Namespace): string {
  const base = escapeName(name);
  return ns === 'fn' ? `${base}_fn` : ns === 'mod' ? `${base}$mod` : base;
}

// The JS name of a bound parameter, let binding or loop variable
function bindJsName(node: {name: string; binding?: Binding | null | undefined}):
    string {
  return node.binding ? node.binding.jsName : escapeName(node.name);
}

// Scope that top-level names resolve in for the unit currently being emitted
let currentScope: Scope|undefined;

function globalJsName(name: string, ns: Namespace): string {
  const b = currentScope ? lookup(currentScope, name, ns) : null;
  return b ? b.jsName : legacyJsName(name, ns);
}

function declJsName(
    stmt: {name: string; binding?: Binding | undefined},
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
      // assignment
      if (opts?.assignmentOnly) {
        return withLeading(`${name} = ${compileExpr(stmt.value)};`);
      }

      const declType = globalVarDeclKeyword === 'let' ?
          inferDeclaredType(stmt.value) :
          'any';
      return withLeading(`${globalVarDeclKeyword} ${name}: ${declType} = ${
          compileExpr(stmt.value)};`);
    }

    case 'moduleDecl': {
      const dedup = deduplicateParams(stmt.params);
      const declKey = `mod:${stmt.name}`;
      const isDyn = (n: string) => n.startsWith('$') && n !== '$children';
      const renamedParams: string[] = [];
      const params =
          dedup
              .map((p, i) => {
                const base = bindJsName(p);
                const selfRef = !!p.defaultValue &&
                    nodeReferencesIdentifier(p.defaultValue, p.name);
                let pname: string;
                if (isDyn(p.name)) {
                  pname = `${base}${T('arg')}`;
                } else if (selfRef) {
                  pname = `${base}${T('arg')}`;
                  renamedParams.push(base);
                } else {
                  pname = base;
                }
                if (slotUsesNoArg(declKey, p, i)) return `${pname}: any`;
                return p.defaultValue ?
                    `${pname}: any = ${compileExpr(p.defaultValue)}` :
                    `${pname}: any`;
              })
              .join(', ');
      const defaultsPrologue = emitNoArgDefaults(declKey, dedup, '  ');
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
      return withLeading(`function ${declJsName(stmt, 'mod')}(${params}): ${
          GEOMETRY_TYPE} {\n${defaultsPrologue}${body}\n}`);
    }

    case 'functionDecl': {
      const dedup = deduplicateParams(stmt.params);
      const declKey = `fn:${stmt.name}`;
      const renamedParams: string[] = [];
      const params =
          dedup
              .map((p, i) => {
                const base = bindJsName(p);
                const selfRef = !!p.defaultValue &&
                    nodeReferencesIdentifier(p.defaultValue, p.name);
                let pname = base;
                if (selfRef) {
                  pname = `${base}${T('arg')}`;
                  renamedParams.push(base);
                }
                if (slotUsesNoArg(declKey, p, i)) return `${pname}: any`;
                return p.defaultValue ?
                    `${pname}: any = ${compileExpr(p.defaultValue)}` :
                    `${pname}: any`;
              })
              .join(', ');
      const localParams = dedup.map(bindJsName);
      const rebinds =
          renamedParams.map(n => `  let ${n}: any = ${n}${T('arg')};\n`)
              .join('');
      const defaultsPrologue = emitNoArgDefaults(declKey, dedup, '  ');
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
        return withLeading(
            `function ${declJsName(stmt, 'fn')}(${params}): any {\n${rebinds}${
                defaultsPrologue}  while (true) {\n${loopBody}\n  }\n}`);
      }
      const bodyExpr = (compileExpr(stmt.body));
      return withLeading(
          `function ${declJsName(stmt, 'fn')}(${params}): any {\n${rebinds}${
              defaultsPrologue}  return ${bodyExpr};\n}`);
    }

    default:
      return `/* unsupported declaration: ${(stmt as Statement).kind}${
          locTag(stmt)} */`;
  }
}

// Apply OpenSCAD defaults: missing or sentinel args use the default; explicit
// `undef` stays `undef`
function emitNoArgDefaults(
    key: string, params: Parameter[], indent: string): string {
  let out = '';
  params.forEach((p, i) => {
    if (!slotUsesNoArg(key, p, i)) return;
    const pname = bindJsName(p);
    out += `${indent}if (${pname} === ${T('NO_ARG')} || arguments.length <= ${
        i}) ${pname} = ${compileExpr(p.defaultValue!)};\n`;
  });
  return out;
}

// True when an expression references the identifier name anywhere within it
function nodeReferencesIdentifier(
    node: KindedNode|undefined, name: string): boolean {
  return someNode(node, n => n.kind === 'identifier' && n.name === name);
}

// Compile-time divergence detection for non-tail recursion
let userFunctionDefs: Map<string, FunctionDeclStmt> = new Map();

// Depth at which constant evaluation gives up and declares divergence. Kept
// below the JS engine's own stack limit so the cap fires before evalConstExpr
// overflows
const CONST_EVAL_DEPTH_CAP = 4000;
const CONST_UNKNOWN = Symbol('const-unknown');
const CONST_DIVERGE = Symbol('const-diverge');


function containsCallTo(node: KindedNode|undefined, name: string): boolean {
  return someNode(node, n => n.kind === 'call' && n.name === name);
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
        return CONST_UNKNOWN;  // builtin / unknown - not statically evaluable
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

// Reject top-level calls that are proven to never terminate
function reportDivergentCalls(calls: FunctionCallExpr[]): void {
  // Cache each function's termination check since many calls may share the same
  // body
  const runsAway = new Map<string, boolean>();
  for (const expr of calls) {
    const fn = userFunctionDefs.get(expr.name);
    if (!fn) continue;
    let risky = runsAway.get(expr.name);
    if (risky === undefined) {
      // Only non-tail self-recursive functions can run away on the stack, tail
      // ones are compiled to loops
      risky = containsCallTo(fn.body, fn.name) &&
          !hasSelfTailCall(fn.body, fn.name);
      runsAway.set(expr.name, risky);
    }
    if (!risky) continue;
    try {
      evalConstExpr(expr, new Map(), 0);
    } catch (e) {
      if (e === CONST_DIVERGE || e instanceof RangeError) {
        const base = currentMainFilename ? path.basename(currentMainFilename) :
                                           '<unknown>';
        const line = fn.loc?.start.line ?? expr.loc?.start.line ?? 0;
        throw new Error(`Recursion detected calling function '${
            fn.name}' in file ${base}, line ${line}`);
      }
      throw e;
    }
  }
}

// In case of deduplicate parameters - keep last occurrence of each name
function deduplicateParams(params: Parameter[]): Parameter[] {
  const seen = new Map<string, number>();
  for (let i = 0; i < params.length; i++) {
    seen.set(params[i]!.name, i);
  }
  return params.filter((p, i) => seen.get(p.name) === i);
}

// Tail-recursion elimination

// True when expression can reach a tail call to funcName in loop-lowered
// positions
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

// True when every tail path recurses into funcName with no terminating branch
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

// True when instantiating stmt unconditionally calls module `moduleName`
// i.e. the module recurses on every path with no base case
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

// Map call arguments to parameters, matching compileArgList's positional/named
// rules
function resolveArgsToParams(
    args: Argument[], params: Parameter[]): (Expr|undefined)[] {
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

// Rebind parameters for a self tail call, evaluating arguments before
// overwriting old values
function emitSelfTailCall(
    call: Extract<Expr, {kind: 'call'}>, params: Parameter[],
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
      const tmp = `${RT.tc}${tailTempCounter++}`;
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

// Lower a tail expression into a return or a loop iteration
function emitTailBody(
    expr: Expr, funcName: string, params: Parameter[], indent: string): string {
  switch (expr.kind) {
    case 'group':
      return emitTailBody(expr.expr, funcName, params, indent);
    case 'ternary': {
      const cond = compileExpr(expr.condition);
      const t = emitTailBody(expr.ifTrue, funcName, params, indent + '  ');
      const f = emitTailBody(expr.ifFalse, funcName, params, indent + '  ');
      return `${indent}if (${RT.truthy}(${cond})) {\n${t}\n${indent}} else {\n${
          f}\n${indent}}`;
    }
    case 'assert': {
      const condition = expr.args[0] ? compileExpr(expr.args[0].value) : 'true';
      const message =
          expr.args[1] ? compileExpr(expr.args[1].value) : '"Assertion failed"';
      return `${indent}${RT.assert}(${condition}, ${message});\n${
          emitTailBody(expr.expr, funcName, params, indent)}`;
    }
    case 'echo': {
      const eArgs =
          expr.args
              .map(
                  a => a.name ? `(${JSON.stringify(a.name + ' = ')} + ${
                                    RT.oecho}(${compileExpr(a.value)}))` :
                                `${RT.oecho}(${compileExpr(a.value)})`)
              .join(', ');
      return `${indent}${RT.echo}(${eArgs});\n${
          emitTailBody(expr.expr, funcName, params, indent)}`;
    }
    case 'let': {
      if (expr.assignments.some(
              a => a.name.startsWith('$') || a.name === funcName)) {
        return `${indent}return ${compileExpr(expr)};`;
      }

      return (() => {
        const savedNames = expr.assignments.map(a => a.binding?.jsName);
        const lines: string[] = [];
        // Sequential let: each value sees the bindings established before it
        for (const a of expr.assignments) {
          const tmp = `${T('tl')}${tailTempCounter++}`;
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

  const decls: string[] = [];
  const geos: string[] = [];
  const scanned: string[] = [];
  const dollarSaves: string[] = [];
  const dollarRestores: string[] = [];
  const dollarParamSets: string[] = [];

  for (const dp of dollarParamNames) {
    dollarSaves.push(`  let ${T(`save_${dp}`)}: any = ${svTarget(dp)};`);
    dollarParamSets.push(`  ${svTarget(dp)} = ${dp}${T('arg')};`);
    dollarRestores.push(`  ${svTarget(dp)} = ${T(`save_${dp}`)};`);
  }

  const shadowLocals = new Set<string>();
  for (const s of stmts) {
    if (s.kind !== 'variableDecl' || !s.binding) continue;
    if (s.binding.kind === 'local' && shadowsOuterVar(s.binding))
      shadowLocals.add(bindJsName(s));
  }

  // Rename shadowing locals and restore the binding afterward since bodies may
  // be emitted multiple times
  const renamedLocals: {binding: Binding; saved: string}[] = [];

  const declaredInBody = new Set<string>(localParamNames);
  const savedDollars = new Set<string>(dollarParamNames);

  // Body locals follow the file-scope slot rule: hoist names read before
  // assignment
  const bodySlots: {name: string; expr: Expr}[] = [];
  const bodyAssignCounts = new Map<string, number>();
  for (const s of stmts) {
    if (s.kind !== 'variableDecl') continue;
    if (s.name.startsWith('$') && s.name !== '$children') continue;
    const bn = bindJsName(s);
    const en = shadowLocals.has(bn) ? `${bn}_sl` : bn;
    bodySlots.push({name: en, expr: s.value});
    bodyAssignCounts.set(en, (bodyAssignCounts.get(en) ?? 0) + 1);
  }
  const bodyPredeclared = namesNeedingPredeclaration(bodySlots);
  for (const name of bodyPredeclared) {
    if (declaredInBody.has(name)) continue;
    declaredInBody.add(name);
    decls.push(`  let ${name}: any = undef;`);
  }

  {
    for (const s of stmts) {
      if (s.kind === 'empty') continue;
      if (s.kind === 'variableDecl') {
        const name = bindJsName(s);
        const valueExpr = compileExpr(s.value);
        const commentsBefore = leadingCommentLines(s, '  ');
        const commentAfter = trailingCommentText(s);
        if (s.name.startsWith('$') && s.name !== '$children') {
          // Dynamic scoping: save/assign/restore for $ variables (in ctx)
          if (!savedDollars.has(name)) {
            savedDollars.add(name);
            dollarSaves.push(
                `  let ${T(`save_${name}`)}: any = ${svTarget(name)};`);
            dollarRestores.push(`  ${svTarget(name)} = ${T(`save_${name}`)};`);
          }
          decls.push(...commentsBefore);
          decls.push(`  ${svTarget(name)} = ${valueExpr};${commentAfter}`);
          scanned.push(`${svTarget(name)} = ${valueExpr}`);
        } else {
          const emitName = shadowLocals.has(name) ? `${name}_sl` : name;
          decls.push(...commentsBefore);
          if (declaredInBody.has(emitName)) {
            decls.push(`  ${emitName} = ${valueExpr};${commentAfter}`);
          } else {
            declaredInBody.add(emitName);
            const t = bodyAssignCounts.get(emitName) === 1 ?
                inferDeclaredType(s.value) :
                'any';
            decls.push(
                `  let ${emitName}: ${t} = ${valueExpr};${commentAfter}`);
          }
          scanned.push(valueExpr);
          if (shadowLocals.has(name) && s.binding) {
            renamedLocals.push({binding: s.binding, saved: s.binding.jsName});
            s.binding.jsName = emitName;
          }
        }
      } else if (s.kind === 'functionDecl' || s.kind === 'moduleDecl') {
        // Indent the nested declaration
        const decl = compileDeclaration(s);
        decls.push('  ' + decl.split('\n').join('\n  '));
        if (s.kind === 'functionDecl') scanned.push(decl);
      } else {
        const geo = compileGeometry(s);
        if (!geo) continue;
        scanned.push(geo);
        if (hasBackgroundModifier(s)) {
          pushCommentedLine(
              geos, s, `  ${T('background_items')}.push(${geo});`, '  ');
        } else {
          pushCommentedLine(geos, s, `  ${T('items')}.push(${geo});`, '  ');
        }
      }
    }
  };

  for (const r of renamedLocals) r.binding.jsName = r.saved;

  // Only bodies that reach for the module scope pay for it
  const scanText = scanned.join('\n');
  const usesChildrenFn = /(?<![\w$.])children\s*\(/.test(scanText);
  const usesChildrenCount = /(?<![\w$])\$children\b/.test(scanText);
  const usesParentModules = parentModulesReadInFunction ||
      /(?<![\w$])\$parent_modules\b/.test(scanText);


  if (usesChildrenFn || usesChildrenCount) {
    lines.push(`  let ${T('c')}: any = ${RT.children_stack}.length > 0 ? ${
        RT.children_stack}[${
        RT.children_stack}.length - 1] : { fn: undefined, count: 0 };`);
  }
  if (usesChildrenCount) lines.push(`  let $children: any = ${T('c')}.count;`);
  if (usesChildrenFn) {
    lines.push(`  function children(i: any): any { return ${T('c')}.fn ? ${
        T('c')}.fn(i) : Manifold.union([]); }`);
  }
  if (usesParentModules) {
    lines.push(
        `  let ${T('save_$parent_modules')}: any = ${RT.ctx}.$parent_modules;`);
    lines.push(`  ${RT.ctx}.$parent_modules = ${RT.children_stack}.length;`);
  }
  if (dollarRestores.length > 0) {
    lines.push(...dollarSaves);
  }
  lines.push(`  const ${T('items')}: ${GEOMETRY_TYPE}[] = [];`);

  lines.push(...dollarParamSets);

  // Rebind renamed self-referential parameters to their OpenSCAD names
  lines.push(
      ...renamedParamNames.map(n => `  let ${n}: any = ${n}${T('arg')};`));
  lines.push(...decls);
  lines.push(...geos);

  if (dollarRestores.length > 0 || usesParentModules) {
    lines.push(`  try {`);
    lines.push(`    return ${RT.union}(${T('items')});`);
    lines.push(`  } finally {`);
    lines.push(...dollarRestores.map(r => `  ${r}`));
    if (usesParentModules) {
      lines.push(`    ${RT.ctx}.$parent_modules = ${
          T('save_$parent_modules')};`);  // ← restore
    }
    lines.push(`  }`);
  } else {
    lines.push(`  return ${RT.union}(${T('items')});`);
  }

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

function buildWithChildrenCall(
    callExpr: string, children: string[], moduleName: string): string {
  if (children.length === 0) {
    return `${RT.with_children}(() => Manifold.union([]), 0, () => ${
        callExpr}, ${JSON.stringify(moduleName)})`;
  }

  const childrenCode = children.map(child => `() => (${child})`).join(',\n  ');
  const hasAwait =
      childrenCode.includes('await ') || callExpr.includes('await ');

  if (hasAwait) {
    return `await (() => { ` +
        `const ${T('childFns')} = [\n  ${
               children.map(child => `async () => (${child})`)
                   .join(',\n  ')}\n]; ` +
        `return ${RT.with_children}(async (i) => ` +
        `${RT.union}(await Promise.all(${RT.pick_children}(${
               T('childFns')}, i).map(fn => fn())))` +
        `, ${T('childFns')}.length, async () => await ${callExpr}, ${
               JSON.stringify(moduleName)}); ` +
        `})()`;
  }

  return `(() => { ` +
      `const ${T('childFns')} = [\n  ${
             children.map(child => `() => (${child})`).join(',\n  ')}\n]; ` +
      `return ${RT.with_children}((i) => ` +
      `${RT.union}(${RT.pick_children}(${T('childFns')}, i).map(fn => fn()))` +
      `, ${T('childFns')}.length, () => ${callExpr}, ${
             JSON.stringify(moduleName)}); ` +
      `})()`;
}

function compileGeometry(stmt: Statement): string {
  const modifier = (stmt as {modifier?: string}).modifier;
  if (typeof modifier === 'string' && modifier.includes('*'))
    return '';  // disable modifier: subtree is ignored
  const geo = compileGeometryDispatch(stmt);
  if (geo && typeof modifier === 'string' && modifier.includes('!')) {
    return `${RT.rootMod}(${geo})`;
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
  const userSig = signatures.get(`mod:${stmt.name}`);
  const extraArgs = (moduleDeclRegistry.has(stmt.name) && userSig) ?
      stmt.args.filter(
          a => a.name && !a.name.startsWith('$') &&
              !userSig.params.includes(a.name)) :
      [];

  let result: string;
  // User/library modules shadow builtins like any other declaration. A
  // library's wrapper still reaches the builtin because use resolves in its own
  // scope.
  if (stmt.ref?.mod && stmt.ref.mod.kind !== 'builtin') {
    result = compileUserModuleCall(stmt);
  } else
    switch (stmt.name) {
      // Primitives
      case 'cube':
        result = compileSquareOrCube(stmt.args, 'cube');
        break;
      case 'sphere':
        result = compileCircleOrSphere(stmt.args, 'sphere');
        break;
      case 'cylinder':
        result = compileCylinder(stmt.args);
        break;
      case 'circle':
        result = compileCircleOrSphere(stmt.args, 'circle');
        break;
      case 'square':
        result = compileSquareOrCube(stmt.args, 'square');
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
    // $-vars live in ctx; non-$ extra args remain module-level bindings
    if (!name.startsWith('$')) {
      dynamicScopeVars.add(name);
      shadowNames.push(name);
    }
    const valStr = compileExpr(arg.value);
    saves.push(`let ${T(`save_${name}`)}: any = ${svTarget(name)};`);
    decls.push(`${svTarget(name)} = ${valStr};`);
    restores.push(`${svTarget(name)} = ${T(`save_${name}`)};`);
  }

  // Extra (non-$) args are local to the module, so restore their original
  // values while compiling the child block to preserve the caller's scope
  if (shadowNames.length > 0) {
    const params = shadowNames.map(n => `${n}: any`).join(', ');
    const vals = shadowNames.map(n => `${T(`save_${n}`)}`).join(', ');
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
  const args = stmt.args
                   .map(
                       a => a.name ? `(${JSON.stringify(a.name + ' = ')} + ${
                                         RT.oecho}(${compileExpr(a.value)}))` :
                                     `${RT.oecho}(${compileExpr(a.value)})`)
                   .join(', ');
  if (stmt.child && stmt.child.kind !== 'empty') {
    const child = compileGeometry(stmt.child);
    return `(${RT.echo}(${args}), ${child || 'Manifold.union([])'})`;
  }
  return `(${RT.echo}(${args}), Manifold.union([]))`;
}

function compileAssertModule(stmt: ModuleCallStmt): string {
  const condition = stmt.args[0] ? compileExpr(stmt.args[0].value) : 'true';
  const message =
      stmt.args[1] ? compileExpr(stmt.args[1].value) : '"Assertion failed"';
  if (stmt.child && stmt.child.kind !== 'empty') {
    const child = compileGeometry(stmt.child);
    return `(${RT.assert}(${condition}, ${message}), ${
        child || 'Manifold.union([])'})`;
  }
  return `(${RT.assert}(${condition}, ${message}), Manifold.union([]))`;
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
      `${RT.radius}(${argOr(d1)}, ${argOr(r1)}, ${argOr(d)}, ${argOr(r)}, 1)`;
  const rHigh =
      `${RT.radius}(${argOr(d2)}, ${argOr(r2)}, ${argOr(d)}, ${argOr(r)}, 1)`;

  const fnStr = fn ? compileExpr(fn.value) : `${RT.ctx}.$fn`;
  const centerStr = center ? compileExpr(center.value) : 'false';
  const faStr = fa ? compileExpr(fa.value) : `${RT.ctx}.$fa`;
  const fsStr = fs ? compileExpr(fs.value) : `${RT.ctx}.$fs`;

  return `${RT.cylinder}(${hStr}, ${rLow}, ${rHigh}, ${fnStr}, ${centerStr}, ${
      faStr}, ${fsStr})`;
}

function compileCircleOrSphere(args: Argument[], primitive: string): string {
  const r = findArg(args, 'r', 0);
  const d = findArg(args, 'd');
  const fn = findArg(args, '$fn');
  const fa = findArg(args, '$fa');
  const fs = findArg(args, '$fs');

  // Resolve d-vs-r at runtime (see radius).
  const argOr = (a: Argument|undefined) =>
      (a ? compileExpr(a.value) : 'undefined');
  const radiusStr =
      `${RT.radius}(undefined, undefined, ${argOr(d)}, ${argOr(r)}, 1)`;

  const fnStr = fn ? compileExpr(fn.value) : `${RT.ctx}.$fn`;
  const faStr = fa ? compileExpr(fa.value) : `${RT.ctx}.$fa`;
  const fsStr = fs ? compileExpr(fs.value) : `${RT.ctx}.$fs`;

  if (primitive == 'circle') {
    return `${RT.circle}(${radiusStr}, ${fnStr}, ${faStr}, ${fsStr})`;
  } else {
    return `${RT.sphere}(${radiusStr}, ${fnStr}, ${faStr}, ${fsStr})`;
  }
}

function compileSquareOrCube(args: Argument[], primitive: string): string {
  const size = findArg(args, 'size', 0);
  const center = findArg(args, 'center', 1);

  const sizeStr = size ? compileExpr(size.value) : '1';
  const centerStr = center ? compileExpr(center.value) : 'false';

  if (primitive == 'square') {
    return `${RT.square}(${sizeStr}, ${centerStr})`;
  } else {
    return `${RT.cube}(${sizeStr}, ${centerStr})`;
  }
}

function compilePolygon(args: Argument[]): string {
  const points = findArg(args, 'points', 0);
  const paths = findArg(args, 'paths', 1);
  if (!points) return `${RT.polygon}(/* missing points */[])`;
  const pointsStr = compileExpr(points.value);
  const pathsStr = paths ? compileExpr(paths.value) : 'undefined';
  return `${RT.polygon}(${emitArgs(pointsStr, pathsStr)})`;
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
  const fnStr = fn ? compileExpr(fn.value) : `${RT.ctx}.$fn`;

  // Track font for base64 generation and resolve variable name.
  const rawFontSpec = font && font.value.kind === 'string' ? font.value.value :
                                                             DEFAULT_FONT_SPEC;
  encounteredFonts.add(rawFontSpec);

  return `${RT.text}(${txtStr}, ${sizeStr}, ${fontStr}, ${halignStr}, ${
      valignStr}, ${spacingStr}, ${dirStr}, ${fnStr}, ${RT.font_registry})`;
}

function compilePolyhedron(args: Argument[]): string {
  const points = findArg(args, 'points', 0);
  const triangles = findArg(args, 'triangles', 1);
  let faces = findArg(args, 'faces', 2);
  if (triangles) faces = triangles;

  if (!points || !faces) return `/* polyhedron: missing points or faces */`;

  return `${RT.polyhedron}(${compileExpr(points.value)}, ${
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
    return `${RT.rotate}(${
        emitArgs(
            child, a ? compileExpr(a.value) : 'undefined',
            v ? compileExpr(v.value) : 'undefined')})`;
  }
  const vec = stmt.args[0];
  const defaultVec = method === 'translate' ? '[0, 0, 0]' : '[1, 1, 1]';
  const vecStr = vec ? compileExpr(vec.value) : defaultVec;
  return `${RT[method]}(${child}, ${vecStr})`;
}

function compileMirror(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'Manifold.union([])';
  const child = compileGeometry(stmt.child);
  const vec = stmt.args[0];
  const vecStr = vec ? compileExpr(vec.value) : '[1, 0, 0]';
  return `${RT.mirror}(${child}, ${vecStr})`;
}

function compileMultMatrix(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'Manifold.union([])';
  const child = compileGeometry(stmt.child);
  const mat = stmt.args[0];
  if (!mat) return `${child}`;
  return `${RT.transform}(${child}, ${compileExpr(mat.value)})`;
}

function compileColor(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'Manifold.union([])';
  const child = compileGeometry(stmt.child);
  const c = findArg(stmt.args, 'c', 0);
  const alpha = findArg(stmt.args, 'alpha', 1);
  const cExpr = c ? compileExpr(c.value) : 'undefined';
  const aExpr = alpha ? compileExpr(alpha.value) : 'undefined';
  if (child.includes('await ')) {
    return `await (async () => { let ${T('save_$color')}: any = ${
        RT.ctx}.$color; ${RT.ctx}.$color = ${RT.parse_color_for_scope}(${
        emitArgs(cExpr, aExpr)}); try { return await ${RT.color}(${
        emitArgs(child, cExpr, aExpr)}); } finally { ${RT.ctx}.$color = ${
        T('save_$color')}; } })()`;
  }
  return `(() => { let ${T('save_$color')}: any = ${RT.ctx}.$color; ${
      RT.ctx}.$color = ${RT.parse_color_for_scope}(${
      emitArgs(cExpr, aExpr)}); try { return ${RT.color}(${
      emitArgs(
          child, cExpr,
          aExpr)}); } finally { ${RT.ctx}.$color = ${T('save_$color')}; } })()`;
}

function compileResize(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'Manifold.union([])';
  const child = compileGeometry(stmt.child);
  const newsize = findArg(stmt.args, 'newsize', 0);
  const auto = findArg(stmt.args, 'auto', 1);
  const ns = newsize ? compileExpr(newsize.value) : 'undefined';
  const au = auto ? compileExpr(auto.value) : 'undefined';
  return `${RT.resize}(${emitArgs(child, ns, au)})`;
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
  return `${RT.offset}(${child}, ${amt}, "Round", 2, ${RT.ctx}.$fn, ${
      RT.ctx}.$fa, ${RT.ctx}.$fs)`;
}

function compileProjection(stmt: ModuleCallStmt): string {
  if (!stmt.child) return 'CrossSection.square(0)';
  const child = compileGeometry(stmt.child);
  if (child === 'Manifold.union([])') return 'CrossSection.square(0)';
  const cut = findArg(stmt.args, 'cut', 0);
  const cutStr = cut ? compileExpr(cut.value) : 'false';
  return `${RT.projection}(${child}, ${cutStr})`;
}

// echo()/assert() with no child are pure side-effect statements, not geometry
function isSideEffectOnlyModule(s: Statement): boolean {
  return s.kind === 'moduleCall' &&
      (s.name === 'echo' || s.name === 'assert') && !s.child;
}

function collectChildrenWithDecls(
    stmt: ModuleCallStmt, sideEffectsAsChildren = false):
    {decls: string[]; geos: string[]; dollars: {name: string; code: string}[]} {
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
    {decls: string[]; geos: string[]; dollars: {name: string; code: string}[]} {
  const varDecls = new Map < string, {
    stmt: Statement&{kind: 'variableDecl'};
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
          if (prior >= 0)
            dollars[prior] = entry;
          else
            dollars.push(entry);
          continue;
        }
        const existing = varDecls.get(name);
        // Last assignment wins
        varDecls.set(
            name, {stmt: s, order: existing ? existing.order : order++});
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

  const slots =
      [...varDecls.entries()]
          .sort((a, b) => a[1].order - b[1].order)
          .map(([name, v]) => ({name, expr: v.stmt.value, stmt: v.stmt}));
  const predeclared = namesNeedingPredeclaration(slots);

  const hoisted = slots.filter(s => predeclared.has(s.name))
                      .map(s => `let ${s.name}: any = undef;`);
  const orderedVars = slots.map(s => {
    const lead = `${leadingCommentLines(s.stmt).join('\n')}${
        s.stmt.leadingComments?.length ? '\n' : ''}`;
    const trail = trailingCommentText(s.stmt);
    const value = compileExpr(s.expr);
    // A hoisted name is already declared, so its slot is a bare assignment
    if (predeclared.has(s.name)) return `${lead}${s.name} = ${value};${trail}`;
    return `${lead}let ${s.name}: ${inferDeclaredType(s.expr)} = ${value};${
        trail}`;
  });
  const decls = [...hoisted, ...orderedVars, ...otherDecls];

  return {decls, geos, dollars};
}

function wrapDollarScope(
    body: string, dollars: {name: string; code: string}[]): string {
  let out = body;
  for (let i = dollars.length - 1; i >= 0; i--) {
    const d = dollars[i]!;
    const t = svTarget(d.name);
    if (out.includes('await ')) {
      out = `await (async () => { let ${T(`save_${d.name}`)}: any = ${t}; ${
          t} = ${d.code}; try { return await ${
          returnExpr(
              out, '  ')}; } finally { ${t} = ${T(`save_${d.name}`)}; } })()`;
    } else {
      out = `(() => { let ${T(`save_${d.name}`)}: any = ${t}; ${t} = ${
          d.code}; try { return ${returnExpr(out, '  ')}; } finally { ${t} = ${
          T(`save_${d.name}`)}; } })()`;
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
    result = `${RT.union}([\n  ${geos.join(',\n  ')}\n])`;
  } else if (op === 'intersection') {
    result = `${RT.intersection}([\n  ${geos.join(',\n  ')}\n])`;
  } else if (op === 'hull') {
    result = `${RT.hull}([\n  ${geos.join(',\n  ')}\n])`;
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
    result = `${RT.difference}(${first}, [\n  ${rest.join(',\n  ')}\n])`;
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
    result = `${RT.minkowski}([\n  ${geos.join(',\n  ')}\n])`;
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

  opts.push(`fn: ${fn ? compileExpr(fn.value) : `${RT.ctx}.$fn`}`);

  opts.push(`fa: ${fa ? compileExpr(fa.value) : `${RT.ctx}.$fa`}`);

  opts.push(`fs: ${fs ? compileExpr(fs.value) : `${RT.ctx}.$fs`}`);

  opts.push(`fe: ${fe ? compileExpr(fe.value) : `${RT.ctx}.$fe`}`);

  if (slices) {
    opts.push(`slices: ${compileExpr(slices.value)}`);
  }

  if (opts.length) {
    return `${RT.linear_extrude}(${child}, ${hStr}, { ${opts.join(', ')} })`;
  }
  return `${RT.linear_extrude}(${child}, ${hStr})`;
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
  const fnStr = fn ? compileExpr(fn.value) : `${RT.ctx}.$fn`;
  const faStr = fa ? compileExpr(fa.value) : `${RT.ctx}.$fa`;
  const fsStr = fs ? compileExpr(fs.value) : `${RT.ctx}.$fs`;
  return `${RT.rotate_extrude}(${child}, ${fnStr}, ${faStr}, ${fsStr}, ${
      aStr})`;
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
                                     `${RT.union}([\n  ${geos.join(',\n  ')}\n])`;

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
        body = `await (async () => { let ${T(`save_${d.name}`)}: any = ${t}; ${
            t} = ${d.code}; try { return await ${
            returnExpr(
                body,
                '  ')}; } finally { ${t} = ${T(`save_${d.name}`)}; } })()`;
      } else {
        body = `(() => { let ${T(`save_${d.name}`)}: any = ${t}; ${t} = ${
            d.code}; try { return ${returnExpr(body, '  ')}; } finally { ${
            t} = ${T(`save_${d.name}`)}; } })()`;
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
    `  const ${T('items')}: ${GEOMETRY_TYPE}[] = [];`,
    ...buildNestedForStatements(variables, 0, stmt.child, 1),
    `  return ${RT.intersection}(${T('items')});`,
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

// For / If geometry
function compileForGeometry(stmt: ForStmt): string {
  if (stmt.variables.length === 0) return 'Manifold.union([])';
  const lines = [
    '(() => {',
    `  const ${T('items')}: ${GEOMETRY_TYPE}[] = [];`,
    ...buildNestedForStatements(stmt.variables, 0, stmt.body, 1),
    `  return ${RT.union}(${T('items')});`,
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
      pushCommentedLine(
          lines, body, `${indent}${T('items')}.push(${geo});`, indent);
    return lines;
  }

  const v = vars[idx]!;
  const vName = bindJsName(v);
  if (v.range.kind === 'range') {
    const start = compileExpr(v.range.start);
    const end = compileExpr(v.range.end);
    const step = v.range.step ? compileExpr(v.range.step) : '1';
    const stepName = `${T(`step_${idx}`)}`;
    return [
      `${indent}{`,
      `${indent}  const ${T(`start_${idx}`)}: any = ${start}, ${
          stepName}: any = ${step}, ${T(`end_${idx}`)}: any = ${end};`,
      `${indent}  const ${T(`cnt_${idx}`)}: any = ${RT.rangeCount}(${
          T(`start_${idx}`)}, ${stepName}, ${T(`end_${idx}`)});`,
      `${indent}  for (let ${T(`i_${idx}`)} = 0; ${T(`i_${idx}`)} < ${
          T(`cnt_${idx}`)}; ${T(`i_${idx}`)}++) {`,
      `${indent}    const ${vName}: any = ${T(`i_${idx}`)} === 0 ? ${
          T(`start_${idx}`)} : ${T(`start_${idx}`)} + ${T(`i_${idx}`)} * ${
          stepName};`,
      ...buildNestedForStatements(vars, idx + 1, body, indentLevel + 2),
      `${indent}  }`,
      `${indent}}`,
    ];
  }

  const iterName = `${T(`iter_${idx}`)}`;
  const idxName = `${T(`idx_${idx}`)}`;
  return [
    `${indent}{`,
    `${indent}  const ${iterName}: any = ${RT.each}(${compileExpr(v.range)});`,
    `${indent}  for (let ${idxName} = 0; ${idxName} < ${iterName}.length; ${
        idxName}++) {`,
    `${indent}    const ${vName}: any = ${iterName}[${idxName}];`,
    `${indent}    let ${T('save_$idx')}: any = ${RT.ctx}.$idx; ${
        RT.ctx}.$idx = ${idxName};`,
    `${indent}    try {`,
    ...buildNestedForStatements(vars, idx + 1, body, indentLevel + 3),
    `${indent}    } finally { ${RT.ctx}.$idx = ${T('save_$idx')}; }`,
    `${indent}  }`,
    `${indent}}`,
  ];
}

function compileIfGeometry(stmt: IfStmt): string {
  const cond = `${RT.truthy}(${compileExpr(stmt.condition)})`;
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
  const argList = compileArgList(`mod:${stmt.name}`, stmt.args);
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

// A generated `surface()` data file containing decoded pixels or raw matrix
// data
interface SurfaceAsset {
  stem: string;
  exportName: string;
  kind: 'image'|'text';
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

// Write a `surface()` data file and return its exported symbol, or warn and
// skip on failure
function emitSurfaceData(filenameStr: string, basePath: string): SurfaceAsset|
    undefined {
  const filePath = path.resolve(basePath, filenameStr);
  if (!fs.existsSync(filePath)) {
    console.warn(`Warning: surface("${filenameStr}"): can't open file "${
        filePath}", ignoring.`);
    return undefined;
  }

  // OpenSCAD treats only PNG as an image; everything else is a text matrix
  const isImage = path.extname(filePath).toLowerCase() === '.png';
  const stem = path.basename(filenameStr, path.extname(filenameStr))
                   .replace(/[^a-zA-Z0-9_]/g, '_');

  const currentFileDir = typeof __dirname !== 'undefined' ?
      __dirname :
      path.dirname(
          new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/i, '$1'));
  const surfaceDataDir =
      path.join(path.resolve(currentFileDir, '..'), 'runtime', 'surface_data');
  fs.mkdirSync(surfaceDataDir, {recursive: true});

  const write = (body: string) => fs.writeFileSync(
      path.join(surfaceDataDir, `${stem}_data.ts`),
      `// Auto-generated by OpenSCAD compiler — do not edit\n` +
          `// Source: ${filePath}\n` + body,
      'utf8');

  if (isImage) {
    const pixels = decodeImagePixels(filePath);
    if (!pixels) {
      console.warn(`Warning: surface("${filenameStr}"): can't decode image "${
          filePath}", ignoring.`);
      return undefined;
    }
    const exportName = T(`img_${stem}`);
    write(
        `// ${pixels.width}x${pixels.height}, base64 of 3 bytes (RGB) per ` +
        `pixel, row-major from the top-left\n` +
        `export const ${exportName} = { width: ${pixels.width}, height: ${
            pixels.height}, rgb: "${pixels.rgb}" };\n`);
    return {stem, exportName, kind: 'image'};
  }

  // Text matrix (.dat / .txt): embed the raw content as a string literal
  const exportName = T(`surfacedata_${stem}`);
  write(`export const ${exportName} = ${
      JSON.stringify(fs.readFileSync(filePath, 'utf8'))};\n`);
  return {stem, exportName, kind: 'text'};
}

// compile surface
export function compileSurface(args: Argument[]): string {
  const file = findArg(args, 'file', 0);
  const center = findArg(args, 'center', 1);
  const invert = findArg(args, 'invert', 2);

  if (!file?.value || file.value.kind !== 'string') {
    console.warn(`Warning: surface(): no file argument given, ignoring.`);
    return 'Manifold.union([])';
  }
  const filenameStr = file.value.value;

  // OpenSCAD resolves a surface() file relative to the .scad file that contains
  // the call
  const sourceFile = currentSourceFilename || currentMainFilename;
  const basePath =
      sourceFile ? path.dirname(path.resolve(sourceFile)) : process.cwd();

  // emitSurfaceData warns and yields undefined for a file that is missing
  // or cannot be decoded
  const asset = emitSurfaceData(filenameStr, basePath);
  if (!asset) return 'Manifold.union([])';
  encounteredSurfaceData.set(filenameStr, asset);

  const centerStr = center ? compileExpr(center.value) : 'false';
  // invert only reaches the image path; a text matrix has no pixels to flip
  const opts = asset.kind === 'image' ?
      `center: ${centerStr}, invert: ${
          invert ? compileExpr(invert.value) : 'false'}, kind: "image"` :
      `center: ${centerStr}, kind: "text"`;

  return `${RT.surface}(${asset.exportName}, { ${opts}, fn: ${
      RT.ctx}.$fn, fa: ${RT.ctx}.$fa, fs: ${RT.ctx}.$fs })`;
}

// Local numeric type inference

// Arithmetic whose result is a number when both operands are numbers
const NUMERIC_RESULT_OPS = new Set(['+', '-', '*', '/', '%']);

// Operators safe to emit natively when both operands are numbers
const NATIVE_OPS =
    new Set(['+', '-', '*', '/', '%', '<', '>', '<=', '>=', '==', '!=']);

// JS precedence for the operators above; higher binds tighter
const JS_PRECEDENCE: Record<string, number> = {
  '==': 8,
  '!=': 8,
  '<': 9,
  '>': 9,
  '<=': 9,
  '>=': 9,
  '+': 11,
  '-': 11,
  '*': 12,
  '/': 12,
  '%': 12,
};
const PREC_UNARY = 14;

// OpenSCAD's == and != on two numbers are exact identity, which is === in JS
const NATIVE_OP_SPELLING: Record<string, string> = {
  '==': '===',
  '!=': '!=='
};

// True when this call site really reaches the runtime builtin rather than a
// user or library definition that shadows the name
function callsUnshadowedBuiltin(expr: FunctionCallExpr): boolean {
  if (isLexicalVar(expr.ref?.value)) return false;
  return expr.ref?.fn?.kind === 'builtin';
}

function isNumericBuiltinCall(expr: FunctionCallExpr): boolean {
  const arity = NUMERIC_BUILTINS.get(expr.name);
  if (arity === undefined || expr.args.length !== arity) return false;
  if (expr.args.some(a => a.name)) return false;
  if (!callsUnshadowedBuiltin(expr)) return false;
  return expr.args.every(a => isNumericExpr(a.value));
}

// Whether an expression provably evaluates to a number, for every possible
// input
function isNumericExpr(expr: Expr): boolean {
  switch (expr.kind) {
    case 'number':
      return true;
    case 'group':
      return isNumericExpr(expr.expr);
    case 'unary':
      return (expr.op === '-' || expr.op === '+') &&
          isNumericExpr(expr.operand);
    case 'binary':
      if (expr.op === '^' || NUMERIC_RESULT_OPS.has(expr.op)) {
        return isNumericExpr(expr.left) && isNumericExpr(expr.right);
      }
      return false;
    case 'ternary':
      return isNumericExpr(expr.ifTrue) && isNumericExpr(expr.ifFalse);
    case 'call':
      return isNumericBuiltinCall(expr);
    default:
      return false;
  }
}

// Add parentheses only when JS precedence would change the grouping
function parenthesizeIfNeeded(
    code: string, prec: number, parentPrec: number, isRight: boolean): string {
  const needed = prec < parentPrec || (prec === parentPrec && isRight);
  return needed ? `(${code})` : code;
}

// Emit numeric expressions with JS operators, adding parentheses only when
// needed
function emitNativeNumeric(
    expr: Expr, parentPrec: number, isRight: boolean): string {
  switch (expr.kind) {
    case 'group':
      return emitNativeNumeric(expr.expr, parentPrec, isRight);
    case 'unary':
      if (expr.op === '+') {
        return emitNativeNumeric(expr.operand, parentPrec, isRight);
      }
      if (expr.op === '-') {
        const operand = emitNativeNumeric(expr.operand, PREC_UNARY, false);
        const negated = /^[-+]/.test(operand) ? `-(${operand})` : `-${operand}`;
        return parenthesizeIfNeeded(negated, PREC_UNARY, parentPrec, isRight);
      }
      break;
    case 'binary':
      if (NATIVE_OPS.has(expr.op)) {
        const prec = JS_PRECEDENCE[expr.op]!;
        const op = NATIVE_OP_SPELLING[expr.op] ?? expr.op;
        const left = emitNativeNumeric(expr.left, prec, false);
        const right = emitNativeNumeric(expr.right, prec, true);
        return parenthesizeIfNeeded(
            `${left} ${op} ${right}`, prec, parentPrec, isRight);
      }
      break;
    default:
      break;
  }

  return compileExpr(expr);
}

// Use native JS operators for numeric binary expressions; otherwise use the
// runtime helper
function tryLowerBinary(expr: BinaryExpr): string|undefined {
  if (!NATIVE_OPS.has(expr.op)) return undefined;
  if (!isNumericExpr(expr.left) || !isNumericExpr(expr.right)) return undefined;
  return emitNativeNumeric(expr, 0, false);
}

// Type for folded initializers; use a specific type only when certain,
// otherwise `any`
type DeclaredType = 'number'|'string'|'boolean'|'any';

// Operators whose emitted form is a boolean
const ALWAYS_BOOLEAN_OPS = new Set(['==', '!=', '&&', '||']);

function inferDeclaredType(expr: Expr): DeclaredType {
  if (isNumericExpr(expr)) return 'number';
  switch (expr.kind) {
    case 'string':
      return 'string';
    case 'boolean':
      return 'boolean';
    case 'group':
      return inferDeclaredType(expr.expr);
    case 'unary':
      return expr.op === '!' ? 'boolean' : 'any';
    case 'binary':
      if (ALWAYS_BOOLEAN_OPS.has(expr.op)) return 'boolean';
      if (JS_PRECEDENCE[expr.op] !== undefined && isNumericExpr(expr.left) &&
          isNumericExpr(expr.right)) {
        return 'boolean';
      }
      return 'any';
    default:
      return 'any';
  }
}

interface SlotUse {
  reads: Set<string>;
  callsUserCode: boolean;
}

function analyzeSlotExpr(expr: Expr): SlotUse {
  const use: SlotUse = {reads: new Set(), callsUserCode: false};
  walk(expr as unknown as KindedNode, node => {
    if (node.kind === 'identifier') {
      const b = (node as {binding?: Binding}).binding;
      if (b && b.ns === 'var') use.reads.add(b.jsName);
    } else if (node.kind === 'call') {
      const call = node as unknown as FunctionCallExpr;
      if (call.ref?.fn?.kind !== 'builtin') use.callsUserCode = true;
      if (call.ref?.value) use.reads.add(call.ref.value.jsName);
    } else if (node.kind === 'moduleCall') {
      use.callsUserCode = true;
    }
  });
  return use;
}

// Variables that need a hoisted undef declaration
function namesNeedingPredeclaration(slots: {name: string; expr: Expr}[]):
    Set<string> {
  const needed = new Set<string>();
  const uses = slots.map(s => analyzeSlotExpr(s.expr));
  const firstUserCall = uses.findIndex(u => u.callsUserCode);
  const unsafeFrom = firstUserCall < 0 ? slots.length : firstUserCall;
  // A name may be assigned more than once, but the declaration goes at its
  // first assignment even when only the final value survives.
  const firstIndex = new Map<string, number>();
  slots.forEach((s, i) => {
    if (!firstIndex.has(s.name)) firstIndex.set(s.name, i);
  });
  for (const [name, j] of firstIndex) {
    if (j >= unsafeFrom) {
      needed.add(name);
      continue;
    }

    for (let i = 0; i <= j; i++) {
      if (uses[i]!.reads.has(name)) {
        needed.add(name);
        break;
      }
    }
  }
  return needed;
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
      // Other special variables use the shared runtime context across compiled
      // files
      if (expr.name.startsWith('$')) return `${RT.ctx}.${expr.name}`;
      const en = escapeName(expr.name);
      return expr.binding ? expr.binding.jsName : en;
    }
    case 'vector':
      return `[${expr.elements.map(compileExpr).join(', ')}]`;
    case 'range':
      if (expr.step) {
        return `${RT.range}(${compileExpr(expr.start)}, ${
            compileExpr(expr.step)}, ${compileExpr(expr.end)})`;
      }
      return `${RT.range}(${compileExpr(expr.start)}, 1, ${
          compileExpr(expr.end)})`;
    case 'binary': {
      // Both operands provably numeric: plain JS operators are equivalent
      const native = tryLowerBinary(expr);
      if (native !== undefined) return native;
      if (expr.op === '^') {
        return `Math.pow(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      }
      if (expr.op === '==') {
        return `${RT.eq}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      }
      if (expr.op === '!=') {
        return `(!${RT.eq}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)}))`;
      }
      if (expr.op === '+')
        return `${RT.add}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '-')
        return `${RT.sub}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '*')
        return `${RT.mul}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '/')
        return `${RT.div}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '%')
        return `${RT.mod}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '&')
        return `${RT.band}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '|')
        return `${RT.bor}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '<<')
        return `${RT.shl}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '>>')
        return `${RT.shr}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '<')
        return `${RT.lt}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '>')
        return `${RT.gt}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '<=')
        return `${RT.le}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '>=')
        return `${RT.ge}(${compileExpr(expr.left)}, ${
            compileExpr(expr.right)})`;
      if (expr.op === '&&' || expr.op === '||') {
        return `(${RT.truthy}(${compileExpr(expr.left)}) ${expr.op} ${
            RT.truthy}(${compileExpr(expr.right)}))`;
      }
      return `(${compileExpr(expr.left)} ${expr.op} ${
          compileExpr(expr.right)})`;
    }
    case 'unary':
      if ((expr.op === '-' || expr.op === '+') && isNumericExpr(expr.operand)) {
        return emitNativeNumeric(expr, 0, false);
      }
      if (expr.op === '-') return `${RT.neg}(${compileExpr(expr.operand)})`;
      if (expr.op === '+') return `${RT.pos}(${compileExpr(expr.operand)})`;
      if (expr.op === '!')
        return `(!${RT.truthy}(${compileExpr(expr.operand)}))`;
      if (expr.op === '~') return `${RT.bnot}(${compileExpr(expr.operand)})`;
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
        return `...(${RT.truthy}(${compileExpr(expr.condition)}) ? ${
            ifTrue} : ${ifFalse})`;
      }
      return `(${RT.truthy}(${compileExpr(expr.condition)}) ? ${ifTrue} : ${
          ifFalse})`;
    }
    case 'call':
      return compileCallExpr(expr);
    case 'index':
      return `${RT.index}(${compileExpr(expr.object)}, ${
          compileExpr(expr.index)})`;
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
                  a => a.name ? `(${JSON.stringify(a.name + ' = ')} + ${
                                    RT.oecho}(${compileExpr(a.value)}))` :
                                `${RT.oecho}(${compileExpr(a.value)})`)
              .join(', ');
      return `(${RT.echo}(${eArgs}), ${compileExpr(expr.expr)})`;
    }
    case 'assert': {
      const condition = expr.args[0] ? compileExpr(expr.args[0].value) : 'true';
      const message =
          expr.args[1] ? compileExpr(expr.args[1].value) : '"Assertion failed"';
      return `(${RT.assert}(${condition}, ${message}), ${
          compileExpr(expr.expr)})`;
    }
    case 'let': {
      const localAssignNames =
          expr.assignments.filter(a => !a.name.startsWith('$')).map(bindJsName);
      return (() => {
        const bound: string[] = [];
        const vals = expr.assignments.map(a => {
          const selfRec = a.value.kind === 'lambda' && !a.name.startsWith('$');
          const suppress = selfRec ? [...bound, bindJsName(a)] : bound;
          const val = (compileExpr(a.value));
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
            result = `(() => { const ${T(`save_${name}`)}: any = ${t}; ${t} = ${
                val}; try { return ${result}; } finally { ${t} = ${
                T(`save_${name}`)}; } })()`;
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
      // each <generator> expands every yielded element, so wrap generator
      // results in each to preserve that expansion
      if (expr.expr.kind === 'listComp') {
        return `...(${RT.flat_map_iter}(${
            compileListComp(expr.expr.generator)}, (${T('ev')}: any) => ${
            RT.each}(${T('ev')})))`;
      }
      const inner = compileExpr(expr.expr);
      if (inner.startsWith('...')) return inner;
      return `...${RT.each}(${inner})`;
    }
    case 'lambda': {
      const localParams = expr.params.map(bindJsName);
      const params =
          expr.params
              .map(
                  p => p.defaultValue ?
                      `${bindJsName(p)} = ${compileExpr(p.defaultValue)}` :
                      bindJsName(p))
              .join(', ');
      // Compile the body in tail position so function-value tail calls become
      // tc thunks that the trampoline can run iteratively.
      const bodyExpr = (compileExprTail(expr.body));

      return `${RT.fnlit}((${params}) => ${bodyExpr}, ${
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
      return `${RT.call}(${callee}${args ? `, ${args}` : ''})`;
    }
    default:
      return `/* unsupported expr: ${(expr as Expr).kind}${
          locTag(expr as ASTNode)} */`;
  }
}

// Reconstruct the OpenSCAD source form of an expression
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

function oscadParamSource(p: Parameter): string {
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
      return `(${RT.truthy}(${compileExpr(expr.condition)}) ? ${
          compileExprTail(expr.ifTrue)} : ${compileExprTail(expr.ifFalse)})`;
    }
    case 'echo': {
      const eArgs =
          expr.args
              .map(
                  a => a.name ? `(${JSON.stringify(a.name + ' = ')} + ${
                                    RT.oecho}(${compileExpr(a.value)}))` :
                                `${RT.oecho}(${compileExpr(a.value)})`)
              .join(', ');
      return `(${RT.echo}(${eArgs}), ${compileExprTail(expr.expr)})`;
    }
    case 'assert': {
      const condition = expr.args[0] ? compileExpr(expr.args[0].value) : 'true';
      const message =
          expr.args[1] ? compileExpr(expr.args[1].value) : '"Assertion failed"';
      return `(${RT.assert}(${condition}, ${message}), ${
          compileExprTail(expr.expr)})`;
    }
    case 'let': {
      const localAssignNames =
          expr.assignments.filter(a => !a.name.startsWith('$')).map(bindJsName);
      return (() => {
        const bound: string[] = [];
        const vals = expr.assignments.map(a => {
          const selfRec = a.value.kind === 'lambda' && !a.name.startsWith('$');
          const suppress = selfRec ? [...bound, bindJsName(a)] : bound;
          const val = (compileExpr(a.value));
          if (!a.name.startsWith('$')) bound.push(bindJsName(a));
          return val;
        });
        let result = (compileExprTail(expr.body));
        for (let i = expr.assignments.length - 1; i >= 0; i--) {
          const a = expr.assignments[i]!;
          const name = bindJsName(a);
          const val = vals[i]!;
          if (a.name.startsWith('$')) {
            const t = svTarget(name);
            result = `(() => { const ${T(`save_${name}`)}: any = ${t}; ${t} = ${
                val}; try { return ${result}; } finally { ${t} = ${
                T(`save_${name}`)}; } })()`;
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
      return `${RT.tc}(${callee}${args ? `, ${args}` : ''})`;
    }
    default:
      return compileExpr(expr);
  }
}

function compileCallExpr(
    expr: {
      kind: 'call'; name: string; args: Argument[];
      loc?: ASTNode['loc'];
      ref?: CallRef | undefined;
    },
    tail = false): string {
  const escaped = escapeName(expr.name);
  const isKnownFunction =
      BUILTIN_FUNCTIONS.has(expr.name) || signatures.has(`fn:${expr.name}`);

  // Other special variables use the shared runtime context across compiled
  // files
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
    return `${RT.unknown_fn}(${JSON.stringify(expr.name)})`;
  }

  // OpenSCAD keeps variables and functions separate, so prefer a
  // function-valued variable when one exists
  const shadowingValue = !isSpecialVarCallee && callsValue;
  const dualDispatch = isKnownFunction && shadowingValue;

  // When the callee isn't a known function definition but resolves to a value
  const fnName = expr.ref?.fn?.jsName ?? `${escaped}_fn`;
  const valueName = expr.ref?.value?.jsName ?? escaped;
  const name = isSpecialVarCallee          ? svTarget(escaped) :
      (shadowingValue && !isKnownFunction) ? valueName :
                                             fnName;

  const isValueCall = name !== fnName;

  const sig = signatures.get(`fn:${expr.name}`);
  const dollarArgs = expr.args.filter(
      a => a.name && a.name.startsWith('$') &&
          !(sig && sig.params.includes(a.name)));
  const positionalArgs = dollarArgs.length === 0 ?
      expr.args :
      expr.args.filter(a => !dollarArgs.includes(a));

  const argList = compileArgList(`fn:${expr.name}`, positionalArgs);
  const call = dualDispatch ? (() => {
    // A call through a value has no declared signature to match against
    const valueArgList = compileArgList(`var:${expr.name}`, positionalArgs);
    return `(typeof ${valueName} === "function" ? ${
        tail ? `${RT.tc}` : `${RT.call}`}(${valueName}${
        valueArgList ? `, ${valueArgList}` : ''}) : ${name}(${argList}))`;
  })() :
      isValueCall ? `${tail ? `${RT.tc}` : `${RT.call}`}(${name}${
                        argList ? `, ${argList}` : ''})` :
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
    saves.push(`let ${T(`save_${dn}`)}: any = ${t};`);
    decls.push(`${t} = ${compileExpr(value)};`);
    restores.push(`${t} = ${T(`save_${dn}`)};`);
  }
  return `(() => { ${saves.join(' ')} ${decls.join(' ')} try { return ${
      call}; } finally { ${restores.join(' ')} } })()`;
}

// List comprehension
function compileListComp(gen: ListCompGenerator): string {
  switch (gen.kind) {
    case 'lcFor': {
      const bound: string[] = [];
      const ranges = gen.variables.map(v => {
        const parts =
            (v.range.kind === 'range' ?
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
          result = `(() => { const ${T('r')} = []; const ${T('start')}: any = ${
              start}, ${T('step')}: any = ${step}, ${T('end')}: any = ${
              end}; const ${T('cnt')}: any = ${RT.rangeCount}(${T('start')}, ${
              T('step')}, ${T('end')}); for (let ${T('i')} = 0; ${T('i')} < ${
              T('cnt')}; ${T('i')}++) { const ${vName}: any = ${
              T('i')} === 0 ? ${T('start')} : ${T('start')} + ${T('i')} * ${
              T('step')}; ${T('r')}.push(...(${result})); } return ${
              T('r')}; })()`;
        } else {
          result =
              `${RT.flat_map_iter}(${ranges[i]![0]}, (${vName}) => ${result})`;
        }
      }
      return result;
    }
    case 'lcIf': {
      const cond = compileExpr(gen.condition);
      let ifTrue = compileListComp(gen.ifTrue);
      let ifFalse = gen.ifFalse ? compileListComp(gen.ifFalse) : '[]';
      // Both branches are now guaranteed to evaluate to an array.
      return `(${RT.truthy}(${cond}) ? ${ifTrue} : ${ifFalse})`;
    }
    case 'lcLet': {
      // Sequential let: each value sees earlier bindings, and each binding
      // shadows outer locals
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
      // Init values use the outer scope; loop names shadow outer locals
      // afterward
      const inits =
          gen.inits.map(a => `${bindJsName(a)} = ${compileExpr(a.value)}`);
      const loopNames = new Set(gen.inits.map(bindJsName));
      const [cond, updates, inner] = ([
        compileExpr(gen.condition),
        gen.updates.map(a => `${bindJsName(a)} = ${compileExpr(a.value)}`)
            .join(', '),
        compileListComp(gen.body),
      ]);
      // Update-only names belong to the loop, so declare them with the init
      // bindings
      for (const name of new Set(gen.updates.map(bindJsName)))
        if (!loopNames.has(name)) inits.push(`${name} = undefined`);
      // Abort the loop once its counter exceeds the limit
      const base = currentMainFilename ? path.basename(currentMainFilename) :
                                         '<unknown>';
      const line = gen.loc?.start.line ?? 0;
      const errMsg =
          JSON.stringify(`ERROR: For loop counter exceeded limit in file ${
              base}, line ${line}`);
      return `(() => { const ${T('r')} = []; let ${T('fc')} = 0; for (let ${
          inits.join(', ')}; ${RT.truthy}(${cond}); ${updates}) { if (${
          T('fc')}++ >= ${MAX_FOR_ITERATIONS}) throw new Error(${errMsg}); ${
          T('r')}.push(...(${inner})); } return ${T('r')}; })()`;
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
