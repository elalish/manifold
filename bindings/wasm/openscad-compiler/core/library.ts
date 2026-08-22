// Separate library compilation
import path from 'path';

import type {Program, ScopeStmt, Statement} from './ast.js';
import {bindLibrary, lookup} from './binder.js';
import {BUILTIN_MODULES, BUILTIN_SIGNATURES, BUILTIN_VAR_CONSTANTS, RUNTIME_SYMBOLS,} from './builtins.js';
import {formatCode} from './format.js';
import {compileDeclaration, PRE_DECLARED_VARS} from './geometry.js';
import {assignPrettyNames, bindJsName, buildRuntimeImport, builtinConstantsFor, builtinSymbolNames, declJsName, globalJsName, namesBlockingRuntimeLocals, namesInUse, reservedNames, resetTempNames, resolveRuntimeLocals, T,} from './naming.js';
import {collectDeclarations, paramUsesNoArg, scanProgram,} from './scan.js';
import {currentBindOptions, currentMainFilename, currentScope, currentSourceFilename, dynamicScopeVars, encounteredFonts, encounteredSurfaceData, externalFunctionNames, externalModuleNames, externalVariableNames, localDecls, noArgDemotions, resetTailTemps, RT, setBindResult, setCurrentRuntimePath, setCurrentScope, setCurrentSourceFilename, setGlobalVarDeclKeyword, setMainFilename, setModuleDecls, setParentModulesReadInFunction, signatures,} from './state.js';
import type {Binding, CompiledLibrary, CompiledLibraryFile, LibraryClosure, LibraryManifest, Namespace, Scope,} from './types.js';

// Separate library compilation

type LibDeclKind = 'module'|'function'|'variable';
type DeclStmt = Extract<Statement, {kind: keyof typeof DECL_INFO}>;

const LIB_BUILTIN_CONSTS = new Set(BUILTIN_VAR_CONSTANTS);
export const MANIFEST_VERSION = 1;

// Maps the three declaration kinds to their library manifest keys: `ns` is the
// namespace, while `kind` and `plural` are singular/plural keys.
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

export function isDecl(stmt: Statement): stmt is DeclStmt {
  return stmt.kind in DECL_INFO;
}

export function declKey(stmt: DeclStmt): string {
  const {ns} = DECL_INFO[stmt.kind];
  return `${ns === 'var' ? 'var' : 'fn'}:${declJsName(stmt, ns)}`;
}

// Library-manifest view of a declaration, or undefined for non-declarations
// Special variables are dynamically scoped and never exported
export function declKindAndName(stmt: Statement): DeclInfo&{name: string}|
    undefined {
  if (!(isDecl(stmt))) return undefined;
  if (stmt.kind === 'variableDecl' && stmt.name.startsWith('$'))
    return undefined;
  return {...DECL_INFO[stmt.kind], name: stmt.name};
}

export async function compileLibrary(
    closure: LibraryClosure,
    opts: {runtimeVersion: string; runtimePathFor: (outRel: string) => string},
    ): Promise<CompiledLibrary> {
  const sourceRels = [...closure.files.keys()].sort();
  const outRelOf = (sourceRel: string) => sourceRel.replace(/\.scad$/i, '.ts');

  setGlobalVarDeclKeyword('var');

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
  setModuleDecls(collectDeclarations(allStatements));

  // Each file of the closure resolves in a scope of its own, following the
  // library's own include/use graph
  const libBind = bindLibrary(
      sourceRels.map(rel => ({
                       rel,
                       program: closure.files.get(rel)!,
                       edges: closure.edges.get(rel) ?? [],
                     })),
      closure.entryRels, currentBindOptions);
  setBindResult(libBind);

  const scan = scanProgram(allStatements);
  setParentModulesReadInFunction(scan.parentModulesInFunction);

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
      code: await emitLibraryFile(rel, outRel, program, {
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

  setGlobalVarDeclKeyword('let');
  return {manifest, files};
}

async function emitLibraryFile(
    sourceRel: string,
    outRel: string,
    program: Program,
    ctx: {
      deps: string[]; outRelOf: (sourceRel: string) => string;
      runtimePath: string;
      scope?: Scope | undefined;
    },
    ): Promise<string> {
  // Reset per-file emitter state
  setCurrentScope(ctx.scope);
  resetTailTemps();
  dynamicScopeVars.clear();
  encounteredFonts.clear();
  encounteredSurfaceData.clear();
  setCurrentRuntimePath(ctx.runtimePath);
  setMainFilename(program.filename ?? '');
  setCurrentSourceFilename(currentMainFilename);

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
    declMap.set(key, {stmt, code: await compileDeclaration(stmt)});
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

  // A reference reaches another file only if it resolves to a binding that
  // another file declares Builtins and unresolved names are never imported
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

  // Fallback for names that aren't local, imported, builtin, or special.
  // OpenSCAD resolves unknown reads to `undef`
  const referenced = fileScan.unresolved;
  // Names declared by this file, using their emitted names. Everything else
  // needs an `undefined` fallback
  const declaredHere = (names: Set<string>, ns: Namespace) => [...names].map(
      n => globalJsName(n, ns, currentScope));
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
  for (const name of undefinedNames)
    body += `let ${name}: undefined = undefined;\n`;
  body += '\n';
  if (declarations.length) body += declarations.join('\n') + '\n';
  if (exportedSymbols.length)
    body += `\nexport { ${exportedSymbols.join(', ')} };\n`;

  const out = sideEffectBlock + importBlock + builtinConstantsFor(body) + body;
  return formatCode(buildRuntimeImport(ctx.runtimePath, out) + out);
}

// Relative ES import specifier from one output file to another
export function relImportSpecifier(
    fromOutRel: string, toOutRel: string): string {
  let rel =
      path.relative(path.dirname(fromOutRel), toOutRel).replace(/\\/g, '/');
  rel = rel.replace(/\.ts$/i, '.js');
  if (!rel.startsWith('.')) rel = './' + rel;
  return rel;
}

// Emit a `use`d file in its own IIFE so its variables stay private while
// exported modules/functions close over them
export async function compileUsedFileScope(scope: ScopeStmt, unitName: string):
    Promise<{code: string; exports: string[]}> {
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

  return await (async () => {
    const declCode = new Map<string, string>();
    const declOrder: string[] = [];
    const exports: string[] = [];
    const savedSourceFilename = currentSourceFilename;

    for (const s of scope.statements) {
      if (s.filename) setCurrentSourceFilename(s.filename);
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
      declCode.set(key, await compileDeclaration(s, {assignmentOnly: true}));
    }

    setCurrentSourceFilename(savedSourceFilename);

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
