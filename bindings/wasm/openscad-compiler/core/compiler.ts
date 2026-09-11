import path from 'path';

import type {Expr, Program, Statement,} from './ast.js';
import {bindProgram} from './binder.js';
import {BUILTIN_FUNCTIONS, BUILTIN_SIGNATURES} from './builtins.js';
import {namesNeedingPredeclaration,} from './expr.js';
import {formatCode} from './format.js';
import {compileDeclaration, compileGeometry, GEOMETRY_TYPE, hasBackgroundModifier, isModuleCallBackgroundOnly, PRE_DECLARED_VARS, pushCommentedLine} from './geometry.js';
import {compileUsedFileScope, declKey, isDecl} from './library.js';
import {assignPrettyNames, buildRuntimeImport, builtinConstantsFor, builtinSymbolNames, declJsName, escapeName, globalJsName, namesBlockingRuntimeLocals, namesInUse, reservedNames, resetTempNames, resolveRuntimeLocals, svTarget, T,} from './naming.js';
import {collectDeclarations, openNoArgSlots, scanProgram} from './scan.js';
import {setModuleDecls} from './state.js';
import type {Signature} from './state.js';
import {currentBindOptions, currentMainFilename, currentScope, dynamicScopeVars, externalFunctionNames, externalModuleNames, externalVariableNames, globalVarDeclKeyword, localDecls, moduleDeclRegistry, noArgDemotions, resetTailTemps, RT, setBindResult, setCurrentRuntimePath, setCurrentScope, setCurrentSourceFilename, setMainFilename, setParentModulesReadInFunction, signatures} from './state.js';
import {reportDivergentCalls} from './tailcall.js';
import type {CompileOptions, ModuleDeclStmtType} from './types.js';


// Path used in emitted `// <source>` comments, anchored to the entry file so
// output stays consistent regardless of the working directory
function sourceComment(filename: string): string {
  const base = currentMainFilename ? path.dirname(currentMainFilename) : '';
  const rel = base ? path.relative(base, filename) : filename;
  return rel.replace(/\\/g, '/');
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

async function processStmt(
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
    signatures: Map<string, Signature>): Promise<void> {
  if (stmt.kind === 'empty') return;
  if (stmt.filename) setCurrentSourceFilename(stmt.filename);
  // `{}` doesn't create a new scope - merge its assignments into
  // the enclosing scope
  if (stmt.kind === 'block') {
    for (const s of stmt.statements)
      await processStmt(
          s, scopeUnits, declMap, declOrder, lastGeoFilename, geometryLines,
          predeclared, moduleDeclRegistry, signatures);
    return;
  }
  // A `use`d file is a scope: its variables stay inside
  // and only its modules and functions are published, as forwarders so they
  // dedupe against the consumer's own declarations of the same name
  if (stmt.kind === 'scope') {
    const unit = `${T(`scope${scopeUnits.length}`)}`;
    const {code, exports} = await compileUsedFileScope(stmt, unit);
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
    declMap.set(key, {
      stmt,
      code: await compileDeclaration(stmt, {assignmentOnly: hoisted})
    });
  } else if (stmt.kind === 'use' || stmt.kind === 'include') {
    const key = `comment:${stmt.path}`;
    if (!declMap.has(key)) declOrder.push(key);
    declMap.set(key, {stmt, code: `// ${stmt.kind} <${stmt.path}>`});
  } else {
    const geo = await compileGeometry(stmt);
    if (geo) {
      const filename = stmt.filename;
      if (filename && filename !== lastGeoFilename) {
        geometryLines.push(`\n// ${sourceComment(filename)}`);
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
export async function compile(
    program: Program, options?: CompileOptions): Promise<string> {
  setCurrentRuntimePath(options?.runtimePath ?? './runtime/runtime.js');
  setMainFilename(program.filename ?? '');
  setCurrentSourceFilename(currentMainFilename);
  dynamicScopeVars.clear();
  externalModuleNames.clear();
  externalFunctionNames.clear();
  externalVariableNames.clear();
  resetTailTemps();
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
  setModuleDecls(collectDeclarations(program.statements));

  const bind = bindProgram(program, currentBindOptions);
  setBindResult(bind);

  const openSlots = openNoArgSlots();

  const scan = scanProgram(
      program.statements, {noArgSlots: openSlots, divergence: true});
  for (const [key, slots] of openSlots) {
    if (slots.some(Boolean)) noArgDemotions.set(key, slots);
  }
  setParentModulesReadInFunction(scan.parentModulesInFunction);

  resolveRuntimeLocals(namesBlockingRuntimeLocals(bind, scan, externalSymbols));
  assignPrettyNames(bind, {
    reserved: reservedNames(),
    externalSymbols,
    builtinSymbols: builtinSymbolNames(),
  });

  resetTempNames(namesInUse(bind, scan.unresolved));
  setCurrentScope(bind.global);

  // Reject top-level constant-argument calls to non-tail recursive functions
  reportDivergentCalls(scan.divergenceCandidates, scan.functionDefs);

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
    await processStmt(
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
      declarations.push(`\n// ${sourceComment(filename)}`);
      lastFilename = filename;
    }
    declarations.push(entry.code);
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
        if (file)
          addImp(
              lib.importSpecifierFor(file),
              globalJsName(m, 'mod', currentScope));
      }
      for (const f of refs.functions) {
        const file = lib.manifest.exports.functions[f];
        if (file)
          addImp(
              lib.importSpecifierFor(file),
              globalJsName(f, 'fn', currentScope));
      }
      for (const v of refs.variables) {
        const file = lib.manifest.exports.variables[v];
        if (file)
          addImp(
              lib.importSpecifierFor(file),
              globalJsName(v, 'var', currentScope));
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
    output += `let ${name}: undefined = undefined;\n`;
  }

  if (declarations.length) {
    output += (output ? '\n' : '') + declarations.join('\n') + '\n\n';
  }

  // children() used outside a module's scope - warns and yields nothing. The
  // children stack is empty at top level, so this resolves to empty geometry
  if (scan.topLevelChildren) {
    output += `function children(i?: any): ${GEOMETRY_TYPE} { const ${
        T('c')}: any = ${RT.children_stack}.length > 0 ? ${RT.children_stack}[${
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
  return formatCode(
      buildRuntimeImport(options?.runtimePath ?? './runtime/runtime.js', full) +
      full);
}
