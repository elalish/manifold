import path from 'path';

import type {Argument, ASTNode, BlockStmt, Expr, ForStmt, ForVariable, IfStmt, ModuleCallStmt, Parameter, Statement} from './ast.js';
import {shadowsOuterVar} from './binder.js';
import {BUILTIN_VAR_CONSTANTS} from './builtins.js';
import {compileArgList, compileExpr, findArg, inferDeclaredType, locTag, namesNeedingPredeclaration} from './expr.js';
import {DEFAULT_FONT_SPEC} from './fonts.js';
import {bindJsName, declJsName, escapeName, svTarget, T,} from './naming.js';
import {nodeReferencesIdentifier, slotUsesNoArg} from './scan.js';
import {currentMainFilename, currentSourceFilename, dynamicScopeVars, encounteredFonts, externalModuleNames, globalVarDeclKeyword, moduleDeclRegistry, parentModulesReadInFunction, RT, signatures} from './state.js';
import {compileSurface} from './surface.js';
import {deduplicateParams, emitTailBody, hasSelfTailCall, moduleAlwaysRecurses, tailAlwaysRecurses} from './tailcall.js';
import type {Binding, ModuleDeclStmtType} from './types.js';

export const GEOMETRY_TYPE =
    'InstanceType<typeof Manifold | typeof CrossSection>';

export function leadingCommentLines(
    node: ASTNode|undefined, indent = ''): string[] {
  return (node?.leadingComments ?? [])
      .flatMap(
          comment =>
              comment.value.split(/\r?\n/).map(line => `${indent}${line}`));
}

export function trailingCommentText(node: ASTNode|undefined): string {
  const comments = node?.trailingComments ?? [];
  if (comments.length === 0) return '';
  return ` ${
      comments.map(comment => comment.value.replace(/\r?\n/g, ' ')).join(' ')}`;
}

export function returnExpr(expr: string, indent = ''): string {
  const trimmed = expr.trim();
  if (trimmed.startsWith('//') || trimmed.startsWith('/*')) {
    return `(\n${expr}\n${indent})`;
  }
  return expr;
}

export function pushCommentedLine(
    lines: string[], node: ASTNode, line: string, indent = ''): void {
  lines.push(...leadingCommentLines(node, indent));
  lines.push(`${line}${trailingCommentText(node)}`);
}

export function emitArgs(...args: string[]): string {
  let end = args.length;
  while (end > 0 && args[end - 1] === 'undefined') end--;
  return args.slice(0, end).join(', ');
}

// Declarations
export const PRE_DECLARED_VARS = new Set([
  '$fn', '$fa', '$fs', '$vpr', '$vpt', '$vpd', '$vpf', '$parent_modules', '$t',
  '$preview', '$color', '$idx', ...BUILTIN_VAR_CONSTANTS
]);

export async function compileDeclaration(
    stmt: Statement, opts?: {assignmentOnly?: boolean}): Promise<string> {
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
      const body = await compileModuleBody(
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
export function emitNoArgDefaults(
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

// Module body compilation
export async function compileModuleBody(
    body: Statement, moduleName?: string, localParamNames: string[] = [],
    dollarParamNames: string[] = [],
    renamedParamNames: string[] = []): Promise<string> {
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
        const decl = await compileDeclaration(s);
        decls.push('  ' + decl.split('\n').join('\n  '));
        if (s.kind === 'functionDecl') scanned.push(decl);
      } else {
        const geo = await compileGeometry(s);
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

export function hasBackgroundModifier(stmt: Statement): boolean {
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

export function isModuleCallBackgroundOnly(
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

export async function compileGeometry(stmt: Statement): Promise<string> {
  const modifier = (stmt as {modifier?: string}).modifier;
  if (typeof modifier === 'string' && modifier.includes('*'))
    return '';  // disable modifier: subtree is ignored
  const geo = await compileGeometryDispatch(stmt);
  if (geo && typeof modifier === 'string' && modifier.includes('!')) {
    return `${RT.rootMod}(${geo})`;
  }
  return geo;
}

export async function compileGeometryDispatch(stmt: Statement):
    Promise<string> {
  switch (stmt.kind) {
    case 'moduleCall':
      return await compileModuleCall(stmt);
    case 'block':
      return await compileBlockGeometry(stmt);
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
async function compileModuleCall(stmt: ModuleCallStmt): Promise<string> {
  const dollarArgs =
      stmt.args.filter(arg => arg.name && arg.name.startsWith('$'));
  const userSig = signatures.get(`mod:${stmt.name}`);
  const extraArgs = (moduleDeclRegistry.has(stmt.name) && userSig) ?
      stmt.args.filter(
          a => a.name && !a.name.startsWith('$') &&
              !userSig.params.includes(a.name)) :
      [];

  let result: string;
  // User/library modules can shadow builtins. A library wrapper can still
  // access the builtin because `use` resolves within its own scope
  if (stmt.ref?.mod && stmt.ref.mod.kind !== 'builtin') {
    result = await compileUserModuleCall(stmt);
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
        result = await compileSurface(
            stmt.args, currentSourceFilename || currentMainFilename);
        break;

      // Transforms
      case 'translate':
        result = await compileTransform(stmt, 'translate');
        break;
      case 'rotate':
        result = await compileTransform(stmt, 'rotate');
        break;
      case 'scale':
        result = await compileTransform(stmt, 'scale');
        break;
      case 'mirror':
        result = await compileMirror(stmt);
        break;
      case 'multmatrix':
        result = await compileMultMatrix(stmt);
        break;
      case 'resize':
        result = await compileResize(stmt);
        break;
      case 'offset':
        result = await compileOffset(stmt);
        break;
      case 'color':
        result = await compileColor(stmt);
        break;
      case 'render':
        result = await compilePassthrough(stmt, 'render');
        break;
      case 'projection':
        result = await compileProjection(stmt);
        break;

      // Boolean operations
      case 'group':
        result = await compileBoolOp(stmt, 'union');
        break;  // group() == implicit union
      case 'union':
        result = await compileBoolOp(stmt, 'union');
        break;
      case 'difference':
        result = await compileDifference(stmt);
        break;
      case 'intersection':
        result = await compileBoolOp(stmt, 'intersection');
        break;
      case 'hull':
        result = await compileBoolOp(stmt, 'hull');
        break;
      case 'minkowski':
        result = await compileMinkowski(stmt);
        break;

      // Extrusion
      case 'linear_extrude':
        result = await compileLinearExtrude(stmt);
        break;
      case 'rotate_extrude':
        result = await compileRotateExtrude(stmt);
        break;

      // Builtin statement modifiers
      case 'echo':
        result = await compileEchoModule(stmt);
        break;
      case 'assert':
        result = await compileAssertModule(stmt);
        break;
      case 'let':
        result = await compileLetModule(stmt);
        break;
      case 'children':
        result = compileChildrenModule(stmt);
        break;
      case 'intersection_for':
        result = await compileIntersectionFor(stmt);
        break;

      default:
        result = await compileUserModuleCall(stmt);
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
async function compileEchoModule(stmt: ModuleCallStmt): Promise<string> {
  const args = stmt.args
                   .map(
                       a => a.name ? `(${JSON.stringify(a.name + ' = ')} + ${
                                         RT.oecho}(${compileExpr(a.value)}))` :
                                     `${RT.oecho}(${compileExpr(a.value)})`)
                   .join(', ');
  if (stmt.child && stmt.child.kind !== 'empty') {
    const child = await compileGeometry(stmt.child);
    return `(${RT.echo}(${args}), ${child || 'Manifold.union([])'})`;
  }
  return `(${RT.echo}(${args}), Manifold.union([]))`;
}

async function compileAssertModule(stmt: ModuleCallStmt): Promise<string> {
  const condition = stmt.args[0] ? compileExpr(stmt.args[0].value) : 'true';
  const message =
      stmt.args[1] ? compileExpr(stmt.args[1].value) : '"Assertion failed"';
  if (stmt.child && stmt.child.kind !== 'empty') {
    const child = await compileGeometry(stmt.child);
    return `(${RT.assert}(${condition}, ${message}), ${
        child || 'Manifold.union([])'})`;
  }
  return `(${RT.assert}(${condition}, ${message}), Manifold.union([]))`;
}

async function compileLetModule(stmt: ModuleCallStmt): Promise<string> {
  let child = 'Manifold.union([])';
  if (stmt.child && stmt.child.kind !== 'empty') {
    child = await compileGeometry(stmt.child) || child;
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
async function compileTransform(
    stmt: ModuleCallStmt,
    method: string,
    ): Promise<string> {
  if (!stmt.child) return 'Manifold.union([])';

  const child = await compileGeometry(stmt.child);
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

async function compileMirror(stmt: ModuleCallStmt): Promise<string> {
  if (!stmt.child) return 'Manifold.union([])';
  const child = await compileGeometry(stmt.child);
  const vec = stmt.args[0];
  const vecStr = vec ? compileExpr(vec.value) : '[1, 0, 0]';
  return `${RT.mirror}(${child}, ${vecStr})`;
}

async function compileMultMatrix(stmt: ModuleCallStmt): Promise<string> {
  if (!stmt.child) return 'Manifold.union([])';
  const child = await compileGeometry(stmt.child);
  const mat = stmt.args[0];
  if (!mat) return `${child}`;
  return `${RT.transform}(${child}, ${compileExpr(mat.value)})`;
}

async function compileColor(stmt: ModuleCallStmt): Promise<string> {
  if (!stmt.child) return 'Manifold.union([])';
  const child = await compileGeometry(stmt.child);
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

async function compileResize(stmt: ModuleCallStmt): Promise<string> {
  if (!stmt.child) return 'Manifold.union([])';
  const child = await compileGeometry(stmt.child);
  const newsize = findArg(stmt.args, 'newsize', 0);
  const auto = findArg(stmt.args, 'auto', 1);
  const ns = newsize ? compileExpr(newsize.value) : 'undefined';
  const au = auto ? compileExpr(auto.value) : 'undefined';
  return `${RT.resize}(${emitArgs(child, ns, au)})`;
}

async function compilePassthrough(
    stmt: ModuleCallStmt, tag: string): Promise<string> {
  if (!stmt.child) return 'Manifold.union([])';
  return `/* ${tag}(${
      stmt.args.map(a => compileExpr(a.value)).join(', ')}) */ ${
      await compileGeometry(stmt.child)}`;
}

async function compileOffset(stmt: ModuleCallStmt): Promise<string> {
  if (!stmt.child) return 'CrossSection.square(0)';
  const child = await compileGeometry(stmt.child);
  const r = findArg(stmt.args, 'r', 0);
  const delta = findArg(stmt.args, 'delta');
  const amount = r ?? delta;
  const amt = amount ? compileExpr(amount.value) : '0';
  return `${RT.offset}(${child}, ${amt}, "Round", 2, ${RT.ctx}.$fn, ${
      RT.ctx}.$fa, ${RT.ctx}.$fs)`;
}

async function compileProjection(stmt: ModuleCallStmt): Promise<string> {
  if (!stmt.child) return 'CrossSection.square(0)';
  const child = await compileGeometry(stmt.child);
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

async function collectChildrenWithDecls(
    stmt: ModuleCallStmt, sideEffectsAsChildren = false): Promise<{
  decls: string[]; geos: string[]; dollars: {name: string; code: string}[]
}> {
  if (!stmt.child) return {decls: [], geos: [], dollars: []};
  if (stmt.child.kind === 'block') {
    return compileBlockStatementsWithDecls(
        stmt.child.statements, sideEffectsAsChildren);
  }
  if (hasBackgroundModifier(stmt.child))
    return {decls: [], geos: [], dollars: []};
  if (isSideEffectOnlyModule(stmt.child) && !sideEffectsAsChildren) {
    return {
      decls: [`${await compileGeometry(stmt.child)};`],
      geos: [],
      dollars: []
    };
  }
  const g = await compileGeometry(stmt.child);
  return {decls: [], geos: g ? [g] : [], dollars: []};
}

async function compileBlockStatementsWithDecls(
    stmts: Statement[], sideEffectsAsChildren = false): Promise<{
  decls: string[]; geos: string[]; dollars: {name: string; code: string}[]
}> {
  const varDecls = new Map < string, {
    stmt: Statement&{kind: 'variableDecl'};
    order: number
  }
  >();
  const otherDecls: string[] = [];
  const geos: string[] = [];
  const dollars: {name: string; code: string}[] = [];
  let order = 0;

  const collect = async (list: Statement[]) => {
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
        otherDecls.push(await compileDeclaration(s));
      } else if (s.kind === 'block') {
        await collect(s.statements);
      } else if (isSideEffectOnlyModule(s) && !sideEffectsAsChildren) {
        // Run echo()/assert() for their side effects, but keep them out of geos
        otherDecls.push(`${await compileGeometry(s)};`);
      } else {
        const g = await compileGeometry(s);
        if (g) {
          const leading = leadingCommentLines(s);
          geos.push(`${leading.length ? `${leading.join('\n')}\n` : ''}${g}`);
        }
      }
    }
  };
  await collect(stmts);

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

async function compileBoolOp(
    stmt: ModuleCallStmt, op: string): Promise<string> {
  const {decls, geos} = await collectChildrenWithDecls(stmt);
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

async function compileDifference(stmt: ModuleCallStmt): Promise<string> {
  const {decls, geos} = await collectChildrenWithDecls(stmt);
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

async function compileMinkowski(stmt: ModuleCallStmt): Promise<string> {
  const {decls, geos} = await collectChildrenWithDecls(stmt);
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

async function compileLinearExtrude(stmt: ModuleCallStmt): Promise<string> {
  if (!stmt.child) return 'Manifold.union([])';
  const child = await compileGeometry(stmt.child);
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

async function compileRotateExtrude(stmt: ModuleCallStmt): Promise<string> {
  if (!stmt.child) return 'Manifold.union([])';
  const child = await compileGeometry(stmt.child);
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
export async function compileBlockGeometry(block: BlockStmt): Promise<string> {
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
        items.push({kind: 'func', code: await compileDeclaration(s)});
      } else {
        const g = await compileGeometry(s);
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

async function compileIntersectionFor(stmt: ModuleCallStmt): Promise<string> {
  if (!stmt.child) return 'Manifold.union([])';

  const variables: ForVariable[] = stmt.args.map(arg => ({
                                                   name: arg.name || '_',
                                                   range: arg.value,
                                                   loc: arg.loc,
                                                 }));

  const lines = [
    '(() => {',
    `  const ${T('items')}: ${GEOMETRY_TYPE}[] = [];`,
    ...await buildNestedForStatements(variables, 0, stmt.child, 1),
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
async function compileForGeometry(stmt: ForStmt): Promise<string> {
  if (stmt.variables.length === 0) return 'Manifold.union([])';
  const lines = [
    '(() => {',
    `  const ${T('items')}: ${GEOMETRY_TYPE}[] = [];`,
    ...await buildNestedForStatements(stmt.variables, 0, stmt.body, 1),
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

async function buildNestedForStatements(
    vars: ForVariable[],
    idx: number,
    body: Statement,
    indentLevel: number,
    ): Promise<string[]> {
  const indent = '  '.repeat(indentLevel);
  if (idx >= vars.length) {
    const lines: string[] = [];
    const geo = await compileGeometry(body);
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
      ...await buildNestedForStatements(vars, idx + 1, body, indentLevel + 2),
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
    ...await buildNestedForStatements(vars, idx + 1, body, indentLevel + 3),
    `${indent}    } finally { ${RT.ctx}.$idx = ${T('save_$idx')}; }`,
    `${indent}  }`,
    `${indent}}`,
  ];
}

async function compileIfGeometry(stmt: IfStmt): Promise<string> {
  const cond = `${RT.truthy}(${compileExpr(stmt.condition)})`;
  const then = await compileGeometry(stmt.thenBody);
  if (stmt.elseBody) {
    const els = await compileGeometry(stmt.elseBody);
    const lines = [
      '(() => {',
      `  if (${cond}) {`,
    ];
    pushCommentedLine(
        lines, stmt.thenBody, `    return ${returnExpr(then, '    ')};`,
        '    ');
    lines.push('  } else {');
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
async function compileUserModuleCall(stmt: ModuleCallStmt): Promise<string> {
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
      await collectChildrenWithDecls(stmt, true) :
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
