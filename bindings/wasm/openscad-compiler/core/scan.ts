// Whole-program scan: declarations, signatures and no-arg slot analysis

import {someNode, walk} from './ast.js';
import type {Expr, FunctionCallExpr, KindedNode, ModuleCallStmt, Parameter, Statement,} from './ast.js';
import {BUILTIN_MODULES} from './builtins.js';
import {collectStringLiterals, isFontRelatedName} from './fonts.js';
import {bindJsName, escapeName} from './naming.js';
import {localDecls, noArgDemotions, signatures} from './state.js';
import {resolveArgsToParams} from './tailcall.js';
import type {Binding, FunctionDeclStmtType, ModuleDeclStmtType, ProgramScan, ScanOptions,} from './types.js';


// Variable kinds whose every write is visible in the program being compiled
const ANALYZABLE_VAR_KINDS =
    new Set<string>(['global', 'filePrivate', 'local', 'let']);


// Generic whole-program scan
export function scanProgram(
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

// True when an expression references the identifier name anywhere within it
export function nodeReferencesIdentifier(
    node: KindedNode|undefined, name: string): boolean {
  return someNode(node, n => n.kind === 'identifier' && n.name === name);
}

export function paramUsesNoArg(p: Parameter): boolean {
  return !!p.defaultValue && !p.name.startsWith('$') &&
      !nodeReferencesIdentifier(p.defaultValue, p.name);
}

export function slotUsesNoArg(key: string, p: Parameter, i: number): boolean {
  return paramUsesNoArg(p) && !noArgDemotions.get(key)?.[i];
}

// Slots that could still use the __NO_ARG prologue, keyed by signature
export function openNoArgSlots(): Map<string, boolean[]> {
  const open = new Map<string, boolean[]>();
  for (const [key, decl] of localDecls) {
    const sig = signatures.get(key);
    if (decl.count !== 1 || sig?.params.length !== decl.params.length) continue;
    // Duplicate parameters shift slots between the signature and emitted list
    if (new Set(decl.params.map(p => p.name)).size !== decl.params.length)
      continue;
    const slots = decl.params.map(
        (p, i) => paramUsesNoArg(p) &&
            // A default can read a later parameter in the prologue, but JS
            // defaults would hit its temporal dead zone
            !decl.params.slice(i + 1).some(
                q => nodeReferencesIdentifier(p.defaultValue, q.name)));
    if (slots.some(Boolean)) open.set(key, slots);
  }
  return open;
}

// One call site's effect on the open slots of the signature it targets
export function demoteNoArgSlots(
    node: FunctionCallExpr|ModuleCallStmt, open: Map<string, boolean[]>): void {
  const key = `${node.kind === 'call' ? 'fn' : 'mod'}:${node.name}`
  const slots = open.get(key);
  if (!slots) return;
  const supplied = resolveArgsToParams(node.args, localDecls.get(key)!.params);
  supplied.forEach((arg, i) => {
    if (arg && !isDefinitelyDefined(arg, new Set())) slots[i] = false;
  });
}

// True if every declaration writing `b` satisfies `pred`. `seen` prevents
// cycles like `a = b; b = a;`
export function everyDeclValue(
    b: Binding|null|undefined, seen: Set<Binding>,
    pred: (e: Expr, seen: Set<Binding>) => boolean): boolean {
  if (!b || seen.has(b) || !ANALYZABLE_VAR_KINDS.has(b.kind)) return false;
  if (b.decls.length === 0) return false;
  seen.add(b);
  return b.decls.every(d => {
    // Parameters and loop variables have no single value expression to inspect
    const value = (d as {value?: Expr}).value;
    return !!value && pred(value, seen);
  });
}

// Expressions guaranteed to produce a defined value. Others may be `undef`
export function isDefinitelyDefined(e: Expr, seen: Set<Binding>): boolean {
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

export function isDefinitelyNumber(e: Expr, seen: Set<Binding>): boolean {
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

// Visits all function/module declarations reachable from declarative statement
// positions
export function forEachDeclaration(
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

export function recordSignature(decl: FunctionDeclStmtType|ModuleDeclStmtType):
    void {
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
export function collectDeclarations(stmts: Statement[]):
    Map<string, ModuleDeclStmtType> {
  const modules = new Map<string, ModuleDeclStmtType>();
  forEachDeclaration(stmts, decl => {
    recordSignature(decl);
    if (decl.kind === 'moduleDecl') modules.set(decl.name, decl);
  });
  return modules;
}
