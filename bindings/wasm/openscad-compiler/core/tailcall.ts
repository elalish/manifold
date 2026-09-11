import path from 'path';

import {someNode} from './ast.js';
import type {Argument, Expr, FunctionCallExpr, FunctionDeclStmt, KindedNode, Parameter, Statement,} from './ast.js';
import {compileCallExpr, compileExpr} from './expr.js';
import {bindJsName, T} from './naming.js';
import {currentMainFilename, nextTailTemp, RT, userFunctionDefs,} from './state.js';

// Depth limit for constant evaluation to avoid stack overflow
const CONST_EVAL_DEPTH_CAP = 4000;
const CONST_UNKNOWN = Symbol('const-unknown');
const CONST_DIVERGE = Symbol('const-diverge');


function containsCallTo(node: KindedNode|undefined, name: string): boolean {
  return someNode(node, n => n.kind === 'call' && n.name === name);
}

// Evaluates constant expressions. Returns the value, `CONST_UNKNOWN` if
// statically unknown, or `CONST_DIVERGE` if recursion exceeds the limit
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
export function reportDivergentCalls(
    calls: FunctionCallExpr[], defs: Map<string, FunctionDeclStmt>): void {
  userFunctionDefs.clear();
  for (const [k, v] of defs) userFunctionDefs.set(k, v);
  // Cache each function's termination check since multiple calls may share its
  // body
  const runsAway = new Map<string, boolean>();
  for (const expr of calls) {
    const fn = userFunctionDefs.get(expr.name);
    if (!fn) continue;
    let risky = runsAway.get(expr.name);
    if (risky === undefined) {
      // Only non-tail self-recursion can overflow the stack; tail calls become
      // loops
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
export function deduplicateParams(params: Parameter[]): Parameter[] {
  const seen = new Map<string, number>();
  for (let i = 0; i < params.length; i++) {
    seen.set(params[i]!.name, i);
  }
  return params.filter((p, i) => seen.get(p.name) === i);
}

// Tail-recursion elimination

// True when expression can reach a tail call to funcName in loop-lowered
// positions
export function hasSelfTailCall(expr: Expr, funcName: string): boolean {
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
export function tailAlwaysRecurses(expr: Expr, funcName: string): boolean {
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

// True if statement always calls moduleName, meaning recursion has no base case
export function moduleAlwaysRecurses(
    stmt: Statement, moduleName: string): boolean {
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
export function resolveArgsToParams(
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
      const tmp = `${RT.tc}${nextTailTemp()}`;
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
export function emitTailBody(
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
          const tmp = `${T('tl')}${nextTailTemp()}`;
          // Lambdas see their own binding, enabling recursive `let`-bound
          // lambdas
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
