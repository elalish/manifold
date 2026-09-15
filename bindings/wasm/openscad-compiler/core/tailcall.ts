import type {Argument, Expr, Parameter, Statement,} from './ast.js';
import {compileCallExpr, compileExpr} from './expr.js';
import {bindJsName, T} from './naming.js';
import {nextTailTemp, RT,} from './state.js';

// Tail-recursion elimination

// True when expression can reach a tail call to funcName in tail position
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

// In case of deduplicate parameters - keep last occurrence of each name
export function deduplicateParams(params: Parameter[]): Parameter[] {
  const seen = new Map<string, number>();
  for (let i = 0; i < params.length; i++) {
    seen.set(params[i]!.name, i);
  }
  return params.filter((p, i) => seen.get(p.name) === i);
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

// Emit a tail-recursive function call as a trampoline thunk
function emitSelfTailCall(
    call: Extract<Expr, {kind: 'call'}>, fnJsName: string, params: Parameter[],
    indent: string): string {
  if (call.args.some(a => a.name && a.name.startsWith('$'))) {
    // special variable args need dynamic-scope saves around the call
    return `${indent}return ${compileCallExpr(call)};`;
  }
  const provided = resolveArgsToParams(call.args, params);

  const stage: string[] = [];
  const argExprs: string[] = [];
  for (let i = 0; i < params.length; i++) {
    if (provided[i] !== undefined) {
      const tmp = `${RT.tc}${nextTailTemp()}`;
      stage.push(`${indent}const ${tmp}: any = ${compileExpr(provided[i]!)};`);
      argExprs.push(tmp);
    } else if (params[i]!.defaultValue) {
      argExprs.push(compileExpr(params[i]!.defaultValue!));
    } else {
      argExprs.push('undefined');
    }
  }
  const tcArgs = argExprs.length > 0 ? `, ${argExprs.join(', ')}` : '';
  return [
    ...stage,
    `${indent}return ${RT.tc}(${fnJsName}${tcArgs});`,
  ].join('\n');
}

// Lower tail-recursive function body to use trampoline
export function emitTailBody(
    expr: Expr, funcName: string, funcJsName: string, params: Parameter[],
    indent: string): string {
  switch (expr.kind) {
    case 'group':
      return emitTailBody(expr.expr, funcName, funcJsName, params, indent);
    case 'ternary': {
      const cond = compileExpr(expr.condition);
      const t =
          emitTailBody(expr.ifTrue, funcName, funcJsName, params, indent + '  ');
      const f = emitTailBody(
          expr.ifFalse, funcName, funcJsName, params, indent + '  ');
      return `${indent}if (${RT.truthy}(${cond})) {\n${t}\n${indent}} else {\n${
          f}\n${indent}}`;
    }
    case 'assert': {
      const condition = expr.args[0] ? compileExpr(expr.args[0].value) : 'true';
      const message =
          expr.args[1] ? compileExpr(expr.args[1].value) : '"Assertion failed"';
      return `${indent}${RT.assert}(${condition}, ${message});\n${
          emitTailBody(expr.expr, funcName, funcJsName, params, indent)}`;
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
          emitTailBody(expr.expr, funcName, funcJsName, params, indent)}`;
    }
    case 'let': {
      if (expr.assignments.some(
              a => a.name.startsWith('$') || a.name === funcName)) {
        return `${indent}return ${compileExpr(expr)};`;
      }

      return (() => {
        const savedNames = expr.assignments.map(a => a.binding?.jsName);
        const lines: string[] = [];
        
        for (const a of expr.assignments) {
          const tmp = `${T('tl')}${nextTailTemp()}`;
          if (a.value.kind === 'lambda' && a.binding) a.binding.jsName = tmp;
          lines.push(`${indent}const ${tmp}: any = ${compileExpr(a.value)};`);
          if (a.binding) a.binding.jsName = tmp;
        }
        const body =
            emitTailBody(expr.body, funcName, funcJsName, params, indent);
        expr.assignments.forEach((a, i) => {
          const saved = savedNames[i];
          if (a.binding && saved !== undefined) a.binding.jsName = saved;
        });
        return lines.join('\n') + '\n' + body;
      })();
    }
    case 'call':
      if (expr.name === funcName) {
        return emitSelfTailCall(expr, funcJsName, params, indent);
      }
      return `${indent}return ${compileExpr(expr)};`;
    default:
      return `${indent}return ${compileExpr(expr)};`;
  }
}
