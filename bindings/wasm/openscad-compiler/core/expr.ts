import path from 'path';

import {walk} from './ast.js';
import type {Argument, ASTNode, BinaryExpr, Expr, FunctionCallExpr, KindedNode, ListCompGenerator, Parameter,} from './ast.js';
import {isLexicalVar} from './binder.js';
import {BUILTIN_FUNCTIONS, EXPERIMENTAL_BUILTIN_FUNCTIONS, NUMERIC_BUILTINS,} from './builtins.js';
import {bindJsName, escapeName, svTarget, T} from './naming.js';
import {currentMainFilename, noArgDemotions, RT, signatures} from './state.js';
import type {Binding, CallRef} from './types.js';

// Type for folded initializers; use a specific type only when certain,
// otherwise `any`
type DeclaredType = 'number'|'string'|'boolean'|'any';

interface SlotUse {
  reads: Set<string>;
  callsUserCode: boolean;
}

const MAX_FOR_ITERATIONS = 1000000;

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

// Operators whose emitted form is a boolean
const ALWAYS_BOOLEAN_OPS = new Set(['==', '!=', '&&', '||']);

// OpenSCAD's == and != on two numbers are exact identity, which is === in JS
const NATIVE_OP_SPELLING: Record<string, string> = {
  '==': '===',
  '!=': '!=='
};

export function locTag(node: ASTNode): string {
  if (!node.loc) return '';
  const s = node.loc.start;
  return ` @${s.line}:${s.column}`;
}

export function compileArgList(key: string, args: Argument[]): string {
  const sig = signatures.get(key);
  if (!sig) {
    return args
        .map(
            a => a.name ? `/* ${a.name} = */ ${compileExpr(a.value)}` :
                          compileExpr(a.value))
        .join(', ');
  }

  // Missing args use a sentinel for defaults; explicit `undef` stays unchanged.
  // Demoted slots use JS defaults, which `undefined` triggers automatically
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

export function inferDeclaredType(expr: Expr): DeclaredType {
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
export function namesNeedingPredeclaration(slots: {name: string; expr: Expr}[]):
    Set<string> {
  const needed = new Set<string>();
  const uses = slots.map(s => analyzeSlotExpr(s.expr));
  const firstUserCall = uses.findIndex(u => u.callsUserCode);
  const unsafeFrom = firstUserCall < 0 ? slots.length : firstUserCall;
  // A name may be assigned more than once, but the declaration goes at its
  // first assignment even when only the final value survives
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
export function compileExpr(expr: Expr): string {
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

export function compileCallExpr(
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
export function findArg(
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
