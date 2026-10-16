// Choose the JS name for each binding after resolution and runtime names are
// known

import {lookup, NAMESPACES} from './binder.js';
import {BUILTIN_VAR_CONSTANTS, RUNTIME_BUILTIN_FUNCTIONS, RUNTIME_BUILTIN_MODULES, RUNTIME_SYMBOLS,} from './builtins.js';
import {RT, setRT, tempNames, unitTakenNames} from './state.js';
import type {Binding, BindResult, Namespace, PrettyNameOptions, ProgramScan, Scope,} from './types.js';

// JavaScript reserved words
const JS_RESERVED = new Set([
  'abstract',     'arguments', 'await',    'boolean',    'break',
  'byte',         'case',      'catch',    'char',       'class',
  'const',        'continue',  'debugger', 'default',    'delete',
  'do',           'double',    'else',     'enum',       'eval',
  'export',       'extends',   'false',    'final',      'finally',
  'float',        'for',       'function', 'goto',       'if',
  'implements',   'import',    'in',       'instanceof', 'int',
  'interface',    'let',       'long',     'native',     'new',
  'null',         'package',   'private',  'protected',  'public',
  'return',       'short',     'static',   'super',      'switch',
  'synchronized', 'this',      'throw',    'throws',     'transient',
  'true',         'try',       'typeof',   'var',        'void',
  'volatile',     'while',     'with',     'yield',
]);

const RUNTIME_NS = 'rt';

// Conflicting names use the highest-priority namespace for stable output
const NS_PRIORITY: Record<Namespace, number> = {
  var : 0,
  fn: 1,
  mod: 2
};

// Smallest edit that makes an OpenSCAD name a legal JS identifier
export function escapeName(name: string): string {
  if (JS_RESERVED.has(name)) return `${name}_`;
  if (/^[0-9]/.test(name)) return `_${name}`;
  return name;
}

export function assignPrettyNames(
    result: BindResult, opts: PrettyNameOptions): void {
  const reserved = new Set(opts.reserved);
  const visit =
      (scope: Scope, ancestors: Set<string>,
       ancestorCallables: Set<string>) => {
        // Names claimed in this scope; siblings must not collide with each
        // other
        const taken = new Set<string>();

        const own: Binding[] = [];
        for (const ns of NAMESPACES)
          for (const b of scope.bindings[ns].values())
            if (b.scope === scope) own.push(b);
        own.sort(
            (a, b) => NS_PRIORITY[a.ns] - NS_PRIORITY[b.ns] || a.id - b.id);

        for (const b of own) {
          if (b.kind === 'special') {
            b.jsName = escapeName(b.name);
            continue;
          }
          // Builtins and library exports keep their existing runtime or library
          // names
          if (b.kind === 'builtin' || b.kind === 'external') {
            b.jsName = fixedName(b, opts);
            // Only names that actually appear in the output can be clashed with
            if (b.ns !== 'mod') taken.add(b.jsName);
            continue;
          }
          b.jsName = pick(b, taken, ancestors, ancestorCallables, reserved);
          taken.add(b.jsName);
        }

        const below = new Set([...ancestors, ...taken]);
        const callablesBelow = new Set(ancestorCallables);
        for (const ns of NAMESPACES) {
          if (ns === 'var') continue;
          for (const b of scope.bindings[ns].values())
            callablesBelow.add(b.jsName);
        }
        for (const child of scope.children) visit(child, below, callablesBelow);
      };
  visit(result.root, new Set(), new Set());
}

function fixedName(b: Binding, opts: PrettyNameOptions): string {
  const key = `${b.ns}:${b.name}`;
  if (b.kind === 'builtin') {
    const fromRuntime = opts.builtinSymbols?.get(key);
    if (fromRuntime) return fromRuntime;
  } else {
    const fromLib = opts.externalSymbols?.get(key);
    if (fromLib) return fromLib;
  }
  const base = escapeName(b.name);
  return b.ns === 'fn' ? `${base}_fn` : b.ns === 'mod' ? `${base}$mod` : base;
}

function pick(
    b: Binding, taken: Set<string>, ancestors: Set<string>,
    ancestorCallables: Set<string>, reserved: Set<string>): string {
  const base = escapeName(b.name);
  // Source names can shadow outer variables, but never outer functions or
  // modules
  if (!taken.has(base) && !reserved.has(base) && !ancestorCallables.has(base))
    return base;

  const blocked = (c: string) => taken.has(c) || ancestors.has(c) ||
      ancestorCallables.has(c) || reserved.has(c);
  const suffixed = b.ns === 'fn' ? `${base}_fn` :
      b.ns === 'mod' ? `${base}_mod` :
                       base;
  if (suffixed !== base && !blocked(suffixed)) return suffixed;
  for (let n = 2;; n++) {
    const c = `${suffixed}_${n}`;
    if (!blocked(c)) return c;
  }
}

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

// Gives each runtime export a local name - uses its own name first, if it
// clashes with a user declaration, import it as `runtime_...` instead
export function resolveRuntimeLocals(blocked: ReadonlySet<string>): void {
  const claimed = new Set<string>();
  const locals: Record<string, string> = {};
  for (const name of RUNTIME_SYMBOLS) {
    let local = name;
    for (let n = 1; blocked.has(local) || claimed.has(local); n++)
      local = n === 1 ? `runtime_${name}` : `runtime_${name}_${n}`;
    claimed.add(local);
    locals[name] = local;
  }
  setRT(locals);
}

export function resetTempNames(taken: Iterable<string>): void {
  unitTakenNames.clear();
  for (const name of taken) unitTakenNames.add(name);
  tempNames.clear();
}

export function T(base: string): string {
  const memo = tempNames.get(base);
  if (memo !== undefined) return memo;
  let name = base;
  for (let n = 1; unitTakenNames.has(name); n++) name = `${base}_${n}`;
  tempNames.set(base, name);
  // so two different bases cannot converge on one name either
  unitTakenNames.add(name);
  return name;
}

export function namesInUse(
    bind: BindResult, unresolved: Set<string>): string[] {
  const used = [...reservedNames()];
  for (const b of bind.bindings) used.push(b.jsName);
  used.push(...unresolved);
  return used;
}

export function builtinSymbolNames(): Map<string, string> {
  const byKey = new Map<string, string>();
  for (const name of RUNTIME_BUILTIN_FUNCTIONS)
    if (RT[name]) byKey.set(`fn:${name}`, RT[name]!);
  for (const name of RUNTIME_BUILTIN_MODULES)
    if (RT[name]) byKey.set(`mod:${name}`, RT[name]!);
  return byKey;
}

// Names emitted as declarations and unavailable to runtime imports
// Includes declared bindings, unresolved reads, and non-$ named argument slots
export function namesBlockingRuntimeLocals(
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
// regular lexical bindings
export function svTarget(name: string): string {
  return name.startsWith('$') ? `${RT.ctx}.${name}` : name;
}

// Identifiers mentioned in code
function mentionedIdentifiers(body: string): Set<string> {
  return new Set(body.match(/(?<![\w$])(?<!(?<!\.)\.)[A-Za-z_$][\w$]*/g) ?? []);
}

// The OpenSCAD constant declarations this body needs, in their fixed order
export function builtinConstantsFor(body: string): string {
  const mentioned = mentionedIdentifiers(body);
  return builtinConstantDecls()
      .filter(([name]) => mentioned.has(name))
      .map(([, decl]) => `${decl}\n`)
      .join('');
}

export function buildRuntimeImport(runtimePath: string, body: string): string {
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

// Names the emitter or the runtime already owns, which no binding may take
export function reservedNames(): string[] {
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

function legacyJsName(name: string, ns: Namespace): string {
  const base = escapeName(name);
  return ns === 'fn' ? `${base}_fn` : ns === 'mod' ? `${base}$mod` : base;
}

// The JS name of a bound parameter, let binding or loop variable
export function bindJsName(
    node: {name: string; binding?: Binding | null | undefined}): string {
  return node.binding ? node.binding.jsName : escapeName(node.name);
}

export function globalJsName(
    name: string, ns: Namespace, scope: Scope|undefined): string {
  const b = scope ? lookup(scope, name, ns) : null;
  return b ? b.jsName : legacyJsName(name, ns);
}

export function declJsName(
    stmt: {name: string; binding?: Binding | undefined},
    ns: Namespace): string {
  if (stmt.binding) return stmt.binding.jsName;
  const base = escapeName(stmt.name);
  return ns === 'fn' ? `${base}_fn` : ns === 'mod' ? `${base}$mod` : base;
}
