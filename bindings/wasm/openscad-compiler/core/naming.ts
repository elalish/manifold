// Choose the JS name for each binding after resolution and runtime names are
// known

import {NAMESPACES} from './binder.js';
import type {Binding, BindResult, Namespace, PrettyNameOptions, Scope,} from './types.js';

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

// Smallest edit that makes an OpenSCAD name a legal JS identifier
export function escapeName(name: string): string {
  if (JS_RESERVED.has(name)) return `${name}_`;
  if (/^[0-9]/.test(name)) return `_${name}`;
  return name;
}

// Conflicting names use the highest-priority namespace for stable output
const NS_PRIORITY: Record<Namespace, number> = {
  var : 0,
  fn: 1,
  mod: 2
};

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
            // Only names that actually appear in the output can be clashed
            // with.
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
