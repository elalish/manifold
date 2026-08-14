// Runs between resolve and emit. Walks the AST building a scope tree and
// creates one `Binding` object per declared name, then points every reference
// at the binding it resolves to

import type {Expr, ForVariable, LetAssignment, ListCompGenerator, Parameter, Program, ScopeStmt, Statement,} from './ast.js';

export type Namespace = 'var'|'fn'|'mod';

export type BindingKind =
    // Provided by the runtime
    'builtin'|
    // Exported by a separately compiled library and imported
    'external'|
    // Top-level declaration of the program being compiled
    'global'|
    // Top-level variable of a `use`d file, private to that file's scope
    'filePrivate'|
    // Module/function/lambda parameter
    'param'|
    // Variable declared inside a module body or a block
    'local'|
    // `let` binding
    'let'|
    // for loop variable
    'loop'|
    // Dynamically scoped `$` variable
    'special';

export interface Binding {
  id: number;
  name: string;
  ns: Namespace;
  kind: BindingKind;
  scope: Scope;
  // Every declaration site that writes this binding. OpenSCAD scopes are
  // declarative and last-wins, so one name can have several declarations but
  // is still a single binding
  decls: (Statement|Parameter|LetAssignment|ForVariable)[];
  // Library closures only: the file whose top level declares this binding,
  // relative to the library root
  file?: string;
  jsName: string;  // Assigned by the naming pass
  reads: number;
}

export type ScopeKind = 'root'|'external'|'global'|'file'|'module'|'function'|
    'lambda'|'let'|'for'|'block'|'comprehension';

export interface Scope {
  id: number;
  kind: ScopeKind;
  parent: Scope|null;
  children: Scope[];
  bindings: Record<Namespace, Map<string, Binding>>;
}

// a call site resolves to a CallRef
export interface CallRef {
  fn: Binding|null;
  value: Binding|null;
  mod: Binding|null;
}

export interface BindOptions {
  builtinFunctions: Iterable<string>;
  builtinModules: Iterable<string>;
  builtinConstants: Iterable<string>;
  externalFunctions?: Iterable<string>;
  externalModules?: Iterable<string>;
  externalVariables?: Iterable<string>;
}

export const NAMESPACES: readonly Namespace[] = ['var', 'fn', 'mod'];

// One file of a library closure, with the directives that pull other files in
export interface LibraryFileInput {
  rel: string;
  program: Program;
  edges: {rel: string; mode: 'include' | 'use'}[];
}

export interface BindResult {
  root: Scope;
  global: Scope;
  bindings: Binding[];
  scopes: Scope[];
}

export interface LibraryBindResult extends BindResult {
  // Resolution scope for each file of the closure, keyed by its path relative
  // to the library root
  fileScopes: Map<string, Scope>;
}

function isSpecial(name: string): boolean {
  return name.startsWith('$');
}

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

export function escapeName(name: string): string {
  if (JS_RESERVED.has(name)) return `${name}_`;
  if (/^[0-9]/.test(name)) return `_${name}`;
  return name;
}

export function lookup(scope: Scope|null, name: string, ns: Namespace): Binding|
    null {
  for (let s = scope; s; s = s.parent) {
    const b = s.bindings[ns].get(name);
    if (b) return b;
  }
  return null;
}

class Binder {
  private scopeCounter = 0;
  private bindingCounter = 0;
  readonly bindings: Binding[] = [];
  readonly scopes: Scope[] = [];
  readonly root: Scope;
  // Exports of separately compiled libraries
  readonly externals: Scope;
  global!: Scope;
  private current: Scope;
  // Library closures: the file whose declarations are being hoisted
  private currentFile: string|undefined;

  constructor(opts: BindOptions) {
    this.root = this.newScope('root', null);
    this.current = this.root;
    for (const n of opts.builtinConstants) this.declare(n, 'var', 'builtin');
    for (const n of opts.builtinFunctions) this.declare(n, 'fn', 'builtin');
    for (const n of opts.builtinModules) this.declare(n, 'mod', 'builtin');

    this.externals = this.newScope('external', this.root);
    this.current = this.externals;
    for (const n of opts.externalVariables ?? [])
      this.declare(n, 'var', 'external');
    for (const n of opts.externalFunctions ?? [])
      this.declare(n, 'fn', 'external');
    for (const n of opts.externalModules ?? [])
      this.declare(n, 'mod', 'external');
    this.current = this.root;
  }

  private newScope(kind: ScopeKind, parent: Scope|null): Scope {
    const scope: Scope = {
      id: this.scopeCounter++,
      kind,
      parent,
      children: [],
      bindings: {var : new Map(), fn: new Map(), mod: new Map()},
    };
    parent?.children.push(scope);
    this.scopes.push(scope);
    return scope;
  }

  private withScope<T>(kind: ScopeKind, fn: (scope: Scope) => T): T {
    const saved = this.current;
    this.current = this.newScope(kind, saved);
    try {
      return fn(this.current);
    } finally {
      this.current = saved;
    }
  }

  // Declare `name` in the current scope, or return the existing binding if the
  // name is already declared there
  private declare(
      name: string, ns: Namespace, kind: BindingKind,
      decl?: Statement|Parameter|LetAssignment|ForVariable,
      scope: Scope = this.current): Binding {
    const existing = scope.bindings[ns].get(name);
    if (existing) {
      if (decl) existing.decls.push(decl);
      // Last declaration wins ownership, matching the export map
      if (decl && this.currentFile !== undefined)
        existing.file = this.currentFile;
      return existing;
    }
    const binding: Binding = {
      id: this.bindingCounter++,
      name,
      ns,
      kind,
      scope,
      decls: decl ? [decl] : [],
      ...(decl && this.currentFile !== undefined ? {file: this.currentFile} :
                                                   {}),
      jsName: '',
      reads: 0,
    };
    scope.bindings[ns].set(name, binding);
    this.bindings.push(binding);
    return binding;
  }

  private special(name: string): Binding {
    return this.declare(name, 'var', 'special', undefined, this.root);
  }

  private resolve(name: string, ns: Namespace): Binding|null {
    if (ns === 'var' && isSpecial(name)) return this.special(name);
    const b = lookup(this.current, name, ns);
    if (b) b.reads++;
    return b;
  }


  // Hoist the declarations of a declarative scope (top level, module body,
  // block). Every name gets its binding before any body is walked, so a
  // declaration may refer to one that appears later in the file
  private hoist(stmts: Statement[], kind: BindingKind, flattenBlocks: boolean):
      void {
    for (const s of stmts) {
      switch (s.kind) {
        case 'block':
          if (flattenBlocks) this.hoist(s.statements, kind, flattenBlocks);
          break;
        case 'variableDecl': {
          if (isSpecial(s.name)) {
            s.binding = this.special(s.name);
            break;
          }
          // A file-scope assignment to a name the runtime pre-declares (PI,
          // _EPSILON, etc.) overwrites that variable
          const outer = this.current.bindings.var.get(s.name) ??
              lookup(this.current.parent, s.name, 'var');
          if ((kind === 'global' || kind === 'filePrivate') &&
              outer?.kind === 'builtin') {
            outer.decls.push(s);
            s.binding = outer;
            break;
          }
          s.binding = this.declare(s.name, 'var', kind, s);
          break;
        }
        case 'functionDecl':
          s.binding = this.declare(s.name, 'fn', kind, s);
          break;
        case 'moduleDecl':
          s.binding = this.declare(s.name, 'mod', kind, s);
          break;
        default:
          break;
      }
    }
  }

  // Bind a declarative scope - hoist its declarations, then walk everything
  private declarative(
      stmts: Statement[], kind: BindingKind, flattenBlocks: boolean): void {
    this.hoist(stmts, kind, flattenBlocks);
    for (const s of stmts) this.stmt(s, flattenBlocks);
  }


  private stmt(s: Statement, flattenBlocks = false): void {
    switch (s.kind) {
      case 'variableDecl':
        this.expr(s.value);
        break;

      case 'functionDecl':
        this.withScope('function', () => {
          this.params(s.params);
          this.expr(s.body);
        });
        break;

      case 'moduleDecl':
        this.withScope('module', () => {
          this.params(s.params);
          const body = s.body.kind === 'block' ? s.body.statements : [s.body];
          // The body block is the module's own scope; nested blocks are not.
          this.declarative(body, 'local', false);
        });
        break;

      case 'block':
        // At the declarative top level a block was already hoisted into the
        // enclosing scope, so only walk its statements. Anywhere else it is
        // emitted as a JS scope of its own.
        if (flattenBlocks) {
          for (const inner of s.statements) this.stmt(inner, flattenBlocks);
        } else {
          this.withScope(
              'block', () => this.declarative(s.statements, 'local', false));
        }
        break;

      case 'scope':
        this.usedFileScope(s);
        break;

      case 'moduleCall':
        s.ref = {
          fn: null,
          value: null,
          mod: this.resolve(s.name, 'mod'),
        };
        for (const a of s.args) this.expr(a.value);
        if (s.child) this.stmt(s.child);
        break;

      case 'for':
        this.withScope('for', () => {
          this.forVariables(s.variables);
          this.stmt(s.body);
        });
        break;

      case 'if':
        this.expr(s.condition);
        this.stmt(s.thenBody);
        if (s.elseBody) this.stmt(s.elseBody);
        break;

      case 'empty':
      case 'use':
      case 'include':
        break;
    }
  }

  // A `use`d file is one scope. Its top-level variables stay private to it, its
  // modules and functions are published to the enclosing scope
  private usedFileScope(s: ScopeStmt): void {
    const outer = this.current;
    this.withScope('file', () => {
      for (const inner of s.statements) {
        switch (inner.kind) {
          case 'variableDecl':
            if (isSpecial(inner.name)) {
              inner.binding = this.special(inner.name);
            } else if (lookup(outer, inner.name, 'var')?.kind === 'builtin') {
              // Pre-declared names refer to the program-wide binding, so they
              // never become private to the file.
              inner.binding = this.resolve(inner.name, 'var')!;
            } else {
              inner.binding =
                  this.declare(inner.name, 'var', 'filePrivate', inner);
            }
            break;
          case 'functionDecl':
            inner.binding =
                this.declare(inner.name, 'fn', 'global', inner, outer);
            break;
          case 'moduleDecl':
            inner.binding =
                this.declare(inner.name, 'mod', 'global', inner, outer);
            break;
          default:
            break;
        }
      }
      for (const inner of s.statements) this.stmt(inner);
    });
  }

  // Parameters bind left to right
  private params(params: Parameter[]): void {
    for (const p of params) {
      if (p.defaultValue) this.expr(p.defaultValue);
      if (isSpecial(p.name)) {
        p.binding = this.special(p.name);
      } else {
        p.binding = this.declare(p.name, 'var', 'param', p);
      }
    }
  }

  // Each loop variable's range sees the variables bound before it
  private forVariables(vars: ForVariable[]): void {
    for (const v of vars) {
      this.expr(v.range);
      if (isSpecial(v.name)) {
        v.binding = this.special(v.name);
      } else {
        v.binding = this.declare(v.name, 'var', 'loop', v);
      }
    }
  }

  // `let` is sequential: each value sees the bindings established before it
  private letAssignments(assignments: LetAssignment[], kind: BindingKind):
      void {
    for (const a of assignments) {
      const selfRecursive = a.value.kind === 'lambda' && !isSpecial(a.name);
      if (selfRecursive) {
        a.binding = this.declare(a.name, 'var', kind, a);
        this.expr(a.value);
        continue;
      }
      this.expr(a.value);
      if (isSpecial(a.name)) {
        a.binding = this.special(a.name);
      } else {
        a.binding = this.declare(a.name, 'var', kind, a);
      }
    }
  }

  private expr(e: Expr): void {
    switch (e.kind) {
      case 'number':
      case 'string':
      case 'boolean':
      case 'undef':
        break;

      case 'identifier':
        e.binding = this.resolve(e.name, 'var');
        break;

      case 'call':
        e.ref = {
          fn: this.resolve(e.name, 'fn'),
          value: this.resolve(e.name, 'var'),
          mod: null,
        };
        for (const a of e.args) this.expr(a.value);
        break;

      case 'vector':
        for (const el of e.elements) this.expr(el);
        break;

      case 'range':
        this.expr(e.start);
        if (e.step) this.expr(e.step);
        this.expr(e.end);
        break;

      case 'binary':
        this.expr(e.left);
        this.expr(e.right);
        break;

      case 'unary':
        this.expr(e.operand);
        break;

      case 'ternary':
        this.expr(e.condition);
        this.expr(e.ifTrue);
        this.expr(e.ifFalse);
        break;

      case 'index':
        this.expr(e.object);
        this.expr(e.index);
        break;

      case 'member':
        this.expr(e.object);
        break;

      case 'group':
        this.expr(e.expr);
        break;

      case 'echo':
      case 'assert':
        for (const a of e.args) this.expr(a.value);
        this.expr(e.expr);
        break;

      case 'let':
        this.withScope('let', () => {
          this.letAssignments(e.assignments, 'let');
          this.expr(e.body);
        });
        break;

      case 'each':
        this.expr(e.expr);
        break;

      case 'lambda':
        this.withScope('lambda', () => {
          this.params(e.params);
          this.expr(e.body);
        });
        break;

      case 'listComp':
        this.listComp(e.generator);
        break;

      case 'dynCall':
        this.expr(e.callee);
        for (const a of e.args) this.expr(a.value);
        break;
    }
  }

  private listComp(gen: ListCompGenerator): void {
    switch (gen.kind) {
      case 'lcFor':
        this.withScope('comprehension', () => {
          this.forVariables(gen.variables);
          this.listComp(gen.body);
        });
        break;

      case 'lcCFor':
        // Init values evaluate in the outer scope; the loop names cover the
        // condition, the updates and the body.
        this.withScope('comprehension', () => {
          this.letAssignments(gen.inits, 'loop');
          this.expr(gen.condition);
          for (const u of gen.updates) {
            this.expr(u.value);
            u.binding = isSpecial(u.name) ? this.special(u.name) :
                                            this.resolve(u.name, 'var') ??
                    this.declare(u.name, 'var', 'loop', u);
          }
          this.listComp(gen.body);
        });
        break;

      case 'lcIf':
        this.expr(gen.condition);
        this.listComp(gen.ifTrue);
        if (gen.ifFalse) this.listComp(gen.ifFalse);
        break;

      case 'lcLet':
        this.withScope('comprehension', () => {
          this.letAssignments(gen.assignments, 'let');
          this.listComp(gen.body);
        });
        break;

      case 'lcExpr':
        this.expr(gen.expr);
        break;
    }
  }

  bindProgram(program: Program): void {
    this.global = this.newScope('global', this.externals);
    this.current = this.global;
    // The top level flattens blocks: their assignments merge into file scope.
    this.declarative(program.statements, 'global', true);
    this.current = this.root;
  }

  /* Bind a library closure. Each file gets a scope of its own holding its own
    top-level declarations, then borrows from the files it pulls in: `include`
    contributes everything, transitively, while `use` contributes only what a
    used file publishes. A used file therefore does NOT see its user's
    declarations

    `global` is left as an empty scope parented to the externals, so anything
    asking for "the library's globals" without naming a file resolves to the
    builtins rather than to an arbitrary file's declarations.
   */
  bindLibraryClosure(files: LibraryFileInput[], entryRels: string[]):
      Map<string, Scope> {
    this.global = this.newScope('global', this.externals);
    const byRel = new Map(files.map(f => [f.rel, f]));

    // `include`: everything it includes end up in one flat scope
    const includeReachable = (roots: string[]): string[] => {
      const seen = new Set<string>();
      const stack = [...roots];
      while (stack.length) {
        const r = stack.pop()!;
        if (seen.has(r) || !byRel.has(r)) continue;
        seen.add(r);
        for (const e of byRel.get(r)!.edges)
          if (e.mode === 'include') stack.push(e.rel);
      }
      return [...seen].sort();
    };

    const useRoots = new Set<string>();
    for (const f of files)
      for (const e of f.edges)
        if (e.mode === 'use' && byRel.has(e.rel)) useRoots.add(e.rel);

    const ENTRY = '';
    const unitRoots = [ENTRY, ...[...useRoots].sort()];
    const unitMembers = new Map<string, string[]>();
    unitMembers.set(ENTRY, includeReachable(entryRels));
    for (const r of useRoots) unitMembers.set(r, includeReachable([r]));

    const unitScopes = new Map<string, Scope>();
    for (const r of unitRoots)
      unitScopes.set(r, this.newScope('file', this.externals));

    // A file belongs to the unit it roots if someone `use`s it, otherwise to
    // the entry unit.
    const unitOf = (rel: string) => useRoots.has(rel) ? rel : ENTRY;

    // Own declarations of each unit's members, last-wins across the unit
    for (const r of unitRoots) {
      this.current = unitScopes.get(r)!;
      for (const rel of unitMembers.get(r)!) {
        const f = byRel.get(rel);
        if (!f) continue;
        this.currentFile = rel;
        this.hoist(f.program.statements, 'global', true);
      }
    }
    this.currentFile = undefined;

    // What each unit publishes to units that `use` it
    for (const r of unitRoots) {
      const scope = unitScopes.get(r)!;
      const seenUse = new Set<string>();
      for (const rel of unitMembers.get(r)!) {
        for (const e of byRel.get(rel)?.edges ?? []) {
          if (e.mode !== 'use' || seenUse.has(e.rel)) continue;
          seenUse.add(e.rel);
          const from = unitScopes.get(unitOf(e.rel));
          if (!from) continue;
          for (const ns of NAMESPACES) {
            for (const [n, b] of from.bindings[ns]) {
              if (ns === 'var' && isSpecial(n)) continue;
              if (scope.bindings[ns].has(n)) continue;
              scope.bindings[ns].set(n, b);
            }
          }
        }
      }
    }

    // Bodies last, so every unit scope is fully populated first
    const fileScopes = new Map<string, Scope>();
    for (const f of files) {
      const scope = unitScopes.get(unitOf(f.rel))!;
      fileScopes.set(f.rel, scope);
      this.current = scope;
      for (const st of f.program.statements) this.stmt(st, true);
    }
    this.current = this.root;
    return fileScopes;
  }
}

export function bindProgram(program: Program, opts: BindOptions): BindResult {
  const binder = new Binder(opts);
  binder.bindProgram(program);
  return {
    root: binder.root,
    global: binder.global,
    bindings: binder.bindings,
    scopes: binder.scopes,
  };
}

export function bindLibrary(
    files: LibraryFileInput[], entryRels: string[],
    opts: BindOptions): LibraryBindResult {
  const binder = new Binder(opts);
  const fileScopes = binder.bindLibraryClosure(files, entryRels);
  return {
    root: binder.root,
    global: binder.global,
    bindings: binder.bindings,
    scopes: binder.scopes,
    fileScopes,
  };
}


export function isLexicalVar(b: Binding|null|undefined): boolean {
  if (!b) return false;
  switch (b.kind) {
    case 'param':
    case 'local':
    case 'let':
    case 'loop':
    case 'global':
    case 'filePrivate':
      return true;
    default:
      return false;
  }
}

// checks if a local share its name with a variable of an enclosing scope
export function shadowsOuterVar(b: Binding): boolean {
  return isLexicalVar(lookup(b.scope.parent, b.name, 'var'));
}

export interface PrettyNameOptions {
  // Runtime symbols and other fixed names no binding may take
  reserved: Iterable<string>;
  // `<ns>:<name>` -> the symbol a separately compiled library exported it as
  externalSymbols?: Map<string, string>;
}

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
          // Builtins and library exports already have names the emitter must
          // match: the runtime's, or the ones the library chose for itself.
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
  const fromLib = opts.externalSymbols?.get(`${b.ns}:${b.name}`);
  if (fromLib) return fromLib;
  const base = escapeName(b.name);
  return b.ns === 'fn' ? `${base}_fn` : b.ns === 'mod' ? `${base}$mod` : base;
}

function pick(
    b: Binding, taken: Set<string>, ancestors: Set<string>,
    ancestorCallables: Set<string>, reserved: Set<string>): string {
  const base = escapeName(b.name);
  // The name the source used may shadow an enclosing *variable*, but never an
  // enclosing function or module.
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

export function assignLegacyNames(result: BindResult): void {
  for (const b of result.bindings) {
    const base = escapeName(b.name);
    b.jsName = b.ns === 'fn' ? `${base}_fn` :
        b.ns === 'mod' ? `${base}$mod` :
                         base;
  }
}
