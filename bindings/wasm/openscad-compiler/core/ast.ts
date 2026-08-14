import type {Binding, CallRef} from './binder.js';
import type {SourceRange} from './lexer.js';

export interface Comment {
  kind: 'line'|'block';
  value: string;
  loc: SourceRange;
}

export interface ASTNode {
  loc?: SourceRange|undefined;
  leadingComments?: Comment[]|undefined;
  trailingComments?: Comment[]|undefined;
  filename?: string|undefined;
}

export type Expr =|NumberLiteral|StringLiteral|BooleanLiteral|UndefLiteral|
    IdentifierExpr|VectorExpr|RangeExpr|BinaryExpr|UnaryExpr|TernaryExpr|
    FunctionCallExpr|IndexExpr|MemberExpr|GroupExpr|EchoExpr|LetExpr|AssertExpr|
    ListCompExpr|EachExpr|LambdaExpr|DynCallExpr;

export interface NumberLiteral extends ASTNode {
  kind: 'number';
  value: number;
}
export interface StringLiteral extends ASTNode {
  kind: 'string';
  value: string;
}
export interface BooleanLiteral extends ASTNode {
  kind: 'boolean';
  value: boolean;
}
export interface UndefLiteral extends ASTNode {
  kind: 'undef';
}
export interface IdentifierExpr extends ASTNode {
  kind: 'identifier';
  name: string;
  // Attached by the binder pass. `null` means the name is not bound anywhere,
  // which OpenSCAD resolves to `undef` rather than an error.
  binding?: Binding|null|undefined;
}
export interface VectorExpr extends ASTNode {
  kind: 'vector';
  elements: Expr[];
}
export interface RangeExpr extends ASTNode {
  kind: 'range';
  start: Expr;
  end: Expr;
  step?: Expr|undefined;
}
export interface BinaryExpr extends ASTNode {
  kind: 'binary';
  op: string;
  left: Expr;
  right: Expr;
}
export interface UnaryExpr extends ASTNode {
  kind: 'unary';
  op: string;
  operand: Expr;
}
export interface TernaryExpr extends ASTNode {
  kind: 'ternary';
  condition: Expr;
  ifTrue: Expr;
  ifFalse: Expr;
}
export interface FunctionCallExpr extends ASTNode {
  kind: 'call';
  name: string;
  args: Argument[];
  // Attached by the binder pass
  ref?: CallRef|undefined;
}
export interface IndexExpr extends ASTNode {
  kind: 'index';
  object: Expr;
  index: Expr;
}
export interface MemberExpr extends ASTNode {
  kind: 'member';
  object: Expr;
  property: string;
}
export interface GroupExpr extends ASTNode {
  kind: 'group';
  expr: Expr;
}
export interface EchoExpr extends ASTNode {
  kind: 'echo';
  args: Argument[];
  expr: Expr;
}
export interface LetExpr extends ASTNode {
  kind: 'let';
  assignments: LetAssignment[];
  body: Expr;
}
export interface AssertExpr extends ASTNode {
  kind: 'assert';
  args: Argument[];
  expr: Expr;
}
export interface ListCompExpr extends ASTNode {
  kind: 'listComp';
  generator: ListCompGenerator;
}
export interface EachExpr extends ASTNode {
  kind: 'each';
  expr: Expr;
}
export interface LambdaExpr extends ASTNode {
  kind: 'lambda';
  params: Parameter[];
  body: Expr;
}
export interface DynCallExpr extends ASTNode {
  kind: 'dynCall';
  callee: Expr;
  args: Argument[];
}

export interface Argument extends ASTNode {
  name?: string|undefined;
  value: Expr;
}

export type Statement =
    |ModuleCallStmt|BlockStmt|VariableDeclStmt|ModuleDeclStmt|FunctionDeclStmt|
    ForStmt|IfStmt|EmptyStmt|UseStmt|IncludeStmt|ScopeStmt;

export interface ModuleCallStmt extends ASTNode {
  kind: 'moduleCall';
  name: string;
  args: Argument[];
  child?: Statement|undefined;
  modifier?: string|undefined;  // *, !, #, %
  // Attached by the binder pass
  ref?: CallRef|undefined;
}

export interface BlockStmt extends ASTNode {
  kind: 'block';
  statements: Statement[];
  modifier?: string|undefined;  // *, !, #, %
}

export interface VariableDeclStmt extends ASTNode {
  kind: 'variableDecl';
  name: string;
  value: Expr;
  binding?: Binding|undefined;
}

export interface ModuleDeclStmt extends ASTNode {
  kind: 'moduleDecl';
  name: string;
  params: Parameter[];
  body: Statement;
  binding?: Binding|undefined;
}

export interface FunctionDeclStmt extends ASTNode {
  kind: 'functionDecl';
  name: string;
  params: Parameter[];
  body: Expr;
  binding?: Binding|undefined;
}

export interface ForStmt extends ASTNode {
  kind: 'for';
  variables: ForVariable[];
  body: Statement;
  modifier?: string|undefined;  // *, !, #, %
}

export interface IfStmt extends ASTNode {
  kind: 'if';
  condition: Expr;
  thenBody: Statement;
  elseBody?: Statement|undefined;
  modifier?: string|undefined;  // *, !, #, %
}

export interface EmptyStmt extends ASTNode {
  kind: 'empty';
}

export interface ScopeStmt extends ASTNode {
  kind: 'scope';
  statements: Statement[];
}

export interface Parameter extends ASTNode {
  name: string;
  defaultValue?: Expr|undefined;
  binding?: Binding|undefined;
}

export interface ForVariable extends ASTNode {
  name: string;
  range: Expr;
  binding?: Binding|undefined;
}

export interface UseStmt extends ASTNode {
  kind: 'use';
  path: string;
}

export interface IncludeStmt extends ASTNode {
  kind: 'include';
  path: string;
}

export interface LetAssignment extends ASTNode {
  name: string;
  value: Expr;
  binding?: Binding|undefined;
}

export type ListCompGenerator =|LCForGenerator|LCCStyleForGenerator|
    LCIfGenerator|LCLetGenerator|LCExprGenerator;

export interface LCForGenerator extends ASTNode {
  kind: 'lcFor';
  variables: ForVariable[];
  body: ListCompGenerator;
}
export interface LCCStyleForGenerator extends ASTNode {
  kind: 'lcCFor';
  inits: LetAssignment[];
  condition: Expr;
  updates: LetAssignment[];
  body: ListCompGenerator;
}
export interface LCIfGenerator extends ASTNode {
  kind: 'lcIf';
  condition: Expr;
  ifTrue: ListCompGenerator;
  ifFalse?: ListCompGenerator|undefined;
}
export interface LCLetGenerator extends ASTNode {
  kind: 'lcLet';
  assignments: LetAssignment[];
  body: ListCompGenerator;
}
export interface LCExprGenerator extends ASTNode {
  kind: 'lcExpr';
  expr: Expr;
}

export interface Program extends ASTNode {
  kind: 'program';
  statements: Statement[];
}

// Every AST node that carries a `kind` discriminant
export type KindedNode = Expr|Statement|ListCompGenerator|Program;

type NodeOfKind<K extends KindedNode['kind']> = Extract<KindedNode, {kind: K}>;

// The child-bearing fields of each node kind.
const CHILD_KEYS: {[K in KindedNode['kind']]: readonly(keyof NodeOfKind<K>)[]} =
    {
      // Expressions
      number: [],
      string: [],
      boolean: [],
      undef: [],
      identifier: [],
      vector: ['elements'],
      range: ['start', 'end', 'step'],
      binary: ['left', 'right'],
      unary: ['operand'],
      ternary: ['condition', 'ifTrue', 'ifFalse'],
      call: ['args'],
      index: ['object', 'index'],
      member: ['object'],
      group: ['expr'],
      echo: ['args', 'expr'],
      let: ['assignments', 'body'],
      assert: ['args', 'expr'],
      listComp: ['generator'],
      each: ['expr'],
      lambda: ['params', 'body'],
      dynCall: ['callee', 'args'],
      // Statements
      moduleCall: ['args', 'child'],
      block: ['statements'],
      scope: ['statements'],
      variableDecl: ['value'],
      moduleDecl: ['params', 'body'],
      functionDecl: ['params', 'body'],
      for: ['variables', 'body'],
      if: ['condition', 'thenBody', 'elseBody'],
      empty: [],
      use: [],
      include: [],
      // List-comprehension generators
      lcFor: ['variables', 'body'],
      lcCFor: ['inits', 'condition', 'updates', 'body'],
      lcIf: ['condition', 'ifTrue', 'ifFalse'],
      lcLet: ['assignments', 'body'],
      lcExpr: ['expr'],
      // Root
      program: ['statements'],
    };

const WRAPPER_CHILD_KEYS = ['value', 'defaultValue', 'range'] as const;

function visitChildValue(
    value: unknown, visit: (child: KindedNode) => void): void {
  if (!value || typeof value !== 'object') return;
  if (Array.isArray(value)) {
    for (const item of value) visitChildValue(item, visit);
    return;
  }
  if ('kind' in value) {
    visit(value as KindedNode);
    return;
  }
  for (const key of WRAPPER_CHILD_KEYS)
    visitChildValue((value as Record<string, unknown>)[key], visit);
}

// Traverses through the AST nodes and invoke a callback for each child
export function forEachChild(
    node: KindedNode, visit: (child: KindedNode) => void): void {
  const keys = (CHILD_KEYS as Record<string, readonly string[]>)[node.kind];
  if (!keys) return;
  for (const key of keys)
    visitChildValue((node as unknown as Record<string, unknown>)[key], visit);
}
