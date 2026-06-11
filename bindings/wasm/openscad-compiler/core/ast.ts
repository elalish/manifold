import type { SourceRange } from "./lexer.js";

export interface Comment {
  kind: "line" | "block";
  value: string;
  loc: SourceRange;
}

export interface ASTNode {
  loc?: SourceRange | undefined;
  leadingComments?: Comment[] | undefined;
  trailingComments?: Comment[] | undefined;
  filename?: string | undefined;
}

export type Expr =
  | NumberLiteral
  | StringLiteral
  | BooleanLiteral
  | UndefLiteral
  | IdentifierExpr
  | VectorExpr
  | RangeExpr
  | BinaryExpr
  | UnaryExpr
  | TernaryExpr
  | FunctionCallExpr
  | IndexExpr
  | MemberExpr
  | GroupExpr
  | EchoExpr
  | LetExpr
  | AssertExpr
  | ListCompExpr
  | EachExpr
  | LambdaExpr
  | DynCallExpr;

export interface NumberLiteral   extends ASTNode { kind: "number";     value: number; }
export interface StringLiteral   extends ASTNode { kind: "string";     value: string; }
export interface BooleanLiteral  extends ASTNode { kind: "boolean";    value: boolean; }
export interface UndefLiteral    extends ASTNode { kind: "undef"; }
export interface IdentifierExpr  extends ASTNode { kind: "identifier"; name: string; }
export interface VectorExpr      extends ASTNode { kind: "vector";     elements: Expr[]; }
export interface RangeExpr       extends ASTNode { kind: "range";      start: Expr; end: Expr; step?: Expr | undefined; }
export interface BinaryExpr      extends ASTNode { kind: "binary";     op: string; left: Expr; right: Expr; }
export interface UnaryExpr       extends ASTNode { kind: "unary";      op: string; operand: Expr; }
export interface TernaryExpr     extends ASTNode { kind: "ternary";    condition: Expr; ifTrue: Expr; ifFalse: Expr; }
export interface FunctionCallExpr extends ASTNode{ kind: "call";       name: string; args: Argument[]; }
export interface IndexExpr       extends ASTNode { kind: "index";      object: Expr; index: Expr; }
export interface MemberExpr      extends ASTNode { kind: "member";     object: Expr; property: string; }
export interface GroupExpr       extends ASTNode { kind: "group";      expr: Expr; }
export interface EchoExpr        extends ASTNode { kind: "echo";       args: Argument[]; expr: Expr; }
export interface LetExpr         extends ASTNode { kind: "let";        assignments: LetAssignment[]; body: Expr; }
export interface AssertExpr      extends ASTNode { kind: "assert";     args: Argument[]; expr: Expr; }
export interface ListCompExpr    extends ASTNode { kind: "listComp";   generator: ListCompGenerator; }
export interface EachExpr        extends ASTNode { kind: "each";       expr: Expr; }
export interface LambdaExpr      extends ASTNode { kind: "lambda";     params: Parameter[]; body: Expr; }
export interface DynCallExpr     extends ASTNode { kind: "dynCall";    callee: Expr; args: Argument[]; }

export interface Argument extends ASTNode {
  name?: string | undefined;
  value: Expr;
}

export type Statement =
  | ModuleCallStmt
  | BlockStmt
  | VariableDeclStmt
  | ModuleDeclStmt
  | FunctionDeclStmt
  | ForStmt
  | IfStmt
  | EmptyStmt
  | UseStmt
  | IncludeStmt;

export interface ModuleCallStmt extends ASTNode {
  kind: "moduleCall";
  name: string;
  args: Argument[];
  child?: Statement | undefined;
  modifier?: string | undefined;   // *, !, #, %
}

export interface BlockStmt extends ASTNode {
  kind: "block";
  statements: Statement[];
}

export interface VariableDeclStmt extends ASTNode {
  kind: "variableDecl";
  name: string;
  value: Expr;
}

export interface ModuleDeclStmt extends ASTNode {
  kind: "moduleDecl";
  name: string;
  params: Parameter[];
  body: Statement;
}

export interface FunctionDeclStmt extends ASTNode {
  kind: "functionDecl";
  name: string;
  params: Parameter[];
  body: Expr;
}

export interface ForStmt extends ASTNode {
  kind: "for";
  variables: ForVariable[];
  body: Statement;
}

export interface IfStmt extends ASTNode {
  kind: "if";
  condition: Expr;
  thenBody: Statement;
  elseBody?: Statement | undefined;
}

export interface EmptyStmt extends ASTNode {
  kind: "empty";
}

export interface Parameter extends ASTNode {
  name: string;
  defaultValue?: Expr | undefined;
}

export interface ForVariable extends ASTNode {
  name: string;
  range: Expr;
}

export interface UseStmt extends ASTNode {
  kind: "use";
  path: string;
}

export interface IncludeStmt extends ASTNode {
  kind: "include";
  path: string;
}

export interface LetAssignment extends ASTNode {
  name: string;
  value: Expr;
}

export type ListCompGenerator =
  | LCForGenerator
  | LCCStyleForGenerator
  | LCIfGenerator
  | LCLetGenerator
  | LCExprGenerator;

export interface LCForGenerator  extends ASTNode { kind: "lcFor";  variables: ForVariable[]; body: ListCompGenerator; }
export interface LCCStyleForGenerator extends ASTNode { kind: "lcCFor"; inits: LetAssignment[]; condition: Expr; updates: LetAssignment[]; body: ListCompGenerator; }
export interface LCIfGenerator   extends ASTNode { kind: "lcIf";   condition: Expr; ifTrue: ListCompGenerator; ifFalse?: ListCompGenerator | undefined; }
export interface LCLetGenerator  extends ASTNode { kind: "lcLet";  assignments: LetAssignment[]; body: ListCompGenerator; }
export interface LCExprGenerator extends ASTNode { kind: "lcExpr"; expr: Expr; }

export interface Program extends ASTNode {
  kind: "program";
  statements: Statement[];
}
