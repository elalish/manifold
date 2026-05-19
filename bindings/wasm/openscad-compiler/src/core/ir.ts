import type { Argument, ASTNode, Expr, ForVariable, Statement } from "./ast.js";

export type PrimitiveKind =
  | "cube"
  | "sphere"
  | "cylinder"
  | "circle"
  | "square"
  | "polygon"
  | "polyhedron"
  | "text";

export type TransformKind =
  | "translate"
  | "rotate"
  | "scale"
  | "mirror"
  | "multmatrix"
  | "resize"
  | "offset"
  | "color"
  | "render"
  | "projection";

export type BooleanKind =
  | "union"
  | "difference"
  | "intersection"
  | "hull"
  | "minkowski";

interface IRBase {
  loc?: ASTNode["loc"];
}

export interface IREmptyNode extends IRBase {
  kind: "empty";
}

export interface IRPrimitiveNode extends IRBase {
  kind: "primitive";
  primitive: PrimitiveKind;
  args: Argument[];
}

export interface IRTransformNode extends IRBase {
  kind: "transform";
  transform: TransformKind;
  args: Argument[];
  child: IRNode;
}

export interface IRBooleanNode extends IRBase {
  kind: "boolean";
  op: BooleanKind;
  children: IRNode[];
}

export interface IRModuleCallNode extends IRBase {
  kind: "moduleCall";
  name: string;
  args: Argument[];
  children: IRNode[];
}

export interface IRChildrenNode extends IRBase {
  kind: "children";
  indexExpr?: Expr | undefined;
}

export interface IRSequenceNode extends IRBase {
  kind: "sequence";
  items: IRNode[];
}

export interface IRIfNode extends IRBase {
  kind: "if";
  condition: Expr;
  thenNode: IRNode;
  elseNode?: IRNode | undefined;
}

export interface IRForNode extends IRBase {
  kind: "for";
  variables: ForVariable[];
  body: IRNode;
}

// Safety valve for syntax we haven't lowered yet.
export interface IRAstFallbackNode extends IRBase {
  kind: "astFallback";
  statement: Statement;
}

export type IRNode =
  | IREmptyNode
  | IRPrimitiveNode
  | IRTransformNode
  | IRBooleanNode
  | IRModuleCallNode
  | IRChildrenNode
  | IRSequenceNode
  | IRIfNode
  | IRForNode
  | IRAstFallbackNode;
