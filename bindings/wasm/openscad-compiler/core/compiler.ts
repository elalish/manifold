import type {
  Program, Statement, Expr, Argument,
  ModuleCallStmt, BlockStmt, ForStmt, IfStmt,
  ForVariable, ListCompGenerator, ASTNode,
  Comment,
} from "./ast.js";
import type {
  IRNode,
  IRPrimitiveNode,
  IRTransformNode,
  IRBooleanNode,
  IRModuleCallNode,
  IRChildrenNode,
  IRSequenceNode,
  IRIfNode,
  IRForNode,
} from "./ir.js";
import fs from "fs";
import path from "path";
import { getFontPath } from "./resolver.js";

function locTag(node: ASTNode): string {
  if (!node.loc) return "";
  const s = node.loc.start;
  return ` @${s.line}:${s.column}`;
}

function commentLines(comment: Comment, indent = ""): string[] {
  return comment.value.split(/\r?\n/).map(line => `${indent}${line}`);
}

function leadingCommentLines(node: ASTNode | undefined, indent = ""): string[] {
  return (node?.leadingComments ?? []).flatMap(comment => commentLines(comment, indent));
}

function trailingCommentText(node: ASTNode | undefined): string {
  const comments = node?.trailingComments ?? [];
  if (comments.length === 0) return "";
  return ` ${comments.map(comment => comment.value.replace(/\r?\n/g, " ")).join(" ")}`;
}

function returnExpr(expr: string, indent = ""): string {
  const trimmed = expr.trim();
  if (trimmed.startsWith("//") || trimmed.startsWith("/*")) {
    return `(\n${expr}\n${indent})`;
  }
  return expr;
}

function pushCommentedLine(lines: string[], node: ASTNode, line: string, indent = ""): void {
  lines.push(...leadingCommentLines(node, indent));
  lines.push(`${line}${trailingCommentText(node)}`);
}

// JavaScript reserved words
const JS_RESERVED = new Set([
  "abstract", "arguments", "await", "boolean", "break", "byte", "case", "catch",
  "char", "class", "const", "continue", "debugger", "default", "delete", "do",
  "double", "else", "enum", "eval", "export", "extends", "false", "final",
  "finally", "float", "for", "function", "goto", "if", "implements", "import",
  "in", "instanceof", "int", "interface", "let", "long", "native", "new",
  "null", "package", "private", "protected", "public", "return", "short",
  "static", "super", "switch", "synchronized", "this", "throw", "throws",
  "transient", "true", "try", "typeof", "var", "void", "volatile", "while",
  "with", "yield",
]);

function escapeName(name: string): string {
  if (JS_RESERVED.has(name)) return `${name}_`;
  if (/^[0-9]/.test(name)) return `_${name}`;
  return name;
}

// Signatures
interface Signature {
  params: string[];
  defaults: (Expr | undefined)[];
}
const signatures = new Map<string, Signature>();

type ModuleDeclStmtType = Extract<Statement, { kind: "moduleDecl" }>;

interface IRLowerContext {
  modules: Map<string, ModuleDeclStmtType>;
  children: IRNode[];
  callStack: string[];
}

let moduleDeclRegistry = new Map<string, ModuleDeclStmtType>();

// Track unique fonts encountered during compilation for base64 generation.
let encounteredFonts = new Set<string>();

// Track unique images encountered during compilation for base64 generation.
let encounteredImages = new Map<string, { stem: string; exportName: string }>();

let currentRuntimePath: string = "./runtime/runtime.js";

function fontSpecToFilename(fontSpec: string): string {
  const cleaned = fontSpec.replace(/"/g, "").trim();
  const parts = cleaned.split(":");
  const family = (parts[0] || "Liberation Sans").trim().replace(/\s+/g, "");

  let style = "Regular";
  for (let i = 1; i < parts.length; i++) {
    const part = parts[i]!.trim();
    const match = part.match(/^style\s*=\s*(.+)$/i);
    if (match) {
      style = match[1]!.trim().replace(/\s+/g, "");
      break;
    }
  }

  return `${family}-${style}`;
}

function generateFontBase64(fontSpec: string, compilerDir: string): string | undefined {
  const fontDir = getFontPath();
  if (!fontDir) {
    console.warn(`Warning: FONTPATH not set in .env — cannot load font "${fontSpec}". Text will render as empty cross-section.`);
    return undefined;
  }

  let filename = fontSpec;
  let ttfPath = path.join(fontDir, `${filename}.ttf`);
  let otfPath = path.join(fontDir, `${filename}.otf`);

  if (!fs.existsSync(ttfPath) && !fs.existsSync(otfPath)) {
    filename = fontSpecToFilename(fontSpec);
    ttfPath = path.join(fontDir, `${filename}.ttf`);
    otfPath = path.join(fontDir, `${filename}.otf`);
  }

  let fontFilePath: string | undefined;
  let mimeType: string;

  if (fs.existsSync(ttfPath)) {
    fontFilePath = ttfPath;
    mimeType = "font/ttf";
  } else if (fs.existsSync(otfPath)) {
    fontFilePath = otfPath;
    mimeType = "font/otf";
  } else {
    console.warn(`Warning: Font file not found at "${ttfPath}" or "${otfPath}" — text using "${fontSpec}" will render as empty cross-section.`);
    return undefined;
  }

  const fontBytes = fs.readFileSync(fontFilePath);
  const base64 = fontBytes.toString("base64");

  const fontsDir = path.join(compilerDir, "runtime", "fonts");
  fs.mkdirSync(fontsDir, { recursive: true });

  const outFile = path.join(fontsDir, `${filename}_base64.ts`);
  const content = `// Auto-generated by openscad-to-manifold compiler — do not edit.\nexport const fontBase64 = "data:${mimeType};base64,${base64}";\n`;
  fs.writeFileSync(outFile, content, "utf8");
  console.log(`Generated font base64: ${outFile} (${(fontBytes.length / 1024).toFixed(1)} KB)`);

  return filename;
}

const MAX_IR_INLINE_DEPTH = 2;
const MAX_IR_INLINE_COMPLEXITY = 120;

const BUILTIN_SIGNATURES: Record<string, string[]> = {
  "cube$mod": ["size", "center"],
  "cylinder$mod": ["h", "r", "r1", "r2", "d", "d1", "d2", "center", "$fn", "$fa", "$fs"],
  "sphere$mod": ["r", "d", "$fn", "$fa", "$fs"],
  "square$mod": ["size", "center"],
  "circle$mod": ["r", "d", "$fn", "$fa", "$fs"],
  "polygon$mod": ["points", "paths", "convexity"],
  "polyhedron$mod": ["points", "faces", "convexity"],
  "linear_extrude$mod": ["height", "center", "convexity", "twist", "slices", "scale", "$fn", "$fa", "$fs"],
  "rotate_extrude$mod": ["angle", "convexity", "$fn"],
  "text$mod": ["text", "size", "font", "halign", "valign", "spacing", "direction", "language", "script", "$fn"],
  "surface$mod": ["file", "center", "invert", "convexity"],
  "import$mod": ["file", "convexity", "layer"],
  "projection$mod": ["cut"],
  "translate$mod": ["v"],
  "rotate$mod": ["a", "v"],
  "scale$mod": ["v"],
  "resize$mod": ["newsize", "auto"],
  "mirror$mod": ["v"],
  "multmatrix$mod": ["m"],
  "color$mod": ["c", "alpha"],
  "offset$mod": ["r", "delta", "chamfer"],
};

function collectSignatures(stmts: Statement[]) {
  for (const stmt of stmts) {
    if (stmt.kind === "functionDecl" || stmt.kind === "moduleDecl") {
      const name = stmt.kind === "functionDecl" ? escapeName(stmt.name) + "_fn" : escapeName(stmt.name) + "$mod";
      signatures.set(name, {
        params: stmt.params.map(p => p.name),
        defaults: stmt.params.map(p => p.defaultValue)
      });
      if (stmt.kind === "moduleDecl" && stmt.body.kind === "block") {
        collectSignatures(stmt.body.statements);
      }
    } else if (stmt.kind === "block") {
      collectSignatures(stmt.statements);
    } else if (stmt.kind === "if") {
      if (stmt.thenBody.kind === "block") collectSignatures(stmt.thenBody.statements);
      if (stmt.elseBody && stmt.elseBody.kind === "block") collectSignatures(stmt.elseBody.statements);
    }
  }
}

function collectModuleDeclarations(
  stmts: Statement[],
  into: Map<string, ModuleDeclStmtType> = new Map<string, ModuleDeclStmtType>(),
): Map<string, ModuleDeclStmtType> {
  for (const stmt of stmts) {
    if (stmt.kind === "moduleDecl") {
      into.set(stmt.name, stmt);
      if (stmt.body.kind === "block") {
        collectModuleDeclarations(stmt.body.statements, into);
      }
      continue;
    }
    if (stmt.kind === "block") {
      collectModuleDeclarations(stmt.statements, into);
      continue;
    }
    if (stmt.kind === "if") {
      if (stmt.thenBody.kind === "block") collectModuleDeclarations(stmt.thenBody.statements, into);
      if (stmt.elseBody && stmt.elseBody.kind === "block") collectModuleDeclarations(stmt.elseBody.statements, into);
    }
  }
  return into;
}

function baseIRContext(modules = moduleDeclRegistry): IRLowerContext {
  return { modules, children: [], callStack: [] };
}

function estimateExprComplexity(expr: Expr): number {
  switch (expr.kind) {
    case "number":
    case "string":
    case "boolean":
    case "undef":
    case "identifier":
      return 1;
    case "vector":
      return 1 + expr.elements.reduce((sum, item) => sum + estimateExprComplexity(item), 0);
    case "range":
      return 1 + estimateExprComplexity(expr.start) + estimateExprComplexity(expr.end) + (expr.step ? estimateExprComplexity(expr.step) : 0);
    case "binary":
      return 1 + estimateExprComplexity(expr.left) + estimateExprComplexity(expr.right);
    case "unary":
      return 1 + estimateExprComplexity(expr.operand);
    case "group":
      return 1 + estimateExprComplexity(expr.expr);
    case "each":
      return 1 + estimateExprComplexity(expr.expr);
    case "ternary":
      return 1 + estimateExprComplexity(expr.condition) + estimateExprComplexity(expr.ifTrue) + estimateExprComplexity(expr.ifFalse);
    case "call":
      return 1 + expr.args.reduce((sum, arg) => sum + estimateExprComplexity(arg.value), 0);
    case "echo":
      return 1 + expr.args.reduce((sum, arg) => sum + estimateExprComplexity(arg.value), 0) + estimateExprComplexity(expr.expr);
    case "assert":
      return 1 + expr.args.reduce((sum, arg) => sum + estimateExprComplexity(arg.value), 0) + estimateExprComplexity(expr.expr);
    case "index":
      return 1 + estimateExprComplexity(expr.object) + estimateExprComplexity(expr.index);
    case "member":
      return 1 + estimateExprComplexity(expr.object);
    case "let":
      return 1 + expr.assignments.reduce((sum, item) => sum + estimateExprComplexity(item.value), 0) + estimateExprComplexity(expr.body);
    case "listComp":
      return 1 + estimateListCompComplexity(expr.generator);
    case "lambda":
      return 1 + expr.params.reduce((sum, item) => sum + (item.defaultValue ? estimateExprComplexity(item.defaultValue) : 0), 0) + estimateExprComplexity(expr.body);
    case "dynCall":
      return 1 + estimateExprComplexity(expr.callee) + expr.args.reduce((sum, arg) => sum + estimateExprComplexity(arg.value), 0);
    default:
      return 1;
  }
}

function estimateListCompComplexity(generator: ListCompGenerator): number {
  switch (generator.kind) {
    case "lcFor":
      return 1 + generator.variables.reduce((sum, item) => sum + estimateExprComplexity(item.range), 0) + estimateListCompComplexity(generator.body);
    case "lcCFor":
      return 1
        + generator.inits.reduce((sum, item) => sum + estimateExprComplexity(item.value), 0)
        + estimateExprComplexity(generator.condition)
        + generator.updates.reduce((sum, item) => sum + estimateExprComplexity(item.value), 0)
        + estimateListCompComplexity(generator.body);
    case "lcIf":
      return 1 + estimateExprComplexity(generator.condition) + estimateListCompComplexity(generator.ifTrue) + (generator.ifFalse ? estimateListCompComplexity(generator.ifFalse) : 0);
    case "lcLet":
      return 1 + generator.assignments.reduce((sum, item) => sum + estimateExprComplexity(item.value), 0) + estimateListCompComplexity(generator.body);
    case "lcExpr":
      return 1 + estimateExprComplexity(generator.expr);
    default:
      return 1;
  }
}

function estimateStatementComplexity(stmt: Statement): number {
  switch (stmt.kind) {
    case "empty":
    case "use":
    case "include":
      return 1;
    case "variableDecl":
      return 1 + estimateExprComplexity(stmt.value);
    case "functionDecl":
      return 1 + stmt.params.reduce((sum, item) => sum + (item.defaultValue ? estimateExprComplexity(item.defaultValue) : 0), 0) + estimateExprComplexity(stmt.body);
    case "moduleDecl":
      return 1 + stmt.params.reduce((sum, item) => sum + (item.defaultValue ? estimateExprComplexity(item.defaultValue) : 0), 0) + estimateStatementComplexity(stmt.body);
    case "moduleCall":
      return 1 + stmt.args.reduce((sum, arg) => sum + estimateExprComplexity(arg.value), 0) + (stmt.child ? estimateStatementComplexity(stmt.child) : 0);
    case "block":
      return 1 + stmt.statements.reduce((sum, item) => sum + estimateStatementComplexity(item), 0);
    case "for":
      return 1 + stmt.variables.reduce((sum, item) => sum + estimateExprComplexity(item.range), 0) + estimateStatementComplexity(stmt.body);
    case "if":
      return 1 + estimateExprComplexity(stmt.condition) + estimateStatementComplexity(stmt.thenBody) + (stmt.elseBody ? estimateStatementComplexity(stmt.elseBody) : 0);
    default:
      return 1;
  }
}

function exprUsesModuleScope(expr: Expr): boolean {
  switch (expr.kind) {
    case "identifier":
      return expr.name === "$children";
    case "number":
    case "string":
    case "boolean":
    case "undef":
      return false;
    case "vector":
      return expr.elements.some(exprUsesModuleScope);
    case "range":
      return exprUsesModuleScope(expr.start) || exprUsesModuleScope(expr.end) || (expr.step ? exprUsesModuleScope(expr.step) : false);
    case "binary":
      return exprUsesModuleScope(expr.left) || exprUsesModuleScope(expr.right);
    case "unary":
      return exprUsesModuleScope(expr.operand);
    case "ternary":
      return exprUsesModuleScope(expr.condition) || exprUsesModuleScope(expr.ifTrue) || exprUsesModuleScope(expr.ifFalse);
    case "call":
      return expr.args.some(arg => exprUsesModuleScope(arg.value));
    case "index":
      return exprUsesModuleScope(expr.object) || exprUsesModuleScope(expr.index);
    case "member":
      return exprUsesModuleScope(expr.object);
    case "group":
      return exprUsesModuleScope(expr.expr);
    case "echo":
    case "assert":
      return expr.args.some(arg => exprUsesModuleScope(arg.value)) || exprUsesModuleScope(expr.expr);
    case "let":
      return expr.assignments.some(item => exprUsesModuleScope(item.value)) || exprUsesModuleScope(expr.body);
    case "listComp":
      return listCompUsesModuleScope(expr.generator);
    case "each":
      return exprUsesModuleScope(expr.expr);
    case "lambda":
      return expr.params.some(item => item.defaultValue ? exprUsesModuleScope(item.defaultValue) : false) || exprUsesModuleScope(expr.body);
    case "dynCall":
      return exprUsesModuleScope(expr.callee) || expr.args.some(arg => exprUsesModuleScope(arg.value));
    default:
      return false;
  }
}

function listCompUsesModuleScope(generator: ListCompGenerator): boolean {
  switch (generator.kind) {
    case "lcFor":
      return generator.variables.some(item => exprUsesModuleScope(item.range)) || listCompUsesModuleScope(generator.body);
    case "lcCFor":
      return generator.inits.some(item => exprUsesModuleScope(item.value))
        || exprUsesModuleScope(generator.condition)
        || generator.updates.some(item => exprUsesModuleScope(item.value))
        || listCompUsesModuleScope(generator.body);
    case "lcIf":
      return exprUsesModuleScope(generator.condition)
        || listCompUsesModuleScope(generator.ifTrue)
        || (generator.ifFalse ? listCompUsesModuleScope(generator.ifFalse) : false);
    case "lcLet":
      return generator.assignments.some(item => exprUsesModuleScope(item.value)) || listCompUsesModuleScope(generator.body);
    case "lcExpr":
      return exprUsesModuleScope(generator.expr);
    default:
      return false;
  }
}

function statementUsesModuleScope(stmt: Statement): boolean {
  switch (stmt.kind) {
    case "empty":
    case "use":
    case "include":
      return false;
    case "variableDecl":
      return exprUsesModuleScope(stmt.value);
    case "functionDecl":
      return stmt.params.some(item => item.defaultValue ? exprUsesModuleScope(item.defaultValue) : false) || exprUsesModuleScope(stmt.body);
    case "moduleDecl":
      return stmt.params.some(item => item.defaultValue ? exprUsesModuleScope(item.defaultValue) : false) || statementUsesModuleScope(stmt.body);
    case "moduleCall":
      return stmt.name === "children"
        || stmt.args.some(arg => exprUsesModuleScope(arg.value))
        || (stmt.child ? statementUsesModuleScope(stmt.child) : false);
    case "block":
      return stmt.statements.some(statementUsesModuleScope);
    case "for":
      return stmt.variables.some(item => exprUsesModuleScope(item.range)) || statementUsesModuleScope(stmt.body);
    case "if":
      return exprUsesModuleScope(stmt.condition) || statementUsesModuleScope(stmt.thenBody) || (stmt.elseBody ? statementUsesModuleScope(stmt.elseBody) : false);
    default:
      return false;
  }
}

function shouldInlineModuleToIR(decl: ModuleDeclStmtType, ctx: IRLowerContext): boolean {
  if (ctx.callStack.length >= MAX_IR_INLINE_DEPTH) return false;
  if (statementUsesModuleScope(decl.body)) return false;
  return estimateStatementComplexity(decl.body) <= MAX_IR_INLINE_COMPLEXITY;
}

function compileArgList(name: string, args: Argument[]): string {
  const sig = signatures.get(name);
  if (!sig) {
    return args.map(a => a.name ? `/* ${a.name} = */ ${compileExpr(a.value)}` : compileExpr(a.value)).join(", ");
  }

  const compiledArgs: string[] = new Array(sig.params.length).fill("undefined");
  const extraArgs: string[] = [];

  let pos = 0;
  while (pos < args.length && !args[pos]!.name) {
    if (pos < sig.params.length) compiledArgs[pos] = compileExpr(args[pos]!.value);
    else extraArgs.push(compileExpr(args[pos]!.value));
    pos++;
  }

  for (let i = pos; i < args.length; i++) {
    const a = args[i]!;
    if (a.name) {
      const idx = sig.params.indexOf(a.name);
      if (idx >= 0) compiledArgs[idx] = `/* ${a.name} = */ ${compileExpr(a.value)}`;
      else extraArgs.push(`/* ${a.name} = */ ${compileExpr(a.value)}`);
    } else {
      extraArgs.push(compileExpr(a.value));
    }
  }

  while (compiledArgs.length > 0 && compiledArgs[compiledArgs.length - 1] === "undefined") {
    compiledArgs.pop();
  }

  return compiledArgs.concat(extraArgs).join(", ");
}

// OpenSCAD built-in function names
const BUILTIN_FUNCTIONS = new Set([
  "is_undef", "is_bool", "is_num", "is_string", "is_list", "is_function",
  "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
  "abs", "sign", "floor", "ceil", "round", "sqrt", "exp", "ln", "log",
  "min", "max", "norm", "cross",
  "len", "str", "chr", "ord", "concat", "search", "lookup",
  "version", "version_num",
  "parent_module",
]);

// Track $ variables that need module-level declarations for dynamic scoping
let dynamicScopeVars: Set<string> = new Set();
const localScopes: Set<string>[] = [];

function withLocalScope<T>(names: string[], fn: () => T): T {
  localScopes.push(new Set(names));
  try {
    return fn();
  } finally {
    localScopes.pop();
  }
}

function isLocalName(name: string): boolean {
  for (let i = localScopes.length - 1; i >= 0; i--) {
    if (localScopes[i]!.has(name)) return true;
  }
  return false;
}

function registerLocalCallable(name: string) {
  if (localScopes.length === 0) return;
  localScopes[localScopes.length - 1]!.add(name);
}

function collectLocalVariableNames(stmts: Statement[]): string[] {
  const names: string[] = [];
  for (const s of stmts) {
    if (s.kind === "variableDecl") {
      names.push(escapeName(s.name));
    }
  }
  return names;
}

function collectStringLiteralsInExpr(expr: Expr, literals: Set<string>): void {
  if (!expr) return;
  switch (expr.kind) {
    case "string":
      literals.add(expr.value);
      break;
    case "vector":
      for (const el of expr.elements) collectStringLiteralsInExpr(el, literals);
      break;
    case "range":
      collectStringLiteralsInExpr(expr.start, literals);
      collectStringLiteralsInExpr(expr.end, literals);
      if (expr.step) collectStringLiteralsInExpr(expr.step, literals);
      break;
    case "binary":
      collectStringLiteralsInExpr(expr.left, literals);
      collectStringLiteralsInExpr(expr.right, literals);
      break;
    case "unary":
      collectStringLiteralsInExpr(expr.operand, literals);
      break;
    case "ternary":
      collectStringLiteralsInExpr(expr.condition, literals);
      collectStringLiteralsInExpr(expr.ifTrue, literals);
      collectStringLiteralsInExpr(expr.ifFalse, literals);
      break;
    case "call":
      for (const arg of expr.args) collectStringLiteralsInExpr(arg.value, literals);
      break;
    case "index":
      collectStringLiteralsInExpr(expr.object, literals);
      collectStringLiteralsInExpr(expr.index, literals);
      break;
    case "member":
      collectStringLiteralsInExpr(expr.object, literals);
      break;
    case "group":
      collectStringLiteralsInExpr(expr.expr, literals);
      break;
    case "echo":
    case "assert":
      for (const arg of expr.args) collectStringLiteralsInExpr(arg.value, literals);
      collectStringLiteralsInExpr(expr.expr, literals);
      break;
    case "let":
      for (const assign of expr.assignments) collectStringLiteralsInExpr(assign.value, literals);
      collectStringLiteralsInExpr(expr.body, literals);
      break;
    case "each":
      collectStringLiteralsInExpr(expr.expr, literals);
      break;
    case "lambda":
      for (const param of expr.params) {
        if (param.defaultValue) collectStringLiteralsInExpr(param.defaultValue, literals);
      }
      collectStringLiteralsInExpr(expr.body, literals);
      break;
    case "dynCall":
      collectStringLiteralsInExpr(expr.callee, literals);
      for (const arg of expr.args) collectStringLiteralsInExpr(arg.value, literals);
      break;
    case "listComp":
      collectStringLiteralsInGenerator(expr.generator, literals);
      break;
  }
}

function collectStringLiteralsInGenerator(gen: ListCompGenerator, literals: Set<string>): void {
  if (!gen) return;
  switch (gen.kind) {
    case "lcFor":
      for (const v of gen.variables) collectStringLiteralsInExpr(v.range, literals);
      collectStringLiteralsInGenerator(gen.body, literals);
      break;
    case "lcCFor":
      for (const init of gen.inits) collectStringLiteralsInExpr(init.value, literals);
      collectStringLiteralsInExpr(gen.condition, literals);
      for (const update of gen.updates) collectStringLiteralsInExpr(update.value, literals);
      collectStringLiteralsInGenerator(gen.body, literals);
      break;
    case "lcIf":
      collectStringLiteralsInExpr(gen.condition, literals);
      collectStringLiteralsInGenerator(gen.ifTrue, literals);
      if (gen.ifFalse) collectStringLiteralsInGenerator(gen.ifFalse, literals);
      break;
    case "lcLet":
      for (const assign of gen.assignments) collectStringLiteralsInExpr(assign.value, literals);
      collectStringLiteralsInGenerator(gen.body, literals);
      break;
    case "lcExpr":
      collectStringLiteralsInExpr(gen.expr, literals);
      break;
  }
}

function resolveCallArgs(moduleCallName: string, callArgs: Argument[]): Map<string, Expr> {
  const resolved = new Map<string, Expr>();
  const name = `${escapeName(moduleCallName)}$mod`;
  const sig = signatures.get(name);
  if (!sig) return resolved;

  // Initialize with default values
  for (let i = 0; i < sig.params.length; i++) {
    const paramName = sig.params[i]!;
    const defaultVal = sig.defaults[i];
    if (defaultVal) {
      resolved.set(paramName, defaultVal);
    }
  }

  // Map positional arguments
  let pos = 0;
  while (pos < callArgs.length && !callArgs[pos]!.name) {
    if (pos < sig.params.length) {
      resolved.set(sig.params[pos]!, callArgs[pos]!.value);
    }
    pos++;
  }

  // Map named arguments
  for (let i = pos; i < callArgs.length; i++) {
    const a = callArgs[i]!;
    if (a.name) {
      resolved.set(a.name, a.value);
    }
  }

  return resolved;
}

function collectFontRelatedLiterals(program: Program): Set<string> {
  const modulesCallingText = new Set<string>(["text"]);
  
  let changed = true;
  while (changed) {
    changed = false;
    for (const stmt of program.statements) {
      if (stmt.kind === "moduleDecl" && !modulesCallingText.has(stmt.name)) {
        let callsText = false;
        
        const checkCalls = (s: Statement) => {
          if (s.kind === "moduleCall") {
            if (modulesCallingText.has(s.name)) {
              callsText = true;
            }
            if (s.child) checkCalls(s.child);
          } else if (s.kind === "block") {
            for (const sub of s.statements) checkCalls(sub);
          } else if (s.kind === "for") {
            checkCalls(s.body);
          } else if (s.kind === "if") {
            checkCalls(s.thenBody);
            if (s.elseBody) checkCalls(s.elseBody);
          }
        };
        
        checkCalls(stmt.body);
        if (callsText) {
          modulesCallingText.add(stmt.name);
          changed = true;
        }
      }
    }
  }

  const literals = new Set<string>();

  // Helper to collect all string literals from an expression
  const collectExpr = (expr: Expr) => {
    if (!expr) return;
    collectStringLiteralsInExpr(expr, literals);
  };

  // Traverse AST to collect literals from related definitions and invocations
  const traverse = (s: Statement, insideTextModule: boolean) => {
    if (!s) return;
    
    if (s.kind === "moduleDecl") {
      const isTextMod = modulesCallingText.has(s.name);
      if (isTextMod) {
        for (const p of s.params) {
          if (p.defaultValue) collectExpr(p.defaultValue);
        }
      }
      traverse(s.body, isTextMod);
    } else if (s.kind === "moduleCall") {
      if (modulesCallingText.has(s.name)) {
        // Resolve arguments including default parameter values if omitted
        const argMap = resolveCallArgs(s.name, s.args);
        for (const [paramName, expr] of argMap.entries()) {
          collectExpr(expr);
        }
      }
      if (s.child) traverse(s.child, insideTextModule);
    } else if (s.kind === "variableDecl") {
      const nameLower = s.name.toLowerCase();
      const isRelated = insideTextModule || nameLower.includes("font") || nameLower.includes("style") || nameLower.includes("family");
      if (isRelated) {
        collectExpr(s.value);
      }
    } else if (s.kind === "block") {
      for (const sub of s.statements) traverse(sub, insideTextModule);
    } else if (s.kind === "for") {
      traverse(s.body, insideTextModule);
    } else if (s.kind === "if") {
      traverse(s.thenBody, insideTextModule);
      if (s.elseBody) traverse(s.elseBody, insideTextModule);
    }
  };

  for (const stmt of program.statements) {
    traverse(stmt, false);
  }

  return literals;
}

// Main entry
export function compile(program: Program, options?: { runtimePath?: string }): string {
  currentRuntimePath = options?.runtimePath ?? "./runtime/runtime.js";
  dynamicScopeVars = new Set();
  encounteredFonts = new Set();
  encounteredImages.clear();
  signatures.clear();
  for (const [k, v] of Object.entries(BUILTIN_SIGNATURES)) {
    signatures.set(k, { params: v, defaults: new Array(v.length).fill(undefined) });
  }
  collectSignatures(program.statements);
  moduleDeclRegistry = collectModuleDeclarations(program.statements);

  // Gather all font-related string literals from the program
  const fontLiterals = collectFontRelatedLiterals(program);

  // Scan FONTPATH and match fonts
  const fontDir = getFontPath();
  if (fontDir && fs.existsSync(fontDir)) {
    try {
      const files = fs.readdirSync(fontDir);
      const cleanedLiterals = Array.from(fontLiterals).map(lit =>
        lit.toLowerCase().replace(/[^a-z0-9]/g, "")
      );
      
      for (const file of files) {
        const ext = path.extname(file).toLowerCase();
        if (ext === ".ttf" || ext === ".otf") {
          const basename = path.basename(file, ext);
          const dashIdx = basename.indexOf("-");
          const family = dashIdx >= 0 ? basename.slice(0, dashIdx) : basename;
          const style = dashIdx >= 0 ? basename.slice(dashIdx + 1) : "Regular";

          const cleanedFamily = family.toLowerCase().replace(/[^a-z0-9]/g, "");
          const cleanedStyle = style.toLowerCase().replace(/[^a-z0-9]/g, "");

          // Check if family matches any of the cleaned literals
          const familyMatched = cleanedLiterals.some(lit =>
            lit.includes(cleanedFamily) || (lit.length >= 4 && cleanedFamily.includes(lit))
          );
          if (familyMatched) {
            const styleMatched = cleanedStyle === "regular" || cleanedLiterals.some(lit => {
              if (cleanedStyle === "bolditalic") {
                return (lit.includes("bold") && lit.includes("italic")) || lit.includes("bolditalic");
              }
              return lit.includes(cleanedStyle);
            });
            if (styleMatched) {
              encounteredFonts.add(basename);
            }
          }
        }
      }
    } catch (e) {
      console.warn("Warning: failed to read font directory for matching:", e);
    }
  }

  // Build declarations, deduplicating by output name (last wins, matching OpenSCAD semantics)
  const declMap = new Map<string, { stmt: Statement; code: string }>();
  const declOrder: string[] = [];
  const geometryLines: string[] = [];

  let lastGeoFilename = "";
  for (const stmt of program.statements) {
    if (
      stmt.kind === "variableDecl" ||
      stmt.kind === "moduleDecl" ||
      stmt.kind === "functionDecl"
    ) {
      // Compute the output name to detect duplicates
      let key: string;
      if (stmt.kind === "variableDecl") key = `var:${escapeName(stmt.name)}`;
      else if (stmt.kind === "moduleDecl") key = `fn:${escapeName(stmt.name)}$mod`;
      else key = `fn:${escapeName(stmt.name)}_fn`;

      if (!declMap.has(key)) declOrder.push(key);
      declMap.set(key, { stmt, code: compileDeclaration(stmt) });
    } else if (stmt.kind === "use" || stmt.kind === "include") {
      const key = `comment:${stmt.path}`;
      if (!declMap.has(key)) declOrder.push(key);
      declMap.set(key, { stmt, code: `// ${stmt.kind} <${stmt.path}>` });
    } else {
      const geo = compileGeometry(stmt);
      if (geo) {
        const filename = stmt.filename;
        if (filename && filename !== lastGeoFilename) {
          const relativePath = path.relative(process.cwd(), filename).replace(/\\/g, "/");
          geometryLines.push(`\n// ${relativePath}`);
          lastGeoFilename = filename;
        }

        if (stmt.kind === "moduleCall" && !stmt.modifier) {
          if (isModuleCallBackgroundOnly(stmt, moduleDeclRegistry)) {
            pushCommentedLine(geometryLines, stmt, `__background_items.push(${geo});`);
          } else {
            pushCommentedLine(geometryLines, stmt, `__result_items.push(${geo});`);
          }
        } else if (stmt.kind === "moduleCall" && stmt.modifier === "%") {
          pushCommentedLine(geometryLines, stmt, `__background_items.push(${geo});`);
        } else {
          pushCommentedLine(geometryLines, stmt, `__result_items.push(${geo});`);
        }
      }
    }
  }

  const declarations: string[] = [];
  let lastFilename = "";
  for (const k of declOrder) {
    const entry = declMap.get(k)!;
    const filename = entry.stmt.filename;
    if (filename && filename !== lastFilename) {
      const relativePath = path.relative(process.cwd(), filename).replace(/\\/g, "/");
      declarations.push(`\n// ${relativePath}`);
      lastFilename = filename;
    }
    declarations.push(entry.code);
  }

  const currentFileDir = typeof __dirname !== "undefined"
    ? __dirname
    : path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/i, "$1"));
  const compilerDir = path.resolve(currentFileDir, "..");
  const fontImports: string[] = [];
  const resolvedFonts = new Map<string, string>(); // fontFamily → sanitized name (if resolved)

  for (const fontFamily of encounteredFonts) {
    const sanitized = generateFontBase64(fontFamily, compilerDir);
    if (sanitized) {
      resolvedFonts.set(fontFamily, sanitized);
    }
  }

  const RUNTIME_IMPORT =
    `import * as __rt from "${options?.runtimePath ?? "./runtime/runtime.js"}";\n` +
    `const { __sphere, __cylinder, __circle, __polygon, Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, ` +
    `sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, ` +
    `min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, parent_module_fn, openscad_assert_fn, ` +
    `__eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, ` +
    `__children_stack, __with_children, ` +
    `__is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve, __rotate, __translate, __scale, __mirror, __text, __parse_color_for_scope, __surface } = __rt;\n`;

  let output = RUNTIME_IMPORT;

  // Add font base64 imports for each resolved font.
  const seenImports = new Set<string>();
  for (const [fontFamily, sanitized] of resolvedFonts) {
    if (seenImports.has(sanitized)) continue;
    seenImports.add(sanitized);
    const runtimeDir = options?.runtimePath ? path.dirname(options.runtimePath).replace(/\\/g, "/") : "./runtime";
    const importPath = `${runtimeDir}/fonts/${sanitized}_base64.js`;
    const varName = `__font_${sanitized.replace(/-/g, "_")}`;
    output += `import { fontBase64 as ${varName} } from "${importPath}";\n`;
  }

  // Add image base64 imports for each resolved image.
  for (const [filename, info] of encounteredImages) {
    const runtimeDir = options?.runtimePath ? path.dirname(options.runtimePath).replace(/\\/g, "/") : "./runtime";
    const importPath = `${runtimeDir}/images/${info.stem}_base64.js`;
    output += `import { ${info.exportName} } from "${importPath}";\n`;
  }
  
  output += `const __font_registry: Record<string, string> = {\n`;
  const seenSanitized = new Set<string>();
  for (const [fontFamily, sanitized] of resolvedFonts) {
    if (seenSanitized.has(sanitized)) continue;
    seenSanitized.add(sanitized);
    const varName = `__font_${sanitized.replace(/-/g, "_")}`;
    output += `  ${JSON.stringify(sanitized)}: ${varName},\n`;
  }
  output += `};\n\n`;

  output += `var PI: any = __rt.PI;\n` +
    `var INF: any = __rt.INF;\n` +
    `var NAN: any = __rt.NAN;\n` +
    `var undef: any = __rt.undef;\n` +
    `var _EPSILON: any = __rt._EPSILON;\n\n`;

  // Add minimum special variables before user declarations
  output += `var $fn: any = 0, $fa: any = 12, $fs: any = 2;\n`;
  output += `var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500, $vpf: any = 22.5;\n`;
  output += `var $parent_modules: any = 0;\n`;
  output += `var $t: any = 0, $preview: any = false;\n`;
  output += `var $color: any = undefined;\n`;
  output += `var $idx: any = undefined;\n`;

  if (declarations.length) {
    output += declarations.join("\n") + "\n\n";
  }

  // Declare any $ variables that were only set inside modules/blocks
  const topLevelVarKeys = new Set(declOrder.filter(k => k.startsWith("var:")).map(k => k.slice(4)));
  for (const v of dynamicScopeVars) {
    if (!topLevelVarKeys.has(v)) {
      output += `var ${v}: any;\n`;
    }
  }

  // Provide a global container for user variables and helpers
  output += `var _NO_ARG: any = Symbol("NO_ARG");\n`;

  if (geometryLines.length === 0) {
    output += `export const result = Manifold.union([]);\n`;
  } else {
    output += `const __result_items: any[] = [];\n`;
    output += `const __background_items: any[] = [];\n`;
    output += `${geometryLines.join("\n")}\n`;
    output += `export const result = __union2d3d(__result_items);\n`;
    output += `export const background = __union2d3d(__background_items);\n`;
  }
  output += `export const __viewport = { vpr: $vpr, vpt: $vpt, vpd: $vpd, vpf: $vpf };\n`;

  return output;
}

// build geometry IR trees from top level statements
export function buildProgramIR(program: Program): IRNode[] {
  const modules = collectModuleDeclarations(program.statements);
  const ctx = baseIRContext(modules);
  const out: IRNode[] = [];
  for (const stmt of program.statements) {
    const ir = lowerGeometryToIR(stmt, ctx);
    if (ir && ir.kind !== "empty") out.push(ir);
  }
  return out;
}

// Declarations
const PRE_DECLARED_VARS = new Set([
  "$fn", "$fa", "$fs",
  "$vpr", "$vpt", "$vpd", "$vpf",
  "$parent_modules",
  "$t", "$preview",
  "PI", "INF", "NAN", "undef", "_EPSILON"
]);

function compileDeclaration(stmt: Statement): string {
  const withLeading = (code: string) => {
    const leading = leadingCommentLines(stmt);
    const suffix = trailingCommentText(stmt);
    return `${leading.length ? `${leading.join("\n")}\n` : ""}${code}${suffix}`;
  };

  switch (stmt.kind) {
    case "variableDecl": {
      const name = escapeName(stmt.name);
      if (PRE_DECLARED_VARS.has(stmt.name)) {
        return withLeading(`${name} = ${compileExpr(stmt.value)};`);
      }
      return withLeading(`var ${name}: any = ${compileExpr(stmt.value)};`);
    }

    case "moduleDecl": {
      const params = compileParams(stmt.params);
      const localParams = deduplicateParams(stmt.params).map(p => escapeName(p.name));
      const body = compileModuleBody(stmt.body, stmt.name, localParams);
      return withLeading(`function ${escapeName(stmt.name)}$mod(${params}): any {\n${body}\n}`);
    }

    case "functionDecl": {
      const params = compileParams(stmt.params);
      const localParams = deduplicateParams(stmt.params).map(p => escapeName(p.name));
      const bodyExpr = withLocalScope(localParams, () => compileExpr(stmt.body));
      return withLeading(`function ${escapeName(stmt.name)}_fn(${params}): any {\n  return ${bodyExpr};\n}`);
    }

    default:
      return `/* unsupported declaration: ${(stmt as Statement).kind}${locTag(stmt)} */`;
  }
}

// Deduplicate parameters: keep last occurrence of each name (OpenSCAD allows duplicates)
function deduplicateParams(params: import("./ast.js").Parameter[]): import("./ast.js").Parameter[] {
  const seen = new Map<string, number>();
  for (let i = 0; i < params.length; i++) {
    seen.set(params[i]!.name, i);
  }
  return params.filter((p, i) => seen.get(p.name) === i);
}

function compileParams(params: import("./ast.js").Parameter[]): string {
  return deduplicateParams(params)
    .map((p) =>
      p.defaultValue
        ? `${escapeName(p.name)}: any = ${compileExpr(p.defaultValue)}`
        : `${escapeName(p.name)}: any`
    )
    .join(", ");
}

// Module body compilation
function compileModuleBody(body: Statement, moduleName?: string, localParamNames: string[] = []): string {
  const stmts = body.kind === "block" ? body.statements : [body];
  const localVarNames = collectLocalVariableNames(stmts);

  const lines: string[] = [];

  // Capture children from the stack at the start of every module body
  lines.push("  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };");
  lines.push("  var $children: any = __c.count;");
  lines.push("  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }");
  lines.push("  var __save_$parent_modules: any = $parent_modules;");
  lines.push("  $parent_modules = __children_stack.length;");
  lines.push("  const __items: any[] = [];");

  const decls: string[] = [];
  const geos: string[] = [];
  const dollarSaves: string[] = [];
  const dollarRestores: string[] = [];

  withLocalScope([...localParamNames, ...localVarNames], () => {
    for (const s of stmts) {
      if (s.kind === "empty") continue;
      if (s.kind === "variableDecl") {
        const name = escapeName(s.name);
        const valueExpr = compileExpr(s.value);
        const commentsBefore = leadingCommentLines(s, "  ");
        const commentAfter = trailingCommentText(s);
        if (s.name.startsWith("$") && s.name !== "$children") {
          // Dynamic scoping: save/assign/restore for $ variables
          dynamicScopeVars.add(name);
          dollarSaves.push(`  var __save_${name}: any = ${name};`);
          decls.push(...commentsBefore);
          decls.push(`  ${name} = ${valueExpr};${commentAfter}`);
          dollarRestores.push(`  ${name} = __save_${name};`);
        } else {
          decls.push(...commentsBefore);
          decls.push(`  var ${name}: any = ${valueExpr};${commentAfter}`);
        }
      } else if (s.kind === "functionDecl" || s.kind === "moduleDecl") {
        // Indent the nested declaration
        const decl = compileDeclaration(s);
        decls.push("  " + decl.split("\n").join("\n  "));
      } else {
        const geo = compileGeometry(s);
        if (geo) pushCommentedLine(geos, s, `  __items.push(${geo});`, "  ");
      }
    }
  });

  if (dollarRestores.length > 0) {
    lines.splice(4, 0, ...dollarSaves);
  }

  lines.push(...decls);
  lines.push(...geos);

  lines.push(`  try {`);
  lines.push(`    return __union2d3d(__items);`);
  lines.push(`  } finally {`);
  if (dollarRestores.length > 0) {
    lines.push(...dollarRestores.map(r => `  ${r}`));
  }
  lines.push(`    $parent_modules = __save_$parent_modules;`);   // ← restore
  lines.push(`  }`);
  
  return lines.join("\n");
}

function isStatementBackgroundOnly(stmt: Statement, modules: Map<string, ModuleDeclStmtType>, visited: Set<string>): boolean {
  switch (stmt.kind) {
    case "empty":
    case "variableDecl":
    case "moduleDecl":
    case "functionDecl":
    case "use":
    case "include":
      return true;
    case "block":
      return stmt.statements.every(s => isStatementBackgroundOnly(s, modules, visited));
    case "for":
      return isStatementBackgroundOnly(stmt.body, modules, visited);
    case "if":
      return isStatementBackgroundOnly(stmt.thenBody, modules, visited) &&
             (!stmt.elseBody || isStatementBackgroundOnly(stmt.elseBody, modules, visited));
    case "moduleCall":
      if (stmt.modifier === "%") {
        return true;
      }
      if (modules.has(stmt.name)) {
        if (visited.has(stmt.name)) {
          return true;
        }
        visited.add(stmt.name);
        const decl = modules.get(stmt.name)!;
        const res = isStatementBackgroundOnly(decl.body, modules, visited);
        visited.delete(stmt.name);
        return res;
      }
      return false;
    default:
      return false;
  }
}

function isModuleCallBackgroundOnly(stmt: ModuleCallStmt, modules: Map<string, ModuleDeclStmtType>): boolean {
  if (stmt.modifier === "%") return true;
  return isStatementBackgroundOnly(stmt, modules, new Set<string>());
}

// Geometry compilation
function compileGeometry(stmt: Statement): string {
  return compileGeometryLegacy(stmt);
}

const IR_PRIMITIVES = new Set([
  "cube", "sphere", "cylinder", "circle", "square", "polygon", "polyhedron", "text", "surface",
]);

const IR_TRANSFORMS = new Set([
  "translate", "rotate", "scale", "mirror", "multmatrix", "resize", "offset", "color", "render", "projection",
]);

const IR_BOOLEANS = new Set(["union", "difference", "intersection", "hull", "minkowski"]);

function wrapWithLetBindings(node: IRNode, bindings: Argument[], loc?: ASTNode["loc"]): IRNode {
  if (bindings.length === 0) return node;
  return {
    kind: "moduleCall",
    name: "let",
    args: bindings,
    children: [node],
    loc,
  } as IRModuleCallNode;
}

function resolveChildrenReference(
  args: Argument[],
  boundChildren: IRNode[],
  loc?: ASTNode["loc"],
): IRNode {
  if (boundChildren.length === 0) return { kind: "empty", loc };

  const iArg = findArg(args, "i", 0);
  if (!iArg) {
    if (boundChildren.length === 1) return boundChildren[0]!;
    return { kind: "sequence", items: boundChildren, loc } as IRSequenceNode;
  }

  if (iArg.value.kind === "number") {
    const idx = Math.trunc(iArg.value.value);
    if (idx < 0 || idx >= boundChildren.length) return { kind: "empty", loc };
    return boundChildren[idx]!;
  }

  let out: IRNode = { kind: "empty", loc };
  for (let idx = boundChildren.length - 1; idx >= 0; idx--) {
    out = {
      kind: "if",
      condition: {
        kind: "binary",
        op: "==",
        left: iArg.value,
        right: { kind: "number", value: idx },
      },
      thenNode: boundChildren[idx]!,
      elseNode: out,
      loc,
    } as IRIfNode;
  }
  return out;
}

function buildModuleParamBindings(
  params: import("./ast.js").Parameter[],
  callArgs: Argument[],
): Argument[] {
  const deduped = deduplicateParams(params);
  const bound = new Map<string, Expr>();

  for (const p of deduped) {
    bound.set(p.name, p.defaultValue ?? { kind: "undef" });
  }

  let pos = 0;
  while (pos < callArgs.length && !callArgs[pos]!.name) {
    if (pos < deduped.length) {
      bound.set(deduped[pos]!.name, callArgs[pos]!.value);
    }
    pos++;
  }

  for (let i = pos; i < callArgs.length; i++) {
    const a = callArgs[i]!;
    if (a.name && bound.has(a.name)) {
      bound.set(a.name, a.value);
    }
  }

  return deduped.map((p) => ({ name: p.name, value: bound.get(p.name)! }));
}

function tryExpandUserModuleCallToIR(
  stmt: ModuleCallStmt,
  decl: ModuleDeclStmtType,
  loweredChildren: IRNode[],
  ctx: IRLowerContext,
): IRNode | undefined {
  const paramBindings = buildModuleParamBindings(decl.params, stmt.args);
  const innerCtx: IRLowerContext = {
    modules: ctx.modules,
    children: loweredChildren,
    callStack: [...ctx.callStack, decl.name],
  };

  const loweredBody = lowerGeometryToIR(decl.body, innerCtx);
  if (!loweredBody) return undefined;
  return wrapWithLetBindings(loweredBody, paramBindings, stmt.loc);
}

function lowerGeometryToIR(stmt: Statement, ctx: IRLowerContext): IRNode | undefined {
  switch (stmt.kind) {
    case "moduleCall":
      return lowerModuleCallToIR(stmt, ctx);
    case "block": {
      const items: IRNode[] = [];
      const letBindings: Argument[] = [];
      let activeModules = new Map(ctx.modules);

      for (const s of stmt.statements) {
        if (s.kind === "empty" || s.kind === "use" || s.kind === "include") continue;

        if (s.kind === "variableDecl") {
          letBindings.push({ name: s.name, value: s.value });
          continue;
        }

        if (s.kind === "moduleDecl") {
          activeModules.set(s.name, s);
          continue;
        }

        if (s.kind === "functionDecl") {
          return undefined;
        }

        const lowered = lowerGeometryToIR(s, { ...ctx, modules: activeModules });
        if (!lowered) return undefined;
        if (lowered.kind !== "empty") items.push(wrapWithLetBindings(lowered, letBindings, s.loc));
      }
      return { kind: "sequence", items, loc: stmt.loc } as IRSequenceNode;
    }
    case "for": {
      const body = lowerGeometryToIR(stmt.body, ctx);
      if (!body) return undefined;
      return { kind: "for", variables: stmt.variables, body, loc: stmt.loc } as IRForNode;
    }
    case "if": {
      const thenNode = lowerGeometryToIR(stmt.thenBody, ctx);
      if (!thenNode) return undefined;
      const elseNode = stmt.elseBody ? lowerGeometryToIR(stmt.elseBody, ctx) : undefined;
      if (stmt.elseBody && !elseNode) return undefined;
      return { kind: "if", condition: stmt.condition, thenNode, elseNode, loc: stmt.loc } as IRIfNode;
    }
    case "empty":
      return { kind: "empty", loc: stmt.loc };
    case "variableDecl":
    case "moduleDecl":
    case "functionDecl":
    case "use":
    case "include":
      return { kind: "empty", loc: stmt.loc };
    default:
      return undefined;
  }
}

function lowerModuleCallToIR(stmt: ModuleCallStmt, ctx: IRLowerContext): IRNode | undefined {
  const name = stmt.name;
  const children = lowerModuleChildrenToIR(stmt.child, ctx);
  if (stmt.child && !children) return undefined;
  const loweredChildren = children ?? [];

  if (!ctx.callStack.includes(name)) {
    const decl = ctx.modules.get(name);
    if (decl && shouldInlineModuleToIR(decl, ctx)) {
      const expanded = tryExpandUserModuleCallToIR(stmt, decl, loweredChildren, ctx);
      if (expanded) return expanded;
    }
  }

  if (name === "children") {
    return resolveChildrenReference(stmt.args, ctx.children, stmt.loc);
  }

  if (IR_PRIMITIVES.has(name)) {
    return {
      kind: "primitive",
      primitive: name as IRPrimitiveNode["primitive"],
      args: stmt.args,
      loc: stmt.loc,
    };
  }

  if (IR_TRANSFORMS.has(name)) {
    const child = loweredChildren.length === 1
      ? loweredChildren[0]!
      : ({ kind: "sequence", items: loweredChildren, loc: stmt.loc } as IRSequenceNode);
    return {
      kind: "transform",
      transform: name as IRTransformNode["transform"],
      args: stmt.args,
      child,
      loc: stmt.loc,
    };
  }

  if (IR_BOOLEANS.has(name)) {
    return {
      kind: "boolean",
      op: name as IRBooleanNode["op"],
      children: loweredChildren,
      loc: stmt.loc,
    };
  }

  return {
    kind: "moduleCall",
    name,
    args: stmt.args,
    children: loweredChildren,
    loc: stmt.loc,
  } as IRModuleCallNode;
}

function lowerModuleChildrenToIR(child: Statement | undefined, ctx: IRLowerContext): IRNode[] | undefined {
  if (!child || child.kind === "empty") return [];
  if (child.kind === "block") {
    const items: IRNode[] = [];
    const letBindings: Argument[] = [];
    let activeModules = new Map(ctx.modules);

    for (const s of child.statements) {
      if (s.kind === "empty" || s.kind === "use" || s.kind === "include") continue;

      if (s.kind === "variableDecl") {
        letBindings.push({ name: s.name, value: s.value });
        continue;
      }

      if (s.kind === "moduleDecl") {
        activeModules.set(s.name, s);
        continue;
      }

      if (s.kind === "functionDecl") {
        return undefined;
      }

      const lowered = lowerGeometryToIR(s, { ...ctx, modules: activeModules });
      if (!lowered) return undefined;
      if (lowered.kind !== "empty") {
        items.push(wrapWithLetBindings(lowered, letBindings, s.loc));
      }
    }
    return items;
  }
  const lowered = lowerGeometryToIR(child, ctx);
  if (!lowered) return undefined;
  return lowered.kind === "empty" ? [] : [lowered];
}

function compileIRNode(node: IRNode): string {
  switch (node.kind) {
    case "empty":
      return "Manifold.union([])";

    case "primitive":
      return compileIRPrimitive(node);

    case "transform":
      return compileIRTransform(node);

    case "boolean":
      return compileIRBoolean(node);

    case "moduleCall":
      return compileIRModuleCall(node);

    case "children":
      return node.indexExpr ? `children(${compileExpr(node.indexExpr)})` : "children()";

    case "sequence": {
      const items = node.items
        .map(compileIRNode)
        .filter(x => x && x !== "Manifold.union([])");
      if (items.length === 0) return "Manifold.union([])";
      if (items.length === 1) return items[0]!;
      return `__union2d3d([\n  ${items.join(",\n  ")}\n])`;
    }

    case "if": {
      const cond = compileExpr(node.condition);
      const thenNode = compileIRNode(node.thenNode);
      const elseNode = node.elseNode ? compileIRNode(node.elseNode) : "Manifold.union([])";
      return `(${cond} ? ${thenNode} : ${elseNode})`;
    }

    case "for":
      return buildNestedFor(node.variables, 0, compileIRNode(node.body));

    case "astFallback":
      return compileGeometryLegacy(node.statement);

    default:
      return `/* unsupported ir node */`;
  }
}

function compileIRPrimitive(node: IRPrimitiveNode): string {
  switch (node.primitive) {
    case "cube": return compileCube(node.args);
    case "sphere": return compileSphere(node.args);
    case "cylinder": return compileCylinder(node.args);
    case "circle": return compileCircle(node.args);
    case "square": return compileSquare(node.args);
    case "polygon": return compilePolygon(node.args);
    case "polyhedron": return compilePolyhedron(node.args);
    case "text": return compileText(node.args);
    case "surface": return compileSurface(node.args);
    default: return "/* unsupported primitive */";
  }
}

function compileIRTransform(node: IRTransformNode): string {
  const child = compileIRNode(node.child);
  switch (node.transform) {
    case "translate":
      return `__translate(${child}, ${node.args[0] ? compileExpr(node.args[0].value) : "[0, 0, 0]"})`;
    case "rotate": {
      const a = node.args[0] ? compileExpr(node.args[0].value) : "undefined";
      const v = node.args[1] ? compileExpr(node.args[1].value) : "undefined";
      return `__rotate(${child}, ${a}, ${v})`;
    }
    case "scale":
      return `__scale(${child}, ${node.args[0] ? compileExpr(node.args[0].value) : "[1, 1, 1]"})`;
    case "mirror":
      return `__mirror(${child}, ${node.args[0] ? compileExpr(node.args[0].value) : "[1, 0, 0]"})`;
    case "multmatrix":
      return node.args[0] ? `__safe_transform(${child}, ${compileExpr(node.args[0].value)})` : child;
    case "resize":
      return `/* resize(${node.args.map(a => compileExpr(a.value)).join(", ")}) */ ${child}`;
    case "offset": {
      const r = findArg(node.args, "r", 0);
      const delta = findArg(node.args, "delta");
      const amount = r ?? delta;
      const amt = amount ? compileExpr(amount.value) : "0";
      return `__safe_offset2d(${child}, ${amt}, "Round", 2, $fn, $fa, $fs)`;
    }
    case "color": {
      const c = findArg(node.args, "c", 0);
      const alpha = findArg(node.args, "alpha", 1);
      const cExpr = c ? compileExpr(c.value) : "undefined";
      const aExpr = alpha ? compileExpr(alpha.value) : "undefined";
      return `(() => { var __save_$color: any = $color; $color = __parse_color_for_scope(${cExpr}, ${aExpr}); try { return __apply_color(${child}, ${cExpr}, ${aExpr}); } finally { $color = __save_$color; } })()`;
    }
    case "render":
      return `/* render(${node.args.map(a => compileExpr(a.value)).join(", ")}) */ ${child}`;
    case "projection": {
      const cut = findArg(node.args, "cut", 0);
      const cutStr = cut ? compileExpr(cut.value) : "false";
      return `__safe_project3d(${child}, ${cutStr})`;
    }
    default:
      return child;
  }
}

function compileIRBoolean(node: IRBooleanNode): string {
  const children = node.children
    .map(compileIRNode)
    .filter(x => x && x !== "Manifold.union([])");

  if (children.length === 0) return "Manifold.union([])";
  if (children.length === 1) return children[0]!;

  switch (node.op) {
    case "union":
      return `__union2d3d([\n  ${children.join(",\n  ")}\n])`;
    case "difference": {
      const [first, ...rest] = children;
      return `__difference2d3d(${first}, [\n  ${rest.join(",\n  ")}\n])`;
    }
    case "intersection":
      return `__intersection2d3d([\n  ${children.join(",\n  ")}\n])`;
    case "hull":
      return `__hull2d3d([\n  ${children.join(",\n  ")}\n])`;
    case "minkowski":
      return `__minkowski2d3d([\n  ${children.join(",\n  ")}\n])`;
    default:
      return `Manifold.union([\n  ${children.join(",\n  ")}\n])`;
  }
}

function buildWithChildrenCall(callExpr: string, children: string[], moduleName: string): string {
  if (children.length === 0) {
    return `__with_children(() => Manifold.union([]), 0, () => ${callExpr}, ${JSON.stringify(moduleName)})`;
  }

  const childrenCode = children.map(child => `() => (${child})`).join(",\n  ");
  const hasAwait = childrenCode.includes("await ") || callExpr.includes("await ");
  
  if (hasAwait) {
    return `await (() => { ` +
      `const __childFns = [\n  ${children.map(child => `async () => (${child})`).join(",\n  ")}\n]; ` +
      `return __with_children(async (i) => (` +
      `i === undefined ? __union2d3d(await Promise.all(__childFns.map(fn => fn()))) : ` +
      `((i >= 0 && i < __childFns.length) ? await __childFns[i]() : Manifold.union([]))` +
      `), __childFns.length, async () => await ${callExpr}, ${JSON.stringify(moduleName)}); ` +
      `})()`;
  }

  return `(() => { ` +
    `const __childFns = [\n  ${children.map(child => `() => (${child})`).join(",\n  ")}\n]; ` +
    `return __with_children((i) => (` +
    `i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ` +
    `((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))` +
    `), __childFns.length, () => ${callExpr}, ${JSON.stringify(moduleName)}); ` +
    `})()`;
}

function compileIRModuleCall(node: IRModuleCallNode): string {
  switch (node.name) {
    case "linear_extrude": {
      const child = node.children.length === 0
        ? "Manifold.union([])"
        : node.children.length === 1
          ? compileIRNode(node.children[0]!)
          : `__union2d3d([\n  ${node.children.map(compileIRNode).join(",\n  ")}\n])`;
      const height = findArg(node.args, "height", 0);
      const hStr = height ? compileExpr(height.value) : "1";
      const twist = findArg(node.args, "twist");
      const slices = findArg(node.args, "slices");
      const scale = findArg(node.args, "scale");
      const center = findArg(node.args, "center");

      const fn = findArg(node.args, "$fn");
      const fa = findArg(node.args, "$fa");
      const fs = findArg(node.args, "$fs");

      const opts: string[] = [];

      if (twist) opts.push(`twist: ${compileExpr(twist.value)}`);
      if (scale) opts.push(`scale: ${compileExpr(scale.value)}`);
      if (center) opts.push(`center: ${compileExpr(center.value)}`);

      opts.push(
        `fn: ${fn ? compileExpr(fn.value) : "$fn"}`
      );

      opts.push(
        `fa: ${fa ? compileExpr(fa.value) : "$fa"}`
      );

      opts.push(
        `fs: ${fs ? compileExpr(fs.value) : "$fs"}`
      );

      if (slices) {
        opts.push(`slices: ${compileExpr(slices.value)}`);
      }
      return opts.length
        ? `__extrude(${child}, ${hStr}, { ${opts.join(", ")} })`
        : `__extrude(${child}, ${hStr})`;
    }

    case "rotate_extrude": {
      const child = node.children.length === 0
        ? "Manifold.union([])"
        : node.children.length === 1
          ? compileIRNode(node.children[0]!)
          : `__union2d3d([\n  ${node.children.map(compileIRNode).join(",\n  ")}\n])`;
      const angle = findArg(node.args, "angle", 0);
      const aStr = angle ? compileExpr(angle.value) : "360";
      const fn = findArg(node.args, "$fn");
      const fa = findArg(node.args, "$fa");
      const fs = findArg(node.args, "$fs");
      const fnStr = fn ? compileExpr(fn.value) : "$fn";
      const faStr = fa ? compileExpr(fa.value) : "$fa";
      const fsStr = fs ? compileExpr(fs.value) : "$fs";
      return `__revolve(${child}, ${fnStr}, ${faStr}, ${fsStr}, ${aStr})`;
    }

    case "echo": {
      const args = node.args.map(a =>
        a.name ? `"${a.name} = ", ${compileExpr(a.value)}` : compileExpr(a.value)
      ).join(", ");
      const child = node.children.length === 0
        ? "Manifold.union([])"
        : node.children.length === 1
          ? compileIRNode(node.children[0]!)
          : `__union2d3d([\n  ${node.children.map(compileIRNode).join(",\n  ")}\n])`;
      return `(console.log(${args}), ${child})`;
    }

    case "assert": {
      const condition = node.args[0] ? compileExpr(node.args[0].value) : "true";
      const message = node.args[1] ? compileExpr(node.args[1].value) : '"Assertion failed"';
      const child = node.children.length === 0
        ? "Manifold.union([])"
        : node.children.length === 1
          ? compileIRNode(node.children[0]!)
          : `__union2d3d([\n  ${node.children.map(compileIRNode).join(",\n  ")}\n])`;
      return `(openscad_assert_fn(${condition}, ${message}), ${child})`;
    }

    case "let": {
      let child = node.children.length === 0
        ? "Manifold.union([])"
        : node.children.length === 1
          ? compileIRNode(node.children[0]!)
          : `__union2d3d([\n  ${node.children.map(compileIRNode).join(",\n  ")}\n])`;
      for (let i = node.args.length - 1; i >= 0; i--) {
        const a = node.args[i]!;
        const name = a.name ? escapeName(a.name) : "_";
        if (child.includes("await ")) {
          child = `await (async (${name}) => (${child}))(${compileExpr(a.value)})`;
        } else {
          child = `((${name}) => (${child}))(${compileExpr(a.value)})`;
        }
      }
      return child;
    }

    case "intersection_for": {
      const variables: ForVariable[] = node.args.map(arg => ({
        name: arg.name || "_",
        range: arg.value,
        loc: arg.loc,
      }));
      const body = node.children.length === 0
        ? "Manifold.union([])"
        : node.children.length === 1
          ? compileIRNode(node.children[0]!)
          : `__union2d3d([\n  ${node.children.map(compileIRNode).join(",\n  ")}\n])`;
      return buildNestedIntersectionFor(variables, 0, body);
    }

    default: {
      if (!moduleDeclRegistry.has(node.name)) {
        const line = node.loc?.start.line;
        const where = line ? ` at line ${line}` : "";
        console.warn(`Warning: Ignoring unknown module '${node.name}'${where}`);
        return "Manifold.union([])";
      }

      const callName = `${escapeName(node.name)}$mod`;
      const argList = compileArgList(callName, node.args);
      const children = node.children.map(compileIRNode).filter(Boolean);
      return buildWithChildrenCall(`${callName}(${argList})`, children, node.name);
    }
  }
}

function compileGeometryLegacy(stmt: Statement): string {
  switch (stmt.kind) {
    case "moduleCall":
      return compileModuleCall(stmt);
    case "block":
      return compileBlockGeometry(stmt);
    case "for":
      return compileForGeometry(stmt);
    case "if":
      return compileIfGeometry(stmt);
    case "empty":
      return "";
    case "variableDecl":
    case "moduleDecl":
    case "functionDecl":
      // Standalone declarations in geometry context are handled by
      // compileBlockGeometry and compileModuleBody. If we get here,
      // it means a declaration appeared outside a block (unusual).
      return "";
    case "use":
    case "include":
      return "";
    default:
      return `/* unsupported: ${(stmt as Statement).kind}${locTag(stmt)} */`;
  }
}

// Module call dispatch
function compileModuleCall(stmt: ModuleCallStmt): string {
  switch (stmt.name) {
    // Primitives
    case "cube": return compileCube(stmt.args);
    case "sphere": return compileSphere(stmt.args);
    case "cylinder": return compileCylinder(stmt.args);
    case "circle": return compileCircle(stmt.args);
    case "square": return compileSquare(stmt.args);
    case "polygon": return compilePolygon(stmt.args);
    case "polyhedron": return compilePolyhedron(stmt.args);
    case "text": return compileText(stmt.args);
    case "surface": return compileSurface(stmt.args);

    // Transforms
    case "translate": return compileTransform(stmt, "translate");
    case "rotate": return compileTransform(stmt, "rotate");
    case "scale": return compileTransform(stmt, "scale");
    case "mirror": return compileMirror(stmt);
    case "multmatrix": return compileMultMatrix(stmt);
    case "resize": return compilePassthrough(stmt, "resize");
    case "offset": return compileOffset(stmt);
    case "color": return compileColor(stmt);
    case "render": return compilePassthrough(stmt, "render");
    case "projection": return compileProjection(stmt);

    // Boolean operations
    case "union": return compileBoolOp(stmt, "union");
    case "difference": return compileDifference(stmt);
    case "intersection": return compileBoolOp(stmt, "intersection");
    case "hull": return compileBoolOp(stmt, "hull");
    case "minkowski": return compileMinkowski(stmt);

    // Extrusion
    case "linear_extrude": return compileLinearExtrude(stmt);
    case "rotate_extrude": return compileRotateExtrude(stmt);

    // Builtin statement modifiers
    case "echo": return compileEchoModule(stmt);
    case "assert": return compileAssertModule(stmt);
    case "let": return compileLetModule(stmt);
    case "children": return compileChildrenModule(stmt);
    case "intersection_for": return compileIntersectionFor(stmt);

    default:
      return compileUserModuleCall(stmt);
  }
}

// Builtin module helpers
function compileEchoModule(stmt: ModuleCallStmt): string {
  const args = stmt.args.map(a =>
    a.name ? `"${a.name} = ", ${compileExpr(a.value)}` : compileExpr(a.value)
  ).join(", ");
  if (stmt.child && stmt.child.kind !== "empty") {
    const child = compileGeometry(stmt.child);
    return `(console.log(${args}), ${child || "Manifold.union([])"})`;
  }
  return `(console.log(${args}), Manifold.union([]))`;
}

function compileAssertModule(stmt: ModuleCallStmt): string {
  const condition = stmt.args[0] ? compileExpr(stmt.args[0].value) : "true";
  const message = stmt.args[1] ? compileExpr(stmt.args[1].value) : '"Assertion failed"';
  if (stmt.child && stmt.child.kind !== "empty") {
    const child = compileGeometry(stmt.child);
    return `(openscad_assert_fn(${condition}, ${message}), ${child || "Manifold.union([])"})`;
  }
  return `(openscad_assert_fn(${condition}, ${message}), Manifold.union([]))`;
}

function compileLetModule(stmt: ModuleCallStmt): string {
  let child = "Manifold.union([])";
  if (stmt.child && stmt.child.kind !== "empty") {
    child = compileGeometry(stmt.child) || child;
  }
  let result = child;
  for (let i = stmt.args.length - 1; i >= 0; i--) {
    const a = stmt.args[i]!;
    const name = a.name ? escapeName(a.name) : "_";
    if (result.includes("await ")) {
      result = `await (async (${name}: any) => (${result}))(${compileExpr(a.value)})`;
    } else {
      result = `((${name}: any) => (${result}))(${compileExpr(a.value)})`;
    }
  }
  return result;
}

function compileChildrenModule(stmt: ModuleCallStmt): string {
  if (stmt.args.length > 0) {
    return `children(${compileExpr(stmt.args[0]!.value)})`;
  }
  return `children()`;
}

// Primitive compilation
function compileCube(args: Argument[]): string {
  const size = findArg(args, "size", 0);
  const center = findArg(args, "center", 1);

  const sizeStr = size ? compileExpr(size.value) : "1";
  const centerStr = center ? compileExpr(center.value) : "false";

  // `size` can be a scalar or a runtime vector expression
  return `((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), ${centerStr}))(${sizeStr})`;
}

function compileSphere(args: Argument[]): string {
  const r = findArg(args, "r", 0);
  const d = findArg(args, "d");
  const fn = findArg(args, "$fn");
  const fa = findArg(args, "$fa");
  const fs = findArg(args, "$fs");

  let radiusStr: string;
  if (d) {
    radiusStr = `(${compileExpr(d.value)}) / 2`;
  } else {
    radiusStr = r ? compileExpr(r.value) : "1";
  }

  const fnStr = fn ? compileExpr(fn.value) : "$fn";
  const faStr = fa ? compileExpr(fa.value) : "$fa";
  const fsStr = fs ? compileExpr(fs.value) : "$fs";
  return `__sphere(${radiusStr}, ${fnStr}, ${faStr}, ${fsStr})`;
}

function compileCylinder(args: Argument[]): string {
  const h = findArg(args, "h", 0);
  const r = findArg(args, "r");
  const r1 = findArg(args, "r1", 1);
  const r2 = findArg(args, "r2", 2);
  const d = findArg(args, "d");
  const d1 = findArg(args, "d1");
  const d2 = findArg(args, "d2");
  const center = findArg(args, "center", 3);
  const fn = findArg(args, "$fn");
  const fa = findArg(args, "$fa");
  const fs = findArg(args, "$fs");

  const hStr = h ? compileExpr(h.value) : "1";

  let rLow: string, rHigh: string;
  if (d) {
    rLow = rHigh = `(${compileExpr(d.value)}) / 2`;
  } else if (d1 || d2) {
    rLow = d1 ? `(${compileExpr(d1.value)}) / 2` : "1";
    rHigh = d2 ? `(${compileExpr(d2.value)}) / 2` : "1";
  } else if (r) {
    rLow = rHigh = compileExpr(r.value);
  } else {
    rLow = r1 ? compileExpr(r1.value) : "1";
    rHigh = r2 ? compileExpr(r2.value) : "1";
  }

  const fnStr = fn ? compileExpr(fn.value) : "$fn";
  const centerStr = center ? compileExpr(center.value) : "false";
  const faStr = fa ? compileExpr(fa.value) : "$fa";
  const fsStr = fs ? compileExpr(fs.value) : "$fs";

  return `__cylinder(${hStr}, ${rLow}, ${rHigh}, ${fnStr}, ${centerStr}, ${faStr}, ${fsStr})`;
}

function compileCircle(args: Argument[]): string {
  const r = findArg(args, "r", 0);
  const d = findArg(args, "d");
  const fn = findArg(args, "$fn");
  const fa = findArg(args, "$fa");
  const fs = findArg(args, "$fs");

  let radiusStr: string;
  if (d) {
    radiusStr = `(${compileExpr(d.value)}) / 2`;
  } else {
    radiusStr = r ? compileExpr(r.value) : "1";
  }

  const fnStr = fn ? compileExpr(fn.value) : "$fn";
  const faStr = fa ? compileExpr(fa.value) : "$fa";
  const fsStr = fs ? compileExpr(fs.value) : "$fs";
  return `__circle(${radiusStr}, ${fnStr}, ${faStr}, ${fsStr})`;
}

function compileSquare(args: Argument[]): string {
  const size = findArg(args, "size", 0);
  const center = findArg(args, "center", 1);

  const sizeStr = size ? compileExpr(size.value) : "1";
  const centerStr = center ? compileExpr(center.value) : "false";

  // `size` can be a scalar or a runtime vector expression
  return `((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), ${centerStr}))(${sizeStr})`;
}

function compilePolygon(args: Argument[]): string {
  const points = findArg(args, "points", 0);
  const paths = findArg(args, "paths", 1);
  if (!points) return `__polygon(/* missing points */[])`;
  const pointsStr = compileExpr(points.value);
  const pathsStr = paths ? compileExpr(paths.value) : "undefined";
  return `__polygon(${pointsStr}, ${pathsStr})`;
}

function compileText(args: Argument[]): string {
  const txt = findArg(args, "text", 0);
  const size = findArg(args, "size", 1);
  const font = findArg(args, "font");
  const halign = findArg(args, "halign");
  const valign = findArg(args, "valign");
  const spacing = findArg(args, "spacing");
  const dir = findArg(args, "direction");
  const fn = findArg(args, "$fn");

  const txtStr = txt ? compileExpr(txt.value) : `""`;
  const sizeStr = size ? compileExpr(size.value) : `10`;
  const fontStr = font ? compileExpr(font.value) : `"Liberation Sans:style=Regular"`;
  const halignStr = halign ? compileExpr(halign.value) : `"left"`;
  const valignStr  = valign  ? compileExpr(valign.value)  : `"baseline"`;
  const spacingStr = spacing ? compileExpr(spacing.value) : `1`;
  const dirStr = dir     ? compileExpr(dir.value)     : `"ltr"`;
  const fnStr = fn      ? compileExpr(fn.value)      : `$fn`;

  // Track font for base64 generation and resolve variable name.
  const rawFontSpec = font && font.value.kind === "string" ? font.value.value : "Liberation Sans:style=Regular";
  encounteredFonts.add(rawFontSpec);
  const filename = fontSpecToFilename(rawFontSpec);
  console.log("Font file name: ", filename);

  return `__text(${txtStr}, ${sizeStr}, ${fontStr}, ${halignStr}, ${valignStr}, ${spacingStr}, ${dirStr}, ${fnStr}, __font_registry)`;
}

function compilePolyhedron(args: Argument[]): string {
  const points = findArg(args, "points", 0);
  const triangles = findArg(args, "triangles", 1);
  let faces = findArg(args, "faces", 2);
  if (triangles) faces = triangles;

  if (!points || !faces) return `/* polyhedron: missing points or faces */`;

  const modifiedFaces: number[][] = [];
  if (faces.value.kind === "vector") {
    const faceElements = faces.value.elements;
    for (let i = 0; i < faceElements.length; i++) {
      const el = faceElements[i]!;
      if (el.kind === "vector") {
        const face = el.elements.map((v: any) => v.value);
        if (face.length > 3) {
          modifiedFaces.push([face[0], face[1], face[2]].reverse());
          modifiedFaces.push([face[0], face[2], face[3]].reverse());
        } else {
          modifiedFaces.push(face.reverse());
        }
      }
    }
  }

  return `new Manifold(new wasm.Mesh({ numProp: 3, vertProperties: new Float32Array(${compileExpr(points.value)}.flat()), triVerts: new Uint32Array(${JSON.stringify(modifiedFaces)}.flat()) }))`;
}

// Transforms
function compileTransform(
  stmt: ModuleCallStmt,
  method: string,
): string {
  if (!stmt.child) return "Manifold.union([])";

  const child = compileGeometry(stmt.child);
  if (method === "rotate") {
    const a = stmt.args[0];
    const v = stmt.args[1];
    return `__rotate(${child}, ${a ? compileExpr(a.value) : "undefined"}, ${v ? compileExpr(v.value) : "undefined"})`;
  }
  const vec = stmt.args[0];
  const defaultVec = method === "translate" ? "[0, 0, 0]" : "[1, 1, 1]";
  const vecStr = vec ? compileExpr(vec.value) : defaultVec;
  return `__${method}(${child}, ${vecStr})`;
}

function compileMirror(stmt: ModuleCallStmt): string {
  if (!stmt.child) return "Manifold.union([])";
  const child = compileGeometry(stmt.child);
  const vec = stmt.args[0];
  const vecStr = vec ? compileExpr(vec.value) : "[1, 0, 0]";
  return `__mirror(${child}, ${vecStr})`;
}

function compileMultMatrix(stmt: ModuleCallStmt): string {
  if (!stmt.child) return "Manifold.union([])";
  const child = compileGeometry(stmt.child);
  const mat = stmt.args[0];
  if (!mat) return `${child}`;
  return `__safe_transform(${child}, ${compileExpr(mat.value)})`;
}

function compileColor(stmt: ModuleCallStmt): string {
  if (!stmt.child) return "Manifold.union([])";
  const child = compileGeometry(stmt.child);
  const c = findArg(stmt.args, "c", 0);
  const alpha = findArg(stmt.args, "alpha", 1);
  const cExpr = c ? compileExpr(c.value) : "undefined";
  const aExpr = alpha ? compileExpr(alpha.value) : "undefined";
  if (child.includes("await ")) {
    return `await (async () => { var __save_$color: any = $color; $color = __parse_color_for_scope(${cExpr}, ${aExpr}); try { return await __apply_color(${child}, ${cExpr}, ${aExpr}); } finally { $color = __save_$color; } })()`;
  }
  return `(() => { var __save_$color: any = $color; $color = __parse_color_for_scope(${cExpr}, ${aExpr}); try { return __apply_color(${child}, ${cExpr}, ${aExpr}); } finally { $color = __save_$color; } })()`;
}

function compilePassthrough(stmt: ModuleCallStmt, tag: string): string {
  if (!stmt.child) return "Manifold.union([])";
  return `/* ${tag}(${stmt.args.map(a => compileExpr(a.value)).join(", ")}) */ ${compileGeometry(stmt.child)}`;
}

function compileOffset(stmt: ModuleCallStmt): string {
  if (!stmt.child) return "CrossSection.square(0)";
  const child = compileGeometry(stmt.child);
  const r = findArg(stmt.args, "r", 0);
  const delta = findArg(stmt.args, "delta");
  const amount = r ?? delta;
  const amt = amount ? compileExpr(amount.value) : "0";
  return `__safe_offset2d(${child}, ${amt}, "Round", 2, $fn, $fa, $fs)`;
}

function compileProjection(stmt: ModuleCallStmt): string {
  if (!stmt.child) return "CrossSection.square(0)";
  const child = compileGeometry(stmt.child);
  if (child === "Manifold.union([])") return "CrossSection.square(0)";
  const cut = findArg(stmt.args, "cut", 0);
  const cutStr = cut ? compileExpr(cut.value) : "false";
  return `__safe_project3d(${child}, ${cutStr})`;
}

// Boolean / collection 
function collectChildrenWithDecls(stmt: ModuleCallStmt): { decls: string[]; geos: string[] } {
  if (!stmt.child) return { decls: [], geos: [] };
  if (stmt.child.kind === "block") {
    return compileBlockStatementsWithDecls(stmt.child.statements);
  }
  const g = compileGeometry(stmt.child);
  return { decls: [], geos: g ? [g] : [] };
}

function compileBlockStatementsWithDecls(stmts: Statement[]): { decls: string[]; geos: string[] } {
  const decls: string[] = [];
  const geos: string[] = [];

  for (const s of stmts) {
    if (s.kind === "empty") continue;
    if (s.kind === "variableDecl") {
      decls.push(`${leadingCommentLines(s).join("\n")}${s.leadingComments?.length ? "\n" : ""}var ${escapeName(s.name)}: any = ${compileExpr(s.value)};${trailingCommentText(s)}`);
    } else if (s.kind === "functionDecl" || s.kind === "moduleDecl") {
      decls.push(compileDeclaration(s));
    } else {
      const g = compileGeometry(s);
      if (g) {
        const leading = leadingCommentLines(s);
        geos.push(`${leading.length ? `${leading.join("\n")}\n` : ""}${g}`);
      }
    }
  }

  return { decls, geos };
}

function compileBoolOp(stmt: ModuleCallStmt, op: string): string {
  const { decls, geos } = collectChildrenWithDecls(stmt);
  if (geos.length === 0) return "Manifold.union([])";

  let result: string;
  if (geos.length === 1) {
    result = geos[0]!;
  } else if (op === "union") {
    result = `__union2d3d([\n  ${geos.join(",\n  ")}\n])`;
  } else if (op === "intersection") {
    result = `__intersection2d3d([\n  ${geos.join(",\n  ")}\n])`;
  } else if (op === "hull") {
    result = `__hull2d3d([\n  ${geos.join(",\n  ")}\n])`;
  } else {
    result = `Manifold.${op}([\n  ${geos.join(",\n  ")}\n])`;
  }

  if (decls.length > 0) {
    if (result.includes("await ")) {
      return `await (async () => {\n  ${decls.join("\n  ")}\n  return await ${returnExpr(result, "  ")};\n})()`;
    }
    return `(() => {\n  ${decls.join("\n  ")}\n  return ${returnExpr(result, "  ")};\n})()`;
  }
  return result;
}

function compileDifference(stmt: ModuleCallStmt): string {
  const { decls, geos } = collectChildrenWithDecls(stmt);
  if (geos.length === 0) return "Manifold.union([])";

  let result: string;
  if (geos.length === 1) {
    result = geos[0]!;
  } else {
    const [first, ...rest] = geos;
    result = `__difference2d3d(${first}, [\n  ${rest.join(",\n  ")}\n])`;
  }

  if (decls.length > 0) {
    if (result.includes("await ")) {
      return `await (async () => {\n  ${decls.join("\n  ")}\n  return await ${returnExpr(result, "  ")};\n})()`;
    }
    return `(() => {\n  ${decls.join("\n  ")}\n  return ${returnExpr(result, "  ")};\n})()`;
  }
  return result;
}

function compileMinkowski(stmt: ModuleCallStmt): string {
  const { decls, geos } = collectChildrenWithDecls(stmt);
  if (geos.length === 0) return "Manifold.union([])";

  let result: string;
  if (geos.length === 1) {
    result = geos[0]!;
  } else {
    result = `__minkowski2d3d([\n  ${geos.join(",\n  ")}\n])`;
  }

  if (decls.length > 0) {
    if (result.includes("await ")) {
      return `await (async () => {\n  ${decls.join("\n  ")}\n  return await ${returnExpr(result, "  ")};\n})()`;
    }
    return `(() => {\n  ${decls.join("\n  ")}\n  return ${returnExpr(result, "  ")};\n})()`;
  }
  return result;
}

function compileLinearExtrude(stmt: ModuleCallStmt): string {
  if (!stmt.child) return "Manifold.union([])";
  const child = compileGeometry(stmt.child);
  if (child === "Manifold.union([])") return "Manifold.union([])";
  const height = findArg(stmt.args, "height", 0);
  const hStr = height ? compileExpr(height.value) : "1";

  const twist = findArg(stmt.args, "twist");
  const slices = findArg(stmt.args, "slices");
  const scale = findArg(stmt.args, "scale");
  const center = findArg(stmt.args, "center");
  const fn = findArg(stmt.args, "$fn");
  const fa = findArg(stmt.args, "$fa");
  const fs = findArg(stmt.args, "$fs");

  const opts: string[] = [];

  if (twist) {
    opts.push(`twist: ${compileExpr(twist.value)}`);
  }

  if (scale) {
    opts.push(`scale: ${compileExpr(scale.value)}`);
  }

  if (center) {
    opts.push(`center: ${compileExpr(center.value)}`);
  }

  opts.push(
    `fn: ${fn ? compileExpr(fn.value) : "$fn"}`
  );

  opts.push(
    `fa: ${fa ? compileExpr(fa.value) : "$fa"}`
  );

  opts.push(
    `fs: ${fs ? compileExpr(fs.value) : "$fs"}`
  );

  if (slices) {
    opts.push(`slices: ${compileExpr(slices.value)}`);
  }

  if (opts.length) {
    return `__extrude(${child}, ${hStr}, { ${opts.join(", ")} })`;
  }
  return `__extrude(${child}, ${hStr})`;
}

function compileRotateExtrude(stmt: ModuleCallStmt): string {
  if (!stmt.child) return "Manifold.union([])";
  const child = compileGeometry(stmt.child);
  if (child === "Manifold.union([])") return "Manifold.union([])";
  const angle = findArg(stmt.args, "angle", 0);
  const aStr = angle ? compileExpr(angle.value) : "360";
  const fn = findArg(stmt.args, "$fn");
  const fa = findArg(stmt.args, "$fa");
  const fs = findArg(stmt.args, "$fs");
  const fnStr = fn ? compileExpr(fn.value) : "$fn";
  const faStr = fa ? compileExpr(fa.value) : "$fa";
  const fsStr = fs ? compileExpr(fs.value) : "$fs";
  return `__revolve(${child}, ${fnStr}, ${faStr}, ${fsStr}, ${aStr})`;
}

// Block geometry 
function compileBlockGeometry(block: BlockStmt): string {
  const items: { kind: "var" | "dollar" | "func" | "geo"; name?: string; code: string }[] = [];
  const localVarNames = collectLocalVariableNames(block.statements);

  withLocalScope(localVarNames, () => {
    for (const s of block.statements) {
      if (s.kind === "empty") continue;
      if (s.kind === "variableDecl") {
        const name = escapeName(s.name);
        const code = compileExpr(s.value);
        if (s.name.startsWith("$")) {
          dynamicScopeVars.add(name);
          items.push({ kind: "dollar", name, code });
        } else {
          items.push({ kind: "var", name, code });
        }
      } else if (s.kind === "functionDecl" || s.kind === "moduleDecl") {
        items.push({ kind: "func", code: compileDeclaration(s) });
      } else {
        const g = compileGeometry(s);
        if (g) {
          const leading = leadingCommentLines(s);
          items.push({ kind: "geo", code: `${leading.length ? `${leading.join("\n")}\n` : ""}${g}` });
        }
      }
    }
  });

  // Collect geometry expressions
  const geos = items.filter(i => i.kind === "geo").map(i => i.code);
  const result =
    geos.length === 0 ? "Manifold.union([])"
      : geos.length === 1 ? geos[0]!
        : `__union2d3d([\n  ${geos.join(",\n  ")}\n])`;

  // Collect declarations (var, dollar, func) in order
  const decls = items.filter(i => i.kind !== "geo");
  if (decls.length === 0) return result;

  // Build inside-out so OpenSCAD let() semantics work -> vars capture outer values, $vars use dynamic scoping, and functions wrap the remaining body.
  let body = result;

  for (let i = decls.length - 1; i >= 0; i--) {
    const d = decls[i]!;
    if (d.kind === "var") {
      if (body.includes("await ")) {
        body = `await (async (${d.name}: any) => (${body}))(${d.code})`;
      } else {
        body = `((${d.name}: any) => (${body}))(${d.code})`;
      }
    } else if (d.kind === "dollar") {
      if (body.includes("await ")) {
        body = `await (async () => { var __save_${d.name}: any = ${d.name}; ${d.name} = ${d.code}; try { return await ${returnExpr(body, "  ")}; } finally { ${d.name} = __save_${d.name}; } })()`;
      } else {
        body = `(() => { var __save_${d.name}: any = ${d.name}; ${d.name} = ${d.code}; try { return ${returnExpr(body, "  ")}; } finally { ${d.name} = __save_${d.name}; } })()`;
      }
    } else {
      // Wrap remaining body in IIFE with the declaration
      if (body.includes("await ")) {
        body = `await (async () => {\n  ${d.code}\n  return await ${returnExpr(body, "  ")};\n})()`;
      } else {
        body = `(() => {\n  ${d.code}\n  return ${returnExpr(body, "  ")};\n})()`;
      }
    }
  }

  return body;
}

function compileIntersectionFor(stmt: ModuleCallStmt): string {
  if (!stmt.child) return "Manifold.union([])";

  const variables: ForVariable[] = stmt.args.map(arg => ({
    name: arg.name || "_",
    range: arg.value,
    loc: arg.loc,
  }));

  const lines = [
    "(() => {",
    "  const __items: any[] = [];",
    ...buildNestedForStatements(variables, 0, stmt.child, 1),
    "  return __intersection2d3d(__items);",
    "})()",
  ];
  const code = lines.join("\n");
  if (code.includes("await ")) {
    lines[0] = "async () => {";
    lines[lines.length - 1] = "})()";
    return "await (" + lines.join("\n");
  }
  return code;
}

function buildNestedIntersectionFor(vars: ForVariable[], idx: number, body: string): string {
  if (idx >= vars.length) return body;

  const v = vars[idx]!;
  const inner = buildNestedIntersectionFor(vars, idx + 1, body);

  if (v.range.kind === "range") {
    const start = compileExpr(v.range.start);
    const end = compileExpr(v.range.end);
    const step = v.range.step ? compileExpr(v.range.step) : "1";
    return `__intersection2d3d((() => {\n` +
      `  const __items = [];\n` +
      `  for (let ${escapeName(v.name)}: any = ${start}; ${escapeName(v.name)} <= ${end}; ${escapeName(v.name)} += ${step}) {\n` +
      `    __items.push(${inner});\n` +
      `  }\n` +
      `  return __items;\n` +
      `})())`;
  }

  // vector iteration
  const rangeExpr = compileExpr(v.range);
  return `__intersection2d3d(__flat_map_iter(${rangeExpr}, (${escapeName(v.name)}: any, __i: any) => { var __save_$idx: any = $idx; $idx = __i; try { return [${inner}]; } finally { $idx = __save_$idx; } }))`;
}

// For / If geometry
function compileForGeometry(stmt: ForStmt): string {
  const lines = [
    "(() => {",
    "  const __items = [];",
    ...buildNestedForStatements(stmt.variables, 0, stmt.body, 1),
    "  return __union2d3d(__items);",
    "})()",
  ];
  const code = lines.join("\n");
  if (code.includes("await ")) {
    lines[0] = "async () => {";
    lines[lines.length - 1] = "})()";
    return "await (" + lines.join("\n");
  }
  return code;
}

function buildNestedFor(vars: ForVariable[], idx: number, body: string): string {
  if (idx >= vars.length) return body;

  const v = vars[idx]!;
  const inner = buildNestedFor(vars, idx + 1, body);

  if (v.range.kind === "range") {
    const start = compileExpr(v.range.start);
    const end = compileExpr(v.range.end);
    const step = v.range.step ? compileExpr(v.range.step) : "1";
    return `__union2d3d((() => {\n` +
      `  const __items = [];\n` +
      `  for (let ${escapeName(v.name)}: any = ${start}; ${escapeName(v.name)} <= ${end}; ${escapeName(v.name)} += ${step}) {\n` +
      `    __items.push(${inner});\n` +
      `  }\n` +
      `  return __items;\n` +
      `})())`;
  }

  // vector iteration
  const rangeExpr = compileExpr(v.range);
  return `__union2d3d(__flat_map_iter(${rangeExpr}, (${escapeName(v.name)}: any, __i: any) => { var __save_$idx: any = $idx; $idx = __i; try { return [${inner}]; } finally { $idx = __save_$idx; } }))`;
}

function buildNestedForStatements(
  vars: ForVariable[],
  idx: number,
  body: Statement,
  indentLevel: number,
): string[] {
  const indent = "  ".repeat(indentLevel);
  if (idx >= vars.length) {
    const lines: string[] = [];
    const geo = compileGeometry(body);
    if (geo) pushCommentedLine(lines, body, `${indent}__items.push(${geo});`, indent);
    return lines;
  }

  const v = vars[idx]!;
  const vName = escapeName(v.name);
  if (v.range.kind === "range") {
    const start = compileExpr(v.range.start);
    const end = compileExpr(v.range.end);
    const step = v.range.step ? compileExpr(v.range.step) : "1";
    const stepName = `__step_${idx}`;
    return [
      `${indent}{`,
      `${indent}  const ${stepName}: any = ${step};`,
      `${indent}  for (let ${vName}: any = ${start}; (${stepName} >= 0) ? ${vName} <= ${end} : ${vName} >= ${end}; ${vName} += ${stepName}) {`,
      ...buildNestedForStatements(vars, idx + 1, body, indentLevel + 2),
      `${indent}  }`,
      `${indent}}`,
    ];
  }

  const iterName = `__iter_${idx}`;
  const idxName = `__idx_${idx}`;
  return [
    `${indent}{`,
    `${indent}  const ${iterName}: any = ${compileExpr(v.range)};`,
    `${indent}  for (let ${idxName} = 0; ${idxName} < ${iterName}.length; ${idxName}++) {`,
    `${indent}    const ${vName}: any = ${iterName}[${idxName}];`,
    `${indent}    var __save_$idx: any = $idx; $idx = ${idxName};`,
    `${indent}    try {`,
    ...buildNestedForStatements(vars, idx + 1, body, indentLevel + 3),
    `${indent}    } finally { $idx = __save_$idx; }`,
    `${indent}  }`,
    `${indent}}`,
  ];
}

function compileIfGeometry(stmt: IfStmt): string {
  const cond = compileExpr(stmt.condition);
  const then = compileGeometry(stmt.thenBody);
  if (stmt.elseBody) {
    const els = compileGeometry(stmt.elseBody);
    const lines = [
      "(() => {",
      `  if (${cond}) {`,
    ];
    pushCommentedLine(lines, stmt.thenBody, `    return ${returnExpr(then, "    ")};`, "    ");
    lines.push("  }");
    lines.push("  else {");
    pushCommentedLine(lines, stmt.elseBody, `    return ${returnExpr(els, "    ")};`, "    ");
    lines.push("  }");
    lines.push("})()");
    return lines.join("\n");
  }
  const lines = [
    "(() => {",
    `  if (${cond}) {`,
  ];
  pushCommentedLine(lines, stmt.thenBody, `    return ${returnExpr(then, "    ")};`, "    ");
  lines.push("  }");
  lines.push("  return Manifold.union([]);");
  lines.push("})()");
  return lines.join("\n");
}

// User module call
function compileUserModuleCall(stmt: ModuleCallStmt): string {
  if (!moduleDeclRegistry.has(stmt.name)) {
    const line = stmt.loc?.start.line;
    const where = line ? ` at line ${line}` : "";
    console.warn(`Warning: Ignoring unknown module '${stmt.name}'${where}`);
    return "Manifold.union([])";
  }

  const name = `${escapeName(stmt.name)}$mod`;
  const argList = compileArgList(name, stmt.args);
  const { decls, geos } = stmt.child && stmt.child.kind !== "empty" ? collectChildrenWithDecls(stmt) : { decls: [], geos: [] };
  const result = buildWithChildrenCall(`${name}(${argList})`, geos, stmt.name);

  if (decls.length > 0) {
    if (result.includes("await ")) {
      return `await (async () => {\n  ${decls.join("\n  ")}\n  return await ${returnExpr(result, "  ")};\n})()`;
    }
    return `(() => {\n  ${decls.join("\n  ")}\n  return ${returnExpr(result, "  ")};\n})()`;
  }
  return result;
}

function guessMimeType(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  const map: Record<string, string> = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
  };
  return map[ext] ?? "image/png";
}

// compile surface
export function compileSurface(args: Argument[]): string {
  const file = findArg(args, "file", 0);
  const center = findArg(args, "center", 1);
  const invert = findArg(args, "invert", 2);

  if (!file?.value || file.value.kind !== "string") {
    throw new Error(`surface(): expected string literal for file`);
  }
  const filenameStr = file.value.value;

  // Resolve the image path
  const basePath = process.env.IMAGEBASEPATH;
  if (!basePath) {
    throw new Error(`surface("${filenameStr}"): IMAGEBASEPATH environment variable is not set.`);
  }

  const imagePath = path.join(basePath, path.basename(filenameStr));
  if (!fs.existsSync(imagePath)) {
    throw new Error(`surface("${filenameStr}"): image not found at "${imagePath}". Check IMAGEBASEPATH="${basePath}".`);
  }

  // Encode to base64
  const base64 = fs.readFileSync(imagePath).toString("base64");
  const mimeType = guessMimeType(imagePath);
  const dataUrl = `data:${mimeType};base64,${base64}`;

  // Write ./runtime/images/<stem>_base64.ts
  const stem = path.basename(filenameStr, path.extname(filenameStr)).replace(/[^a-zA-Z0-9_]/g, "_");
  const exportName = `__img_${stem}`;

  const currentFileDir = typeof __dirname !== "undefined"
    ? __dirname
    : path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/i, "$1"));
  const compilerDir = path.resolve(currentFileDir, "..");
  const imagesDir = path.join(compilerDir, "runtime", "images");
  fs.mkdirSync(imagesDir, { recursive: true });

  const tsContent =
    `// Auto-generated by OpenSCAD compiler — do not edit\n` +
    `// Source: ${imagePath}\n` +
    `export const ${exportName} = "${dataUrl}";\n`;

  const tsPath = path.join(imagesDir, `${stem}_base64.ts`);
  fs.writeFileSync(tsPath, tsContent, "utf8");

  // Track the image for top-level import
  encounteredImages.set(filenameStr, { stem, exportName });

  const centerStr = center ? compileExpr(center.value) : "false";

  return `await __surface(${exportName}, { center: ${centerStr}, fn: $fn, fa: $fa, fs: $fs })`;
}

// Expression compilation
function compileExpr(expr: Expr): string {
  switch (expr.kind) {
    case "number":
      return String(expr.value);
    case "string":
      return JSON.stringify(expr.value);
    case "boolean":
      return String(expr.value);
    case "undef":
      return "undefined";
    case "identifier":
      // $children stays as $children (the count variable set in module body)
      if (expr.name === "$children") return "$children";
      return escapeName(expr.name);
    case "vector":
      return `[${expr.elements.map(compileExpr).join(", ")}]`;
    case "range":
      if (expr.step) {
        return `__range(${compileExpr(expr.start)}, ${compileExpr(expr.step)}, ${compileExpr(expr.end)})`;
      }
      return `__range(${compileExpr(expr.start)}, 1, ${compileExpr(expr.end)})`;
    case "binary":
      if (expr.op === "^") {
        return `Math.pow(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      }
      if (expr.op === "==") {
        return `__eq(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      }
      if (expr.op === "!=") {
        return `(!__eq(${compileExpr(expr.left)}, ${compileExpr(expr.right)}))`;
      }
      if (expr.op === "+") return `__add(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === "-") return `__sub(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === "*") return `__mul(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === "/") return `__div(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === "%") return `__mod(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      return `(${compileExpr(expr.left)} ${expr.op} ${compileExpr(expr.right)})`;
    case "unary":
      if (expr.op === "-") return `__neg(${compileExpr(expr.operand)})`;
      if (expr.op === "+") return `__pos(${compileExpr(expr.operand)})`;
      return `(${expr.op}${compileExpr(expr.operand)})`;
    case "ternary": {
      let ifTrue = compileExpr(expr.ifTrue);
      let ifFalse = compileExpr(expr.ifFalse);
      const trueSpread = ifTrue.startsWith("...");
      const falseSpread = ifFalse.startsWith("...");
      if (trueSpread || falseSpread) {
        if (trueSpread) ifTrue = ifTrue.slice(3); else ifTrue = `[${ifTrue}]`;
        if (falseSpread) ifFalse = ifFalse.slice(3); else ifFalse = `[${ifFalse}]`;
        return `...(${compileExpr(expr.condition)} ? ${ifTrue} : ${ifFalse})`;
      }
      return `(${compileExpr(expr.condition)} ? ${ifTrue} : ${ifFalse})`;
    }
    case "call":
      return compileCallExpr(expr);
    case "index":
      return `${compileExpr(expr.object)}[${compileExpr(expr.index)}]`;
    case "member": {
      const memberMap: Record<string, string> = { x: "0", y: "1", z: "2" };
      const idx = memberMap[expr.property];
      if (idx !== undefined) {
        return `${compileExpr(expr.object)}[${idx}]`;
      }
      return `${compileExpr(expr.object)}.${expr.property}`;
    }
    case "group": {
      const inner = compileExpr(expr.expr);
      if (inner.startsWith("...")) return inner;
      return `(${inner})`;
    }
    case "echo": {
      const eArgs = expr.args.map(a =>
        a.name ? `"${a.name} = ", ${compileExpr(a.value)}` : compileExpr(a.value)
      ).join(", ");
      return `(console.log(${eArgs}), ${compileExpr(expr.expr)})`;
    }
    case "assert": {
      const condition = expr.args[0] ? compileExpr(expr.args[0].value) : "true";
      const message = expr.args[1] ? compileExpr(expr.args[1].value) : '"Assertion failed"';
      return `(openscad_assert_fn(${condition}, ${message}), ${compileExpr(expr.expr)})`;
    }
    case "let": {
      const localAssignNames = expr.assignments.map(a => escapeName(a.name));
      return withLocalScope(localAssignNames, () => {
        let result = compileExpr(expr.body);
        for (let i = expr.assignments.length - 1; i >= 0; i--) {
          const a = expr.assignments[i]!;
          result = `((${escapeName(a.name)}) => (${result}))(${compileExpr(a.value)})`;
        }
        return result;
      });
    }
    case "each": {
      const inner = compileExpr(expr.expr);
      if (inner.startsWith("...")) return inner;
      return `...${inner}`;
    }
    case "lambda": {
      const params = expr.params.map(p => p.defaultValue ? `${escapeName(p.name)} = ${compileExpr(p.defaultValue)}` : escapeName(p.name)).join(", ");
      const localParams = expr.params.map(p => escapeName(p.name));
      const bodyExpr = withLocalScope(localParams, () => compileExpr(expr.body));
      return `(${params}) => ${bodyExpr}`;
    }
    case "listComp": {
      return `...(${compileListComp(expr.generator)})`;
    }
    case "dynCall": {
      const callee = compileExpr(expr.callee);
      const args = expr.args.map(a => a.name ? `/* ${a.name} = */ ${compileExpr(a.value)}` : compileExpr(a.value)).join(", ");
      return `(${callee})(${args})`;
    }
    default:
      return `/* unsupported expr: ${(expr as Expr).kind}${locTag(expr as ASTNode)} */`;
  }
}

function compileCallExpr(expr: { kind: "call"; name: string; args: Argument[] }): string {
  const escaped = escapeName(expr.name);
  const isKnownFunction = BUILTIN_FUNCTIONS.has(expr.name) || signatures.has(`${escaped}_fn`);
  const name = (!isKnownFunction && isLocalName(escaped)) ? escaped : `${escaped}_fn`;
  const argList = compileArgList(name, expr.args);
  if (name === "_attach_transform_fn") {
    return `__safe_attach_transform(${argList})`;
  }
  return `${name}(${argList})`;
}

// List comprehension
function compileListComp(gen: ListCompGenerator): string {
  switch (gen.kind) {
    case "lcFor": {
      let result = compileListComp(gen.body);
      for (let i = gen.variables.length - 1; i >= 0; i--) {
        const v = gen.variables[i]!;
        const vName = escapeName(v.name);
        if (v.range.kind === "range") {
          const start = compileExpr(v.range.start);
          const end = compileExpr(v.range.end);
          const step = v.range.step ? compileExpr(v.range.step) : "1";
          result = `(() => { const __r = []; for (let ${vName} = ${start}; ${vName} <= ${end}; ${vName} += ${step}) __r.push(...(${result})); return __r; })()`;
        } else {
          result = `__flat_map_iter(${compileExpr(v.range)}, (${vName}) => ${result})`;
        }
      }
      return result;
    }
    case "lcIf": {
      const cond = compileExpr(gen.condition);
      let ifTrue = compileListComp(gen.ifTrue);
      let ifFalse = gen.ifFalse ? compileListComp(gen.ifFalse) : "[]";
      // Both branches are now guaranteed to evaluate to an array.
      return `(${cond} ? ${ifTrue} : ${ifFalse})`;
    }
    case "lcLet": {
      let result = compileListComp(gen.body);
      for (let i = gen.assignments.length - 1; i >= 0; i--) {
        const a = gen.assignments[i]!;
        result = `((${escapeName(a.name)}) => (${result}))(${compileExpr(a.value)})`;
      }
      return result;
    }
    case "lcExpr": {
      const expr = compileExpr(gen.expr);
      if (expr.startsWith("...")) {
        return `[${expr}]`;
      }
      return `[${expr}]`;
    }
    case "lcCFor": {
      const inits = gen.inits.map(a => `${escapeName(a.name)} = ${compileExpr(a.value)}`).join(", ");
      const cond = compileExpr(gen.condition);
      const updates = gen.updates.map(a => `${escapeName(a.name)} = ${compileExpr(a.value)}`).join(", ");
      const inner = compileListComp(gen.body);
      return `(() => { const __r = []; for (let ${inits}; ${cond}; ${updates}) __r.push(${inner}); return __r; })()`;
    }
  }
}

// Argument lookup
function findArg(
  args: Argument[],
  name: string,
  positionalIndex?: number,
): Argument | undefined {
  const named = args.find((a) => a.name === name);
  if (named) return named;
  if (positionalIndex !== undefined && positionalIndex < args.length) {
    const a = args[positionalIndex]!;
    if (!a.name) return a;
  }
  return undefined;
}
