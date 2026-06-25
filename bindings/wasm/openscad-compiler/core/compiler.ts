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
import type { LibraryClosure } from "./resolver.js";

export interface LibraryManifest {
  library: string;
  compiledAt: string;
  runtimeVersion: string;
  files: Record<string, { out: string; modules: string[]; functions: string[]; variables: string[] }>;
  exports: {
    modules: Record<string, string>;   // module name -> owning source file (relPath)
    functions: Record<string, string>; // function name -> owning source file
    variables: Record<string, string>; // variable name -> owning source file
  };
  // name -> [file, file, ...] for cross-file redefinitions (last-wins applied).
  ambiguous: Record<string, string[]>;
  // Compiled export symbol ("name$mod" / "name_fn") -> parameter names, so a consumer can map positional/named args to a separately compiled callable.
  signatures: Record<string, string[]>;
}

export interface ResolvedExternalLib {
  name: string;
  manifest: LibraryManifest;
  importSpecifierFor: (sourceRelPath: string) => string;
  sideEffectSpecifiers?: string[];
}

export interface CompileOptions {
  runtimePath?: string;
  externalLibraries?: ResolvedExternalLib[];
}

const BUILTIN_MODULES = new Set([
  "cube", "sphere", "cylinder", "circle", "square", "polygon", "polyhedron", "text", "surface", "import",
  "translate", "rotate", "scale", "mirror", "multmatrix", "resize", "offset", "color", "render", "projection",
  "group", "union", "difference", "intersection", "hull", "minkowski", "linear_extrude", "rotate_extrude",
  "echo", "assert", "let", "children", "intersection_for", "parent_module",
]);

interface ProgramReferences {
  modules: Set<string>;    // module-call statements
  functions: Set<string>;  // function-call expressions
  variables: Set<string>;  // bare identifier reads (non-$)
}

function collectProgramReferences(stmts: Statement[]): ProgramReferences {
  const modules = new Set<string>();
  const functions = new Set<string>();
  const variables = new Set<string>();

  const visitExpr = (e: Expr | undefined): void => {
    if (!e) return;
    switch (e.kind) {
      case "identifier":
        if (e.name !== "$children" && !e.name.startsWith("$")) variables.add(e.name);
        break;
      case "call":
        functions.add(e.name);
        e.args.forEach(a => visitExpr(a.value));
        break;
      case "vector": e.elements.forEach(visitExpr); break;
      case "range": visitExpr(e.start); visitExpr(e.end); visitExpr(e.step); break;
      case "binary": visitExpr(e.left); visitExpr(e.right); break;
      case "unary": visitExpr(e.operand); break;
      case "group": case "each": visitExpr(e.expr); break;
      case "ternary": visitExpr(e.condition); visitExpr(e.ifTrue); visitExpr(e.ifFalse); break;
      case "index": visitExpr(e.object); visitExpr(e.index); break;
      case "member": visitExpr(e.object); break;
      case "echo": case "assert": e.args.forEach(a => visitExpr(a.value)); visitExpr(e.expr); break;
      case "let": e.assignments.forEach(a => visitExpr(a.value)); visitExpr(e.body); break;
      case "lambda": e.params.forEach(p => visitExpr(p.defaultValue)); visitExpr(e.body); break;
      case "dynCall": visitExpr(e.callee); e.args.forEach(a => visitExpr(a.value)); break;
      case "listComp": visitGenerator(e.generator); break;
    }
  };
  const visitGenerator = (g: ListCompGenerator | undefined): void => {
    if (!g) return;
    switch (g.kind) {
      case "lcFor": g.variables.forEach(v => visitExpr(v.range)); visitGenerator(g.body); break;
      case "lcCFor":
        g.inits.forEach(a => visitExpr(a.value)); visitExpr(g.condition);
        g.updates.forEach(a => visitExpr(a.value)); visitGenerator(g.body); break;
      case "lcIf": visitExpr(g.condition); visitGenerator(g.ifTrue); visitGenerator(g.ifFalse); break;
      case "lcLet": g.assignments.forEach(a => visitExpr(a.value)); visitGenerator(g.body); break;
      case "lcExpr": visitExpr(g.expr); break;
    }
  };
  const visitStmt = (s: Statement | undefined): void => {
    if (!s) return;
    switch (s.kind) {
      case "variableDecl": visitExpr(s.value); break;
      case "functionDecl": s.params.forEach(p => visitExpr(p.defaultValue)); visitExpr(s.body); break;
      case "moduleDecl": s.params.forEach(p => visitExpr(p.defaultValue)); visitStmt(s.body); break;
      case "moduleCall":
        if (s.name !== "children") modules.add(s.name);
        s.args.forEach(a => visitExpr(a.value));
        visitStmt(s.child);
        break;
      case "block": s.statements.forEach(visitStmt); break;
      case "for": s.variables.forEach(v => visitExpr(v.range)); visitStmt(s.body); break;
      case "if": visitExpr(s.condition); visitStmt(s.thenBody); visitStmt(s.elseBody); break;
    }
  };

  stmts.forEach(visitStmt);
  return { modules, functions, variables };
}

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

// Track unique surface data encountered during compilation for base64 generation.
let encounteredSurfaceData = new Map<string, { stem: string; exportName: string; kind: "image" | "text" }>();

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
    return args
      .map(a => a.name ? `/* ${a.name} = */ ${compileExpr(a.value)}` : compileExpr(a.value))
      .join(", ");
  }

  const compiledArgs: string[] = new Array(sig.params.length).fill("undefined");
  const namedClaimed: boolean[] = new Array(sig.params.length).fill(false);
  const extraArgs: string[] = [];

  let posCursor = 0;
  for (const a of args) {
    if (a.name) {
      if (a.name.startsWith("$")) continue;
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

  while (compiledArgs.length > 0 && compiledArgs[compiledArgs.length - 1] === "undefined") {
    compiledArgs.pop();
  }

  return compiledArgs.concat(extraArgs).join(", ");
}

const BUILTIN_FUNCTIONS = new Set([
  "is_undef", "is_bool", "is_num", "is_string", "is_list", "is_function", "is_object",
  "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
  "abs", "sign", "floor", "ceil", "round", "sqrt", "exp", "ln", "log", "pow",
  "min", "max", "norm", "cross",
  "len", "str", "chr", "ord", "concat", "search", "lookup",
  "version", "version_num",
  "parent_module", 
]);


const RUNTIME_SYMBOLS: string[] = [
  "__sphere", "__cylinder", "__circle", "__radius", "__polygon", "__polyhedron",
  "Manifold", "CrossSection", "wasm",
  "is_undef_fn", "is_object_fn", "is_bool_fn", "is_num_fn", "is_string_fn", "is_list_fn", "is_function_fn",
  "sin_fn", "cos_fn", "tan_fn", "asin_fn", "acos_fn", "atan_fn", "atan2_fn",
  "abs_fn", "sign_fn", "floor_fn", "ceil_fn", "round_fn", "sqrt_fn", "exp_fn", "ln_fn", "log_fn", "pow_fn",
  "min_fn", "max_fn", "norm_fn", "cross_fn",
  "len_fn", "str_fn", "chr_fn", "ord_fn", "concat_fn", "search_fn", "lookup_fn",
  "parent_module_fn", "openscad_assert_fn",
  "__truthy", "__eq", "__lt", "__gt", "__le", "__ge",
  "__add", "__sub", "__mul", "__div", "__mod", "__band", "__bor", "__shl", "__shr", "__bnot", "__neg", "__pos", "__index",
  "version_fn", "version_num_fn",
  "__ctx", "__withSpecials",
  "__children_stack", "__with_children",
  "__is_finite_matrix4", "__to_manifold_mat4", "__safe_transform", "__identity4",
  "__safe_offset2d", "__safe_project3d", "__apply_color",
  "__flat_map_iter", "__range", "__rangeCount",
  "__union2d3d", "__difference2d3d", "__intersection2d3d", "__hull2d3d", "__minkowski2d3d",
  "__extrude", "__revolve", "__rotate", "__translate", "__scale", "__mirror",
  "__text", "__parse_color_for_scope", "__surface",
];

// Emit target for a (possibly special) variable name. Special $-prefixed vars
// are stored in the shared runtime context object (`__ctx`); every other name is
// a plain lexical binding. Used wherever the emitter reads/writes a variable
// that might be a special variable (dynamic-scope save/assign/restore sites).
function svTarget(name: string): string {
  return name.startsWith("$") ? `__ctx.${name}` : name;
}

function buildRuntimeImport(runtimePath: string): string {
  return (
    `import * as __rt from "${runtimePath}";\n` +
    `const { ${RUNTIME_SYMBOLS.join(", ")} } = __rt;\n`
  );
}

// Names of modules/functions/variables that are defined in a separately
// compiled external library and imported (not inlined). Populated from the
// libraries' manifests at the start of compile(); consulted so calls to them
// are recognized rather than reported as unknown modules.
let externalModuleNames: Set<string> = new Set();
let externalFunctionNames: Set<string> = new Set();
let externalVariableNames: Set<string> = new Set();

// Keyword for top-level (file-scope) variable declarations. `let` for normal
// single-file output; `var` for library files, where cross-file circular ES
// imports would otherwise hit the temporal dead zone on `let` exports. `var`
// hoists to `undefined`, matching OpenSCAD's declarative `undef`-by-default.
let globalVarDeclKeyword = "let";

// Track $ variables that need module-level declarations for dynamic scoping
let dynamicScopeVars: Set<string> = new Set();
const localScopes: Set<string>[] = [];

let globalVarNames: Set<string> = new Set();
let activeShadowRenames: Map<string, string> = new Map();

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

// Collect every identifier referenced as a value, and every name bound anywhere
// in the program (variable/param/let/for/comprehension binding). A name that is
// referenced but never bound is an undefined variable: OpenSCAD evaluates such a
// reference to `undef` rather than erroring, so we declare these as `undefined`
// to avoid a ReferenceError in the compiled code.
interface IdentifierUsage {
  referenced: Set<string>;
  bound: Set<string>;
}

function collectIdentifierUsage(program: Program): IdentifierUsage {
  const referenced = new Set<string>();
  const bound = new Set<string>();

  const visitGenerator = (gen: ListCompGenerator | undefined): void => {
    if (!gen) return;
    switch (gen.kind) {
      case "lcFor":
        gen.variables.forEach(v => { bound.add(escapeName(v.name)); visitExpr(v.range); });
        visitGenerator(gen.body);
        break;
      case "lcCFor":
        gen.inits.forEach(a => { bound.add(escapeName(a.name)); visitExpr(a.value); });
        visitExpr(gen.condition);
        gen.updates.forEach(a => { bound.add(escapeName(a.name)); visitExpr(a.value); });
        visitGenerator(gen.body);
        break;
      case "lcIf":
        visitExpr(gen.condition);
        visitGenerator(gen.ifTrue);
        visitGenerator(gen.ifFalse);
        break;
      case "lcLet":
        gen.assignments.forEach(a => { bound.add(escapeName(a.name)); visitExpr(a.value); });
        visitGenerator(gen.body);
        break;
      case "lcExpr":
        visitExpr(gen.expr);
        break;
    }
  };

  const visitExpr = (expr: Expr | undefined): void => {
    if (!expr) return;
    switch (expr.kind) {
      case "identifier":
        if (expr.name !== "$children") referenced.add(escapeName(expr.name));
        break;
      case "number":
      case "string":
      case "boolean":
      case "undef":
        break;
      case "vector":
        expr.elements.forEach(visitExpr);
        break;
      case "range":
        visitExpr(expr.start); visitExpr(expr.end); visitExpr(expr.step);
        break;
      case "binary":
        visitExpr(expr.left); visitExpr(expr.right);
        break;
      case "unary":
        visitExpr(expr.operand);
        break;
      case "group":
      case "each":
        visitExpr(expr.expr);
        break;
      case "ternary":
        visitExpr(expr.condition); visitExpr(expr.ifTrue); visitExpr(expr.ifFalse);
        break;
      case "call":
        expr.args.forEach(a => visitExpr(a.value));
        break;
      case "index":
        visitExpr(expr.object); visitExpr(expr.index);
        break;
      case "member":
        visitExpr(expr.object);
        break;
      case "echo":
      case "assert":
        expr.args.forEach(a => visitExpr(a.value));
        visitExpr(expr.expr);
        break;
      case "let":
        expr.assignments.forEach(a => { bound.add(escapeName(a.name)); visitExpr(a.value); });
        visitExpr(expr.body);
        break;
      case "lambda":
        expr.params.forEach(p => { bound.add(escapeName(p.name)); visitExpr(p.defaultValue); });
        visitExpr(expr.body);
        break;
      case "dynCall":
        visitExpr(expr.callee);
        expr.args.forEach(a => visitExpr(a.value));
        break;
      case "listComp":
        visitGenerator(expr.generator);
        break;
    }
  };

  const visitStmt = (stmt: Statement | undefined): void => {
    if (!stmt) return;
    switch (stmt.kind) {
      case "variableDecl":
        bound.add(escapeName(stmt.name));
        visitExpr(stmt.value);
        break;
      case "functionDecl":
        stmt.params.forEach(p => { bound.add(escapeName(p.name)); visitExpr(p.defaultValue); });
        visitExpr(stmt.body);
        break;
      case "moduleDecl":
        stmt.params.forEach(p => { bound.add(escapeName(p.name)); visitExpr(p.defaultValue); });
        visitStmt(stmt.body);
        break;
      case "moduleCall":
        stmt.args.forEach(a => visitExpr(a.value));
        visitStmt(stmt.child);
        break;
      case "block":
        stmt.statements.forEach(visitStmt);
        break;
      case "for":
        stmt.variables.forEach(v => { bound.add(escapeName(v.name)); visitExpr(v.range); });
        visitStmt(stmt.body);
        break;
      case "if":
        visitExpr(stmt.condition);
        visitStmt(stmt.thenBody);
        visitStmt(stmt.elseBody);
        break;
    }
  };

  program.statements.forEach(visitStmt);
  return { referenced, bound };
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
export function compile(program: Program, options?: CompileOptions): string {
  currentRuntimePath = options?.runtimePath ?? "./runtime/runtime.js";
  dynamicScopeVars = new Set();
  encounteredFonts = new Set();
  encounteredSurfaceData.clear();
  externalModuleNames = new Set();
  externalFunctionNames = new Set();
  externalVariableNames = new Set();
  signatures.clear();
  for (const [k, v] of Object.entries(BUILTIN_SIGNATURES)) {
    signatures.set(k, { params: v, defaults: new Array(v.length).fill(undefined) });
  }
  // Register external-library export signatures and names BEFORE collecting
  // local ones, so a local declaration of the same name overrides the import.
  const externalLibraries = options?.externalLibraries ?? [];
  for (const lib of externalLibraries) {
    for (const [sym, params] of Object.entries(lib.manifest.signatures)) {
      signatures.set(sym, { params, defaults: new Array(params.length).fill(undefined) });
    }
    for (const name of Object.keys(lib.manifest.exports.modules)) externalModuleNames.add(name);
    for (const name of Object.keys(lib.manifest.exports.functions)) externalFunctionNames.add(name);
    for (const name of Object.keys(lib.manifest.exports.variables)) externalVariableNames.add(name);
  }
  collectSignatures(program.statements);
  moduleDeclRegistry = collectModuleDeclarations(program.statements);

  globalVarNames = new Set();
  activeShadowRenames = new Map();
  for (const s of program.statements) {
    if (s.kind === "variableDecl") globalVarNames.add(escapeName(s.name));
  }

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
  const processStmt = (stmt: Statement) => {
    if (stmt.kind === "empty") return;
    // An anonymous `{ }` block is not a separate scope in OpenSCAD: its
    // assignments merge into (and may override, last-wins) the enclosing scope,
    // and its actions run in place. Flatten it into the top-level scope.
    if (stmt.kind === "block") {
      for (const s of stmt.statements) processStmt(s);
      return;
    }
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

        if (hasBackgroundModifier(stmt)) {
          pushCommentedLine(geometryLines, stmt, `__background_items.push(${geo});`);
        } else if (stmt.kind === "moduleCall" && !stmt.modifier && isModuleCallBackgroundOnly(stmt, moduleDeclRegistry)) {
          pushCommentedLine(geometryLines, stmt, `__background_items.push(${geo});`);
        } else {
          pushCommentedLine(geometryLines, stmt, `__result_items.push(${geo});`);
        }
      }
    }
  };
  for (const stmt of program.statements) processStmt(stmt);

  // Order declarations to respect dependencies. Variable declarations compile to
  // `let name = expr` (not hoisted-initialized), so a variable that references
  // another top-level binding must be emitted after it — even when the reference
  // appears textually earlier (OpenSCAD scopes are declarative). Modules and
  // functions compile to hoisted `function` declarations and impose no ordering.
  // A stable topological sort keeps the original order wherever possible and
  // falls back to original order for any variables caught in a dependency cycle.
  const sortedDeclOrder = ((): string[] => {
    const varKeyByName = new Map<string, string>();
    for (const k of declOrder) {
      const e = declMap.get(k)!;
      if (e.stmt.kind === "variableDecl") varKeyByName.set(e.stmt.name, k);
    }
    // deps: key -> set of declaration keys that must be emitted before it.
    const indeg = new Map<string, number>();
    const adj = new Map<string, string[]>();
    for (const k of declOrder) { indeg.set(k, 0); adj.set(k, []); }
    for (const k of declOrder) {
      const e = declMap.get(k)!;
      if (e.stmt.kind !== "variableDecl") continue;
      const ids = new Set<string>();
      collectIdentifiers((e.stmt as any).value, ids);
      for (const id of ids) {
        const dk = varKeyByName.get(id);
        if (dk && dk !== k) {
          adj.get(dk)!.push(k);
          indeg.set(k, indeg.get(k)! + 1);
        }
      }
    }
    const origIndex = new Map(declOrder.map((k, i) => [k, i] as const));
    const ready = declOrder.filter(k => indeg.get(k) === 0);
    const result: string[] = [];
    const emitted = new Set<string>();
    while (ready.length) {
      // Pick the lowest original index among ready nodes to stay stable.
      ready.sort((a, b) => origIndex.get(a)! - origIndex.get(b)!);
      const k = ready.shift()!;
      result.push(k);
      emitted.add(k);
      for (const m of adj.get(k)!) {
        indeg.set(m, indeg.get(m)! - 1);
        if (indeg.get(m) === 0) ready.push(m);
      }
    }
    // Cycle fallback: append any unresolved declarations in original order.
    if (result.length < declOrder.length) {
      for (const k of declOrder) if (!emitted.has(k)) result.push(k);
    }
    return result;
  })();

  const declarations: string[] = [];
  let lastFilename = "";
  for (const k of sortedDeclOrder) {
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

  const RUNTIME_IMPORT = buildRuntimeImport(options?.runtimePath ?? "./runtime/runtime.js");

  let output = RUNTIME_IMPORT;

  // Inject imports for names referenced from separately compiled external libraries (resolved per kind against each library's manifest exports).
  if (externalLibraries.length > 0) {
    const refs = collectProgramReferences(program.statements);
    const importsBySpec = new Map<string, Set<string>>();
    const addImp = (spec: string, sym: string) => {
      let set = importsBySpec.get(spec);
      if (!set) { set = new Set(); importsBySpec.set(spec, set); }
      set.add(sym);
    };
    for (const lib of externalLibraries) {
      for (const m of refs.modules) {
        const file = lib.manifest.exports.modules[m];
        if (file) addImp(lib.importSpecifierFor(file), `${escapeName(m)}$mod`);
      }
      for (const f of refs.functions) {
        const file = lib.manifest.exports.functions[f];
        if (file) addImp(lib.importSpecifierFor(file), `${escapeName(f)}_fn`);
      }
      for (const v of refs.variables) {
        const file = lib.manifest.exports.variables[v];
        if (file) addImp(lib.importSpecifierFor(file), escapeName(v));
      }
    }
    // Side-effect imports first so library top-level statements (e.g. setting __ctx.$slop) run before the consumer body, matching include semantics
    const seenSideEffect = new Set<string>();
    for (const lib of externalLibraries) {
      for (const spec of lib.sideEffectSpecifiers ?? []) {
        if (seenSideEffect.has(spec)) continue;
        seenSideEffect.add(spec);
        output += `import "${spec}";\n`;
      }
    }
    for (const [spec, syms] of importsBySpec) {
      output += `import { ${[...syms].join(", ")} } from "${spec}";\n`;
    }
  }

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
  for (const [filename, info] of encounteredSurfaceData) {
    const runtimeDir = options?.runtimePath ? path.dirname(options.runtimePath).replace(/\\/g, "/") : "./runtime";
    const importPath = info.kind === "image" ? `${runtimeDir}/surface_data/${info.stem}_base64.js` : `${runtimeDir}/surface_data/${info.stem}_data.js`;
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

  output += `let PI: any = __rt.PI;\n` +
    `let INF: any = __rt.INF;\n` +
    `let NAN: any = __rt.NAN;\n` +
    `let undef: any = __rt.undef;\n` +
    `let _EPSILON: any = __rt._EPSILON;\n\n`;

  if (declarations.length) {
    output += declarations.join("\n") + "\n\n";
  }

  const topLevelVarKeys = new Set(declOrder.filter(k => k.startsWith("var:")).map(k => k.slice(4)));
  const alreadyDeclaredAtTop = new Set<string>([
    ...PRE_DECLARED_VARS,
    "$color", "$idx",
    ...topLevelVarKeys,
  ]);

  for (const v of dynamicScopeVars) {
    if (!v.startsWith("$") && !alreadyDeclaredAtTop.has(v)) {
      output += `let ${v}: any;\n`;
    }
  }

  const { referenced } = collectIdentifierUsage(program);
  const moduleLevelDeclared = new Set<string>([
    ...alreadyDeclaredAtTop,
    ...dynamicScopeVars,
    ...BUILTIN_FUNCTIONS,
    // Variables imported from external libraries
    ...[...externalVariableNames].map(escapeName),
    // Reserved module-level names emitted by the compiler/runtime
    "result", "background", "Manifold", "CrossSection", "wasm", "__NO_ARG",
  ]);

  const undefinedNames = [...referenced]
    .filter(n => !n.startsWith("__") && !n.startsWith("$") && !moduleLevelDeclared.has(n))
    .sort();
  for (const name of undefinedNames) {
    output += `let ${name}: any = undefined;\n`;
  }

  // Provide a global container for user variables and helpers
  output += `let __NO_ARG: any = Symbol("NO_ARG");\n`;

  if (geometryLines.length === 0) {
    output += `export const result = Manifold.union([]);\n`;
  } else {
    output += `const __result_items: any[] = [];\n`;
    output += `const __background_items: any[] = [];\n`;
    output += `${geometryLines.join("\n")}\n`;
    output += `export const result = __union2d3d(__result_items);\n`;
    output += `export const background = __union2d3d(__background_items);\n`;
  }
  output += `export const __viewport = { vpr: __ctx.$vpr, vpt: __ctx.$vpt, vpd: __ctx.$vpd, vpf: __ctx.$vpf };\n`;

  return output;
}

// Separate library compilation 
export interface CompiledLibraryFile {
  sourceRel: string;
  outRel: string;
  code: string;
}
export interface CompiledLibrary {
  manifest: LibraryManifest;
  files: CompiledLibraryFile[];
}

type LibDeclKind = "module" | "function" | "variable";
function declKindAndName(stmt: Statement): { kind: LibDeclKind; name: string } | undefined {
  if (stmt.kind === "moduleDecl") return { kind: "module", name: stmt.name };
  if (stmt.kind === "functionDecl") return { kind: "function", name: stmt.name };
  if (stmt.kind === "variableDecl" && !(stmt.name.startsWith("$"))) return { kind: "variable", name: stmt.name };
  return undefined;
}

const LIB_BUILTIN_CONSTS = new Set(["PI", "INF", "NAN", "undef", "_EPSILON"]);

export function compileLibrary(
  closure: LibraryClosure,
  opts: { runtimeVersion: string; runtimePathFor: (outRel: string) => string },
): CompiledLibrary {
  const sourceRels = [...closure.files.keys()].sort();
  const outRelOf = (sourceRel: string) => sourceRel.replace(/\.scad$/i, ".ts");
  // Library files use `var` for top-level vars to survive circular ESM imports.
  globalVarDeclKeyword = "var";

  signatures.clear();
  for (const [k, v] of Object.entries(BUILTIN_SIGNATURES)) {
    signatures.set(k, { params: v, defaults: new Array(v.length).fill(undefined) });
  }
  externalModuleNames = new Set();
  externalFunctionNames = new Set();
  externalVariableNames = new Set();
  const allStatements: Statement[] = [];
  for (const rel of sourceRels) allStatements.push(...closure.files.get(rel)!.statements);
  collectSignatures(allStatements);
  moduleDeclRegistry = collectModuleDeclarations(allStatements);
  globalVarNames = new Set();
  for (const s of allStatements) {
    if (s.kind === "variableDecl") globalVarNames.add(escapeName(s.name));
  }

  // Build the per-kind export map (name -> owning source file), last-wins with collisions recorded, plus the manifest signatures and per-file decl lists
  const exportsByKind = {
    module: new Map<string, string>(),
    function: new Map<string, string>(),
    variable: new Map<string, string>(),
  };
  const ambiguous: Record<string, string[]> = {};
  const manifestSignatures: Record<string, string[]> = {};
  const perFileDecls = new Map<string, { modules: string[]; functions: string[]; variables: string[] }>();

  for (const rel of sourceRels) {
    const program = closure.files.get(rel)!;
    const lists = { modules: [] as string[], functions: [] as string[], variables: [] as string[] };
    for (const stmt of program.statements) {
      const dk = declKindAndName(stmt);
      if (!dk) continue;
      const map = exportsByKind[dk.kind];
      const prior = map.get(dk.name);
      if (prior !== undefined && prior !== rel) {
        const key = `${dk.kind}:${dk.name}`;
        if (!ambiguous[key]) ambiguous[key] = [prior];
        ambiguous[key].push(rel);
      }
      map.set(dk.name, rel); // last-wins
      if (dk.kind === "module") {
        lists.modules.push(dk.name);
        manifestSignatures[`${escapeName(dk.name)}$mod`] = (stmt as any).params.map((p: any) => p.name);
      } else if (dk.kind === "function") {
        lists.functions.push(dk.name);
        manifestSignatures[`${escapeName(dk.name)}_fn`] = (stmt as any).params.map((p: any) => p.name);
      } else {
        lists.variables.push(dk.name);
      }
    }
    perFileDecls.set(rel, lists);
  }

  // Emit each file
  const files: CompiledLibraryFile[] = [];
  for (const rel of sourceRels) {
    const program = closure.files.get(rel)!;
    const outRel = outRelOf(rel);
    files.push({
      sourceRel: rel,
      outRel,
      code: emitLibraryFile(rel, outRel, program, {
        exportsByKind, deps: closure.deps.get(rel) ?? [], outRelOf,
        runtimePath: opts.runtimePathFor(outRel),
      }),
    });
  }

  const manifestFiles: LibraryManifest["files"] = {};
  for (const rel of sourceRels) {
    const lists = perFileDecls.get(rel)!;
    manifestFiles[rel] = { out: outRelOf(rel), ...lists };
  }

  const manifest: LibraryManifest = {
    library: closure.name,
    compiledAt: new Date().toISOString(),
    runtimeVersion: opts.runtimeVersion,
    files: manifestFiles,
    exports: {
      modules: Object.fromEntries(exportsByKind.module),
      functions: Object.fromEntries(exportsByKind.function),
      variables: Object.fromEntries(exportsByKind.variable),
    },
    ambiguous,
    signatures: manifestSignatures,
  };

  globalVarDeclKeyword = "let";
  return { manifest, files };
}

function emitLibraryFile(
  sourceRel: string,
  outRel: string,
  program: Program,
  ctx: {
    exportsByKind: { module: Map<string, string>; function: Map<string, string>; variable: Map<string, string> };
    deps: string[];
    outRelOf: (sourceRel: string) => string;
    runtimePath: string;
  },
): string {
  // Reset per-file emitter state
  dynamicScopeVars = new Set();
  activeShadowRenames = new Map();
  encounteredFonts = new Set();
  encounteredSurfaceData.clear();
  currentRuntimePath = ctx.runtimePath;

  // Top-level declarations, deduped last-wins (matching compile())
  const declMap = new Map<string, { stmt: Statement; code: string }>();
  const declOrder: string[] = [];
  const ownNames = { module: new Set<string>(), function: new Set<string>(), variable: new Set<string>() };
  const exportedSymbols: string[] = [];

  for (const stmt of program.statements) {
    if (stmt.kind === "variableDecl" || stmt.kind === "moduleDecl" || stmt.kind === "functionDecl") {
      let key: string;
      if (stmt.kind === "variableDecl") key = `var:${escapeName(stmt.name)}`;
      else if (stmt.kind === "moduleDecl") key = `fn:${escapeName(stmt.name)}$mod`;
      else key = `fn:${escapeName(stmt.name)}_fn`;
      if (!declMap.has(key)) declOrder.push(key);
      declMap.set(key, { stmt, code: compileDeclaration(stmt) });
      const dk = declKindAndName(stmt);
      if (dk) ownNames[dk.kind].add(dk.name);
    }
  }

  const declarations: string[] = [];
  for (const key of declOrder) {
    const entry = declMap.get(key)!;
    declarations.push(entry.code);
    const dk = declKindAndName(entry.stmt);
    if (dk?.kind === "module") exportedSymbols.push(`${escapeName(dk.name)}$mod`);
    else if (dk?.kind === "function") exportedSymbols.push(`${escapeName(dk.name)}_fn`);
    else if (dk?.kind === "variable") exportedSymbols.push(escapeName(dk.name));
  }

  // Resolve cross-file references to imports
  const refs = collectProgramReferences(program.statements);
  const importsBySpec = new Map<string, Set<string>>();
  const importedVarNames = new Set<string>();
  const addImp = (ownerRel: string, sym: string) => {
    const spec = relImportSpecifier(outRel, ctx.outRelOf(ownerRel));
    let set = importsBySpec.get(spec);
    if (!set) { set = new Set(); importsBySpec.set(spec, set); }
    set.add(sym);
  };

  for (const m of refs.modules) {
    if (BUILTIN_MODULES.has(m) || ownNames.module.has(m)) continue;
    const owner = ctx.exportsByKind.module.get(m);
    if (owner && owner !== sourceRel) addImp(owner, `${escapeName(m)}$mod`);
    else if (!owner) console.warn(`Warning: library ${sourceRel}: unresolved module '${m}' (emitting no-op call)`);
  }
  for (const f of refs.functions) {
    if (BUILTIN_FUNCTIONS.has(f) || ownNames.function.has(f)) continue;
    const owner = ctx.exportsByKind.function.get(f);
    if (owner && owner !== sourceRel) addImp(owner, `${escapeName(f)}_fn`);
  }
  for (const v of refs.variables) {
    if (LIB_BUILTIN_CONSTS.has(v) || ownNames.variable.has(v)) continue;
    const owner = ctx.exportsByKind.variable.get(v);
    if (owner && owner !== sourceRel) { addImp(owner, escapeName(v)); importedVarNames.add(escapeName(v)); }
  }

  // Side-effect imports for under-root deps, to preserve include-time execution order
  let sideEffectBlock = "";
  for (const dep of ctx.deps) {
    if (dep === sourceRel) continue;
    sideEffectBlock += `import "${relImportSpecifier(outRel, ctx.outRelOf(dep))}";\n`;
  }

  let importBlock = "";
  for (const [spec, syms] of importsBySpec) {
    importBlock += `import { ${[...syms].join(", ")} } from "${spec}";\n`;
  }

  // Undefined fallbacks: referenced identifiers that aren't local, imported, a builtin const, or a $-special. OpenSCAD resolves unknown reads to undef
  const { referenced } = collectIdentifierUsage(program);
  const localDeclared = new Set<string>([
    ...[...ownNames.variable].map(escapeName),
    ...[...ownNames.module].map(n => `${escapeName(n)}$mod`),
    ...[...ownNames.function].map(n => `${escapeName(n)}_fn`),
    ...LIB_BUILTIN_CONSTS, ...BUILTIN_FUNCTIONS, ...importedVarNames,
    "Manifold", "CrossSection", "wasm", "__NO_ARG",
  ]);
  const undefinedNames = [...referenced]
    .filter(n => !n.startsWith("__") && !n.startsWith("$") && !localDeclared.has(n))
    .sort();

  let out = buildRuntimeImport(ctx.runtimePath);
  out += sideEffectBlock;
  out += importBlock;
  out += `const __font_registry: Record<string, string> = {};\n`;
  out += `let PI: any = __rt.PI;\n`;
  out += `let INF: any = __rt.INF;\n`;
  out += `let NAN: any = __rt.NAN;\n`;
  out += `let undef: any = __rt.undef;\n`;
  out += `let _EPSILON: any = __rt._EPSILON;\n`;
  out += `let __NO_ARG: any = Symbol("NO_ARG");\n`;
  for (const name of undefinedNames) out += `let ${name}: any = undefined;\n`;
  out += "\n";
  if (declarations.length) out += declarations.join("\n") + "\n";
  if (exportedSymbols.length) out += `\nexport { ${exportedSymbols.join(", ")} };\n`;
  return out;
}

// Relative ES import specifier from one output file to another
function relImportSpecifier(fromOutRel: string, toOutRel: string): string {
  let rel = path.relative(path.dirname(fromOutRel), toOutRel).replace(/\\/g, "/");
  rel = rel.replace(/\.ts$/i, ".js");
  if (!rel.startsWith(".")) rel = "./" + rel;
  return rel;
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
  "$color", "$idx",
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
      if (stmt.name.startsWith("$") && stmt.name !== "$children") {
        return withLeading(`${svTarget(stmt.name)} = ${compileExpr(stmt.value)};`);
      }
      if (PRE_DECLARED_VARS.has(stmt.name)) {
        return withLeading(`${name} = ${compileExpr(stmt.value)};`);
      }
      return withLeading(`${globalVarDeclKeyword} ${name}: any = ${compileExpr(stmt.value)};`);
    }

    case "moduleDecl": {
      const dedup = deduplicateParams(stmt.params);
      const isDyn = (n: string) => n.startsWith("$") && n !== "$children";
      const renamedParams: string[] = [];
      const params = dedup.map(p => {
        const base = escapeName(p.name);
        const selfRef = !!p.defaultValue && nodeReferencesIdentifier(p.defaultValue, p.name);
        let pname: string;
        if (isDyn(p.name)) {
          pname = `${base}__arg`;
        } else if (selfRef) {
          pname = `${base}__arg`;
          renamedParams.push(base);
        } else {
          pname = base;
        }
        return p.defaultValue
          ? `${pname}: any = ${compileExpr(p.defaultValue)}`
          : `${pname}: any`;
      }).join(", ");
      const localParams = dedup.map(p => escapeName(p.name));
      const dollarParams = dedup.filter(p => isDyn(p.name)).map(p => escapeName(p.name));
      const body = compileModuleBody(stmt.body, stmt.name, localParams, dollarParams, renamedParams);
      return withLeading(`function ${escapeName(stmt.name)}$mod(${params}): any {\n${body}\n}`);
    }

    case "functionDecl": {
      const dedup = deduplicateParams(stmt.params);
      const renamedParams: string[] = [];
      const params = dedup.map(p => {
        const base = escapeName(p.name);
        const selfRef = !!p.defaultValue && nodeReferencesIdentifier(p.defaultValue, p.name);
        let pname = base;
        if (selfRef) {
          pname = `${base}__arg`;
          renamedParams.push(base);
        }
        return p.defaultValue
          ? `${pname}: any = ${compileExpr(p.defaultValue)}`
          : `${pname}: any`;
      }).join(", ");
      const localParams = dedup.map(p => escapeName(p.name));
      const bodyExpr = withLocalScope(localParams, () => compileExpr(stmt.body));
      const rebinds = renamedParams.map(n => `  let ${n}: any = ${n}__arg;\n`).join("");
      return withLeading(`function ${escapeName(stmt.name)}_fn(${params}): any {\n${rebinds}  return ${bodyExpr};\n}`);
    }

    default:
      return `/* unsupported declaration: ${(stmt as Statement).kind}${locTag(stmt)} */`;
  }
}

// True when an expression references the identifier name anywhere within it
function nodeReferencesIdentifier(node: unknown, name: string): boolean {
  if (!node || typeof node !== "object") return false;
  const n = node as Record<string, unknown>;
  if (n.kind === "identifier" && n.name === name) return true;
  for (const key in n) {
    if (key === "loc" || key === "leadingComments" || key === "trailingComments") continue;
    const v = n[key];
    if (Array.isArray(v)) {
      for (const item of v) if (nodeReferencesIdentifier(item, name)) return true;
    } else if (v && typeof v === "object") {
      if (nodeReferencesIdentifier(v, name)) return true;
    }
  }
  return false;
}

function collectIdentifiers(node: unknown, out: Set<string>): void {
  if (!node || typeof node !== "object") return;
  const n = node as Record<string, unknown>;
  if (n.kind === "identifier" && typeof n.name === "string") out.add(n.name);
  for (const key in n) {
    if (key === "loc" || key === "leadingComments" || key === "trailingComments") continue;
    const v = n[key];
    if (Array.isArray(v)) {
      for (const item of v) collectIdentifiers(item, out);
    } else if (v && typeof v === "object") {
      collectIdentifiers(v, out);
    }
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

// Module body compilation
function compileModuleBody(body: Statement, moduleName?: string, localParamNames: string[] = [], dollarParamNames: string[] = [], renamedParamNames: string[] = []): string {
  const stmts = body.kind === "block" ? body.statements : [body];
  const localVarNames = collectLocalVariableNames(stmts);

  const lines: string[] = [];

  // Capture children from the stack at the start of every module body
  lines.push("  let __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };");
  lines.push("  let $children: any = __c.count;");
  lines.push("  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }");
  lines.push("  let __save_$parent_modules: any = __ctx.$parent_modules;");
  lines.push("  __ctx.$parent_modules = __children_stack.length;");
  lines.push("  const __items: any[] = [];");

  const decls: string[] = [];
  const geos: string[] = [];
  const dollarSaves: string[] = [];
  const dollarRestores: string[] = [];
  const dollarParamSets: string[] = [];

  for (const dp of dollarParamNames) {
    dollarSaves.push(`  let __save_${dp}: any = ${svTarget(dp)};`);
    dollarParamSets.push(`  ${svTarget(dp)} = ${dp}__arg;`);
    dollarRestores.push(`  ${svTarget(dp)} = __save_${dp};`);
  }

  const shadowLocals = new Set([...localVarNames].filter(n => globalVarNames.has(n)));
  const savedShadowRenames = activeShadowRenames;
  activeShadowRenames = new Map();

  const declaredInBody = new Set<string>(localParamNames);
  const savedDollars = new Set<string>(dollarParamNames);

  withLocalScope([...localParamNames, ...localVarNames], () => {
    for (const s of stmts) {
      if (s.kind === "empty") continue;
      if (s.kind === "variableDecl") {
        const name = escapeName(s.name);
        const valueExpr = compileExpr(s.value);
        const commentsBefore = leadingCommentLines(s, "  ");
        const commentAfter = trailingCommentText(s);
        if (s.name.startsWith("$") && s.name !== "$children") {
          // Dynamic scoping: save/assign/restore for $ variables (in __ctx)
          if (!savedDollars.has(name)) {
            savedDollars.add(name);
            dollarSaves.push(`  let __save_${name}: any = ${svTarget(name)};`);
            dollarRestores.push(`  ${svTarget(name)} = __save_${name};`);
          }
          decls.push(...commentsBefore);
          decls.push(`  ${svTarget(name)} = ${valueExpr};${commentAfter}`);
        } else {
          const emitName = shadowLocals.has(name) ? `${name}__sl` : name;
          decls.push(...commentsBefore);
          if (declaredInBody.has(emitName)) {
            decls.push(`  ${emitName} = ${valueExpr};${commentAfter}`);
          } else {
            declaredInBody.add(emitName);
            decls.push(`  let ${emitName}: any = ${valueExpr};${commentAfter}`);
          }
          if (shadowLocals.has(name)) activeShadowRenames.set(name, emitName);
        }
      } else if (s.kind === "functionDecl" || s.kind === "moduleDecl") {
        // Indent the nested declaration
        const decl = compileDeclaration(s);
        decls.push("  " + decl.split("\n").join("\n  "));
      } else {
        const geo = compileGeometry(s);
        if (!geo) continue;
        if (hasBackgroundModifier(s)) {
          pushCommentedLine(geos, s, `  __background_items.push(${geo});`, "  ");
        } else {
          pushCommentedLine(geos, s, `  __items.push(${geo});`, "  ");
        }
      }
    }
  });

  activeShadowRenames = savedShadowRenames;

  if (dollarRestores.length > 0) {
    lines.splice(4, 0, ...dollarSaves);
  }

  lines.push(...dollarParamSets);
  // Rebind self-referential parameters (renamed to `<name>__arg` in the signature) to their OpenSCAD names so the body can use them normally.
  lines.push(...renamedParamNames.map(n => `  let ${n}: any = ${n}__arg;`));
  lines.push(...decls);
  lines.push(...geos);

  lines.push(`  try {`);
  lines.push(`    return __union2d3d(__items);`);
  lines.push(`  } finally {`);
  if (dollarRestores.length > 0) {
    lines.push(...dollarRestores.map(r => `  ${r}`));
  }
  lines.push(`    __ctx.$parent_modules = __save_$parent_modules;`);   // ← restore
  lines.push(`  }`);
  
  return lines.join("\n");
}

function hasBackgroundModifier(stmt: Statement): boolean {
  const m = (stmt as { modifier?: string }).modifier;
  return typeof m === "string" && m.includes("%");
}

function isStatementBackgroundOnly(stmt: Statement, modules: Map<string, ModuleDeclStmtType>, visited: Set<string>): boolean {
  if (hasBackgroundModifier(stmt)) return true;
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

const OVERRIDABLE_BUILTINS = new Set([
  "cube", "sphere", "cylinder", "circle", "square", "polygon", "text",
]);

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

  // A library (e.g. BOSL2) may redefine a built-in geometry primitive as an
  // attachable module. When such a call carries a children block (e.g.
  // `cylinder(...) { attach(...) ... }`) we must route to the module so the
  // children aren't dropped — a primitive IR node has no children slot. We
  // gate on children presence so the library's own childless internal calls
  // (cuboid/attachable drawing helper cubes, etc.) keep using the cheap
  // built-in primitive and don't recurse back through attachable().
  if (loweredChildren.length > 0 && OVERRIDABLE_BUILTINS.has(name) && ctx.modules.has(name)) {
    return {
      kind: "moduleCall",
      name,
      args: stmt.args,
      children: loweredChildren,
      loc: stmt.loc,
    } as IRModuleCallNode;
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
      return `(__truthy(${cond}) ? ${thenNode} : ${elseNode})`;
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
      return `__safe_offset2d(${child}, ${amt}, "Round", 2, __ctx.$fn, __ctx.$fa, __ctx.$fs)`;
    }
    case "color": {
      const c = findArg(node.args, "c", 0);
      const alpha = findArg(node.args, "alpha", 1);
      const cExpr = c ? compileExpr(c.value) : "undefined";
      const aExpr = alpha ? compileExpr(alpha.value) : "undefined";
      return `(() => { let __save_$color: any = __ctx.$color; __ctx.$color = __parse_color_for_scope(${cExpr}, ${aExpr}); try { return __apply_color(${child}, ${cExpr}, ${aExpr}); } finally { __ctx.$color = __save_$color; } })()`;
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
        `fn: ${fn ? compileExpr(fn.value) : "__ctx.$fn"}`
      );

      opts.push(
        `fa: ${fa ? compileExpr(fa.value) : "__ctx.$fa"}`
      );

      opts.push(
        `fs: ${fs ? compileExpr(fs.value) : "__ctx.$fs"}`
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
      const fnStr = fn ? compileExpr(fn.value) : "__ctx.$fn";
      const faStr = fa ? compileExpr(fa.value) : "__ctx.$fa";
      const fsStr = fs ? compileExpr(fs.value) : "__ctx.$fs";
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
  const dollarArgs = stmt.args.filter(arg => arg.name && arg.name.startsWith("$"));
  const userSig = signatures.get(`${escapeName(stmt.name)}$mod`);
  const extraArgs = (moduleDeclRegistry.has(stmt.name) && userSig)
    ? stmt.args.filter(a => a.name && !a.name.startsWith("$") && !userSig.params.includes(a.name))
    : [];

  let result: string;
  const hasChildBlock = !!stmt.child && stmt.child.kind !== "empty";
  if (hasChildBlock && OVERRIDABLE_BUILTINS.has(stmt.name) && moduleDeclRegistry.has(stmt.name)) {
    result = compileUserModuleCall(stmt);
  } else
  switch (stmt.name) {
    // Primitives
    case "cube": result = compileCube(stmt.args); break;
    case "sphere": result = compileSphere(stmt.args); break;
    case "cylinder": result = compileCylinder(stmt.args); break;
    case "circle": result = compileCircle(stmt.args); break;
    case "square": result = compileSquare(stmt.args); break;
    case "polygon": result = compilePolygon(stmt.args); break;
    case "polyhedron": result = compilePolyhedron(stmt.args); break;
    case "text": result = compileText(stmt.args); break;
    case "surface": result = compileSurface(stmt.args); break;

    // Transforms
    case "translate": result = compileTransform(stmt, "translate"); break;
    case "rotate": result = compileTransform(stmt, "rotate"); break;
    case "scale": result = compileTransform(stmt, "scale"); break;
    case "mirror": result = compileMirror(stmt); break;
    case "multmatrix": result = compileMultMatrix(stmt); break;
    case "resize": result = compilePassthrough(stmt, "resize"); break;
    case "offset": result = compileOffset(stmt); break;
    case "color": result = compileColor(stmt); break;
    case "render": result = compilePassthrough(stmt, "render"); break;
    case "projection": result = compileProjection(stmt); break;

    // Boolean operations
    case "group": result = compileBoolOp(stmt, "union"); break; // group() == implicit union
    case "union": result = compileBoolOp(stmt, "union"); break;
    case "difference": result = compileDifference(stmt); break;
    case "intersection": result = compileBoolOp(stmt, "intersection"); break;
    case "hull": result = compileBoolOp(stmt, "hull"); break;
    case "minkowski": result = compileMinkowski(stmt); break;

    // Extrusion
    case "linear_extrude": result = compileLinearExtrude(stmt); break;
    case "rotate_extrude": result = compileRotateExtrude(stmt); break;

    // Builtin statement modifiers
    case "echo": result = compileEchoModule(stmt); break;
    case "assert": result = compileAssertModule(stmt); break;
    case "let": result = compileLetModule(stmt); break;
    case "children": result = compileChildrenModule(stmt); break;
    case "intersection_for": result = compileIntersectionFor(stmt); break;

    default:
      result = compileUserModuleCall(stmt); break;
  }

  const dynArgs = [...dollarArgs, ...extraArgs];
  if (dynArgs.length === 0) {
    return result;
  }

  const decls: string[] = [];
  const saves: string[] = [];
  const restores: string[] = [];

  for (const arg of dynArgs) {
    const name = escapeName(arg.name!);
    // $-vars live in __ctx; non-$ extra args remain module-level bindings
    if (!name.startsWith("$")) dynamicScopeVars.add(name);
    const valStr = compileExpr(arg.value);
    saves.push(`let __save_${name}: any = ${svTarget(name)};`);
    decls.push(`${svTarget(name)} = ${valStr};`);
    restores.push(`${svTarget(name)} = __save_${name};`);
  }

  const hasAwait = result.includes("await ");
  if (hasAwait) {
    return `await (async () => { ${saves.join(" ")} ${decls.join(" ")} try { return await ${result}; } finally { ${restores.join(" ")} } })()`;
  } else {
    return `(() => { ${saves.join(" ")} ${decls.join(" ")} try { return ${result}; } finally { ${restores.join(" ")} } })()`;
  }
}

// Builtin module helpers
function compileEchoModule(stmt: ModuleCallStmt): string {
  const args = stmt.args.map(a =>
    a.name ? `"${a.name} =", ${compileExpr(a.value)}` : compileExpr(a.value)
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

  // Resolve d vs r at runtime: a pass-through wrapper may forward both with one
  const argOr = (a: Argument | undefined) => (a ? compileExpr(a.value) : "undefined");
  const radiusStr = `__radius(undefined, undefined, ${argOr(d)}, ${argOr(r)}, 1)`;

  const fnStr = fn ? compileExpr(fn.value) : "__ctx.$fn";
  const faStr = fa ? compileExpr(fa.value) : "__ctx.$fa";
  const fsStr = fs ? compileExpr(fs.value) : "__ctx.$fs";
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

  // Resolve each radius at runtime following OpenSCAD precedence
  const argOr = (a: Argument | undefined) => (a ? compileExpr(a.value) : "undefined");
  const rLow = `__radius(${argOr(d1)}, ${argOr(r1)}, ${argOr(d)}, ${argOr(r)}, 1)`;
  const rHigh = `__radius(${argOr(d2)}, ${argOr(r2)}, ${argOr(d)}, ${argOr(r)}, 1)`;

  const fnStr = fn ? compileExpr(fn.value) : "__ctx.$fn";
  const centerStr = center ? compileExpr(center.value) : "false";
  const faStr = fa ? compileExpr(fa.value) : "__ctx.$fa";
  const fsStr = fs ? compileExpr(fs.value) : "__ctx.$fs";

  return `__cylinder(${hStr}, ${rLow}, ${rHigh}, ${fnStr}, ${centerStr}, ${faStr}, ${fsStr})`;
}

function compileCircle(args: Argument[]): string {
  const r = findArg(args, "r", 0);
  const d = findArg(args, "d");
  const fn = findArg(args, "$fn");
  const fa = findArg(args, "$fa");
  const fs = findArg(args, "$fs");

  // Resolve d-vs-r at runtime (see __radius).
  const argOr = (a: Argument | undefined) => (a ? compileExpr(a.value) : "undefined");
  const radiusStr = `__radius(undefined, undefined, ${argOr(d)}, ${argOr(r)}, 1)`;

  const fnStr = fn ? compileExpr(fn.value) : "__ctx.$fn";
  const faStr = fa ? compileExpr(fa.value) : "__ctx.$fa";
  const fsStr = fs ? compileExpr(fs.value) : "__ctx.$fs";
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
  const fnStr = fn      ? compileExpr(fn.value)      : `__ctx.$fn`;

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

  return `__polyhedron(${compileExpr(points.value)}, ${compileExpr(faces.value)})`;
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
    return `await (async () => { let __save_$color: any = __ctx.$color; __ctx.$color = __parse_color_for_scope(${cExpr}, ${aExpr}); try { return await __apply_color(${child}, ${cExpr}, ${aExpr}); } finally { __ctx.$color = __save_$color; } })()`;
  }
  return `(() => { let __save_$color: any = __ctx.$color; __ctx.$color = __parse_color_for_scope(${cExpr}, ${aExpr}); try { return __apply_color(${child}, ${cExpr}, ${aExpr}); } finally { __ctx.$color = __save_$color; } })()`;
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
  return `__safe_offset2d(${child}, ${amt}, "Round", 2, __ctx.$fn, __ctx.$fa, __ctx.$fs)`;
}

function compileProjection(stmt: ModuleCallStmt): string {
  if (!stmt.child) return "CrossSection.square(0)";
  const child = compileGeometry(stmt.child);
  if (child === "Manifold.union([])") return "CrossSection.square(0)";
  const cut = findArg(stmt.args, "cut", 0);
  const cutStr = cut ? compileExpr(cut.value) : "false";
  return `__safe_project3d(${child}, ${cutStr})`;
}

// echo()/assert() with no child are pure side-effect statements, not geometry
function isSideEffectOnlyModule(s: Statement): boolean {
  return s.kind === "moduleCall" && (s.name === "echo" || s.name === "assert") && !s.child;
}

function collectChildrenWithDecls(stmt: ModuleCallStmt, sideEffectsAsChildren = false): { decls: string[]; geos: string[] } {
  if (!stmt.child) return { decls: [], geos: [] };
  if (stmt.child.kind === "block") {
    return compileBlockStatementsWithDecls(stmt.child.statements, sideEffectsAsChildren);
  }
  if (hasBackgroundModifier(stmt.child)) return { decls: [], geos: [] };
  if (isSideEffectOnlyModule(stmt.child) && !sideEffectsAsChildren) {
    return { decls: [`${compileGeometry(stmt.child)};`], geos: [] };
  }
  const g = compileGeometry(stmt.child);
  return { decls: [], geos: g ? [g] : [] };
}

function compileBlockStatementsWithDecls(stmts: Statement[], sideEffectsAsChildren = false): { decls: string[]; geos: string[] } {
  const declItems: { name?: string; code: string }[] = [];
  const geos: string[] = [];

  const collect = (list: Statement[]) => {
    for (const s of list) {
      if (s.kind === "empty") continue;
      if (hasBackgroundModifier(s)) continue;
      if (s.kind === "variableDecl") {
        declItems.push({
          name: escapeName(s.name),
          code: `${leadingCommentLines(s).join("\n")}${s.leadingComments?.length ? "\n" : ""}let ${escapeName(s.name)}: any = ${compileExpr(s.value)};${trailingCommentText(s)}`,
        });
      } else if (s.kind === "functionDecl" || s.kind === "moduleDecl") {
        declItems.push({ code: compileDeclaration(s) });
      } else if (s.kind === "block") {
        collect(s.statements);
      } else if (isSideEffectOnlyModule(s) && !sideEffectsAsChildren) {
        // Run echo()/assert() for their side effects, but keep them out of geos
        declItems.push({ code: `${compileGeometry(s)};` });
      } else {
        const g = compileGeometry(s);
        if (g) {
          const leading = leadingCommentLines(s);
          geos.push(`${leading.length ? `${leading.join("\n")}\n` : ""}${g}`);
        }
      }
    }
  };
  collect(stmts);

  // Keep only the last decl per variable name (named decls only; function/module declarations have no name key and are all kept)
  const lastIdx = new Map<string, number>();
  declItems.forEach((it, i) => { if (it.name) lastIdx.set(it.name, i); });
  const decls = declItems
    .filter((it, i) => it.name === undefined || lastIdx.get(it.name) === i)
    .map(it => it.code);

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
    `fn: ${fn ? compileExpr(fn.value) : "__ctx.$fn"}`
  );

  opts.push(
    `fa: ${fa ? compileExpr(fa.value) : "__ctx.$fa"}`
  );

  opts.push(
    `fs: ${fs ? compileExpr(fs.value) : "__ctx.$fs"}`
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
  const fnStr = fn ? compileExpr(fn.value) : "__ctx.$fn";
  const faStr = fa ? compileExpr(fa.value) : "__ctx.$fa";
  const fsStr = fs ? compileExpr(fs.value) : "__ctx.$fs";
  return `__revolve(${child}, ${fnStr}, ${faStr}, ${fsStr}, ${aStr})`;
}

// Block geometry 
function compileBlockGeometry(block: BlockStmt): string {
  const items: { kind: "var" | "dollar" | "func" | "geo"; name?: string; code: string }[] = [];
  const localVarNames = collectLocalVariableNames(block.statements);

  withLocalScope(localVarNames, () => {
    for (const s of block.statements) {
      if (s.kind === "empty") continue;
      // A '%' (background) subtree is excluded from the enclosing union.
      if (hasBackgroundModifier(s)) continue;
      if (s.kind === "variableDecl") {
        const name = escapeName(s.name);
        const code = compileExpr(s.value);
        if (s.name.startsWith("$")) {
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
      const t = svTarget(d.name!);
      if (body.includes("await ")) {
        body = `await (async () => { let __save_${d.name}: any = ${t}; ${t} = ${d.code}; try { return await ${returnExpr(body, "  ")}; } finally { ${t} = __save_${d.name}; } })()`;
      } else {
        body = `(() => { let __save_${d.name}: any = ${t}; ${t} = ${d.code}; try { return ${returnExpr(body, "  ")}; } finally { ${t} = __save_${d.name}; } })()`;
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
      `  const __start: any = ${start}, __step: any = ${step}, __end: any = ${end};\n` +
      `  const __cnt: any = __rangeCount(__start, __step, __end);\n` +
      `  for (let __i = 0; __i < __cnt; __i++) {\n` +
      `    const ${escapeName(v.name)}: any = __start + __i * __step;\n` +
      `    __items.push(${inner});\n` +
      `  }\n` +
      `  return __items;\n` +
      `})())`;
  }

  // vector iteration
  const rangeExpr = compileExpr(v.range);
  return `__intersection2d3d(__flat_map_iter(${rangeExpr}, (${escapeName(v.name)}: any, __i: any) => { let __save_$idx: any = __ctx.$idx; __ctx.$idx = __i; try { return [${inner}]; } finally { __ctx.$idx = __save_$idx; } }))`;
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
      `  const __start: any = ${start}, __step: any = ${step}, __end: any = ${end};\n` +
      `  const __cnt: any = __rangeCount(__start, __step, __end);\n` +
      `  for (let __i = 0; __i < __cnt; __i++) {\n` +
      `    const ${escapeName(v.name)}: any = __start + __i * __step;\n` +
      `    __items.push(${inner});\n` +
      `  }\n` +
      `  return __items;\n` +
      `})())`;
  }

  // vector iteration
  const rangeExpr = compileExpr(v.range);
  return `__union2d3d(__flat_map_iter(${rangeExpr}, (${escapeName(v.name)}: any, __i: any) => { let __save_$idx: any = __ctx.$idx; __ctx.$idx = __i; try { return [${inner}]; } finally { __ctx.$idx = __save_$idx; } }))`;
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
      `${indent}  const __start_${idx}: any = ${start}, ${stepName}: any = ${step}, __end_${idx}: any = ${end};`,
      `${indent}  const __cnt_${idx}: any = __rangeCount(__start_${idx}, ${stepName}, __end_${idx});`,
      `${indent}  for (let __i_${idx} = 0; __i_${idx} < __cnt_${idx}; __i_${idx}++) {`,
      `${indent}    const ${vName}: any = __start_${idx} + __i_${idx} * ${stepName};`,
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
    `${indent}    let __save_$idx: any = __ctx.$idx; __ctx.$idx = ${idxName};`,
    `${indent}    try {`,
    ...buildNestedForStatements(vars, idx + 1, body, indentLevel + 3),
    `${indent}    } finally { __ctx.$idx = __save_$idx; }`,
    `${indent}  }`,
    `${indent}}`,
  ];
}

function compileIfGeometry(stmt: IfStmt): string {
  const cond = `__truthy(${compileExpr(stmt.condition)})`;
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
  if (!moduleDeclRegistry.has(stmt.name) && !externalModuleNames.has(stmt.name)) {
    const line = stmt.loc?.start.line;
    const where = line ? ` at line ${line}` : "";
    console.warn(`Warning: Ignoring unknown module '${stmt.name}'${where}`);
    return "Manifold.union([])";
  }

  const name = `${escapeName(stmt.name)}$mod`;
  const argList = compileArgList(name, stmt.args);
  const { decls, geos } = stmt.child && stmt.child.kind !== "empty" ? collectChildrenWithDecls(stmt, true) : { decls: [], geos: [] };
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

  // OpenSCAD does not abort when a surface() file is missing or cannot be opened - it prints a warning and the module yields no geometry
  if (!file?.value || file.value.kind !== "string") {
    console.warn(`Warning: surface(): no file argument given, ignoring.`);
    return "Manifold.union([])";
  }
  const filenameStr = file.value.value;

  const basePath = process.env.IMAGEBASEPATH;
  if (!basePath) {
    console.warn(`Warning: surface("${filenameStr}"): IMAGEBASEPATH is not set, can't open file, ignoring.`);
    return "Manifold.union([])";
  }

  const filePath = path.join(basePath, path.basename(filenameStr));
  if (!fs.existsSync(filePath)) {
    console.warn(`Warning: surface("${filenameStr}"): can't open file "${filePath}", ignoring.`);
    return "Manifold.union([])";
  }

  const ext = path.extname(filePath).toLowerCase();
  const isImage = ext === ".png"; // OpenSCAD treats only PNG as an image; everything else is a text matrix

  const stem = path.basename(filenameStr, path.extname(filenameStr)).replace(/[^a-zA-Z0-9_]/g, "_");

  const currentFileDir = typeof __dirname !== "undefined"
    ? __dirname
    : path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/i, "$1"));
  const compilerDir = path.resolve(currentFileDir, "..");
  const surfaceDataDir = path.join(compilerDir, "runtime", "surface_data");
  fs.mkdirSync(surfaceDataDir, { recursive: true });

  const centerStr = center ? compileExpr(center.value) : "false";

  if (isImage) {
    const base64 = fs.readFileSync(filePath).toString("base64");
    const mimeType = guessMimeType(filePath);
    const dataUrl = `data:${mimeType};base64,${base64}`;
    const exportName = `__img_${stem}`;

    const tsContent =
      `// Auto-generated by OpenSCAD compiler — do not edit\n` +
      `// Source: ${filePath}\n` +
      `export const ${exportName} = "${dataUrl}";\n`;

    fs.writeFileSync(path.join(surfaceDataDir, `${stem}_base64.ts`), tsContent, "utf8");
    encounteredSurfaceData.set(filenameStr, { stem, exportName, kind: "image" });

    return `await __surface(${exportName}, { center: ${centerStr}, kind: "image", fn: __ctx.$fn, fa: __ctx.$fa, fs: __ctx.$fs })`;
  } else {
    // Text matrix (.dat / .txt) - embed raw content as a string literal.
    const raw = fs.readFileSync(filePath, "utf8");
    const exportName = `__surfacedata_${stem}`;

    const tsContent =
      `// Auto-generated by OpenSCAD compiler — do not edit\n` +
      `// Source: ${filePath}\n` +
      `export const ${exportName} = ${JSON.stringify(raw)};\n`;

    fs.writeFileSync(path.join(surfaceDataDir, `${stem}_data.ts`), tsContent, "utf8");
    encounteredSurfaceData.set(filenameStr, { stem, exportName, kind: "text" });

    return `await __surface(${exportName}, { center: ${centerStr}, kind: "text", fn: __ctx.$fn, fa: __ctx.$fa, fs: __ctx.$fs })`;
  }
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
    case "identifier": {
      // $children stays as $children (the count variable set in module body)
      if (expr.name === "$children") return "$children";
      // All other special ($-prefixed) variables read through the shared runtime context so a value set in one compiled file is visible in another.
      if (expr.name.startsWith("$")) return `__ctx.${expr.name}`;
      const en = escapeName(expr.name);
      return activeShadowRenames.get(en) ?? en;
    }
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
      if (expr.op === "&") return `__band(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === "|") return `__bor(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === "<<") return `__shl(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === ">>") return `__shr(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === "<") return `__lt(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === ">") return `__gt(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === "<=") return `__le(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === ">=") return `__ge(${compileExpr(expr.left)}, ${compileExpr(expr.right)})`;
      if (expr.op === "&&" || expr.op === "||") {
        return `(__truthy(${compileExpr(expr.left)}) ${expr.op} __truthy(${compileExpr(expr.right)}))`;
      }
      return `(${compileExpr(expr.left)} ${expr.op} ${compileExpr(expr.right)})`;
    case "unary":
      if (expr.op === "-") return `__neg(${compileExpr(expr.operand)})`;
      if (expr.op === "+") return `__pos(${compileExpr(expr.operand)})`;
      if (expr.op === "!") return `(!__truthy(${compileExpr(expr.operand)}))`;
      if (expr.op === "~") return `__bnot(${compileExpr(expr.operand)})`;
      return `(${expr.op}${compileExpr(expr.operand)})`;
    case "ternary": {
      let ifTrue = compileExpr(expr.ifTrue);
      let ifFalse = compileExpr(expr.ifFalse);
      const trueSpread = ifTrue.startsWith("...");
      const falseSpread = ifFalse.startsWith("...");
      if (trueSpread || falseSpread) {
        if (trueSpread) ifTrue = ifTrue.slice(3); else ifTrue = `[${ifTrue}]`;
        if (falseSpread) ifFalse = ifFalse.slice(3); else ifFalse = `[${ifFalse}]`;
        return `...(__truthy(${compileExpr(expr.condition)}) ? ${ifTrue} : ${ifFalse})`;
      }
      return `(__truthy(${compileExpr(expr.condition)}) ? ${ifTrue} : ${ifFalse})`;
    }
    case "call":
      return compileCallExpr(expr);
    case "index":
      return `__index(${compileExpr(expr.object)}, ${compileExpr(expr.index)})`;
    case "member": {
      const memberMap: Record<string, string> = { x: "0", y: "1", z: "2" };
      const idx = memberMap[expr.property];
      if (idx !== undefined) {
        return `${compileExpr(expr.object)}?.[${idx}]`;
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
        a.name ? `"${a.name} =", ${compileExpr(a.value)}` : compileExpr(a.value)
      ).join(", ");
      return `(console.log(${eArgs}), ${compileExpr(expr.expr)})`;
    }
    case "assert": {
      const condition = expr.args[0] ? compileExpr(expr.args[0].value) : "true";
      const message = expr.args[1] ? compileExpr(expr.args[1].value) : '"Assertion failed"';
      return `(openscad_assert_fn(${condition}, ${message}), ${compileExpr(expr.expr)})`;
    }
    case "let": {
      const localAssignNames = expr.assignments
        .filter(a => !a.name.startsWith("$"))
        .map(a => escapeName(a.name));
      return withLocalScope(localAssignNames, () => {
        let result = compileExpr(expr.body);
        for (let i = expr.assignments.length - 1; i >= 0; i--) {
          const a = expr.assignments[i]!;
          const name = escapeName(a.name);
          const val = compileExpr(a.value);
          if (a.name.startsWith("$")) {
            const t = svTarget(name);
            result = `(() => { const __save_${name}: any = ${t}; ${t} = ${val}; try { return ${result}; } finally { ${t} = __save_${name}; } })()`;
          } else {
            result = `((${name}) => (${result}))(${val})`;
          }
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

  const sig = signatures.get(name);
  const dollarArgs = expr.args.filter(a =>
    a.name && a.name.startsWith("$") && !(sig && sig.params.includes(a.name)));
  const positionalArgs = dollarArgs.length === 0 ? expr.args : expr.args.filter(a => !dollarArgs.includes(a));

  const argList = compileArgList(name, positionalArgs);
  const call = `${name}(${argList})`;
  if (dollarArgs.length === 0) {
    return call;
  }

  const saves: string[] = [];
  const decls: string[] = [];
  const restores: string[] = [];
  for (const arg of dollarArgs) {
    const dn = escapeName(arg.name!);
    const t = svTarget(dn);
    saves.push(`let __save_${dn}: any = ${t};`);
    decls.push(`${t} = ${compileExpr(arg.value)};`);
    restores.push(`${t} = __save_${dn};`);
  }
  return `(() => { ${saves.join(" ")} ${decls.join(" ")} try { return ${call}; } finally { ${restores.join(" ")} } })()`;
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
          result = `(() => { const __r = []; const __start: any = ${start}, __step: any = ${step}, __end: any = ${end}; const __cnt: any = __rangeCount(__start, __step, __end); for (let __i = 0; __i < __cnt; __i++) { const ${vName}: any = __start + __i * __step; __r.push(...(${result})); } return __r; })()`;
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
      return `(__truthy(${cond}) ? ${ifTrue} : ${ifFalse})`;
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
      return `(() => { const __r = []; for (let ${inits}; __truthy(${cond}); ${updates}) __r.push(...(${inner})); return __r; })()`;
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
