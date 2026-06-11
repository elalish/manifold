import fs from "fs";
import path from "path";
import { describe, expect, test } from "vitest";
import { Lexer } from "../core/lexer.js";
import { Parser } from "../core/parser.js";
import type { SourceRange } from "../core/lexer.js";
import type {
  Program,
  Statement,
  Expr,
  Argument,
  Parameter,
  ForVariable,
  LetAssignment,
  ListCompGenerator,
} from "../core/ast.js";


interface Violation {
  nodeKind: string;
  problem: string;
  loc: SourceRange;
  sliced?: string;
}

function offsetToLineCol(source: string, offset: number): { line: number; column: number } {
  let line = 1;
  let column = 1;
  for (let i = 0; i < offset && i < source.length; i++) {
    if (source[i] === "\n") { line++; column = 1; }
    else { column++; }
  }
  return { line, column };
}

function checkRange(source: string, range: SourceRange, nodeKind: string, parentRange?: SourceRange): Violation[] {
  const violations: Violation[] = [];
  const { start, end } = range;

  if (start.offset < 0 || end.offset < 0) {
    violations.push({ nodeKind, problem: "negative offset", loc: range });
  }
  if (start.offset > end.offset) {
    violations.push({ nodeKind, problem: `start.offset (${start.offset}) > end.offset (${end.offset})`, loc: range });
  }

  const sliced = source.slice(start.offset, end.offset);
  if (sliced.length === 0 && start.offset !== end.offset) {
    violations.push({ nodeKind, problem: "empty slice despite non-zero range", loc: range, sliced });
  }
  
  if (start.offset > source.length) {
    violations.push({ nodeKind, problem: `start.offset (${start.offset}) past source length (${source.length})`, loc: range });
  }
  if (end.offset > source.length) {
    violations.push({ nodeKind, problem: `end.offset (${end.offset}) past source length (${source.length})`, loc: range });
  }

  const computedStart = offsetToLineCol(source, start.offset);
  if (computedStart.line !== start.line) {
    violations.push({ nodeKind, problem: `start.line is ${start.line} but offset ${start.offset} is on line ${computedStart.line}`, loc: range });
  }
  if (computedStart.column !== start.column) {
    violations.push({ nodeKind, problem: `start.column is ${start.column} but offset ${start.offset} is column ${computedStart.column}`, loc: range });
  }

  const computedEnd = offsetToLineCol(source, end.offset);
  if (computedEnd.line !== end.line) {
    violations.push({ nodeKind, problem: `end.line is ${end.line} but offset ${end.offset} is on line ${computedEnd.line}`, loc: range });
  }
  if (computedEnd.column !== end.column) {
    violations.push({ nodeKind, problem: `end.column is ${end.column} but offset ${end.offset} is column ${computedEnd.column}`, loc: range });
  }

  if (parentRange) {
    if (start.offset < parentRange.start.offset) {
      violations.push({ nodeKind, problem: `start.offset (${start.offset}) is before parent start (${parentRange.start.offset})`, loc: range });
    }
    if (end.offset > parentRange.end.offset) {
      violations.push({ nodeKind, problem: `end.offset (${end.offset}) is after parent end (${parentRange.end.offset})`, loc: range });
    }
  }

  return violations;
}

function walkExpr(source: string, expr: Expr, violations: Violation[], parentRange?: SourceRange): void {
  const range = expr.loc;
  if (range) violations.push(...checkRange(source, range, expr.kind, parentRange));
  const r = range ?? parentRange;

  switch (expr.kind) {
    case "number":
    case "string":
    case "boolean":
    case "undef":
    case "identifier":
      break;

    case "vector":
      for (const el of expr.elements) walkExpr(source, el, violations, r);
      break;

    case "range":
      walkExpr(source, expr.start, violations, r);
      if (expr.step) walkExpr(source, expr.step, violations, r);
      walkExpr(source, expr.end, violations, r);
      break;

    case "binary":
      walkExpr(source, expr.left, violations, r);
      walkExpr(source, expr.right, violations, r);
      break;

    case "unary":
      walkExpr(source, expr.operand, violations, r);
      break;

    case "ternary":
      walkExpr(source, expr.condition, violations, r);
      walkExpr(source, expr.ifTrue, violations, r);
      walkExpr(source, expr.ifFalse, violations, r);
      break;

    case "call":
      for (const arg of expr.args) walkArgument(source, arg, violations, r);
      break;

    case "dynCall":
      walkExpr(source, expr.callee, violations, r);
      for (const arg of expr.args) walkArgument(source, arg, violations, r);
      break;

    case "index":
      walkExpr(source, expr.object, violations, r);
      walkExpr(source, expr.index, violations, r);
      break;

    case "member":
      walkExpr(source, expr.object, violations, r);
      break;

    case "group":
      walkExpr(source, expr.expr, violations, r);
      break;

    case "echo":
      for (const arg of expr.args) walkArgument(source, arg, violations, r);
      walkExpr(source, expr.expr, violations, r);
      break;

    case "let":
      for (const a of expr.assignments) walkLetAssignment(source, a, violations, r);
      walkExpr(source, expr.body, violations, r);
      break;

    case "assert":
      for (const arg of expr.args) walkArgument(source, arg, violations, r);
      walkExpr(source, expr.expr, violations, r);
      break;

    case "listComp":
      walkListCompGenerator(source, expr.generator, violations, r);
      break;

    case "each":
      walkExpr(source, expr.expr, violations, r);
      break;

    case "lambda":
      for (const p of expr.params) walkParameter(source, p, violations, r);
      walkExpr(source, expr.body, violations, r);
      break;

    default: {
      const _never: never = expr;
      violations.push({ nodeKind: "unknown-expr", problem: `unhandled expr kind: ${(_never as any).kind}`, loc: (expr as any).loc });
    }
  }
}

function walkArgument(source: string, arg: Argument, violations: Violation[], parentRange?: SourceRange): void {
  if (arg.loc) violations.push(...checkRange(source, arg.loc, "argument", parentRange));
  const r = arg.loc ?? parentRange;
  walkExpr(source, arg.value, violations, r);
}

function walkParameter(source: string, param: Parameter, violations: Violation[], parentRange?: SourceRange): void {
  if (param.loc) violations.push(...checkRange(source, param.loc, "parameter", parentRange));
  const r = param.loc ?? parentRange;
  if (param.defaultValue) walkExpr(source, param.defaultValue, violations, r);
}

function walkLetAssignment(source: string, la: LetAssignment, violations: Violation[], parentRange?: SourceRange): void {
  if (la.loc) violations.push(...checkRange(source, la.loc, "letAssignment", parentRange));
  const r = la.loc ?? parentRange;
  walkExpr(source, la.value, violations, r);
}

function walkForVariable(source: string, fv: ForVariable, violations: Violation[], parentRange?: SourceRange): void {
  if (fv.loc) violations.push(...checkRange(source, fv.loc, "forVariable", parentRange));
  const r = fv.loc ?? parentRange;
  walkExpr(source, fv.range, violations, r);
}

function walkListCompGenerator(source: string, gen: ListCompGenerator, violations: Violation[], parentRange?: SourceRange): void {
  if (gen.loc) violations.push(...checkRange(source, gen.loc, gen.kind, parentRange));
  const r = gen.loc ?? parentRange;

  switch (gen.kind) {
    case "lcFor":
      for (const v of gen.variables) walkForVariable(source, v, violations, r);
      walkListCompGenerator(source, gen.body, violations, r);
      break;
    case "lcCFor":
      for (const a of gen.inits) walkLetAssignment(source, a, violations, r);
      walkExpr(source, gen.condition, violations, r);
      for (const a of gen.updates) walkLetAssignment(source, a, violations, r);
      walkListCompGenerator(source, gen.body, violations, r);
      break;
    case "lcIf":
      walkExpr(source, gen.condition, violations, r);
      walkListCompGenerator(source, gen.ifTrue, violations, r);
      if (gen.ifFalse) walkListCompGenerator(source, gen.ifFalse, violations, r);
      break;
    case "lcLet":
      for (const a of gen.assignments) walkLetAssignment(source, a, violations, r);
      walkListCompGenerator(source, gen.body, violations, r);
      break;
    case "lcExpr":
      walkExpr(source, gen.expr, violations, r);
      break;
  }
}

function walkStatement(source: string, stmt: Statement, violations: Violation[], parentRange?: SourceRange): void {
  if (stmt.loc) violations.push(...checkRange(source, stmt.loc, stmt.kind, parentRange));
  const r = stmt.loc ?? parentRange;

  switch (stmt.kind) {
    case "variableDecl":
      walkExpr(source, stmt.value, violations, r);
      break;

    case "moduleCall":
      for (const arg of stmt.args) walkArgument(source, arg, violations, r);
      if (stmt.child) walkStatement(source, stmt.child, violations, r);
      break;

    case "block":
      for (const s of stmt.statements) walkStatement(source, s, violations, r);
      break;

    case "moduleDecl":
      for (const p of stmt.params) walkParameter(source, p, violations, r);
      walkStatement(source, stmt.body, violations, r);
      break;

    case "functionDecl":
      for (const p of stmt.params) walkParameter(source, p, violations, r);
      walkExpr(source, stmt.body, violations, r);
      break;

    case "for":
      for (const v of stmt.variables) walkForVariable(source, v, violations, r);
      walkStatement(source, stmt.body, violations, r);
      break;

    case "if":
      walkExpr(source, stmt.condition, violations, r);
      walkStatement(source, stmt.thenBody, violations, r);
      if (stmt.elseBody) walkStatement(source, stmt.elseBody, violations, r);
      break;

    case "empty":
    case "use":
    case "include":
      break; // leaf / no children

    default: {
      const _never: never = stmt;
      violations.push({ nodeKind: "unknown-stmt", problem: `unhandled stmt kind: ${(_never as any).kind}`, loc: (stmt as any).loc });
    }
  }
}

function walkProgram(source: string, program: Program): Violation[] {
  const violations: Violation[] = [];
  if (program.loc) violations.push(...checkRange(source, program.loc, "program"));
  for (const stmt of program.statements) {
    walkStatement(source, stmt, violations, program.loc);
  }
  return violations;
}

function formatViolation(source: string, v: Violation, filename: string): string {
  const snippet = source.slice(
    Math.max(0, v.loc.start.offset - 20),
    Math.min(source.length, v.loc.end.offset + 20),
  ).replace(/\n/g, "↵");
  return (
    `[${v.nodeKind}] ${v.problem}\n` +
    `  at ${filename}:${v.loc.start.line}:${v.loc.start.column} ` +
    `(offset ${v.loc.start.offset}–${v.loc.end.offset})\n` +
    `  context: "…${snippet}…"`
  );
}

function getAllFiles(dir: string): string[] {
    let results: string[] = [];

    const items = fs.readdirSync(dir, {
      withFileTypes: true
    });

    for (const item of items) {
      const fullPath = path.join(dir, item.name);

      if (item.isDirectory()) {
        results = results.concat(getAllFiles(fullPath));
      } else {
        results.push(fullPath);
      }
    }

    return results;
}

describe("AST source locations", () => {
  const openscadFiles = getAllFiles("./examples");

  for (const file of openscadFiles) {
    if (file.endsWith(".scad")) {
      test(`Test for ${file}`, () => {
        const source = fs.readFileSync(file, "utf8");

        // Parse
        const lexer = new Lexer(source, file);
        const parser = new Parser(lexer);
        const program = parser.parseProgram();

        // Walk and collect all violations
        const violations = walkProgram(source, program);

        if (violations.length > 0) {
          const messages = violations
            .map(v => formatViolation(source, v, path.relative(process.cwd(), file)))
            .join("\n\n");
          expect.fail(
            `${violations.length} source-location violation(s) in ${path.relative(process.cwd(), file)}:\n\n${messages}`
          );
        }
      });
    }
  }
});