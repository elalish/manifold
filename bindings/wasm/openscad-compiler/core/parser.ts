import { Lexer, TokenType, fmtLoc, tokenTypeToString } from "./lexer.js";
import type { Token, SourceLocation, SourceRange } from "./lexer.js";
import type {
  Expr, Statement, Program, Argument, Parameter, ForVariable,
  ModuleCallStmt, BlockStmt, ListCompGenerator, LetAssignment,
  Comment, ASTNode,
} from "./ast.js";

export class Parser {
  private tokens: Token[] = [];
  private pos = 0;
  private current!: Token;
  private prev!: Token;
  private filename: string;
  private comments: Comment[] = [];
  readonly warnings: string[] = [];

  constructor(lexer: Lexer) {
    this.filename = lexer.filename;

    // Pre-tokenize for easy lookahead
    let tok: Token;
    do {
      tok = lexer.nextToken();
      if (tok.type === TokenType.LineComment || tok.type === TokenType.BlockComment) {
        this.comments.push({
          kind: tok.type === TokenType.LineComment ? "line" : "block",
          value: tok.value ?? "",
          loc: tok.range,
        });
      } else {
        this.tokens.push(tok);
      }
    } while (tok.type !== TokenType.EOF);
    this.current = this.tokens[0]!;
    this.prev = this.current;
  }

  // Position helpers
  private startLoc(): SourceLocation {
    return this.current.range.start;
  }

  private rangeSince(start: SourceLocation): SourceRange {
    return { start, end: this.prev.range.end };
  }

  // Token helpers
  private advance(): Token {
    const tok = this.current;
    this.prev = tok;
    this.pos++;
    this.current = this.tokens[this.pos] ?? this.tokens[this.tokens.length - 1]!;
    return tok;
  }

  private expect(type: TokenType): Token {
    if (this.current.type !== type) {
      throw new Error(
        `Expected ${tokenTypeToString(type)} but got ${tokenTypeToString(this.current.type)} at ${fmtLoc(this.current.range.start, this.filename)}`
      );
    }
    return this.advance();
  }

  private match(type: TokenType): boolean {
    if (this.current.type === type) {
      this.advance();
      return true;
    }
    return false;
  }

  private peekNext(): Token {
    return this.tokens[this.pos + 1] ?? this.tokens[this.tokens.length - 1]!;
  }

  private isIdentifier(name?: string): boolean {
    return (
      this.current.type === TokenType.Identifier &&
      (name === undefined || this.current.value === name)
    );
  }

  // Program
  parseProgram(): Program {
    const start = this.startLoc();
    const statements: Statement[] = [];
    while (this.current.type !== TokenType.EOF) {
      const posBefore = this.pos;
      try {
        const stmt = this.parseStatement();
        stmt.filename = this.filename;
        statements.push(stmt);
      } catch (err) {
        // Recover instead of aborting the whole file: record the error, skip past the broken statement, and keep emitting the surrounding code.
        this.warnings.push((err as Error).message);
        this.synchronize();
        if (this.pos === posBefore) this.advance();
      }
    }
    const program: Program = { kind: "program", statements, loc: this.rangeSince(start) };
    program.filename = this.filename;
    attachComments(program, this.comments);
    return program;
  }

  private synchronize(): void {
    let depth = 0;
    while (this.current.type !== TokenType.EOF) {
      const t = this.current.type;
      if (t === TokenType.LBrace || t === TokenType.LParen || t === TokenType.LBracket) {
        depth++;
        this.advance();
        continue;
      }
      if (t === TokenType.RBrace) {
        if (depth <= 0) return;
        depth--;
        this.advance();
        continue;
      }
      if (t === TokenType.RParen || t === TokenType.RBracket) {
        if (depth > 0) depth--;
        this.advance();
        continue;
      }
      if (t === TokenType.Semicolon && depth <= 0) {
        this.advance();
        return;
      }
      this.advance();
    }
  }

  // Statements
  private parseStatement(): Statement {
    let modifier: string | undefined;
    while (
      this.current.type === TokenType.Hash ||
      this.current.type === TokenType.Bang ||
      this.current.type === TokenType.Star ||
      this.current.type === TokenType.Percent
    ) {
      const ch =
        this.current.type === TokenType.Hash ? "#"
        : this.current.type === TokenType.Bang ? "!"
        : this.current.type === TokenType.Star ? "*"
        : "%";
      modifier = (modifier ?? "") + ch;
      this.advance();
    }

    const stmt = this.parseStatementInner();
    if (modifier !== undefined) {
      switch (stmt.kind) {
        case "moduleCall":
        case "block":
        case "for":
        case "if":
          stmt.modifier = modifier;
          break;
        default:
          break;
      }
    }
    return stmt;
  }

  private parseStatementInner(): Statement {
    if (this.isIdentifier("module")) return this.parseModuleDecl();
    if (this.isIdentifier("function")) return this.parseFunctionDecl();
    if (this.isIdentifier("use") || this.isIdentifier("include")) return this.parseUseInclude();
    if (this.isIdentifier("for")) return this.parseForStmt();
    if (this.isIdentifier("let") && this.peekNext().type === TokenType.LParen) return this.parseLetStmt();
    if (this.isIdentifier("if")) return this.parseIfStmt();
    if (this.current.type === TokenType.LBrace) return this.parseBlock();

    if (this.current.type === TokenType.Semicolon) {
      const start = this.startLoc();
      this.advance();
      return { kind: "empty", loc: this.rangeSince(start) };
    }

    if (
      this.current.type === TokenType.Identifier &&
      this.peekNext().type === TokenType.Equals
    ) {
      return this.parseVariableDecl();
    }

    return this.parseModuleCallStmt();
  }

  private parseModuleDecl(): Statement {
    const start = this.startLoc();
    this.advance(); // consume 'module'
    const name = this.expect(TokenType.Identifier).value!;
    this.expect(TokenType.LParen);
    const params = this.parseParameterList();
    this.expect(TokenType.RParen);
    const body = this.parseStatement();
    return { kind: "moduleDecl", name, params, body, loc: this.rangeSince(start) };
  }

  private parseFunctionDecl(): Statement {
    const start = this.startLoc();
    this.advance(); // consume 'function'
    const name = this.expect(TokenType.Identifier).value!;
    this.expect(TokenType.LParen);
    const params = this.parseParameterList();
    this.expect(TokenType.RParen);
    this.expect(TokenType.Equals);
    const body = this.parseExpr();
    this.expect(TokenType.Semicolon);
    return { kind: "functionDecl", name, params, body, loc: this.rangeSince(start) };
  }

  private parseForStmt(): Statement {
    const start = this.startLoc();
    this.advance(); // consume 'for'
    this.expect(TokenType.LParen);

    const variables: ForVariable[] = [];
    do {
      if (this.current.type === TokenType.RParen) break;
      const vs = this.startLoc();
      const name = this.expect(TokenType.Identifier).value!;
      this.expect(TokenType.Equals);
      const range = this.parseExpr();
      variables.push({ name, range, loc: this.rangeSince(vs) });
    } while (this.match(TokenType.Comma));

    this.expect(TokenType.RParen);
    const body = this.parseStatement();
    return { kind: "for", variables, body, loc: this.rangeSince(start) };
  }

  private parseIfStmt(): Statement {
    const start = this.startLoc();
    this.advance();
    this.expect(TokenType.LParen);
    const condition = this.parseExpr();
    this.expect(TokenType.RParen);
    const thenBody = this.parseStatement();

    let elseBody: Statement | undefined;
    if (this.isIdentifier("else")) {
      this.advance();
      elseBody = this.parseStatement();
    }

    return { kind: "if", condition, thenBody, elseBody, loc: this.rangeSince(start) };
  }

  private parseBlock(): BlockStmt {
    const start = this.startLoc();
    this.expect(TokenType.LBrace);
    const statements: Statement[] = [];
    while (
      this.current.type !== TokenType.RBrace &&
      this.current.type !== TokenType.EOF
    ) {
      statements.push(this.parseStatement());
    }
    this.expect(TokenType.RBrace);
    return { kind: "block", statements, loc: this.rangeSince(start) };
  }

  private parseVariableDecl(): Statement {
    const start = this.startLoc();
    const name = this.expect(TokenType.Identifier).value!;
    this.expect(TokenType.Equals);
    const value = this.parseExpr();
    this.expect(TokenType.Semicolon);
    return { kind: "variableDecl", name, value, loc: this.rangeSince(start) };
  }

  private parseModuleCallStmt(): ModuleCallStmt {
    const start = this.startLoc();
    const name = this.expect(TokenType.Identifier).value!;
    this.expect(TokenType.LParen);
    const args = this.parseArgumentList();
    this.expect(TokenType.RParen);

    let child: Statement | undefined;
    if (this.current.type === TokenType.Semicolon) {
      this.advance();
    } else if (
      this.current.type === TokenType.EOF ||
      this.current.type === TokenType.RBrace
    ) {
      throw new Error(
        `Expected ';' or child statement after module call '${name}' at ${fmtLoc(this.current.range.start, this.filename)}`
      );
    } else {
      child = this.parseStatement();
    }

    return { kind: "moduleCall", name, args, child, loc: this.rangeSince(start) };
  }

  // Argument / Parameter lists
  private parseArgumentList(): Argument[] {
    const args: Argument[] = [];
    if ((this.current as Token).type === TokenType.RParen) return args;

    do {
      if ((this.current as Token).type === TokenType.RParen) break;
      const as = this.startLoc();
      if (
        this.current.type === TokenType.Identifier &&
        this.peekNext().type === TokenType.Equals
      ) {
        const name = this.advance().value!;
        this.advance(); // consume '='
        const value = this.parseExpr();
        args.push({ name, value, loc: this.rangeSince(as) });
      } else {
        const value = this.parseExpr();
        args.push({ value, loc: this.rangeSince(as) });
      }
    } while (this.match(TokenType.Comma));

    return args;
  }

  private parseParameterList(): Parameter[] {
    const params: Parameter[] = [];
    if ((this.current as Token).type === TokenType.RParen) return params;

    do {
      if ((this.current as Token).type === TokenType.RParen) break;
      const ps = this.startLoc();
      const name = this.expect(TokenType.Identifier).value!;
      let defaultValue: Expr | undefined;
      if (this.match(TokenType.Equals)) {
        defaultValue = this.parseExpr();
      }
      params.push({ name, defaultValue, loc: this.rangeSince(ps) });
    } while (this.match(TokenType.Comma));

    return params;
  }

  // Expressions 
  parseExpr(): Expr {
    return this.parseTernary();
  }

  private parseTernary(): Expr {
    const start = this.startLoc();
    let expr = this.parseOr();
    if (this.match(TokenType.Question)) {
      const ifTrue = this.parseExpr();
      this.expect(TokenType.Colon);
      const ifFalse = this.parseExpr();
      return { kind: "ternary", condition: expr, ifTrue, ifFalse, loc: this.rangeSince(start) };
    }
    return expr;
  }

  private parseOr(): Expr {
    let left = this.parseAnd();
    while (this.match(TokenType.Or)) {
      const right = this.parseAnd();
      left = { kind: "binary", op: "||", left, right, loc: { start: left.loc!.start, end: right.loc!.end } };
    }
    return left;
  }

  private parseAnd(): Expr {
    let left = this.parseEquality();
    while (this.match(TokenType.And)) {
      const right = this.parseEquality();
      left = { kind: "binary", op: "&&", left, right, loc: { start: left.loc!.start, end: right.loc!.end } };
    }
    return left;
  }

  private parseEquality(): Expr {
    let left = this.parseRelational();
    while (this.current.type === TokenType.EqEq || this.current.type === TokenType.BangEq) {
      const op = this.current.type === TokenType.EqEq ? "==" : "!=";
      this.advance();
      const right = this.parseRelational();
      left = { kind: "binary", op, left, right, loc: { start: left.loc!.start, end: right.loc!.end } };
    }
    return left;
  }

  private parseRelational(): Expr {
    let left = this.parseBitOr();
    const opMap: Partial<Record<TokenType, string>> = {
      [TokenType.Lt]: "<",
      [TokenType.Gt]: ">",
      [TokenType.LtEq]: "<=",
      [TokenType.GtEq]: ">=",
    };
    let op = opMap[this.current.type];
    while (op) {
      this.advance();
      const right = this.parseBitOr();
      left = { kind: "binary", op, left, right, loc: { start: left.loc!.start, end: right.loc!.end } };
      op = opMap[this.current.type];
    }
    return left;
  }

  private parseBitOr(): Expr {
    let left = this.parseBitAnd();
    while (this.current.type === TokenType.Pipe) {
      this.advance();
      const right = this.parseBitAnd();
      left = { kind: "binary", op: "|", left, right, loc: { start: left.loc!.start, end: right.loc!.end } };
    }
    return left;
  }

  private parseBitAnd(): Expr {
    let left = this.parseShift();
    while (this.current.type === TokenType.Amp) {
      this.advance();
      const right = this.parseShift();
      left = { kind: "binary", op: "&", left, right, loc: { start: left.loc!.start, end: right.loc!.end } };
    }
    return left;
  }

  private parseShift(): Expr {
    let left = this.parseAddition();
    while (this.current.type === TokenType.Shl || this.current.type === TokenType.Shr) {
      const op = this.current.type === TokenType.Shl ? "<<" : ">>";
      this.advance();
      const right = this.parseAddition();
      left = { kind: "binary", op, left, right, loc: { start: left.loc!.start, end: right.loc!.end } };
    }
    return left;
  }

  private parseAddition(): Expr {
    let left = this.parseMultiplication();
    while (
      this.current.type === TokenType.Plus ||
      this.current.type === TokenType.Minus
    ) {
      const op = this.current.type === TokenType.Plus ? "+" : "-";
      this.advance();
      const right = this.parseMultiplication();
      left = { kind: "binary", op, left, right, loc: { start: left.loc!.start, end: right.loc!.end } };
    }
    return left;
  }

  private parseMultiplication(): Expr {
    let left = this.parseExponentiation();
    while (
      this.current.type === TokenType.Star ||
      this.current.type === TokenType.Slash ||
      this.current.type === TokenType.Percent
    ) {
      const op =
        this.current.type === TokenType.Star ? "*" :
          this.current.type === TokenType.Slash ? "/" : "%";
      this.advance();
      const right = this.parseExponentiation();
      left = { kind: "binary", op, left, right, loc: { start: left.loc!.start, end: right.loc!.end } };
    }
    return left;
  }

  private parseExponentiation(): Expr {
    const start = this.startLoc();
    let left = this.parseUnary();
    if (this.current.type === TokenType.Caret) {
      this.advance();
      const right = this.parseExponentiation(); // right-associative
      left = { kind: "binary", op: "^", left, right, loc: this.rangeSince(start) };
    }
    return left;
  }

  private parseUnary(): Expr {
    const start = this.startLoc();
    if (this.current.type === TokenType.Minus) {
      this.advance();
      const operand = this.parseUnary();
      return { kind: "unary", op: "-", operand, loc: this.rangeSince(start) };
    }
    if (this.current.type === TokenType.Plus) {
      this.advance();
      const operand = this.parseUnary();
      return { kind: "unary", op: "+", operand, loc: this.rangeSince(start) };
    }
    if (this.current.type === TokenType.Bang) {
      this.advance();
      const operand = this.parseUnary();
      return { kind: "unary", op: "!", operand, loc: this.rangeSince(start) };
    }
    if (this.current.type === TokenType.Tilde) {
      this.advance();
      const operand = this.parseUnary();
      return { kind: "unary", op: "~", operand, loc: this.rangeSince(start) };
    }
    return this.parsePostfix();
  }

  private parsePostfix(): Expr {
    let expr = this.parsePrimary();

    while (true) {
      if (this.current.type === TokenType.LBracket) {
        this.advance();
        const index = this.parseExpr();
        this.expect(TokenType.RBracket);
        expr = { kind: "index", object: expr, index, loc: { start: expr.loc!.start, end: this.prev.range.end } };
      } else if (this.current.type === TokenType.Dot) {
        this.advance();
        const property = this.expect(TokenType.Identifier).value!;
        expr = { kind: "member", object: expr, property, loc: { start: expr.loc!.start, end: this.prev.range.end } };
      } else if (this.current.type === TokenType.LParen && expr.kind !== "number" && expr.kind !== "string" && expr.kind !== "boolean" && expr.kind !== "undef") {
        this.advance();
        const args = this.parseArgumentList();
        this.expect(TokenType.RParen);
        if (expr.kind === "identifier") {
          expr = { kind: "call", name: expr.name, args, loc: { start: expr.loc!.start, end: this.prev.range.end } };
        } else {
          expr = { kind: "dynCall", callee: expr, args, loc: { start: expr.loc!.start, end: this.prev.range.end } };
        }
      } else {
        break;
      }
    }

    return expr;
  }

  private parsePrimary(): Expr {
    const tok = this.current;
    const start = this.startLoc();

    // Number
    if (tok.type === TokenType.Number) {
      this.advance();
      return { kind: "number", value: Number(tok.value), loc: this.rangeSince(start) };
    }

    // String
    if (tok.type === TokenType.String) {
      this.advance();
      return { kind: "string", value: tok.value!, loc: this.rangeSince(start) };
    }

    // Booleans & undef
    if (tok.type === TokenType.Identifier) {
      if (tok.value === "true") { this.advance(); return { kind: "boolean", value: true, loc: this.rangeSince(start) }; }
      if (tok.value === "false") { this.advance(); return { kind: "boolean", value: false, loc: this.rangeSince(start) }; }
      if (tok.value === "undef") { this.advance(); return { kind: "undef", loc: this.rangeSince(start) }; }

      // Anonymous function: function(params) expr
      if (tok.value === "function" && this.peekNext().type === TokenType.LParen) {
        this.advance(); // consume 'function'
        this.advance(); // consume '('
        const params = this.parseParameterList();
        this.expect(TokenType.RParen);
        const body = this.parseExpr();
        return { kind: "lambda", params, body, loc: this.rangeSince(start) };
      }

      // Echo expression modifier: echo(args) [expr]
      if (tok.value === "echo" && this.peekNext().type === TokenType.LParen) {
        this.advance(); // consume 'echo'
        this.advance(); // consume '('
        const args = this.parseArgumentList();
        this.expect(TokenType.RParen);
        const expr = this.canStartExpr() ? this.parseExpr() : { kind: "undef" } as const;
        return { kind: "echo", args, expr, loc: this.rangeSince(start) };
      }

      // Let expression: let(assignments) expr
      if (tok.value === "let" && this.peekNext().type === TokenType.LParen) {
        this.advance(); // consume 'let'
        this.advance(); // consume '('
        const assignments = this.parseLetAssignments();
        this.expect(TokenType.RParen);
        const body = this.canStartExpr() ? this.parseExpr() : { kind: "undef" } as const;
        return { kind: "let", assignments, body, loc: this.rangeSince(start) };
      }

      // Assert expression modifier: assert(args) [expr]
      if (tok.value === "assert" && this.peekNext().type === TokenType.LParen) {
        this.advance(); // consume 'assert'
        this.advance(); // consume '('
        const args = this.parseArgumentList();
        this.expect(TokenType.RParen);
        const expr = this.canStartExpr() ? this.parseExpr() : { kind: "undef" } as const;
        return { kind: "assert", args, expr, loc: this.rangeSince(start) };
      }

      // Each expression: each expr
      if (tok.value === "each") {
        this.advance(); // consume 'each'
        return { kind: "each", expr: this.parseExpr(), loc: this.rangeSince(start) };
      }

      if (this.peekNext().type === TokenType.LParen) {
        const name = this.advance().value!;
        this.advance();
        const args = this.parseArgumentList();
        this.expect(TokenType.RParen);
        return { kind: "call", name, args, loc: this.rangeSince(start) };
      }

      // Plain identifier
      this.advance();
      return { kind: "identifier", name: tok.value!, loc: this.rangeSince(start) };
    }

    // Vector or range
    if (tok.type === TokenType.LBracket) {
      return this.parseVectorOrRange();
    }

    // Grouped expression
    if (tok.type === TokenType.LParen) {
      this.advance();
      const expr = this.parseExpr();
      this.expect(TokenType.RParen);
      return { kind: "group", expr, loc: this.rangeSince(start) };
    }

    throw new Error(
      `Unexpected token ${tokenTypeToString(tok.type)}` +
      (tok.value ? ` '${tok.value}'` : "") +
      ` at ${fmtLoc(tok.range.start, this.filename)}`
    );
  }

  // Vector / Range
  private parseVectorOrRange(): Expr {
    const start = this.startLoc();
    this.expect(TokenType.LBracket);

    // empty vector
    if (this.current.type === TokenType.RBracket) {
      this.advance();
      return { kind: "vector", elements: [], loc: this.rangeSince(start) };
    }

    const first = this.parseVectorElement();

    // Range  [start : end] or [start : step : end]
    if (first.kind !== "listComp" && first.kind !== "each" && this.current.type === TokenType.Colon) {
      this.advance();
      const second = this.parseExpr();
      if (this.current.type === TokenType.Colon) {
        this.advance();
        const third = this.parseExpr();
        this.expect(TokenType.RBracket);
        return { kind: "range", start: first, step: second, end: third, loc: this.rangeSince(start) };
      }
      this.expect(TokenType.RBracket);
      return { kind: "range", start: first, end: second, loc: this.rangeSince(start) };
    }

    // Vector
    const elements: Expr[] = [first];
    while (this.match(TokenType.Comma)) {
      if ((this.current as Token).type === TokenType.RBracket) break;
      elements.push(this.parseVectorElement());
    }
    this.expect(TokenType.RBracket);
    return { kind: "vector", elements, loc: this.rangeSince(start) };
  }

  private parseVectorElement(): Expr {
    const start = this.startLoc();
    if (this.isIdentifier("each")) {
      this.advance();
      if (this.isIdentifier("for") || this.isIdentifier("if") || this.isIdentifier("let")) {
        const generator = this.parseListCompGenerator();
        return { kind: "each", expr: { kind: "listComp", generator, loc: this.rangeSince(start) }, loc: this.rangeSince(start) };
      }
      return { kind: "each", expr: this.parseExpr(), loc: this.rangeSince(start) };
    }
    if (this.isIdentifier("for") || this.isIdentifier("if") || this.isIdentifier("let")) {
      const generator = this.parseListCompGenerator();
      return { kind: "listComp", generator, loc: this.rangeSince(start) };
    }
    return this.parseExpr();
  }

  // List comprehension generators
  private parseListCompGenerator(): ListCompGenerator {
    const start = this.startLoc();

    if (this.isIdentifier("for")) {
      this.advance();
      this.expect(TokenType.LParen);
      const variables: ForVariable[] = [];
      if ((this.current as Token).type !== TokenType.Semicolon && (this.current as Token).type !== TokenType.RParen) {
        do {
          if ((this.current as Token).type === TokenType.Semicolon || (this.current as Token).type === TokenType.RParen) break;
          const vs = this.startLoc();
          const name = this.expect(TokenType.Identifier).value!;
          this.expect(TokenType.Equals);
          const range = this.parseExpr();
          variables.push({ name, range, loc: this.rangeSince(vs) });
        } while (this.match(TokenType.Comma));
      }

      // C-style for: for(init ; condition ; update)
      if (this.current.type === TokenType.Semicolon) {
        this.advance();
        const inits = variables.map(v => ({ name: v.name, value: v.range, loc: v.loc }));
        const condition = this.parseExpr();
        this.expect(TokenType.Semicolon);
        const updates: LetAssignment[] = [];
        if ((this.current as Token).type !== TokenType.RParen) {
          do {
            if ((this.current as Token).type === TokenType.RParen) break;
            const us = this.startLoc();
            const name = this.expect(TokenType.Identifier).value!;
            this.expect(TokenType.Equals);
            const value = this.parseExpr();
            updates.push({ name, value, loc: this.rangeSince(us) });
          } while (this.match(TokenType.Comma));
        }
        this.expect(TokenType.RParen);
        const body = this.parseListCompGenerator();
        return { kind: "lcCFor", inits, condition, updates, body, loc: this.rangeSince(start) };
      }

      this.expect(TokenType.RParen);
      const body = this.parseListCompGenerator();
      return { kind: "lcFor", variables, body, loc: this.rangeSince(start) };
    }

    if (this.isIdentifier("if")) {
      this.advance();
      this.expect(TokenType.LParen);
      const condition = this.parseExpr();
      this.expect(TokenType.RParen);
      const ifTrue = this.parseListCompGenerator();
      let ifFalse: ListCompGenerator | undefined;
      if (this.isIdentifier("else")) {
        this.advance();
        ifFalse = this.parseListCompGenerator();
      }
      return { kind: "lcIf", condition, ifTrue, ifFalse, loc: this.rangeSince(start) };
    }

    if (this.isIdentifier("let")) {
      this.advance();
      this.expect(TokenType.LParen);
      const assignments = this.parseLetAssignments();
      this.expect(TokenType.RParen);
      const body = this.parseListCompGenerator();
      return { kind: "lcLet", assignments, body, loc: this.rangeSince(start) };
    }

    if (this.isIdentifier("each")) {
      this.advance();
      if (this.isIdentifier("for") || this.isIdentifier("if") || this.isIdentifier("let")) {
        const generator = this.parseListCompGenerator();
        return { kind: "lcExpr", expr: { kind: "each", expr: { kind: "listComp", generator, loc: this.rangeSince(start) }, loc: this.rangeSince(start) }, loc: this.rangeSince(start) };
      }
      return { kind: "lcExpr", expr: { kind: "each", expr: this.parseExpr(), loc: this.rangeSince(start) }, loc: this.rangeSince(start) };
    }

    const expr = this.parseExpr();
    return { kind: "lcExpr", expr, loc: this.rangeSince(start) };
  }

  // Let assignments
  private parseLetAssignments(): LetAssignment[] {
    const assignments: LetAssignment[] = [];
    if ((this.current as Token).type === TokenType.RParen) return assignments;

    do {
      if ((this.current as Token).type === TokenType.RParen) break;
      const as = this.startLoc();
      const name = this.expect(TokenType.Identifier).value!;
      this.expect(TokenType.Equals);
      const value = this.parseExpr();
      assignments.push({ name, value, loc: this.rangeSince(as) });
    } while (this.match(TokenType.Comma));

    return assignments;
  }

  // Use / Include
  private parseUseInclude(): Statement {
    const start = this.startLoc();
    const keyword = this.advance().value! as "use" | "include";
    this.expect(TokenType.Lt);

    let path = '';
    while ((this.current as Token).type !== TokenType.Gt && (this.current as Token).type !== TokenType.EOF) {
      const tok = this.advance();
      path += this.tokenToString(tok);
    }
    this.expect(TokenType.Gt);

    return { kind: keyword, path, loc: this.rangeSince(start) };
  }

  // Let statement
  private parseLetStmt(): Statement {
    const start = this.startLoc();
    const name = this.advance().value!;
    this.expect(TokenType.LParen);
    const args = this.parseArgumentList();
    this.expect(TokenType.RParen);

    let child: Statement | undefined;
    if (this.current.type === TokenType.Semicolon) {
      this.advance();
    } else {
      child = this.parseStatement();
    }

    return { kind: "moduleCall", name, args, child, loc: this.rangeSince(start) };
  }

  // Utilities
  private tokenToString(tok: Token): string {
    if (tok.value !== undefined) return tok.value;
    switch (tok.type) {
      case TokenType.Slash: return '/';
      case TokenType.Dot: return '.';
      case TokenType.Minus: return '-';
      case TokenType.Plus: return '+';
      case TokenType.Star: return '*';
      case TokenType.Colon: return ':';
      default: return '';
    }
  }

  private canStartExpr(): boolean {
    const t = this.current.type;
    return (
      t === TokenType.Number ||
      t === TokenType.String ||
      t === TokenType.Identifier ||
      t === TokenType.LBracket ||
      t === TokenType.LParen ||
      t === TokenType.Minus ||
      t === TokenType.Bang ||
      t === TokenType.Tilde ||
      t === TokenType.Plus
    );
  }
}

function attachComments(program: Program, comments: Comment[]): void {
  const targets = collectStatementTargets(program.statements)
    .filter((node): node is Statement & { loc: NonNullable<ASTNode["loc"]> } => Boolean(node.loc))
    .sort((a, b) => a.loc.start.offset - b.loc.start.offset);

  const byEnd = [...targets].sort((a, b) => a.loc.end.offset - b.loc.end.offset);

  for (const comment of [...comments].sort((a, b) => a.loc.start.offset - b.loc.start.offset)) {
    const trailing = [...byEnd]
      .reverse()
      .find(node =>
        node.loc.end.line === comment.loc.start.line &&
        node.loc.end.offset <= comment.loc.start.offset
      );

    if (trailing) {
      trailing.trailingComments ??= [];
      trailing.trailingComments.push(comment);
      continue;
    }

    const leading = targets.find(node => node.loc.start.offset >= comment.loc.end.offset);
    if (leading) {
      leading.leadingComments ??= [];
      leading.leadingComments.push(comment);
    }
  }
}

function collectStatementTargets(statements: Statement[]): Statement[] {
  const out: Statement[] = [];
  for (const stmt of statements) {
    out.push(stmt);
    switch (stmt.kind) {
      case "block":
        out.push(...collectStatementTargets(stmt.statements));
        break;
      case "moduleDecl":
        out.push(...collectStatementTargets([stmt.body]));
        break;
      case "moduleCall":
        if (stmt.child) out.push(...collectStatementTargets([stmt.child]));
        break;
      case "for":
        out.push(...collectStatementTargets([stmt.body]));
        break;
      case "if":
        out.push(...collectStatementTargets([stmt.thenBody]));
        if (stmt.elseBody) out.push(...collectStatementTargets([stmt.elseBody]));
        break;
    }
  }
  return out;
}
