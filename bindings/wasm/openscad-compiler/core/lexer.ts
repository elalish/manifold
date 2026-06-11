export enum TokenType {
  // Literals
  Identifier,
  Number,
  String,

  // Punctuation
  LParen,    // (
  RParen,    // )
  LBracket,  // [
  RBracket,  // ]
  LBrace,    // {
  RBrace,    // }
  Comma,     // ,
  Semicolon, // ;
  Colon,     // :
  Hash,      // #

  // Assignment
  Equals,    // =

  // Arithmetic
  Plus,      // +
  Minus,     // -
  Star,      // *
  Slash,     // /
  Percent,   // %
  Caret,     // ^

  // Comparison
  Lt,        // <
  Gt,        // >
  LtEq,     // <=
  GtEq,     // >=
  EqEq,     // ==
  BangEq,   // !=

  // Logical
  And,       // &&
  Or,        // ||
  Bang,      // !

  // Member access
  Dot,       // .

  // Ternary
  Question,  // ?

  LineComment,
  BlockComment,

  EOF,
}

// Position in the source code
export interface SourceLocation {
  line: number;
  column: number;
  offset: number;
}

// A range in the source code (start and end locations)
export interface SourceRange {
  start: SourceLocation;
  end: SourceLocation;
}

export interface Token {
  type: TokenType;
  value?: string;
  range: SourceRange;
}

// Format a source location for display in error messages
export function fmtLoc(loc: SourceLocation, filename: string): string {
  return `${filename}:${loc.line}:${loc.column}`;
}

export function tokenTypeToString(type: TokenType): string {
  switch (type) {
    case TokenType.Identifier: return "Identifier";
    case TokenType.Number: return "Number";
    case TokenType.String: return "String";
    case TokenType.LParen: return "'('";
    case TokenType.RParen: return "')'";
    case TokenType.LBracket: return "'['";
    case TokenType.RBracket: return "']'";
    case TokenType.LBrace: return "'{'";
    case TokenType.RBrace: return "'}'";
    case TokenType.Comma: return "','";
    case TokenType.Semicolon: return "';'";
    case TokenType.Colon: return "':'";
    case TokenType.Hash: return "'#'";
    case TokenType.Equals: return "'='";
    case TokenType.Plus: return "'+'";
    case TokenType.Minus: return "'-'";
    case TokenType.Star: return "'*'";
    case TokenType.Slash: return "'/'";
    case TokenType.Percent: return "'%'";
    case TokenType.Caret: return "'^'";
    case TokenType.Lt: return "'<'";
    case TokenType.Gt: return "'>'";
    case TokenType.LtEq: return "'<='";
    case TokenType.GtEq: return "'>='";
    case TokenType.EqEq: return "'=='";
    case TokenType.BangEq: return "'!='";
    case TokenType.And: return "'&&'";
    case TokenType.Or: return "'||'";
    case TokenType.Bang: return "'!'";
    case TokenType.Dot: return "'.'";
    case TokenType.Question: return "'?'";
    case TokenType.LineComment: return "LineComment";
    case TokenType.BlockComment: return "BlockComment";
    case TokenType.EOF: return "EOF";
    default: return TokenType[type] || "Unknown";
  }
}

// Format a source range for display in error messages
export function fmtRange(range: SourceRange, filename: string): string {
  if (range.start.line === range.end.line) {
    return `${filename}:${range.start.line}:${range.start.column}-${range.end.column}`;
  }
  return `${filename}:${range.start.line}:${range.start.column} – ${filename}:${range.end.line}:${range.end.column}`;
}

export class Lexer {
  private pos = 0;
  private line = 1;
  private column = 1;

  constructor(private input: string, public filename: string = "<unknown>") { }

  // Snapshot the current position
  private loc(): SourceLocation {
    return { line: this.line, column: this.column, offset: this.pos };
  }

  private peek(): string {
    return this.pos < this.input.length ? (this.input[this.pos] ?? "\0") : "\0";
  }

  private peekNext(): string {
    return this.pos + 1 < this.input.length ? (this.input[this.pos + 1] ?? "\0") : "\0";
  }

  private advance(): string {
    const ch = this.input[this.pos++] ?? "\0";
    if (ch === "\n") { this.line++; this.column = 1; }
    else { this.column++; }
    return ch;
  }

  private isAtEnd(): boolean {
    return this.pos >= this.input.length;
  }

  private skipWhitespace(): void {
    while (!this.isAtEnd()) {
      const ch = this.peek();

      // Whitespace
      if (/\s/.test(ch)) {
        this.advance();
        continue;
      }

      break;
    }
  }

  nextToken(): Token {
    this.skipWhitespace();

    const start = this.loc();

    if (this.isAtEnd()) return { type: TokenType.EOF, range: { start, end: start } };

    const ch = this.peek();

    if (ch === "/" && this.peekNext() === "/") {
      return this.readLineComment(start);
    }

    if (ch === "/" && this.peekNext() === "*") {
      return this.readBlockComment(start);
    }

    // Number or Identifier (handles identifiers starting with digits)
    const remaining = this.input.slice(this.pos);
    const idMatch = remaining.match(/^[0-9]*[a-zA-Z_$][a-zA-Z0-9_$]*/);
    const numMatch = remaining.match(/^(?:(?:[0-9]*\.[0-9]+)|(?:[0-9]+(?:\.[0-9]*)?))(?:[eE][+-]?[0-9]+)?/);

    if (idMatch || numMatch) {
      const idLen = idMatch ? idMatch[0].length : 0;
      const numLen = numMatch ? numMatch[0].length : 0;

      if (idLen > numLen) {
        return this.readIdentifier(start);
      } else if (numLen > 0) {
        return this.readNumber(start);
      }
    }

    // String literal
    if (ch === '"') {
      return this.readString(start);
    }

    // Operators and punctuation
    this.advance();

    const mk = (type: TokenType): Token => ({ type, range: { start, end: this.loc() } });

    switch (ch) {
      case "(": return mk(TokenType.LParen);
      case ")": return mk(TokenType.RParen);
      case "[": return mk(TokenType.LBracket);
      case "]": return mk(TokenType.RBracket);
      case "{": return mk(TokenType.LBrace);
      case "}": return mk(TokenType.RBrace);
      case ",": return mk(TokenType.Comma);
      case ";": return mk(TokenType.Semicolon);
      case ":": return mk(TokenType.Colon);
      case "#": return mk(TokenType.Hash);
      case "?": return mk(TokenType.Question);
      case ".": return mk(TokenType.Dot);
      case "+": return mk(TokenType.Plus);
      case "-": return mk(TokenType.Minus);
      case "*": return mk(TokenType.Star);
      case "%": return mk(TokenType.Percent);
      case "^": return mk(TokenType.Caret);

      case "/": return mk(TokenType.Slash);

      case "=":
        if (this.peek() === "=") { this.advance(); return mk(TokenType.EqEq); }
        return mk(TokenType.Equals);

      case "!":
        if (this.peek() === "=") { this.advance(); return mk(TokenType.BangEq); }
        return mk(TokenType.Bang);

      case "<":
        if (this.peek() === "=") { this.advance(); return mk(TokenType.LtEq); }
        return mk(TokenType.Lt);

      case ">":
        if (this.peek() === "=") { this.advance(); return mk(TokenType.GtEq); }
        return mk(TokenType.Gt);

      case "&":
        if (this.peek() === "&") { this.advance(); return mk(TokenType.And); }
        throw new Error(`Unexpected character '&' (did you mean '&&'?) at ${fmtLoc(start, this.filename)}`);

      case "|":
        if (this.peek() === "|") { this.advance(); return mk(TokenType.Or); }
        throw new Error(`Unexpected character '|' (did you mean '||'?) at ${fmtLoc(start, this.filename)}`);
    }

    throw new Error(`Unexpected character '${ch}' at ${fmtLoc(start, this.filename)}`);
  }

  private readNumber(start: SourceLocation): Token {
    const startPos = this.pos;

    // Integer part
    while (/\d/.test(this.peek())) this.advance();

    // Fractional part
    if (this.peek() === "." && /[\d\0]/.test(this.peekNext())) {
      this.advance();
      while (/\d/.test(this.peek())) this.advance();
    } else if (this.peek() === ".") {
      this.advance(); // trailing dot
    }

    // Scientific notation
    if (this.peek() === "e" || this.peek() === "E") {
      this.advance();
      if (this.peek() === "+" || this.peek() === "-") this.advance();
      while (/\d/.test(this.peek())) this.advance();
    }

    return { type: TokenType.Number, value: this.input.slice(startPos, this.pos), range: { start, end: this.loc() } };
  }

  private readString(start: SourceLocation): Token {
    this.advance(); // consume opening "
    let result = "";

    while (!this.isAtEnd() && this.peek() !== '"') {
      if (this.peek() === "\\") {
        this.advance();
        const esc = this.advance();
        switch (esc) {
          case "n": result += "\n"; break;
          case "t": result += "\t"; break;
          case "r": result += "\r"; break;
          case "\\": result += "\\"; break;
          case '"': result += '"'; break;
          case "x": {
            let hex = "";
            for (let i = 0; i < 2; i++) {
              if (/[0-9a-fA-F]/.test(this.peek())) {
                hex += this.advance();
              } else {
                break;
              }
            }
            if (hex.length === 2) {
              result += String.fromCharCode(parseInt(hex, 16));
            } else {
              result += "x" + hex;
            }
            break;
          }
          case "u": {
            let hex = "";
            for (let i = 0; i < 4; i++) {
              if (/[0-9a-fA-F]/.test(this.peek())) {
                hex += this.advance();
              } else {
                break;
              }
            }
            if (hex.length === 4) {
              result += String.fromCharCode(parseInt(hex, 16));
            } else {
              result += "u" + hex;
            }
            break;
          }
          case "U": {
            let hex = "";
            for (let i = 0; i < 6; i++) {
              if (/[0-9a-fA-F]/.test(this.peek())) {
                hex += this.advance();
              } else {
                break;
              }
            }
            if (hex.length === 6) {
              result += String.fromCodePoint(parseInt(hex, 16));
            } else {
              result += "U" + hex;
            }
            break;
          }
          default: result += esc; break;
        }
      } else {
        result += this.advance();
      }
    }

    if (this.isAtEnd()) throw new Error(`Unterminated string literal at ${fmtLoc(start, this.filename)}`);
    this.advance();

    return { type: TokenType.String, value: result, range: { start, end: this.loc() } };
  }

  private readIdentifier(start: SourceLocation): Token {
    const startPos = this.pos;
    while (/[a-zA-Z_$0-9]/.test(this.peek())) this.advance();
    return { type: TokenType.Identifier, value: this.input.slice(startPos, this.pos), range: { start, end: this.loc() } };
  }

  private readLineComment(start: SourceLocation): Token {
    const startPos = this.pos;
    this.advance(); this.advance();
    while (!this.isAtEnd() && this.peek() !== "\n") this.advance();
    return {
      type: TokenType.LineComment,
      value: this.input.slice(startPos, this.pos),
      range: { start, end: this.loc() },
    };
  }

  private readBlockComment(start: SourceLocation): Token {
    const startPos = this.pos;
    this.advance(); this.advance();
    while (!this.isAtEnd()) {
      if (this.peek() === "*" && this.peekNext() === "/") {
        this.advance(); this.advance();
        return {
          type: TokenType.BlockComment,
          value: this.input.slice(startPos, this.pos),
          range: { start, end: this.loc() },
        };
      }
      this.advance();
    }
    throw new Error(`Unterminated block comment at ${fmtLoc(start, this.filename)}`);
  }
}
