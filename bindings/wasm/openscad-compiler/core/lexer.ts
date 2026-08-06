export enum TokenType {
  // Literals
  Identifier,
  Number,
  String,

  // Punctuation
  LParen,     // (
  RParen,     // )
  LBracket,   // [
  RBracket,   // ]
  LBrace,     // {
  RBrace,     // }
  Comma,      // ,
  Semicolon,  // ;
  Colon,      // :
  Hash,       // #

  // Assignment
  Equals,  // =

  // Arithmetic
  Plus,     // +
  Minus,    // -
  Star,     // *
  Slash,    // /
  Percent,  // %
  Caret,    // ^

  // Comparison
  Lt,      // <
  Gt,      // >
  LtEq,    // <=
  GtEq,    // >=
  EqEq,    // ==
  BangEq,  // !=

  // Logical
  And,   // &&
  Or,    // ||
  Bang,  // !

  // Bitwise
  Amp,    // &
  Pipe,   // |
  Tilde,  // ~
  Shl,    // <<
  Shr,    // >>

  // Member access
  Dot,  // .

  // Ternary
  Question,  // ?

  LineComment,
  BlockComment,

  EOF,
}

// Operator and punctuation lexemes. This single table
// drives both lexing (text -> TokenType) and printing (TokenType -> text)
const OPERATOR_TABLE: ReadonlyArray<readonly[string, TokenType]> = [
  ['(', TokenType.LParen],
  [')', TokenType.RParen],
  ['[', TokenType.LBracket],
  [']', TokenType.RBracket],
  ['{', TokenType.LBrace],
  ['}', TokenType.RBrace],
  [',', TokenType.Comma],
  [';', TokenType.Semicolon],
  [':', TokenType.Colon],
  ['#', TokenType.Hash],
  ['=', TokenType.Equals],
  ['+', TokenType.Plus],
  ['-', TokenType.Minus],
  ['*', TokenType.Star],
  ['/', TokenType.Slash],
  ['%', TokenType.Percent],
  ['^', TokenType.Caret],
  ['<', TokenType.Lt],
  ['>', TokenType.Gt],
  ['<=', TokenType.LtEq],
  ['>=', TokenType.GtEq],
  ['==', TokenType.EqEq],
  ['!=', TokenType.BangEq],
  ['&&', TokenType.And],
  ['||', TokenType.Or],
  ['!', TokenType.Bang],
  ['&', TokenType.Amp],
  ['|', TokenType.Pipe],
  ['~', TokenType.Tilde],
  ['<<', TokenType.Shl],
  ['>>', TokenType.Shr],
  ['.', TokenType.Dot],
  ['?', TokenType.Question],
];

const OPERATOR_TYPES = new Map<string, TokenType>(OPERATOR_TABLE);
const OPERATOR_LEXEMES = new Map<TokenType, string>(
    OPERATOR_TABLE.map(([text, type]) => [type, text]));

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
  const lexeme = OPERATOR_LEXEMES.get(type);
  if (lexeme !== undefined) return `'${lexeme}'`;

  return TokenType[type] ?? 'Unknown';
}

// Format a source range for display in error messages
export function fmtRange(range: SourceRange, filename: string): string {
  if (range.start.line === range.end.line) {
    return `${filename}:${range.start.line}:${range.start.column}-${
        range.end.column}`;
  }
  return `${filename}:${range.start.line}:${range.start.column} – ${filename}:${
      range.end.line}:${range.end.column}`;
}

// Regular expressions
const HEX_NUMBER_REGEX = /^0[xX][0-9a-fA-F]+/;
const IDENTIFIER_REGEX = /^[0-9]*[a-zA-Z_$][a-zA-Z0-9_$]*/;
const NUMBER_REGEX =
    /^(?:(?:[0-9]*\.[0-9]+)|(?:[0-9]+(?:\.[0-9]*)?))(?:[eE][+-]?[0-9]+)?/;
const OCTAL_DIGIT_REGEX = /[0-7]/;
const HEX_DIGIT_REGEX = /[0-9a-fA-F]/;
const WHITESPACE_REGEX = /\s/;

export class Lexer {
  private pos = 0;
  private line = 1;
  private column = 1;

  private pathState: 'none'|'expectOpen'|'inPath' = 'none';

  constructor(private input: string, public filename: string = '<unknown>') {}

  // Snapshot the current position
  private loc(): SourceLocation {
    return {line: this.line, column: this.column, offset: this.pos};
  }

  private peek(): string {
    return this.pos < this.input.length ? (this.input[this.pos] ?? '\0') : '\0';
  }

  private peekNext(): string {
    return this.pos + 1 < this.input.length ?
        (this.input[this.pos + 1] ?? '\0') :
        '\0';
  }

  private advance(count: number = 1): string {
    let lastCh = '\0';
    for (let i = 0; i < count; i++) {
      const ch = this.input[this.pos++] ?? '\0';
      if (ch === '\n') {
        this.line++;
        this.column = 1;
      } else {
        this.column++;
      }
      lastCh = ch;
    }
    return lastCh;
  }

  private isAtEnd(): boolean {
    return this.pos >= this.input.length;
  }

  private skipWhitespace(): void {
    while (!this.isAtEnd()) {
      const ch = this.peek();

      // Whitespace
      if (WHITESPACE_REGEX.test(ch)) {
        this.advance();
        continue;
      }

      break;
    }
  }

  nextToken(): Token {
    this.skipWhitespace();

    const start = this.loc();

    if (this.isAtEnd())
      return {type: TokenType.EOF, range: {start, end: start}};

    const ch = this.peek();

    if (this.pathState === 'inPath') {
      if (ch === '>') {
        this.advance();
        this.pathState = 'none';
        return {type: TokenType.Gt, range: {start, end: this.loc()}};
      }
      return this.readIncludePath(start);
    }
    if (this.pathState === 'expectOpen') {
      if (ch === '<') {
        this.advance();
        this.pathState = 'inPath';
        return {type: TokenType.Lt, range: {start, end: this.loc()}};
      }
      this.pathState = 'none';
    }

    if (ch === '/' && this.peekNext() === '/') {
      return this.readLineComment(start);
    }

    if (ch === '/' && this.peekNext() === '*') {
      return this.readBlockComment(start);
    }

    // Number or Identifier (handles identifiers starting with digits)
    const remaining = this.input.slice(this.pos);

    const hexMatch = remaining.match(HEX_NUMBER_REGEX);
    if (hexMatch) {
      this.advance(hexMatch[0].length);
      return {
        type: TokenType.Number,
        value: hexMatch[0],
        range: {start, end: this.loc()}
      };
    }

    const idMatch = remaining.match(IDENTIFIER_REGEX);
    const numMatch = remaining.match(NUMBER_REGEX);

    const idLen = idMatch?.[0].length ?? 0;
    const numLen = numMatch?.[0].length ?? 0;

    if (idMatch && idLen > numLen) {
      return this.readIdentifier(start, idMatch[0]);
    } else if (numMatch) {
      // NUMBER_REGEX always consumes at least one digit, so a match is a token.
      return this.readNumber(start, numMatch[0]);
    }

    // String literal
    if (ch === '"') {
      return this.readString(start);
    }

    // Prefer the two-character lexeme, then fall back to the single-character one.
    for (const lexeme of [ch + this.peekNext(), ch]) {
      const type = OPERATOR_TYPES.get(lexeme);
      if (type !== undefined) {
        this.advance(lexeme.length);
        return {type, range: {start, end: this.loc()}};
      }
    }

    throw new Error(
        `Unexpected character '${ch}' at ${fmtLoc(start, this.filename)}`);
  }

  private readNumber(start: SourceLocation, value: string): Token {
    this.advance(value.length);
    return {type: TokenType.Number, value, range: {start, end: this.loc()}};
  }

  private readString(start: SourceLocation): Token {
    this.advance();  // consume opening "
    let result = '';

    while (!this.isAtEnd() && this.peek() !== '"') {
      if (this.peek() === '\\') {
        this.advance();
        const esc = this.advance();
        switch (esc) {
          case 'n':
            result += '\n';
            break;
          case 't':
            result += '\t';
            break;
          case 'r':
            result += '\r';
            break;
          case '\\':
            result += '\\';
            break;
          case '"':
            result += '"';
            break;
          case 'x': {
            if (OCTAL_DIGIT_REGEX.test(this.peek())) {
              const hi = this.advance();
              if (HEX_DIGIT_REGEX.test(this.peek())) {
                const code = parseInt(hi + this.advance(), 16);
                if (code !== 0) result += String.fromCharCode(code);
              } else {
                result += 'x' + hi;
              }
            } else {
              result += 'x';
            }
            break;
          }
          case 'u': {
            let hex = '';
            for (let i = 0; i < 4; i++) {
              if (HEX_DIGIT_REGEX.test(this.peek())) {
                hex += this.advance();
              } else {
                break;
              }
            }
            if (hex.length === 4) {
              const code = parseInt(hex, 16);
              if (code !== 0)
                result += String.fromCharCode(code);  // OpenSCAD drops NUL
            } else {
              result += 'u' + hex;
            }
            break;
          }
          case 'U': {
            let hex = '';
            for (let i = 0; i < 6; i++) {
              if (HEX_DIGIT_REGEX.test(this.peek())) {
                hex += this.advance();
              } else {
                break;
              }
            }
            if (hex.length === 6) {
              const code = parseInt(hex, 16);
              if (code !== 0)
                result += String.fromCodePoint(code);  // OpenSCAD drops NUL
            } else {
              result += 'U' + hex;
            }
            break;
          }
          default:
            result += esc;
            break;
        }
      } else {
        result += this.advance();
      }
    }

    if (this.isAtEnd())
      throw new Error(
          `Unterminated string literal at ${fmtLoc(start, this.filename)}`);
    this.advance();

    return {
      type: TokenType.String,
      value: result,
      range: {start, end: this.loc()}
    };
  }

  private readIdentifier(start: SourceLocation, value: string): Token {
    this.advance(value.length);
    if (value === 'include' || value === 'use') this.pathState = 'expectOpen';
    return {type: TokenType.Identifier, value, range: {start, end: this.loc()}};
  }

  private readIncludePath(start: SourceLocation): Token {
    const startPos = this.pos;
    while (!this.isAtEnd() && this.peek() !== '>' && this.peek() !== '\n') {
      this.advance();
    }
    return {
      type: TokenType.String,
      value: this.input.slice(startPos, this.pos),
      range: {start, end: this.loc()}
    };
  }

  private readLineComment(start: SourceLocation): Token {
    const startPos = this.pos;
    this.advance();
    this.advance();
    while (!this.isAtEnd() && this.peek() !== '\n') this.advance();
    return {
      type: TokenType.LineComment,
      value: this.input.slice(startPos, this.pos),
      range: {start, end: this.loc()},
    };
  }

  private readBlockComment(start: SourceLocation): Token {
    const startPos = this.pos;
    this.advance();
    this.advance();
    while (!this.isAtEnd()) {
      if (this.peek() === '*' && this.peekNext() === '/') {
        this.advance();
        this.advance();
        return {
          type: TokenType.BlockComment,
          value: this.input.slice(startPos, this.pos),
          range: {start, end: this.loc()},
        };
      }
      this.advance();
    }
    throw new Error(
        `Unterminated block comment at ${fmtLoc(start, this.filename)}`);
  }
}
