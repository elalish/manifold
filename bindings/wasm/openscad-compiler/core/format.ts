const INDENT = '  ';

interface OpenBracket {
  ch: string;
  line: number;
  object: boolean;
}

const OPENER_OF: Record<string, string> = {
  '}': '{',
  ')': '(',
  ']': '['
};

// A `{` starts a block after a header, statement, or nothing; otherwise
// it's an object literal or destructuring pattern.
const BLOCK_BRACE = /(?:^|[)};]|\b(?:else|try|finally|do)|=>)$/;

export function formatCode(code: string): string {
  const out: string[] = [];
  const stack: OpenBracket[] = [];

  let line = '';
  let lineDepth = 0;
  let lineNo = 0;
  let endedBySource = true;

  const depth = () => {
    let result = 0;
    let lastLine = -1;

    for (const bracket of stack) {
      if (bracket.line !== lastLine) result++;
      lastLine = bracket.line;
    }

    return result;
  };

  const flush = () => {
    if (line !== '') {
      out.push(INDENT.repeat(lineDepth) + line.trimEnd());
      lineNo++;
    }

    line = '';
    endedBySource = false;
  };

  const put = (text: string, indent?: number) => {
    if (!line && text.trim()) {
      lineDepth = indent ?? depth();
    }

    line += text;
  };

  const hasTrailingComment = (from: number) => {
    while (from < code.length && /[ \t]/.test(code[from]!)) from++;
    return code[from] === '/' &&
        (code[from + 1] === '/' || code[from + 1] === '*');
  };

  const isStructuralSeparator = (c: string) => {
    const top = stack.at(-1);

    if (c === ';') {
      return !top || top.ch === '{';
    }

    return top?.ch === '{' && top.object;
  };

  let i = 0;

  while (i < code.length) {
    const c = code[i]!;

    // Keep strings intact.
    if (c === '"' || c === '\'' || c === '`') {
      const start = i++;

      while (i < code.length && code[i] !== c) {
        i += code[i] === '\\' ? 2 : 1;
      }

      put(code.slice(start, Math.min(i + 1, code.length)));
      i++;
      continue;
    }

    // Line comment.
    if (c === '/' && code[i + 1] === '/') {
      const end = code.indexOf('\n', i);
      put(code.slice(i, end === -1 ? code.length : end).trimEnd());
      i = end === -1 ? code.length : end;
      continue;
    }

    // Block comment.
    if (c === '/' && code[i + 1] === '*') {
      const end = code.indexOf('*/', i + 2);
      put(code.slice(i, end === -1 ? code.length : end + 2));
      i = end === -1 ? code.length : end + 2;
      continue;
    }

    // Source newline.
    if (c === '\n') {
      if (line) {
        flush();
        endedBySource = true;
      } else if (endedBySource && out.at(-1) !== '') {
        out.push('');
        lineNo++;
      } else {
        endedBySource = true;
      }

      i++;
      continue;
    }

    // Whitespace.
    if (c === ' ' || c === '\t' || c === '\r') {
      if (line && !line.endsWith(' ')) line += ' ';
      i++;
      continue;
    }

    // Opening brace.
    if (c === '{') {
      let j = i + 1;

      while (j < code.length && /\s/.test(code[j]!)) j++;

      // Keep `{}` inline.
      if (code[j] === '}') {
        put('{}');
        i = j + 1;
        continue;
      }

      const object = !BLOCK_BRACE.test(line.trimEnd());

      put('{');
      stack.push({ch: '{', line: lineNo, object});

      flush();
      i++;
      continue;
    }

    // Other opening brackets.
    if (c === '(' || c === '[') {
      put(c);
      stack.push({ch: c, line: lineNo, object: false});
      i++;
      continue;
    }

    // Closing bracket.
    if (c === '}' || c === ')' || c === ']') {
      if (c === '}' && line) flush();

      const top = stack.at(-1);
      const matched = top?.ch === OPENER_OF[c] ? stack.pop() : undefined;

      put(c, matched && depth());
      i++;
      continue;
    }

    // Structural separators.
    if (c === ';' || c === ',') {
      put(c);
      i++;

      if (isStructuralSeparator(c) && !hasTrailingComment(i)) {
        flush();
      }

      continue;
    }

    put(c);
    i++;
  }

  flush();

  while (out.at(-1) === '') {
    out.pop();
  }

  return out.join('\n') + '\n';
}