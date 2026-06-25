import { describe, test, expect } from "vitest";
import { execFileSync } from "node:child_process";
import { readFileSync, existsSync, mkdtempSync } from "node:fs";
import path from "path";
import { tmpdir } from "node:os";

const inputFile = process.env.TEST_FILE!;

if (!inputFile) throw new Error("Missing input file");

const tsForMatch = (scadPath: string) => path.join(path.dirname(scadPath).replace("examples", "out"), `${path.basename(scadPath, ".scad")}.ts`);

describe("echo equality", () => {
  test(`Test for ${inputFile}`, async () => {
    const tsPath = tsForMatch(inputFile);
    if (!existsSync(tsPath)) throw new Error(`No compiled file at ${tsPath}`);
    const expected = normalize(runOpenscad(inputFile), "openscad");
    const actual = normalize(runCompiled(tsPath), "ts");
    expect(actual.length).toBe(expected.length);
    expected.forEach((e, i) => expectEchoEqual(actual[i], e, i));
  })
});

function runOpenscad(scadPath: string): string[] {
  const out = path.join(mkdtempSync(path.join(tmpdir(), "scad-")), "out.echo");
  execFileSync("openscad", ["-o", out, "--backend=manifold", scadPath], { stdio: "ignore" });
  return readFileSync(out, "utf8").split(/\r?\n/);
}

function runCompiled(tsPath: string): string[] {
  try {
    const out = execFileSync(
      process.execPath,
      ["--import", "tsx", tsPath],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }
    );
    return out.split(/\r?\n/);
  } catch (e: any) {
    throw new Error(`Compiled run failed (${tsPath}):\n${e.stderr ?? e.message}`);
  }
}

function normalize(lines: string[], source: "openscad" | "ts"): Tok[][] {
  return lines
    .map((l) => l.replace(/\r$/, ""))
    .map((l) =>
      source === "openscad"
        ? (l.startsWith("ECHO: ") ? l.slice(6) : null)
        : (l.trim() === "" ? null : l.trim())
    )
    .filter((l): l is string => l !== null)
    .map((l) => expandRanges(lexLine(l)));
}

// numeric token as number and anything else as string
type Tok = number | string;

const NUMERIC = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;

const LITERAL_MAP: Record<string, string> = {
  undefined: "undef",
  null: "undef",
  undef: "undef",
  NaN: "undef",
  nan: "undef",
  Infinity: "inf",
  "-Infinity": "-inf",
};

// Split a printed echo line into a flat token stream dropping quotes
function lexLine(line: string): Tok[] {
  const s = line.replace(/["']/g, "");
  const seps = " \t,[]:=";
  const toks: Tok[] = [];
  let i = 0;
  while (i < s.length) {
    const ch = s[i]!;
    if (ch === " " || ch === "\t" || ch === ",") { i++; continue; }
    if (ch === "[" || ch === "]" || ch === ":" || ch === "=") { toks.push(ch); i++; continue; }
    let j = i;
    while (j < s.length && !seps.includes(s[j]!)) j++;
    const word = s.slice(i, j);
    i = j;
    if (word === "") continue;
    toks.push(NUMERIC.test(word) ? Number(word) : (LITERAL_MAP[word] ?? word));
  }
  return toks;
}

// expand OpenSCAD range syntax to match compiled range output
function expandRanges(toks: Tok[]): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < toks.length) {
    if (
      toks[i] === "[" &&
      typeof toks[i + 1] === "number" && toks[i + 2] === ":" &&
      typeof toks[i + 3] === "number" && toks[i + 4] === ":" &&
      typeof toks[i + 5] === "number" && toks[i + 6] === "]"
    ) {
      const start = toks[i + 1] as number;
      const step = toks[i + 3] as number;
      const end = toks[i + 5] as number;
      out.push("[");
      if (step > 0) for (let v = start; v <= end; v += step) out.push(v);
      else if (step < 0) for (let v = start; v >= end; v += step) out.push(v);
      out.push("]");
      i += 7;
    } else {
      out.push(toks[i]!);
      i++;
    }
  }
  return out;
}

// compare one echo line, numbers with a relative tolerance
function expectEchoEqual(a: Tok[], e: Tok[], lineIdx: number) {
  expect(a.length, `line ${lineIdx} token count\n  ts: ${JSON.stringify(a)}\n  os: ${JSON.stringify(e)}`).toBe(e.length);
  a.forEach((tok, i) => {
    const want = e[i]!;
    if (typeof tok === "number" && typeof want === "number") {
      const tol = 1e-5 * Math.max(1, Math.abs(tok), Math.abs(want));
      expect(
        Math.abs(tok - want),
        `line ${lineIdx} token ${i}: ${tok} vs ${want}`,
      ).toBeLessThanOrEqual(tol);
    } else {
      expect(tok, `line ${lineIdx} token ${i}`).toEqual(want);
    }
  });
}