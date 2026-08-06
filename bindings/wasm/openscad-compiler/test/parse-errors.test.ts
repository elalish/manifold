import fs from 'fs';
import path from 'path';
import {afterEach, beforeEach, describe, expect, test, vi} from 'vitest';

import type {Statement} from '../core/ast.js';
import {resolveProgram, resolveProgramWithLibraries} from '../core/resolver.js';


const fixtureDir = path.resolve(__dirname, 'fixtures/parse-errors');

interface Expectation {
  expect: 'throw'|'resolve';
  at?: string;
  from?: Record<string, number>;
  warns?: boolean;
}

const manifest: Record<string, Expectation> =
    JSON.parse(fs.readFileSync(path.join(fixtureDir, 'expected.json'), 'utf8'));

const fixtures =
    fs.readdirSync(fixtureDir).filter((f: string) => f.endsWith('.scad')).sort();

// Statements grouped by the file they came from, keyed by basename
function countByFile(statements: Statement[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const stmt of statements) {
    const file = path.basename(stmt.filename ?? '<unknown>');
    counts[file] = (counts[file] ?? 0) + 1;
  }
  return counts;
}


describe.each([
  ['resolveProgram', (f: string) => resolveProgram(f, [])],
  [
    'resolveProgramWithLibraries',
    (f: string) => resolveProgramWithLibraries(f, []),
  ],
] as const)('%s', (_resolverName, resolve) => {
  let warn: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // Dropping a use'd file is expected to warn; keep it out of test output
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warn.mockRestore();
  });

  test.each(fixtures)('%s', (fixture) => {
    const want = manifest[fixture];
    if (!want) throw new Error(`No expected.json entry for ${fixture}`);
    const file = path.join(fixtureDir, fixture);

    if (want.expect === 'throw') {
      let message: string|undefined;
      let statements: Statement[]|undefined;
      try {
        statements = resolve(file).statements;
      } catch (err) {
        message = (err as Error).message;
      }
     
      expect(statements).toBeUndefined();
      expect(message).toMatch(/failed to parse/);
      if (want.at) {
        expect(message).toContain(`${want.at}:`);
      }
      return;
    }

    const {statements} = resolve(file);
    expect(countByFile(statements)).toEqual(want.from ?? {});
    expect(warn.mock.calls.length > 0).toBe(want.warns ?? false);
  });
});
