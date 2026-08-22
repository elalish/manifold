import path from 'path';

import {walk} from './ast.js';
import type {Argument, Expr, KindedNode, Statement} from './ast.js';
import {getFontPath} from './resolver.js';
import {globalFileResolver, signatures} from './state.js';
import type {FontScan, FontTargets, ModuleDeclStmtType,} from './types.js';

// OpenSCAD's default typeface, used when a text() call names none
export const DEFAULT_FONT_SPEC = 'Liberation Sans:style=Regular';

// mapping of lowercased filename to the name as it actually sits on disk, per
// font directory
const fontDirListings = new Map<string, Map<string, string>>();

export function fontCandidateNames(stmts: Statement[]): FontTargets {
  const names = new Set<string>(['text']);
  const decls = new Set<ModuleDeclStmtType>();
  const visit = (list: Statement[]): void => {
    for (const s of list) {
      if (s.kind === 'scope') {
        visit(s.statements);
      } else if (s.kind === 'moduleDecl') {
        names.add(s.name);
        decls.add(s);
      }
    }
  };
  visit(stmts);
  return {names, decls};
}

export function isFontRelatedName(name: string): boolean {
  const lower = name.toLowerCase();
  return lower.includes('font') || lower.includes('style') ||
      lower.includes('family');
}

// Scan FONTPATH and return the basenames of the .ttf/.otf files whose family
// and style match one of the program's font literals
export async function fontsMatchingLiterals(fontLiterals: Set<string>):
    Promise<string[]> {
  const fontDir = await getFontPath();
  if (!fontDir || !await globalFileResolver?.exists(fontDir)) return [];

  const matched: string[] = [];
  try {
    const files = await globalFileResolver?.readDir(fontDir)!;
    const cleanedLiterals =
        Array.from(fontLiterals)
            .map(lit => lit.toLowerCase().replace(/[^a-z0-9]/g, ''));

    for (const file of files) {
      const ext = path.extname(file).toLowerCase();
      if (ext !== '.ttf' && ext !== '.otf') continue;

      const basename = path.basename(file, ext);
      const dashIdx = basename.indexOf('-');
      const family = dashIdx >= 0 ? basename.slice(0, dashIdx) : basename;
      const style = dashIdx >= 0 ? basename.slice(dashIdx + 1) : 'Regular';

      const cleanedFamily = family.toLowerCase().replace(/[^a-z0-9]/g, '');
      const cleanedStyle = style.toLowerCase().replace(/[^a-z0-9]/g, '');

      // Check if family matches any of the cleaned literals
      const familyMatched = cleanedLiterals.some(
          lit => lit.includes(cleanedFamily) ||
              (lit.length >= 4 && cleanedFamily.includes(lit)));
      if (!familyMatched) continue;

      const styleMatched =
          cleanedStyle === 'regular' || cleanedLiterals.some(lit => {
            if (cleanedStyle === 'bolditalic') {
              return (lit.includes('bold') && lit.includes('italic')) ||
                  lit.includes('bolditalic');
            }
            return lit.includes(cleanedStyle);
          });
      if (styleMatched) matched.push(basename);
    }
  } catch (e) {
    console.warn('Warning: failed to read font directory for matching:', e);
  }
  return matched;
}

function fontSpecToFilename(fontSpec: string): string {
  const cleaned = fontSpec.replace(/"/g, '').trim();
  const parts = cleaned.split(':');
  const family = (parts[0] || 'Liberation Sans').trim().replace(/\s+/g, '');

  let style = 'Regular';
  for (let i = 1; i < parts.length; i++) {
    const part = parts[i]!.trim();
    const match = part.match(/^style\s*=\s*(.+)$/i);
    if (match) {
      style = match[1]!.trim().replace(/\s+/g, '');
      break;
    }
  }

  return `${family}-${style}`;
}

async function fontDirListing(fontDir: string): Promise<Map<string, string>> {
  let listing = fontDirListings.get(fontDir);
  if (listing) return listing;

  listing = new Map<string, string>();
  try {
    const files = await globalFileResolver?.readDir(fontDir);
    if (!files) return listing;
    for (const file of files) {
      listing.set(file.toLowerCase(), file);
    }
  } catch (e) {
    console.warn(`Warning: failed to read font directory "${fontDir}":`, e);
  }
  fontDirListings.set(fontDir, listing);
  return listing;
}

// Resolves `basename` to a font file, ignoring case. Returns the name as it is
// on disk
async function resolveFontFile(fontDir: string, basename: string):
    Promise<{filePath: string; basename: string; mimeType: string}|undefined> {
  const listing = await fontDirListing(fontDir);
  const candidates: ReadonlyArray<[string, string]> =
      [['.ttf', 'font/ttf'], ['.otf', 'font/otf']];

  for (const [ext, mimeType] of candidates) {
    const file = listing.get(`${basename}${ext}`.toLowerCase());
    if (file) {
      return {
        filePath: path.join(fontDir, file),
        basename: path.basename(file, path.extname(file)),
        mimeType,
      };
    }
  }
  return undefined;
}

export async function generateFontBase64(
    fontSpec: string, compilerDir: string): Promise<string|undefined> {
  const fontDir = await getFontPath();
  if (!fontDir) {
    console.warn(
        `Warning: FONTPATH environment variable not set — cannot load font "${
            fontSpec}". Text will render as empty cross-section.`);
    return undefined;
  }

  const canonical = fontSpecToFilename(fontSpec);
  const resolved = await resolveFontFile(fontDir, fontSpec) ??
      await resolveFontFile(fontDir, canonical);

  if (!resolved) {
    console.warn(`Warning: No "${fontSpec}" or "${canonical}" .ttf/.otf in "${
        fontDir}" — text using "${
        fontSpec}" will render as empty cross-section.`);
    return undefined;
  }

  const {filePath: fontFilePath, basename: filename, mimeType} = resolved;
  const fontBytes =
      (await globalFileResolver?.readBinary(fontFilePath)) as Uint8Array;
  const base64 = Buffer.from(fontBytes).toString('base64');

  const fontsDir = path.join(compilerDir, 'runtime', 'fonts');
  await globalFileResolver?.makeDir(fontsDir);

  const outFile = path.join(fontsDir, `${filename}_base64.ts`);
  const content =
      `// Auto-generated by openscad-to-manifold compiler — do not edit.\nexport const fontBase64 = "data:${
          mimeType};base64,${base64}";\n`;
  await globalFileResolver?.writeText(outFile, content);
  console.log(`Generated font base64: ${outFile} (${
      (fontBytes.length / 1024).toFixed(1)} KB)`);

  return filename;
}

export function resolveFontLiterals(
    font: FontScan, candidates: Set<string>): Set<string> {
  const textModules = new Set<string>(['text']);
  let changed = true;
  while (changed) {
    changed = false;
    for (const name of candidates) {
      if (textModules.has(name)) continue;
      const called = font.edges.get(name);
      if (!called) continue;
      for (const callee of called) {
        if (!textModules.has(callee)) continue;
        textModules.add(name);
        changed = true;
        break;
      }
    }
  }

  const literals = font.literals;
  for (const {module, exprs} of font.paramDefaults)
    if (textModules.has(module))
      for (const e of exprs) collectStringLiterals(e, literals);
  for (const {name, args} of font.calls)
    if (textModules.has(name))
      for (const e of resolveCallArgs(name, args).values())
        collectStringLiterals(e, literals);
  for (const {module, value} of font.scopedVars)
    if (textModules.has(module)) collectStringLiterals(value, literals);
  return literals;
}

export function collectStringLiterals(
    node: KindedNode, literals: Set<string>): void {
  walk(node, n => {
    if (n.kind === 'string') literals.add(n.value);
  });
}

function resolveCallArgs(
    moduleCallName: string, callArgs: Argument[]): Map<string, Expr> {
  const resolved = new Map<string, Expr>();
  const sig = signatures.get(`mod:${moduleCallName}`);
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
