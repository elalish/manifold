import path from 'path';

import {runtimeFileResolver} from './host.js';

// Font directory listing
const fontDirListings = new Map<string, Map<string, string>>();

// Font spec to its base64 data URL
const fontDataCache = new Map<string, string|undefined>();

// `Family:style=Style` to the `Family-Style` basename the file carries on disk
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

function fontDirListing(fontDir: string): Map<string, string> {
  let listing = fontDirListings.get(fontDir);
  if (listing) return listing;

  listing = new Map<string, string>();
  try {
    const fontFiles = runtimeFileResolver.readDir(fontDir);
    for (const file of fontFiles) listing.set(file.toLowerCase(), file);
  } catch (e) {
    console.warn(`Warning: failed to read font directory "${fontDir}":`, e);
  }
  fontDirListings.set(fontDir, listing);
  return listing;
}

function resolveFontFile(fontDir: string, basename: string):
    {filePath: string; mimeType: string}|undefined {
  const listing = fontDirListing(fontDir);
  const candidates: ReadonlyArray<[string, string]> =
      [['.ttf', 'font/ttf'], ['.otf', 'font/otf']];

  for (const [ext, mimeType] of candidates) {
    const file = listing.get(`${basename}${ext}`.toLowerCase());
    if (file) return {filePath: path.join(fontDir, file), mimeType};
  }
  return undefined;
}

// Reads the font file a text() spec names from FONTPATH and returns it as a
// base64 data URL
export function computeFontData(fontSpec: string): string|undefined {
  if (fontDataCache.has(fontSpec)) return fontDataCache.get(fontSpec);

  let data = loadFontData(fontSpec);
  fontDataCache.set(fontSpec, data);
  return data;
}

function loadFontData(fontSpec: string): string|undefined {
  const fontDir = process.env.FONTPATH?.trim();
  if (!fontDir || !runtimeFileResolver.exists(fontDir)) {
    console.warn(
        `Warning: FONTPATH environment variable not set — cannot load font "${
            fontSpec}". Text will render as empty cross-section.`);
    return undefined;
  }

  const canonical = fontSpecToFilename(fontSpec);
  const resolved =
      resolveFontFile(fontDir, fontSpec) ?? resolveFontFile(fontDir, canonical);

  if (!resolved) {
    console.warn(`Warning: No "${fontSpec}" or "${canonical}" .ttf/.otf in "${
        fontDir}" — text using "${
        fontSpec}" will render as empty cross-section.`);
    return undefined;
  }

  const {filePath, mimeType} = resolved;
  try {
    const buffer = runtimeFileResolver.readBinary(filePath);
    // convert
    const base64 = buffer?.toString('base64');
    return `data:${mimeType};base64,${base64}`;
  } catch (e) {
    console.warn(`Warning: failed to read font file "${filePath}":`, e);
    return undefined;
  }
}
