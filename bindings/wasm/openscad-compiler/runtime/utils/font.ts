import path from 'path-browserify';

import {environmentResolver, runtimeFileResolver} from './host.js';

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

async function fetchAndStoreFontFile(fontSpec: string) {
  const [family, ...props] = fontSpec.split(':');

  const style =
      props.find(p => p.startsWith('style='))?.split('=')[1]?.toLowerCase() ===
          'italic' ?
      'italic' :
      'normal';

  const weight =
      Number(props.find(p => p.startsWith('weight='))?.split('=')[1]) || 400;

  // resolve font through fontsource
  const url = `https://api.fontsource.org/v1/fonts?family=${
      encodeURIComponent(family.trim())}`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to resolve font: ${family}`);
  }

  const fonts = await response.json();

  if (!fonts.length) {
    throw new Error(`Font not found: ${family}`);
  }

  const font = fonts[0];
  const variant = font.variants?.[weight]?.[style];

  if (!variant?.url?.ttf) {
    throw new Error(
        `Variant not found: ${family} ${weight} ${style}`,
    );
  }

  // download font
  const fontResponse = await fetch(variant.url.ttf);

  if (!fontResponse.ok) {
    throw new Error(`Failed to download font: ${variant.url.ttf}`);
  }

  const blob = await fontResponse.blob();

  // save the downloaded font file
  const fontPath = `/fonts/${font.id}-${weight}-${style}.ttf`;

  runtimeFileResolver.writeFile(
      fontPath, Buffer.from(await blob.arrayBuffer()));

  return {filePath: fontPath, mimeType: 'font/ttf'};
}

async function resolveFontFile(fontDir: string, basename: string):
    Promise<{filePath: string; mimeType: string}|undefined> {
  const listing = fontDirListing(fontDir);
  const candidates: ReadonlyArray<[string, string]> =
      [['.ttf', 'font/ttf'], ['.otf', 'font/otf']];

  for (const [ext, mimeType] of candidates) {
    const file = listing.get(`${basename}${ext}`.toLowerCase());
    if (file) return {filePath: path.join(fontDir, file), mimeType};
  }

  // if font is not found then fetch the font and store it into VFS if in web
  // mode
  if (environmentResolver.mode == 'web') {
    return await fetchAndStoreFontFile(basename);
  }

  return undefined;
}

// Reads the font file a text() spec names from FONTPATH and returns it as a
// base64 data URL
export async function computeFontData(fontSpec: string):
    Promise<string|undefined> {
  if (fontDataCache.has(fontSpec)) return fontDataCache.get(fontSpec);

  let data = await loadFontData(fontSpec);
  fontDataCache.set(fontSpec, data);
  return data;
}

async function loadFontData(fontSpec: string): Promise<string|undefined> {
  // const fontDir = process.env.FONTPATH?.trim();
  const fontDir = environmentResolver.fontDir;
  if (!fontDir || !runtimeFileResolver.exists(fontDir)) {
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
