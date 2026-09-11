import path from 'path';

import type {SurfaceImage} from '../runtime.js';

import {runtimeCanvasResolver, runtimeFileResolver} from './host.js';

// Decodes an image to raw pixels so the mesh builder only sees luminance
function decodeImagePixels(filePath: string): SurfaceImage|undefined {
  const img = runtimeCanvasResolver.image();
  // A Buffer src decodes in place; the typings only admit a string
  (img as {src: unknown}).src = runtimeFileResolver.readBinary(filePath);
  const {width, height} = img;
  if (!width || !height) return undefined;

  const canvas = runtimeCanvasResolver.createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const {data} = ctx.getImageData(0, 0, width, height);

  // Drop the alpha channel: surface() only reads luminance
  const rgb = Buffer.allocUnsafe(width * height * 3);
  for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) {
    rgb[j] = data[i]!;
    rgb[j + 1] = data[i + 1]!;
    rgb[j + 2] = data[i + 2]!;
  }
  return {width, height, rgb: rgb.toString('base64')};
}

// Reads a `surface()` source file, decoding PNGs to pixels and returning any
// other file as its raw text. Warns and returns undefined when unreadable.
export function computeSurfaceData(filePath: string): string|SurfaceImage|
    undefined {
  if (!runtimeFileResolver.exists(filePath)) {
    console.warn(`Warning: surface("${filePath}"): can't open file "${
        filePath}", ignoring.`);
    return undefined;
  }

  // OpenSCAD treats only PNG as an image; everything else is a text matrix
  if (path.extname(filePath).toLowerCase() === '.png') {
    const pixels = decodeImagePixels(filePath);
    if (!pixels) {
      console.warn(`Warning: surface("${filePath}"): can't decode image "${
          filePath}", ignoring.`);
    }
    return pixels;
  }

  const result = runtimeFileResolver.readText(filePath);

  return result as string;
}
