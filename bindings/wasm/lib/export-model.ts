// Copyright 2024-25 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Export models out of manifoldCAD.
 *
 * ManifoldCAD uses [gltf-transform](https://gltf-transform.dev/) internally to
 * represent scenes. Exporters must accept in-memory gltf-transform Documents.
 *
 * Exporters and importers can each accept one or more type of file.
 * Each format must have a defined extension and mime type.  In the case of
 * duplicates, order is not guaranteed. While an implementation may support both
 * import and export of a format, this is not required.  At the time of writing,
 * manifoldCAD supports both import and export of glTF, but only exports 3mf.
 *
 * @packageDocumentation
 */

import * as GLTFTransform from '@gltf-transform/core';

import {UnsupportedFormatError} from './error.ts';
import * as export3MF from './export-3mf.ts';
import * as gltfIO from './gltf-io.ts';
import {findExtension, findMimeType, isNode} from './util.ts';

/**
 * @inline
 */
interface Format {
  extension: string;
  mimetype: string;
}

export interface Exporter {
  exportFormats: Array<Format>;
  toArrayBuffer: (doc: GLTFTransform.Document) => Promise<ArrayBuffer>;
}

/**
 * @inline
 */
interface ExportOptions {
  /**
   * Use `mimetype` to determine the format of the imported model, rather than
   * inferring it.
   */
  mimetype?: string;
}

const exporters: Array<Exporter> = [];
register(gltfIO);
register(export3MF);

function getFormat(identifier: string): Format {
  const formats = exporters.flatMap(ex => ex.exportFormats);
  const format = (findMimeType(identifier, formats) ??
                  findExtension(identifier, formats)) as Format;
  if (!format) throw new UnsupportedFormatError(identifier, formats);
  return format;
}

function getExporter(identifier: Format|string) {
  const format =
      typeof identifier === 'string' ? getFormat(identifier) : identifier;
  return exporters.find(ex => ex.exportFormats.includes(format))!;
}

/**
 * Returns true if a given extension or mimetype can be exported.
 *
 * @param filetype
 * @param throwOnFailure If true, throw an `UnsupportedFormatException` rather
 *     than return false.
 * @group Management Functions
 */
export function supports(filetype: string, throwOnFailure: false): boolean {
  if (throwOnFailure) return !!getFormat(filetype);

  try {
    return !!getFormat(filetype);
  } catch (e) {
    return false;
  }
}

/**
 * Register an exporter.
 *
 * Supported formats will be inferred.
 * @group Management Functions
 * @param exporter
 */
export function register(exporter: Exporter) {
  exporters.push(exporter);
}

/**
 * Convert an in-memory GLTF document to a binary Blob.
 *
 * @param doc The GLTF document.
 * @param identifier The target format extension or mimetype.
 * @returns A URL encoded blob.
 * @group Low Level Functions
 */
export async function toBlob(
    doc: GLTFTransform.Document, identifier: string): Promise<Blob> {
  const format = getFormat(identifier);
  const buffer = await getExporter(identifier).toArrayBuffer(doc);
  return new Blob([buffer], {type: format.mimetype});
}

/**
 * Convert an in-memory GLTF document to an ArrayBuffer.
 *
 * @param doc The GLTF document.
 * @param identifier The target format extension or mimetype.
 * @returns A URL encoded blob.
 * @group Low Level Functions
 */
export async function toArrayBuffer(
    doc: GLTFTransform.Document, identifier: string): Promise<ArrayBuffer> {
  return await getExporter(identifier).toArrayBuffer(doc);
}

/**
 * Write a model to disk.
 * @group Low Level Functions
 */
export async function writeFile(
    filename: string, doc: GLTFTransform.Document,
    options: ExportOptions = {}) {
  if (!isNode()) {
    throw new Error('Must have a filesystem to write files.');
  }
  const fs = await import('node:fs/promises');

  const exporter = getExporter(options.mimetype ?? filename);
  const buffer = await exporter.toArrayBuffer(doc);
  return await fs.writeFile(filename, new Uint8Array(buffer));
}
