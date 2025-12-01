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
 * @packageDocumentation
 */

import * as GLTFTransform from '@gltf-transform/core';

import * as export3MF from './export-3mf.ts';
import * as gltfIO from './gltf-io.ts';
import {isNode} from './util.ts';

export interface Format {
  extension: string;
  mimetype: string;
}

export interface Exporter {
  exportFormats: Array<Format>;
  toBlob: (doc: GLTFTransform.Document) => Promise<Blob>;
  toArrayBuffer:
      (doc: GLTFTransform.Document) => Promise<Uint8Array<ArrayBufferLike>>;

  writeFile?: (filename: string, doc: GLTFTransform.Document) => Promise<void>;
}

/**
 * @inline
 */
export interface ExportOptions {
  mimetype?: string;
}

const exporters: Array<Exporter> = [gltfIO, export3MF];

export function getExporterByExtension(extension: string) {
  const hasExtension =
      (format: Format) => [format.extension, `.${format.extension}`].includes(
          extension);
  const exporter = exporters.find(ex => ex.exportFormats.find(hasExtension));

  if (!exporter) {
    const extensionList =
        exporters.flatMap(ex => ex.exportFormats.map(f => f.extension))
            .map(ext => `\`.${ext}\``)
            .reduceRight(
                (prev, cur, index) => cur + (index ? ', or ' : ', ') + prev);
    throw new Error(
        `Cannot import \`${extension}\`.  ` +
        `Format must be one of ${extensionList}`);
  }
  return exporter;
}

export function getExporterByMimeType(mimetype: string) {
  const hasMimetype = (format: Format) => format.mimetype === mimetype;
  const exporter = exporters.find(ex => ex.exportFormats.find(hasMimetype));

  if (!exporter) {
    const mimetypeList =
        exporters.flatMap(ex => ex.exportFormats.map(f => f.mimetype))
            .reduceRight(
                (prev, cur, index) => cur + (index ? ', or ' : ', ') + prev);
    throw new Error(
        `Cannot export \`${mimetype}\`.  ` +
        `Format must be one of ${mimetypeList}`);
  }
  return exporter;
}

/**
 * Convert an in-memory GLTF document to a binary blob.
 *
 * @param doc The GLTF document.
 * @param extension The target file extension.
 * @returns A URL encoded blob.
 */
export async function toBlob(
    doc: GLTFTransform.Document, extension: string): Promise<Blob> {
  return await getExporterByExtension(extension).toBlob(doc);
}

/**
 * Write a model to disk.
 *
 * If the matching exporter has a `writeFile()` method, delegate to it.
 */
export async function writeFile(
    filename: string, doc: GLTFTransform.Document,
    options: ExportOptions = {}) {
  if (!isNode()) {
    throw new Error('Must have a filesystem to write files.');
  }
  const fs = await import('node:fs/promises');

  const [ext] = filename.match(/(\.[^\.]+)$/)!;
  const exporter = options.mimetype ? getExporterByMimeType(options.mimetype) :
                                      getExporterByExtension(ext);

  if (typeof exporter.writeFile === 'function' && isNode()) {
    return exporter.writeFile(filename, doc)
  }

  const blob = await exporter.toBlob(doc);
  return await fs.writeFile(filename, await blob.bytes());
}
