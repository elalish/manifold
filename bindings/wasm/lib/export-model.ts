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
import * as exportGLTF from './export-gltf.ts';

export interface Format {
  extension: string;
  mimetype: string;
}

export interface Exporter {
  supportedFormats: Array<Format>;
  asBlob: (doc: GLTFTransform.Document) => Promise<Blob>;
}

const exporters: Array<Exporter> = [exportGLTF, export3MF];

export function getExporterByExtension(extension: string) {
  const hasExtension =
      (format: Format) => [format.extension, `.${format.extension}`].includes(
          extension);
  const exporter = exporters.find(ex => ex.supportedFormats.find(hasExtension));

  if (!exporter) {
    const extensionList =
        exporters
            .map(exporter => exporter.supportedFormats.map(f => f.extension))
            .reduce((acc, cur) => ([...acc, ...cur]))
            .map(ext => `\`.${ext}\``)
            .reduceRight(
                (prev, cur, index) => cur + (index ? ', or ' : ', ') + prev);
    throw new Error(
        `Cannot import \`${extension}\`.  ` +
        `Format must be one of ${extensionList}`);
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
export async function asBlob(
    doc: GLTFTransform.Document, extension: string): Promise<Blob> {
  return await getExporterByExtension(extension).asBlob(doc);
}
