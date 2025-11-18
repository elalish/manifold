// Copyright 2023-2025 The Manifold Authors.
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

import * as GLTFTransform from '@gltf-transform/core';
import {KHRONOS_EXTENSIONS} from '@gltf-transform/extensions';

import {setupIO} from './gltf-io.ts';

const binaryFormat = {
  extension: 'glb',
  mimetype: 'model/gltf-binary'
};

const jsonFormat = {
  extension: 'gltf',
  mimetype: 'model/gltf+json'
};

export const supportedFormats = [binaryFormat, jsonFormat];

/**
 * Convert a GLTF-Transform document to a blob.
 *
 * @param doc The GLTF document to convert.
 * @returns A blob containing the converted model.
 */
export async function asBlob(doc: GLTFTransform.Document) {
  const io = setupIO(new GLTFTransform.WebIO());
  io.registerExtensions(KHRONOS_EXTENSIONS);

  const glb = await io.writeBinary(doc);
  return new Blob(
      [glb as Uint8Array<ArrayBuffer>], {type: binaryFormat.mimetype});
}
