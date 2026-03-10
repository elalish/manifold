// Copyright 2026 The Manifold Authors.
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
 * Import 3MF models into manifoldCAD.
 *
 * @packageDocumentation
 * @group ManifoldCAD
 * @category Input/Output
 */

import * as GLTFTransform from '@gltf-transform/core';
import {unzipSync} from 'fflate';

export const importFormats = [{extension: '3mf', mimetype: 'model/3mf'}];

/**
 * Parse a 3MF ArrayBuffer into a gltf-transform Document.
 *
 * Note: 3MF files store geometry in millimetres with +Z up, while glTF uses
 * metres with +Y up. This importer normalizes geometry to glTF conventions so
 * the shared import pipeline can apply `importTransform()` consistently across
 * all formats.
 */
export async function fromArrayBuffer(buffer: ArrayBuffer):
    Promise<GLTFTransform.Document> {
  const files = unzipSync(new Uint8Array(buffer));
  const modelData = files['3D/3dmodel.model'];
  if (!modelData) {
    throw new Error('Invalid 3MF file: missing 3D/3dmodel.model');
  }
  return parse3mfXml(new TextDecoder().decode(modelData));
}

function parse3mfXml(xml: string): GLTFTransform.Document {
  const doc = new GLTFTransform.Document();
  const buf = doc.createBuffer();
  // A default material is required so that gltfDocToManifold can process
  // the primitives (it calls getMaterial() on each primitive).
  const defaultMaterial = doc.createMaterial();

  const scene = doc.createScene();

  // Find every <object> that has a <mesh> child and create one
  // gltf-transform node per mesh.  Component objects (those with only
  // <components>) are skipped; their transforms are all identity for
  // geometry exported by manifoldCAD, so no information is lost.
  const objectRe = /<object\b[^>]*>([\s\S]*?)<\/object>/g;
  let objMatch: RegExpExecArray|null;

  while ((objMatch = objectRe.exec(xml)) !== null) {
    const body = objMatch[1];
    const meshBlock = body.match(/<mesh>([\s\S]*?)<\/mesh>/);
    if (!meshBlock) continue;
    const meshXml = meshBlock[1];

    // --- vertices ---
    const positions: number[] = [];
    const vertRe = /<vertex\b[^>]*>/g;
    let vMatch: RegExpExecArray|null;
    while ((vMatch = vertRe.exec(meshXml)) !== null) {
      const tag = vMatch[0];
      const x = tag.match(/\bx="([^"]+)"/)?.[1];
      const y = tag.match(/\by="([^"]+)"/)?.[1];
      const z = tag.match(/\bz="([^"]+)"/)?.[1];
      if (x && y && z) {
        // 3MF is mm/+Z-up. Convert to glTF m/+Y-up by applying:
        // scale = 1/1000 and rotation = -90 degrees around X.
        const xf = +x / 1000;
        const yf = +y / 1000;
        const zf = +z / 1000;
        positions.push(xf, zf, -yf);
      }
    }
    if (!positions.length) continue;

    // --- triangles ---
    const indices: number[] = [];
    const triRe = /<triangle\b[^>]*>/g;
    let tMatch: RegExpExecArray|null;
    while ((tMatch = triRe.exec(meshXml)) !== null) {
      const tag = tMatch[0];
      const v1 = tag.match(/\bv1="(\d+)"/)?.[1];
      const v2 = tag.match(/\bv2="(\d+)"/)?.[1];
      const v3 = tag.match(/\bv3="(\d+)"/)?.[1];
      if (v1 && v2 && v3) indices.push(+v1, +v2, +v3);
    }

    const posAcc = doc.createAccessor()
                       .setBuffer(buf)
                       .setType(GLTFTransform.Accessor.Type.VEC3)
                       .setArray(new Float32Array(positions));
    const indAcc = doc.createAccessor()
                       .setBuffer(buf)
                       .setType(GLTFTransform.Accessor.Type.SCALAR)
                       .setArray(new Uint32Array(indices));
    const prim = doc.createPrimitive()
                     .setAttribute('POSITION', posAcc)
                     .setIndices(indAcc)
                     .setMaterial(defaultMaterial);
    scene.addChild(
        doc.createNode().setMesh(doc.createMesh().addPrimitive(prim)));
  }

  return doc;
}
