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
import {transformMesh} from '@gltf-transform/functions';
import {XMLParser} from 'fast-xml-parser';
import {unzipSync} from 'fflate';

import {euler2quat} from './math.ts';

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
  let modelPath = '3D/3dmodel.model';
  const relsData = files['_rels/.rels'];
  if (relsData) {
    const parser = new XMLParser({
      ignoreAttributes: false,
      attributeNamePrefix: '',
      removeNSPrefix: true,
    });
    const rels = parser.parse(new TextDecoder().decode(relsData));
    const startPart =
        asArray<{Type?: string; Target?: string; TargetMode?: string;}>(
            rels.Relationships?.Relationship)
            .find(
                rel => rel.Type ===
                    'http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel');
    if (startPart) {
      if (!startPart.Target)
        throw new Error('Invalid 3MF file: StartPart has no target');
      const base = 'https://3mf.invalid/';
      const target = new URL(startPart.Target, base);
      if (startPart.TargetMode === 'External' ||
          target.origin !== new URL(base).origin)
        throw new Error('External 3MF model relationships are not supported');
      modelPath = target.pathname.slice(1);
      if (!files[modelPath]) modelPath = decodeURIComponent(modelPath);
    }
  }
  const modelData = files[modelPath];
  if (!modelData) {
    throw new Error(`Invalid 3MF file: missing ${modelPath}`);
  }
  return parse3mfXml(new TextDecoder().decode(modelData));
}

interface ObjectRef3MF {
  objectID: string;
  transform: number[];
}

interface Vertex3MF {
  x?: number|string;
  y?: number|string;
  z?: number|string;
}

interface Triangle3MF {
  v1?: number|string;
  v2?: number|string;
  v3?: number|string;
}

interface Mesh3MF {
  vertices?: {vertex?: Vertex3MF|Array<Vertex3MF>;};
  triangles?: {triangle?: Triangle3MF|Array<Triangle3MF>;};
}

interface ComponentRef3MF {
  objectid?: number|string;
  transform?: string;
}

interface ObjectNode3MF {
  id?: number|string;
  name?: string;
  mesh?: Mesh3MF;
  components?: {component?: ComponentRef3MF|Array<ComponentRef3MF>;};
}

interface Model3MF {
  unit?: string;
  resources?: {object?: ObjectNode3MF|Array<ObjectNode3MF>;};
  build?: {item?: ComponentRef3MF|Array<ComponentRef3MF>;};
}

interface Object3MF {
  id: string;
  mesh: GLTFTransform.Mesh|null;
  children: Array<ObjectRef3MF>;
  name?: string;
}

const identityMatrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function asArray<T>(value?: T|Array<T>): Array<T> {
  if (value == null) return [];
  return Array.isArray(value) ? value : [value];
}

function toFiniteNumber(value: unknown): number|null {
  if (value == null || (typeof value === 'string' && value.trim() === '')) {
    return null;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function isVertexIndex(
    index: number|null, vertexCount: number): index is number {
  return index != null && Number.isInteger(index) && index >= 0 &&
      index < vertexCount;
}

function parseUnitMeters(unit?: string): number {
  switch ((unit ?? 'millimeter').toLowerCase()) {
    case 'micron':
      return 1e-6;
    case 'millimeter':
      return 1e-3;
    case 'centimeter':
      return 1e-2;
    case 'inch':
      return 0.0254;
    case 'foot':
      return 0.3048;
    case 'meter':
      return 1.0;
    default:
      return 1e-3;
  }
}

function parseTransform(value?: string): number[] {
  if (!value) return [...identityMatrix];
  const nums = value.trim().split(/\s+/).map(Number);
  if (nums.length !== 12 || nums.some((n) => !Number.isFinite(n))) {
    return [...identityMatrix];
  }

  // 3MF transform attribute stores affine matrix values as:
  // m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32.
  const m = [...identityMatrix];
  m[0] = nums[0];
  m[1] = nums[1];
  m[2] = nums[2];
  m[4] = nums[3];
  m[5] = nums[4];
  m[6] = nums[5];
  m[8] = nums[6];
  m[9] = nums[7];
  m[10] = nums[8];
  m[12] = nums[9];
  m[13] = nums[10];
  m[14] = nums[11];
  return m;
}

function parseMesh(
    mesh: Mesh3MF|undefined, doc: GLTFTransform.Document,
    buf: GLTFTransform.Buffer,
    defaultMaterial: GLTFTransform.Material): GLTFTransform.Mesh|null {
  if (!mesh) return null;

  // --- vertices ---
  const positions: number[] = [];
  for (const [index, vertex] of asArray(mesh.vertices?.vertex).entries()) {
    const x = toFiniteNumber(vertex.x);
    const y = toFiniteNumber(vertex.y);
    const z = toFiniteNumber(vertex.z);
    if (x == null || y == null || z == null) {
      throw new Error(
          `Invalid 3MF mesh: vertex ${index} has invalid coordinates`);
    }

    positions.push(x, y, z);
  }
  if (!positions.length) return null;

  // --- triangles ---
  const indices: number[] = [];
  const vertexCount = positions.length / 3;
  for (const [triangleIndex, tri] of asArray(mesh.triangles?.triangle)
           .entries()) {
    const v1 = toFiniteNumber(tri.v1);
    const v2 = toFiniteNumber(tri.v2);
    const v3 = toFiniteNumber(tri.v3);
    if (!isVertexIndex(v1, vertexCount) || !isVertexIndex(v2, vertexCount) ||
        !isVertexIndex(v3, vertexCount)) {
      throw new Error(`Invalid 3MF mesh: triangle ${
          triangleIndex} has an invalid vertex index`);
    }
    indices.push(v1, v2, v3);
  }
  if (!indices.length) return null;

  const posAcc = doc.createAccessor()
                     .setBuffer(buf)
                     .setType(GLTFTransform.Accessor.Type.VEC3)
                     .setArray(new Float32Array(positions));
  const indAcc = doc.createAccessor()
                     .setBuffer(buf)
                     .setType(GLTFTransform.Accessor.Type.SCALAR)
                     .setArray(new Uint32Array(indices));
  const primitive = doc.createPrimitive()
                        .setAttribute('POSITION', posAcc)
                        .setIndices(indAcc)
                        .setMaterial(defaultMaterial);
  return doc.createMesh().addPrimitive(primitive);
}

function parseComponentRefs(object: ObjectNode3MF): Array<ObjectRef3MF> {
  const refs: Array<ObjectRef3MF> = [];
  for (const component of asArray(object.components?.component)) {
    const objectID = component.objectid?.toString();
    if (!objectID) continue;
    refs.push({objectID, transform: parseTransform(component.transform)});
  }
  return refs;
}

function parseBuildRefs(model: Model3MF): Array<ObjectRef3MF> {
  const refs: Array<ObjectRef3MF> = [];
  for (const item of asArray(model.build?.item)) {
    const objectID = item.objectid?.toString();
    if (!objectID) continue;
    refs.push({objectID, transform: parseTransform(item.transform)});
  }
  return refs;
}

function matchesTransform(node: GLTFTransform.Node, matrix: number[]): boolean {
  const restored = node.getMatrix();
  for (let column = 0; column < 4; column++) {
    const start = column * 4;
    // Six-decimal 3MF rotations are not exactly orthogonal. Allow their print
    // and decomposition error relative to each basis column, without an
    // absolute floor that could hide shear on very small affine scales.
    const tolerance = 2e-6 *
        Math.max(
            Math.abs(matrix[start]), Math.abs(matrix[start + 1]),
            Math.abs(matrix[start + 2]));
    for (let row = 0; row < 4; row++) {
      if (!(Math.abs(restored[start + row] - matrix[start + row]) <=
            tolerance)) {
        return false;
      }
    }
  }
  return true;
}

function multiplyTransforms(parent: number[], child: number[]): number[] {
  const result = [...identityMatrix];
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 3; row++) {
      result[column * 4 + row] = parent[row] * child[column * 4] +
          parent[row + 4] * child[column * 4 + 1] +
          parent[row + 8] * child[column * 4 + 2] +
          (column === 3 ? parent[row + 12] : 0);
    }
  }
  return result;
}

function instantiateObjectNode(
    doc: GLTFTransform.Document, objects: Map<string, Object3MF>,
    ref: ObjectRef3MF, visiting: Set<string>): GLTFTransform.Node|null {
  const object = objects.get(ref.objectID);
  if (!object) return null;
  if (visiting.has(object.id)) return null;

  visiting.add(object.id);
  const node = doc.createNode(object.name);
  node.setMatrix(ref.transform as GLTFTransform.mat4);
  // glTF nodes store TRS, which cannot represent every 3MF affine matrix.
  // Allow print/decomposition roundoff, but bake shear into this instance's
  // mesh.
  const bake = !matchesTransform(node, ref.transform);
  if (bake) node.setMatrix(identityMatrix as GLTFTransform.mat4);
  if (object.mesh) {
    const mesh = bake ? object.mesh.clone() : object.mesh;
    if (bake) transformMesh(mesh, ref.transform as GLTFTransform.mat4);
    node.setMesh(mesh);
  }
  for (const childRef of object.children) {
    // An identity parent leaves its full affine transform to each child.
    const transformedRef = bake ? {
      ...childRef,
      transform: multiplyTransforms(ref.transform, childRef.transform)
    } :
                                  childRef;
    const childNode =
        instantiateObjectNode(doc, objects, transformedRef, visiting);
    if (childNode) node.addChild(childNode);
  }
  visiting.delete(object.id);
  return node;
}

function parse3mfXml(xml: string): GLTFTransform.Document {
  const parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: '',
    removeNSPrefix: true,
    trimValues: true,
    parseTagValue: false,
    parseAttributeValue: false
  });
  const parsed = parser.parse(xml) as {model?: Model3MF};
  const model = parsed.model;
  if (!model) {
    throw new Error('Invalid 3MF file: missing <model> root');
  }

  const doc = new GLTFTransform.Document();
  const buf = doc.createBuffer();
  // A default material is required so that gltfDocToManifold can process
  // the primitives (it calls getMaterial() on each primitive).
  const defaultMaterial = doc.createMaterial();
  const scene = doc.createScene();
  const unitToMeters = parseUnitMeters(model.unit);
  const objects = new Map<string, Object3MF>();

  // Keep vertex data untouched and normalize once at the root:
  // 3MF (unit/+Z-up) -> glTF (m/+Y-up).
  const rootTransform = doc.createNode('3mf-import-root');
  rootTransform.setRotation(euler2quat([-90, 0, 0]));
  rootTransform.setScale([unitToMeters, unitToMeters, unitToMeters]);
  scene.addChild(rootTransform);

  // Parse all resource objects first (mesh objects and component objects).
  for (const object of asArray(model.resources?.object)) {
    const id = object.id?.toString();
    if (!id) continue;

    objects.set(id, {
      id,
      name: object.name,
      mesh: parseMesh(object.mesh, doc, buf, defaultMaterial),
      children: parseComponentRefs(object)
    });
  }

  // Build roots from the <build> list and instantiate object hierarchies.
  const buildRefs = parseBuildRefs(model);
  for (const ref of buildRefs) {
    const node = instantiateObjectNode(doc, objects, ref, new Set());
    if (node) rootTransform.addChild(node);
  }

  // Fallback for malformed 3MF files missing <build>: show all mesh objects.
  if (!buildRefs.length) {
    for (const object of objects.values()) {
      if (!object.mesh) continue;
      rootTransform.addChild(doc.createNode(object.name).setMesh(object.mesh));
    }
  }

  return doc;
}
