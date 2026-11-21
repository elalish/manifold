import {Accessor, Document} from '@gltf-transform/core';
import {expect, test} from 'vitest';

import {disposeMesh} from './gltf-io.ts';
import {EXTManifold} from './manifold-gltf.ts';

function fixture() {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const positions =
      doc.createAccessor()
          .setBuffer(buffer)
          .setType(Accessor.Type.VEC3)
          .setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
  const makeIndices = () => doc.createAccessor()
                                .setBuffer(buffer)
                                .setType(Accessor.Type.SCALAR)
                                .setArray(new Uint32Array([0, 1, 2]));
  const first = doc.createPrimitive()
                    .setAttribute('POSITION', positions)
                    .setIndices(makeIndices());
  const second = doc.createPrimitive()
                     .setAttribute('POSITION', positions)
                     .setIndices(makeIndices());
  const firstMesh = doc.createMesh().addPrimitive(first);
  const secondMesh = doc.createMesh().addPrimitive(second);
  return {doc, positions, first, second, firstMesh, secondMesh};
}

test('preserves shared accessors until their last mesh is disposed', () => {
  const {doc, positions, first, second, firstMesh, secondMesh} = fixture();
  const uniqueIndices = first.getIndices()!;
  disposeMesh(firstMesh);
  expect(second.getAttribute('POSITION')).toBe(positions);
  expect(positions.isDisposed()).toBe(false);
  expect(first.isDisposed()).toBe(true);
  expect(uniqueIndices.isDisposed()).toBe(true);
  expect(doc.getRoot().listAccessors()).toHaveLength(2);
  disposeMesh(secondMesh);
  expect(positions.isDisposed()).toBe(true);
  expect(doc.getRoot().listAccessors()).toHaveLength(0);
});

test('preserves primitives referenced by another mesh', () => {
  const {positions, first, firstMesh, secondMesh} = fixture();
  secondMesh.addPrimitive(first);
  disposeMesh(firstMesh);
  expect(first.isDisposed()).toBe(false);
  expect(first.getAttribute('POSITION')).toBe(positions);
  expect(first.getIndices()).not.toBeNull();
  disposeMesh(secondMesh);
  expect(first.isDisposed()).toBe(true);
  expect(positions.isDisposed()).toBe(true);
});

test('disposes orphaned extension objects without deleting shared data', () => {
  const {doc, firstMesh, second, secondMesh} = fixture();
  const indices = second.getIndices()!;
  const extension = doc.createExtension(EXTManifold)
                        .createManifoldPrimitive()
                        .setIndices(indices)
                        .setRunIndex([0, 3]);
  firstMesh.setExtension('EXT_mesh_manifold', extension);
  disposeMesh(firstMesh);
  expect(extension.isDisposed()).toBe(true);
  expect(second.getIndices()).toBe(indices);
  expect(indices.isDisposed()).toBe(false);
  disposeMesh(secondMesh);
  expect(indices.isDisposed()).toBe(true);
});
