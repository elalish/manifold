import {Accessor, Document} from '@gltf-transform/core';
import {expect, test} from 'vitest';

import {readMesh} from './gltf-io.ts';
import {EXTManifold} from './manifold-gltf.ts';

function fixture() {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const positions =
      doc.createAccessor()
          .setBuffer(buffer)
          .setType(Accessor.Type.VEC3)
          .setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
  const indices = doc.createAccessor()
                      .setBuffer(buffer)
                      .setType(Accessor.Type.SCALAR)
                      .setArray(new Uint32Array([0, 1, 2]));
  const primitive = doc.createPrimitive()
                        .setAttribute('POSITION', positions)
                        .setIndices(indices);
  const mesh = doc.createMesh().addPrimitive(primitive);
  return {doc, buffer, positions, indices, primitive, mesh};
}

test.each([false, true])(
    'zero-fills missing and skipped properties (extension %s)', extension => {
      const {doc, mesh, indices} = fixture();
      if (extension) {
        mesh.setExtension(
            'EXT_mesh_manifold',
            doc.createExtension(EXTManifold)
                .createManifoldPrimitive()
                .setIndices(indices)
                .setRunIndex([0, 3]));
      }
      const result = readMesh(mesh, ['POSITION', 'SKIP_2', 'NORMAL'])!;
      expect(result.mesh.numProp).toBe(8);
      expect(Array.from(result.mesh.vertProperties)).toEqual([
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      ]);
    });

test(
    'keeps complete vertex strides across primitives with different attributes',
    () => {
      const {doc, buffer, mesh, positions, indices} = fixture();
      const normals =
          doc.createAccessor()
              .setBuffer(buffer)
              .setType(Accessor.Type.VEC3)
              .setArray(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]));
      mesh.addPrimitive(doc.createPrimitive()
                            .setAttribute('POSITION', positions)
                            .setAttribute('NORMAL', normals)
                            .setIndices(indices));
      const result = readMesh(mesh)!;
      expect(result.mesh.numProp).toBe(6);
      expect(Array.from(result.mesh.triVerts)).toEqual([0, 1, 2, 3, 4, 5]);
      expect(Array.from(result.mesh.vertProperties)).toEqual([
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1,
      ]);
    });
