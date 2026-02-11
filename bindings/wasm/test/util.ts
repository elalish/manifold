
import {distanceVec3} from '../lib/math.ts';
import type {Mesh, Vec3} from '../manifold.d.ts';

export function inVec3Array(
    haystack: Array<Vec3>, needle: Vec3, margin: number = 1.0e-6): boolean {
  return !!haystack.find(x => distanceVec3(needle, x) <= margin);
}

export function equalsVec3Array(
    a: Array<Vec3>, b: Array<Vec3>, margin: number = 1.0e-6): boolean {
  if (a.length != b.length) return false;

  for (const pt of a) {
    if (!inVec3Array(b, pt, margin)) return false;
  }

  for (const pt of b) {
    if (!inVec3Array(a, pt, margin)) return false;
  }

  return true;
}

/**
 * Turn a `Mesh` into a point cloud.
 */
export function meshToVec3Array(mesh: Mesh): Array<Vec3> {
  const pts: Array<Vec3> = [];
  for (let i = 0; i < mesh.numVert; i++) {
    pts.push(mesh.position(i) as unknown as Vec3);
  }
  return pts;
}