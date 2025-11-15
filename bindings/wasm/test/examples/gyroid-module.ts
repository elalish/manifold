// Recreates Modular Gyroid Puzzle by Emmett Lalish:
// https://www.thingiverse.com/thing:25477. This sample demonstrates the
// use of a Signed Distance Function (SDF) to create smooth, complex
// manifolds.

import {Box, getGLTFNodes, GLTFNode, Manifold} from 'manifold-3d/manifoldCAD';

// number of modules along pyramid edge (use 1 for print orientation)
const m = 4;
// module size
const size = 20;
// SDF resolution
const n = 20;

const pi = 3.14159;

function gyroid(p) {
  const x = p[0] - pi / 4;
  const y = p[1] - pi / 4;
  const z = p[2] - pi / 4;
  return Math.cos(x) * Math.sin(y) + Math.cos(y) * Math.sin(z) +
      Math.cos(z) * Math.sin(x);
}

function gyroidOffset(level) {
  const period = 2 * pi;
  const box = {
    min: [-period, -period, -period],
    max: [period, period, period]
  } as Box;
  return Manifold.levelSet(gyroid, box, period / n, level).scale(size / period);
};

function rhombicDodecahedron() {
  const box = Manifold.cube([1, 1, 2], true).scale(size * Math.sqrt(2));
  const result = box.rotate([90, 45, 0]).intersect(box.rotate([90, 45, 90]));
  return result.intersect(box.rotate([0, 0, 45]));
}

const gyroidModule = rhombicDodecahedron()
                         .intersect(gyroidOffset(-0.4))
                         .subtract(gyroidOffset(0.4));

if (m > 1) {
  for (let i = 0; i < m; ++i) {
    for (let j = i; j < m; ++j) {
      for (let k = j; k < m; ++k) {
        const node = new GLTFNode();
        node.manifold = gyroidModule;
        node.translation = [(k + i - j) * size, (k - i) * size, (-j) * size];
        node.material = {
          baseColorFactor: [(k + i - j + 1) / m, (k - i + 1) / m, (j + 1) / m]
        };
      }
    }
  }
}

// Get a list of GLTF nodes that have been created in this model.  This
// function only works at the top level; in a library it will always return
// an empty array, and nodes created in libraries will not be included in
// the result. This is intentional; libraries must not create geometry as
// a side effect.
const nodes = getGLTFNodes();
export default nodes;