// This example demonstrates how symbolic perturbation correctly creates
// holes even though the subtracted objects are exactly coplanar.
import {vec2} from 'gl-matrix';

function fractal(holes, hole, w, position, depth, maxDepth) {
  w /= 3;
  holes.push(
      hole.scale([w, w, 1.0]).translate([position[0], position[1], 0.0]));
  if (depth == maxDepth) return;
  const offsets = [
    vec2.fromValues(-w, -w), vec2.fromValues(-w, 0.0),
    vec2.fromValues(-w, w), vec2.fromValues(0.0, w),
    vec2.fromValues(w, w), vec2.fromValues(w, 0.0),
    vec2.fromValues(w, -w), vec2.fromValues(0.0, -w)
  ];
  for (let offset of offsets)
    fractal(
        holes, hole, w, vec2.add(offset, position, offset), depth + 1,
        maxDepth);
}

function mengerSponge(n) {
  let result = Manifold.cube([1, 1, 1], true);
  const holes = [];
  fractal(holes, result, 1.0, [0.0, 0.0], 1, n);

  const hole = Manifold.compose(holes);

  result = Manifold.difference([
    result,
    hole,
    hole.rotate([90, 0, 0]),
    hole.rotate([0, 90, 0]),
  ]);
  return result;
}

const posColors = (newProp, pos) => {
  for (let i = 0; i < 3; ++i) {
    newProp[i] = (1 - pos[i]) / 2;
  }
};

const result = mengerSponge(3)
                    .trimByPlane([1, 1, 1], 0)
                    .setProperties(3, posColors)
                    .scale(100);

const node = new GLTFNode();
node.manifold = result;
node.material = {baseColorFactor: [1, 1, 1], attributes: ['COLOR_0']};
