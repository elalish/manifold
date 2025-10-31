// Smooth, complex manifolds can be created using the warp() function.
// This example recreates the Exploitable Heart by Emmett Lalish:
// https://www.thingiverse.com/thing:6190
// It also demonstrates the use of setMorph to animate a warping function.

import {Manifold, GLTFNode, setMorphStart} from 'manifold-3d/manifoldCAD';


const func = (v) => {
  const x2 = v[0] * v[0];
  const y2 = v[1] * v[1];
  const z = v[2];
  const z2 = z * z;
  const a = x2 + 9 / 4 * y2 + z2;
  const b = z * z2 * (x2 + 9 / 80 * y2);
  const a2 = a * a;
  const a3 = a * a2;

  const step = (r) => {
    const r2 = r * r;
    const r4 = r2 * r2;
    // Taubin's function: https://mathworld.wolfram.com/HeartSurface.html
    const f = a3 * r4 * r2 - b * r4 * r - 3 * a2 * r4 + 3 * a * r2 - 1;
    // Derivative
    const df =
        6 * a3 * r4 * r - 5 * b * r4 - 12 * a2 * r2 * r + 6 * a * r;
    return f / df;
  };
  // Newton's method for root finding
  let r = 1.5;
  let dr = 1;
  while (Math.abs(dr) > 0.0001) {
    dr = step(r);
    r -= dr;
  }
  // Update radius
  v[0] *= r;
  v[1] *= r;
  v[2] *= r;
};

const ball = Manifold.sphere(1, 200);

const heart = ball.warp(func);
const box = heart.boundingBox();
const scale = 100 / (box.max[0] - box.min[0]);

setMorphStart(ball, func);
const node = new GLTFNode();
node.manifold = ball;
node.scale = [scale, scale, scale];

globalDefaults.animationLength = 5;  // seconds
globalDefaults.animationMode = 'ping-pong';

export default node;