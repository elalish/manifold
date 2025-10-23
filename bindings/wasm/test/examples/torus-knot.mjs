// Creates a classic torus knot, defined as a string wrapping periodically
// around the surface of an imaginary donut. If p and q have a common
// factor then you will get multiple separate, interwoven knots. This is
// an example of using the warp() method, thus avoiding any direct
// handling of triangles.

// The number of times the thread passes through the donut hole.
const p = 1;
// The number of times the thread circles the donut.
const q = 3;
// Radius of the interior of the imaginary donut.
const majorRadius = 25;
// Radius of the small cross-section of the imaginary donut.
const minorRadius = 10;
// Radius of the small cross-section of the actual object.
const threadRadius = 3.75;
// Number of linear segments making up the threadRadius circle. Default is
// getCircularSegments(threadRadius).
const circularSegments = -1;
// Number of segments along the length of the knot. Default makes roughly
// square facets.
const linearSegments = -1;

// These default values recreate Matlab Knot by Emmett Lalish:
// https://www.thingiverse.com/thing:7080

import {vec3} from 'gl-matrix';

function gcd(a, b) {
  return b == 0 ? a : gcd(b, a % b);
}

const kLoops = gcd(p, q);
const pk = p / kLoops;
const qk = q / kLoops;
const n = circularSegments > 2 ? circularSegments :
                                 getCircularSegments(threadRadius);
const m = linearSegments > 2 ? linearSegments :
                               n * qk * majorRadius / threadRadius;

const offset = 2
const circle = CrossSection.circle(1, n).translate([offset, 0]);

const func = (v) => {
  const psi = qk * Math.atan2(v[0], v[1]);
  const theta = psi * pk / qk;
  const x1 = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
  const phi = Math.atan2(x1 - offset, v[2]);
  vec3.set(
      v, threadRadius * Math.cos(phi), 0, threadRadius * Math.sin(phi));
  const center = vec3.fromValues(0, 0, 0);
  const r = majorRadius + minorRadius * Math.cos(theta);
  vec3.rotateX(v, v, center, -Math.atan2(pk * minorRadius, qk * r));
  v[0] += minorRadius;
  vec3.rotateY(v, v, center, theta);
  v[0] += majorRadius;
  vec3.rotateZ(v, v, center, psi);
};

const result = Manifold.revolve(circle, m).warp(func);
export default result;
