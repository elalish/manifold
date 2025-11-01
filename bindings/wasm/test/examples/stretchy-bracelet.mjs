// Recreates Stretchy Bracelet by Emmett Lalish:
// https://www.thingiverse.com/thing:13505

import {Manifold} from 'manifold-3d/manifoldCAD';
import {vec2} from 'gl-matrix';

function base(
    width, radius, decorRadius, twistRadius, nDecor, innerRadius,
    outerRadius, cut, nCut, nDivision) {
  let b = Manifold.cylinder(width, radius + twistRadius / 2);
  const circle = [];
  const dPhiDeg = 180 / nDivision;
  for (let i = 0; i < 2 * nDivision; ++i) {
    circle.push([
      decorRadius * Math.cos(dPhiDeg * i * Math.PI / 180) + twistRadius,
      decorRadius * Math.sin(dPhiDeg * i * Math.PI / 180)
    ]);
  }
  let decor = Manifold.extrude(circle, width, nDivision, 180)
                  .scale([1, 0.5, 1])
                  .translate([0, radius, 0]);
  for (let i = 0; i < nDecor; i++)
    b = b.add(decor.rotate([0, 0, (360.0 / nDecor) * i]));
  const stretch = [];
  const dPhiRad = 2 * Math.PI / nCut;

  const o = vec2.fromValues(0, 0);
  const p0 = vec2.fromValues(outerRadius, 0);
  const p1 = vec2.fromValues(innerRadius, -cut);
  const p2 = vec2.fromValues(innerRadius, cut);
  for (let i = 0; i < nCut; ++i) {
    stretch.push(vec2.rotate([0, 0], p0, o, dPhiRad * i));
    stretch.push(vec2.rotate([0, 0], p1, o, dPhiRad * i));
    stretch.push(vec2.rotate([0, 0], p2, o, dPhiRad * i));
    stretch.push(vec2.rotate([0, 0], p0, o, dPhiRad * i));
  }
  const result =
      Manifold.intersection(Manifold.extrude(stretch, width), b);
  return result;
}

function stretchyBracelet(
    radius = 30, height = 8, width = 15, thickness = 0.4, nDecor = 20,
    nCut = 27, nDivision = 30) {
  const twistRadius = Math.PI * radius / nDecor;
  const decorRadius = twistRadius * 1.5;
  const outerRadius = radius + (decorRadius + twistRadius) * 0.5;
  const innerRadius = outerRadius - height;
  const cut = 0.5 * (Math.PI * 2 * innerRadius / nCut - thickness);
  const adjThickness = 0.5 * thickness * height / cut;

  return Manifold.difference(
      base(
          width, radius, decorRadius, twistRadius, nDecor,
          innerRadius + thickness, outerRadius + adjThickness,
          cut - adjThickness, nCut, nDivision),
      base(
          width, radius - thickness, decorRadius, twistRadius, nDecor,
          innerRadius, outerRadius + 3 * adjThickness, cut, nCut,
          nDivision));
}

const result = stretchyBracelet();
export default result;