// Demonstrate how to build a library for manifoldCAD
//
// This example is based on the involute gear modules from the Gear Bearing by
// Emmett Lalish: https://www.thingiverse.com/thing:53451

import {CrossSection, getCircularSegments, Vec2} from 'manifold-3d/manifoldCAD';

const {sin, cos, tan, atan2, sqrt, pow, max, PI} = Math;

/**
 * Create a 3D spur or helical gear.
 * @param {number} teeth - number of teeth
 * @param {number} height - height/thickness of the gear
 * @param {object} params - gear parameters: should match for meshing gears
 * @param {number} params.circularPitch - tooth spacing - default 10mm
 * @param {number} params.pressureDeg - contact angle - default 20 degrees
 * @param {number} params.depthRatio - tooth length / tooth spacing - default
 *     0.75
 * @param {number} params.clearance - gap between meshing teeth - default 0mm
 * @param {number} params.n - number of points per curve - default is
 * calculated by the static Quality defaults according to the circularPitch
 * @param {number} params.twist - helical pitch twist / height - default 0
 * @returns {Manifold} - the spur gear manifold
 */
export function spurGear(teeth: number, height: number, {
  circularPitch = 10,
  pressureDeg = 20,
  depthRatio = 0.75,
  clearance = 0,
  n = getCircularSegments(circularPitch / 6),
  twist = 0
} = {}) {
  const nDiv = n * twist * height / circularPitch;
  const twistDeg = twist * height * 360 / (teeth * circularPitch);
  return gear2D(teeth, {circularPitch, pressureDeg, depthRatio, clearance, n})
      .extrude(height, nDiv, twistDeg);
}

/* Create a 2D CrossSection of an involute gear.
 * @param {number} teeth - number of teeth
 * @param {object} params - gear parameters: should match for meshing gears
 * @param {number} params.circularPitch - tooth spacing (mm)
 * @param {number} params.pressureDeg - contact angle
 * @param {number} params.depthRatio - tooth length / tooth spacing
 * @param {number} params.clearance - gap between meshing teeth
 * @param {number} params.n - number of points per curve
 * @returns {CrossSection} - the spur gear cross section
 */
export function gear2D(teeth: number, {
  circularPitch = 10,
  pressureDeg = 20,
  depthRatio = 0.75,
  clearance = 0,
  n = getCircularSegments(circularPitch / 6),
} = {}) {
  const pressureRad = pressureDeg * PI / 180;
  const pitchRadius = teeth * circularPitch / (2 * PI);
  const baseRadius = pitchRadius * cos(pressureRad);
  const depth = circularPitch / (2 * tan(pressureRad));
  const outerRadius = clearance < 0 ? pitchRadius + depth / 2 - clearance :
                                      pitchRadius + depth / 2;
  const rootRadius1 = pitchRadius - depth / 2 - clearance / 2;
  const rootRadius =
      (clearance < 0 && rootRadius1 < baseRadius) ? baseRadius : rootRadius1;

  const backlashDeg = clearance / (pitchRadius * cos(pressureRad)) * 180 / PI;
  const halfThickDeg = 90 / teeth - backlashDeg / 2;

  const pitchPoint =
      involute(baseRadius, involuteIntersectRad(baseRadius, pitchRadius));
  const pitchDeg = atan2(pitchPoint[1], pitchPoint[0]) * 180 / PI;
  const minRadius = max(baseRadius, rootRadius);

  const startRad = max(involuteIntersectRad(baseRadius, minRadius) - n, 0);
  const stopRad = involuteIntersectRad(baseRadius, outerRadius);
  const step = (stopRad - startRad) / n;

  const p = [[0, 0] as Vec2];
  for (let i = 0; i <= n; ++i) {
    p.push(involute(baseRadius, i * step + startRad));
  }

  const halftooth = new CrossSection(p)
                        .rotate(-pitchDeg - halfThickDeg)
                        .subtract(CrossSection.square(2 * outerRadius));
  const tooth = halftooth.add(halftooth.mirror([0, 1]));

  let gear =
      CrossSection
          .circle(
              max(rootRadius,
                  pitchRadius - depthRatio * circularPitch / 2 - clearance / 2),
              teeth * 2)
          .rotate(90 / teeth);
  for (let i = 0; i < teeth; ++i) {
    gear = gear.add(tooth.rotate(i * 360 / teeth));
  }
  return gear.intersect(
      CrossSection
          .circle(
              pitchRadius + depthRatio * circularPitch / 2 - clearance / 2,
              teeth * 3)
          .rotate(90 / teeth));
}

// Calculate the involute angle at a given radius larger than the base radius.
function involuteIntersectRad(baseRadius, radius) {
  return sqrt(pow(radius / baseRadius, 2) - 1);
}

// Calculate the involute position for a given base radius and involute angle.
function involute(baseRadius, involuteRad): Vec2 {
  return [
    baseRadius * (cos(involuteRad) + involuteRad * sin(involuteRad)),
    baseRadius * (sin(involuteRad) - involuteRad * cos(involuteRad))
  ]
}

// ManifoldCAD will attempt to render to the default export of a model.
// This can be a Manifold object or a function that returns a Manifold object.
// It will not run when importing this file, unless explicitly called.
export default () => {
  return spurGear(15, 5, {circularPitch: 5});
};
