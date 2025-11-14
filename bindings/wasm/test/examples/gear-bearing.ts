// Demonstrate using a library.
// This example recreates the Gear Bearing by Emmett Lalish:
// https://www.thingiverse.com/thing:53451

import {getGLTFNodes, GLTFNode, Manifold} from 'manifold-3d/manifoldCAD';

import {spurGear} from './involute-gear-library';

const {round, min, cos, tan, sqrt, PI} = Math;

const diameter = 51.7;
const thickness = 15;
const clearance = 0.15;
const planets = 5;
const planetTeeth = 7;
const sunTeeth = 9;
const pressureDeg = 45;
const holeWidth = 6.7;
setMinCircularAngle(2);
setMinCircularEdgeLength(0.5);

const depthRatio = 0.5;
const m = round(planets);
const np = round(planetTeeth);
const k1 = round(2 / m * (sunTeeth + np));
const k = k1 * m % 2 != 0 ? k1 + 1 : k1;
const ns = k * m / 2 - np;
const nr = ns + 2 * np;
const pitchD = 0.9 * diameter /
    (1 +
     min(PI / (2 * nr * tan(pressureDeg * PI / 180)), PI * depthRatio / nr));
const circularPitch = pitchD * PI / nr;
const twist = circularPitch * 2 / thickness;
const holeRadius = holeWidth / sqrt(3);

const gearParams = {
  circularPitch,
  pressureDeg,
  clearance,
  depthRatio,
  twist
};

function chevronGear(teeth, clearanceFactor = 1) {
  const params = {...gearParams};
  params.clearance *= clearanceFactor;
  const top = spurGear(teeth, thickness / 2, params);
  return top.add(top.mirror([0, 0, 1]));
}

const sun = chevronGear(ns).mirror([0, 1, 0]).subtract(
    Manifold.cylinder(thickness + 1, holeRadius, holeRadius, 6, true));
const ring = Manifold.cylinder(thickness, diameter / 2, diameter / 2, 0, true)
                 .subtract(chevronGear(nr, -1));
const planet = chevronGear(np);

const shift = new GLTFNode();
shift.translation = [0, 0, thickness / 2];

const ringNode = new GLTFNode(shift);
ringNode.manifold = ring;
ringNode.material = {
  baseColorFactor: [0.3, 0.3, 0.3]
};

const sunNode = new GLTFNode(shift);
sunNode.manifold = sun;
sunNode.rotation =
    (t) => [0, 0, (np + 1) * 180 / ns + (t * 360) * (ns + np) * 2 / ns]
sunNode.material = {
  baseColorFactor: [0.7, 0.7, 0.7]
};

for (let i = 0; i < planets; ++i) {
  const planetCarrier = new GLTFNode(shift);
  planetCarrier.rotation = (t) => [0, 0, (i / m + t) * 360];

  const planetNode = new GLTFNode(planetCarrier);
  planetNode.manifold = planet;
  planetNode.rotation =
      (t) => [0, 0, i * ns / m * 360 / np - (t * 360) * (1 + (ns + np) / np)];
  planetNode.translation = [pitchD / 2 * (ns + np) / nr, 0, 0];
  planetNode.material = {
    baseColorFactor: [
      0.3 * cos(2 * PI * i / m) + 0.5,
      0.2 * cos(2 * PI * (i / m + 1 / 3)) + 0.5,
      0.8 * cos(2 * PI * (i / m + 2 / 3)) + 0.5
    ]
  };
}

setAnimationDuration(5);  // GLTF animation length in seconds

const bearing = getGLTFNodes();
export default bearing;