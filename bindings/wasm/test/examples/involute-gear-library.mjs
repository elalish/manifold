// Demonstrate how to build a library for manifoldCAD
//
// Create a CrossSection of an involute gear.
// Based on the openJScad example written by Joost Nieuwenhuijse and Simon Clark
// https://github.com/jscad/OpenJSCAD.org/blob/master/packages/examples/parameters/gear.js

import {CrossSection} from "manifold-3d/manifoldCAD";
const {circle, hull, union} = CrossSection;
const {PI, atan, cos, sin, sqrt} = Math;
const TAU = PI*2;

const involute = (r, psi) => ([
  r*(cos(psi)+psi*sin(psi)),
  r*(sin(psi)-psi*cos(psi))
]);

const gearParams = (args = {}) => {
  const opts = {
    module: 1,
    teeth: 12,
    pressureAngle: 20,
    addendum: 1,
    dedendum: 1.25,
    ...args
  };

  const { pressureAngle, module, teeth, addendum, dedendum } = opts;
  const pitchRadius = module * teeth / 2;
  const tipRadius = pitchRadius + addendum * module;
  const rootRadius = pitchRadius - dedendum * module;
  const baseRadius = pitchRadius * cos(pressureAngle * TAU/360);

  return {
    ...opts,
    pitchRadius, tipRadius, rootRadius, baseRadius,
  };
};

const tooth2d = (params) => {
  const {baseRadius, pitchRadius, tipRadius, teeth} = params;
  const maxTanLength = sqrt(tipRadius**2 - baseRadius**2);
  const maxAngle = maxTanLength / baseRadius;

  const n = 8;
  const points = [[0,0]];
  for(let i=0; i<=n; i++) {
    const angle = i*maxAngle/n;
    points.push(involute(baseRadius,angle))
  }

  const toothWidthAtPitchCircle = sqrt(pitchRadius**2-baseRadius**2);
  const angleAtPitchCircle = toothWidthAtPitchCircle / baseRadius;
  const diffAngle = angleAtPitchCircle - atan(angleAtPitchCircle);
  const angularToothWidthAtBase = (TAU/teeth) + diffAngle;

  const halfTooth = hull(points)
    .rotate(-2*angularToothWidthAtBase)
    .rotate(-360/teeth * 1/4);
  return hull([halfTooth,halfTooth.mirror([0,1])]);
};

// Export this function.  ManifoldCAD will not call it directly,
// but other models can call it.  See 'Animated Gears' for an example.
export const involuteGear2d = (args = {}) => {
  const params = gearParams(args);
  const {teeth, rootRadius} = params;

  const t = tooth2d(params);
  return union(
    [...Array(teeth).keys()]
      .map(n => t.rotate(n * 360/teeth))
  ).add(
    circle(rootRadius)
  );
};

// ManifoldCAD will attempt to render to the default export of a model.
// This can be a Manifold object or a function that returns a Manifold object.
// It will not run when importing this file, unless explicitly called.
export default () => {
  return involuteGear2d({teeth: 12}).extrude(1.5);
};
