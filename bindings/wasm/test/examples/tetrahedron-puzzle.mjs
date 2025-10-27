// A tetrahedron cut into two identical halves that can screw together as
// a puzzle. This demonstrates how redundant points along a polygon can be
// used to make twisted extrusions smoother. It also showcases animated
// assemblies. Based on the screw puzzle by George Hart:
// https://www.thingiverse.com/thing:186372

const edgeLength = 50;  // Length of each edge of the overall tetrahedron.
const gap = 0.2;  // Spacing between the two halves to allow sliding.
const nDivisions = 50;  // Divisions (both ways) in the screw surface.

const scale = edgeLength / (2 * Math.sqrt(2));

const tet = Manifold.tetrahedron().intersect(
    Manifold.tetrahedron().rotate([0, 0, 90]).scale(2.5));

const box = [];
box.push([2, -2], [2, 2]);
for (let i = 0; i <= nDivisions; ++i) {
  box.push([gap / (2 * scale), 2 - i * 4 / nDivisions]);
}

const cyan = [0, 1, 1];
const magenta = [1, 0, 1];
const fade = (color, pos) => {
  for (let i = 0; i < 3; ++i) {
    const x = pos[2] / 2;
    color[i] = cyan[i] * x + magenta[i] * (1 - x);
  }
};

// setProperties(3, fade) creates three channels of vertex properties
// according to the above fade function. setMaterial assigns these
// channels as colors, and sets the factor to white, since our default is
// yellow.
const screw = setMaterial(
    Manifold.extrude(box, 2, nDivisions, 270).setProperties(3, fade),
    {baseColorFactor: [1, 1, 1], attributes: ['COLOR_0']});

const result =
    tet.intersect(screw.rotate([0, 0, -45]).translate([0, 0, -1]))
        .scale(scale);

// Assigned materials are only applied to a GLTFNode. Note that material
// definitions cascade, applying recursively to all child surfaces, but
// overridden by any materials defined lower down. The default material
// properties, as well as animation parameters can be set via
// globalDefaults.

const layFlat = new GLTFNode();
layFlat.rotation =
    [45, -Math.atan(1 / Math.sqrt(2)) * 180 / Math.PI, 120];
layFlat.translation = [0, 0, scale * Math.sqrt(3) / 3];

const fixed = new GLTFNode(layFlat);
fixed.manifold = result;
fixed.rotation = [0, 0, 180];

// For 3MF export, only top-level objects are independently arrangeable.
const layFlat2 = layFlat.clone();

const moving = new GLTFNode(layFlat2);
moving.manifold = result;
// Use functions to create animation, which runs from t=0 to t=1.
moving.translation = (t) => {
  const a = 1 - t;
  const x = a > 0.5 ? scale * 2 * (0.5 - a) : 0;
  return [x, x, -2 * x + scale * 4 * (0.5 - Math.abs(0.5 - a))];
};
moving.rotation = (t) => [0, 0, 270 * 2 * (1 - t)];

globalDefaults.animationLength = 10;  // seconds
globalDefaults.animationMode = 'ping-pong';

const nodes = getGLTFNodes();
export default nodes;