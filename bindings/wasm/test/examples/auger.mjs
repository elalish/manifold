const outerRadius = 20;
const beadRadius = 2;
const height = 40;
const twist = 90;

import {CrossSection, Manifold, setMinCircularEdgeLength} from 'manifold-3d/manifoldCAD';
const {revolve, sphere, union, extrude} = Manifold;
const {circle} = CrossSection;
setMinCircularEdgeLength(0.1);

const bead1 =
    revolve(circle(beadRadius).translate([outerRadius, 0]), 50, 90)
        .add(sphere(beadRadius).translate([outerRadius, 0, 0]))
        .translate([0, -outerRadius, 0]);

const beads = [];
for (let i = 0; i < 3; i++) {
  beads.push(bead1.rotate(0, 0, 120 * i));
}
const bead = union(beads);

const auger = extrude(bead.slice(0), height, 50, twist);

const result =
    auger.add(bead).add(bead.translate(0, 0, height).rotate(0, 0, twist));
export default result;