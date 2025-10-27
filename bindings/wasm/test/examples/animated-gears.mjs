// Demonstrate using a library.

import {involuteGear2d} from './Involute Gear Library';

const gear1teeth = 12;
const gear2teeth = 15;

const gear1 = new GLTFNode();
gear1.manifold = involuteGear2d({teeth:gear1teeth}).extrude(3);
gear1.translation=[gear1teeth/2,0,-1.5]

const gear2 = new GLTFNode();
gear2.manifold = involuteGear2d({teeth:gear2teeth}).extrude(1);
gear2.translation = [-gear2teeth/2,0,-0.5]
gear2.material = {baseColorFactor: [1, 0, 1]};

globalDefaults.animationLength = 10;  // GLTF animation length in seconds
const speed = 360; // degrees of rotation of gear1 over animationLength
const ratio = gear1teeth / gear2teeth;
gear1.rotation = (t) => [0,0, t * speed]
gear2.rotation = (t) => [0,0, -(t * speed * ratio) + (1/2 * 360/gear2teeth)]

const gears = getGLTFNodes();
export default gears;