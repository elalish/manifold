import auger from '../test/examples/auger.mjs?raw';
import gearBearing from '../test/examples/gear-bearing.ts?raw';
import gyroidModule from '../test/examples/gyroid-module.mjs?raw';
import heart from '../test/examples/heart.mjs?raw';
import intro from '../test/examples/intro.mjs?raw';
import involuteGearLibrary from '../test/examples/involute-gear-library.ts?raw';
import mengerSponge from '../test/examples/menger-sponge.mjs?raw';
import roundedFrame from '../test/examples/rounded-frame.mjs?raw';
import scallop from '../test/examples/scallop.mjs?raw';
import stretchyBracelet from '../test/examples/stretchy-bracelet.mjs?raw';
import tetrahedronPuzzle from '../test/examples/tetrahedron-puzzle.mjs?raw';
import torusKnot from '../test/examples/torus-knot.mjs?raw';
import voronoi from '../test/examples/voronoi.mjs?raw';

const examples = new Map();
examples.set('Intro', intro);
examples.set('Auger', auger);
examples.set('Tetrahedron Puzzle', tetrahedronPuzzle);
examples.set('Rounded Frame', roundedFrame);
examples.set('Heart', heart);
examples.set('Scallop', scallop);
examples.set('Torus Knot', torusKnot);
examples.set('Menger Sponge', mengerSponge);
examples.set('Stretchy Bracelet', stretchyBracelet);
examples.set('Gyroid Module', gyroidModule);
examples.set('Gear Bearing', gearBearing);
examples.set('involute-gear-library', involuteGearLibrary);
examples.set('Voronoi', voronoi);

if (typeof self !== 'undefined') {
  self.examples = examples;
}