import auger from '../../manifold-cad/test/examples/auger.mjs?raw';
import gearBearing from '../../manifold-cad/test/examples/gear-bearing.ts?raw';
import godelEscherBach from '../../manifold-cad/test/examples/godel-escher-bach.ts?raw';
import gyroidModule from '../../manifold-cad/test/examples/gyroid-module.ts?raw';
import heart from '../../manifold-cad/test/examples/heart.mjs?raw';
import importManifold from '../../manifold-cad/test/examples/import-manifold.ts?raw';
import importModel from '../../manifold-cad/test/examples/import-model.ts?raw';
import intro from '../../manifold-cad/test/examples/intro.mjs?raw';
import involuteGearLibrary from '../../manifold-cad/test/examples/involute-gear-library.ts?raw';
import mengerSponge from '../../manifold-cad/test/examples/menger-sponge.mjs?raw';
import roundedFrame from '../../manifold-cad/test/examples/rounded-frame.mjs?raw';
import scallop from '../../manifold-cad/test/examples/scallop.mjs?raw';
import stretchyBracelet from '../../manifold-cad/test/examples/stretchy-bracelet.mjs?raw';
import tetrahedronPuzzle from '../../manifold-cad/test/examples/tetrahedron-puzzle.mjs?raw';
import torusKnot from '../../manifold-cad/test/examples/torus-knot.mjs?raw';
import voronoi from '../../manifold-cad/test/examples/voronoi.mjs?raw';

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
examples.set('Import Model', importModel);
examples.set('Import Manifold', importManifold);
examples.set('Gödel, Escher, Bach', godelEscherBach)

if (typeof self !== 'undefined') {
  self.examples = examples;
}