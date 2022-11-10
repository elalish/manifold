// Copyright 2022 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const expect = require('chai').expect;
const wasm = require('../manifold.js');
const examples = require('../examples/examples');
const THREE = require('three');

// manifold member functions that returns a new manifold
const memberFunctions = [
  'add', 'subtract', 'intersect', 'refine', 'transform', 'translate', 'rotate',
  'scale', 'asOriginal', 'decompose'
];
// top level functions that constructs a new manifold
const constructors = [
  'cube', 'cylinder', 'sphere', 'tetrahedron', 'extrude', 'revolve', 'union',
  'difference', 'intersection', 'compose', 'levelSet', 'smooth'
];
const utils = [
  'setMinCircularAngle', 'setMinCircularEdgeLength', 'setCircularSegments',
  'getCircularSegments', 'Mesh'
];
const exposedFunctions = constructors.concat(utils);

wasm().then(function(Module) {
  Module.setup();
  // Setup memory management, such that users don't have to care about
  // calling `delete` manually.
  // Note that this only fixes memory leak across different runs: the memory
  // will only be freed when the compilation finishes.

  let manifoldRegistry = [];
  for (const name of memberFunctions) {
    const originalFn = Module.Manifold.prototype[name];
    Module.Manifold.prototype['_' + name] = originalFn;
    Module.Manifold.prototype[name] = function(...args) {
      const result = this['_' + name](...args);
      manifoldRegistry.push(result);
      return result;
    }
  }

  for (const name of constructors) {
    const originalFn = Module[name];
    Module[name] = function(...args) {
      const result = originalFn(...args);
      manifoldRegistry.push(result);
      return result;
    }
  }

  Module.cleanup =
      function() {
    for (const obj of manifoldRegistry) {
      // decompose result is an array of manifolds
      if (obj instanceof Array)
        for (const elem of obj) elem.delete();
      else
        obj.delete();
    }
    manifoldRegistry = [];
  }

  function runExample(name) {
    try {
      const content = examples.functionBodies.get(name) + '\nreturn result;\n';
      ;
      const f = new Function(...exposedFunctions, 'THREE', content);
      const manifold = f(...exposedFunctions.map(name => Module[name]), THREE);
      const prop = manifold.getProperties();
      const genus = manifold.genus();
      return {...prop, genus};
    } finally {
      Module.cleanup();
    }
  }

  suite('Examples', () => {
    test('Intro', () => {
      const result = runExample('Intro');
      expect(result.genus).to.equal(5, 'Genus');
      expect(result.volume).to.be.closeTo(203164, 1, 'Volume');
      expect(result.surfaceArea).to.be.closeTo(62046, 1, 'Surface Area');
    });

    test('Tetrahedron Puzzle', () => {
      const result = runExample('Tetrahedron Puzzle');
      expect(result.genus).to.equal(0, 'Genus');
      expect(result.volume).to.be.closeTo(7297, 1, 'Volume');
      expect(result.surfaceArea).to.be.closeTo(3303, 1, 'Surface Area');
    });

    test('Rounded Frame', () => {
      const result = runExample('Rounded Frame');
      expect(result.genus).to.equal(5, 'Genus');
      expect(result.volume).to.be.closeTo(353706, 1, 'Volume');
      expect(result.surfaceArea).to.be.closeTo(68454, 1, 'Surface Area');
    });

    test('Heart', () => {
      const result = runExample('Heart');
      expect(result.genus).to.equal(0, 'Genus');
      expect(result.volume).to.be.closeTo(282743, 1, 'Volume');
      expect(result.surfaceArea).to.be.closeTo(22187, 1, 'Surface Area');
    });

    test('Scallop', () => {
      const result = runExample('Scallop');
      expect(result.genus).to.equal(0, 'Genus');
      expect(result.volume).to.be.closeTo(41284, 1, 'Volume');
      expect(result.surfaceArea).to.be.closeTo(7810, 1, 'Surface Area');
    });

    test('Torus Knot', () => {
      const result = runExample('Torus Knot');
      expect(result.genus).to.equal(1, 'Genus');
      expect(result.volume).to.be.closeTo(20960, 1, 'Volume');
      expect(result.surfaceArea).to.be.closeTo(11202, 1, 'Surface Area');
    });

    test('Menger Sponge', () => {
      const result = runExample('Menger Sponge');
      expect(result.genus).to.equal(1409, 'Genus');
      expect(result.volume).to.be.closeTo(406457, 1, 'Volume');
      expect(result.surfaceArea).to.be.closeTo(247590, 1, 'Surface Area');
    });

    test('Stretchy Bracelet', () => {
      const result = runExample('Stretchy Bracelet');
      expect(result.genus).to.equal(1, 'Genus');
      expect(result.volume).to.be.closeTo(3992, 1, 'Volume');
      expect(result.surfaceArea).to.be.closeTo(22267, 1, 'Surface Area');
    });

    test('Gyroid Module', () => {
      const result = runExample('Gyroid Module');
      expect(result.genus).to.equal(15, 'Genus');
      expect(result.volume).to.be.closeTo(4167, 1, 'Volume');
      expect(result.surfaceArea).to.be.closeTo(5642, 1, 'Surface Area');
    });
  });
});
