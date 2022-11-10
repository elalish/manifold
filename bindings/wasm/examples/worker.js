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

const threePath = 'https://cdn.jsdelivr.net/npm/three@0.144.0/';
importScripts(
    'manifold.js', threePath + 'build/three.js',
    threePath + 'examples/js/exporters/GLTFExporter.js');

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

let wasm;
Module().then(function(tmp) {
  wasm = tmp;
  wasm.setup();
  // Setup memory management, such that users don't have to care about
  // calling `delete` manually.
  // Note that this only fixes memory leak across different runs: the memory
  // will only be freed when the compilation finishes.

  let manifoldRegistry = [];
  for (const name of memberFunctions) {
    const originalFn = wasm.Manifold.prototype[name];
    wasm.Manifold.prototype['_' + name] = originalFn;
    wasm.Manifold.prototype[name] = function(...args) {
      const result = this['_' + name](...args);
      manifoldRegistry.push(result);
      return result;
    }
  }

  for (const name of constructors) {
    const originalFn = wasm[name];
    wasm[name] = function(...args) {
      const result = originalFn(...args);
      manifoldRegistry.push(result);
      return result;
    }
  }

  wasm.cleanup =
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

  postMessage(null);
});

const oldLog = console.log;
console.log = function(...args) {
  let message = '';
  for (const arg of args) {
    if (typeof arg == 'object') {
      message += JSON.stringify(arg, null, 4);
    } else {
      message += arg.toString();
    }
  }
  postMessage({log: message});
  oldLog(...args);
};

onmessage =
    (e) => {
      const content = e.data + '\nexportGLB(result);\n';
      try {
        const f = new Function(...exposedFunctions, content);
        f(...exposedFunctions.map(name => wasm[name]));
      } catch (error) {
        console.log(error.toString());
        postMessage({objectURL: null});
      } finally {
        wasm.cleanup();
      }
    }

// Export & Rendering
// ------------------------------------------------------------
const mesh = new THREE.Mesh(
    undefined,
    new THREE.MeshStandardMaterial(
        {color: 'yellow', metalness: 1, roughness: 0.2}));
const rotation = new THREE.Matrix4();
rotation.set(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);
mesh.setRotationFromMatrix(rotation);  // Z-up -> Y-up
mesh.scale.setScalar(0.001);           // mm -> m

const exporter = new THREE.GLTFExporter();

function exportGLB(manifold) {
  console.log(`Triangles: ${manifold.numTri().toLocaleString()}`);
  const box = manifold.boundingBox();
  const size = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    size[i] = Math.round((box.max[i] - box.min[i]) * 10) / 10;
  }
  console.log(`Bounding Box: X = ${size[0].toLocaleString()} mm, Y = ${
      size[1].toLocaleString()} mm, Z = ${size[2].toLocaleString()} mm`);
  const volume = Math.round(manifold.getProperties().volume / 10);
  console.log(`Genus: ${manifold.genus().toLocaleString()}, Volume: ${
      (volume / 100).toLocaleString()} cm^3`);

  mesh.geometry?.dispose();
  mesh.geometry = mesh2geometry(manifold.getMesh());
  exporter.parse(
      mesh,
      (gltf) => {
        const blob = new Blob([gltf], {type: 'application/octet-stream'});
        postMessage({objectURL: URL.createObjectURL(blob)});
      },
      () => {
        console.log('glTF export failed!');
        postMessage({objectURL: null});
      },
      {binary: true});
}

function mesh2geometry(mesh) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(mesh.vertPos, 3));
  geometry.setIndex(new THREE.BufferAttribute(mesh.triVerts, 1));
  return geometry;
}