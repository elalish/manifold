// Copyright 2024 The Manifold Authors.
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

import {BufferAttribute, BufferGeometry, IcosahedronGeometry, Mesh as ThreeMesh, MeshNormalMaterial, PerspectiveCamera, Scene, WebGLRenderer} from 'three';
import {mergeVertices} from 'three/examples/jsm/utils/BufferGeometryUtils.js';

import Module, {Mesh} from './built/manifold.js';

type Boolean = 'union'|'difference'|'intersection';

const wasm = await Module();
wasm.setup();

const {Manifold, Mesh} = wasm;

// we have manifold module, let's do some three.js
const camera = new PerspectiveCamera(30, 1, 0.01, 10);
camera.position.z = 1;

const scene = new Scene();
const mesh =
    new ThreeMesh(undefined, new MeshNormalMaterial({flatShading: true}));
scene.add(mesh);

const icosahedron = simplify(new IcosahedronGeometry(0.16));

const manifold_1 = Manifold.cube([0.2, 0.2, 0.2], true);
const manifold_2 = new Manifold(geometry2mesh(icosahedron));

const csg = function(operation: Boolean) {
  mesh.geometry?.dispose();
  mesh.geometry =
      mesh2geometry(Manifold[operation](manifold_1, manifold_2).getMesh());
};

const selectElement = document.querySelector('select') as HTMLSelectElement;

selectElement.onchange = function() {
  csg(selectElement.value as Boolean);
};

csg('difference');

const output = document.querySelector('#output')!;
const renderer = new WebGLRenderer({canvas: output, antialias: true});
const dim = output.getBoundingClientRect();
renderer.setSize(dim.width, dim.height);
renderer.setAnimationLoop(function(time: number) {
  mesh.rotation.x = time / 2000;
  mesh.rotation.y = time / 1000;
  renderer.render(scene, camera);
});

// functions to convert between three.js and wasm
function geometry2mesh(geometry: BufferGeometry) {
  const vertProperties = geometry.attributes.position.array as Float32Array;
  const triVerts = geometry.index!.array as Uint32Array;
  return new Mesh({numProp: 3, vertProperties, triVerts});
}

function mesh2geometry(mesh: Mesh) {
  const geometry = new BufferGeometry();
  geometry.setAttribute(
      'position', new BufferAttribute(mesh.vertProperties, 3));
  geometry.setIndex(new BufferAttribute(mesh.triVerts, 1));
  return geometry;
}

// most of three.js geometries aren't manifolds, so...
function simplify(geometry: BufferGeometry) {
  delete geometry.attributes.normal;
  delete geometry.attributes.uv;
  const simplified = mergeVertices(geometry);
  simplified.computeVertexNormals();
  return simplified;
}