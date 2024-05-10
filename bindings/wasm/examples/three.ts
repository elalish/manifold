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

import {BoxGeometry, BufferAttribute, BufferGeometry, IcosahedronGeometry, Mesh as ThreeMesh, MeshLambertMaterial, MeshNormalMaterial, PerspectiveCamera, Scene, WebGLRenderer} from 'three';

import Module, {Mesh} from './built/manifold.js';

type BooleanOp = 'union'|'difference'|'intersection';

const wasm = await Module();
wasm.setup();

const {Manifold, Mesh} = wasm;

// we have manifold module, let's do some three.js
const camera = new PerspectiveCamera(30, 1, 0.01, 10);
camera.position.z = 1;

const scene = new Scene();
const materials = [
  new MeshNormalMaterial({flatShading: true}),
  new MeshLambertMaterial({color: 'red', flatShading: true}),
  new MeshLambertMaterial({color: 'blue', flatShading: true})
];
const result =
    new ThreeMesh(undefined, new MeshNormalMaterial({flatShading: true}));
scene.add(result);

const cube = new BoxGeometry(0.2, 0.2, 0.2);
cube.addGroup(0, 18, 0);
cube.addGroup(18, Infinity, 1);

const icosahedron = new IcosahedronGeometry(0.16);
icosahedron.addGroup(0, 30, 0);
icosahedron.addGroup(30, Infinity, 2);

const manifold_1 = new Manifold(geometry2mesh(cube));
const manifold_2 = new Manifold(geometry2mesh(icosahedron));

const csg = function(operation: BooleanOp) {
  result.geometry?.dispose();
  result.geometry =
      mesh2geometry(Manifold[operation](manifold_1, manifold_2).getMesh());
};

const selectElement = document.querySelector('select') as HTMLSelectElement;

selectElement.onchange = function() {
  csg(selectElement.value as BooleanOp);
};

csg('difference');

const output = document.querySelector('#output')!;
const renderer = new WebGLRenderer({canvas: output, antialias: true});
const dim = output.getBoundingClientRect();
renderer.setSize(dim.width, dim.height);
renderer.setAnimationLoop(function(time: number) {
  result.rotation.x = time / 2000;
  result.rotation.y = time / 1000;
  renderer.render(scene, camera);
});

// functions to convert between three.js and wasm
function geometry2mesh(geometry: BufferGeometry) {
  const vertProperties = geometry.attributes.position.array as Float32Array;
  const triVerts = geometry.index != null ?
      geometry.index.array as Uint32Array :
      new Uint32Array(vertProperties.length / 3).map((_, idx) => idx);
  const mesh = new Mesh({numProp: 3, vertProperties, triVerts});
  mesh.merge();
  return mesh;
}

function mesh2geometry(mesh: Mesh) {
  const geometry = new BufferGeometry();
  geometry.setAttribute(
      'position', new BufferAttribute(mesh.vertProperties, 3));
  geometry.setIndex(new BufferAttribute(mesh.triVerts, 1));
  return geometry;
}
