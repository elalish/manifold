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

import {BoxGeometry, BufferAttribute, BufferGeometry, IcosahedronGeometry, Mesh as ThreeMesh, MeshLambertMaterial, MeshNormalMaterial, PerspectiveCamera, PointLight, Scene, WebGLRenderer} from 'three';

import type {Mesh as MeshType} from '../../manifold';
import Module from '../../manifold';

// Load Manifold WASM library
const wasm = await Module();
wasm.setup();
const {Manifold, Mesh} = wasm;

// Define our set of materials
const materials = [
  new MeshNormalMaterial({flatShading: true}),
  new MeshLambertMaterial({color: 'red', flatShading: true}),
  new MeshLambertMaterial({color: 'blue', flatShading: true})
];
const result = new ThreeMesh(undefined, materials);

// Set up Manifold IDs corresponding to materials
const firstID = Manifold.reserveIDs(materials.length);
// ids vector is parallel to materials vector - same indexing
const ids = [...Array<number>(materials.length)].map((_, idx) => firstID + idx);
// Build a mapping to get back from ID to material index
const id2matIndex = new Map();
ids.forEach((id, idx) => id2matIndex.set(id, idx));

// Set up Three.js scene
const scene = new Scene();
const camera = new PerspectiveCamera(30, 1, 0.01, 10);
camera.position.z = 1;
camera.add(new PointLight(0xffffff, 1));
scene.add(camera);
scene.add(result);

// Set up Three.js renderer
const output = document.querySelector('#output')!;
const renderer = new WebGLRenderer({canvas: output, antialias: true});
const dim = output.getBoundingClientRect();
renderer.setSize(dim.width, dim.height);
renderer.setAnimationLoop(function(time: number) {
  result.rotation.x = time / 2000;
  result.rotation.y = time / 1000;
  renderer.render(scene, camera);
});

// Create input meshes in Three.js
const cube = new BoxGeometry(0.2, 0.2, 0.2);
cube.clearGroups();
cube.addGroup(0, 18, 0);         // First 6 faces colored by normal
cube.addGroup(18, Infinity, 1);  // Rest of faces are red

const icosahedron = new IcosahedronGeometry(0.16);
icosahedron.clearGroups();
icosahedron.addGroup(30, Infinity, 2);  // Last faces are blue
icosahedron.addGroup(0, 30, 0);         // First 10 faces colored by normal
// The above groups are in reversed order to demonstrate the need for sorting.

// Convert Three.js input meshes to Manifolds
const manifoldCube = new Manifold(geometry2mesh(cube));
const manifoldIcosahedron = new Manifold(geometry2mesh(icosahedron));

// Set up UI for operations
type BooleanOp = 'union'|'difference'|'intersection';

function csg(operation: BooleanOp) {
  result.geometry?.dispose();
  result.geometry = mesh2geometry(
      Manifold[operation](manifoldCube, manifoldIcosahedron).getMesh());
}

csg('union');
const selectElement = document.querySelector('select')!;
selectElement.onchange = function() {
  csg(selectElement.value as BooleanOp);
};

// Convert Three.js BufferGeometry to Manifold Mesh
function geometry2mesh(geometry: BufferGeometry) {
  // Only using position in this sample for simplicity. Can interleave any other
  // desired attributes here such as UV, normal, etc.
  const vertProperties = geometry.attributes.position.array as Float32Array;
  // Manifold only uses indexed geometry, so generate an index if necessary.
  const triVerts = geometry.index != null ?
      geometry.index.array as Uint32Array :
      new Uint32Array(vertProperties.length / 3).map((_, idx) => idx);
  // Create a triangle run for each group (material) - akin to a draw call.
  const starts = [...Array(geometry.groups.length)].map(
      (_, idx) => geometry.groups[idx].start);
  // Map the materials to ID.
  const originalIDs = [...Array(geometry.groups.length)].map(
      (_, idx) => ids[geometry.groups[idx].materialIndex!]);
  // List the runs in sequence.
  const indices = Array.from(starts.keys())
  indices.sort((a, b) => starts[a] - starts[b])
  const runIndex = new Uint32Array(indices.map(i => starts[i]));
  const runOriginalID = new Uint32Array(indices.map(i => originalIDs[i]));
  // Create the MeshGL for I/O with Manifold library.
  const mesh =
      new Mesh({numProp: 3, vertProperties, triVerts, runIndex, runOriginalID});
  // Automatically merge vertices with nearly identical positions to create a
  // Manifold. This only fills in the mergeFromVert and mergeToVert vectors -
  // these are automatically filled in for any mesh returned by Manifold. These
  // are necessary because GL drivers require duplicate verts when any
  // properties change, e.g. a UV boundary or sharp corner.
  mesh.merge();
  return mesh;
}

// Convert Manifold Mesh to Three.js BufferGeometry
function mesh2geometry(mesh: MeshType) {
  const geometry = new BufferGeometry();
  // Assign buffers
  geometry.setAttribute(
      'position', new BufferAttribute(mesh.vertProperties, 3));
  geometry.setIndex(new BufferAttribute(mesh.triVerts, 1));
  // Create a group (material) for each ID. Note that there may be multiple
  // triangle runs returned with the same ID, though these will always be
  // sequential since they are sorted by ID. In this example there are two runs
  // for the MeshNormalMaterial, one corresponding to each input mesh that had
  // this ID. This allows runTransform to return the total transformation matrix
  // applied to each triangle run from its input mesh - even after many
  // consecutive operations.
  let id = mesh.runOriginalID[0];
  let start = mesh.runIndex[0];
  for (let run = 0; run < mesh.numRun; ++run) {
    const nextID = mesh.runOriginalID[run + 1];
    if (nextID !== id) {
      const end = mesh.runIndex[run + 1];
      geometry.addGroup(start, end - start, id2matIndex.get(id));
      id = nextID;
      start = end;
    }
  }
  return geometry;
}
