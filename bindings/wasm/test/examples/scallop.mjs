// A smoothed manifold demonstrating selective edge sharpening with
// smooth() and refine(), see more details at:
// https://elalish.blogspot.com/2022/03/smoothing-triangle-meshes.html

import {Manifold, Mesh, GLTFNode} from 'manifold-3d/manifoldCAD';

const height = 10;
const radius = 30;
const offset = 20;
const wiggles = 12;
const sharpness = 0.8;
const n = 50;

const positions = [];
const triangles = [];
positions.push(-offset, 0, height, -offset, 0, -height);
const sharpenedEdges = [];

const delta = 3.14159 / wiggles;
for (let i = 0; i < 2 * wiggles; ++i) {
  const theta = (i - wiggles) * delta;
  const amp = 0.5 * height * Math.max(Math.cos(0.8 * theta), 0);

  positions.push(
      radius * Math.cos(theta), radius * Math.sin(theta),
      amp * (i % 2 == 0 ? 1 : -1));
  let j = i + 1;
  if (j == 2 * wiggles) j = 0;

  const smoothness = 1 - sharpness * Math.cos((theta + delta / 2) / 2);
  let halfedge = triangles.length + 1;
  sharpenedEdges.push({halfedge, smoothness});
  triangles.push(0, 2 + i, 2 + j);

  halfedge = triangles.length + 1;
  sharpenedEdges.push({halfedge, smoothness});
  triangles.push(1, 2 + j, 2 + i);
}

const triVerts = Uint32Array.from(triangles);
const vertProperties = Float32Array.from(positions);
const scallop = new Mesh({numProp: 3, triVerts, vertProperties});

const colorCurvature = (color, pos, oldProp) => {
  const a = Math.max(0, Math.min(1, oldProp[0] / 3 + 0.5));
  const b = a * a * (3 - 2 * a);
  const red = [1, 0, 0];
  const blue = [0, 0, 1];
  for (let i = 0; i < 3; ++i) {
    color[i] = (1 - b) * blue[i] + b * red[i];
  }
};
const result = Manifold.smooth(scallop, sharpenedEdges)
                    .refine(n)
                    .calculateCurvature(-1, 0)
                    .setProperties(3, colorCurvature);

const node = new GLTFNode();
node.manifold = result;
node.material = {
  baseColorFactor: [1, 1, 1],
  metallic: 0,
  attributes: ['COLOR_0']
};
export default node;