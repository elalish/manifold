// A smoothed manifold demonstrating manual smoothing by assigning tangents to
// halfedges. Use refine() before export to see the curvature.

import {GLTFNode, Manifold, Mesh} from 'manifold-3d/manifoldCAD';

const height = 10;
const radius = 30;
const offset = 20;
const wiggles = 12;
const lean = 0.3;
const frontalSharpness = 1;
const n = 50;

const lerp = (a, b, t) => a.map((value, i) => value + (b[i] - value) * t);

const positions = [];
const triangles = [];
const halfedgeTangent = [];
positions.push(-offset, 0, height, -offset, 0, -height);

const len = Math.PI * radius / (2 * wiggles);
const topNormal = [-lean, 0, Math.sqrt(1 - lean * lean)];
const delta = Math.PI / wiggles;
const centerTangents = new Array(2 * wiggles);
const edgeTangents = new Array(2 * wiggles);

for (let i = 0; i < 2 * wiggles; ++i) {
  const theta = i * delta;
  const amp = 0.5 * height * (Math.cos(theta) + 1) / 2;
  const v = [
    radius * Math.cos(theta), radius * Math.sin(theta),
    amp * (i % 2 === 0 ? 1 : -1)
  ];

  positions.push(...v);

  const centerTan = [
    (v[0] + offset) / 2,
    v[1] / 2,
    (v[2] - height) / 2,
  ];
  const dot = centerTan[0] * topNormal[0] + centerTan[1] * topNormal[1] +
      centerTan[2] * topNormal[2];
  centerTangents[i] = [
    centerTan[0] - dot * topNormal[0],
    centerTan[1] - dot * topNormal[1],
    centerTan[2] - dot * topNormal[2],
  ];

  edgeTangents[i] = [-len * Math.sin(theta), len * Math.cos(theta), 0];
}

for (let i = 0; i < 2 * wiggles; ++i) {
  const next = i + 1 === 2 * wiggles ? 0 : i + 1;

  const radial = centerTangents[i];
  const nextRadial = centerTangents[next];
  const edge = edgeTangents[i];
  const nextEdge = edgeTangents[next].map(value => -value);
  const sharpness = frontalSharpness * (Math.cos(i * delta) + 1) / 4;
  const up = lerp([0, 0, len], [nextEdge[1], -nextEdge[0], 0], sharpness);
  const down = lerp([0, 0, -len], [-edge[1], edge[0], 0], sharpness);

  triangles.push(0, 2 + i, 2 + next);
  halfedgeTangent.push(
      radial[0], radial[1], radial[2], 1, edge[0], edge[1], edge[2], 1, up[0],
      up[1], up[2], 1);

  triangles.push(1, 2 + next, 2 + i);
  halfedgeTangent.push(
      nextRadial[0], nextRadial[1], -nextRadial[2], 1, nextEdge[0], nextEdge[1],
      nextEdge[2], 1, down[0], down[1], down[2], 1);
}

const triVerts = Uint32Array.from(triangles);
const vertProperties = Float32Array.from(positions);
const scallop = new Mesh({
  numProp: 3,
  triVerts,
  vertProperties,
  halfedgeTangent: Float32Array.from(halfedgeTangent),
});

const colorCurvature = (color, pos, oldProp) => {
  const a = Math.max(0, Math.min(1, oldProp[0] / 3 + 0.5));
  const b = a * a * (3 - 2 * a);
  const red = [1, 0, 0];
  const blue = [0, 0, 1];
  for (let i = 0; i < 3; ++i) {
    color[i] = (1 - b) * blue[i] + b * red[i];
  }
};
const result =
    new Manifold(scallop).refine(n).calculateCurvature(-1, 0).setProperties(
        3, colorCurvature);

const node = new GLTFNode();
node.manifold = result;
node.translation = [0, 0, -result.boundingBox().min[2]];
node.material = {
  baseColorFactor: [1, 1, 1],
  metallic: 0,
  attributes: ['COLOR_0']
};
export default node;