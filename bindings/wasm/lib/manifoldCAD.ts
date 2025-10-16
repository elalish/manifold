import {initialize} from './worker.ts';

const evaluator = initialize();
const context = await evaluator.getFullContext();

export const {
  Manifold,
  CrossSection,
  Mesh,
  triangulate,
  setMinCircularAngle,
  setMinCircularEdgeLength,
  setCircularSegments,
  getCircularSegments,
  resetToCircularDefaults,
  show,
  only,
  setMaterial,
  setMorphStart,
  setMorphEnd,
  GLTFNode
} = context;