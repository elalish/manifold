import {Accessor, Animation, AnimationSampler, Document, Mesh as GLTFMesh, Node} from '@gltf-transform/core';
import {quat} from 'gl-matrix';

import {Manifold, Mesh, Vec3} from '../examples/built/manifold';
import {Quat} from '../examples/public/editor';

import {globalDefaults, GLTFNode} from './export'

const FPS = 30;


interface Morph {
  start?: (v: Vec3) => void;
  end?: (v: Vec3) => void;
}

const manifold2morph = new Map<Manifold, Morph>();
let animation: Animation;
let timesAccessor: Accessor;
let weightsAccessor: Accessor;
let weightsSampler: AnimationSampler;
let hasAnimation: boolean;

export function cleanup() {
  manifold2morph.clear();
}

export function euler2quat(rotation: Vec3): Quat {
  const deg2rad = Math.PI / 180;
  const q = [0, 0, 0, 1] as Quat;
  quat.rotateZ(q, q, deg2rad * rotation[2]);
  quat.rotateY(q, q, deg2rad * rotation[1]);
  quat.rotateX(q, q, deg2rad * rotation[0]);
  return q;
}

export function addMotion(
    doc: Document, type: 'translation'|'rotation'|'scale', node: GLTFNode,
    out: Node): Vec3|null {
  const motion = node[type];
  if (motion == null) {
    return null;
  }
  if (typeof motion !== 'function') {
    return motion;
  }

  const nFrames = timesAccessor.getCount();
  const nEl = type == 'rotation' ? 4 : 3;
  const frames = new Float32Array(nEl * nFrames);
  for (let i = 0; i < nFrames; ++i) {
    const x = i / (nFrames - 1);
    const m = motion(
        globalDefaults.animationMode !== 'ping-pong' ?
            x :
            (1 - Math.cos(x * 2 * Math.PI)) / 2);
    frames.set(nEl === 4 ? euler2quat(m) : m, nEl * i);
  }

  const framesAccessor =
      doc.createAccessor(node.name + ' ' + type + ' frames')
          .setBuffer(doc.getRoot().listBuffers()[0])
          .setArray(frames)
          .setType(nEl === 4 ? Accessor.Type.VEC4 : Accessor.Type.VEC3);
  const sampler = doc.createAnimationSampler()
                      .setInput(timesAccessor)
                      .setOutput(framesAccessor)
                      .setInterpolation('LINEAR');
  const channel = doc.createAnimationChannel()
                      .setTargetPath(type)
                      .setTargetNode(out)
                      .setSampler(sampler);
  animation.addSampler(sampler);
  animation.addChannel(channel);
  hasAnimation = true;
  return motion(0);
}

export function setMorph(doc: Document, node: Node, manifold: Manifold) {
  if (manifold2morph.has(manifold)) {
    const channel = doc.createAnimationChannel()
                        .setTargetPath('weights')
                        .setTargetNode(node)
                        .setSampler(weightsSampler);
    animation.addChannel(channel);
    hasAnimation = true;
  }
}

export const getMorph = (manifold: Manifold) => manifold2morph.get(manifold)

export function morphStart(manifoldMesh: Mesh, morph?: Morph):
    number[] {
      const inputPositions: number[] = [];
      if (morph == null) {
        return inputPositions;
      }

      for (let i = 0; i < manifoldMesh.numVert; ++i) {
        for (let j = 0; j < 3; ++j)
          inputPositions[i * 3 + j] =
              manifoldMesh.vertProperties[i * manifoldMesh.numProp + j];
      }
      if (morph.start) {
        for (let i = 0; i < manifoldMesh.numVert; ++i) {
          const vertProp = manifoldMesh.vertProperties;
          const offset = i * manifoldMesh.numProp;
          const pos = inputPositions.slice(offset, offset + 3) as Vec3;
          morph.start(pos);
          for (let j = 0; j < 3; ++j) vertProp[offset + j] = pos[j];
        }
      }
      return inputPositions;
    }

export function morphEnd(
    doc: Document, manifoldMesh: Mesh, mesh: GLTFMesh, inputPositions: number[],
    morph?: Morph) {
  if (morph == null) {
    return;
  }

  mesh.setWeights([0]);

  mesh.listPrimitives().forEach((primitive, i) => {
    if (morph.end) {
      for (let i = 0; i < manifoldMesh.numVert; ++i) {
        const pos = inputPositions.slice(3 * i, 3 * (i + 1)) as Vec3;
        morph.end(pos);
        inputPositions.splice(3 * i, 3, ...pos);
      }
    }

    const startPosition = primitive.getAttribute('POSITION')!.getArray()!;
    const array = new Float32Array(startPosition.length);

    const offset = manifoldMesh.runIndex[i];
    for (let j = 0; j < array.length; ++j) {
      array[j] = inputPositions[offset + j] - startPosition[j];
    }

    const morphAccessor = doc.createAccessor(mesh.getName() + ' morph target')
                              .setBuffer(doc.getRoot().listBuffers()[0])
                              .setArray(array)
                              .setType(Accessor.Type.VEC3);
    const morphTarget =
        doc.createPrimitiveTarget().setAttribute('POSITION', morphAccessor);
    primitive.addTarget(morphTarget);
  });
}

export const setMorphStart =
    (manifold: Manifold, func: (v: Vec3) => void): void => {
      const morph = manifold2morph.get(manifold);
      if (morph != null) {
        morph.start = func;
      } else {
        manifold2morph.set(manifold, {start: func});
      }
    };

export const setMorphEnd =
    (manifold: Manifold, func: (v: Vec3) => void): void => {
      const morph = manifold2morph.get(manifold);
      if (morph != null) {
        morph.end = func;
      } else {
        manifold2morph.set(manifold, {end: func});
      }
    };


export function addAnimationToDoc(doc: Document) {
  animation = doc.createAnimation('');
  hasAnimation = false;
  const nFrames = Math.round(globalDefaults.animationLength * FPS) + 1;
  const times = new Float32Array(nFrames);
  const weights = new Float32Array(nFrames);
  for (let i = 0; i < nFrames; ++i) {
    const x = i / (nFrames - 1);
    times[i] = x * globalDefaults.animationLength;
    weights[i] = globalDefaults.animationMode !== 'ping-pong' ?
        x :
        (1 - Math.cos(x * 2 * Math.PI)) / 2;
  }
  timesAccessor = doc.createAccessor('animation times')
                      .setBuffer(doc.createBuffer())
                      .setArray(times)
                      .setType(Accessor.Type.SCALAR);
  weightsAccessor = doc.createAccessor('animation weights')
                        .setBuffer(doc.getRoot().listBuffers()[0])
                        .setArray(weights)
                        .setType(Accessor.Type.SCALAR);
  weightsSampler = doc.createAnimationSampler()
                       .setInput(timesAccessor)
                       .setOutput(weightsAccessor)
                       .setInterpolation('LINEAR');
  animation.addSampler(weightsSampler);
}

export function cleanupAnimation() {
  if (!hasAnimation) {
    timesAccessor.dispose();
    weightsAccessor.dispose();
    weightsSampler.dispose();
    animation.dispose();
  }
}