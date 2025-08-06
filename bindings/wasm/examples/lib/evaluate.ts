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

import * as glMatrix from 'gl-matrix';

import {CrossSection, Manifold} from '../built/manifold';

import {exportModels, GlobalDefaults} from './export';

var _module = null as any;

// Faster on modern browsers than Float32Array
glMatrix.glMatrix.setMatrixArrayType(Array);

// manifold static methods (that return a new manifold)
const manifoldStaticFunctions = [
  'cube', 'cylinder', 'sphere', 'tetrahedron', 'extrude', 'revolve', 'compose',
  'union', 'difference', 'intersection', 'levelSet', 'smooth', 'ofMesh', 'hull'
];
// manifold member functions (that return a new manifold)
const manifoldMemberFunctions = [
  'add',
  'subtract',
  'intersect',
  'decompose',
  'warp',
  'transform',
  'translate',
  'rotate',
  'scale',
  'mirror',
  'calculateCurvature',
  'calculateNormals',
  'smoothByNormals',
  'smoothOut',
  'refine',
  'refineToLength',
  'refineToTolerance',
  'setProperties',
  'setTolerance',
  'simplify',
  'asOriginal',
  'trimByPlane',
  'split',
  'splitByPlane',
  'slice',
  'project',
  'hull'
];
// CrossSection static methods (that return a new cross-section)
const crossSectionStaticFunctions = [
  'square', 'circle', 'union', 'difference', 'intersection', 'compose',
  'ofPolygons', 'hull'
];
// CrossSection member functions (that return a new cross-section)
const crossSectionMemberFunctions = [
  'add', 'subtract', 'intersect', 'rectClip', 'decompose', 'transform',
  'translate', 'rotate', 'scale', 'mirror', 'simplify', 'offset', 'hull'
];
// top level functions that construct a new manifold/mesh
const toplevelConstructors = ['show', 'only', 'setMaterial'];
const toplevel = [
  'setMinCircularAngle', 'setMinCircularEdgeLength', 'setCircularSegments',
  'getCircularSegments', 'resetToCircularDefaults', 'Mesh', 'GLTFNode',
  'Manifold', 'CrossSection', 'setMorphStart', 'setMorphEnd'
];
const exposedFunctions = toplevelConstructors.concat(toplevel);

// Setup memory management, such that users don't have to care about
// calling `delete` manually.
// Note that this only fixes memory leak across different runs: the memory
// will only be freed when the compilation finishes.

const memoryRegistry = new Array<Manifold|CrossSection>();

export function setup(module: any) {
  _module = module;

  function addMembers(
      className: string, methodNames: Array<string>, areStatic: boolean) {
    //@ts-ignore
    const cls = module[className];
    const obj = areStatic ? cls : cls.prototype;
    for (const name of methodNames) {
      const originalFn = obj[name];
      obj[name] = function(...args: any) {
        //@ts-ignore
        const result = originalFn(...args);
        memoryRegistry.push(result);
        return result;
      };
    }
  }

  addMembers('Manifold', manifoldMemberFunctions, false);
  addMembers('Manifold', manifoldStaticFunctions, true);
  addMembers('CrossSection', crossSectionMemberFunctions, false);
  addMembers('CrossSection', crossSectionStaticFunctions, true);

  for (const name of toplevelConstructors) {
    //@ts-ignore
    const originalFn = module[name];
    //@ts-ignore
    module[name] = function(...args: any) {
      const result = originalFn(...args);
      memoryRegistry.push(result);
      return result;
    };
  }
}

export function cleanup() {
  for (const obj of memoryRegistry) {
    // decompose result is an array of manifolds
    if (obj instanceof Array)
      for (const elem of obj) elem.delete();
    else
      obj.delete();
  }
  memoryRegistry.length = 0;
}

export function evaluateCADToManifold(code: string) {
  const globalDefaults = {} as GlobalDefaults;
  const module = _module;
  const context = {
    globalDefaults,
    exportModels,
    glMatrix,
    module,
    ...Object.fromEntries(
        exposedFunctions.map((name) => [name, (module as any)[name]]),
        ),
  };
  const evalFn = new Function(
      ...Object.keys(context),
      'resetToCircularDefaults();\n' + code +
          '\n return typeof result === "undefined" ? undefined : result;',
  );
  const manifold = evalFn(...Object.values(context));
  return {globalDefaults, manifold};
}
