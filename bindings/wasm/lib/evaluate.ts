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

import {CrossSection, Manifold, ManifoldToplevel} from '../examples/built/manifold';

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

/**
 * An object that will evaluate ManifoldCAD scripts on demand.
 *
 * It inserts the Manifold instance (`module`) into the evaluation
 * context, as well as a selection of available methods.  Additional
 * values can be directly inserted into the `context` property.
 *
 * This class provides some simple garbage collection.  It does this by
 * intercepting calls to a white-list of functions, tracking new
 * instances of `Manifold` and `CrossSection`.  This way, users don't
 * have to care about calling `delete` manually.  Note that this only
 * fixes memory leak across different runs: the memory will only be freed
 * when `cleanup()` is called.
 *
 * @param module A Manifold WASM instance, already set up.
 * @property context Additional objects inserted into the evaluation
 * context.
 * @property beforeScript Boilerplate script run before the supplied
 * code.
 * @property afterScript Boilerplate code run after the supplied code.
 */
export class Evaluator {
  context: any = {};
  beforeScript: string = 'resetToCircularDefaults();';
  afterScript: string =
      'return typeof result === "undefined" ? undefined : result;';

  protected module?: ManifoldToplevel;
  protected memoryRegistry: Array<Manifold|CrossSection>;


  /**
   * Construct a new evaluator.
   *
   */
  constructor(module: ManifoldToplevel) {
    this.module = module;
    this.memoryRegistry = new Array<Manifold|CrossSection>();

    this.addMembers('Manifold', manifoldMemberFunctions, false);
    this.addMembers('Manifold', manifoldStaticFunctions, true);
    this.addMembers('CrossSection', crossSectionMemberFunctions, false);
    this.addMembers('CrossSection', crossSectionStaticFunctions, true);

    for (const name of toplevelConstructors) {
      //@ts-ignore
      const originalFn = module[name];
      //@ts-ignore
      this.module[name] = (...args: any) => {
        const result = originalFn(...args);
        this.memoryRegistry.push(result);
        return result;
      };
    }
  }

  /**
   * Intercept calls and add their results to our garbage collection
   * list.
   *
   * @param className The class to intercept.
   * @param methodNames An array of methods to intercept.
   * @param areStatic Are these static methods?  If so, intercept them at
   * the prototype level.
   */
  protected addMembers(
      className: string, methodNames: Array<string>, areStatic: boolean) {
    //@ts-ignore
    const cls = this.module[className];
    const obj = areStatic ? cls : cls.prototype;
    for (const name of methodNames) {
      const originalFn = obj[name];
      obj[name] = (...args: any) => {
        //@ts-ignore
        const result = originalFn(...args);
        this.memoryRegistry.push(result);
        return result;
      };
    }
  }

  /**
   * Delete any objects tagged for garbage collection.
   */
  cleanup() {
    for (const obj of this.memoryRegistry) {
      // decompose result is an array of manifolds
      if (obj instanceof Array)
        for (const elem of obj) elem.delete();
      else
        obj.delete();
    }
    this.memoryRegistry.length = 0;
  }

  /**
   * Evaluate a string as javascript code creating a Manifold model.
   *
   * This function assembles the final execution context.  It then runs
   * `beforeScript`, `code` and `afterScript` in order.  Finally, it
   * returns the end result.
   *
   * @param code The input string.
   * @returns any By default, this script will return either `undefined`
   * or a `Manifold` object.  Changing `afterScript` will affect this
   * behaviour.
   */
  evaluate(code: string) {
    const exposedFunctions = toplevelConstructors.concat(toplevel).map(
        (name) => [name, (this.module as any)[name]]);
    const context = {
      ...Object.fromEntries(exposedFunctions),
      module: this.module,
      ...this.context
    };

    const evalFn = new Function(
        ...Object.keys(context),
        this.beforeScript + '\n' + code + '\n' + this.afterScript + '\n');
    return evalFn(...Object.values(context));
  }
}
