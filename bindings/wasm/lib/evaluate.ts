// Copyright 2022-2025 The Manifold Authors.
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

import {CrossSection, Manifold} from '../manifold-encapsulated-types';
import type {ManifoldToplevel} from '../manifold.d.ts';
import Module from '../manifold.js';

// Instantiate Manifold WASM
const manifoldwasm = await Module();
manifoldwasm.setup();

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

// top level functions exposed in the evaluation context.
const toplevel = [
  'setMinCircularAngle', 'setMinCircularEdgeLength', 'setCircularSegments',
  'getCircularSegments', 'resetToCircularDefaults', 'Mesh', 'Manifold',
  'CrossSection', 'triangulate'
];

/**
 * An object that will evaluate ManifoldCAD scripts on demand.
 *
 * It inserts the Manifold instance (`module`) into the evaluation
 * context, as well as a selection of available methods.  Additional
 * properties can be inserted through `addContext`,
 * `addContextMethodWithCleanup` or can be directly added to the
 * `context` property.
 *
 * This class provides some simple garbage collection.  It does this by
 * intercepting calls to a white-list of functions, tracking new
 * instances of `Manifold` and `CrossSection`.  This way, users don't
 * have to care about calling `delete` manually.  Note that this only
 * fixes memory leak across different runs: the memory will only be freed
 * when `cleanup()` is called.
 *
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

  protected module: ManifoldToplevel = manifoldwasm;
  protected memoryRegistry: Array<Manifold|CrossSection>;

  /**
   * Construct a new evaluator.
   */
  constructor() {
    this.memoryRegistry = new Array<Manifold|CrossSection>();

    this.addMembers('Manifold', manifoldMemberFunctions, false);
    this.addMembers('Manifold', manifoldStaticFunctions, true);
    this.addMembers('CrossSection', crossSectionMemberFunctions, false);
    this.addMembers('CrossSection', crossSectionStaticFunctions, true);
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
   * Clear the evaluation context.
   */
  clearContext() {
    this.context = {};
  }

  /**
   * Add objects to the evaluation context.
   *
   * @param moreContext An object containing properties or methods.
   */
  addContext(moreContext: Record<string, any>) {
    Object.assign(this.context, moreContext)
  }

  /**
   * Add a method to the evaluation context, with cleanup.
   *
   * Calls to the method will be intercepted, and their results
   * added to the cleanup list.  If your function does not
   * generate new Manifold or CrossSection objects, you can
   * add it to the context directly.
   * @param name The name for the method in the context.
   * @param originalFn The function to intercept and include.
   */
  addContextMethodWithCleanup(name: string, originalFn: any) {
    this.context[name] = (...args: any) => {
      //@ts-ignore
      const result = originalFn(...args);
      this.memoryRegistry.push(result);
      return result;
    };
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
  async evaluate(code: string): Promise<any> {
    const exposedFunctions =
        toplevel.map((name) => [name, (this.module as any)[name]]);
    const context = {
      ...Object.fromEntries(exposedFunctions),
      module: this.module,
      ...this.context
    };
    const AsyncFunction =
        Object.getPrototypeOf(async function() {}).constructor;
    const evalFn = new AsyncFunction(
        ...Object.keys(context),
        this.beforeScript + '\n' + code + '\n' + this.afterScript + '\n');
    return await evalFn(...Object.values(context));
  }

  /**
   * Get the instantiated manifold WASM instance owned by this module.
   *
   * Note that function calls that have been intercepted for garbage
   * collection will continue to be intercepted, even outside of the evaluator.
   */
  getModule(): ManifoldToplevel {
    return this.module;
  }
}
