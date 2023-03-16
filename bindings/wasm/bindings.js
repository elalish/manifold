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

var _ManifoldInitialized = false;
Module.setup = function() {
  if (_ManifoldInitialized) return;
  _ManifoldInitialized = true;

  function toVec(vec, list, f = x => x) {
    if (list) {
      for (let x of list) {
        vec.push_back(f(x));
      }
    }
    return vec;
  }

  function fromVec(vec, f = x => x) {
    const result = [];
    const size = vec.size();
    for (let i = 0; i < size; i++) result.push(f(vec.get(i)));
    return result;
  }

  function polygons2vec(polygons) {
    if (polygons[0].length < 3) {
      polygons = [polygons];
    }
    return toVec(
        new Module.Vector2_vec2(), polygons,
        poly => toVec(new Module.Vector_vec2(), poly, p => {
          if (p instanceof Array) return {x: p[0], y: p[1]};
          return p;
        }));
  }

  function disposePolygons(polygonsVec) {
    for (let i = 0; i < polygonsVec.size(); i++) polygonsVec.get(i).delete();
    polygonsVec.delete();
  }

  function vararg2vec(vec) {
    if (vec[0] instanceof Array)
      return {x: vec[0][0], y: vec[0][1], z: vec[0][2]};
    if (typeof (vec[0]) == 'number')
      // default to 0
      return {x: vec[0] || 0, y: vec[1] || 0, z: vec[2] || 0};
    return vec[0];
  }

  Module.Manifold.prototype.warp = function(func) {
    const wasmFuncPtr = addFunction(function(vec3Ptr) {
      const x = getValue(vec3Ptr, 'float');
      const y = getValue(vec3Ptr + 4, 'float');
      const z = getValue(vec3Ptr + 8, 'float');
      const vert = [x, y, z];
      func(vert);
      setValue(vec3Ptr, vert[0], 'float');
      setValue(vec3Ptr + 4, vert[1], 'float');
      setValue(vec3Ptr + 8, vert[2], 'float');
    }, 'vi');
    const out = this._Warp(wasmFuncPtr);
    removeFunction(wasmFuncPtr);
    return out;
  };

  Module.Manifold.prototype.transform = function(mat) {
    const vec = new Module.Vector_f32();
    console.assert(mat.length == 16, 'expects a 4x4 matrix');
    // assuming glMatrix format (column major)
    // skip the last row
    for (let i = 0; i < 16; i++)
      if (i % 4 != 3) vec.push_back(mat[i]);

    const result = this._Transform(vec);
    vec.delete();
    return result;
  };

  Module.Manifold.prototype.translate = function(...vec) {
    return this._Translate(vararg2vec(vec));
  };

  Module.Manifold.prototype.rotate = function(vec) {
    return this._Rotate(...vec);
  };

  Module.Manifold.prototype.scale = function(vec) {
    if (typeof vec == 'number') {
      return this._Scale({x: vec, y: vec, z: vec});
    }
    return this._Scale(vararg2vec([vec]));
  };

  Module.Manifold.prototype.decompose = function() {
    const vec = this._Decompose();
    const result = fromVec(vec);
    vec.delete();
    return result;
  };

  Module.Manifold.prototype.getCurvature = function() {
    const result = this._getCurvature();
    const oldMeanCurvature = result.vertMeanCurvature;
    const oldGaussianCurvature = result.vertGaussianCurvature;
    result.vertMeanCurvature = fromVec(oldMeanCurvature);
    result.vertGaussianCurvature = fromVec(oldGaussianCurvature);
    oldMeanCurvature.delete();
    oldGaussianCurvature.delete();
    return result;
  };

  class Mesh {
    constructor({
      numProp = 3,
      triVerts = new Uint32Array(),
      vertProperties = new Float32Array(),
      mergeFromVert,
      mergeToVert,
      runIndex,
      runOriginalID,
      faceID,
      halfedgeTangent,
      runTransform
    } = {}) {
      this.numProp = numProp;
      this.triVerts = triVerts;
      this.vertProperties = vertProperties;
      this.mergeFromVert = mergeFromVert;
      this.mergeToVert = mergeToVert;
      this.runIndex = runIndex;
      this.runOriginalID = runOriginalID;
      this.faceID = faceID;
      this.halfedgeTangent = halfedgeTangent;
      this.runTransform = runTransform;
    }

    get numTri() {
      return this.triVerts.length / 3;
    }

    get numVert() {
      return this.vertProperties.length / this.numProp;
    }

    get numRun() {
      return this.runOriginalID.length;
    }

    verts(tri) {
      return this.triVerts.subarray(3 * tri, 3 * (tri + 1));
    }

    position(vert) {
      return this.vertProperties.subarray(numProp * vert, numProp * vert + 3);
    }

    extras(vert) {
      return this.vertProperties.subarray(
          numProp * vert + 3, numProp * (vert + 1));
    }

    tangent(halfedge) {
      return this.halfedgeTangent.subarray(4 * halfedge, 4 * (halfedge + 1));
    }

    transform(run) {
      const mat4 = new Array(16);
      for (const col of [0, 1, 2, 3]) {
        for (const row of [0, 1, 2]) {
          mat4[4 * col + row] = this.runTransform[12 * run + 3 * col + row];
        }
      }
      mat4[15] = 1;
      return mat4;
    }
  }

  Module.Mesh = Mesh;

  Module.Manifold.prototype.getMesh = function(normalIdx = [0, 0, 0]) {
    if (normalIdx instanceof Array)
      normalIdx = {0: normalIdx[0], 1: normalIdx[1], 2: normalIdx[2]};
    return new Mesh(this._GetMeshJS(normalIdx));
  };

  Module.Manifold.prototype.boundingBox = function() {
    const result = this._boundingBox();
    return {
      min: ['x', 'y', 'z'].map(f => result.min[f]),
      max: ['x', 'y', 'z'].map(f => result.max[f]),
    };
  };

  Module.ManifoldError = function ManifoldError(code, ...args) {
    let message = 'Unknown error';
    switch (code) {
      case Module.status.NonFiniteVertex.value:
        message = 'Non-finite vertex';
        break;
      case Module.status.NotManifold.value:
        message = 'Not manifold';
        break;
      case Module.status.VertexOutOfBounds.value:
        message = 'Vertex index out of bounds';
        break;
      case Module.status.PropertiesWrongLength.value:
        message = 'Properties have wrong length';
        break;
      case Module.status.MissingPositionProperties.value:
        message = 'Less than three properties';
        break;
      case Module.status.MergeVectorsDifferentLengths.value:
        message = 'Merge vectors have different lengths';
        break;
      case Module.status.MergeIndexOutOfBounds.value:
        message = 'Merge index out of bounds';
        break;
      case Module.status.TransformWrongLength.value:
        message = 'Transform vector has wrong length';
        break;
      case Module.status.RunIndexWrongLength.value:
        message = 'Run index vector has wrong length';
        break;
      case Module.status.FaceIDWrongLength.value:
        message = 'Face ID vector has wrong length';
    }

    const base = Error.apply(this, [message, ...args]);
    base.name = this.name = 'ManifoldError';
    this.message = base.message;
    this.stack = base.stack;
    this.code = code;
  };

  Module.ManifoldError.prototype = Object.create(Error.prototype, {
    constructor:
        {value: Module.ManifoldError, writable: true, configurable: true}
  });

  const ManifoldCtor = Module.Manifold;
  Module.Manifold = function(mesh) {
    const manifold = new ManifoldCtor(mesh);

    const status = manifold.status();
    if (status.value !== 0) {
      throw new Module.ManifoldError(status.value);
    }

    return manifold;
  };

  Module.Manifold.prototype = Object.create(ManifoldCtor.prototype);

  Module.cube = function(...args) {
    let size = undefined;
    if (args.length == 0)
      size = {x: 1, y: 1, z: 1};
    else if (typeof args[0] == 'number')
      size = {x: args[0], y: args[0], z: args[0]};
    else
      size = vararg2vec(args);
    const center = args[1] || false;
    return Module._Cube(size, center);
  };

  Module.cylinder = function(
      height, radiusLow, radiusHigh = -1.0, circularSegments = 0,
      center = false) {
    return Module._Cylinder(
        height, radiusLow, radiusHigh, circularSegments, center);
  };

  Module.sphere = function(radius, circularSegments = 0) {
    return Module._Sphere(radius, circularSegments);
  };

  Module.smooth = function(mesh, sharpenedEdges = []) {
    const sharp = new Module.Vector_smoothness();
    toVec(sharp, sharpenedEdges);
    const result = Module._Smooth(mesh, sharp);
    sharp.delete();
    return result;
  };

  Module.extrude = function(
      polygons, height, nDivisions = 0, twistDegrees = 0.0,
      scaleTop = [1.0, 1.0]) {
    if (scaleTop instanceof Array) scaleTop = {x: scaleTop[0], y: scaleTop[1]};
    const polygonsVec = polygons2vec(polygons);
    const result = Module._Extrude(
        polygonsVec, height, nDivisions, twistDegrees, scaleTop);
    disposePolygons(polygonsVec);
    return result;
  };

  Module.revolve = function(polygons, circularSegments = 0) {
    const polygonsVec = polygons2vec(polygons);
    const result = Module._Revolve(polygonsVec, circularSegments);
    disposePolygons(polygonsVec);
    return result;
  };

  Module.compose = function(manifolds) {
    const vec = new Module.Vector_manifold();
    toVec(vec, manifolds);
    const result = Module._Compose(vec);
    vec.delete();
    return result;
  };

  Module.levelSet = function(sdf, bounds, edgeLength, level = 0) {
    const bounds2 = {
      min: {x: bounds.min[0], y: bounds.min[1], z: bounds.min[2]},
      max: {x: bounds.max[0], y: bounds.max[1], z: bounds.max[2]},
    };
    const wasmFuncPtr = addFunction(function(vec3Ptr) {
      const x = getValue(vec3Ptr, 'float');
      const y = getValue(vec3Ptr + 4, 'float');
      const z = getValue(vec3Ptr + 8, 'float');
      const vert = [x, y, z];
      return sdf(vert);
    }, 'fi');
    const out = Module._LevelSet(wasmFuncPtr, bounds2, edgeLength, level);
    removeFunction(wasmFuncPtr);
    return out;
  };

  function batchbool(name) {
    return function(...args) {
      if (args.length == 1) args = args[0];
      const v = new Module.Vector_manifold();
      for (const m of args) v.push_back(m);
      const result = Module['_' + name + 'N'](v);
      v.delete();
      return result;
    };
  }

  Module.union = batchbool('union');
  Module.difference = batchbool('difference');
  Module.intersection = batchbool('intersection');
};
