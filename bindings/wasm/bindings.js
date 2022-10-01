var _ManifoldInitialized = false;
Module.setup = function () {
  if (_ManifoldInitialized) return;
  _ManifoldInitialized = true;

  function toVec(vec, list, f = x => x) {
    for (let x of list) {
      vec.push_back(f(x));
    }
    return vec;
  }

  function fromVec(vec, f = x => x) {
    let result = [];
    const size = vec.size();
    for (let i = 0; i < size; i++)
      result.push(f(vec.get(i)));
    return result;
  }

  function polygons2vec(polygons) {
    return toVec(
      new Module.Vector2_vec2(), polygons,
      poly => toVec(new Module.Vector_vec2(), poly, p => {
        if (p instanceof Array) return { x: p[0], y: p[1] };
        return p;
      }));
  }

  function vararg2vec(vec) {
    if (vec[0] instanceof Array)
      return { x: vec[0][0], y: vec[0][1], z: vec[0][2] };
    if (typeof (vec[0]) == 'number')
      // default to 0
      return { x: vec[0] || 0, y: vec[1] || 0, z: vec[2] || 0 };
    return vec[0];
  }

  Module.Manifold.prototype.warp = function (func) {
    const wasmFuncPtr = addFunction(function (vec3Ptr) {
      const x = getValue(vec3Ptr, 'float');
      const y = getValue(vec3Ptr + 1, 'float');
      const z = getValue(vec3Ptr + 2, 'float');
      const vert = [x, y, z];
      func(vert);
      setValue(vec3Ptr, vert[0], 'float');
      setValue(vec3Ptr + 1, vert[1], 'float');
      setValue(vec3Ptr + 2, vert[2], 'float');
    }, 'vi');
    const out = this._Warp(wasmFuncPtr);
    removeFunction(wasmFuncPtr);
    return out;
  };

  // note that the matrix is using column major (same as glm)
  Module.Manifold.prototype.transform = function (mat) {
    console.assert(mat.length == 4, 'expects a 3x4 matrix');
    let vec = new Module.Vector_f32();
    for (let col of mat) {
      console.assert(col.length == 3, 'expects a 3x4 matrix');
      for (let x of col) mat.push_back(x);
    }
    const result = this._Transform(vec);
    vec.delete();
    return result;
  };

  Module.Manifold.prototype.translate = function (...vec) {
    return this._Translate(vararg2vec(vec));
  };

  Module.Manifold.prototype.rotate = function (vec) {
    return this._Rotate(...vec);
  };

  Module.Manifold.prototype.scale = function (...vec) {
    return this._Scale(vararg2vec(vec));
  };

  Module.Manifold.prototype.smooth = function (sharpenedEdges = []) {
    let vec = new Module.Vector_smoothness();
    toVec(vec, sharpenedEdges);
    const result = this._Smooth(vec);
    vec.delete();
    return result;
  };

  Module.Manifold.prototype.decompose = function () {
    const vec = this._Decompose();
    const result = fromVec(vec);
    vec.delete();
    return result;
  };

  Module.Manifold.prototype.getCurvature = function () {
    let result = this._getCurvature();
    let oldMeanCurvature = result.vertMeanCurvature;
    let oldGaussianCurvature = result.vertGaussianCurvature;
    result.vertMeanCurvature = fromVec(oldMeanCurvature);
    result.vertGaussianCurvature = fromVec(oldGaussianCurvature);
    oldMeanCurvature.delete();
    oldGaussianCurvature.delete();
    return result;
  };

  Module.Manifold.prototype.getMeshRelation = function () {
    let result = this._getMeshRelation();
    let oldBarycentric = result.barycentric;
    let oldTriBary = result.triBary;
    let conversion1 = v => ['x', 'y', 'z'].map(f => v[f]);
    let conversion2 = v => {
      return {
        meshID: v.meshID,
        originalID: v.originalID,
        tri: v.tri,
        vertBary: conversion1(v.vertBary)
      };
    };
    result.barycentric = fromVec(oldBarycentric, conversion1);
    result.triBary = fromVec(oldTriBary, conversion2);
    oldBarycentric.delete();
    oldTriBary.delete();
    return result;
  };

  Module.Manifold.prototype.boundingBox = function () {
    const result = this._boundingBox();
    return {
      min: ['x', 'y', 'z'].map(f => result.min[f]),
      max: ['x', 'y', 'z'].map(f => result.max[f]),
    };
  };

  Module.cube = function (...args) {
    let size = undefined;
    if (args.length == 0)
      size = { x: 1, y: 1, z: 1 };
    else if (typeof args[0] == 'number')
      size = { x: args[0], y: args[0], z: args[0] };
    else
      size = vararg2vec(args);
    const center = args[1] || false;
    return Module._Cube(size, center);
  };

  Module.cylinder = function (
    height, radiusLow, radiusHigh = -1.0, circularSegments = 0,
    center = false) {
    return Module._Cylinder(
      height, radiusLow, radiusHigh, circularSegments, center);
  };

  Module.sphere = function (radius, circularSegments = 0) {
    return Module._Sphere(radius, circularSegments);
  };

  Module.extrude = function (
    polygons, height, nDivisions = 0, twistDegrees = 0.0,
    scaleTop = [1.0, 1.0]) {
    if (scaleTop instanceof Array) scaleTop = { x: scaleTop[0], y: scaleTop[1] };
    const polygonsVec = polygons2vec(polygons);
    const result = Module._Extrude(
      polygonsVec, height, nDivisions, twistDegrees, scaleTop);
    for (let i = 0; i < polygonsVec.size(); i++) polygonsVec.get(i).delete();
    polygonsVec.delete();
    return result;
  };

  Module.revolve = function (polygons, circularSegments = 0) {
    const polygonsVec = polygons2vec(polygons);
    const result = Module._Revolve(polygonsVec, circularSegments);
    for (let i = 0; i < polygonsVec.size(); i++) polygonsVec.get(i).delete();
    polygonsVec.delete();
    return result;
  };

  Module.compose = function (manifolds) {
    let vec = new Module.Vector_manifold();
    toVec(vec, manifolds);
    const result = Module._Compose(vec);
    vec.delete();
    return result;
  };

  Module.levelSet = function (sdf, bounds, edgeLength, level = 0) {
    let bounds2 = {
      min: { x: bounds.min[0], y: bounds.min[1], z: bounds.min[2] },
      max: { x: bounds.max[0], y: bounds.max[1], z: bounds.max[2] },
    };
    const wasmFuncPtr = addFunction(function (vec3Ptr) {
      const x = getValue(vec3Ptr, 'float');
      const y = getValue(vec3Ptr + 1, 'float');
      const z = getValue(vec3Ptr + 2, 'float');
      const vert = [x, y, z];
      return sdf(vert);
    }, 'fi');
    const out = Module._LevelSet(wasmFuncPtr, bounds2, edgeLength, level);
    removeFunction(wasmFuncPtr);
    return out;
  }

  function batchbool(name) {
    return function (...args) {
      if (args.length == 1)
        args = args[0];
      let v = new Module.Vector_manifold();
      for (const m of args)
        v.push_back(m);
      const result = Module['_' + name + 'N'](v);
      v.delete();
      return result;
    }
  }

  Module.union = batchbool('union');
  Module.difference = batchbool('difference');
  Module.intersection = batchbool('intersection');
};
