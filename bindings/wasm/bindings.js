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

  // conversion utilities

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

  function vec2polygons(vec, f = x => x) {
    const result = [];
    const nPoly = vec.size();
    for (let i = 0; i < nPoly; i++) {
      const v = vec.get(i);
      const nPts = v.size();
      const poly = [];
      for (let j = 0; j < nPts; j++) {
        poly.push(f(v.get(j)));
      }
      result.push(poly);
    }
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

  function vararg2vec2(vec) {
    if (vec[0] instanceof Array) return {x: vec[0][0], y: vec[0][1]};
    if (typeof (vec[0]) == 'number')
      // default to 0
      return {x: vec[0] || 0, y: vec[1] || 0};
    return vec[0];
  }

  function vararg2vec3(vec) {
    if (vec[0] instanceof Array)
      return {x: vec[0][0], y: vec[0][1], z: vec[0][2]};
    if (typeof (vec[0]) == 'number')
      // default to 0
      return {x: vec[0] || 0, y: vec[1] || 0, z: vec[2] || 0};
    return vec[0];
  }

  function fillRuleToInt(fillRule) {
    return fillRule == 'EvenOdd' ? 0 :
        fillRule == 'NonZero'    ? 1 :
        fillRule == 'Negative'   ? 3 :
                                   /* Positive */ 2;
  }

  function joinTypeToInt(joinType) {
    return joinType == 'Round' ? 1 : joinType == 'Miter' ? 2 : /* Square */ 0;
  }

  // CrossSection methods

  const CrossSectionCtor = Module.CrossSection;

  function cross(polygons, fillRule = 'Positive') {
    if (polygons instanceof CrossSectionCtor) {
      return polygons;
    } else {
      const polygonsVec = polygons2vec(polygons);
      const cs = new CrossSectionCtor(polygonsVec, fillRuleToInt(fillRule));
      disposePolygons(polygonsVec);
      return cs;
    }
  };

  Module.CrossSection.prototype.translate = function(...vec) {
    return this._Translate(vararg2vec2(vec));
  };

  Module.CrossSection.prototype.scale = function(vec) {
    // if only one factor provided, scale both x and y with it
    if (typeof vec == 'number') {
      return this._Scale({x: vec, y: vec});
    }
    return this._Scale(vararg2vec2([vec]));
  };

  Module.CrossSection.prototype.mirror = function(vec) {
    return this._Mirror(vararg2vec2([vec]));
  };

  Module.CrossSection.prototype.warp = function(func) {
    const wasmFuncPtr = addFunction(function(vec2Ptr) {
      const x = getValue(vec2Ptr, 'double');
      const y = getValue(vec2Ptr + 8, 'double');
      const vert = [x, y];
      func(vert);
      setValue(vec2Ptr, vert[0], 'double');
      setValue(vec2Ptr + 8, vert[1], 'double');
    }, 'vi');
    const out = this._Warp(wasmFuncPtr);
    removeFunction(wasmFuncPtr);
    return out;
  };

  Module.CrossSection.prototype.decompose = function() {
    const vec = this._Decompose();
    const result = fromVec(vec);
    vec.delete();
    return result;
  };

  Module.CrossSection.prototype.bounds = function() {
    const result = this._Bounds();
    return {
      min: ['x', 'y'].map(f => result.min[f]),
      max: ['x', 'y'].map(f => result.max[f]),
    };
  };

  Module.CrossSection.prototype.offset = function(
      delta, joinType = 'Square', miterLimit = 2.0, circularSegments = 0) {
    return this._Offset(
        delta, joinTypeToInt(joinType), miterLimit, circularSegments);
  };

  Module.CrossSection.prototype.extrude = function(
      height, nDivisions = 0, twistDegrees = 0.0, scaleTop = [1.0, 1.0],
      center = false) {
    scaleTop = vararg2vec2([scaleTop]);
    const man = Module._Extrude(
        this._ToPolygons(), height, nDivisions, twistDegrees, scaleTop);
    return (center ? man.translate([0., 0., -height / 2.]) : man);
  };

  Module.CrossSection.prototype.revolve = function(
      circularSegments = 0, revolveDegrees = 360.0) {
    return Module._Revolve(
        this._ToPolygons(), circularSegments, revolveDegrees);
  };

  Module.CrossSection.prototype.add = function(other) {
    return this._add(cross(other));
  };

  Module.CrossSection.prototype.subtract = function(other) {
    return this._subtract(cross(other));
  };

  Module.CrossSection.prototype.intersect = function(other) {
    return this._intersect(cross(other));
  };

  Module.CrossSection.prototype.toPolygons = function() {
    const vec = this._ToPolygons();
    const result = vec2polygons(vec, v => [v.x, v.y]);
    vec.delete();
    return result;
  };

  // Manifold methods

  Module.Manifold.prototype.smoothOut = function(
      minSharpAngle = 60, minSmoothness = 0) {
    return this._SmoothOut(minSharpAngle, minSmoothness);
  };

  Module.Manifold.prototype.warp = function(func) {
    const wasmFuncPtr = addFunction(function(vec3Ptr) {
      const x = getValue(vec3Ptr, 'double');
      const y = getValue(vec3Ptr + 8, 'double');
      const z = getValue(vec3Ptr + 16, 'double');
      const vert = [x, y, z];
      func(vert);
      setValue(vec3Ptr, vert[0], 'double');
      setValue(vec3Ptr + 8, vert[1], 'double');
      setValue(vec3Ptr + 16, vert[2], 'double');
    }, 'vi');
    const out = this._Warp(wasmFuncPtr);
    removeFunction(wasmFuncPtr);

    const status = out.status();
    if (status.value !== 0) {
      throw new Module.ManifoldError(status.value);
    }
    return out;
  };

  Module.Manifold.prototype.calculateNormals = function(
      normalIdx, minSharpAngle = 60) {
    return this._CalculateNormals(normalIdx, minSharpAngle);
  };

  Module.Manifold.prototype.asOriginal = function(propertyTolerance = []) {
    const tol = new Module.Vector_f64();
    toVec(tol, propertyTolerance);
    const result = this._AsOriginal(tol);
    tol.delete();
    return result
  };

  Module.Manifold.prototype.setProperties = function(numProp, func) {
    const oldNumProp = this.numProp();
    const wasmFuncPtr = addFunction(function(newPtr, vec3Ptr, oldPtr) {
      const newProp = [];
      for (let i = 0; i < numProp; ++i) {
        newProp[i] = getValue(newPtr + 8 * i, 'double');
      }
      const pos = [];
      for (let i = 0; i < 3; ++i) {
        pos[i] = getValue(vec3Ptr + 8 * i, 'double');
      }
      const oldProp = [];
      for (let i = 0; i < oldNumProp; ++i) {
        oldProp[i] = getValue(oldPtr + 8 * i, 'double');
      }

      func(newProp, pos, oldProp);

      for (let i = 0; i < numProp; ++i) {
        setValue(newPtr + 8 * i, newProp[i], 'double');
      }
    }, 'viii');
    const out = this._SetProperties(numProp, wasmFuncPtr);
    removeFunction(wasmFuncPtr);
    return out;
  };

  Module.Manifold.prototype.translate = function(...vec) {
    return this._Translate(vararg2vec3(vec));
  };

  Module.Manifold.prototype.rotate = function(xOrVec, y, z) {
    if (Array.isArray(xOrVec)) {
      return this._Rotate(...xOrVec);
    } else {
      return this._Rotate(xOrVec, y || 0, z || 0);
    }
  };

  Module.Manifold.prototype.scale = function(vec) {
    // if only one factor provided, scale all three dimensions (xyz) with it
    if (typeof vec == 'number') {
      return this._Scale({x: vec, y: vec, z: vec});
    }
    return this._Scale(vararg2vec3([vec]));
  };

  Module.Manifold.prototype.mirror = function(vec) {
    return this._Mirror(vararg2vec3([vec]));
  };

  Module.Manifold.prototype.trimByPlane = function(normal, offset = 0.) {
    return this._TrimByPlane(vararg2vec3([normal]), offset);
  };

  Module.Manifold.prototype.slice = function(height = 0.) {
    const polygonsVec = this._Slice(height);
    const result = new CrossSectionCtor(polygonsVec, fillRuleToInt('Positive'));
    disposePolygons(polygonsVec);
    return result;
  };

  Module.Manifold.prototype.project = function() {
    const polygonsVec = this._Project();
    const result = new CrossSectionCtor(polygonsVec, fillRuleToInt('Positive'));
    disposePolygons(polygonsVec);
    return result.simplify(this.precision);
  };

  Module.Manifold.prototype.split = function(manifold) {
    const vec = this._Split(manifold);
    const result = fromVec(vec);
    vec.delete();
    return result;
  };

  Module.Manifold.prototype.splitByPlane = function(normal, offset = 0.) {
    const vec = this._SplitByPlane(vararg2vec3([normal]), offset);
    const result = fromVec(vec);
    vec.delete();
    return result;
  };

  Module.Manifold.prototype.decompose = function() {
    const vec = this._Decompose();
    const result = fromVec(vec);
    vec.delete();
    return result;
  };

  Module.Manifold.prototype.boundingBox = function() {
    const result = this._boundingBox();
    return {
      min: ['x', 'y', 'z'].map(f => result.min[f]),
      max: ['x', 'y', 'z'].map(f => result.max[f]),
    };
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

    merge() {
      const {changed, mesh} = Module._Merge(this);
      Object.assign(this, {...mesh});
      return changed;
    }

    verts(tri) {
      return this.triVerts.subarray(3 * tri, 3 * (tri + 1));
    }

    position(vert) {
      return this.vertProperties.subarray(
          this.numProp * vert, this.numProp * vert + 3);
    }

    extras(vert) {
      return this.vertProperties.subarray(
          this.numProp * vert + 3, this.numProp * (vert + 1));
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
      case Module.status.InvalidConstruction.value:
        message = 'Manifold constructed with invalid parameters';
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

  // CrossSection Constructors

  Module.CrossSection = function(polygons, fillRule = 'Positive') {
    const polygonsVec = polygons2vec(polygons);
    const cs = new CrossSectionCtor(polygonsVec, fillRuleToInt(fillRule));
    disposePolygons(polygonsVec);
    return cs;
  };

  Module.CrossSection.ofPolygons = function(polygons, fillRule = 'Positive') {
    return new Module.CrossSection(polygons, fillRule);
  };

  Module.CrossSection.square = function(...args) {
    let size = undefined;
    if (args.length == 0)
      size = {x: 1, y: 1};
    else if (typeof args[0] == 'number')
      size = {x: args[0], y: args[0]};
    else
      size = vararg2vec2(args);
    const center = args[1] || false;
    return Module._Square(size, center);
  };

  Module.CrossSection.circle = function(radius, circularSegments = 0) {
    return Module._Circle(radius, circularSegments);
  };

  // allows args to be either CrossSection or polygons (constructed with
  // Positive fill)
  function crossSectionBatchbool(name) {
    return function(...args) {
      if (args.length == 1) args = args[0];
      const v = new Module.Vector_crossSection();
      for (const cs of args) v.push_back(cross(cs));
      const result = Module['_crossSection' + name](v);
      v.delete();
      return result;
    };
  }

  Module.CrossSection.compose = crossSectionBatchbool('Compose');
  Module.CrossSection.union = crossSectionBatchbool('UnionN');
  Module.CrossSection.difference = crossSectionBatchbool('DifferenceN');
  Module.CrossSection.intersection = crossSectionBatchbool('IntersectionN');

  function pushVec2(vec, ps) {
    toVec(vec, ps, p => {
      if (p instanceof Array) return {x: p[0], y: p[1]};
      return p;
    })
  }

  Module.CrossSection.hull = function(...args) {
    if (args.length == 1) args = args[0];
    let pts = new Module.Vector_vec2();
    for (const cs of args) {
      if (cs instanceof CrossSectionCtor) {
        Module._crossSectionCollectVertices(pts, cs);
      } else if (
          cs instanceof Array && cs.length == 2 && typeof cs[0] == 'number') {
        pts.push_back({x: cs[0], y: cs[1]});
      } else if (cs.x) {
        pts.push_back(cs);
      } else {
        const wrap =
            ((cs[0].length == 2 && typeof cs[0][0] == 'number') || cs[0].x);
        const polys = wrap ? [cs] : cs;
        for (const poly of polys) pushVec2(pts, poly);
      }
    }
    const result = Module._crossSectionHullPoints(pts);
    pts.delete();
    return result;
  };

  Module.CrossSection.prototype = Object.create(CrossSectionCtor.prototype);

  // Because the constructor and prototype are being replaced, instanceof will
  // not work as desired unless we refer back to the original like this
  Object.defineProperty(Module.CrossSection, Symbol.hasInstance, {
    get: () => (t) => {
      return (t instanceof CrossSectionCtor);
    }
  });

  // Manifold Constructors

  const ManifoldCtor = Module.Manifold;
  Module.Manifold = function(mesh) {
    const manifold = new ManifoldCtor(mesh);

    const status = manifold.status();
    if (status.value !== 0) {
      throw new Module.ManifoldError(status.value);
    }

    return manifold;
  };

  Module.Manifold.ofMesh = function(mesh) {
    return new Module.Manifold(mesh);
  };

  Module.Manifold.tetrahedron = function() {
    return Module._Tetrahedron();
  };

  Module.Manifold.cube = function(...args) {
    let size = undefined;
    if (args.length == 0)
      size = {x: 1, y: 1, z: 1};
    else if (typeof args[0] == 'number')
      size = {x: args[0], y: args[0], z: args[0]};
    else
      size = vararg2vec3(args);
    const center = args[1] || false;
    return Module._Cube(size, center);
  };

  Module.Manifold.cylinder = function(
      height, radiusLow, radiusHigh = -1.0, circularSegments = 0,
      center = false) {
    return Module._Cylinder(
        height, radiusLow, radiusHigh, circularSegments, center);
  };

  Module.Manifold.sphere = function(radius, circularSegments = 0) {
    return Module._Sphere(radius, circularSegments);
  };

  Module.Manifold.smooth = function(mesh, sharpenedEdges = []) {
    const sharp = new Module.Vector_smoothness();
    toVec(sharp, sharpenedEdges);
    const result = Module._Smooth(mesh, sharp);
    sharp.delete();
    return result;
  };

  Module.Manifold.extrude = function(
      polygons, height, nDivisions = 0, twistDegrees = 0.0,
      scaleTop = [1.0, 1.0], center = false) {
    const cs = (polygons instanceof CrossSectionCtor) ?
        polygons :
        Module.CrossSection(polygons, 'Positive');
    return cs.extrude(height, nDivisions, twistDegrees, scaleTop, center);
  };

  Module.Manifold.revolve = function(
      polygons, circularSegments = 0, revolveDegrees = 360.0) {
    const cs = (polygons instanceof CrossSectionCtor) ?
        polygons :
        Module.CrossSection(polygons, 'Positive');
    return cs.revolve(circularSegments, revolveDegrees);
  };

  Module.Manifold.reserveIDs = function(n) {
    return Module._ReserveIDs(n);
  };

  Module.Manifold.compose = function(manifolds) {
    const vec = new Module.Vector_manifold();
    toVec(vec, manifolds);
    const result = Module._manifoldCompose(vec);
    vec.delete();
    return result;
  };

  function manifoldBatchbool(name) {
    return function(...args) {
      if (args.length == 1) args = args[0];
      const v = new Module.Vector_manifold();
      for (const m of args) v.push_back(m);
      const result = Module['_manifold' + name + 'N'](v);
      v.delete();
      return result;
    };
  }

  Module.Manifold.union = manifoldBatchbool('Union');
  Module.Manifold.difference = manifoldBatchbool('Difference');
  Module.Manifold.intersection = manifoldBatchbool('Intersection');

  Module.Manifold.levelSet = function(
      sdf, bounds, edgeLength, level = 0, precision = -1) {
    const bounds2 = {
      min: {x: bounds.min[0], y: bounds.min[1], z: bounds.min[2]},
      max: {x: bounds.max[0], y: bounds.max[1], z: bounds.max[2]},
    };
    const wasmFuncPtr = addFunction(function(vec3Ptr) {
      const x = getValue(vec3Ptr, 'double');
      const y = getValue(vec3Ptr + 8, 'double');
      const z = getValue(vec3Ptr + 16, 'double');
      const vert = [x, y, z];
      return sdf(vert);
    }, 'di');
    const out =
        Module._LevelSet(wasmFuncPtr, bounds2, edgeLength, level, precision);
    removeFunction(wasmFuncPtr);
    return out;
  };

  function pushVec3(vec, ps) {
    toVec(vec, ps, p => {
      if (p instanceof Array) return {x: p[0], y: p[1], z: p[2]};
      return p;
    })
  }

  Module.Manifold.hull = function(...args) {
    if (args.length == 1) args = args[0];
    let pts = new Module.Vector_vec3();
    for (const m of args) {
      if (m instanceof ManifoldCtor) {
        Module._manifoldCollectVertices(pts, m);
      } else if (
          m instanceof Array && m.length == 3 && typeof m[0] == 'number') {
        pts.push_back({x: m[0], y: m[1], z: m[2]});
      } else if (m.x) {
        pts.push_back(m);
      } else {
        pushVec3(pts, m);
      }
    }
    const result = Module._manifoldHullPoints(pts);
    pts.delete();
    return result;
  };

  Module.Manifold.prototype = Object.create(ManifoldCtor.prototype);

  // Because the constructor and prototype are being replaced, instanceof will
  // not work as desired unless we refer back to the original like this
  Object.defineProperty(Module.Manifold, Symbol.hasInstance, {
    get: () => (t) => {
      return (t instanceof ManifoldCtor);
    }
  });

  // Top-level functions

  Module.triangulate = function(polygons, precision = -1) {
    const polygonsVec = polygons2vec(polygons);
    const result = fromVec(
        Module._Triangulate(polygonsVec, precision), (x) => [x[0], x[1], x[2]]);
    disposePolygons(polygonsVec);
    return result;
  };
};
