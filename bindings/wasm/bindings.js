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

  // warmup tbb for emscripten, according to
  // https://github.com/oneapi-src/oneTBB/blob/master/WASM_Support.md#limitations
  Module.initTBB();
  // conversion utilities

  function toVec(vec, list, f = x => x) {
    if (list) {
      for (let x of list) {
        vec.push_back(f(x));
      }
    }
    return vec;
  }

  // A number sets the segments, an object any of the CircularQuality fields.
  function toQuality(quality = 0) {
    if (typeof quality != 'object') quality = {segments: quality};
    return {...Module._DefaultCircularQuality(), ...quality};
  }

  Module.segmentsFor = function(radius, quality) {
    return Module._SegmentsFor(toQuality(quality), radius);
  };

  Module.forDivisions = function(divisions, degrees) {
    return Module._ForDivisions(divisions, degrees);
  };

  function fromVec(vec, f = x => x) {
    const result = [];
    const size = vec.size();
    for (let i = 0; i < size; i++) result.push(f(vec.get(i)));
    return result;
  }

  function withCallback(makeCallback, signature, operation) {
    const state = {failed: false, failure: undefined};
    const pointer = addFunction(makeCallback(state), signature);
    let result;
    try {
      result = operation(pointer);
    } finally {
      removeFunction(pointer);
    }
    if (state.failed) {
      result.delete();
      throw state.failure;
    }
    return result;
  }

  function checkedManifold(manifold) {
    const status = manifold.status();
    if (status !== 'NoError') {
      manifold.delete();
      throw new Module.ManifoldError(status);
    }
    return manifold;
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
      v.delete();
      result.push(poly);
    }
    return result;
  }

  // A point is [x, y] or {x, y}.
  function isPoint(p) {
    return typeof p?.[0] == 'number' || typeof p?.x == 'number';
  }

  // Polygons is either one contour, whose first element is a point, or a list
  // of contours.
  function polygons2vec(polygons) {
    if (isPoint(polygons[0])) polygons = [polygons];
    const polygonsVec = new Module.Vector2_vec2();
    for (const poly of polygons) {
      const polyVec = toVec(new Module.Vector_vec2(), poly, p => {
        if (p instanceof Array) return {x: p[0], y: p[1]};
        return p;
      });
      polygonsVec.push_back(polyVec);
      polyVec.delete();
    }
    return polygonsVec;
  }

  function vararg2vec2(vec) {
    if (vec[0] instanceof Array) return {x: vec[0][0], y: vec[0][1]};
    if (typeof (vec[0]) == 'number')
      // missing components default to 0
      return {x: vec[0], y: vec[1] ?? 0};
    return vec[0];
  }

  function vararg2vec3(vec) {
    if (vec[0] instanceof Array)
      return {x: vec[0][0], y: vec[0][1], z: vec[0][2]};
    if (typeof (vec[0]) == 'number')
      // missing components default to 0
      return {x: vec[0], y: vec[1] ?? 0, z: vec[2] ?? 0};
    return vec[0];
  }

  function joinTypeToInt(joinType) {
    return joinType == 'Round' ? 1 :
        joinType == 'Miter'    ? 2 :
        joinType == 'Bevel'    ? 3 :
                                 /* Square */ 0;
  }

  // CrossSection methods

  const CrossSectionCtor = Module.CrossSection;

  function withCrossSection(polygons, f) {
    if (polygons instanceof CrossSectionCtor) return f(polygons);
    const polygonsVec = polygons2vec(polygons);
    const cs = new CrossSectionCtor(polygonsVec);
    polygonsVec.delete();
    const result = f(cs);
    cs.delete();
    return result;
  }

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
    return withCallback(state => function(vec2Ptr) {
      if (state.failed) return;
      try {
        const x = getValue(vec2Ptr, 'double');
        const y = getValue(vec2Ptr + 8, 'double');
        const vert = [x, y];
        func(vert);
        setValue(vec2Ptr, vert[0], 'double');
        setValue(vec2Ptr + 8, vert[1], 'double');
      } catch (error) {
        // Return normally so C++ can destroy its local allocations.
        state.failed = true;
        state.failure = error;
      }
    }, 'vi', pointer => this._Warp(pointer));
  };

  Module.CrossSection.prototype.warpBatch = function(func) {
    return withCallback(state => function(ptr, count) {
      if (state.failed) return;
      try {
        const heapF64 = Module.HEAPF64 ?? HEAPF64;
        if (!heapF64) {
          throw new Error('WASM heap is not initialized (HEAPF64 unavailable)');
        }
        // Commit only after success, as in scalar warp. The temporary also
        // stays valid if the callback allocates enough to grow WASM memory.
        const verts = new Float64Array(heapF64.buffer, ptr, count * 2).slice();
        func(verts, count);
        const currentHeap = Module.HEAPF64 ?? HEAPF64;
        new Float64Array(currentHeap.buffer, ptr, count * 2).set(verts);
      } catch (error) {
        // Return normally so C++ can destroy its local allocations.
        state.failed = true;
        state.failure = error;
      }
    }, 'vii', pointer => this._WarpBatch(pointer));
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
      delta, joinType = 'Round', miterLimit = 2.0, quality = 0) {
    return this._Offset(
        delta, joinTypeToInt(joinType), miterLimit, toQuality(quality));
  };

  Module.CrossSection.prototype.simplify = function(tolerance = 0) {
    return this._Simplify(tolerance);
  };

  Module.CrossSection.prototype.extrude = function(
      height, nDivisions = 0, twistDegrees = 0.0, scaleTop = [1.0, 1.0],
      center = false) {
    // if only one factor provided, scale both x and y with it
    if (typeof scaleTop == 'number') {
      scaleTop = [scaleTop, scaleTop];
    }
    scaleTop = vararg2vec2([scaleTop]);
    const polygonsVec = this._ToPolygons();
    const man = Module._Extrude(
        polygonsVec, height, nDivisions, twistDegrees, scaleTop);
    polygonsVec.delete();
    if (!center) return man;
    const centered = man.translate([0., 0., -height / 2.]);
    man.delete();
    return centered;
  };

  Module.CrossSection.prototype.revolve = function(
      quality = 0, revolveDegrees = 360.0) {
    const polygonsVec = this._ToPolygons();
    const man =
        Module._Revolve(polygonsVec, toQuality(quality), revolveDegrees);
    polygonsVec.delete();
    return man;
  };

  Module.CrossSection.prototype.add = function(other) {
    return withCrossSection(other, cs => this._add(cs));
  };

  Module.CrossSection.prototype.subtract = function(other) {
    return withCrossSection(other, cs => this._subtract(cs));
  };

  Module.CrossSection.prototype.intersect = function(other) {
    return withCrossSection(other, cs => this._intersect(cs));
  };

  Module.CrossSection.prototype.toPolygons = function() {
    const vec = this._ToPolygons();
    const result = vec2polygons(vec, v => [v.x, v.y]);
    vec.delete();
    return result;
  };

  // Manifold methods

  Module.Manifold.prototype.warp = function(func) {
    const out = withCallback(state => function(vec3Ptr) {
      if (state.failed) return;
      try {
        const x = getValue(vec3Ptr, 'double');
        const y = getValue(vec3Ptr + 8, 'double');
        const z = getValue(vec3Ptr + 16, 'double');
        const vert = [x, y, z];
        func(vert);
        setValue(vec3Ptr, vert[0], 'double');
        setValue(vec3Ptr + 8, vert[1], 'double');
        setValue(vec3Ptr + 16, vert[2], 'double');
      } catch (error) {
        // Return normally so C++ can destroy its local allocations.
        state.failed = true;
        state.failure = error;
      }
    }, 'vi', pointer => this._Warp(pointer));
    return checkedManifold(out);
  };

  Module.Manifold.prototype.warpBatch = function(func) {
    const out = withCallback(state => function(ptr, count) {
      if (state.failed) return;
      try {
        const heapF64 = Module.HEAPF64 ?? HEAPF64;
        if (!heapF64) {
          throw new Error('WASM heap is not initialized (HEAPF64 unavailable)');
        }
        // Commit only after success, as in scalar warp. The temporary also
        // stays valid if the callback allocates enough to grow WASM memory.
        const verts = new Float64Array(heapF64.buffer, ptr, count * 3).slice();
        func(verts, count);
        const currentHeap = Module.HEAPF64 ?? HEAPF64;
        new Float64Array(currentHeap.buffer, ptr, count * 3).set(verts);
      } catch (error) {
        // Return normally so C++ can destroy its local allocations.
        state.failed = true;
        state.failure = error;
      }
    }, 'vii', pointer => this._WarpBatch(pointer));
    return checkedManifold(out);
  };

  Module.Manifold.prototype.calculateNormals = function(minSharpAngle = 52.5) {
    return this._CalculateNormals(minSharpAngle);
  };

  Module.Manifold.prototype.smoothByNormals = function() {
    return this._SmoothByNormals();
  };

  Module.Manifold.prototype.setProperties = function(numProp, func) {
    const oldNumProp = this.numProp();
    return withCallback(state => function(newPtr, vec3Ptr, oldPtr) {
      if (state.failed) return;
      try {
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
      } catch (error) {
        // Return normally so C++ can destroy its local allocations.
        state.failed = true;
        state.failure = error;
      }
    }, 'viii', pointer => this._SetProperties(numProp, pointer));
  };

  Module.Manifold.prototype.translate = function(...vec) {
    return this._Translate(vararg2vec3(vec));
  };

  Module.Manifold.prototype.rotate = function(xOrVec, y, z) {
    if (Array.isArray(xOrVec)) {
      return this._Rotate(...xOrVec);
    } else {
      return this._Rotate(xOrVec, y ?? 0, z ?? 0);
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
    const result = new CrossSectionCtor(polygonsVec);
    polygonsVec.delete();
    return result;
  };

  Module.Manifold.prototype.project = function() {
    const polygonsVec = this._Project();
    const result = new CrossSectionCtor(polygonsVec);
    polygonsVec.delete();
    return result;
  };

  Module.Manifold.prototype.windingNumber = function(points) {
    const inVec = new Module.Vector_vec3();
    for (const p of points) inVec.push_back(vararg2vec3([p]));
    const outVec = this._WindingNumber(inVec);
    inVec.delete();
    const result = fromVec(outVec);
    outVec.delete();
    return result;
  };

  Module.Manifold.prototype.rayCast = function(origin, endpoint) {
    const vec = this._RayCast(vararg2vec3([origin]), vararg2vec3([endpoint]));
    const result =
        fromVec(vec, hit => ({
                       originalID: hit.originalID,
                       faceID: hit.faceID,
                       distance: hit.distance,
                       position: ['x', 'y', 'z'].map(f => hit.position[f]),
                       normal: ['x', 'y', 'z'].map(f => hit.normal[f]),
                       crossing: hit.crossing,
                     }));
    vec.delete();
    return result;
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

  Module.Manifold.prototype.simplify = function(tolerance = 0) {
    return this._Simplify(tolerance);
  };

  // Shared by Mesh and Mesh64, which differ only in float precision.
  class MeshBase {
    constructor(Float, {
      numProp = 3,
      triVerts = new Uint32Array(),
      vertProperties = new Float(),
      mergeFromVert,
      mergeToVert,
      runIndex,
      runOriginalID,
      faceID,
      halfedgeTangent,
      runTransform,
      runFlags
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
      this.runFlags = runFlags;
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

    // Identity when the run has no transform, as for an original.
    transform(run) {
      const mat4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
      if (!(run >= 0 && 12 * (run + 1) <= (this.runTransform?.length ?? 0))) {
        return mat4;
      }
      for (const col of [0, 1, 2, 3]) {
        for (const row of [0, 1, 2]) {
          mat4[4 * col + row] = this.runTransform[12 * run + 3 * col + row];
        }
      }
      return mat4;
    }

    backside(run) {
      return this.runFlags != null && run < this.runFlags.length &&
          (this.runFlags[run] & 1) !== 0;
    }

    hasNormals(run) {
      return this.runFlags != null && run < this.runFlags.length &&
          (this.runFlags[run] & 2) !== 0;
    }
  }

  class Mesh extends MeshBase {
    constructor(options) {
      super(Float32Array, options);
    }

    merge(tolerance = 0) {
      return new Mesh(Module._Merge(this, tolerance));
    }
  }

  class Mesh64 extends MeshBase {
    constructor(options) {
      super(Float64Array, options);
    }

    merge(tolerance = 0) {
      return new Mesh64(Module._Merge64(this, tolerance));
    }
  }

  // A registered symbol, so a Mesh64 from another instance of this module is
  // also read as double.
  const isMesh64 = Symbol.for('manifold-3d.Mesh64');
  Mesh64.prototype[isMesh64] = true;

  Module.Mesh = Mesh;
  Module.Mesh64 = Mesh64;

  Module.Manifold.prototype.getMesh = function() {
    return new Mesh(this._GetMeshJS());
  };

  Module.Manifold.prototype.getMesh64 = function() {
    return new Mesh64(this._GetMesh64JS());
  };

  Module.ManifoldError = function ManifoldError(code, ...args) {
    const message = Module._ErrorMessage(Module.status[code]?.value ?? -1);

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

  Module.CrossSection = function(polygons) {
    const polygonsVec = polygons2vec(polygons);
    const cs = new CrossSectionCtor(polygonsVec);
    polygonsVec.delete();
    return cs;
  };

  Module.CrossSection.evenOdd = function(polygons) {
    const polygonsVec = polygons2vec(polygons);
    const cs = Module._EvenOdd(polygonsVec);
    polygonsVec.delete();
    return cs;
  };

  Module.CrossSection.nonZero = function(polygons) {
    const polygonsVec = polygons2vec(polygons);
    const cs = Module._NonZero(polygonsVec);
    polygonsVec.delete();
    return cs;
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

  Module.CrossSection.circle = function(radius, quality = 0) {
    return Module._Circle(radius, toQuality(quality));
  };

  // allows args to be either CrossSection or polygons (constructed with
  // Positive fill)
  function crossSectionBatchbool(name) {
    return function(...args) {
      if (args.length == 1) args = args[0];
      const v = new Module.Vector_crossSection();
      for (const arg of args) withCrossSection(arg, cs => v.push_back(cs));
      const result = Module['_crossSection' + name](v);
      v.delete();
      return result;
    };
  }

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
      } else if (isPoint(cs)) {
        pushVec2(pts, [cs]);
      } else {
        const polys = isPoint(cs[0]) ? [cs] : cs;
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
    return checkedManifold(
        mesh?.[isMesh64] ? Module._FromMesh64(mesh) : Module._FromMesh(mesh));
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
      height, radiusLow, radiusHigh = -1.0, quality = 0, center = false) {
    return Module._Cylinder(
        height, radiusLow, radiusHigh, toQuality(quality), center);
  };

  Module.Manifold.sphere = function(radius, quality = 0) {
    return Module._Sphere(radius, toQuality(quality));
  };

  Module.Manifold.extrude = function(
      polygons, height, nDivisions = 0, twistDegrees = 0.0,
      scaleTop = [1.0, 1.0], center = false) {
    return withCrossSection(
        polygons,
        cs => cs.extrude(height, nDivisions, twistDegrees, scaleTop, center));
  };

  Module.Manifold.revolve = function(
      polygons, quality = 0, revolveDegrees = 360.0) {
    return withCrossSection(
        polygons, cs => cs.revolve(quality, revolveDegrees));
  };

  Module.Manifold.reserveIDs = function(n) {
    return Module._ReserveIDs(n);
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

  function withLevelSetCallback(sdf, level, operation) {
    return withCallback(state => function(vec3Ptr) {
      // Outside the surface, so later evaluations add no geometry.
      if (state.failed) return level - 1;
      try {
        const x = getValue(vec3Ptr, 'double');
        const y = getValue(vec3Ptr + 8, 'double');
        const z = getValue(vec3Ptr + 16, 'double');
        return sdf([x, y, z]);
      } catch (error) {
        // Return normally so C++ can destroy its local allocations.
        state.failed = true;
        state.failure = error;
        return level - 1;
      }
    }, 'di', operation);
  }

  Module.Manifold.levelSet = function(
      sdf, bounds, edgeLength, level = 0, tolerance = -1) {
    const bounds2 = {
      min: {x: bounds.min[0], y: bounds.min[1], z: bounds.min[2]},
      max: {x: bounds.max[0], y: bounds.max[1], z: bounds.max[2]},
    };
    return withLevelSetCallback(
        sdf, level,
        pointer =>
            Module._LevelSet(pointer, bounds2, edgeLength, level, tolerance));
  };

  // ctx-aware static factories: mirror new Manifold / levelSet but
  // run under this ExecutionContext so progress/cancellation are observed.
  Module.ExecutionContext.prototype.fromMesh = function(mesh) {
    return mesh?.[isMesh64] ? this._FromMesh64(mesh) : this._FromMesh(mesh);
  };

  Module.ExecutionContext.prototype.levelSet = function(
      sdf, bounds, edgeLength, level = 0, tolerance = -1) {
    const bounds2 = {
      min: {x: bounds.min[0], y: bounds.min[1], z: bounds.min[2]},
      max: {x: bounds.max[0], y: bounds.max[1], z: bounds.max[2]},
    };
    return withLevelSetCallback(
        sdf, level,
        pointer =>
            this._LevelSet(pointer, bounds2, edgeLength, level, tolerance));
  };

  function pushVec3(vec, ps) {
    toVec(vec, ps, p => {
      if (p instanceof Array) return {x: p[0], y: p[1], z: p[2]};
      return p;
    })
  }

  Module.Manifold.hull = function(...args) {
    if (args.length == 1) args = args[0];
    const manifolds = [];
    const pts = new Module.Vector_vec3();
    for (const m of args) {
      if (m instanceof ManifoldCtor) {
        manifolds.push(m);
      } else if (isPoint(m)) {
        pushVec3(pts, [m]);
      } else {
        pushVec3(pts, m);
      }
    }
    if (pts.size() == 0) {
      // The hull of Manifolds alone uses their smallest precision.
      const vec = new Module.Vector_manifold();
      toVec(vec, manifolds);
      pts.delete();
      const result = Module._manifoldHull(vec);
      vec.delete();
      return result;
    }
    let relPrecision = Infinity;
    for (const m of manifolds) {
      Module._manifoldCollectVertices(pts, m);
      if (!m.isEmpty()) relPrecision = Math.min(relPrecision, m.getPrecision());
    }
    const result = Module._manifoldHullPoints(pts);
    pts.delete();
    if (!isFinite(relPrecision) || relPrecision == result.getPrecision())
      return result;
    // Like the other operations on several Manifolds, the hull uses the
    // smallest precision among the non-empty input Manifolds, so it is
    // recomputed at that precision.
    const stamped = result.withPrecision(relPrecision);
    result.delete();
    const hull = stamped.hull();
    stamped.delete();
    return hull;
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

  Module.triangulate = function(polygons, epsilon = -1, allowConvex = true) {
    const polygonsVec = polygons2vec(polygons);
    const trianglesVec = Module._Triangulate(polygonsVec, epsilon, allowConvex);
    polygonsVec.delete();
    const result = fromVec(trianglesVec, (x) => [x[0], x[1], x[2]]);
    trianglesVec.delete();
    return result;
  };

  Module.version = function() {
    const v = Module._Version();
    return [v[0], v[1], v[2]];
  };
};
