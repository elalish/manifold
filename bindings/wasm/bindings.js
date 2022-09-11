var _ManifoldInitialized = false;
Module.setup = function() {
  if (_ManifoldInitialized) return;
  _ManifoldInitialized = true;

  function toVec(vec, list, f = x => x) {
    for (let x of list) {
      vec.push_back(f(x));
    }
    return vec;
  }

  function polygons2vec(polygons) {
    return toVec(
        new Module.Vector2_vec2(), polygons,
        poly => toVec(new Module.Vector_vec2(), poly, p => {
          if (p instanceof Array) return {x: p[0], y: p[1]};
          return p;
        }));
  }

  Module.geometry2mesh = function(geometry) {
    const mesh = {
      vertPos: new Module.Vector_vec3(),
      vertNormal: new Module.Vector_vec3(),
      triVerts: new Module.Vector_ivec3(),
      halfedgeTangent: new Module.Vector_vec4()
    };
    const temp = new THREE.Vector3();
    const p = geometry.attributes.position;
    const n = geometry.attributes.normal;
    const x = geometry.index;
    for (let i = 0; i < p.count; i++) {
      temp.fromBufferAttribute(p, i);
      mesh.vertPos.push_back(temp);
      temp.fromBufferAttribute(n, i);
      mesh.vertNormal.push_back(temp);
    }
    for (let i = 0; i < x.count; i += 3) {
      mesh.triVerts.push_back(x.array.subarray(i, i + 3));
    }
    return mesh;
  };

  function mesh2geometry(mesh) {
    const geometry = new THREE.BufferGeometry();
    const p = [], n = [], x = [];
    let i, s, v;
    for (i = 0, s = mesh.vertPos.size(); i < s; i++) {
      v = mesh.vertPos.get(i);
      p.push(v.x, v.y, v.z);
      v = mesh.vertNormal.get(i);
      n.push(v.x, v.y, v.z);
    }
    for (i = 0, s = mesh.triVerts.size(); i < s; i++) {
      v = mesh.triVerts.get(i);
      x.push(v[0], v[1], v[2]);
    }
    geometry.setAttribute(
        'position', new THREE.BufferAttribute(new Float32Array(p), 3));
    geometry.setAttribute(
        'normal', new THREE.BufferAttribute(new Float32Array(n), 3));
    geometry.setIndex(new THREE.BufferAttribute(new Uint8Array(x), 1));
    return geometry;
  }

  function vararg2vec(vec) {
    if (vec[0] instanceof Array)
      return {x: vec[0][0], y: vec[0][1], z: vec[0][2]};
    if (typeof (vec[0]) == 'number')
      // default to 0
      return {x: vec[0] || 0, y: vec[1] || 0, z: vec[2] || 0};
    return vec[0];
  }

  Module.Manifold.prototype.getGeometry = function() {
    let mesh = this.getMesh();
    let geometry = mesh2geometry(mesh);
    for (let name in mesh) mesh[name].delete();
    return geometry;
  };

  Module.Manifold.prototype.transform = function(mat) {
    console.assert(mat.length == 4, 'expects a 4x3 matrix');
    let vec = new Module.Vector_f32();
    for (let row of mat) {
      console.assert(row.length == 3, 'expects a 4x3 matrix');
      for (let x of row) mat.push_back(x);
    }
    const result = this._Transform(vec);
    vec.delete();
    return result;
  };

  Module.Manifold.prototype.translate = function(...vec) {
    return this._Translate(vararg2vec(vec));
  };

  Module.Manifold.prototype.rotate = function(...vec) {
    return this._Rotate(vararg2vec(vec));
  };

  Module.Manifold.prototype.scale = function(...vec) {
    return this._Scale(vararg2vec(vec));
  };

  Module.cube = function(...args) {
    const size = vararg2vec(args);
    const center = args[3] || false;
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

  Module.extrude = function(
      polygons, height, nDivisions = 0, twistDegrees = 0.0,
      scaleTop = [1.0, 1.0]) {
    if (scaleTop instanceof Array) scaleTop = {x: scaleTop[0], y: scaleTop[1]};
    const polygonsVec = polygons2vec(polygons);
    const result = Module._Extrude(
        polygonsVec, height, nDivisions, twistDegrees, scaleTop);
    for (let i = 0; i < polygonsVec.size(); i++) polygonsVec.get(i).delete();
    polygonsVec.delete();
    return result;
  };

  Module.revolve = function(polygons, circularSegments = 0) {
    const polygonsVec = polygons2vec(polygons);
    const result = Module._Revolve(polygonsVec, circularSegments);
    for (let i = 0; i < polygonsVec.size(); i++) polygonsVec.get(i).delete();
    polygonsVec.delete();
    return result;
  };
};
