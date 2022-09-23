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

  Module.Manifold.prototype.rotate = function (...vec) {
    return this._Rotate(...vararg2vec(vec));
  };

  Module.Manifold.prototype.scale = function (...vec) {
    return this._Scale(vararg2vec(vec));
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
};
