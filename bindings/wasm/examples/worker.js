
// manifold member functions that returns a new manifold
const memberFunctions = [
  'add', 'subtract', 'intersect', 'refine', 'transform', 'translate', 'rotate',
  'scale', 'asOriginal', 'smooth', 'decompose'
];
// top level functions that constructs a new manifold
const constructors = [
  'cube', 'cylinder', 'sphere', 'tetrahedron', 'extrude', 'revolve', 'union',
  'difference', 'intersection', 'compose', 'levelSet'
];
const utils = [
  'setMinCircularAngle', 'setMinCircularEdgeLength', 'setCircularSegments',
  'getCircularSegments'
];
const exposedFunctions = constructors.concat(utils);

var Module = {
  onRuntimeInitialized: function () {
    Module.setup();
    // Setup memory management, such that users don't have to care about
    // calling `delete` manually.
    // Note that this only fixes memory leak across different runs: the memory
    // will only be freed when the compilation finishes.

    let manifoldRegistry = [];
    for (const name of memberFunctions) {
      const originalFn = Module.Manifold.prototype[name];
      Module.Manifold.prototype["_" + name] = originalFn;
      Module.Manifold.prototype[name] = function (...args) {
        const result = this["_" + name](...args);
        manifoldRegistry.push(result);
        return result;
      }
    }

    for (const name of constructors) {
      const originalFn = Module[name];
      Module[name] = function (...args) {
        const result = originalFn(...args);
        manifoldRegistry.push(result);
        return result;
      }
    }

    Module.cleanup = function () {
      for (const obj of manifoldRegistry) {
        // decompose result is an array of manifolds
        if (obj instanceof Array)
          for (const elem of obj)
            elem.delete();
        else
          obj.delete();
      }
      manifoldRegistry = [];
    }

    postMessage(null);
  }
};

const threePath = 'https://cdn.jsdelivr.net/npm/three@0.144.0/';
importScripts('manifold.js', threePath + 'build/three.js', threePath + 'examples/js/exporters/GLTFExporter.js');


onmessage = (e) => {
  const content = e.data + '\nexportGLB(result);\n';
  try {
    const f = new Function(...exposedFunctions, content);
    const t0 = performance.now();
    f(...exposedFunctions.map(name => Module[name]));
    const t1 = performance.now();
    const log = consoleElement.textContent;
    // Remove "Running..."
    consoleElement.textContent = log.substring(log.indexOf("\n") + 1);
    console.log(`Took ${Math.round(t1 - t0)} ms`);
    setScript('safe', 'true');
  } catch (error) {
    console.log(error);
  } finally {
    Module.cleanup();
  }
}

// Export & Rendering ------------------------------------------------------------
const mesh = new THREE.Mesh(undefined, new THREE.MeshStandardMaterial({
  color: 'yellow',
  metalness: 1,
  roughness: 0.2
}));
const rotation = new THREE.Matrix4();
rotation.set(
  1, 0, 0, 0,
  0, 0, 1, 0,
  0, -1, 0, 0,
  0, 0, 0, 1);
mesh.setRotationFromMatrix(rotation); // Z-up -> Y-up
mesh.scale.setScalar(0.001); // mm -> m

const exporter = new THREE.GLTFExporter();

function exportGLB(manifold) {
  const box = manifold.boundingBox();
  const size = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    size[i] = Math.round((box.max[i] - box.min[i]) * 10) / 10;
  }
  console.log(`Bounding Box: X = ${size[0]} mm, Y = ${size[1]} mm, Z = ${size[2]} mm`);
  mesh.geometry?.dispose();
  mesh.geometry = mesh2geometry(manifold.getMesh());
  exporter.parse(
    mesh,
    (gltf) => {
      const blob = new Blob([gltf], { type: 'application/octet-stream' });
      postMessage(URL.createObjectURL(blob));
    },
    () => {
      console.log('glTF export failed!');
      postMessage(null);
    },
    { binary: true }
  );
}

function mesh2geometry(mesh) {
  const geometry = new THREE.BufferGeometry();

  const numVert = mesh.vertPos.size();
  const vert = new Float32Array(3 * numVert);
  for (let i = 0; i < numVert; i++) {
    const v = mesh.vertPos.get(i);
    const idx = 3 * i;
    vert[idx] = v.x;
    vert[idx + 1] = v.y;
    vert[idx + 2] = v.z;
  }

  const numTri = mesh.triVerts.size();
  const tri = new Uint32Array(3 * numTri);
  for (let i = 0; i < numTri; i++) {
    const v = mesh.triVerts.get(i);
    const idx = 3 * i;
    tri[idx] = v[0];
    tri[idx + 1] = v[1];
    tri[idx + 2] = v[2];
  }

  mesh.vertPos.delete();
  mesh.triVerts.delete();
  mesh.vertNormal.delete();
  mesh.halfedgeTangent.delete();

  geometry.setAttribute('position', new THREE.BufferAttribute(vert, 3));
  geometry.setIndex(new THREE.BufferAttribute(tri, 1));
  return geometry;
}