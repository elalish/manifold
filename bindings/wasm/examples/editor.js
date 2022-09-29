const examples = new Map();

examples.set('Intro', `
// Write code in TypeScript and this editor will show the API docs.
// Manifold constructors include "cube", "cylinder", "sphere", "extrude", "revolve".
// Type e.g. "box." to see the Manifold API.
// Work in mm to get a GLB in true scale.
const box = cube([100, 100, 100], true);
const ball = sphere(60, 100);
// You must name your final output "result".
const result = box.subtract(ball);`);

examples.set('Warp', `
const ball = sphere(60, 100);
const func = (v: Vec3) => {
  v[2] /= 2;
};
const result = ball.warp(func);`);


let editor = undefined;
let worker = undefined;
const mesh = new THREE.Mesh(undefined, new THREE.MeshStandardMaterial({
  color: 'yellow',
  metalness: 1,
  roughness: 0.2
}));
mesh.scale.setScalar(0.001);
const runButton = document.querySelector('#compile');
var Module = {
  onRuntimeInitialized: function () {
    Module.setup();
    // Setup memory management, such that users don't have to care about
    // calling `delete` manually.
    // Note that this only fixes memory leak across different runs: the memory
    // will only be freed when the compilation finishes.

    let manifoldRegistry = [];
    for (const name of
      ['add', 'subtract', 'intersect', 'refine', 'transform', 'translate',
        'rotate', 'scale']) {
      const originalFn = Module.Manifold.prototype[name];
      Module.Manifold.prototype["_" + name] = originalFn;
      Module.Manifold.prototype[name] = function (...args) {
        const result = this["_" + name](...args);
        manifoldRegistry.push(result);
        return result;
      }
    }

    for (const name
      of ['cube', 'cylinder', 'sphere', 'extrude', 'revolve', 'union',
        'difference', 'intersection']) {
      const originalFn = Module[name];
      Module[name] = function (...args) {
        const result = originalFn(...args);
        manifoldRegistry.push(result);
        return result;
      }
    }

    Module.cleanup = function () {
      for (const obj of manifoldRegistry) {
        obj.delete();
      }
      manifoldRegistry = [];
    }

    runButton.onclick = async function (e) {
      const output = await worker.getEmitOutput(editor.getModel().uri.toString());
      const content = output.outputFiles[0].text + 'push2MV(result);';
      const exposedFunctions = [
        'cube', 'cylinder', 'sphere', 'extrude', 'revolve',
        'union', 'difference', 'intersection',
      ];
      const f = new Function(...exposedFunctions, content);
      const t0 = performance.now();
      f(...exposedFunctions.map(name => Module[name]));
      Module.cleanup();
      const t1 = performance.now();
      console.log(`took ${t1 - t0}ms`);
      runButton.disabled = true;
    };
  }
};

const mv = document.querySelector('model-viewer');
let objectURL = null;
const exporter = new THREE.GLTFExporter();

function push2MV(manifold) {
  mesh.geometry?.dispose();
  mesh.geometry = mesh2geometry(manifold.getMesh());
  exporter.parse(
    mesh,
    (gltf) => {
      const blob = new Blob([gltf], { type: 'application/octet-stream' });
      URL.revokeObjectURL(objectURL);
      objectURL = URL.createObjectURL(blob);
      mv.src = objectURL;
    },
    () => console.log('GLTF export failed!'),
    { binary: true }
  );
}

document.querySelector('#download').onclick = function () {
  const link = document.createElement("a");
  link.download = "manifold.glb";
  link.href = objectURL;
  link.click();
};

const fileButton = document.querySelector('#file');
const arrow = document.querySelector('.uparrow');
const dropdown = document.querySelector('.dropdown');
const hideDropdown = function () {
  dropdown.classList.remove('show');
  arrow.classList.remove('down');
};
const toggleDropdown = function () {
  dropdown.classList.toggle('show');
  arrow.classList.toggle('down');
};
fileButton.onclick = toggleDropdown;

// State
const currentName = document.querySelector('#current');
let nextNum = 1;
let switching = false;

function appendDropdownItem(name) {
  const button = document.createElement('button');
  button.type = 'button';
  button.classList.add('blue', 'item');
  button.textContent = name;
  dropdown.appendChild(button);
  button.onclick = function () {
    if (editor) {
      if (!examples.get(currentName.textContent)) {
        window.localStorage.setItem(currentName.textContent, editor.getValue());
      }
      switching = true;
      hideDropdown();
      const scriptName = button.textContent;
      currentName.textContent = scriptName;
      const code = examples.get(scriptName) ?? window.localStorage.getItem(scriptName) ?? '';
      editor.setValue(code);
    }
  };
  return button;
}

for (const [name] of examples) {
  appendDropdownItem(name);
}

function newItem(code) {
  const name = 'New Script ' + nextNum++;
  window.localStorage.setItem(name, code);
  const nextButton = appendDropdownItem(name);
  nextButton.click();
};

const newButton = document.querySelector('#new');
newButton.onclick = function () { newItem(''); };

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

require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.34.0/min/vs' } });
require(['vs/editor/editor.main'], async function () {
  const content = await fetch('bindings.d.ts').then(response => response.text());
  monaco.languages.typescript.typescriptDefaults.addExtraLib(content);
  editor = monaco.editor.create(document.getElementById('editor'), {
    value: examples.get('Intro'),
    language: 'typescript',
    automaticLayout: true
  });
  const w = await monaco.languages.typescript.getTypeScriptWorker();
  worker = await w(editor.getModel().uri);

  document.querySelector('#compile').click();

  editor.onDidChangeModelContent(e => {
    runButton.disabled = false;
    if (switching) {
      switching = false;
      return;
    }
    if (examples.get(currentName.textContent)) {
      const cursor = editor.getPosition();
      newItem(editor.getValue());
      editor.setPosition(cursor);
    }
  });

  window.onresize = () => {
    editor.layout({});
  };
});