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

// UI
const fileButton = document.querySelector('#file');
const currentElement = document.querySelector('#current');
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

const prefix = 'ManifoldCAD';
function getScript(name) {
  return window.localStorage.getItem(prefix + name);
}
function setScript(name, code) {
  window.localStorage.setItem(prefix + name, code);
}
function nthKey(n) {
  if (n >= window.localStorage.length) return;
  const key = window.localStorage.key(n);
  if (key.startsWith(prefix)) {
    return key.slice(prefix.length);
  }
}

function saveCurrent() {
  if (editor) {
    const currentName = currentElement.textContent;
    if (!examples.get(currentName)) {
      setScript(currentName, editor.getValue());
    }
  }
};

window.onpagehide = saveCurrent;

let switching = false;
let isExample = true;
function switchTo(scriptName) {
  saveCurrent();
  if (editor) {
    switching = true;
    hideDropdown();
    currentElement.textContent = scriptName;
    setScript('currentName', scriptName);
    const code = examples.get(scriptName) ?? getScript(scriptName) ?? '';
    isExample = examples.get(scriptName) != null;
    editor.setValue(code);
  }
}

function appendDropdownItem(name) {
  const button = document.createElement('button');
  button.type = 'button';
  button.classList.add('blue', 'item');
  button.textContent = name;
  dropdown.appendChild(button);
  button.onclick = function () { switchTo(button.textContent); };
  return button;
}

function newItem(code) {
  let num = 1;
  let name = 'New Script ' + num++;
  while (getScript(name) != null) {
    name = 'New Script ' + num++;
  }
  setScript(name, code);
  const nextButton = appendDropdownItem(name);
  nextButton.click();
};

const newButton = document.querySelector('#new');
newButton.onclick = function () { newItem(''); };

// Editor
let worker = undefined;
require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.34.0/min/vs' } });
require(['vs/editor/editor.main'], async function () {
  const content = await fetch('bindings.d.ts').then(response => response.text());
  monaco.languages.typescript.typescriptDefaults.addExtraLib(content);
  editor = monaco.editor.create(document.getElementById('editor'), {
    language: 'typescript',
    automaticLayout: true
  });
  const w = await monaco.languages.typescript.getTypeScriptWorker();
  worker = await w(editor.getModel().uri);

  for (const [name] of examples) {
    appendDropdownItem(name);
  }

  let currentName = currentElement.textContent;
  for (let i = 0; i < window.localStorage.length; i++) {
    const key = nthKey(i);
    if (!key) continue;
    if (key === 'currentName') {
      currentName = getScript(key);
    } else {
      appendDropdownItem(key);
    }
  }
  switchTo(currentName);

  document.querySelector('#compile').click();

  editor.onDidChangeModelContent(e => {
    runButton.disabled = false;
    if (switching) {
      switching = false;
      return;
    }
    if (isExample) {
      const cursor = editor.getPosition();
      newItem(editor.getValue());
      editor.setPosition(cursor);
    }
  });

  window.onresize = () => {
    editor.layout({});
  };
});

// Execution
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
      try {
        f(...exposedFunctions.map(name => Module[name]));
      } catch (error) {
        console.log(error);
      } finally {
        Module.cleanup();
        const t1 = performance.now();
        console.log(`took ${t1 - t0}ms`);
        runButton.disabled = true;
      }
    };
  }
};

// Export & Rendering
const mv = document.querySelector('model-viewer');
const mesh = new THREE.Mesh(undefined, new THREE.MeshStandardMaterial({
  color: 'yellow',
  metalness: 1,
  roughness: 0.2
}));
mesh.scale.setScalar(0.001);
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

document.querySelector('#download').onclick = function () {
  const link = document.createElement("a");
  link.download = "manifold.glb";
  link.href = objectURL;
  link.click();
};