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

import ManifoldWorker from './worker?worker';

const CODE_START = '<code>';
// Loaded globally by examples.js
const exampleFunctions = self.examples.functionBodies;

if (navigator.serviceWorker) {
  navigator.serviceWorker.register(
      '/service-worker.js', {scope: './index.html'});
}

let editor = undefined;

// Edit UI ------------------------------------------------------------

const undoButton = document.querySelector('#undo');
const redoButton = document.querySelector('#redo');
const formatButton = document.querySelector('#format');
const shareButton = document.querySelector('#share');

undoButton.onclick = () => editor.trigger('ignored', 'undo');
redoButton.onclick = () => editor.trigger('ignored', 'redo');
formatButton.onclick = () =>
    editor.trigger('ignored', 'editor.action.formatDocument');
shareButton.onclick = () => {
  const url = new URL(window.location.toString());
  url.hash =
      '#' +
      encodeURIComponent(
          currentFileElement.textContent + CODE_START + editor.getValue());
  navigator.clipboard.writeText(url.toString());
  console.log('Sharable link copied to clipboard!');
};

// File UI ------------------------------------------------------------
const fileButton = document.querySelector('#file');
const currentFileElement = document.querySelector('#current');
const fileArrow = document.querySelector('#file .uparrow');
const fileDropdown = document.querySelector('#fileDropdown');
const saveContainer = document.querySelector('#save');
const saveDropdown = document.querySelector('#saveDropdown');
const saveArrow = document.querySelector('#save .uparrow');

const hideDropdown = function() {
  fileDropdown.classList.remove('show');
  saveDropdown.classList.remove('show');
  fileArrow.classList.remove('down');
  saveArrow.classList.remove('down');
};
const toggleFileDropdown = function(event) {
  event.stopPropagation();
  fileDropdown.classList.toggle('show');
  fileArrow.classList.toggle('down');
};
const toggleSaveDropdown = function(event) {
  event.stopPropagation();
  saveDropdown.classList.toggle('show');
  saveArrow.classList.toggle('down');
};
fileButton.onclick = toggleFileDropdown;
saveArrow.parentElement.onclick = toggleSaveDropdown;
document.body.onclick = hideDropdown;

const prefix = 'ManifoldCAD';
function getScript(name) {
  return window.localStorage.getItem(prefix + name);
}
function setScript(name, code) {
  window.localStorage.setItem(prefix + name, code);
}
function removeScript(name) {
  window.localStorage.removeItem(prefix + name);
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
    const currentName = currentFileElement.textContent;
    if (!exampleFunctions.get(currentName)) {
      setScript(currentName, editor.getValue());
    }
  }
};

window.onpagehide = saveCurrent;
window.beforeunload = saveCurrent;

let switching = false;
let isExample = true;
function switchTo(scriptName) {
  if (editor) {
    switching = true;
    currentFileElement.textContent = scriptName;
    setScript('currentName', scriptName);
    isExample = exampleFunctions.get(scriptName) != null;
    const code = isExample ? exampleFunctions.get(scriptName).substring(1) :
                             getScript(scriptName) ?? '';
    window.location.hash = '#' + scriptName;
    editor.setValue(code);
  }
}

function createDropdownItem(name) {
  const container = document.createElement('div');
  container.classList.add('item');
  const button = document.createElement('button');
  container.appendChild(button);
  button.type = 'button';
  button.classList.add('blue', 'item');
  const label = document.createElement('span');
  button.appendChild(label);
  label.textContent = name;

  button.onclick = function() {
    saveCurrent();
    switchTo(label.textContent);
  };
  // Stop text input spaces from triggering the button
  button.onkeyup = function(event) {
    event.preventDefault();
  };
  return button;
}

function addIcon(button) {
  const icon = document.createElement('button');
  icon.classList.add('icon');
  button.parentElement.appendChild(icon);
  return icon;
}

function uniqueName(name) {
  let num = 1;
  let newName = name;
  while (getScript(newName) != null || exampleFunctions.get(newName) != null) {
    newName = name + ' ' + num++;
  }
  return newName;
}

function addEdit(button) {
  const label = button.firstChild;
  const edit = addIcon(button);
  edit.classList.add('edit');

  edit.onclick = function(event) {
    event.stopPropagation();
    const oldName = label.textContent;
    const code = getScript(oldName);
    const form = document.createElement('form');
    const inputElement = document.createElement('input');
    inputElement.classList.add('name');
    inputElement.value = oldName;
    label.textContent = '';
    button.appendChild(form);
    form.appendChild(inputElement);
    inputElement.focus();
    inputElement.setSelectionRange(0, oldName.length);

    function rename() {
      const input = inputElement.value;
      inputElement.blur();
      if (!input) return;
      const newName = uniqueName(input);
      label.textContent = newName;
      if (currentFileElement.textContent == oldName) {
        currentFileElement.textContent = newName;
      }
      removeScript(oldName);
      setScript(newName, code);
    }

    form.onsubmit = rename;
    inputElement.onclick = function(event) {
      event.stopPropagation();
    };

    inputElement.onblur = function() {
      button.removeChild(form);
      label.textContent = oldName;
    };
  };

  const trash = addIcon(button);
  trash.classList.add('trash');
  let lastClick = 0;

  trash.onclick = function(event) {
    event.stopPropagation();
    if (button.classList.contains('blue')) {
      lastClick = performance.now();
      button.classList.remove('blue');
      button.classList.add('red');
      document.body.addEventListener('click', function() {
        button.classList.add('blue');
        button.classList.remove('red');
      }, {once: true});
    } else if (performance.now() - lastClick > 500) {
      removeScript(label.textContent);
      if (currentFileElement.textContent == label.textContent) {
        switchTo('Intro');
      }
      const container = button.parentElement;
      container.parentElement.removeChild(container);
    }
  };
}

const newButton = document.querySelector('#new');
function newItem(code, scriptName = undefined) {
  const name = uniqueName(scriptName ?? 'New Script');
  setScript(name, code);
  const nextButton = createDropdownItem(name);
  newButton.insertAdjacentElement('afterend', nextButton.parentElement);
  addEdit(nextButton);
  return {button: nextButton, name};
};
newButton.onclick = function() {
  newItem('').button.click();
};

const runButton = document.querySelector('#compile');
const poster = document.querySelector('#poster');
let manifoldInitialized = false;
let autoExecute = true;

function initializeRun() {
  runButton.disabled = false;
  if (autoExecute) {
    runButton.click();
  } else {
    poster.textContent = 'Auto-run disabled';
  }
}

// Editor ------------------------------------------------------------
let tsWorker = undefined;

async function getManifoldDTS() {
  const global = await fetch('/manifold-global-types.d.ts')
                     .then(response => response.text());

  const encapsulated = await fetch('/manifold-encapsulated-types.d.ts')
                           .then(response => response.text());

  return `
${global.replaceAll('export', '')}
${encapsulated.replace(/^import.*$/gm, '').replaceAll('export', 'declare')}
declare interface ManifoldToplevel {
  CrossSection: typeof T.CrossSection;
  Manifold: typeof T.Manifold;
  Mesh: typeof T.Mesh;
  triangulate: typeof T.triangulate;
  setMinCircularAngle: typeof T.setMinCircularAngle;
  setMinCircularEdgeLength: typeof T.setMinCircularEdgeLength;
  setCircularSegments: typeof T.setCircularSegments;
  getCircularSegments: typeof T.getCircularSegments;
  setup: () => void;
}
declare const module: ManifoldToplevel;
`;
}

async function getEditorDTS() {
  const global = await fetch('/editor.d.ts').then(response => response.text());
  return `${global.replace(/^import.*$/gm, '')}`;
}

require.config({
  paths:
      {vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.34.0/min/vs'}
});
require(['vs/editor/editor.main'], async function() {
  monaco.languages.typescript.typescriptDefaults.addExtraLib(
      await getManifoldDTS());
  monaco.languages.typescript.typescriptDefaults.addExtraLib(
      await getEditorDTS());
  editor = monaco.editor.create(
      document.getElementById('editor'),
      {language: 'typescript', automaticLayout: true});
  const w = await monaco.languages.typescript.getTypeScriptWorker();
  tsWorker = await w(editor.getModel().uri);
  editor.getModel().updateOptions({tabSize: 2});

  for (const [name] of exampleFunctions) {
    const button = createDropdownItem(name);
    fileDropdown.appendChild(button.parentElement);
  }

  let currentName = currentFileElement.textContent;

  for (let i = 0; i < window.localStorage.length; i++) {
    const key = nthKey(i);
    if (!key) continue;
    if (key === 'currentName') {
      currentName = getScript(key);
    } else if (key === 'safe') {
      autoExecute = getScript(key) !== 'false';
    } else {
      const button = createDropdownItem(key);
      newButton.insertAdjacentElement('afterend', button.parentElement);
      addEdit(button);
    }
  }

  if (window.location.hash.length > 0) {
    const fragment = decodeURIComponent(window.location.hash.substring(1));
    const codeIdx = fragment.indexOf(CODE_START);
    if (codeIdx != -1) {
      autoExecute = true;
      const name = fragment.substring(0, codeIdx);
      switchTo(
          newItem(fragment.substring(codeIdx + CODE_START.length), name).name);
    } else {
      if (fragment != currentName) {
        autoExecute = true;
      }
      switchTo(fragment);
    }
  } else {
    switchTo(currentName);
  }

  if (manifoldInitialized) {
    initializeRun();
  }

  editor.onDidChangeModelContent(e => {
    runButton.disabled = false;
    if (switching) {
      switching = false;
      editor.setScrollTop(0);
      return;
    }
    if (isExample) {
      const cursor = editor.getPosition();
      newItem(editor.getValue()).button.click();
      editor.setPosition(cursor);
    }
  });

  window.onresize = () => {
    editor.layout({});
  };
});

// Animation ------------------------------------------------------------
const mv = document.querySelector('model-viewer');
const animationContainer = document.querySelector('#animation');
const playButton = document.querySelector('#play');
const scrubber = document.querySelector('#scrubber');
let paused = false;

mv.addEventListener('load', () => {
  const hasAnimation = mv.availableAnimations.length > 0;
  animationContainer.style.display = hasAnimation ? 'flex' : 'none';
  if (hasAnimation) {
    play();
  }
});

function play() {
  mv.play();
  playButton.classList.remove('play');
  playButton.classList.add('pause');
  paused = false;
  scrubber.classList.add('hide');
}

function pause() {
  mv.pause();
  playButton.classList.remove('pause');
  playButton.classList.add('play');
  paused = true;
  scrubber.max = mv.duration;
  scrubber.value = mv.currentTime;
  scrubber.classList.remove('hide');
}

playButton.onclick = function() {
  if (paused) {
    play();
  } else {
    pause();
  }
};

scrubber.oninput = function() {
  mv.currentTime = scrubber.value;
};

// Execution ------------------------------------------------------------
const consoleElement = document.querySelector('#console');
const oldLog = console.log;
console.log = function(message) {
  consoleElement.textContent += message + '\r\n';
  consoleElement.scrollTop = consoleElement.scrollHeight;
  oldLog(message);
};

function clearConsole() {
  consoleElement.textContent = '';
}

function enableCancel() {
  runButton.firstChild.style.visibility = 'hidden';
  runButton.classList.add('red', 'cancel');
}

function disableCancel() {
  runButton.firstChild.style.visibility = 'visible';
  runButton.classList.remove('red', 'cancel');
}

let t0 = performance.now();

function finishRun() {
  disableCancel();
  const t1 = performance.now();
  const log = consoleElement.textContent;
  // Remove "Running..."
  consoleElement.textContent = log.substring(log.indexOf('\n') + 1);
  console.log(
      `Took ${(Math.round((t1 - t0) / 10) / 100).toLocaleString()} seconds`);
}

const output = {
  glbURL: null,
  threeMFURL: null
};
let manifoldWorker = null;

function createWorker() {
  manifoldWorker = new ManifoldWorker();
  manifoldWorker.onmessage = function(e) {
    if (e.data == null) {
      if (tsWorker != null && !manifoldInitialized) {
        initializeRun();
      }
      manifoldInitialized = true;
      return;
    }

    if (e.data.log != null) {
      consoleElement.textContent += e.data.log + '\r\n';
      consoleElement.scrollTop = consoleElement.scrollHeight;
      return;
    }

    finishRun();
    runButton.disabled = true;

    if (output.threeMFURL != null) {
      URL.revokeObjectURL(output.threeMFURL);
      output.threeMFURL = null;
    }
    URL.revokeObjectURL(output.glbURL);
    output.glbURL = e.data.glbURL;
    output.threeMFURL = e.data.threeMFURL;
    threemfButton.disabled = output.threeMFURL == null;
    mv.src = output.glbURL;
    if (output.glbURL == null) {
      mv.showPoster();
      poster.textContent = 'Error';
      createWorker();
    } else {
      setScript('safe', 'true');
    }
  }
}

createWorker();

async function run() {
  saveCurrent();
  setScript('safe', 'false');
  enableCancel();
  clearConsole();
  console.log('Running...');
  const output = await tsWorker.getEmitOutput(editor.getModel().uri.toString());
  manifoldWorker.postMessage(output.outputFiles[0].text);
  t0 = performance.now();
}

function cancel() {
  manifoldWorker.terminate();
  createWorker();
  finishRun();
  console.log('Run canceled');
}

runButton.onclick = function() {
  if (runButton.classList.contains('cancel')) {
    cancel();
  } else {
    run();
  }
};

function clickSave(saveButton, filename, outputName) {
  const container = saveButton.parentElement;
  return () => {
    const oldSave = saveContainer.firstElementChild;
    if (oldSave !== container) {
      saveDropdown.insertBefore(oldSave, saveDropdown.firstElementChild);
      saveContainer.insertBefore(container, saveDropdown);
      container.appendChild(saveArrow.parentElement);
    }
    const link = document.createElement('a');
    link.download = filename;
    link.href = output[outputName];
    link.click();
  };
}

const glbButton = document.querySelector('#glb');
glbButton.onclick = clickSave(glbButton, 'manifold.glb', 'glbURL');

const threemfButton = document.querySelector('#threemf');
threemfButton.onclick = clickSave(threemfButton, 'manifold.3mf', 'threeMFURL');
