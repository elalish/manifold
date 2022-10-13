let editor = undefined;

// File UI ------------------------------------------------------------
const fileButton = document.querySelector('#file');
const currentElement = document.querySelector('#current');
const arrow = document.querySelector('.uparrow');
const dropdown = document.querySelector('.dropdown');

const hideDropdown = function () {
  dropdown.classList.remove('show');
  arrow.classList.remove('down');
};
const toggleDropdown = function (event) {
  event.stopPropagation();
  dropdown.classList.toggle('show');
  arrow.classList.toggle('down');
};
fileButton.onclick = toggleDropdown;
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
    const currentName = currentElement.textContent;
    if (!examples.get(currentName)) {
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
    currentElement.textContent = scriptName;
    setScript('currentName', scriptName);
    const code = examples.get(scriptName) ?? getScript(scriptName) ?? '';
    isExample = examples.get(scriptName) != null;
    editor.setValue(code);
  }
}

function appendDropdownItem(name) {
  const container = document.createElement('div');
  container.classList.add('item');
  const button = document.createElement('button');
  container.appendChild(button);
  button.type = 'button';
  button.classList.add('blue', 'item');
  const label = document.createElement('span');
  button.appendChild(label);
  label.textContent = name;
  dropdown.appendChild(container);

  button.onclick = function () {
    saveCurrent();
    switchTo(label.textContent);
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
  while (getScript(newName) != null || examples.get(newName) != null) {
    newName = name + ' ' + num++;
  }
  return newName;
}

function addEdit(button) {
  const label = button.firstChild;
  const edit = addIcon(button);
  edit.style.backgroundImage = 'url(pencil.png)';
  edit.style.right = '30px';

  edit.onclick = function (event) {
    event.stopPropagation();
    const oldName = label.textContent;
    const code = getScript(oldName);
    const form = document.createElement('form');
    const inputElement = document.createElement('input');
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
      if (currentElement.textContent == oldName) {
        currentElement.textContent = newName;
      }
      removeScript(oldName);
      setScript(newName, code);
    }

    form.onsubmit = rename;
    inputElement.onclick = function (event) { event.stopPropagation(); };

    inputElement.onblur = function () {
      button.removeChild(form);
      label.textContent = oldName;
    };
  };

  const trash = addIcon(button);
  trash.style.backgroundImage = 'url(trash.png)';
  trash.style.right = '0px';
  let lastClick = 0;

  trash.onclick = function (event) {
    event.stopPropagation();
    if (button.classList.contains('blue')) {
      lastClick = performance.now();
      button.classList.remove('blue');
      button.classList.add('red');
      document.body.addEventListener('click', function () {
        button.classList.add('blue');
        button.classList.remove('red');
      }, { once: true });
    } else if (performance.now() - lastClick > 500) {
      removeScript(label.textContent);
      button.parentElement.removeChild(button);
      switchTo('Intro');
    }
  };
}

function newItem(code) {
  const name = uniqueName('New Script');
  setScript(name, code);
  const nextButton = appendDropdownItem(name);
  addEdit(nextButton);
  nextButton.click();
};

const newButton = document.querySelector('#new');
newButton.onclick = function () { newItem(''); };

const runButton = document.querySelector('#compile');
const poster = document.querySelector('#poster');
let manifoldInitialized = false;
let autoExecute = false;

function initializeRun() {
  runButton.disabled = false;
  if (autoExecute) {
    runButton.click();
  } else {
    poster.textContent = 'Auto-run disabled due to prior failure';
  }
}

// Editor ------------------------------------------------------------
let tsWorker = undefined;
require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.34.0/min/vs' } });
require(['vs/editor/editor.main'], async function () {
  const content = await fetch('bindings.d.ts').then(response => response.text());
  monaco.languages.typescript.typescriptDefaults.addExtraLib(content);
  editor = monaco.editor.create(document.getElementById('editor'), {
    language: 'typescript',
    automaticLayout: true
  });
  const w = await monaco.languages.typescript.getTypeScriptWorker();
  tsWorker = await w(editor.getModel().uri);

  for (const [name] of examples) {
    appendDropdownItem(name);
  }

  let currentName = currentElement.textContent;

  for (let i = 0; i < window.localStorage.length; i++) {
    const key = nthKey(i);
    if (!key) continue;
    if (key === 'currentName') {
      currentName = getScript(key);
    } else if (key === 'safe') {
      autoExecute = getScript(key) !== 'false';
    } else {
      const button = appendDropdownItem(key);
      addEdit(button);
    }
  }
  switchTo(currentName);

  if (manifoldInitialized) {
    initializeRun();
  }

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

// Execution ------------------------------------------------------------
const consoleElement = document.querySelector('#console');
const oldLog = console.log;
console.log = function (message) {
  consoleElement.textContent += message.toString() + '\r\n';
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
  consoleElement.textContent = log.substring(log.indexOf("\n") + 1);
  console.log(`Took ${(Math.round((t1 - t0) / 10) / 100).toLocaleString()} seconds`);
}

const mv = document.querySelector('model-viewer');
let objectURL = null;
let manifoldWorker = null;

function createWorker() {
  manifoldWorker = new Worker('worker.js');
  manifoldWorker.onmessage = function (e) {
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
    setScript('safe', 'true');

    URL.revokeObjectURL(objectURL);
    objectURL = e.data.objectURL;
    mv.src = objectURL;
    if (objectURL == null) {
      mv.showPoster();
      poster.textContent = 'Error';
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

runButton.onclick = function () {
  if (runButton.classList.contains('cancel')) {
    cancel();
  } else {
    run();
  }
};

const downloadButton = document.querySelector('#download');
downloadButton.onclick = function () {
  const link = document.createElement("a");
  link.download = "manifold.glb";
  link.href = objectURL;
  link.click();
};
