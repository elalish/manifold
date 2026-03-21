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

// '?url' is vite convention to reference a static asset.
// vite will package the asset and provide a proper URL.
import '@google/model-viewer';

import esbuildWasmUrl from 'esbuild-wasm/esbuild.wasm?url';
import ManifoldWorker from 'manifold-3d/lib/worker.bundled.js?worker';
import manifoldWasmUrl from 'manifold-3d/manifold.wasm?url';
import {AutoTypings, JsDelivrSourceResolver, LocalStorageCache} from 'monaco-editor-auto-typings';
import * as monaco from 'monaco-editor/esm/vs/editor/editor.main';
// '?worker' is vite convention to load a module as a web worker.
import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import tsWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker?worker';

const CODE_START = '<code>';
// Loaded globally by examples.js
const exampleFunctions = self.examples;

if (navigator.serviceWorker) {
  const params = new URLSearchParams(window.location.search);
  const disableServiceWorker = params.has('no-sw');

  if (window.caches) {
    window.caches.keys().then(keys => {
      keys.filter(
              key => key.startsWith('manifoldCAD-cache-') &&
                  key !== 'manifoldCAD-cache-v4')
          .forEach(key => window.caches.delete(key));
    });
  }

  if (disableServiceWorker) {
    // Explicit escape hatch for debugging cache-related issues.
    navigator.serviceWorker.getRegistrations().then(registrations => {
      registrations.forEach(registration => registration.unregister());
    });
  } else {
    // Resolve against the current page URL so production asset paths don't
    // redirect registration into /assets.
    const serviceWorkerUrl =
        new URL('./service-worker.js', window.location.href);
    navigator.serviceWorker
        .register(serviceWorkerUrl, {scope: './', updateViaCache: 'none'})
        .then(async registration => {
          await navigator.serviceWorker.ready;
          if (!navigator.serviceWorker.controller) {
            const key = 'manifoldcad-sw-controller-refresh';
            if (!window.sessionStorage.getItem(key)) {
              // One-time marker for this tab to avoid a reload loop.
              window.sessionStorage.setItem(key, '1');
              window.location.reload();
              return;
            }
          } else {
            window.sessionStorage.removeItem(
                'manifoldcad-sw-controller-refresh');
          }
          registration.update();
        })
        .catch(error => {
          console.error('Service worker registration failed:', error);
        });
  }
}

let editor = undefined;

// Pane resizing - draggable pane dividers ---------------------

const LEFT_PANE_MIN_PERCENT = 20;
const LEFT_PANE_MAX_PERCENT = 80;
const VIEWER_PANE_MIN_PERCENT = 35;
const VIEWER_PANE_MAX_PERCENT = 90;

// Keep percentages within practical bounds so panes stay usable.
function clampToRange(value, minValue, maxValue) {
  return Math.min(maxValue, Math.max(minValue, value));
}

// Attach pointer-drag behavior to a splitter element.
// The callback receives pointer-move events and applies the layout update.
function attachSplitterDrag(splitterElement, handleDragMove) {
  if (!splitterElement) return;

  splitterElement.addEventListener('pointerdown', pointerDownEvent => {
    const isMobileLayout = window.matchMedia('(max-width: 820px)').matches;
    if (isMobileLayout) return;

    pointerDownEvent.preventDefault();
    splitterElement.setPointerCapture(pointerDownEvent.pointerId);

    const onPointerMove = moveEvent => handleDragMove(moveEvent);
    const onPointerEnd = endEvent => {
      splitterElement.releasePointerCapture(endEvent.pointerId);
      splitterElement.removeEventListener('pointermove', onPointerMove);
      splitterElement.removeEventListener('pointerup', onPointerEnd);
      splitterElement.removeEventListener('pointercancel', onPointerEnd);
    };

    splitterElement.addEventListener('pointermove', onPointerMove);
    splitterElement.addEventListener('pointerup', onPointerEnd);
    splitterElement.addEventListener('pointercancel', onPointerEnd);
  });
}

function setupPaneSplitters() {
  const pageElement = document.querySelector('.page');
  const workbenchElement = document.getElementById('workbench');
  const rightPaneElement = document.getElementById('rightPane');
  const horizontalSplitterElement = document.getElementById('split-x');
  const verticalSplitterElement = document.getElementById('split-y');
  const leftPaneStorageKey = 'ManifoldCAD:leftPanePercent';
  const viewerPaneStorageKey = 'ManifoldCAD:viewerPanePercent';

  if (!pageElement || !workbenchElement || !rightPaneElement) return;

  // Restore saved pane percentages on refresh when they are valid numbers.
  const savedLeftPane = Number(window.localStorage.getItem(leftPaneStorageKey));
  if (Number.isFinite(savedLeftPane)) {
    const clampedLeftPanePercent = clampToRange(
        savedLeftPane, LEFT_PANE_MIN_PERCENT, LEFT_PANE_MAX_PERCENT);
    pageElement.style.setProperty('--left-pane', `${clampedLeftPanePercent}%`);
  }

  const savedViewerPane =
      Number(window.localStorage.getItem(viewerPaneStorageKey));
  if (Number.isFinite(savedViewerPane)) {
    const clampedViewerPanePercent = clampToRange(
        savedViewerPane, VIEWER_PANE_MIN_PERCENT, VIEWER_PANE_MAX_PERCENT);
    pageElement.style.setProperty(
        '--viewer-pane', `${clampedViewerPanePercent}%`);
  }

  attachSplitterDrag(horizontalSplitterElement, moveEvent => {
    // Convert pointer X position to a percentage of the full workbench width.
    const workbenchBounds = workbenchElement.getBoundingClientRect();
    const leftPanePercent =
        ((moveEvent.clientX - workbenchBounds.left) / workbenchBounds.width) *
        100;
    const clampedLeftPanePercent = clampToRange(
        leftPanePercent, LEFT_PANE_MIN_PERCENT, LEFT_PANE_MAX_PERCENT);
    pageElement.style.setProperty('--left-pane', `${clampedLeftPanePercent}%`);
    window.localStorage.setItem(leftPaneStorageKey, clampedLeftPanePercent);

    // Monaco reacts to container size changes, but an explicit layout keeps
    // drag updates immediate and smooth.
    editor?.layout({});
  });

  attachSplitterDrag(verticalSplitterElement, moveEvent => {
    // Convert pointer Y position to a percentage of the right pane height.
    const rightPaneBounds = rightPaneElement.getBoundingClientRect();
    const viewerPanePercent =
        ((moveEvent.clientY - rightPaneBounds.top) / rightPaneBounds.height) *
        100;
    const clampedViewerPanePercent = clampToRange(
        viewerPanePercent, VIEWER_PANE_MIN_PERCENT, VIEWER_PANE_MAX_PERCENT);
    pageElement.style.setProperty(
        '--viewer-pane', `${clampedViewerPanePercent}%`);
    window.localStorage.setItem(viewerPaneStorageKey, clampedViewerPanePercent);
  });
}

setupPaneSplitters();

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
  console.log('Shareable link copied to clipboard!');
  console.log('Consider shortening this URL using tinyURL.com');
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

function getAllScripts() {
  const files = {};
  for (const [name, contents] of exampleFunctions) {
    files[name] = contents;
  }

  for (let i = 0; i < window.localStorage.length; i++) {
    const key = nthKey(i);
    if (!key || key === 'currentName' || key === 'safe') continue;
    files[key] = getScript(key)
  }
  return files;
}

function getModelForScript(filename) {
  const uri = monaco.Uri.parse(`inmemory://model/${filename}.ts`);
  const model = monaco.editor.getModel(uri) ||
      monaco.editor.createModel('', 'typescript', uri);
  model.updateOptions({tabSize: 2});
  return model;
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
    const code = isExample ? exampleFunctions.get(scriptName) :
                             getScript(scriptName) ?? '';
    window.location.hash = '#' + scriptName;
    const model = getModelForScript(scriptName);
    editor.setModel(model);

    // Either editor.setValue() or model.setValue() will trigger
    // onDidChangeModelContent.  This will cause some UI updates, but will also
    // get monaco-editor-auto-typings to update types.
    model.setValue(code);
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

async function createEditor() {
  self.MonacoEnvironment = {
    getWorker: (_, label) => {
      if (label === 'typescript' || label === 'javascript') {
        return new tsWorker();
      } else {
        return new editorWorker();
      }
    }
  };

  editor = monaco.editor.create(document.getElementById('editor'), {
    language: 'typescript',
    automaticLayout: true,
    minimap: {enabled: false},


    // make monaco editor to wrap the content,and hide horizontal
    // scrollbar----start----:

    // make text wrap to the next line when it exceeds the width of the editor:
    wordWrap: 'on',

    // remove horizontal scrollbar:
    scrollbar: {
      horizontal: 'hidden',
    },
    // make monaco editor to wrap the content,and hide horizontal
    // scrollbar----end-------.


  });

  monaco.languages.typescript.typescriptDefaults.setCompilerOptions({
    module: monaco.languages.typescript.ScriptTarget.ESNext,
    moduleResolution: monaco.languages.typescript.ScriptTarget.NodeNext,
    allowNonTsExtensions: true,
  });

  // Make sure `manifold-3d/manifoldCAD` types are available for import.
  const manifoldCADTypesUrl =
      new URL('./manifoldCAD.d.ts', window.location.href);
  const manifoldCADGlobalsTypesUrl =
      new URL('./manifoldCADGlobals.d.ts', window.location.href);

  monaco.languages.typescript.typescriptDefaults.addExtraLib(
      await (await fetch(manifoldCADTypesUrl)).text(),
      'inmemory://model/node_modules/manifold-3d/manifoldCAD.d.ts');

  // Types in the global namespace for top-level scripts.
  // This could be improved in the future.  API-Extractor intentionally doesn't
  // global variables, so another tool may be a better fit.
  monaco.languages.typescript.typescriptDefaults.addExtraLib(
      (await (await fetch(manifoldCADGlobalsTypesUrl)).text())
          .replace(/^export /gm, ''));

  // Load up all scripts so that monaco can check types of multi-file models.
  for (const [filename, content] of Object.entries(getAllScripts())) {
    getModelForScript(filename).setValue(content);
  }

  // Initialize auto typing on monaco editor.
  const typeIndicator = document.querySelector('#type-indicator');
  let typeIndicatorFrame = 0;
  let autoTypings = undefined;

  const syncTypeIndicator = () => {
    if (!typeIndicator || !autoTypings) return;
    typeIndicator.textContent =
        autoTypings.isResolving ? 'Fetching types...' : '';
    typeIndicatorFrame =
        autoTypings.isResolving ? requestAnimationFrame(syncTypeIndicator) : 0;
  };

  const showTypeIndicator = () => {
    if (!typeIndicator) return;
    typeIndicator.textContent = 'Fetching types...';
    if (autoTypings && typeIndicatorFrame === 0) {
      typeIndicatorFrame = requestAnimationFrame(syncTypeIndicator);
    }
  };

  self.window.typecache = new LocalStorageCache();

  // We inject manifold-3d typings locally above, and text-shaper publishes
  // broken declaration re-exports to non-existent source files. Avoid CDN
  // probes for those packages to keep refreshes quiet.
  // This skip list only affects Monaco auto-typing CDN lookups, not runtime
  // imports.
  const jsDelivrResolver = new JsDelivrSourceResolver();
  const skippedTypingPackages =
      new Set(['manifold-3d', 'text-shaper', '@types/require']);
  const shouldSkipTypingPackage = packageName => {
    return skippedTypingPackages.has(packageName);
  };
  const sourceResolver = {
    resolvePackageJson: async (packageName, version, subPath) => {
      if (shouldSkipTypingPackage(packageName)) return '';
      return jsDelivrResolver.resolvePackageJson(packageName, version, subPath);
    },
    resolveSourceFile: async (packageName, version, path) => {
      if (shouldSkipTypingPackage(packageName)) return '';
      return jsDelivrResolver.resolveSourceFile(packageName, version, path);
    }
  };

  autoTypings = await AutoTypings.create(editor, {
    sourceResolver,
    sourceCache: self.window.typecache,
    // Conservative limits: resolve shallow imports while avoiding deep fetch
    // fan-out that adds noise and slows editor/offline workflows.
    packageRecursionDepth: 1,
    fileRecursionDepth: 2,
    onUpdate: update => {
      if (update.type === 'ResolveNewImports') {
        showTypeIndicator();
      }
    },
    onError: e => {
      if (String(e?.message ?? e).includes('Not implemented yet')) {
        return;
      }
      console.error(e);
    }
  });
  if (typeIndicator?.textContent) {
    syncTypeIndicator();
  }
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
    const activeName = currentFileElement.textContent;

    // The user switched models.
    if (switching) {
      switching = false;
      editor.setScrollTop(0);
      runButton.disabled = false;
      return;
    }

    // monaco-editor-auto-typings loaded types.  Do nothing.
    if (autoTypings.isResolving && e.changes.isFlush) {
      return;
    }

    // The user edited an example.
    // Copy it into a new script.
    if (isExample && exampleFunctions.get(activeName) != editor.getValue()) {
      const cursor = editor.getPosition();
      newItem(editor.getValue()).button.click();
      editor.setPosition(cursor);
      runButton.disabled = false;
      return;
    }

    // And if we're here, the user made an edit.
    runButton.disabled = false;
  });

  window.onresize = () => {
    editor.layout({});
  };
};

createEditor();

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

function finishRun() {
  disableCancel();
  const log = consoleElement.textContent;
  // Remove "Running..."
  consoleElement.textContent = log.substring(log.indexOf('\n') + 1);
}

const output = {
  glbURL: null,
  threeMFURL: null
};
let manifoldWorker = null;

function createWorker() {
  manifoldWorker = new ManifoldWorker();

  manifoldWorker.onmessage = (e) => {
    const message = e.data;

    if (message?.type === 'ready') {
      if (tsWorker != null && !manifoldInitialized) {
        initializeRun();
      }
      manifoldInitialized = true;

    } else if (message?.type === 'error') {
      // Clean up.
      setScript('safe', 'false');
      finishRun();

      // Show errors.  If the stack trace makes more sense, show that.
      const errorText = `${message.name}: ${message.message}`;
      if (message.stack && message.stack.startsWith(errorText)) {
        consoleElement.textContent += message.stack + '\r\n';
      } else {
        consoleElement.textContent += errorText + '\r\n';
      }
      consoleElement.scrollTop = consoleElement.scrollHeight;
      mv.showPoster();
      poster.textContent = 'Error';

      // Clear models.
      if (output.glbURL) URL.revokeObjectURL(output.glbURL);
      if (output.threeMFURL) URL.revokeObjectURL(output.threeMFURL);
      output.glbURL = null;
      output.threeMFURL = null;
      threemfButton.disabled = true;

      // Start all over again.
      createWorker();

    } else if (message?.type === 'log') {
      const logMessage = String(message.message ?? '');
      // Hide noisy per-module CDN fetch traces, keep other worker logs.
      if (!logMessage.startsWith('Fetching http')) {
        consoleElement.textContent += logMessage + '\r\n';
        consoleElement.scrollTop = consoleElement.scrollHeight;
      }

    } else if (message?.type === 'done') {
      setScript('safe', 'true');
      manifoldWorker.postMessage({type: 'export', extension: 'glb'});
      manifoldWorker.postMessage({type: 'export', extension: '3mf'});

      finishRun();
      runButton.disabled = true;

    } else if (message?.type === 'blob') {
      if (message.extension === 'glb') {
        if (output.glbURL) URL.revokeObjectURL(output.glbURL);
        output.glbURL = message.blobURL;

        mv.src = output.glbURL;
      } else if (message?.extension === '3mf') {
        if (output.threeMFURL) URL.revokeObjectURL(output.threeMFURL);
        output.threeMFURL = message.blobURL;
        threemfButton.disabled = false;
      }
    }
  };

  manifoldWorker.postMessage(
      {type: 'initialize', esbuildWasmUrl, manifoldWasmUrl});
}

createWorker();

async function run() {
  saveCurrent();
  setScript('safe', 'false');
  enableCancel();
  clearConsole();
  console.log('Running...');
  const files = {};
  for (const [filename, contents] of Object.entries(getAllScripts())) {
    files[`./${filename}`] = contents;
  }
  const filename = currentFileElement.textContent;
  const code = editor.getValue();
  manifoldWorker.postMessage({
    type: 'evaluate',
    code,
    filename,
    files,
    jsCDN: 'jsDelivr',
    baseUrl: window.location.href
  });
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

function clickSave(saveButton, extension, outputName) {
  const container = saveButton.parentElement;
  return () => {
    const oldSave = saveContainer.firstElementChild;
    if (oldSave !== container) {
      saveDropdown.insertBefore(oldSave, saveDropdown.firstElementChild);
      saveContainer.insertBefore(container, saveDropdown);
      container.appendChild(saveArrow.parentElement);
    }
    const link = document.createElement('a');

    link.download =
        `${currentFileElement.textContent.trim() || 'manifold'}.${extension}`;

    link.href = output[outputName];
    link.click();
  };
}

const glbButton = document.querySelector('#glb');
glbButton.onclick = clickSave(glbButton, 'glb', 'glbURL');

const threemfButton = document.querySelector('#threemf');
threemfButton.onclick = clickSave(threemfButton, '3mf', 'threeMFURL');