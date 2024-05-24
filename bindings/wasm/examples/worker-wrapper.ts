import { module, exportModels, exposedFunctions, cleanup } from "./worker";
import * as glMatrix from 'gl-matrix';

// Setup complete
self.postMessage(null);

if (self.console) {
  const oldLog = self.console.log;
  self.console.log = function (...args) {
    let message = '';
    for (const arg of args) {
      if (arg == null) {
        message += 'undefined';
      } else if (typeof arg == 'object') {
        message += JSON.stringify(arg, null, 4);
      } else {
        message += arg.toString();
      }
    }
    self.postMessage({ log: message });
    oldLog(...args);
  };
}

self.onmessage = async (e) => {
  const content = 'const globalDefaults = {};\n' + e.data +
    '\nreturn exportModels(globalDefaults, typeof result === "undefined" ? undefined : result);\n';
  try {
    const f = new Function(
      'exportModels', 'glMatrix', 'module', ...exposedFunctions, content);
    const result = await f(
      exportModels, glMatrix, module,  //@ts-ignore
      ...exposedFunctions.map(name => module[name]));
    self.postMessage(result);
  } catch (error: any) {
    console.log(error.toString());
    self.postMessage({ objectURL: null });
  } finally {
    module.cleanup();
    cleanup();
  }
};
