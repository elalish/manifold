import { module, cleanup, evaluateCADToModel } from "./worker";

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
  try {
    const result = await evaluateCADToModel(e.data);
    self.postMessage(result);
  } catch (error: any) {
    console.log(error.toString());
    self.postMessage({ objectURL: null });
  } finally {
    module.cleanup();
    cleanup();
  }
};
