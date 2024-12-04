// https://gist.github.com/jamsinclair/6ad148d0590291077a4ce389c2b274ea
import {createFilter} from 'vite';

function isEmscriptenFile(code) {
  return /var\s+Module\s*=|WebAssembly\.instantiate/.test(code) &&
      /var\s+workerOptions\s*=/.test(code);
}

/**
 * Vite plugin that replaces Emscripten workerOptions with static object literal
 * to fix error with Vite See project issue:
 * https://github.com/emscripten-core/emscripten/issues/22394
 *
 * Defaults to running for all .js and .ts files. If there are any issues you
 * can use the include/exclude options.
 *
 * @param {Object} options
 * @property {string[]} [include] - Glob patterns to include
 * @property {string[]} [exclude] - Glob patterns to exclude
 * @returns {import('vite').Plugin}
 */
export default function emscriptenStaticWorkerOptions(options = {}) {
  const filter = createFilter(options.include || /\.[jt]s$/, options.exclude);

  return {
    name: 'emscripten-static-worker-options',
    enforce: 'pre',
    transform(code, id) {
      if (!filter(id)) return null;

      if (!isEmscriptenFile(code)) return null;

      const workerOptionsMatch =
          code.match(/var\s+workerOptions\s*=\s*({[^}]+})/);
      if (!workerOptionsMatch) return null;

      const optionsObjectStr = workerOptionsMatch[1];
      const optionsDeclarationStr = workerOptionsMatch[0];

      const modifiedCode =
          code.replace(optionsDeclarationStr, '')
              .replace(
                  new RegExp('workerOptions(?![\\w$])', 'g'), optionsObjectStr);

      return {code: modifiedCode, map: null};
    }
  };
}
