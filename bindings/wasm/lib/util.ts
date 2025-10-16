// Copyright 2025 The Manifold Authors.
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

import * as lexer from 'es-module-lexer';

/**
 * Are we in a web worker?
 *
 * @returns A boolean.
 */
export const isWebWorker = (): boolean =>
    typeof self !== 'undefined' && typeof self.document == 'undefined';

const decomment = (code: string): string =>
    code.replaceAll(/\/\*[\s\S]*?\*\/|\/\/.*/gm, '');
const isLocal = (specifier: string): boolean =>
    !!specifier.match(/^(\/|\.\/|\.\.\/)/);
// const isRelative = (specifier: string): boolean =>
//     !!specifier.match(/^(\\.\/|\.\.\/)/);
const isHttp = (specifier: string): boolean =>
    !!specifier.match(/^https?:\/\//);
const replace =
    (source: string, start: number, end: number, sub: string): string =>
        source.substring(0, start) + sub + source.substring(end);

/**
 * Extract target variable names from a static import statement, and rewrite
 * them as either a simple assignment or a destructuring assignment.
 *
 * @param importAssignment The assignment part of a static import statment.
 * @returns A string defining a variable or a destructuring assignment
 */
const rewriteImportAssignment = (statement: string) => {
  let assignment =
      decomment(statement).replace(/^import/, '').replace(/from.+$/, '').trim();

  if (assignment.match(/\{/)) {  // Is the assignee an object?
    // Destructuring assignment.
    // '{ foo as bar }' -> '{ foo: bar }'
    // '{ foo, bar }' => '{ foo, bar }'
    const assignments =
        assignment.trim()
            .replace(/^{/, '')
            .replace(/}$/, '')
            .split(',')
            .map(x => x.split('as').map(x => x.trim()))
            .map(([theirname,
                   ourname]) => `${theirname}${ourname ? ': ' + ourname : ''}`)
            .join(', ');
    return `{ ${assignments} }`;

  } else {
    // Single assignment.
    if (assignment.match(/\*\s+as/)) {
      // Return a named object.
      const namespace = assignment.replace(/\*\s+as/, '').trim();
      return `${namespace}`;

    } else {
      // Assign default export.
      return `{default:${assignment}}`;
    }
  }
};

/**
 * Transform static imports into dynamic imports.
 *
 * Static imports can only be used at the module level, and are typically
 * evaluated when bundling.  Dynamic imports will be evaluated at run time.
 *
 * This is a naive implementation.  Any imports beyond basic
 * static imports will be left untouched.  This includes static imports with
 * assertions, deferred imports and source imports.
 *
 * All heavy lifting is done by
 * [es-module-lexer](https://github.com/guybedford/es-module-lexer).
 *
 * @param code Javascript code potentially including static import statements
 * @returns Transformed code containing dynamic imports
 */
export const transformStaticImportsToDynamic = (code: string): string => {
  let transformed = code;

  lexer.initSync();
  const [imports] = lexer.parse(transformed);
  // Work from back to front.  This way we only change the positions of imports
  // we have already transformed.
  for (const imp of [...imports].reverse()) {
    if (imp.t !== 1) continue;   // Static imports only.
    if (imp.a !== -1) continue;  // No assertions.

    const assignment = rewriteImportAssignment(code.slice(imp.ss, imp.se));
    const specifier = code.slice(imp.s, imp.e);
    const dynamicImport = `const ${assignment} = await import('${specifier}')`;

    transformed = replace(transformed, imp.ss, imp.se, dynamicImport);
  }

  return transformed;
};

/**
 * These content delivery networks provide NPM modules as ES modules,
 * whether they were published that way or not.
 */
export const cdnUrlHelpers: {[key: string]: (specifier: string) => string} = {
  'esm.sh': (specifier) => `https://esm.sh/${specifier}`,
  'jsDelivr': (specifier) => `https://cdn.jsdelivr.net/npm/${specifier}/+esm`,
  'skypack': (specifier) => `https://cdn.skypack.dev/${specifier}`
};

/**
 * Rewrite ES module imports to load from a content distribution network.
 *
 * This will step through all import statements that have static module
 * specifiers. If the specifier is not a local file, and not already an
 * HTTP/HTTPS URL, format it as a URL for a given CDN.
 *
 * If we don't have a helper for the CDN, use it as a prefix.
 *
 * All heavy lifting is done by
 * [es-module-lexer](https://github.com/guybedford/es-module-lexer).
 *
 * @param code javascript code to transform
 * @param cdn The name of the CDN, as specified in `cdnUrlHelpers`
 * @returns Transformed code referencing CDN served modules where possible.
 */
export const transformImportsForCDN = (code: string, cdn: string): string => {
  let transformed = code;

  lexer.initSync();
  const [imports] = lexer.parse(transformed);
  for (const imp of [...imports].reverse()) {
    const specifier = code.slice(imp.s, imp.e);
    if (isLocal(specifier) || isHttp(specifier)) {
      continue;  // Nothing to do here.
    }

    const format = cdnUrlHelpers[cdn];
    transformed = replace(
        transformed, imp.s, imp.e,
        format ? format(specifier) : `${cdn}${specifier}`);
  }

  return transformed;
};

export const dropStaticImport = (code: string, moduleName: string): string => {
  let transformed = code;

  lexer.initSync();
  const [imports] = lexer.parse(transformed);
  // Work from back to front.  This way we only change the positions of imports
  // we have already transformed.
  for (const imp of [...imports].reverse()) {
    if (imp.t !== 1) continue;   // Static imports only.
    if (imp.a !== -1) continue;  // No assertions.

    const specifier = code.slice(imp.s, imp.e);
    if (specifier !== moduleName) continue;

    transformed = replace(transformed, imp.ss, imp.se, '');
    console.log(transformed);
  }

  return transformed;
};
