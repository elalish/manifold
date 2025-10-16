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
 * @param code Javascript code potentially including static import statements.
 * @param remotePrefix Optionally, prefix remote packages with this string.
 *     I.e.: 'https://esm.run/'
 * @returns Transformed code containing dynamic imports.
 */
export const transformStaticImportsToDynamic =
    (code: string, remotePrefix?: string): string => {
      let transformed = code;

      const isLocal = (x: string) => x.match(/^(\/|\.\/|\.\.\/)/);
      const isHttp = (x: string) => x.match(/^https?:\/\//);
      const decomment = (x: string) =>
          x.replaceAll(/\/\*[\s\S]*?\*\/|\/\/.*/gm, '');
      lexer.initSync();

      const [imports] = lexer.parse(transformed);
      const staticImports = imports
                                .filter(imp => imp.t === 1)  // Static imports.
                                .filter(imp => imp.a === -1);  // No assertions.

      // Work from back to front.
      // This way we only change the positions of imports
      // we have already transformed.
      for (const imp of staticImports.reverse()) {
        let specifier = code.slice(imp.s, imp.e);
        if (remotePrefix && !isLocal(specifier) && !isHttp(specifier)) {
          // This is a remote package, but apparently not a URL.
          specifier = remotePrefix + specifier;
        }

        let assignee = code.slice(imp.ss, imp.se)
                           .replace(/^import/, '')
                           .replace(/from.+$/, '');
        assignee = decomment(assignee).trim();

        let dynamicImport = '';
        if (assignee.match(/\{/)) {  // Is the assignee an object?
          // Destructuring assignment.
          // '{ foo as bar }' -> '{ foo: bar }'
          // '{ foo, bar }' => '{ foo, bar }'
          const assignments =
              assignee.trim()
                  .replace(/^{/, '')
                  .replace(/}$/, '')
                  .split(',')
                  .map(x => x.split('as').map(x => x.trim()))
                  .map(
                      ([theirname, ourname]) =>
                          `${theirname}${ourname ? ': ' + ourname : ''}`)
                  .join(', ');
          dynamicImport =
              `const { ${assignments} } = await import('${specifier}')`;
        } else {
          // Single assignment.
          if (assignee.match(/\*\s+as/)) {
            // Return a named object.
            const namespace = assignee.replace(/\*\s+as/, '').trim();
            dynamicImport = `const ${namespace} = await import('${specifier}')`;

          } else {
            // Assign default export.
            dynamicImport =
                `const {default:${assignee}} = await import('${specifier}')`;
          }
        }

        // Replace the import.
        const pre = transformed.substring(0, imp.ss);
        const post = transformed.substring(imp.se);
        transformed = pre + dynamicImport + post;
      }

      return transformed;
    };