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
 * This is a naive implementation.  Rather than building an edit list, it edits
 * and reparses the source code iteratively.  It will return when it cannot
 * find any import statements it is able to modify; any imports beyond basic
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
      lexer.initSync();

      while (true) {
        const [imports] = lexer.parse(transformed);
        const staticImports =
            imports
                .filter(imp => imp.t === 1)    // Static imports.
                .filter(imp => imp.a === -1);  // No assertions.

        const imp = staticImports.pop();
        if (!imp) break;  // Nothing to do.

        let specifier = code.slice(imp.s, imp.e);
        if (!isLocal(specifier) && !isHttp(specifier)) {
          // This is a remote package, but apparently not a URL.
          specifier = remotePrefix + specifier;
        }

        const assignee = code.slice(imp.ss, imp.se)
                             .replace(/^import/, '')
                             .replace(/from.+$/, '')
                             .trim();

        let dynamicImport = '';
        if (assignee.match(/\{/)) {  // Is the assignee an object?
          // Destructuring assignment.
          dynamicImport = `const ${assignee} = await import('${specifier}')`;
        } else {
          // Default assignment.
          dynamicImport =
              `const {default:${assignee}} = await import('${specifier}')`;
        }

        // Replace the import.
        const pre = transformed.substring(0, imp.ss);
        const post = transformed.substring(imp.se);
        transformed = pre + dynamicImport + post;
      }

      return transformed;
    };