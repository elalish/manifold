import {init, parse} from 'es-module-lexer';
import path from 'path-browserify';
import {transform} from 'sucrase';

export function stripTypes(code: string): string {
  return transform(code, {
           transforms: ['typescript'],
           disableESTransforms: true,
         })
      .code;
}

// The runtime is served by the host, while compiled library files live in VFS
export const runtimeUrl = new URL('./runtime/runtime.js', location.href).href;

function resolveSpecifier(
    specifier: string, filePath: string, blobUrls: ReadonlyMap<string, string>,
    runtimeImportUrl: string): string|undefined {
  if (!specifier.startsWith('.') && !specifier.startsWith('/')) {
    return specifier;
  }

  const absPath = path.resolve(path.dirname(filePath), specifier);
  if (absPath === '/runtime/runtime.js') return runtimeImportUrl;

  return blobUrls.get(absPath) ??
      blobUrls.get(absPath.replace(/\.js$/i, '.ts'));
}

// The lexer only returns actual module specifiers, ignoring "import" or "from"
// in strings and comments
export async function rewriteImports(
    code: string, filePath: string, blobUrls: ReadonlyMap<string, string>,
    runtimeImportUrl =
        runtimeUrl): Promise<{code: string; unresolved: string[]}> {
  await init;
  const [imports] = parse(code);
  const unresolved: string[] = [];
  let rewritten = code;

  // replace from the end so the lexer's source offsets remain valid
  for (const entry of [...imports].reverse()) {
    if (entry.d !== -1 || entry.n === undefined) continue;
    const resolved =
        resolveSpecifier(entry.n, filePath, blobUrls, runtimeImportUrl);
    if (resolved === undefined) {
      unresolved.push(entry.n);
      continue;
    }
    rewritten =
        rewritten.slice(0, entry.s) + resolved + rewritten.slice(entry.e);
  }

  return {code: rewritten, unresolved: unresolved.reverse()};
}
