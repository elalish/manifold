import {walk} from './ast.js';
import type {ModuleCallStmt, ModuleDeclStmt} from './ast.js';
import type {BindResult} from './types.js';

const ASYNC_BUILTINS = new Set(['text', 'surface', 'children']);

export const asyncModuleBindings = new Set<number>();
export const externalAsyncModules = new Set<string>();

export function isAsyncModuleCall(stmt: ModuleCallStmt): boolean {
  const binding = stmt.ref?.mod;
  if (!binding) return false;
  if (binding.kind === 'builtin') return ASYNC_BUILTINS.has(stmt.name);
  if (binding.kind === 'external') return externalAsyncModules.has(stmt.name);
  return asyncModuleBindings.has(binding.id);
}

export function moduleBodyNeedsAsync(decl: ModuleDeclStmt): boolean {
  let needsAsync = false;
  walk(decl.body, node => {
    if (needsAsync ||
        node.kind === 'moduleDecl' || node.kind === 'functionDecl' ||
        ('modifier' in node &&
         typeof node.modifier === 'string' &&
         node.modifier.includes('*'))) {
      return false;
    }
    if (node.kind === 'moduleCall' && isAsyncModuleCall(node)) {
      needsAsync = true;
      return false;
    }
  });
  return needsAsync;
}

// Bindings are hoisted before emission, so process each module until async calls stop adding callers
export function findAsyncModules(bind: BindResult): void {
  asyncModuleBindings.clear();
  const declarations = bind.bindings.flatMap(binding => {
    if (binding.ns !== 'mod') return [];
    const decl = binding.decls.at(-1);
    return decl && 'kind' in decl && decl.kind === 'moduleDecl' ?
        [{id: binding.id, decl: decl as ModuleDeclStmt}] :
        [];
  });

  let changed: boolean;
  do {
    changed = false;
    for (const {id, decl} of declarations) {
      if (asyncModuleBindings.has(id)) continue;
      if (moduleBodyNeedsAsync(decl)) {
        asyncModuleBindings.add(id);
        changed = true;
      }
    }
  } while (changed);
}
