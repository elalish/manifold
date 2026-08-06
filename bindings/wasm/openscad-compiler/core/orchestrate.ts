import fs from 'fs';
import path from 'path';

import {compile, compileLibrary} from './compiler.js';
import type {LibraryManifest, ResolvedExternalLib} from './compiler.js';
import {type ExternalLibraryRef, resolveLibraryClosure, resolveProgramWithLibraries,} from './resolver.js';

function getRuntimeVersion(cwd: string): string {
  try {
    const pkg =
        JSON.parse(fs.readFileSync(path.join(cwd, 'package.json'), 'utf8'));
    return String(pkg.version ?? '0.0.0');
  } catch {
    return '0.0.0';
  }
}

function toPosixSpecifier(p: string): string {
  let rel = p.replace(/\\/g, '/').replace(/\.ts$/i, '.js');
  if (!rel.startsWith('.') && !rel.startsWith('/')) rel = './' + rel;
  return rel;
}

// Directory a library is compiled into: runtime/libraries/<lowercased name>
function libraryDir(cwd: string, libName: string): string {
  return path.join(cwd, 'runtime', 'libraries', libName.toLowerCase());
}

export function ensureLibraryCompiled(
    ref: ExternalLibraryRef, libraryPaths: string[], cwd: string,
    log: (msg: string) => void =
        () => {}): {manifest: LibraryManifest; libDir: string} {
  const libDir = libraryDir(cwd, ref.name);
  const manifestPath = path.join(libDir, '.manifest.json');
  const runtimeVersion = getRuntimeVersion(cwd);

  // Carry over files from the previous build so existing references keep
  // working across the full set during recompilation
  let priorFiles: string[] = [];
  let staleRuntime = false;
  if (fs.existsSync(libDir) && fs.existsSync(manifestPath)) {
    const manifest =
        JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as LibraryManifest;
    const relOf = (abs: string) =>
        path.relative(ref.root, abs).replace(/\\/g, '/');
    const missing = ref.entries.filter(e => !(relOf(e.file) in manifest.files));
    // Emitted code is tied to the runtime it was compiled against, so a version change invalidates every cached file regardless of coverage
    staleRuntime = manifest.runtimeVersion !== runtimeVersion;
    if (!staleRuntime && missing.length === 0) {
      log(`Library ${ref.name}: cache hit (${
          Object.keys(manifest.files).length} files)`);
      return {manifest, libDir};
    }
    priorFiles =
        Object.keys(manifest.files).map(rel => path.join(ref.root, rel));
    log(staleRuntime ? `Library ${ref.name}: cached build targets runtime ${
                           manifest.runtimeVersion ?? 'unknown'}, current is ${
                           runtimeVersion}; recompiling...` :
                       `Library ${ref.name}: cache is missing ${
                           missing.map(e => relOf(e.file)).join(', ')
                       }; recompiling...`);
  } else {
    log(`Library ${ref.name}: compiling...`);
  }

  const entryFiles = [...priorFiles];
  for (const e of ref.entries) {
    if (!entryFiles.includes(e.file)) entryFiles.push(e.file);
  }
  const closure =
      resolveLibraryClosure(ref.name, ref.root, entryFiles, libraryPaths);
  const runtimeJsAbs = path.join(cwd, 'runtime', 'runtime.js');
  const runtimePathFor = (outRel: string) => toPosixSpecifier(
      path.relative(path.dirname(path.join(libDir, outRel)), runtimeJsAbs));

  const compiled = compileLibrary(closure, {runtimeVersion, runtimePathFor});

  // Clear out any output the new build does not overwrite by name, so nothing emitted by the old version of compiler
  if (staleRuntime) fs.rmSync(libDir, {recursive: true, force: true});
  fs.mkdirSync(libDir, {recursive: true});
  for (const f of compiled.files) {
    const outPath = path.join(libDir, f.outRel);
    fs.mkdirSync(path.dirname(outPath), {recursive: true});
    fs.writeFileSync(outPath, f.code);
  }
  // Manifest written LAST so its presence marks a complete build
  fs.writeFileSync(manifestPath, JSON.stringify(compiled.manifest, null, 2));
  log(`Library ${ref.name}: compiled ${compiled.files.length} files`);
  return {manifest: compiled.manifest, libDir};
}

export function compileConsumer(
    entryFile: string, outputFile: string, libraryPaths: string[],
    cwd: string = process.cwd(), log: (msg: string) => void = () => {}):
    {code: string; externalLibraries: string[]; resolvedFiles: string[]} {
  const entryAbs = path.resolve(entryFile);
  const resolved = resolveProgramWithLibraries(entryAbs, libraryPaths);

  const outDir = path.dirname(path.resolve(outputFile));
  const externalLibraries: ResolvedExternalLib[] = [];

  for (const [name, ref] of resolved.externalLibraries) {
    const {manifest} = ensureLibraryCompiled(ref, libraryPaths, cwd, log);
    const libDir = libraryDir(cwd, name);

    const importSpecifierFor = (sourceRel: string): string => {
      const out = manifest.files[sourceRel]?.out ??
          sourceRel.replace(/\.scad$/i, '.ts');
      return toPosixSpecifier(path.relative(outDir, path.join(libDir, out)));
    };

    // Side-effect import for each include-mode entry (relative to library root)
    const sideEffectSpecifiers: string[] = [];
    for (const entry of ref.entries) {
      if (entry.mode !== 'include') continue;
      const sourceRel = path.relative(ref.root, entry.file).replace(/\\/g, '/');
      sideEffectSpecifiers.push(importSpecifierFor(sourceRel));
    }

    externalLibraries.push(
        {name, manifest, importSpecifierFor, sideEffectSpecifiers});
  }

  let relPath = path.relative(outDir, cwd);
  if (relPath === '') relPath = '.';
  let rp = relPath.replace(/\\/g, '/');
  if (!rp.startsWith('.') && !rp.startsWith('/')) rp = './' + rp;
  const runtimeJSPath = rp + '/runtime/runtime.js';

  const ast = {
    kind: 'program' as const,
    statements: resolved.statements,
    filename: entryAbs
  };
  const code = compile(ast, {runtimePath: runtimeJSPath, externalLibraries});

  return {
    code,
    externalLibraries: [...resolved.externalLibraries.keys()],
    resolvedFiles: resolved.resolvedFiles,
  };
}
