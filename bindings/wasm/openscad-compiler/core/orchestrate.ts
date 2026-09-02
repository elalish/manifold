import path from 'path';

import {compile} from './compiler.js';
import {compileLibrary, MANIFEST_VERSION} from './library.js';
import {resolveLibraryClosure, resolveProgramWithLibraries,} from './resolver.js';
import {globalFileResolver} from './state.js';
import type {ExternalLibraryRef, LibraryManifest, ResolvedExternalLib} from './types.js';

async function getRuntimeVersion(cwd: string): Promise<string> {
  try {
    const pkg = JSON.parse(
        await globalFileResolver?.readText(path.join(cwd, 'package.json')) as
        string);
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

export async function ensureLibraryCompiled(
    ref: ExternalLibraryRef, entryDir: string, cwd: string,
    log: (msg: string) => void =
        () => {}): Promise<{manifest: LibraryManifest; libDir: string}> {
  const libDir = path.join(cwd, 'runtime', 'libraries', ref.name.toLowerCase());
  const manifestPath = path.join(libDir, '.manifest.json');
  const runtimeVersion = await getRuntimeVersion(cwd);

  // Carry over files from the previous build so existing references keep
  // working across the full set during recompilation
  let priorFiles: string[] = [];
  let staleRuntime = false;
  if (await globalFileResolver?.exists(libDir) &&
      await globalFileResolver?.exists(manifestPath)) {
    const manifest = JSON.parse(
                         await globalFileResolver?.readText(manifestPath) as
                         string) as LibraryManifest;
    const relOf = (abs: string) =>
        path.relative(ref.root, abs).replace(/\\/g, '/');
    const missing = ref.entries.filter(e => !(relOf(e.file) in manifest.files));
    // Emitted code is tied to the runtime it was compiled against, so a version
    // change invalidates every cached file regardless of coverage. The
    // manifest's own shape is versioned too, since its signature keys are read
    // back by the consumer
    staleRuntime = manifest.runtimeVersion !== runtimeVersion ||
        (manifest.manifestVersion ?? 1) !== MANIFEST_VERSION;
    if (!staleRuntime && missing.length === 0) {
      log(`Library ${ref.name}: cache hit (${
          Object.keys(manifest.files).length} files)`);
      return {manifest, libDir};
    }
    priorFiles =
        Object.keys(manifest.files).map(rel => path.join(ref.root, rel));
    log(staleRuntime ?
            `Library ${ref.name}: cached build targets runtime ${
                manifest.runtimeVersion ??
                'unknown'}, current is ${runtimeVersion}; recompiling...` :
            `Library ${ref.name}: cache is missing ${
                missing.map(e => relOf(e.file)).join(', ')}; recompiling...`);
  } else {
    log(`Library ${ref.name}: compiling...`);
  }

  const entryFiles = [...priorFiles];
  for (const e of ref.entries) {
    if (!entryFiles.includes(e.file)) entryFiles.push(e.file);
  }
  const closure =
      await resolveLibraryClosure(ref.name, ref.root, entryFiles, entryDir);
  const runtimeJsAbs = path.join(cwd, 'runtime', 'runtime.js');
  const runtimePathFor = (outRel: string) => toPosixSpecifier(
      path.relative(path.dirname(path.join(libDir, outRel)), runtimeJsAbs));

  const compiled =
      await compileLibrary(closure, {runtimeVersion, runtimePathFor});

  // Clear out any output the new build does not overwrite by name, so nothing
  // emitted by the old version of compiler
  if (staleRuntime) await globalFileResolver?.removeDir(libDir);
  await globalFileResolver?.makeDir(libDir);
  for (const f of compiled.files) {
    const outPath = path.join(libDir, f.outRel);
    await globalFileResolver?.makeDir(path.dirname(outPath));
    await globalFileResolver?.writeText(outPath, f.code);
  }
  // Manifest written LAST so its presence marks a complete build
  await globalFileResolver?.writeText(
      manifestPath, JSON.stringify(compiled.manifest, null, 2));
  log(`Library ${ref.name}: compiled ${compiled.files.length} files`);
  return {manifest: compiled.manifest, libDir};
}

export async function compileConsumer(
    entryFile: string, outputFile: string, cwd: string,
    log: (msg: string) => void = () => {}):
    Promise<
        {code: string; externalLibraries: string[]; resolvedFiles: string[]}> {
  const entryAbs = path.resolve(entryFile);
  const entryDir = path.dirname(entryAbs);
  const resolved = await resolveProgramWithLibraries(entryAbs);

  const outDir = path.dirname(path.resolve(outputFile));
  const externalLibraries: ResolvedExternalLib[] = [];

  for (const [name, ref] of resolved.externalLibraries) {
    const {manifest} = await ensureLibraryCompiled(ref, entryDir, cwd, log);
    const libDir = path.join(cwd, 'runtime', 'libraries', name.toLowerCase())

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
  const code =
      await compile(ast, {runtimePath: runtimeJSPath, externalLibraries});

  return {
    code,
    externalLibraries: [...resolved.externalLibraries.keys()],
    resolvedFiles: resolved.resolvedFiles,
  };
}
