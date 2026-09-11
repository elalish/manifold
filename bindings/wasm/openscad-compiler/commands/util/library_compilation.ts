import {existsSync, mkdirSync, readFileSync, rmSync, writeFileSync} from 'fs';
import path from 'path';

import {compileLibrary} from '../../core/library.js';
import {resolveLibraryClosure, resolveProgramWithLibraries} from '../../core/resolver.js';
import type {ExternalLibraryRef, LibraryManifest, ResolvedExternalLib, ResolvedProgramWithLibraries} from '../../core/types.js';

const MANIFEST_VERSION = 1;

function toPosixSpecifier(p: string): string {
  let rel = p.replace(/\\/g, '/').replace(/\.ts$/i, '.js');
  if (!rel.startsWith('.') && !rel.startsWith('/')) rel = './' + rel;
  return rel;
}

function getRuntimeVersion(cwd: string): string {
  try {
    const pkg = JSON.parse(
        readFileSync(path.join(cwd, 'package.json'), 'utf-8') as string);
    return String(pkg.version ?? '0.0.0');
  } catch {
    return '0.0.0';
  }
}

async function ensureLibraryCompiled(
    ref: ExternalLibraryRef, entryDir: string,
    cwd: string): Promise<{manifest: LibraryManifest; libDir: string}> {
  const libDir = path.join(cwd, 'runtime', 'libraries', ref.name.toLowerCase());
  const manifestPath = path.join(libDir, '.manifest.json');
  const runtimeVersion = getRuntimeVersion(cwd);

  // Carry over files from the previous build so existing references keep
  // working across the full set during recompilation
  let priorFiles: string[] = [];
  let staleRuntime = false;
  if (existsSync(libDir) && existsSync(manifestPath)) {
    const manifest =
        JSON.parse(readFileSync(manifestPath, 'utf-8') as string) as
        LibraryManifest;
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
      console.log(`Library ${ref.name}: cache hit (${
          Object.keys(manifest.files).length} files)`);
      return {manifest, libDir};
    }
    priorFiles =
        Object.keys(manifest.files).map(rel => path.join(ref.root, rel));
    console.log(
        staleRuntime ?
            `Library ${ref.name}: cached build targets runtime ${
                manifest.runtimeVersion ??
                'unknown'}, current is ${runtimeVersion}; recompiling...` :
            `Library ${ref.name}: cache is missing ${
                missing.map(e => relOf(e.file)).join(', ')}; recompiling...`);
  } else {
    console.log(`Library ${ref.name}: compiling...`);
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
  if (staleRuntime) rmSync(libDir, {recursive: true, force: true});
  mkdirSync(libDir, {recursive: true});
  for (const f of compiled.files) {
    const outPath = path.join(libDir, f.outRel);
    mkdirSync(path.dirname(outPath), {recursive: true});
    writeFileSync(outPath, f.code, 'utf-8');
  }
  // Manifest written LAST so its presence marks a complete build
  writeFileSync(manifestPath, JSON.stringify(compiled.manifest, null, 2));
  console.log(`Library ${ref.name}: compiled ${compiled.files.length} files`);
  return {manifest: compiled.manifest, libDir};
}

export async function getExternalLibraries(
    absFile: string, outputFile: any): Promise<{
  externalLibraries: ResolvedExternalLib[],
  resolved: ResolvedProgramWithLibraries
}> {
  const entryAbs = path.resolve(absFile);
  const entryDir = path.dirname(entryAbs);
  const resolved = await resolveProgramWithLibraries(entryAbs);
  const outDir = path.dirname(path.resolve(outputFile));
  const externalLibraries: ResolvedExternalLib[] = [];

  for (const [name, ref] of resolved.externalLibraries) {
    const {manifest} =
        await ensureLibraryCompiled(ref, entryDir, process.cwd());
    const libDir =
        path.join(process.cwd(), 'runtime', 'libraries', name.toLowerCase())

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

  return {externalLibraries, resolved};
}