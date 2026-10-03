import path from 'path-browserify';

import {compileLibrary} from '../../core/library.js';
import {resolveLibraryClosure, resolveProgramWithLibraries} from '../../core/resolver.js';
import type {ExternalLibraryRef, LibraryManifest, ResolvedExternalLib, ResolvedProgramWithLibraries} from '../../core/types.js';
import {vfs} from '../../vfs/vfs.js';

const MANIFEST_VERSION = 1;
const RUNTIME_VERSION = '1.0.0';

function toPosixSpecifier(p: string): string {
  let rel = p.replace(/\\/g, '/').replace(/\.ts$/i, '.js');
  if (!rel.startsWith('.') && !rel.startsWith('/')) rel = './' + rel;
  return rel;
}

async function ensureLibraryCompiled(
    ref: ExternalLibraryRef, entryDir: string,
    cwd: string): Promise<{manifest: LibraryManifest; libDir: string}> {
  const libDir =
      '/' + path.join('runtime', 'libraries', ref.name.toLowerCase());
  const manifestPath = path.join(libDir, '.manifest.json');
  const runtimeVersion = RUNTIME_VERSION;

  // Carry over files from the previous build so existing references keep
  // working across the full set during recompilation
  let priorFiles: string[] = [];
  let staleRuntime = false;
  if (vfs.existsSync(libDir) && vfs.existsSync(manifestPath)) {
    const manifest =
        JSON.parse(vfs.readFileSync(manifestPath, 'utf-8') as string) as
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
  if (staleRuntime) vfs.rmSync(libDir, {recursive: true, force: true});
  vfs.mkdirSync(libDir, {recursive: true});
  for (const f of compiled.files) {
    const outPath = path.join(libDir, f.outRel);
    vfs.mkdirSync(path.dirname(outPath), {recursive: true});
    vfs.writeFileSync(outPath, f.code, 'utf-8');
  }
  // Manifest written LAST so its presence marks a complete build
  vfs.writeFileSync(manifestPath, JSON.stringify(compiled.manifest, null, 2));
  console.log(`Library ${ref.name}: compiled ${compiled.files.length} files`);
  return {manifest: compiled.manifest, libDir};
}

export async function getExternalLibraries(
    absFile: string, outputFile: any): Promise<{
  externalLibraries: ResolvedExternalLib[],
  resolved: ResolvedProgramWithLibraries
}> {
  const entryDir = '/';
  const resolved = await resolveProgramWithLibraries(absFile);
  const outDir = '/';
  const externalLibraries: ResolvedExternalLib[] = [];

  for (const [name, ref] of resolved.externalLibraries) {
    const {manifest} = await ensureLibraryCompiled(ref, entryDir, '/');
    const libDir = '/' + path.join('runtime', 'libraries', name.toLowerCase())

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
