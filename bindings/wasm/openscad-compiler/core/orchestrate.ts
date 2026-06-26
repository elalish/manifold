import fs from "fs";
import path from "path";
import { compile, compileLibrary } from "./compiler.js";
import type { ResolvedExternalLib, LibraryManifest } from "./compiler.js";
import {
  resolveProgramWithLibraries,
  resolveLibraryClosure,
  type ExternalLibraryRef,
} from "./resolver.js";

function getRuntimeVersion(cwd: string): string {
  try {
    const pkg = JSON.parse(fs.readFileSync(path.join(cwd, "package.json"), "utf8"));
    return String(pkg.version ?? "0.0.0");
  } catch {
    return "0.0.0";
  }
}

function toPosixSpecifier(p: string): string {
  let rel = p.replace(/\\/g, "/").replace(/\.ts$/i, ".js");
  if (!rel.startsWith(".") && !rel.startsWith("/")) rel = "./" + rel;
  return rel;
}

// Directory a library is compiled into: runtime/libraries/<lowercased name>
function libraryDir(cwd: string, libName: string): string {
  return path.join(cwd, "runtime", "libraries", libName.toLowerCase());
}

export function ensureLibraryCompiled(ref: ExternalLibraryRef, libraryPaths: string[], cwd: string, log: (msg: string) => void = () => {}): { manifest: LibraryManifest; libDir: string } {
  const libDir = libraryDir(cwd, ref.name);
  const manifestPath = path.join(libDir, ".manifest.json");

  if (fs.existsSync(libDir) && fs.existsSync(manifestPath)) {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8")) as LibraryManifest;
    log(`Library ${ref.name}: cache hit (${Object.keys(manifest.files).length} files)`);
    return { manifest, libDir };
  }

  log(`Library ${ref.name}: compiling...`);
  const closure = resolveLibraryClosure(ref.name, ref.root, ref.entries.map(e => e.file), libraryPaths);
  const runtimeJsAbs = path.join(cwd, "runtime", "runtime.js");
  const runtimePathFor = (outRel: string) =>
    toPosixSpecifier(path.relative(path.dirname(path.join(libDir, outRel)), runtimeJsAbs));

  const compiled = compileLibrary(closure, { runtimeVersion: getRuntimeVersion(cwd), runtimePathFor });

  fs.mkdirSync(libDir, { recursive: true });
  for (const f of compiled.files) {
    const outPath = path.join(libDir, f.outRel);
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, f.code);
  }
  // Manifest written LAST so its presence marks a complete build
  fs.writeFileSync(manifestPath, JSON.stringify(compiled.manifest, null, 2));
  log(`Library ${ref.name}: compiled ${compiled.files.length} files`);
  return { manifest: compiled.manifest, libDir };
}

export function compileConsumer(entryFile: string, outputFile: string, libraryPaths: string[], cwd: string = process.cwd(), log: (msg: string) => void = () => {}): { code: string; externalLibraries: string[]; resolvedFiles: string[] } {
  const entryAbs = path.resolve(entryFile);
  const resolved = resolveProgramWithLibraries(entryAbs, libraryPaths);

  const outDir = path.dirname(path.resolve(outputFile));
  const externalLibraries: ResolvedExternalLib[] = [];

  for (const [name, ref] of resolved.externalLibraries) {
    const { manifest } = ensureLibraryCompiled(ref, libraryPaths, cwd, log);
    const libDir = libraryDir(cwd, name);

    const importSpecifierFor = (sourceRel: string): string => {
      const out = manifest.files[sourceRel]?.out ?? sourceRel.replace(/\.scad$/i, ".ts");
      return toPosixSpecifier(path.relative(outDir, path.join(libDir, out)));
    };

    // Side-effect import for each include-mode entry (relative to library root)
    const sideEffectSpecifiers: string[] = [];
    for (const entry of ref.entries) {
      if (entry.mode !== "include") continue;
      const sourceRel = path.relative(ref.root, entry.file).replace(/\\/g, "/");
      sideEffectSpecifiers.push(importSpecifierFor(sourceRel));
    }

    externalLibraries.push({ name, manifest, importSpecifierFor, sideEffectSpecifiers });
  }

  let relPath = path.relative(outDir, cwd);
  if (relPath === "") relPath = ".";
  let rp = relPath.replace(/\\/g, "/");
  if (!rp.startsWith(".") && !rp.startsWith("/")) rp = "./" + rp;
  const runtimeJSPath = rp + "/runtime/runtime.js";

  const ast = { kind: "program" as const, statements: resolved.statements };
  const code = compile(ast, { runtimePath: runtimeJSPath, externalLibraries });

  return {
    code,
    externalLibraries: [...resolved.externalLibraries.keys()],
    resolvedFiles: resolved.resolvedFiles,
  };
}
