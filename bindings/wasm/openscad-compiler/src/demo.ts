import fs from "fs";
import { compile } from "./core/compiler.js";
import { resolveProgram, getOpenSCADLibraryPaths } from "./core/resolver.js";

import path from "path";

const file = process.argv[2] || "examples/cube.scad";
const absFile = path.resolve(file);

try {
  const libraryPaths = [
    path.dirname(absFile),
    process.cwd(),
    ...getOpenSCADLibraryPaths(),
  ];

  // Resolve all include/use directives recursively
  const resolved = resolveProgram(absFile, libraryPaths);

  if (resolved.resolvedFiles.length > 1) {
    console.log(`Resolved ${resolved.resolvedFiles.length} files:`);
    for (const f of resolved.resolvedFiles) {
      console.log(`  ${path.relative(process.cwd(), f)}`);
    }
  }

  const outputFile = path.join("out", path.basename(file, path.extname(file)) + ".ts");

  let relativePath = path.relative(path.dirname(outputFile), "out");
  if (relativePath === "") relativePath = ".";
  const runtimePath = relativePath.replace(/\\/g, "/") + "/runtime.js";

  const ast = { kind: "program" as const, statements: resolved.statements };
  const js = compile(ast, { runtimePath });
  console.log(`Generated TypeScript (${js.length.toLocaleString()} chars)`);
  fs.mkdirSync(path.dirname(outputFile), { recursive: true });
  fs.writeFileSync(outputFile, js);
  console.log(`Output written to ${outputFile}`);
} catch (err) {
  console.error(`Error: ${(err as Error).message}`);
  process.exitCode = 1;
}