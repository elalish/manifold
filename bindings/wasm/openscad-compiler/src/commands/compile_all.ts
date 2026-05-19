import { Command } from "commander";
import fs from "fs";
import { compile } from "../core/compiler.js";
import { resolveProgram, getOpenSCADLibraryPaths } from "../core/resolver.js";

import path from "path";

const compileAllCommand = new Command();

const exampleDir = "examples";

function getAllScadFiles(dir: string, baseDir: string = dir): string[] {
    let results: string[] = [];
    const list = fs.readdirSync(dir, { withFileTypes: true });
    for (const dirent of list) {
        const fullPath = path.join(dir, dirent.name);
        if (dirent.isDirectory()) {
            results = results.concat(getAllScadFiles(fullPath, baseDir));
        } else if (dirent.isFile() && fullPath.endsWith(".scad")) {
            results.push(path.relative(baseDir, fullPath).replace(/\\/g, '/'));
        }
    }
    return results;
}


compileAllCommand.name("compile-all")
    .description("Compile all OpenSCAD files in the examples directory to manifold mesh files")
    .action(() => {
        try {
            const allFiles = getAllScadFiles(exampleDir);
            console.log("All files to compile:", allFiles);

            const failed: string[] = [];

            for (const file of allFiles) {
                console.log(`\n=== Compiling ${file} ===`);
                const absFile = path.resolve(exampleDir, file);

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

                    const basename = path.basename(file, path.extname(file));
                    const outputFile = path.join("out", path.dirname(file), basename + ".ts");

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
                    const message = (err as Error).message;
                    console.error(`Error compiling ${file}: ${message}`);
                    failed.push(file);
                }
            }

            if (failed.length > 0) {
                console.error(`\nFailed files (${failed.length}): ${failed.join(", ")}`);
                process.exitCode = 1;
            }
        } catch (error) {
            console.log("An Error Occured: " + error);
            process.exitCode = 1;
        }
    });


export default compileAllCommand
