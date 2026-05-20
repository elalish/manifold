import { Command } from "commander";
import fs from "fs";
import { compile } from "../core/compiler.js";
import { resolveProgram, getOpenSCADLibraryPaths } from "../core/resolver.js";
import path from "path";

const compileSingleFileCommand = new Command();

compileSingleFileCommand.name("compile")
    .description("Compile OpenSCAD files to manifold mesh files")
    .argument("<input>", "Input file path")
    .option("--output <output>", "Output file path")
    .action((input, options) => {
        try {
            if (!input) {
                console.log("Error: Input file path is required");
                process.exit(1);
            }

            const userGivenOutPutPath = options.output;

            // check userGivenOuputPath is valid ts file path
            if (userGivenOutPutPath && !userGivenOutPutPath.endsWith(".ts")) {
                console.log("Error: Output file path is not valid");
                process.exit(1);
            }

            const file = input;
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

                const outputFile = userGivenOutPutPath || path.join("test/out", path.basename(file, path.extname(file)) + ".ts");

                let relativePath = path.relative(path.dirname(outputFile), process.cwd());
                if (relativePath === "") relativePath = ".";
                let runtimePath = relativePath.replace(/\\/g, "/");
                if (!runtimePath.startsWith(".") && !runtimePath.startsWith("/")) {
                    runtimePath = "./" + runtimePath;
                }
                const runtimeJSPath = runtimePath + "/runtime.js";

                const ast = { kind: "program" as const, statements: resolved.statements };
                const js = compile(ast, { runtimePath: runtimeJSPath });
                console.log(`Generated TypeScript (${js.length.toLocaleString()} chars)`);
                fs.mkdirSync(path.dirname(outputFile), { recursive: true });
                fs.writeFileSync(outputFile, js);
                console.log(`Output written to ${outputFile}`);
            } catch (err) {
                console.error(`Error: ${(err as Error).message}`);
                process.exitCode = 1;
            }
        } catch (error) {
            console.log("An error occurred: " + error);
            process.exit(1);
        }
    });

export default compileSingleFileCommand;
