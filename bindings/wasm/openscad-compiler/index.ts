#!/usr/bin/env node
import { Command } from "commander";
import compileSingleFileCommand from "./commands/compile.js";
import compileAllCommand from "./commands/compile_all.js";

const program = new Command();

program.name("openscad-to-manifold")
    .description("Convert OpenSCAD files to manifold mesh files")
    .version("1.0.0")

program.addCommand(compileSingleFileCommand);
program.addCommand(compileAllCommand);

program.parse(process.argv);