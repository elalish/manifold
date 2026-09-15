import {Command} from 'commander';
import fs from 'fs';
import path from 'path';

import {compileConsumer} from '../core/orchestrate.js';
import {setGlobalFileResolver} from '../core/state.js';
import {nodeFileResolver} from '../host/node.js';

import {getExternalLibraries} from './util/library_compilation.js';

const compileSingleFileCommand = new Command();

compileSingleFileCommand.name('compile')
    .description('Compile OpenSCAD files to manifold mesh files')
    .argument('<input>', 'Input file path')
    .option('--output <output>', 'Output file path')
    .action(async (input, options) => {
      try {
        if (!input) {
          console.log('Error: Input file path is required');
          process.exit(1);
        }

        const userGivenOutPutPath = options.output;

        // check userGivenOuputPath is valid ts file path
        if (userGivenOutPutPath && !userGivenOutPutPath.endsWith('.ts')) {
          console.log('Error: Output file path is not valid');
          process.exit(1);
        }

        const file = input;
        const absFile = path.resolve(file);

        try {
          const outputFile = userGivenOutPutPath ||
              path.join(
                  'test/out', path.basename(file, path.extname(file)) + '.ts');

          setGlobalFileResolver(nodeFileResolver);

          const {externalLibraries, resolved} =
              await getExternalLibraries(absFile, outputFile);

          const {code: js, resolvedFiles} = await compileConsumer(
              absFile, outputFile, process.cwd(), externalLibraries, resolved);
          if (resolvedFiles.length > 1) {
            console.log(`Resolved ${resolvedFiles.length} local files`);
          }
          if (externalLibraries.length > 0) {
            console.log(`External libraries: ${externalLibraries.join(', ')}`);
          }
          console.log(
              `Generated TypeScript (${js.length.toLocaleString()} chars)`);
          fs.mkdirSync(path.dirname(outputFile), {recursive: true});
          fs.writeFileSync(outputFile, js);
          console.log(`Output written to ${outputFile}`);
        } catch (err) {
          console.error(`Error: ${(err as Error).message}`);
          process.exitCode = 1;
        }
      } catch (error) {
        console.log('An error occurred: ' + error);
        process.exit(1);
      }
    });

export default compileSingleFileCommand;
