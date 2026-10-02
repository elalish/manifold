import { vfs } from '../vfs/vfs.js';
import path from 'path-browserify';

import {compileConsumer} from '../core/orchestrate.js';
import {setGlobalFileResolver, setGlobalPathResolver} from '../core/state.js';
import {webFileResolver, webPathResolver} from '../host/web.js';

import {getExternalLibraries} from './utils/library_compilation.js';
import {rewriteImports, runtimeUrl, stripTypes} from './utils/import_handler.js';


export async function compile(
    inputCode: string,
    options: {runtimeUrl?: string; esbuildWasmUrl?: string} = {}) {
  try {
    if (!inputCode) {
      console.log('Error: Input code is required');
      return;
    }

    const randomId = crypto.randomUUID();
    console.log(randomId);
    
    const inputPath = `${randomId}.scad`;
    vfs.writeFileSync(inputPath, inputCode);

    // derive output path from input path
    const outputPath = `${randomId}.ts`;
    const blobUrls = new Map<string, string>();
    const createdUrls: string[] = [];
    const runtimeImportUrl = options.runtimeUrl ?
        new URL(options.runtimeUrl, location.href).href :
        runtimeUrl;

    try {
      setGlobalFileResolver(webFileResolver);
      setGlobalPathResolver(webPathResolver);

      const {externalLibraries, resolved} = await getExternalLibraries(inputPath, outputPath);

      const {code: js, resolvedFiles} = await compileConsumer(inputPath, outputPath, "/", externalLibraries, resolved);
      vfs.writeFileSync(outputPath, js);

      // collect all compiled library files from vfs the manifest tells us which files exist
      let needsBundle = false;
      for (const extLib of externalLibraries) {
        const libDir = '/' + path.join('runtime', 'libraries', extLib.name.toLowerCase());
        const manifestPath = path.join(libDir, '.manifest.json');
        const manifest = JSON.parse(vfs.readFileSync(manifestPath, 'utf-8') as string);
        
        // collect all library files that need blob urls
        const pending: {vfsPath: string; jsVfsPath: string; code: string}[] = [];
        for (const fileInfo of Object.values(manifest.files)) {
          const outRel = (fileInfo as any).out as string;
          const vfsPath = path.join(libDir, outRel);
          const jsVfsPath = vfsPath.replace(/\.ts$/, '.js');
          const code = stripTypes(vfs.readFileSync(vfsPath, 'utf-8') as string);
          pending.push({ vfsPath, jsVfsPath, code });
        }

        // blob urls have no VFS base path, so create each module only after all relative imports have their own blob url
        while (pending.length > 0) {
          let progressed = false;
          for (let i = pending.length - 1; i >= 0; i--) {
            const entry = pending[i]!;
            const {code: rewritten, unresolved} =
                await rewriteImports(
                    entry.code, entry.vfsPath, blobUrls, runtimeImportUrl);
            if (unresolved.length > 0) continue;

            const blob = new Blob([rewritten], { type: 'application/javascript' });
            const url = URL.createObjectURL(blob);
            createdUrls.push(url);
            blobUrls.set(entry.vfsPath, url);
            blobUrls.set(entry.jsVfsPath, url);
            pending.splice(i, 1);
            progressed = true;
          }
          if (!progressed) {
            needsBundle = true;
            break;
          }
        }
        if (needsBundle) break;
      }

      let executableCode: string;
      if (needsBundle) {
        const {bundleVfsEntry} = await import('./utils/bundle.js');
        executableCode = await bundleVfsEntry(
            '/' + outputPath, runtimeImportUrl, options.esbuildWasmUrl);
      } else {
        // Rewrite the consumer after all acyclic library URLs are available.
        const consumerVfsPath = '/' + outputPath;
        const strippedConsumer = stripTypes(js);
        const {code, unresolved} = await rewriteImports(
            strippedConsumer, consumerVfsPath, blobUrls, runtimeImportUrl);
        if (unresolved.length > 0) {
          throw new Error(`Unresolved consumer imports: ${
              unresolved.join(', ')}`);
        }
        executableCode = code;
      }
      // create blob url for consumer and execute
      const consumerBlob =
          new Blob([executableCode], {type: 'application/javascript'});
      const consumerUrl = URL.createObjectURL(consumerBlob);
      createdUrls.push(consumerUrl);
      await import(/* @vite-ignore */ consumerUrl);

      if (resolvedFiles.length > 1) {
        console.log(`Resolved ${resolvedFiles.length} local files`);
      }
      if (externalLibraries.length > 0) {
        console.log(`External libraries: ${
            externalLibraries.map(lib => lib.name).join(', ')}`);
      }
      console.log(`Generated TypeScript (${js.length.toLocaleString()} chars)`);
      console.log(`Output written to ${outputPath}`);
    } catch (err) {
      console.error(`Error: ${(err as Error).message}`);
    } finally {
      for (const url of createdUrls) URL.revokeObjectURL(url);
    }
  } catch (error) {
    console.log('An error occurred: ' + error);
    return;
  }
}

export default compile;
