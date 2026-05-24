import { pathToFileURL } from 'url';
import path from 'path';

const filename = process.argv[2];

async function run() {
  if (!filename) {
    process.stderr.write('No filename argument provided\n');
    process.exit(1);
  }

  const absolutePath = path.resolve(filename);
  const fileUrl = pathToFileURL(absolutePath).href;

  const mod = await import(fileUrl);

  if (!mod.result) {
    throw new Error(`Module has no 'result' export: ${filename}`);
  }

  const result = {
    volume: mod.result.volume() as number,
    surfaceArea: mod.result.surfaceArea() as number,
  };

  process.stdout.write(JSON.stringify(result));
  process.exit(0);
}

run().catch(e => {
  process.stderr.write(`Worker error: ${String(e)}\n${e?.stack ?? ''}\n`);
  process.exit(1);
});