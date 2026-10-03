#!/usr/bin/env node
import {spawnSync} from 'node:child_process';
import {cpSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync,} from 'node:fs';
import {dirname, join, relative, resolve, sep} from 'node:path';
import {inflateRawSync} from 'node:zlib';

const ZIP = 'test_zip.zip';
const FOLDER = 'test';
const STAGING = '.test_zip_staging';

type Entry = {
  name: string; method: number; compressedSize: number; localOffset: number;
};

function readEntries(buf: Buffer): Entry[] {
  let eocd = -1;
  for (let i = buf.length - 22; i >= 0; i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error(`${ZIP} is not a valid zip file`);

  const count = buf.readUInt16LE(eocd + 10);
  const out: Entry[] = [];
  let p = buf.readUInt32LE(eocd + 16);

  for (let i = 0; i < count; i++) {
    if (buf.readUInt32LE(p) !== 0x02014b50) {
      throw new Error(`${ZIP} has a corrupt central directory`);
    }
    const nameLen = buf.readUInt16LE(p + 28);
    out.push({
      method: buf.readUInt16LE(p + 10),
      compressedSize: buf.readUInt32LE(p + 20),
      localOffset: buf.readUInt32LE(p + 42),
      name: buf.toString('utf8', p + 46, p + 46 + nameLen),
    });
    p += 46 + nameLen + buf.readUInt16LE(p + 30) + buf.readUInt16LE(p + 32);
  }
  return out;
}

function extract(zipPath: string, destDir: string): void {
  const buf = readFileSync(zipPath);

  for (const entry of readEntries(buf)) {
    const target = resolve(destDir, entry.name);
    const rel = relative(destDir, target);
    if (rel.startsWith(`..${sep}`) || rel === '..') {
      throw new Error(`entry escapes the output directory: ${entry.name}`);
    }

    if (entry.name.endsWith('/')) {
      mkdirSync(target, {recursive: true});
      continue;
    }

    // The local header repeats the name and extra field with its own lengths
    const p = entry.localOffset;
    const start = p + 30 + buf.readUInt16LE(p + 26) + buf.readUInt16LE(p + 28);
    const raw = buf.subarray(start, start + entry.compressedSize);

    let data: Buffer;
    if (entry.method === 0)
      data = raw;
    else if (entry.method === 8)
      data = inflateRawSync(raw);
    else
      throw new Error(`unsupported compression in ${entry.name}`);

    mkdirSync(dirname(target), {recursive: true});
    writeFileSync(target, data);
  }
}

function findFolder(root: string, name: string): string|null {
  const queue = [root];
  while (queue.length) {
    const dir = queue.shift()!;
    const kids = readdirSync(dir, {withFileTypes: true})
                     .filter(
                         (d) => d.isDirectory(),
                     );
    for (const kid of kids)
      if (kid.name === name) return join(dir, kid.name);
    for (const kid of kids) queue.push(join(dir, kid.name));
  }
  return null;
}

function run(args: string[]): void {
  console.log(`\n> npm ${args.join(' ')}`);
  const res = spawnSync('npm', args, {
    stdio: 'inherit',
    shell: process.platform === 'win32',  // npm is a .cmd shim there
  });
  if (res.error) throw res.error;
  if (res.status !== 0) throw new Error(`npm ${args.join(' ')} failed`);
}

function main(): void {
  const root = process.cwd();
  const zipPath = join(root, ZIP);
  if (!existsSync(zipPath)) throw new Error(`${ZIP} not found in ${root}`);

  const staging = join(root, STAGING);
  rmSync(staging, {recursive: true, force: true});
  mkdirSync(staging, {recursive: true});

  try {
    extract(zipPath, staging);

    const found = findFolder(staging, FOLDER);
    if (!found) throw new Error(`no '${FOLDER}' folder inside ${ZIP}`);

    const dest = join(root, FOLDER);
    rmSync(dest, {recursive: true, force: true});
    try {
      renameSync(found, dest);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EXDEV') throw err;
      cpSync(found, dest, {recursive: true});
    }
    console.log(`unpacked ${ZIP} -> ${FOLDER}/`);
  } finally {
    rmSync(staging, {recursive: true, force: true});
  }

  run(['i']);
  run(['run', 'test']);
}

try {
  main();
} catch (err) {
  console.error(`\nerror: ${(err as Error).message}`);
  process.exit(1);
}