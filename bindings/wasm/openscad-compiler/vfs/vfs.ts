interface VFSFile {
  name: string;
  type: 'file';
  parent: VFSDirectory|null;
  data: Uint8Array;
}

interface VFSDirectory {
  name: string;
  type: 'directory';
  parent: VFSDirectory|null;
  children: Map<string, VFSNode>;
}

type VFSNode = VFSFile|VFSDirectory;

export class VFS {
  private readonly root: VFSDirectory;

  constructor() {
    this.root = {
      name: '',
      type: 'directory',
      parent: null,
      children: new Map(),
    };
  }

  mkdirSync(path: string, options: {recursive?: boolean} = {}): void {
    const parts = this.parsePath(path);

    if (parts.length === 0) return;

    const recursive = options.recursive ?? false;

    let current = this.root;

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i]!;
      const existing = current.children.get(part);

      if (existing) {
        if (existing.type !== 'directory') {
          throw new Error(`ENOTDIR: ${part} is not a directory`);
        }

        if (i === parts.length - 1 && !recursive) {
          throw new Error(`EEXIST: file already exists, mkdir '${path}'`);
        }

        current = existing;
        continue;
      }

      // if an intermediate directory doesn't exist, recursive mode is required.
      if (i < parts.length - 1 && !recursive) {
        throw new Error(`ENOENT: no such file or directory, mkdir '${path}'`);
      }

      const directory: VFSDirectory = {
        name: part,
        type: 'directory',
        parent: current,
        children: new Map(),
      };

      current.children.set(part, directory);
      current = directory;
    }
  }

  existsSync(path: string): boolean {
    return this.resolve(path) !== null;
  }

  isFileSync(path: string): boolean {
    return this.resolve(path)?.type === 'file';
  }

  readFileSync(path: string, encoding?: string): Uint8Array|string {
    const node = this.resolve(path);

    if (!node) {
      throw new Error(`ENOENT: no such file or directory: ${path}`);
    }

    if (node.type !== 'file') {
      throw new Error(`EISDIR: ${path} is a directory`);
    }

    if (encoding === 'utf8' || encoding === 'utf-8') {
      return new TextDecoder('utf-8').decode(node.data);
    }

    return node.data;
  }

  readdirSync(path: string): string[] {
    const node = this.resolve(path);
    if (!node) {
      throw new Error(`ENOENT: no such directory: ${path}`);
    }
    if (node.type !== 'directory') {
      throw new Error(`ENOTDIR: ${path} is not a directory`);
    }
    return [...node.children.keys()];
  }

  writeFileSync(path: string, data: string|Uint8Array, encoding?: string):
      void {
    const parts = this.parsePath(path);

    if (parts.length === 0) {
      throw new Error('Cannot write to root');
    }

    const fileName = parts.pop()!;

    let current = this.root;

    // Resolve parent directory
    for (const part of parts) {
      const node = current.children.get(part);

      if (!node) {
        throw new Error(`ENOENT: no such file or directory: ${part}`);
      }

      if (node.type !== 'directory') {
        throw new Error(`ENOTDIR: ${part} is not a directory`);
      }

      current = node;
    }

    if (current.children.get(fileName)?.type === 'directory') {
      throw new Error(`EISDIR: ${path} is a directory`);
    }

    let fileData: Uint8Array;

    if (typeof data === 'string') {
      if (encoding !== undefined && encoding !== 'utf8' &&
          encoding !== 'utf-8') {
        throw new Error(`Unsupported encoding: ${encoding}`);
      }

      fileData = new TextEncoder().encode(data);
    } else {
      fileData = data;
    }

    const file: VFSFile = {
      name: fileName,
      type: 'file',
      parent: current,
      data: fileData,
    };

    current.children.set(fileName, file);
  }

  rmSync(path: string, options: {recursive?: boolean; force?: boolean} = {}):
      void {
    const parts = this.parsePath(path);

    // don't allow deleting the root
    if (parts.length === 0) {
      throw new Error('EPERM: cannot remove root directory');
    }

    const name = parts.pop()!;

    let parent = this.root;

    // resolve parent directory
    for (const part of parts) {
      const node = parent.children.get(part);

      if (!node) {
        if (options.force) return;
        throw new Error(`ENOENT: no such file or directory: ${path}`);
      }

      if (node.type !== 'directory') {
        throw new Error(`ENOTDIR: ${part} is not a directory`);
      }

      parent = node;
    }

    const node = parent.children.get(name);

    if (!node) {
      if (options.force) return;
      throw new Error(`ENOENT: no such file or directory: ${path}`);
    }

    if (node.type === 'directory' && node.children.size > 0 &&
        !options.recursive) {
      throw new Error(`ENOTEMPTY: directory not empty: ${path}`);
    }

    parent.children.delete(name);
  }

  private resolve(path: string): VFSNode|null {
    const parts = this.parsePath(path);

    let current: VFSNode = this.root;

    for (const part of parts) {
      if (current.type !== 'directory') {
        return null;
      }

      const child = current.children.get(part);

      if (!child) {
        return null;
      }

      current = child;
    }

    return current;
  }

  private parsePath(path: string): string[] {
    const parts: string[] = [];
    for (const part of path.replace(/\\/g, '/').split('/')) {
      if (!part || part === '.') continue;
      if (part === '..') {
        parts.pop();
      } else {
        parts.push(part);
      }
    }
    return parts;
  }
}


export const vfs = new VFS();
