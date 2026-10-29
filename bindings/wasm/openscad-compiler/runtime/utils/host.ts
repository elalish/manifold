export interface CanvasResolver {
  createCanvas(width: number, height: number): any;
  image(): any;
}

export interface FileResolver {
  exists(filePath: string): boolean;
  readText(filePath: string): string|null;
  readBinary(filePath: string): Buffer|null;
  writeFile(filePath: string, blobBuffer: Buffer): void;
  readDir(path: string): string[];
}

export interface EnvironmentResolver {
  fontDir: string;
  mode: 'web'|'node';
}


export let runtimeFileResolver: FileResolver;
export let runtimeCanvasResolver: CanvasResolver;
export let environmentResolver: EnvironmentResolver;

export function setRunTimeFileResolver(fileResolver: FileResolver) {
  runtimeFileResolver = fileResolver;
}

export function setRunTimeCanvasResolver(canvasResolver: CanvasResolver) {
  runtimeCanvasResolver = canvasResolver;
}

export function setEnvironmentResolver(resolver: EnvironmentResolver) {
  environmentResolver = resolver;
}
