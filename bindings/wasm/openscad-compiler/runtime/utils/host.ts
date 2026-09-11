export interface CanvasResolver {
  createCanvas(width: number, height: number): any;
  image(): any;
}

export interface FileResolver {
  exists(filePath: string): boolean;
  readText(filePath: string): string|null;
  readBinary(filePath: string): Buffer|null;
  readDir(path: string): string[];
}

export let runtimeFileResolver: FileResolver;
export let runtimeCanvasResolver: CanvasResolver;

export function setRunTimeFileResolver(fileResolver: FileResolver) {
  runtimeFileResolver = fileResolver;
}

export function setRunTimeCanvasResolver(canvasResolver: CanvasResolver) {
  runtimeCanvasResolver = canvasResolver;
}