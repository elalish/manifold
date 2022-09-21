type Vec3 = [number, number, number];
type Matrix4x3 = [Vec3, Vec3, Vec3, Vec3];
type Vec2 = [number, number];
type SimplePolygon = Vec2[];
type Polygons = SimplePolygon[];

declare class Manifold {
  transform(matrix: Matrix4x3): Manifold;
  translate(v: Vec3): Manifold;
  rotate(v: Vec3): Manifold;
  scale(v: Vec3): Manifold;
  add(other: Manifold): Manifold;
  subtract(other: Manifold): Manifold;
  intersect(other: Manifold): Manifold;
  refine(n: number): Manifold;
}

declare function cube(size: Vec3, center?: boolean): Manifold;

declare function cylinder(
    height: number, radiusLow: number, radiusHigh?: number,
    circularSegments?: number, center?: boolean): Manifold;

declare function sphere(radius: number, circularSegments?: number): Manifold;

declare function extrude(
    polygon: Polygons, height: number, nDivisions?: number,
    twistDegrees?: number, scaleTop?: Vec2): Manifold;

declare function revolve(polygon: Polygons, circularSegments?: number): Manifold;

declare function union(a: Manifold, b: Manifold): Manifold;
declare function difference(a: Manifold, b: Manifold): Manifold;
declare function intersection(a: Manifold, b: Manifold): Manifold;

