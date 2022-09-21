type Matrix1x3 = [number, number, number];
type Matrix4x3 = [Matrix1x3, Matrix1x3, Matrix1x3, Matrix1x3];
type Vec3 = [number, number, number];
type Vec3Fun = (v: Vec3) => Manifold;
type Vec2 = [number, number];
type SimplePolygon = Vec2[];
type Polygons = SimplePolygon[];

declare class Manifold {
  transform(matrix: Matrix4x3): Manifold;

  translate: Vec3Fun;
  rotate: Vec3Fun;
  scale: Vec3Fun;
  add(other: Manifold): Manifold;
  subtract(other: Manifold): Manifold;
  intersect(other: Manifold): Manifold;
  refine(n: number): Manifold;
}

declare function cube(size: Vec3, center?: boolean): Manifold;
declare function cube(size?: number, center?: boolean): Manifold;

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

