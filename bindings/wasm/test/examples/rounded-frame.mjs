// Demonstrates how at 90-degree intersections, the sphere and cylinder
// facets match up perfectly, for any choice of global resolution
// parameters.
const {sphere, cylinder, union, cube} = Manifold;

function roundedFrame(edgeLength, radius, circularSegments = 0) {
  const edge = cylinder(edgeLength, radius, -1, circularSegments);
  const corner = sphere(radius, circularSegments);

  const edge1 = union(corner, edge).rotate([-90, 0, 0]).translate([
    -edgeLength / 2, -edgeLength / 2, 0
  ]);

  const edge2 = union(
      union(edge1, edge1.rotate([0, 0, 180])),
      edge.translate([-edgeLength / 2, -edgeLength / 2, 0]));

  const edge4 = union(edge2, edge2.rotate([0, 0, 90])).translate([
    0, 0, -edgeLength / 2
  ]);

  return union(edge4, edge4.rotate([180, 0, 0]));
}

setMinCircularAngle(3);
setMinCircularEdgeLength(0.5);
const result = roundedFrame(100, 10);
// Demonstrate how you can use the .split method to perform
// a subtraction and an intersection at once
const [inside, outside] = result.split(cube(100, true));

const outsideNode = new GLTFNode();
outsideNode.manifold = outside;

const insideNode = new GLTFNode();
insideNode.manifold = inside;
insideNode.material = {baseColorFactor: [0, 1, 1]};
const nodes = [outsideNode, insideNode];
export default nodes;