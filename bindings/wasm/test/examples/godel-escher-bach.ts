// This example recreats the cover of 'Gödel, Escher, Bach: an Eternal Golden Braid'
// by Douglas Hofstadter.  It demonstrates the use of CrossSections and CrossSectionGLTFNodes to
// incorporate two dimensional drawings in a model.
//
// By definition, anything 2D is not manifold and can't be 3D printed.  CrossSections
// will be ignored when exporting to formats like 3MF that require manifold objects.
// When exporting to glTF, CrossSections will be visible, but will not
// have `EXT_mesh_manifold` applied.

import {CrossSection, CrossSectionGLTFNode, GLTFNode, Vec3} from 'manifold-3d/manifoldCAD';
const {circle, square, union, hull} = CrossSection;

/**
 * There are lots of ways to represent letters.
 * This is probably not the best way, but it only really needs to do three letters.
 */
function font(height:number=100, stroke:number=18) {
  const letters:{[key: string]: CrossSection} = {};
  const r = (height+stroke)/4; // Outer radius of loops and arcs.

  // 'E' to start.
  letters['E'] = square([height, height], true)
    .subtract(square([height, height-2*stroke], true).translate([stroke,0]))
    .add(square([height, stroke], true));

  // A loop of B, or corners of G.
  const loop = circle(r)
    .subtract(circle(r-stroke))
    .translate([height/2-r,height/2-r]);

  // Mask two loops to make the curved part of B.
  // Invert the mask to clip 'E' for the square part.
  const loops = loop.add(loop.mirror([0,1]))
  const maskB = square([r, height]).translate([height/2-r, -height/2])
  letters['B'] = letters['E'].subtract(maskB).add(loops.intersect(maskB));

  // Hull four loops to make the outline of an O.
  letters['O'] = hull([loops,loops.mirror([1,0])]);
  letters['O'] = letters['O'].subtract(letters['O'].offset(-stroke));

  // Mask O and add a bar to make G.
  const barG = 2*stroke + (height-(stroke*3))/2;
  letters['G'] = letters['O']
      .subtract(square([height,(height/2-r)]))
      .add(square([barG, stroke],true).translate([(height-barG)/2,0]));

  return letters;
}

export default () => {
  const height = 100;
  const offset = height/2;

  const {G, E, B} = font();
  const root = new GLTFNode();
  root.rotation = [0,0,-45];

  // Create a cross section.
  const leftCS = union(
    G.translate([0, (height+offset)/2]),
    E.translate([0,-(height+offset)/2])
  );
  // And also, a display node for it.
  const leftNode = new CrossSectionGLTFNode(leftCS, root);
  leftNode.rotation = [90,0,90];
  leftNode.translation = [-height-offset,0,0];

  // And the same thing twice more.
  const rightCS = union(
    E.translate([0, (height+offset)/2]),
    G.translate([0,-(height+offset)/2])
  );
  const rightNode = new CrossSectionGLTFNode(rightCS, root);
  rightNode.rotation = [90,0,0];
  rightNode.translation = [0,height+offset,0];

  const bottomCS = B;
  const bottomNode = new CrossSectionGLTFNode(bottomCS, root);
  bottomNode.rotation = [0,0,0];
  bottomNode.translation = [0,0,-height*2];

  // Time for some Constructive Solid Geometry.
  const csNodes = [leftNode, rightNode, bottomNode];
  const geometry = csNodes.map(node => 
    (node._crossSection!.extrude(4*height))
      .rotate(node.rotation as Vec3)
      .translate(node.translation as Vec3)
  ).reduce((acc, cur) => acc.intersect(cur));

  // Also put that into a node so it can be oriented
  // in the same context as our CrossSection nodes.
  const geometryNode = new GLTFNode(root);
  geometryNode.manifold = geometry;
  geometryNode.material = {
      metallic: 0,
      baseColorFactor: [0.4,0.4,0.0],
      roughness: 0.4
  };

  return [geometryNode, root, ...csNodes];
}