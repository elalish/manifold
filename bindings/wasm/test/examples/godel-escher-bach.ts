// This example recreates the cover of 'Gödel, Escher, Bach: an Eternal Golden
// Braid' by Douglas Hofstadter.  It demonstrates the use of CrossSections and
// CrossSectionGLTFNodes to incorporate two dimensional drawings in a model.
//
// By definition, anything 2D is not manifold and can't be 3D printed.
// CrossSections will be ignored when exporting to formats like 3MF that require
// manifold objects. When exporting to glTF, CrossSections will be visible, but
// will not have `EXT_mesh_manifold` applied.

import {CrossSection, CrossSectionGLTFNode, GLTFNode, Manifold, Polygons, Vec3} from 'manifold-3d/manifoldCAD';
import {pointsOnPath} from 'points-on-path';
import {Font, getGlyphPath, pathToSVG} from 'text-shaper';

// Orbitron Font from https://github.com/theleagueof/orbitron/
const fonturl =
    'https://raw.githubusercontent.com/theleagueof/orbitron/master/webfonts/orbitron-black-webfont.ttf';

// Return a function that generates a CrossSection for a given letter.
// Getting the function is asynchronous as it depends on a fetch, but
// the function itself is synchronous.
const characterGenerator =
    async (fonturl: string, height = 100) => {
  const font = await Font.fromURL(fonturl);

  // Helper function to center and scale a CrossSection.
  // This particular demo looks best if characters are square, so ignore aspect
  // ratio.
  const resize = (cs: CrossSection) => {
    const {min, max} = cs.bounds();
    return cs.translate([-(min[0] + max[0]) / 2, -(min[1] + max[1]) / 2])
        .scale([height / (max[0] - min[0]), height / (max[1] - min[1])]);
  };

  return (char: string): CrossSection => {
    const glyphpath = getGlyphPath(font, font.glyphIdForChar(char));
    const svgpath = pathToSVG(glyphpath, {flipY: true, scale: 1});
    const cs = new CrossSection(pointsOnPath(svgpath) as Polygons);
    return resize(cs).mirror([0, 1]);
  }
}

export default async () => {
  const height = 100;
  const offset = height / 2;
  const char = await characterGenerator(fonturl, height);

  const csNodes: Array<CrossSectionGLTFNode> = [];
  const rootNode = new GLTFNode();
  rootNode.name = 'Root';

  // Create a cross section for the left side view.
  // And also, a display node for it.
  const leftNode = new CrossSectionGLTFNode(rootNode);
  leftNode.crossSection = CrossSection.union([
    char('G').translate([0, (height + offset) / 2]),
    char('E').translate([0, -(height + offset) / 2])
  ]);
  leftNode.name = 'Left';
  leftNode.rotation = [90, 0, 90];
  leftNode.translation = [-height - offset, 0, 0];
  leftNode.material = {baseColorFactor: [1, 1, 0], unlit: true};
  csNodes.push(leftNode);

  // And the same for the right.
  const rightNode = new CrossSectionGLTFNode(rootNode);
  rightNode.crossSection = CrossSection.union([
    char('E').translate([0, (height + offset) / 2]),
    char('G').translate([0, -(height + offset) / 2])
  ]);
  rightNode.name = 'Right';
  rightNode.rotation = [90, 0, 0];
  rightNode.translation = [0, height + offset, 0];
  rightNode.material = {baseColorFactor: [0, 1, 1], unlit: true};
  csNodes.push(rightNode);

  // Now the bottom.
  const bottomNode = new CrossSectionGLTFNode(rootNode);
  bottomNode.crossSection = char('B');
  bottomNode.name = 'Bottom';
  bottomNode.rotation = [0, 0, 0];
  bottomNode.translation = [0, 0, -height * 2];
  bottomNode.material = {baseColorFactor: [1, 0, 1], unlit: true};
  csNodes.push(bottomNode);

  // Time for some Constructive Solid Geometry.
  // Extrude each CrossSection and apply the same rotation and translation.
  // Then compute the intersection.
  const extrude = (node: CrossSectionGLTFNode) => {
    let extrusion = node.crossSection!.extrude(4 * height);
    extrusion = extrusion.rotate(node.rotation as Vec3);
    extrusion = extrusion.translate(node.translation as Vec3);
    return extrusion;
  };
  const intersect = ((acc: Manifold, cur: Manifold) => acc.intersect(cur));
  const intersection = csNodes.map(extrude).reduce(intersect);

  // Put the result into a node so it can be oriented
  // in the same context as our CrossSection nodes.
  const intersectionNode = new GLTFNode(rootNode);
  intersectionNode.name = 'Intersection';
  intersectionNode.manifold = intersection;
  intersectionNode.material = {
    metallic: 0,
    baseColorFactor: [0.1, 0.1, 0.1],
    roughness: 0.4
  };

  rootNode.rotation = [0, 0, -45];
  return [rootNode, csNodes, intersectionNode];
}