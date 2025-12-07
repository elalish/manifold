// This example shows how to import a 3d model and
// manipulate it within manifoldCAD.
//
// There are two functions that import models with different purposes.
// `importModel()` will import a model for visualization purposes.
//
// `importManifold()`, as shown here, will make a best-effort at converting
// the source model into a fully manifold object that can be manipulated
// or combined like any other manifoldCAD object.

import {importManifold, Manifold} from 'manifold-3d/manifoldCAD';
const {compose} = Manifold;

const csg = async () => {
  // glTF (.glb or .gltf) models have a defined scale of 1 unit to 1 metre.  On
  // the other hand, manifoldCAD has a defined scale of 1 unit to 1 millimetre.
  const metres = 1000;

  // These models are 100m and 120m wide, respectively.
  const moon = await importManifold('/models/moon.glb');
  const space = await importManifold('/models/space.glb');

  // Demonstrate basic CSG operations.
  const geometry = [
    space.add(moon),
    space.subtract(moon),
    space.intersect(moon),
  ];

  // Space out the results.
  const gap = 20 * metres;
  const arranged = geometry.reduce((acc: Manifold, cur: Manifold) => {
    // Left and right sides of the gap between objects.
    const {max: [rightside]} = acc.boundingBox();
    const {min: [leftside]} = cur.boundingBox();

    // Cnter the gap on the origin.
    // Place accumulated product to the left, current element to the right.
    // They will not overlap, so `compose()` is preferable over `union()`.
    return compose([
      acc.translate([-rightside - gap / 2, 0, 0]),
      cur.translate([-leftside + gap / 2, 0, 0])
    ]);
  });

  return arranged;
};

// If the default export is a function, manifoldCAD will execute it,
// expecting a Manifold object, GLTFNode, or an array containing the same.
export default csg;
