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
const {union} = Manifold;

const csg = async () => {
  // glTF (.glb or .gltf) models have a defined scale of 1 unit to 1 metre.  On
  // the other hand, manifoldCAD has a defined scale of 1 unit to 1 millimetre.
  // These models are 100m and 120m wide, respectively.  Smaller than you'd
  // expect for a moon, but larger than you'd expect for a model.
  const metres = 1000;

  // Models can be imported by URL on manifoldCAD.org, or by path with the CLI.
  // `new URL(relativeURL, import.meta.url)` is a way of resolving relative
  // paths that works inside manifoldCAD.  It also works in node.js (after v21)
  // and most modern browsers, albeit with slightly different results.
  //
  // On disk, `import.meta.url` is the absolute url of the file containing the
  // reference.  For example, `new URL('./model.glb', import.meta.url)` will
  // resolve to a file named `model.glb` in the same directory as the file
  // asking for it.
  //
  // In a browser as there is no filesystem.  By default, `import.meta.url`
  // will be set to the tab URL.  On manifoldCAD.org, this means that relative
  // URLs are resolved using `https://manifoldcad.org/` as the base URL.
  const moon =
      await importManifold(new URL('./models/moon.glb', import.meta.url));
  const space =
      await importManifold(new URL('./models/space.glb', import.meta.url));

  // Demonstrate basic CSG operations.
  const geometry = [
    space.add(moon),
    space.subtract(moon),
    space.intersect(moon),
  ];

  // Space out the results.
  const arranged = geometry.reduce((acc: Manifold, cur: Manifold) => {
    // Place accumulated product to the left (x-), with the right side at x=0.
    // The left side of the current object will be on the right, after a gap.
    //
    // The end result will not be centred in model space.  That's fine as
    // the model viewer will centre it for display.
    const {max: [rightside]} = acc.boundingBox();
    const {min: [leftside]} = cur.boundingBox();

    return union([
      acc.translate([-rightside, 0, 0]),
      cur.translate([-leftside + 20 * metres, 0, 0])
    ]);
  });

  return arranged;
};

// If the default export is a function, manifoldCAD will execute it,
// expecting a Manifold object, GLTFNode, or an array containing the same.
export default csg;
