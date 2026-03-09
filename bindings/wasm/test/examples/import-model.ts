// This example shows `importModel()`, which imports a model for display only.
// Unlike `importManifold()`, the result cannot be used in CSG operations.

import {importModel, Manifold} from 'manifold-3d/manifoldCAD';

const csg = async () => {
  // `new URL(..., import.meta.url)` resolves this file relative to the script.
  const moon = await importModel(new URL('./models/moon.glb', import.meta.url));

  // Imported models can still be transformed.
  moon.translation = [0, 0, 50_000];

  // Return display-only and CSG geometry together.
  const platform = Manifold.cylinder(5_000, 60_000);

  return [moon, platform];
};

// If the default export is a function, manifoldCAD will execute it,
// expecting a Manifold object, GLTFNode, or an array containing the same.
export default csg;
