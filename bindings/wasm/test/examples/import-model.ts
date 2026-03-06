// This example shows how to import a 3d model for display-only purposes
// using `importModel()`.
//
// Unlike `importManifold()`, which converts the source model into a fully
// manifold object that can be modified with CSG operations, `importModel()`
// preserves the original model exactly — including all materials, textures
// and geometry — but the result cannot be used in boolean operations.
//
// Use `importModel()` when you want to:
//   - Display a reference model alongside your design.
//   - Include a non-manifold or complex mesh that would be lost in conversion.
//   - Preserve original materials and textures from the source file.

import {importModel, Manifold} from 'manifold-3d/manifoldCAD';

const csg = async () => {
  // glTF (.glb or .gltf) models have a defined scale of 1 unit to 1 metre.  On
  // the other hand, manifoldCAD has a defined scale of 1 unit to 1 millimetre.
  const metres = 1000;

  // `importModel()` loads the model for visualization only and returns a
  // `VisualizationGLTFNode`.  The original materials, textures and geometry
  // are preserved as-is.
  //
  // Models can be imported by URL on manifoldCAD.org, or by path with the CLI.
  // `new URL(relativeURL, import.meta.url)` resolves paths relative to this
  // file on disk, or relative to the tab URL in a browser.
  const moon = await importModel(new URL('./models/moon.glb', import.meta.url));

  // Give the imported model a name so it appears clearly in scene viewers.
  moon.name = 'Moon (display only)';

  // The imported model can be transformed but not used in CSG operations.
  moon.translation = [0, 0, 50 * metres];

  // A regular Manifold object can be returned alongside a VisualizationGLTFNode
  // by placing both in an array.  ManifoldCAD will render them together.
  const platform = Manifold.cylinder(5 * metres, 60 * metres);

  // Return the display-only model and the manifold geometry together.
  return [moon, platform];
};

// If the default export is a function, manifoldCAD will execute it,
// expecting a Manifold object, GLTFNode, or an array containing the same.
export default csg;
