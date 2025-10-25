import {GLTFNode, Manifold} from '../../lib/manifoldCAD';

// Models that build GLTF objects need not export their results.  The
// scene builder will track GLTF nodes as they are created.  Essentially,
// it's possible to build a manifoldCAD model as a side-effect of an import.

const result = Manifold.sphere(1.0);
const node = new GLTFNode();
node.manifold = result;
export {}