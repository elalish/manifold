import {GLTFNode, Manifold} from '../../lib/manifoldCAD';

const result = Manifold.sphere(1.0);
const node = new GLTFNode();
node.manifold = result;
export default node;