import {Manifold, ManifoldError} from '../../lib/manifoldCAD';

let caught = null;
try {
  Manifold.cube([-1, 1, 1]);
} catch (error) {
  caught = error;
}
if (!(caught instanceof ManifoldError) ||
    caught.code !== 'InvalidConstruction') {
  throw new Error('Expected a ManifoldError');
}
const result = Manifold.sphere(1);
export default result;
