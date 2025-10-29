import {Manifold} from '../../lib/manifoldCAD';

const makeSphere: ((radius?: number) => any) = (radius?: number) => {
  return Manifold.sphere(radius ?? 1.0);
};

const result = makeSphere();
export default result;