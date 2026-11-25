import {Manifold} from '../../lib/manifoldCAD';

// A transform is lazy, so the error surfaces at export.
const result = Manifold.cube().translate([NaN, 0, 0]);
export default result;
