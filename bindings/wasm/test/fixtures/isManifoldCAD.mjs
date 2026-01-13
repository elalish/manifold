import {isManifoldCAD, Manifold} from '../../lib/manifoldCAD';
const {cube, sphere} = Manifold;

export const isManifoldCADReturns = isManifoldCAD();

// We don't really have a way of returning a boolean value
// through the worker to vitest.  But we can check the properties
// of our model.

export const result = isManifoldCAD() ? cube(1) : sphere(1);
export default result;
