import Module from '../../manifold.js';

const wasm = await Module();
wasm.setup();
wasm.Manifold.cube().getMesh();
