// Write code in JavaScript or TypeScript and this editor will show the
// API docs. Type e.g. "box." to see the Manifold API. Type
// "CrossSection." or "Manifold." to list the 2D and 3D constructors,
// respectively. Type "module." to see the static API - these functions
// can also be used bare. Use console.log() to print output (lower-right).
// This editor defines Z as up and units of mm.
const {cube, sphere} = Manifold;
const box = cube([100, 100, 100], true);
const ball = sphere(60, 100);

// You must export your model as default.  It can be a Manifold object,
// a GLTFNode, an array of Manifold or GLTFNode objects, or even a function
// that returns one of those options.
// See Menger Sponge, Gyroid Module and Involute Gear Library examples.
const result = box.subtract(ball);
export default result;

// For visual debug, wrap any shape with show() and it and all of its
// copies will be shown in transparent red, akin to # in OpenSCAD. Or try
// only() to ghost out everything else, akin to * in OpenSCAD.

// All changes are automatically saved and restored between sessions.
// This PWA is purely local - there is no server communication.
// This means it will work equally well offline once loaded.
// Consider installing it (icon in the search bar) for easy access.

// See the script drop-down above ("Intro") for usage examples. The
// gl-matrix package from npm is automatically imported for convenience -
// its API is available in the top-level glMatrix object.

// Use GLTFNode for disjoint manifolds rather than compose(), as this will
// keep them better organized in the GLB. This will also allow you to
// specify material properties, and even vertex colors via
// setProperties(). See Tetrahedron Puzzle example.
