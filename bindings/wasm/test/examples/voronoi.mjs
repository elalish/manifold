// Generate an organic looking Voronoi pattern.

import {CrossSection} from 'manifold-3d/manifoldCAD';

// When run through manifoldcad.org or the `manifold-cad` CLI,
// imported npm packages will automatically be retrieved from
// a content delivery network (Such as jsDelivr or esm.sh) and
// bundled with this script.
// When imported directly into node, there is no bundling step.
// These packages would have to be installed via npm.
import PoissonDiskSampling from 'poisson-disk-sampling';
import {DVMesh} from '@thi.ng/geom-voronoi';

const size = 100;    // Side length
const gap = 1;       // Gap between cells
const round = 3;     // Roundover radius
const thickness = 1; // Thickness of the final object.
const boundary = CrossSection.square([size, size]);

// Generate random but evenly distributed points.
// New points will be no closer than minDistance, but no farther
// than maxDistance from each other.
const sampler = new PoissonDiskSampling({
  shape: [size, size],
  minDistance: 10,
  maxDistance: 20,
});

// Generate points until no more will fit.  Then create a Voronoi graph.
// If each point defines the centre of a bubble, a Voronoi graph is
// what happens when each bubble has expands until it meets a
// neighbouring bubble.
const mesh = new DVMesh(sampler.fill());

// For each cell, create a 2D CrossSection object from the points
// defining that cell.  Shrinking and growing the CrossSection results in gaps
// between cells, and rounded corners.  Finally, clip the results to fit
// our predetermined boundary.
const cells = mesh
  .voronoi()
  .map(points => CrossSection.ofPolygons([points]))
  .map(cell => cell.offset(-gap / 2 - round).offset(+round))
  .map(cs => cs.intersect(boundary));

// We now have a list of rounded cells, but it's easier to have a single
// object.  We could union them together, but we do know that they do not
// overlap.  This makes CrossSection.compose() the more efficient option.
// After that, make a Manifold object from the CrossSection by extruding
// it along the Z axis. 
const result = CrossSection
  .compose(cells)
  .extrude(thickness);

export default result;