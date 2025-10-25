// Generate an organic looking Voronoi pattern.
//
// If each point defines the centre of a bubble, a Voronoi graph is
// what happens when each bubble has expands until it meets a
// neighbouring bubble.

// poison-disk-spacing generates random but evenly distributed points.
import PoissonDiskSampling from 'poisson-disk-sampling';
// geom-voronoi converts a list of points into a Voronoi graph. 
import {DVMesh} from '@thi.ng/geom-voronoi';

const size = 100;    // Side length
const gap = 1;       // Gap between cells
const round = 3;     // Roundover radius
const thickness = 1; // Thickness of the final object.
const boundary = CrossSection.square([size, size]);

// New points will be no closer than minDistance, but no farther
// than maxDistance from each other.
const sampler = new PoissonDiskSampling({
  shape: [size, size],
  minDistance: 10,
  maxDistance: 20,
});

// Generate points until no more will fit.  Then create a Voronoi graph.
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