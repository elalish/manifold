import {DVMesh} from '@thi.ng/geom-voronoi';
import PoissonDiskSampling from 'poisson-disk-sampling';

const size = 100;
const gap = 1;
const round = 3;
const thickness = 1;
const boundary = CrossSection.square([size, size]);

const sampler = new PoissonDiskSampling({
  shape: [size, size],
  minDistance: (gap + round * 2) * 2,
  maxDistance: 20,
});

const points = sampler.fill();
const cells = (new DVMesh(points))
                  .voronoi()
                  .map(points => CrossSection.ofPolygons([points]))
                  .map(cell => cell.offset(-gap / 2 - round).offset(+round))
                  .map(cs => cs.intersect(boundary));

const result = CrossSection.compose(cells).extrude(thickness);