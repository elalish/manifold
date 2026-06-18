// Volume: 1996.0648272258063, SurfaceArea: 8903.954859552894
include <BOSL2/std.scad>

// Makes a pseudo-sphere from a rectangular patch and its mirror.
s = 50/sqrt(2);
d = [[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]];
p = s * d;
q =  s * 0.55 * d;
u =  s * 2.5 * UP;
patch1 = [
    [p[2],      p[2]+q[3],   p[3]+q[2],   p[3]     ],
    [p[2]+q[1], p[2]+q[2]+u, p[3]+q[3]+u, p[3]+q[0]],
    [p[1]+q[2], p[1]+q[1]+u, p[0]+q[0]+u, p[0]+q[3]],
    [p[1],      p[1]+q[0],   p[0]+q[1],   p[0]     ],
];
patch2 = bezier_patch_reverse(zflip(p=patch1));
//vnf_polyhedron([bezier_vnf(patch1),bezier_vnf(patch2)]);
debug_bezier_patches([patch1, patch2], splinesteps=16, style="quincunx");


// vim: expandtab tabstop=4 shiftwidth=4 softtabstop=4 nowrap
