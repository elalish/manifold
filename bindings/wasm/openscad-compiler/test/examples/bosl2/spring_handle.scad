// Volume: 11709.322761918755, SurfaceArea: 9834.380914213336
include <BOSL2/std.scad>

$fn = 45;
wire_d = 2;
spring_l = 100;
spring_d = 20;
rod_d = 10;
loops = 17;
tight_loops=3;

tpart = tight_loops/loops;
lpart = wire_d * tight_loops / 100;
r_table = [
    [0.00, 0],
    [0+tpart, 0],
    [0.5, 1],
    [1-tpart, 0],
    [1.00, 0],
];
l_table = [
    [0.00, -0.50],
    [0+tpart, -0.5+lpart],
    [1-tpart, +0.5-lpart],
    [1.00, +0.50],
];
lsteps = 45;
tsteps = loops * lsteps;
path = [
    for (i = [0:1:tsteps])
    let(
        u = i / tsteps,
        a = u * 360 * loops,
        r = lookup(u, r_table) * spring_d/2 + wire_d/2 + rod_d/2,
        z = lookup(u, l_table) * spring_l,
        pt = [r*cos(a), r*sin(a), z]
    ) pt
];
yrot(90) {
    color("lightblue")
        path_sweep(circle(d=wire_d), path);
    cylinder(d=rod_d, h=spring_l+10, center=true);
}


// vim: expandtab tabstop=4 shiftwidth=4 softtabstop=4 nowrap
